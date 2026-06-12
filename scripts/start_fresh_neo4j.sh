#!/usr/bin/env bash
set -euo pipefail

SOURCE_HOME="${KGQA_NEO4J_SOURCE:-/home/u2023312337/CoMaGRAG/external_tools/neo4j}"
JAVA_HOME_DIR="${JAVA_HOME:-/home/u2023312337/CoMaGRAG/external_tools/java17}"
BASE_DIR="${KGQA_NEO4J_BASE_DIR:-$HOME/.local/share}"
INSTANCE_NAME="${KGQA_NEO4J_INSTANCE_NAME:-kgqa-neo4j-openfusion-$(date +%Y%m%d-%H%M%S)}"
NEW_HOME="${KGQA_NEO4J_HOME:-$BASE_DIR/$INSTANCE_NAME}"
START_BOLT_PORT="${KGQA_NEO4J_BOLT_PORT:-7690}"
START_HTTP_PORT="${KGQA_NEO4J_HTTP_PORT:-7477}"
PASSWORD="${KGQA_NEO4J_PASSWORD:-kgqa_neo4j_local_$(date +%Y%m%d%H%M%S)}"
USERNAME="${KGQA_NEO4J_USERNAME:-neo4j}"
DATABASE="${KGQA_NEO4J_DATABASE:-neo4j}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

port_in_use() {
  local port="$1"
  ss -ltn 2>/dev/null | grep -Eq "[:.]${port}[[:space:]]"
}

find_port_pair() {
  local bolt="$START_BOLT_PORT"
  local http="$START_HTTP_PORT"
  for offset in $(seq 0 50); do
    local candidate_bolt=$((bolt + offset))
    local candidate_http=$((http + offset))
    if ! port_in_use "$candidate_bolt" && ! port_in_use "$candidate_http"; then
      printf '%s %s\n' "$candidate_bolt" "$candidate_http"
      return 0
    fi
  done
  return 1
}

[ -d "$SOURCE_HOME" ] || fail "Neo4j source home not found: $SOURCE_HOME"
[ -x "$SOURCE_HOME/bin/neo4j" ] || fail "Neo4j binary not found under: $SOURCE_HOME/bin"
[ -d "$JAVA_HOME_DIR" ] || fail "JAVA_HOME directory not found: $JAVA_HOME_DIR"
[ ! -e "$NEW_HOME" ] || fail "Target Neo4j home already exists: $NEW_HOME"

read -r BOLT_PORT HTTP_PORT < <(find_port_pair) || fail "No free Bolt/HTTP port pair found"

mkdir -p "$NEW_HOME"
cp -a \
  "$SOURCE_HOME/bin" \
  "$SOURCE_HOME/certificates" \
  "$SOURCE_HOME/conf" \
  "$SOURCE_HOME/import" \
  "$SOURCE_HOME/labs" \
  "$SOURCE_HOME/lib" \
  "$SOURCE_HOME/licenses" \
  "$SOURCE_HOME/plugins" \
  "$SOURCE_HOME/products" \
  "$SOURCE_HOME/LICENSES.txt" \
  "$SOURCE_HOME/LICENSE.txt" \
  "$SOURCE_HOME/NOTICE.txt" \
  "$SOURCE_HOME/packaging_info" \
  "$SOURCE_HOME/README.txt" \
  "$SOURCE_HOME/UPGRADE.txt" \
  "$NEW_HOME/"

mkdir -p "$NEW_HOME/data" "$NEW_HOME/logs" "$NEW_HOME/run"

CONF="$NEW_HOME/conf/neo4j.conf"
grep -vE '^(server\.default_listen_address=|server\.default_advertised_address=|server\.bolt\.|server\.http\.|server\.https\.|server\.memory\.heap\.initial_size=|server\.memory\.heap\.max_size=|server\.memory\.pagecache\.size=|dbms\.security\.auth_enabled=)' "$CONF" > "$CONF.tmp"
mv "$CONF.tmp" "$CONF"

cat >> "$CONF" <<EOF

# OpenFusionKGQA fresh Neo4j instance
server.default_listen_address=127.0.0.1
server.default_advertised_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=127.0.0.1:${BOLT_PORT}
server.bolt.advertised_address=127.0.0.1:${BOLT_PORT}
server.http.enabled=true
server.http.listen_address=127.0.0.1:${HTTP_PORT}
server.http.advertised_address=127.0.0.1:${HTTP_PORT}
server.https.enabled=false
server.memory.heap.initial_size=512m
server.memory.heap.max_size=512m
server.memory.pagecache.size=256m
dbms.security.auth_enabled=true
EOF

JAVA_HOME="$JAVA_HOME_DIR" \
NEO4J_HOME="$NEW_HOME" \
NEO4J_CONF="$NEW_HOME/conf" \
  "$NEW_HOME/bin/neo4j-admin" dbms set-initial-password "$PASSWORD" >/dev/null

setsid env \
  JAVA_HOME="$JAVA_HOME_DIR" \
  NEO4J_HOME="$NEW_HOME" \
  NEO4J_CONF="$NEW_HOME/conf" \
  "$NEW_HOME/bin/neo4j" console > "$NEW_HOME/logs/console.out" 2>&1 < /dev/null &
echo $! > "$NEW_HOME/run/neo4j-console-wrapper.pid"

for _ in $(seq 1 60); do
  if port_in_use "$BOLT_PORT"; then
    break
  fi
  sleep 1
done

if ! port_in_use "$BOLT_PORT"; then
  tail -80 "$NEW_HOME/logs/console.out" >&2 || true
  fail "Neo4j Bolt port did not become ready: $BOLT_PORT"
fi

JAVA_HOME="$JAVA_HOME_DIR" "$NEW_HOME/bin/cypher-shell" \
  -a "bolt://127.0.0.1:${BOLT_PORT}" \
  -u "$USERNAME" \
  -p "$PASSWORD" \
  "RETURN 1 AS ok;" >/dev/null

ENV_FILE="$NEW_HOME/kgqa-neo4j.env"
cat > "$ENV_FILE" <<EOF
export NEO4J_URI="bolt://127.0.0.1:${BOLT_PORT}"
export NEO4J_USERNAME="${USERNAME}"
export NEO4J_PASSWORD="${PASSWORD}"
export NEO4J_DATABASE="${DATABASE}"
export KGQA_NEO4J_HOME="${NEW_HOME}"
EOF

cat <<EOF
Fresh Neo4j instance is ready.

home:      ${NEW_HOME}
bolt:      bolt://127.0.0.1:${BOLT_PORT}
http:      http://127.0.0.1:${HTTP_PORT}
env_file:  ${ENV_FILE}

Use:
  source "${ENV_FILE}"

Stop:
  JAVA_HOME="${JAVA_HOME_DIR}" NEO4J_HOME="${NEW_HOME}" NEO4J_CONF="${NEW_HOME}/conf" "${NEW_HOME}/bin/neo4j" stop
EOF
