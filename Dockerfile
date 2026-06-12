FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY graphrag_v2 ./graphrag_v2
COPY scripts ./scripts
COPY examples ./examples
COPY settings.neo4j.example.yaml settings.llm.neo4j.example.yaml settings.compose.neo4j.yaml ./

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

ENTRYPOINT ["kgqa"]
CMD ["--help"]
