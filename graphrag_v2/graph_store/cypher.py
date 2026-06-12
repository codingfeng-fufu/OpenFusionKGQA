"""Cypher snippets for the Neo4j graph store implementation."""

NEO4J_SCHEMA_VERSION = "2026-06-06.v1"

SCHEMA_CONSTRAINTS = {
    "kgqa_index_id_unique": (
        "CREATE CONSTRAINT kgqa_index_id_unique IF NOT EXISTS "
        "FOR (i:KGQAIndex) REQUIRE i.id IS UNIQUE"
    ),
    "kgqa_document_scoped_id_unique": (
        "CREATE CONSTRAINT kgqa_document_scoped_id_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE (d.index_id, d.id) IS UNIQUE"
    ),
    "kgqa_text_unit_scoped_id_unique": (
        "CREATE CONSTRAINT kgqa_text_unit_scoped_id_unique IF NOT EXISTS "
        "FOR (t:TextUnit) REQUIRE (t.index_id, t.id) IS UNIQUE"
    ),
    "kgqa_entity_scoped_id_unique": (
        "CREATE CONSTRAINT kgqa_entity_scoped_id_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE (e.index_id, e.id) IS UNIQUE"
    ),
    "kgqa_relation_schema_scoped_name_unique": (
        "CREATE CONSTRAINT kgqa_relation_schema_scoped_name_unique IF NOT EXISTS "
        "FOR (r:RelationSchema) REQUIRE (r.index_id, r.name) IS UNIQUE"
    ),
}

SCHEMA_INDEXES = {
    "kgqa_entity_scoped_name_index": (
        "CREATE INDEX kgqa_entity_scoped_name_index IF NOT EXISTS "
        "FOR (e:Entity) ON (e.index_id, e.name)"
    ),
    "kgqa_entity_scoped_canonical_name_index": (
        "CREATE INDEX kgqa_entity_scoped_canonical_name_index IF NOT EXISTS "
        "FOR (e:Entity) ON (e.index_id, e.canonical_name)"
    ),
    "kgqa_text_unit_scoped_doc_id_index": (
        "CREATE INDEX kgqa_text_unit_scoped_doc_id_index IF NOT EXISTS "
        "FOR (t:TextUnit) ON (t.index_id, t.doc_id)"
    ),
    "kgqa_relationship_scoped_relation_index": (
        "CREATE INDEX kgqa_relationship_scoped_relation_index IF NOT EXISTS "
        "FOR ()-[r:RELATION]-() ON (r.index_id, r.relation)"
    ),
}

CONSTRAINTS = list(SCHEMA_CONSTRAINTS.values())
INDEXES = list(SCHEMA_INDEXES.values())
EXPECTED_CONSTRAINT_NAMES = sorted(SCHEMA_CONSTRAINTS)
EXPECTED_INDEX_NAMES = sorted(SCHEMA_INDEXES)

CLEAR_INDEX = """
MATCH (n)
WHERE n.index_id = $index_id
DETACH DELETE n
"""

CLEAR_STAGING_INDEX = """
MATCH (n)
WHERE n.index_id = $staging_index_id
DETACH DELETE n
"""

PROMOTE_STAGED_NODES = """
MATCH (n)
WHERE n.index_id = $staging_index_id
SET n.index_id = $canonical_index_id
"""

PROMOTE_STAGED_RELATIONSHIPS = """
MATCH ()-[r]->()
WHERE r.index_id = $staging_index_id
SET r.index_id = $canonical_index_id
"""

PROMOTE_STAGED_INDEX = """
MATCH (i:KGQAIndex {id: $staging_index_id})
SET i.id = $canonical_index_id,
    i.index_id = $canonical_index_id,
    i.promoted_from = $staging_index_id,
    i.promoted_at = datetime()
"""

MERGE_INDEX = """
MERGE (i:KGQAIndex {id: $index_id})
ON CREATE SET i.created_at = datetime()
SET i.index_id = $index_id,
    i.output_path = $output_path,
    i.database = $database,
    i.updated_at = datetime()
"""

MERGE_TEXT_UNIT = """
MERGE (i:KGQAIndex {id: $index_id})
MERGE (d:Document {index_id: $index_id, id: $doc_id})
SET d.source_path = $source_path,
    d.index_id = $index_id
MERGE (t:TextUnit {index_id: $index_id, id: $id})
SET t.index_id = $index_id,
    t.doc_id = $doc_id,
    t.source_path = $source_path,
    t.chunk_index = $chunk_index,
    t.text = $text,
    t.n_tokens = $n_tokens
MERGE (i)-[:HAS_DOCUMENT]->(d)
MERGE (d)-[:HAS_CHUNK]->(t)
"""

MERGE_ENTITY = """
MERGE (i:KGQAIndex {id: $index_id})
MERGE (e:Entity {index_id: $index_id, id: $id})
SET e.index_id = $index_id,
    e.name = $name,
    e.canonical_name = $canonical_name,
    e.type = $type,
    e.description = $description,
    e.aliases = $aliases,
    e.evidence_chunk_ids = $evidence_chunk_ids,
    e.confidence = $confidence,
    e.metadata = $metadata
MERGE (i)-[:HAS_ENTITY]->(e)
WITH e
UNWIND $evidence_chunk_ids AS chunk_id
MATCH (t:TextUnit {index_id: $index_id, id: chunk_id})
MERGE (e)-[:MENTIONED_IN]->(t)
"""

MERGE_RELATIONSHIP = """
MATCH (i:KGQAIndex {id: $index_id})
MATCH (source:Entity {index_id: $index_id, id: $source_entity_id})
MATCH (target:Entity {index_id: $index_id, id: $target_entity_id})
MERGE (source)-[r:RELATION {index_id: $index_id, id: $id}]->(target)
SET r.index_id = $index_id,
    r.relation = $relation,
    r.description = $description,
    r.confidence = $confidence,
    r.extraction_count = $extraction_count,
    r.evidence_chunk_ids = $evidence_chunk_ids,
    r.original_relations = $original_relations,
    r.metadata = $metadata
MERGE (schema:RelationSchema {index_id: $index_id, name: $relation})
SET schema.index_id = $index_id,
    schema.description = $relation
MERGE (i)-[:HAS_RELATION_SCHEMA]->(schema)
"""
