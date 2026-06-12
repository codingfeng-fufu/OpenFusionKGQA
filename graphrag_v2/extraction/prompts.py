"""Prompts for LLM-based knowledge extraction."""

ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT = """You extract knowledge graph candidates from text.

Return only compact valid JSON. Do not include markdown fences or explanatory text.
Every relationship must reference entity names that appear in the entities list.
Use the source text as the only evidence. Do not invent facts.
"""

ENTITY_RELATION_EXTRACTION_PROMPT = """Extract entities and relationships from this text unit.

Return this JSON shape:
{{
  "entities": [
    {{
      "name": "Entity name",
      "type": "Entity type",
      "description": "Short description grounded in the text",
      "confidence": 0.85
    }}
  ],
  "relationships": [
    {{
      "source": "Source entity name",
      "target": "Target entity name",
      "relation": "short_relation_name",
      "description": "Short relationship description grounded in the text",
      "confidence": 0.80
    }}
  ]
}}

Rules:
- Use confidence values between 0 and 1.
- Prefer confidence values between 0.6 and 0.95 for reliable evidence.
- Keep relation names concise, lowercase, and underscore-separated when possible.
- Keep every description under 12 words.
- Return compact one-line JSON when possible.
- Preserve open predicates from the source text; do not force every relation into
  a fixed ontology.
- Also extract explicit type and storage statements as relationships:
  - If the text says "X is a Y", include entity Y and relation "is_a".
  - If the text says "X stores Y", "X is used to store Y", or similar, include
    relation "stores". Use the explicit stored object phrase as the target
    entity when present.
- Extract at most 8 entities and at most 8 relationships.
- Return empty lists if no reliable entities or relationships are present.

Text unit:
chunk_id: {chunk_id}
source_path: {source_path}
text:
{text}
"""


def build_extraction_prompt(
    chunk_id: str,
    source_path: str,
    text: str,
) -> str:
    """Build the user prompt for one text unit."""
    return ENTITY_RELATION_EXTRACTION_PROMPT.format(
        chunk_id=chunk_id,
        source_path=source_path,
        text=text,
    )
