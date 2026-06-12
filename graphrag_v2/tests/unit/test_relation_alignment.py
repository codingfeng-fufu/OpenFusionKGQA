"""Tests for relation alignment."""

from graphrag_v2.graph_fusion.relation_alignment import (
    align_open_relation_detail,
    align_relation,
    align_relation_detail,
)
from graphrag_v2.graph_fusion.relation_schema import (
    OPEN_RELATION_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
)


def test_align_exact_relation():
    relation, score = align_relation("uses", "Technology", "Technology")

    assert relation == "uses"
    assert score > 0.95


def test_align_unknown_relation_to_related_to():
    relation, score = align_relation("mentions vaguely", "Technology", "Database")

    assert relation == "related_to"
    assert score >= 0.55


def test_align_open_relation_preserves_predicate_without_schema_penalty():
    alignment = align_open_relation_detail("answered with")

    assert alignment.canonical_relation == "answered_with"
    assert alignment.score == 1.0
    assert alignment.schema_version == OPEN_RELATION_SCHEMA_VERSION
    assert alignment.endpoint_compatible is True
    assert alignment.endpoint_reason == "schema_disabled"


def test_align_cross_lingual_relation_alias():
    relation, score = align_relation("使用", "Technology", "Technology")

    assert relation == "uses"
    assert score > 0.95


def test_align_detail_reports_schema_version_and_endpoint_rules():
    alignment = align_relation_detail(
        "fallback_to",
        source_type="Person",
        target_type="Format",
    )

    assert alignment.canonical_relation == "has_fallback"
    assert alignment.schema_version == RELATION_SCHEMA_VERSION
    assert alignment.endpoint_compatible is False
    assert alignment.endpoint_reason == "incompatible_source_type"


def test_align_relation_override():
    relation, score = align_relation(
        "employs",
        "Technology",
        "Technology",
        relation_aliases={"employs": "uses"},
    )

    assert relation == "uses"
    assert score > 0.95
