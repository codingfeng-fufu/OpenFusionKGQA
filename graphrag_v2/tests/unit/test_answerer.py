"""Tests for QA answerers."""

import pytest

from graphrag_v2.qa import (
    CandidateAnswerExtractor,
    CommunityEvidence,
    GraphGroundedQA,
    GraphEvidence,
    LinkedEntity,
    LLMAnswerer,
    MockAnswerer,
    RelationshipEvidence,
    TextEvidence,
    format_qa_result_json,
)
from graphrag_v2.qa.prompts import build_prompt
from graphrag_v2.qa.prompts import QA_ANSWER_PROMPT_VERSION
from graphrag_v2.qa.sources import LocalArtifactQADataSource
from graphrag_v2.qa.models import QAResult


def test_mock_answerer_formats_local_answer():
    answerer = MockAnswerer()
    graph_evidence = GraphEvidence(
        linked_entities=[
            LinkedEntity(
                id="entity_1",
                name="GraphRAG",
                canonical_name="graphrag",
                type="Technology",
                description="GraphRAG",
                score=1.0,
            )
        ],
        relationships=[
            RelationshipEvidence(
                id="rel_1",
                source_entity_id="entity_2",
                target_entity_id="entity_1",
                source_name="微软",
                target_name="GraphRAG",
                relation="uses",
                description="微软开发了 GraphRAG",
                confidence=0.9,
                extraction_count=1,
                evidence_chunk_ids=["chunk_1"],
                score=0.9,
            )
        ],
        text_chunk_ids=["chunk_1"],
    )

    answer = answerer.answer(
        "GraphRAG 是什么？",
        "local",
        graph_evidence,
        [],
        [
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text="GraphRAG 是微软开发的一种技术。",
                score=1.0,
            )
        ],
    )

    assert "GraphRAG" in answer
    assert "最相关关系" in answer


def test_llm_answerer_uses_client_when_available():
    class StubClient:
        mock_mode = False
        total_errors = 0
        calls = []

        def chat_completion(self, messages, temperature=0.2, **kwargs):
            self.calls.append(messages)
            return '{"candidate_id": "cand_1", "answer_text": "Toronto", "supported": true, "reason": "directly stated"}'

    client = StubClient()
    answerer = LLMAnswerer(llm_client=client)
    answer = answerer.answer(
        "Where was the director born?",
        "local",
        GraphEvidence(),
        [],
        [
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text="The director was born in Toronto.",
                score=1.0,
            )
        ],
    )

    assert answer == "Toronto"
    assert "Candidate Answers:" in client.calls[0][1]["content"]


def test_llm_answerer_passes_response_format_json_to_supported_client():
    class StubClient:
        mock_mode = False
        total_errors = 0
        supports_response_format_json = True

        def __init__(self):
            self.call_kwargs = []

        def chat_completion(self, **kwargs):
            self.call_kwargs.append(kwargs)
            return '{"candidate_id": "cand_1", "answer_text": "Toronto", "supported": true}'

    client = StubClient()
    answerer = LLMAnswerer(llm_client=client)

    answer = answerer.answer(
        "Where was the director born?",
        "local",
        GraphEvidence(),
        [],
        _toronto_text_evidence(),
    )

    assert answer == "Toronto"
    assert client.call_kwargs[0]["response_format_json"] is True


def test_candidate_answer_extractor_finds_typed_text_spans():
    candidates = CandidateAnswerExtractor().extract(
        question="Where was the director born?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text="The director was born in Toronto on 12 July 1979.",
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Toronto"].answer_type == "location"
    assert by_text["Toronto"].source_chunk_id == "chunk_1"
    assert by_text["Toronto"].source_span == [25, 32]
    assert by_text["12 July 1979"].answer_type == "date"


def test_candidate_answer_extractor_finds_comparison_options_quotes_and_named_birthplace():
    candidates = CandidateAnswerExtractor().extract(
        question="Which film was released more recently, The Flight Of Mr. Mckinley or La Terrazza?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text='The Flight Of Mr. Mckinley is a 1975 film. La Terrazza is a 1980 film. "Cahiers du cinéma" covered it.',
                score=1.0,
            ),
            TextEvidence(
                chunk_id="chunk_2",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=1,
                text="Ricci was born Teodoro Ricci in Rome on October 23, 1927. Selena married Chris Pérez.",
                score=1.0,
            ),
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert "The Flight Of Mr. Mckinley" in by_text
    assert by_text["La Terrazza"].confidence == 0.98
    assert by_text["Cahiers du cinéma"].answer_type == "entity"
    assert by_text["Rome"].answer_type == "location"
    assert by_text["Chris Pérez"].answer_type == "entity"
    assert by_text["1980"].confidence < by_text["La Terrazza"].confidence


def test_candidate_answer_extractor_finds_hotpot_location_and_school_patterns():
    candidates = CandidateAnswerExtractor().extract(
        question="Where did the director of film Dragonheart 3 graduate from?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text=(
                    "David Lynch was born January 20, 1946 in Missoula, Montana. "
                    "Anne Gadegaard was born in Århus, Region Midtjylland. "
                    "Colin Teague studied at Redroofs Theatre School."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Missoula, Montana"].answer_type == "location"
    assert by_text["Århus"].answer_type == "location"
    assert by_text["Redroofs Theatre School"].answer_type == "organization"


def test_candidate_answer_extractor_finds_nationality_and_early_year_answers():
    candidates = CandidateAnswerExtractor().extract(
        question="What is the date of birth and nationality of the father?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text=(
                    "Liudolf of Brunswick (c. 1003 - 23 April 1038) was Matilda's father. "
                    "Kaumualii was a vassal of the unified Kingdom of Hawaiʻi."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["1003"].answer_type == "date"
    assert by_text["Kingdom of Hawaiʻi"].answer_type == "nationality"


def test_candidate_answer_extractor_finds_birthplace_after_father_chain():
    candidates = CandidateAnswerExtractor().extract(
        question="Where was the father of Marianus V Of Arborea born?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text=(
                    "Marianus V of Arborea was the son of Brancaleone Doria. "
                    "Brancaleone Doria was born in Sardinia and belonged to "
                    "an influential family of the Republic of Genoa."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Sardinia"].answer_type == "location"
    assert by_text["Sardinia"].confidence > by_text["Republic of Genoa"].confidence


def test_candidate_answer_extractor_finds_death_place_for_father_question():
    candidates = CandidateAnswerExtractor().extract(
        question="Where did Edward Randolph's father die?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text=(
                    "William Randolph was Edward Randolph's father. "
                    "William Randolph died in the Colony of Virginia in 1671."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Colony of Virginia"].answer_type == "location"
    assert by_text["Colony of Virginia"].confidence >= 0.9


def test_candidate_answer_extractor_finds_sibling_in_law_answer():
    candidates = CandidateAnswerExtractor().extract(
        question="Who is William Stanley (Battle Of Bosworth)'s sibling-in-law?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text=(
                    "William Stanley was the brother of Thomas Stanley. "
                    "Thomas Stanley married Lady Margaret Beaufort. "
                    "Eleanor was another member of the family."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Lady Margaret Beaufort"].answer_type == "entity"
    assert by_text["Lady Margaret Beaufort"].confidence > by_text["Eleanor"].confidence


def test_candidate_answer_extractor_finds_second_wife_sibling_in_law_answer():
    candidates = CandidateAnswerExtractor().extract(
        question="Who is William Stanley (Battle Of Bosworth)'s sibling-in-law?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text=(
                    "Thomas Stanley's marriage to Eleanor constituted a powerful alliance. "
                    "In 1472, he married his second wife Lady Margaret Beaufort, whose son "
                    "Henry Tudor was the leading Lancastrian claimant."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Lady Margaret Beaufort"].answer_type == "entity"
    assert by_text["Lady Margaret Beaufort"].confidence > by_text["Eleanor"].confidence


def test_candidate_answer_extractor_trims_birthplace_before_relative_clause():
    candidates = CandidateAnswerExtractor().extract(
        question="What is the place of birth of the director of film Kailangan Ko'Y Ikaw?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text=(
                    "Joyce E. Bernal (born May 6, 1968) is a Filipina film and "
                    "television director in the Philippines who started as a film editor."
                ),
                score=1.0,
            )
        ],
    )

    by_text = {candidate.answer_text: candidate for candidate in candidates}
    assert by_text["Philippines"].answer_type == "location"
    assert "Philippines who started as a film editor" not in by_text


def test_candidate_answer_extractor_requires_question_options_in_evidence():
    candidates = CandidateAnswerExtractor().extract(
        question="Which city is warmer, Paris or Lyon?",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text="The retrieved evidence discusses Toronto and Vancouver.",
                score=1.0,
            )
        ],
    )

    candidate_texts = {candidate.answer_text for candidate in candidates}
    assert "Paris" not in candidate_texts
    assert "Lyon" not in candidate_texts


def test_llm_answerer_rejects_answer_outside_candidate_set():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.2, **kwargs):
            return '{"candidate_id": "cand_999", "answer_text": "Paris", "supported": true}'

    answerer = LLMAnswerer(llm_client=StubClient())

    with pytest.raises(ValueError, match="outside extracted candidate answers"):
        answerer.answer(
            "Where was the director born?",
            "local",
            GraphEvidence(),
            [],
            [
                TextEvidence(
                    chunk_id="chunk_1",
                    doc_id="doc_1",
                    source_path="/tmp/doc.md",
                    chunk_index=0,
                    text="The director was born in Toronto.",
                    score=1.0,
                )
            ],
        )


def test_llm_answerer_refuses_before_llm_when_no_candidates():
    class ExplodingClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.2, **kwargs):
            raise AssertionError("LLM should not be called without candidates")

    answerer = LLMAnswerer(llm_client=ExplodingClient())

    answer = answerer.answer(
        "Where was the director born?",
        "local",
        GraphEvidence(),
        [],
        [
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text="No usable answer is stated here.",
                score=1.0,
            )
        ],
    )

    assert "证据不足" in answer


def test_llm_answerer_retries_selection_when_supported_false_with_candidates():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def __init__(self):
            self.calls = []

        def chat_completion(self, messages, temperature=0.2, **kwargs):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return '{"candidate_id": "", "answer_text": "", "supported": false}'
            return '{"candidate_id": "cand_1", "answer_text": "Toronto", "supported": true}'

    client = StubClient()
    answerer = LLMAnswerer(llm_client=client)

    answer = answerer.answer(
        "Where was the director born?",
        "local",
        GraphEvidence(),
        [],
        _toronto_text_evidence(),
    )

    assert answer == "Toronto"
    assert len(client.calls) == 2


def test_llm_answerer_prompt_includes_candidate_source_chunks():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def __init__(self):
            self.user_content = ""

        def chat_completion(self, messages, temperature=0.2, **kwargs):
            self.user_content = messages[1]["content"]
            return '{"candidate_id": "cand_1", "answer_text": "Sardinia", "supported": true}'

    client = StubClient()
    answerer = LLMAnswerer(llm_client=client)
    answer = answerer.answer(
        "Where was the father born?",
        "local",
        GraphEvidence(),
        [],
        [
            TextEvidence(
                chunk_id="chunk_birth",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text="Brancaleone Doria was born in Sardinia.",
                score=1.0,
            )
        ],
    )

    assert answer == "Sardinia"
    assert "cand_1" in client.user_content
    assert "chunk_birth" in client.user_content


def test_llm_answerer_system_prompt_requires_exact_candidate_selection():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def __init__(self):
            self.system_content = ""

        def chat_completion(self, messages, temperature=0.2, **kwargs):
            self.system_content = messages[0]["content"]
            return '{"candidate_id": "cand_1", "answer_text": "Sardinia", "supported": true}'

    client = StubClient()
    answerer = LLMAnswerer(llm_client=client)

    answer = answerer.answer(
        "Where was the father born?",
        "local",
        GraphEvidence(),
        [],
        [
            TextEvidence(
                chunk_id="chunk_birth",
                doc_id="doc_1",
                source_path="/tmp/hotpot.md",
                chunk_index=0,
                text="Brancaleone Doria was born in Sardinia.",
                score=1.0,
            )
        ],
    )

    assert answer == "Sardinia"
    assert "select exactly one candidate_id" in client.system_content
    assert "Do not invent a new answer_text" in client.system_content


def test_llm_answer_prompt_keeps_long_text_evidence_tail():
    prompt = build_prompt(
        question="Where did the filmmaker graduate from?",
        route="local",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text=("prefix " * 80)
                + "BA in Critical Social Thought from Mount Holyoke College.",
                score=1.0,
            )
        ],
    )

    assert "Mount Holyoke College" in prompt


def test_llm_answer_prompt_requires_candidate_extraction_before_refusal():
    prompt = build_prompt(
        question="Where was the director born?",
        route="local",
        graph_evidence=GraphEvidence(),
        community_evidence=[],
        text_evidence=[
            TextEvidence(
                chunk_id="chunk_1",
                doc_id="doc_1",
                source_path="/tmp/doc.md",
                chunk_index=0,
                text="The director was born in Toronto.",
                score=1.0,
            )
        ],
    )

    assert "Candidate answer extraction" in prompt
    assert "Before refusing" in prompt
    assert "only if no candidate answer" in prompt


def test_llm_answerer_requires_explicit_client():
    answerer = LLMAnswerer()

    with pytest.raises(ValueError, match="requires a configured real LLM client"):
        answerer.answer("GraphRAG 是什么？", "local", GraphEvidence(), [], [])


def test_llm_answerer_removes_think_blocks():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.2):
            return '<think>hidden reasoning</think>\n{"candidate_id": "cand_1", "answer_text": "Toronto", "supported": true}'

    answerer = LLMAnswerer(llm_client=StubClient())

    answer = answerer.answer(
        "Where was the director born?",
        "local",
        GraphEvidence(),
        [],
        _toronto_text_evidence(),
    )

    assert answer == "Toronto"


def test_llm_answerer_keeps_text_after_stray_closing_think_tag():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.2):
            return 'hidden reasoning</think>\n{"candidate_id": "cand_1", "answer_text": "Toronto", "supported": true}'

    answerer = LLMAnswerer(llm_client=StubClient())

    answer = answerer.answer(
        "Where was the director born?",
        "local",
        GraphEvidence(),
        [],
        _toronto_text_evidence(),
    )

    assert answer == "Toronto"


def test_llm_answerer_rejects_mock_client():
    class StubClient:
        mock_mode = True

        def chat_completion(self, messages, temperature=0.2):
            raise AssertionError("mock client should not be called")

    answerer = LLMAnswerer(llm_client=StubClient())

    try:
        answerer.answer(
            "Where was the director born?",
            "local",
            GraphEvidence(),
            [],
            _toronto_text_evidence(),
        )
    except ValueError as exc:
        assert "requires a configured real LLM client" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_llm_answerer_rejects_empty_response():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.2):
            return "   "

    answerer = LLMAnswerer(llm_client=StubClient())

    try:
        answerer.answer(
            "Where was the director born?",
            "local",
            GraphEvidence(),
            [],
            _toronto_text_evidence(),
        )
    except ValueError as exc:
        assert "empty response" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_llm_answerer_rejects_client_error_after_retry():
    class StubClient:
        mock_mode = False
        total_errors = 0

        def chat_completion(self, messages, temperature=0.2):
            self.total_errors += 1
            return "fallback text"

    answerer = LLMAnswerer(llm_client=StubClient())

    try:
        answerer.answer(
            "Where was the director born?",
            "local",
            GraphEvidence(),
            [],
            _toronto_text_evidence(),
        )
    except ValueError as exc:
        assert "failed after retry attempts" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mock_answerer_returns_insufficient_evidence_message():
    answerer = MockAnswerer()
    answer = answerer.answer("GraphRAG 是什么？", "local", GraphEvidence(), [], [])

    assert "证据不足" in answer


def test_qa_result_json_contains_refusal_contract():
    result = QAResult(
        question="未知问题？",
        route="local",
        answer="证据不足，无法回答该问题。",
        refusal_reason="no_source_evidence",
        source_provider="json",
        metadata={"routing_reason": "test"},
    )

    payload = format_qa_result_json(result)

    assert '"refused": true' in payload
    assert '"refusal_reason": "no_source_evidence"' in payload
    assert '"metadata"' in payload


def test_graph_grounded_qa_refusal_does_not_call_answerer():
    class EmptyDataSource(LocalArtifactQADataSource):
        def __init__(self):
            pass

        provider = "json"

        def metadata(self):
            return {}

        def entities(self):
            return []

        def relationships(self):
            return []

        def communities(self):
            return []

        def community_reports(self):
            return []

        def text_units(self):
            return []

    class ExplodingAnswerer:
        def answer(self, *args, **kwargs):
            raise AssertionError("answerer should not be called without source evidence")

    result = GraphGroundedQA(
        data_source=EmptyDataSource(),
        answerer=ExplodingAnswerer(),
    ).ask("未知问题？")

    assert result.refused is True
    assert result.refusal_reason == "no_source_evidence"


def test_graph_grounded_qa_global_question_uses_text_when_no_communities(tmp_path):
    import pandas as pd

    index_path = tmp_path / "index"
    index_path.mkdir()
    pd.DataFrame(
        [
            {
                "id": "entity_graphrag",
                "name": "GraphRAG",
                "canonical_name": "graphrag",
                "type": "Technology",
                "description": "GraphRAG",
                "aliases": ["GraphRAG"],
                "evidence_chunk_ids": ["chunk_1"],
                "confidence": 1.0,
                "metadata": "{}",
            }
        ]
    ).to_parquet(index_path / "entities.parquet")
    pd.DataFrame(
        [
            {
                "id": "rel_neo4j_graph_database",
                "source_entity_id": "entity_neo4j",
                "target_entity_id": "entity_graph_database",
                "source_name": "Neo4j",
                "target_name": "Graph Database",
                "relation": "is_a",
                "description": "Neo4j is a graph database.",
                "confidence": 0.8,
                "extraction_count": 1,
                "evidence_chunk_ids": ["chunk_a_neo4j"],
                "metadata": "{}",
            }
        ]
    ).to_parquet(index_path / "relationships.parquet")
    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_z_graphrag",
                "doc_id": "doc_1",
                "source_path": "examples/docs/graphrag.md",
                "chunk_index": 0,
                "text": "GraphRAG combines graph evidence and language models for question answering.",
                "n_tokens": 12,
                "metadata": "{}",
            },
            {
                "chunk_id": "chunk_a_neo4j",
                "doc_id": "doc_2",
                "source_path": "examples/docs/neo4j.txt",
                "chunk_index": 0,
                "text": "Neo4j is a graph database used to store entities and relationships.",
                "n_tokens": 10,
                "metadata": "{}",
            }
        ]
    ).to_parquet(index_path / "text_units.parquet")

    qa = GraphGroundedQA.from_index(index_path, prefer_neo4j=False)
    result = qa.ask("这批文档主要讲了哪些主题？")

    assert result.route == "global"
    assert result.refusal_reason is None
    assert "chunk_z_graphrag" in result.citations
    assert "chunk_a_neo4j" in result.citations
    assert "GraphRAG" in result.answer


def test_graph_grounded_qa_metadata_includes_prompt_version_and_query_trace():
    class StubDataSource(LocalArtifactQADataSource):
        def __init__(self):
            pass

        provider = "json"

        def metadata(self):
            return {}

        def entities(self):
            return [
                {
                    "id": "entity_graphrag",
                    "name": "GraphRAG",
                    "canonical_name": "GraphRAG",
                    "type": "Technology",
                    "description": "GraphRAG",
                    "aliases": ["GraphRAG"],
                    "evidence_chunk_ids": ["chunk_1"],
                }
            ]

        def relationships(self):
            return [
                {
                    "id": "rel_1",
                    "source_entity_id": "entity_graphrag",
                    "target_entity_id": "entity_neo4j",
                    "source_name": "GraphRAG",
                    "target_name": "Neo4j",
                    "relation": "uses",
                    "description": "GraphRAG uses Neo4j",
                    "confidence": 0.9,
                    "extraction_count": 1,
                    "evidence_chunk_ids": ["chunk_1"],
                }
            ]

        def communities(self):
            return []

        def community_reports(self):
            return []

        def text_units(self):
            return [
                {
                    "chunk_id": "chunk_1",
                    "doc_id": "doc_1",
                    "source_path": "/tmp/doc.md",
                    "chunk_index": 0,
                    "text": "GraphRAG uses Neo4j as graph evidence.",
                }
            ]

    result = GraphGroundedQA(data_source=StubDataSource()).ask("GraphRAG 是什么？")

    assert result.metadata["answer_prompt_version"] == QA_ANSWER_PROMPT_VERSION
    trace = result.metadata["query_trace"]
    assert trace["route"] == "local"
    assert trace["routing_reason"] == result.metadata["routing_reason"]
    assert trace["linked_entities"] == [
        {"id": "entity_graphrag", "name": "GraphRAG", "score": 1.0}
    ]
    assert trace["retrieved_relationships"] == [
        {
            "id": "rel_1",
            "source": "GraphRAG",
            "relation": "uses",
            "target": "Neo4j",
            "score": 0.7783,
            "hop": 1,
        }
    ]
    assert trace["retrieved_communities"] == []
    assert trace["retrieved_text_chunks"] == [
        {
            "chunk_id": "chunk_1",
            "source_path": "/tmp/doc.md",
            "chunk_index": 0,
        }
    ]


def test_graph_grounded_qa_adaptive_trace_includes_second_hop_relationship():
    class StubDataSource(LocalArtifactQADataSource):
        def __init__(self):
            pass

        provider = "json"

        def metadata(self):
            return {}

        def entities(self):
            return [
                {
                    "id": "entity_film_a",
                    "name": "Film A",
                    "canonical_name": "Film A",
                    "type": "Film",
                    "description": "Film A",
                    "aliases": ["Film A"],
                    "evidence_chunk_ids": ["chunk_film_a"],
                },
                {
                    "id": "entity_film_b",
                    "name": "Film B",
                    "canonical_name": "Film B",
                    "type": "Film",
                    "description": "Film B",
                    "aliases": ["Film B"],
                    "evidence_chunk_ids": ["chunk_film_b"],
                },
                {
                    "id": "entity_director_a",
                    "name": "Director A",
                    "canonical_name": "Director A",
                    "type": "Person",
                    "description": "Director A",
                    "aliases": ["Director A"],
                    "evidence_chunk_ids": ["chunk_director_a"],
                },
                {
                    "id": "entity_director_b",
                    "name": "Director B",
                    "canonical_name": "Director B",
                    "type": "Person",
                    "description": "Director B",
                    "aliases": ["Director B"],
                    "evidence_chunk_ids": ["chunk_director_b"],
                },
            ]

        def relationships(self):
            return [
                {
                    "id": "rel_film_pair",
                    "source_entity_id": "entity_film_a",
                    "target_entity_id": "entity_film_b",
                    "source_name": "Film A",
                    "target_name": "Film B",
                    "relation": "supports_answer",
                    "description": "Film A and Film B are compared.",
                    "confidence": 1.0,
                    "extraction_count": 1,
                    "evidence_chunk_ids": ["chunk_film_a", "chunk_film_b"],
                },
                {
                    "id": "rel_film_b_director_a",
                    "source_entity_id": "entity_film_b",
                    "target_entity_id": "entity_director_a",
                    "source_name": "Film B",
                    "target_name": "Director A",
                    "relation": "supports_answer",
                    "description": "Film B requires Director A evidence.",
                    "confidence": 1.0,
                    "extraction_count": 1,
                    "evidence_chunk_ids": ["chunk_film_b", "chunk_director_a"],
                },
                {
                    "id": "rel_director_a_director_b",
                    "source_entity_id": "entity_director_a",
                    "target_entity_id": "entity_director_b",
                    "source_name": "Director A",
                    "target_name": "Director B",
                    "relation": "supports_answer",
                    "description": "Director B died in 2012.",
                    "confidence": 1.0,
                    "extraction_count": 1,
                    "evidence_chunk_ids": ["chunk_director_a", "chunk_director_b"],
                },
            ]

        def communities(self):
            return []

        def community_reports(self):
            return []

        def text_units(self):
            return [
                _text_unit("chunk_film_a", 0, "Film A was directed by Director A."),
                _text_unit("chunk_film_b", 1, "Film B was directed by Director B."),
                _text_unit("chunk_director_a", 2, "Director A died in 1953."),
                _text_unit("chunk_director_b", 3, "Director B died in 2012."),
            ]

    result = GraphGroundedQA(data_source=StubDataSource()).ask(
        "Which film has the director who died later, Film A or Film B?"
    )

    assert "chunk_director_b" in result.citations
    trace = result.metadata["query_trace"]
    assert trace["adaptive_enabled"] is True
    assert trace["adaptive_triggered"] is True
    assert "died later" in trace["matched_adaptive_cues"]
    assert trace["hop_plan"] == [1, 2, 3]
    assert trace["relationship_count_by_hop"] == {1: 2, 2: 1}
    relationships = {
        relationship["id"]: relationship
        for relationship in trace["retrieved_relationships"]
    }
    assert relationships["rel_director_a_director_b"]["hop"] == 2


def _text_unit(chunk_id: str, chunk_index: int, text: str):
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc_1",
        "source_path": "/tmp/doc.md",
        "chunk_index": chunk_index,
        "text": text,
    }


def _toronto_text_evidence():
    return [
        TextEvidence(
            chunk_id="chunk_1",
            doc_id="doc_1",
            source_path="/tmp/doc.md",
            chunk_index=0,
            text="The director was born in Toronto.",
            score=1.0,
        )
    ]
