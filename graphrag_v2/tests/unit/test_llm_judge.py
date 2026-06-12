"""Tests for the reproducible LLM QA judge utility."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_judge_module():
    module_path = REPO_ROOT / "scripts" / "judge_qa_with_llm.py"
    spec = importlib.util.spec_from_file_location("judge_qa_with_llm", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_judge_response_accepts_fenced_json():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        """```json
        {
          "semantic_label": "correct",
          "grounded_label": "grounded",
          "semantic_score": 1,
          "grounded_score": 1,
          "reason": "same answer"
        }
        ```"""
    )

    assert parsed == {
        "semantic_label": "correct",
        "grounded_label": "grounded",
        "semantic_score": 1.0,
        "grounded_score": 1.0,
        "reason": "same answer",
        "judge_error": None,
    }


def test_parse_judge_response_normalizes_alias_labels_and_10_point_scores():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        json.dumps(
            {
                "semantic_label": "semantically correct",
                "grounded_label": "fully grounded",
                "semantic_score": 10,
                "grounded_score": 10,
                "reason": "same entity and directly supported",
            }
        )
    )

    assert parsed == {
        "semantic_label": "correct",
        "grounded_label": "grounded",
        "semantic_score": 1.0,
        "grounded_score": 1.0,
        "reason": "same entity and directly supported",
        "judge_error": None,
    }


def test_parse_judge_response_falls_back_to_free_form_text():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        "The system answer is semantically correct and fully grounded in the retrieved evidence."
    )

    assert parsed == {
        "semantic_label": "correct",
        "grounded_label": "grounded",
        "semantic_score": 1.0,
        "grounded_score": 1.0,
        "reason": "The system answer is semantically correct and fully grounded in the retrieved evidence.",
        "judge_error": None,
    }


def test_parse_judge_response_accepts_alternate_judge_keys():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        json.dumps(
            {
                "semantic_correctness": "semantically correct",
                "groundedness": "fully supported",
                "semantic_score": 1,
                "grounded_score": 1,
                "reason": "The system answer matches the gold answer and the evidence supports it.",
            }
        )
    )

    assert parsed["semantic_label"] == "correct"
    assert parsed["grounded_label"] == "grounded"
    assert parsed["semantic_score"] == 1.0
    assert parsed["grounded_score"] == 1.0


def test_parse_judge_response_accepts_yes_correct_and_numeric_labels():
    judge = _load_judge_module()

    for semantic_label, grounded_label in [("yes", "yes"), (1, 1), ("correct", "correct")]:
        parsed = judge.parse_judge_response(
            json.dumps(
                {
                    "semantic_label": semantic_label,
                    "grounded_label": grounded_label,
                    "semantic_score": 1,
                    "grounded_score": 1,
                    "reason": "The answer matches the gold answer and is supported by evidence.",
                }
            )
        )
        assert parsed["semantic_label"] == "correct"
        assert parsed["grounded_label"] == "grounded"


def test_parse_judge_response_uses_reason_when_label_keys_are_missing():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        json.dumps(
            {
                "semantic_score": 1,
                "grounded_score": 1,
                "reason": "The answer is semantically correct and fully grounded in the evidence.",
            }
        )
    )

    assert parsed["semantic_label"] == "correct"
    assert parsed["grounded_label"] == "grounded"


def test_parse_judge_response_reconciles_conflicting_labels_with_scores_and_reason():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        json.dumps(
            {
                "semantic_label": "incorrect",
                "grounded_label": "ungrounded",
                "semantic_score": 1,
                "grounded_score": 1,
                "reason": (
                    "The system answer matches the gold answer exactly and is "
                    "fully supported by the retrieved evidence."
                ),
            }
        )
    )

    assert parsed["semantic_label"] == "correct"
    assert parsed["grounded_label"] == "grounded"
    assert parsed["semantic_score"] == 1.0
    assert parsed["grounded_score"] == 1.0


def test_parse_judge_response_rejects_empty_response():
    judge = _load_judge_module()

    try:
        judge.parse_judge_response("")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty judge responses must raise")


def test_parse_judge_response_rejects_truncated_json():
    judge = _load_judge_module()

    try:
        judge.parse_judge_response('{"semantic_label": "correct", "grounded_label"')
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("truncated JSON judge responses must raise")


def test_parse_judge_response_salvages_truncated_json_with_labels():
    judge = _load_judge_module()

    parsed = judge.parse_judge_response(
        '{\n'
        '  "semantic_label": "no",\n'
        '  "grounded_label": "yes",\n'
        '  "semantic_score": 0.0,\n'
        '  "grounded_score": 1.0,\n'
        '  "reason": "The gold answer is not matched'
    )

    assert parsed["semantic_label"] == "incorrect"
    assert parsed["grounded_label"] == "grounded"
    assert parsed["semantic_score"] == 0.0
    assert parsed["grounded_score"] == 1.0
    assert parsed["judge_error"] is None


def test_parse_semantic_judge_response_marks_grounding_not_evaluated():
    judge = _load_judge_module()

    parsed = judge.parse_semantic_judge_response(
        json.dumps(
            {
                "semantic_label": "correct",
                "semantic_score": 1,
                "reason": "The two answers are aliases.",
            }
        )
    )

    assert parsed == {
        "semantic_label": "correct",
        "grounded_label": "not_evaluated",
        "semantic_score": 1.0,
        "grounded_score": 0.0,
        "reason": "The two answers are aliases.",
        "judge_error": None,
    }


def test_semantic_only_judge_prompt_excludes_retrieved_evidence():
    judge = _load_judge_module()

    messages = judge.build_semantic_judge_messages(
        {
            "question": "Where was the director born?",
            "gold_answer": "Toronto",
            "answer": "Toronto, Canada",
            "citation_recall": 1.0,
        }
    )

    prompt = messages[1]["content"]
    assert "Gold Answer: Toronto" in prompt
    assert "System Answer: Toronto, Canada" in prompt
    assert "Retrieved Evidence" not in prompt
    assert "Citation Recall" not in prompt


def test_semantic_only_case_diagnosis_is_lightweight():
    judge = _load_judge_module()

    assert (
        judge.diagnose_semantic_only_case(
            {"answer": "证据不足，无法回答该问题。"},
            {"semantic_label": "incorrect"},
        )
        == "answerer_refusal"
    )
    assert (
        judge.diagnose_semantic_only_case(
            {"answer": "Paris"},
            {"semantic_label": "incorrect"},
        )
        == "semantic_mismatch"
    )


def test_diagnose_retrieval_gap_for_missing_required_citations():
    judge = _load_judge_module()
    case = {
        "answer": "Insufficient evidence.",
        "gold_answer": "Film B",
        "citation_recall": 0.75,
        "required_citations": ["a", "b", "c", "d"],
        "citations": ["a", "b", "c"],
    }
    judgment = {
        "semantic_label": "incorrect",
        "grounded_label": "grounded",
    }

    assert judge.diagnose_case(case, judgment, [], real_extraction=False) == "retrieval_gap"


def test_diagnose_data_or_judge_issue_when_gold_not_in_retrieved_text():
    judge = _load_judge_module()
    case = {
        "answer": "The evidence does not specify where Henry Ford died.",
        "gold_answer": "Dearborn",
        "citation_recall": 1.0,
        "required_citations": ["a"],
        "citations": ["a"],
    }
    judgment = {
        "semantic_label": "incorrect",
        "grounded_label": "grounded",
    }
    text_evidence = [
        {"chunk_id": "a", "text": "Henry Ford founded the Ford Motor Company."}
    ]

    assert (
        judge.diagnose_case(case, judgment, text_evidence, real_extraction=False)
        == "data_or_judge_issue"
    )


def test_gold_support_real_llm_answerer_is_not_real_extraction_report():
    judge = _load_judge_module()

    report = {
        "benchmark": {
            "name": "HotpotQA Isolated",
            "index_mode": "one gold HotpotQA context index per question",
            "answerer": "real_llm",
        }
    }

    assert judge._is_real_extraction_report(report) is False


def test_real_extraction_benchmark_is_real_extraction_report():
    judge = _load_judge_module()

    report = {
        "benchmark": {
            "name": "HotpotQA real LLM extraction graph benchmark",
            "index_mode": "real_llm_extracted_graph",
        }
    }

    assert judge._is_real_extraction_report(report) is True


def test_judge_report_with_fake_client_writes_summary(tmp_path: Path):
    judge = _load_judge_module()
    input_report = tmp_path / "qa-report.json"
    output_json = tmp_path / "judge.json"
    output_md = tmp_path / "judge.md"
    input_report.write_text(
        json.dumps(
            {
                "benchmark": {"name": "fixture"},
                "cases": [
                    {
                        "ordinal": 1,
                        "id": "case_1",
                        "question": "Where?",
                        "gold_answer": "Paris",
                        "answer": "Paris",
                        "citation_recall": 1.0,
                        "strict_pass": True,
                        "required_citations": ["chunk_1"],
                        "citations": ["chunk_1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        provider_name = "fake"
        model_name = "fake-model"
        model = "fake-model"
        mock_mode = False

        def __init__(self):
            self.total_calls = 0

        def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False):
            self.total_calls += 1
            assert "Gold Answer: Paris" in messages[-1]["content"]
            return json.dumps(
                {
                    "semantic_label": "correct",
                    "grounded_label": "grounded",
                    "semantic_score": 1.0,
                    "grounded_score": 1.0,
                    "reason": "matches",
                }
            )

        def get_stats(self):
            return {"total_calls": self.total_calls}

    report = judge.run_judge(
        input_report_path=input_report,
        output_json_path=output_json,
        output_markdown_path=output_md,
        llm_client=FakeClient(),
    )

    assert report["summary"]["semantic_correct"] == 1
    assert report["judgments"][0]["diagnosis_label"] == "ok"
    assert '"semantic_label": "correct"' in report["judgments"][0]["judge_raw_response"]
    assert output_json.exists()
    assert output_md.exists()


def test_judge_passes_response_format_json_to_supported_client(tmp_path: Path):
    judge = _load_judge_module()
    input_report = tmp_path / "qa-report.json"
    input_report.write_text(
        json.dumps(
            {
                "benchmark": {"name": "fixture"},
                "cases": [
                    {
                        "ordinal": 1,
                        "id": "case_1",
                        "question": "Where?",
                        "gold_answer": "Paris",
                        "answer": "Paris",
                        "citation_recall": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class JsonModeClient:
        provider_name = "fake"
        model_name = "fake-model"
        model = "fake-model"
        mock_mode = False
        supports_response_format_json = True

        def __init__(self):
            self.call_kwargs = []

        def chat_completion(self, **kwargs):
            self.call_kwargs.append(kwargs)
            return json.dumps(
                {
                    "semantic_label": "correct",
                    "grounded_label": "grounded",
                    "semantic_score": 1.0,
                    "grounded_score": 1.0,
                    "reason": "matches",
                }
            )

        def get_stats(self):
            return {"total_calls": len(self.call_kwargs)}

    client = JsonModeClient()
    report = judge.run_judge(
        input_report_path=input_report,
        output_json_path=None,
        output_markdown_path=None,
        llm_client=client,
    )

    assert report["summary"]["judge_errors"] == 0
    assert client.call_kwargs[0]["response_format_json"] is True


def test_judge_retries_empty_response_with_shorter_evidence(tmp_path: Path):
    judge = _load_judge_module()
    index_path = tmp_path / "index"
    index_path.mkdir()
    input_report = tmp_path / "qa-report.json"
    output_json = tmp_path / "judge.json"
    output_md = tmp_path / "judge.md"
    input_report.write_text(
        json.dumps(
            {
                "benchmark": {"name": "fixture"},
                "cases": [
                    {
                        "ordinal": 1,
                        "id": "case_1",
                        "question": "Where?",
                        "gold_answer": "Paris",
                        "answer": "Paris",
                        "citation_recall": 1.0,
                        "required_citations": ["chunk_1"],
                        "citations": ["chunk_1"],
                        "case_index_path": str(index_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import pandas as pd

    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "text": "Paris " + ("evidence " * 200),
            }
        ]
    ).to_parquet(index_path / "text_units.parquet")

    class RetryClient:
        provider_name = "fake"
        model_name = "fake-model"
        model = "fake-model"
        mock_mode = False

        def __init__(self):
            self.calls = []

        def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False):
            self.calls.append(messages[-1]["content"])
            if len(self.calls) == 1:
                return ""
            assert len(messages[-1]["content"]) < len(self.calls[0])
            return json.dumps(
                {
                    "semantic_label": "correct",
                    "grounded_label": "grounded",
                    "semantic_score": 1.0,
                    "grounded_score": 1.0,
                    "reason": "retry succeeded",
                }
            )

        def get_stats(self):
            return {"total_calls": len(self.calls)}

    client = RetryClient()
    report = judge.run_judge(
        input_report_path=input_report,
        output_json_path=output_json,
        output_markdown_path=output_md,
        llm_client=client,
        max_evidence_chars=1200,
        retry_evidence_chars=[80],
    )

    judgment = report["judgments"][0]
    assert len(client.calls) == 2
    assert judgment["semantic_label"] == "correct"
    assert judgment["judge_error"] is None
    assert judgment["judge_attempts"] == 2
    assert judgment["judge_retry_evidence_chars"] == [80]
    assert judgment["judge_errors"] == ["ValueError: Judge response was empty."]


def test_judge_checkpoint_resume_reruns_unresolved_only(tmp_path: Path):
    judge = _load_judge_module()
    input_report = tmp_path / "qa-report.json"
    checkpoint = tmp_path / "judge.checkpoint.jsonl"
    input_report.write_text(
        json.dumps(
            {
                "benchmark": {"name": "fixture"},
                "cases": [
                    {
                        "ordinal": 1,
                        "id": "bad",
                        "question": "Bad?",
                        "gold_answer": "Paris",
                        "answer": "Paris",
                        "citation_recall": 1.0,
                    },
                    {
                        "ordinal": 2,
                        "id": "good",
                        "question": "Good?",
                        "gold_answer": "Rome",
                        "answer": "Rome",
                        "citation_recall": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    class FirstClient:
        provider_name = "fake"
        model_name = "fake-model"
        model = "fake-model"
        mock_mode = False

        def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False):
            if "Question: Bad?" in messages[-1]["content"]:
                return ""
            return json.dumps(
                {
                    "semantic_label": "correct",
                    "grounded_label": "grounded",
                    "semantic_score": 1.0,
                    "grounded_score": 1.0,
                    "reason": "ok",
                }
            )

        def get_stats(self):
            return {}

    first = judge.run_judge(
        input_report_path=input_report,
        output_json_path=None,
        output_markdown_path=None,
        llm_client=FirstClient(),
        checkpoint_jsonl_path=checkpoint,
    )

    assert first["summary"]["judge_errors"] == 1
    assert first["summary"]["judge_unresolved"] == 1
    assert [item["judge_unresolved"] for item in first["judgments"]] == [True, False]
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 2

    class ResumeClient:
        provider_name = "fake"
        model_name = "fake-model"
        model = "fake-model"
        mock_mode = False

        def __init__(self):
            self.questions = []

        def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False):
            content = messages[-1]["content"]
            self.questions.append(content)
            assert "Question: Good?" not in content
            return json.dumps(
                {
                    "semantic_label": "correct",
                    "grounded_label": "grounded",
                    "semantic_score": 1.0,
                    "grounded_score": 1.0,
                    "reason": "retry ok",
                }
            )

        def get_stats(self):
            return {}

    resume_client = ResumeClient()
    resumed = judge.run_judge(
        input_report_path=input_report,
        output_json_path=None,
        output_markdown_path=None,
        llm_client=resume_client,
        checkpoint_jsonl_path=checkpoint,
        resume=True,
    )

    assert len(resume_client.questions) == 1
    assert resumed["summary"]["judge_errors"] == 0
    assert resumed["summary"]["judge_unresolved"] == 0
    assert [item["semantic_label"] for item in resumed["judgments"]] == ["correct", "correct"]
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 3


def test_judge_slices_cases_by_ordinal(tmp_path: Path):
    judge = _load_judge_module()
    input_report = tmp_path / "qa-report.json"
    input_report.write_text(
        json.dumps(
            {
                "benchmark": {"name": "fixture"},
                "cases": [
                    {"ordinal": 1, "id": "one", "question": "One?", "gold_answer": "A", "answer": "A"},
                    {"ordinal": 2, "id": "two", "question": "Two?", "gold_answer": "B", "answer": "B"},
                    {"ordinal": 3, "id": "three", "question": "Three?", "gold_answer": "C", "answer": "C"},
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        provider_name = "fake"
        model_name = "fake-model"
        model = "fake-model"
        mock_mode = False

        def chat_completion(self, messages, temperature=0.0, max_tokens=None, stream=False):
            return json.dumps(
                {
                    "semantic_label": "correct",
                    "grounded_label": "grounded",
                    "semantic_score": 1.0,
                    "grounded_score": 1.0,
                    "reason": "ok",
                }
            )

        def get_stats(self):
            return {}

    report = judge.run_judge(
        input_report_path=input_report,
        output_json_path=None,
        output_markdown_path=None,
        llm_client=FakeClient(),
        slice_start=2,
        slice_end=3,
    )

    assert [item["ordinal"] for item in report["judgments"]] == [2, 3]
    assert report["summary"]["total"] == 2


def test_reparse_judge_report_recovers_field_variant_raw_response():
    judge = _load_judge_module()
    report = {
        "judge": {"prompt_version": "test"},
        "summary": {"total": 1, "judge_errors": 1},
        "judgments": [
            {
                "ordinal": 1,
                "id": "case_1",
                "question": "Where?",
                "gold_answer": "Paris",
                "system_answer": "Paris",
                "semantic_label": "incorrect",
                "grounded_label": "ungrounded",
                "semantic_score": 0.0,
                "grounded_score": 0.0,
                "reason": "",
                "judge_error": "ValueError: prior parse failed",
                "judge_raw_response": json.dumps(
                    {
                        "semantic_correctness": "semantically correct",
                        "groundedness": "fully supported",
                        "semantic_score": 1,
                        "grounded_score": 1,
                        "reason": "The answer matches and is supported.",
                    }
                ),
                "diagnosis_label": "answerer_refusal",
            }
        ],
    }

    reparsed = judge.reparse_judge_report(report)

    judgment = reparsed["judgments"][0]
    assert judgment["semantic_label"] == "correct"
    assert judgment["grounded_label"] == "grounded"
    assert judgment["judge_error"] is None
    assert judgment["diagnosis_label"] == "ok"
    assert reparsed["summary"]["semantic_correct"] == 1
    assert reparsed["summary"]["judge_errors"] == 0


def test_merge_judge_reports_replaces_failed_judgment_with_successful_retry():
    judge = _load_judge_module()
    base = {
        "judge": {"prompt_version": "test"},
        "summary": {"total": 2, "judge_errors": 1},
        "judgments": [
            {
                "ordinal": 1,
                "id": "case_1",
                "semantic_label": "incorrect",
                "grounded_label": "ungrounded",
                "semantic_score": 0.0,
                "grounded_score": 0.0,
                "judge_error": "ValueError: empty",
                "diagnosis_label": "data_or_judge_issue",
            },
            {
                "ordinal": 2,
                "id": "case_2",
                "semantic_label": "correct",
                "grounded_label": "grounded",
                "semantic_score": 1.0,
                "grounded_score": 1.0,
                "judge_error": None,
                "diagnosis_label": "ok",
            },
        ],
    }
    retry = {
        "judge": {"prompt_version": "test"},
        "summary": {"total": 1, "judge_errors": 0},
        "judgments": [
            {
                "ordinal": 1,
                "id": "case_1",
                "semantic_label": "correct",
                "grounded_label": "grounded",
                "semantic_score": 1.0,
                "grounded_score": 1.0,
                "judge_error": None,
                "diagnosis_label": "ok",
            }
        ],
    }

    merged = judge.merge_judge_reports(base, [retry])

    assert [item["semantic_label"] for item in merged["judgments"]] == ["correct", "correct"]
    assert merged["summary"]["semantic_correct"] == 2
    assert merged["summary"]["judge_errors"] == 0
    assert merged["judge"]["merge_sources"] == ["memory://patch-1"]
