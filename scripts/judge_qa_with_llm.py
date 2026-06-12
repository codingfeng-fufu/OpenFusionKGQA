#!/usr/bin/env python3
"""Judge QA answers with a reproducible LLM prompt and emit diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from graphrag_v2.config import load_config
from graphrag_v2.config.defaults import DEFAULT_CHAT_MODEL_ID
from graphrag_v2.llm import create_chat_provider


JUDGE_PROMPT_VERSION = "2026-06-09.qa-judge.v1"
SEMANTIC_LABELS = {"correct", "partial", "incorrect"}
GROUNDED_LABELS = {"grounded", "partially_grounded", "ungrounded", "not_evaluated"}
REFUSAL_RE = re.compile(
    r"\b(insufficient evidence|cannot answer|can't answer|does not specify|not provided|无法回答|证据不足)\b",
    flags=re.IGNORECASE,
)


SYSTEM_PROMPT = """You are a strict but fair QA evaluator.
Judge whether the system answer is semantically correct relative to the gold answer,
and whether the system answer is supported by the provided retrieved evidence.
Accept paraphrases, aliases, demonyms/country equivalents, date formatting variants,
and more specific locations when they answer the same question.
Do not require exact string match.
If the system says the evidence is insufficient, judge semantic correctness against
the gold answer, but judge groundedness against the retrieved evidence.
Return only JSON with keys: semantic_label, grounded_label, semantic_score,
grounded_score, reason.
"""

SEMANTIC_ONLY_SYSTEM_PROMPT = """You are a strict but fair QA evaluator.
Judge whether the system answer is semantically correct relative to the gold answer.
Accept paraphrases, aliases, demonyms/country equivalents, date formatting variants,
and more specific locations when they answer the same question.
Do not require exact string match.
Return only JSON with keys: semantic_label, semantic_score, reason.
Use semantic_label as one of: correct, partial, incorrect.
"""


def run_judge(
    *,
    input_report_path: Path,
    output_json_path: Path | None,
    output_markdown_path: Path | None,
    llm_client: Any,
    limit: int | None = None,
    max_evidence_chars: int = 1200,
    retry_evidence_chars: list[int] | None = None,
    checkpoint_jsonl_path: Path | None = None,
    resume: bool = False,
    slice_start: int | None = None,
    slice_end: int | None = None,
    semantic_only: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_report = json.loads(input_report_path.read_text(encoding="utf-8"))
    cases = extract_cases(input_report)
    if limit is not None:
        cases = cases[:limit]
    cases = slice_cases(cases, slice_start=slice_start, slice_end=slice_end)
    real_extraction = _is_real_extraction_report(input_report)
    checkpoint_by_key = (
        load_checkpoint_judgments(checkpoint_jsonl_path) if checkpoint_jsonl_path and resume else {}
    )

    judgments = []
    for ordinal, case in enumerate(cases, start=1):
        case_ordinal = int(case.get("ordinal") or ordinal)
        checkpoint_judgment = _checkpoint_lookup(
            checkpoint_by_key,
            case=case,
            case_ordinal=case_ordinal,
        )
        if checkpoint_judgment and not _is_judge_unresolved(checkpoint_judgment):
            judgments.append(checkpoint_judgment)
            continue

        attempt = judge_case_with_retries(
            case=case,
            llm_client=llm_client,
            max_evidence_chars=max_evidence_chars,
            retry_evidence_chars=retry_evidence_chars,
            semantic_only=semantic_only,
        )
        text_evidence = attempt["text_evidence"]
        response = attempt["response"]
        parsed = attempt["parsed"]
        judgment = {
            "ordinal": case_ordinal,
            "id": str(case.get("id") or ""),
            "question": str(case.get("question") or ""),
            "gold_answer": str(case.get("gold_answer") or case.get("answer_gold") or ""),
            "system_answer": str(case.get("answer") or case.get("system_answer") or ""),
            "semantic_label": parsed["semantic_label"],
            "grounded_label": parsed["grounded_label"],
            "semantic_score": parsed["semantic_score"],
            "grounded_score": parsed["grounded_score"],
            "reason": parsed["reason"],
            "judge_error": parsed["judge_error"],
            "judge_attempts": attempt["attempts"],
            "judge_errors": attempt["errors"],
            "judge_retry_evidence_chars": attempt["retry_evidence_chars"],
            "judge_attempt_records": attempt["attempt_records"],
            "judge_raw_response": response,
            "judge_unresolved": bool(parsed["judge_error"]),
            "answer_em_proxy": bool(case.get("answer_contains_gold", False)),
            "citation_recall": float(case.get("citation_recall", 0.0) or 0.0),
            "strict_pass": bool(case.get("passed", False)),
            "diagnosis_label": (
                diagnose_semantic_only_case(case, parsed)
                if semantic_only
                else diagnose_case(
                    case,
                    parsed,
                    text_evidence,
                    real_extraction=real_extraction,
                )
            ),
        }
        judgments.append(judgment)
        append_checkpoint_judgment(checkpoint_jsonl_path, judgment)

    report = {
        "judge": {
            "provider": str(getattr(llm_client, "provider_name", "unknown")),
            "model": str(
                getattr(llm_client, "model_name", None)
                or getattr(llm_client, "model", "unknown")
            ),
            "mock_mode": bool(getattr(llm_client, "mock_mode", False)),
            "prompt_version": JUDGE_PROMPT_VERSION,
            "input_report": str(input_report_path),
            "checkpoint_jsonl": str(checkpoint_jsonl_path) if checkpoint_jsonl_path else None,
            "resume": resume,
            "slice_start": slice_start,
            "slice_end": slice_end,
            "mode": "semantic_only" if semantic_only else "semantic_and_grounded",
        },
        "summary": summarize_judgments(judgments, elapsed_seconds=time.perf_counter() - started_at),
        "llm_stats": _client_stats(llm_client),
        "judgments": judgments,
    }
    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if output_markdown_path:
        output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        output_markdown_path.write_text(format_markdown_report(report), encoding="utf-8")
    return report


def slice_cases(
    cases: list[dict[str, Any]],
    *,
    slice_start: int | None,
    slice_end: int | None,
) -> list[dict[str, Any]]:
    if slice_start is None and slice_end is None:
        return cases
    start = 1 if slice_start is None else slice_start
    end = len(cases) if slice_end is None else slice_end
    if start < 1:
        raise ValueError("--slice-start must be >= 1")
    if end < start:
        raise ValueError("--slice-end must be >= --slice-start")
    selected = []
    for index, case in enumerate(cases, start=1):
        ordinal = int(case.get("ordinal") or index)
        if start <= ordinal <= end:
            selected.append(case)
    return selected


def load_checkpoint_judgments(
    checkpoint_jsonl_path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if checkpoint_jsonl_path is None or not checkpoint_jsonl_path.exists():
        return {}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(
        checkpoint_jsonl_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            judgment = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{checkpoint_jsonl_path}: malformed checkpoint JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(judgment, dict):
            raise ValueError(f"{checkpoint_jsonl_path}: checkpoint line {line_number} is not an object")
        for key in _judgment_merge_keys(judgment):
            by_key[key] = judgment
    return by_key


def append_checkpoint_judgment(
    checkpoint_jsonl_path: Path | None,
    judgment: dict[str, Any],
) -> None:
    if checkpoint_jsonl_path is None:
        return
    checkpoint_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judgment, ensure_ascii=False, sort_keys=True) + "\n")


def _checkpoint_lookup(
    checkpoint_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    case: dict[str, Any],
    case_ordinal: int,
) -> dict[str, Any] | None:
    keys = [("ordinal", str(case_ordinal))]
    case_id = case.get("id")
    if case_id not in (None, ""):
        keys.append(("id", str(case_id)))
    for key in keys:
        if key in checkpoint_by_key:
            return checkpoint_by_key[key]
    return None


def _is_judge_unresolved(judgment: dict[str, Any]) -> bool:
    return bool(judgment.get("judge_unresolved", bool(judgment.get("judge_error"))))


def reparse_judge_report(report: dict[str, Any]) -> dict[str, Any]:
    """Reparse stored raw judge responses and refresh labels/summary offline."""
    reparsed = dict(report)
    judgments = [dict(item) for item in report.get("judgments", [])]
    for judgment in judgments:
        raw_response = str(judgment.get("judge_raw_response") or "")
        if not raw_response.strip():
            continue
        try:
            parsed = parse_judge_response(raw_response)
        except Exception as exc:
            judgment["judge_error"] = f"{exc.__class__.__name__}: {exc}"
            judgment["judge_unresolved"] = True
            continue
        judgment.update(
            {
                "semantic_label": parsed["semantic_label"],
                "grounded_label": parsed["grounded_label"],
                "semantic_score": parsed["semantic_score"],
                "grounded_score": parsed["grounded_score"],
                "reason": parsed["reason"],
                "judge_error": parsed["judge_error"],
                "judge_unresolved": bool(parsed["judge_error"]),
            }
        )
        if parsed["semantic_label"] == "correct":
            judgment["diagnosis_label"] = "ok"
    reparsed["judgments"] = judgments
    reparsed["summary"] = summarize_judgments(
        judgments,
        elapsed_seconds=float((report.get("summary") or {}).get("elapsed_seconds") or 0.0),
    )
    judge = dict(reparsed.get("judge") or {})
    judge["reparse_applied"] = True
    reparsed["judge"] = judge
    return reparsed


def merge_judge_reports(
    base_report: dict[str, Any],
    patch_reports: list[dict[str, Any]],
    *,
    merge_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Merge retry/reparse judge reports into a base report by ordinal/id."""
    merged = dict(base_report)
    judgments = [dict(item) for item in base_report.get("judgments", [])]
    position_by_key: dict[tuple[str, str], int] = {}
    for index, judgment in enumerate(judgments):
        for key in _judgment_merge_keys(judgment):
            position_by_key[key] = index

    sources = merge_sources or [f"memory://patch-{index}" for index in range(1, len(patch_reports) + 1)]
    for patch_report in patch_reports:
        for patch_judgment in patch_report.get("judgments", []):
            patch = dict(patch_judgment)
            target_index = None
            for key in _judgment_merge_keys(patch):
                if key in position_by_key:
                    target_index = position_by_key[key]
                    break
            if target_index is None:
                target_index = len(judgments)
                judgments.append(patch)
                for key in _judgment_merge_keys(patch):
                    position_by_key[key] = target_index
                continue
            if _should_replace_judgment(judgments[target_index], patch):
                judgments[target_index] = patch

    merged["judgments"] = judgments
    merged["summary"] = summarize_judgments(
        judgments,
        elapsed_seconds=sum(
            float((report.get("summary") or {}).get("elapsed_seconds") or 0.0)
            for report in [base_report, *patch_reports]
        ),
    )
    judge = dict(merged.get("judge") or {})
    judge["merge_sources"] = sources
    merged["judge"] = judge
    return merged


def _judgment_merge_keys(judgment: dict[str, Any]) -> list[tuple[str, str]]:
    keys = []
    ordinal = judgment.get("ordinal")
    if ordinal not in (None, ""):
        keys.append(("ordinal", str(ordinal)))
    case_id = judgment.get("id")
    if case_id not in (None, ""):
        keys.append(("id", str(case_id)))
    return keys


def _should_replace_judgment(
    current: dict[str, Any],
    patch: dict[str, Any],
) -> bool:
    current_has_error = bool(current.get("judge_error"))
    patch_has_error = bool(patch.get("judge_error"))
    if current_has_error and not patch_has_error:
        return True
    if current_has_error == patch_has_error:
        return _judgment_quality_score(patch) >= _judgment_quality_score(current)
    return False


def _judgment_quality_score(judgment: dict[str, Any]) -> float:
    score = 0.0
    if not judgment.get("judge_error"):
        score += 10.0
    score += float(judgment.get("semantic_score") or 0.0)
    score += float(judgment.get("grounded_score") or 0.0)
    if str(judgment.get("semantic_label") or "") == "correct":
        score += 1.0
    if str(judgment.get("grounded_label") or "") == "grounded":
        score += 1.0
    return score


def judge_case_with_retries(
    *,
    case: dict[str, Any],
    llm_client: Any,
    max_evidence_chars: int,
    retry_evidence_chars: list[int] | None = None,
    semantic_only: bool = False,
) -> dict[str, Any]:
    retry_chars = [] if semantic_only else (
        list(retry_evidence_chars) if retry_evidence_chars is not None else [
            max(80, max_evidence_chars // 2),
            max(80, max_evidence_chars // 4),
        ]
    )
    evidence_char_limits = [0] if semantic_only else [max_evidence_chars] + [
        value for value in retry_chars if value > 0 and value != max_evidence_chars
    ]
    errors: list[str] = []
    attempt_records: list[dict[str, Any]] = []
    response = ""
    text_evidence: list[dict[str, str]] = []
    for index, evidence_chars in enumerate(evidence_char_limits):
        if semantic_only:
            text_evidence = []
            messages = build_semantic_judge_messages(case)
        else:
            text_evidence = load_case_text_evidence(
                case,
                max_chars_per_chunk=evidence_chars,
            )
            messages = build_judge_messages(case, text_evidence)
        try:
            request_kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 700,
            }
            if bool(getattr(llm_client, "supports_response_format_json", False)):
                request_kwargs["response_format_json"] = True
            response = llm_client.chat_completion(**request_kwargs)
            parsed = (
                parse_semantic_judge_response(response)
                if semantic_only
                else parse_judge_response(response)
            )
            attempt_records.append(
                {
                    "attempt": index + 1,
                    "evidence_chars": evidence_chars,
                    "raw_response": response,
                    "error": None,
                }
            )
            return {
                "parsed": parsed,
                "response": response,
                "text_evidence": text_evidence,
                "attempts": index + 1,
                "errors": errors,
                "retry_evidence_chars": evidence_char_limits[1:index + 1],
                "attempt_records": attempt_records,
            }
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            errors.append(error)
            attempt_records.append(
                {
                    "attempt": index + 1,
                    "evidence_chars": evidence_chars,
                    "raw_response": response,
                    "error": error,
                }
            )

    return {
        "parsed": {
            "semantic_label": "incorrect",
            "grounded_label": "ungrounded",
            "semantic_score": 0.0,
            "grounded_score": 0.0,
            "reason": "",
            "judge_error": errors[-1] if errors else "Unknown judge error",
        },
        "response": response,
        "text_evidence": text_evidence,
        "attempts": len(evidence_char_limits),
        "errors": errors,
        "retry_evidence_chars": evidence_char_limits[1:],
        "attempt_records": attempt_records,
    }


def build_judge_messages(
    case: dict[str, Any],
    text_evidence: list[dict[str, str]],
) -> list[dict[str, str]]:
    evidence_lines = []
    for item in text_evidence:
        evidence_lines.append(
            f"- {item['chunk_id']}: {item['text']}"
        )
    evidence = "\n".join(evidence_lines) if evidence_lines else "- none"
    user_prompt = "\n".join(
        [
            f"Question: {case.get('question', '')}",
            f"Gold Answer: {case.get('gold_answer') or case.get('answer_gold') or ''}",
            f"System Answer: {case.get('answer') or case.get('system_answer') or ''}",
            f"Citation Recall: {case.get('citation_recall', '')}",
            "Retrieved Evidence:",
            evidence,
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_semantic_judge_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = "\n".join(
        [
            f"Question: {case.get('question', '')}",
            f"Gold Answer: {case.get('gold_answer') or case.get('answer_gold') or ''}",
            f"System Answer: {case.get('answer') or case.get('system_answer') or ''}",
        ]
    )
    return [
        {"role": "system", "content": SEMANTIC_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_semantic_judge_response(response: str) -> dict[str, Any]:
    if not response.strip():
        raise ValueError("Judge response was empty.")
    try:
        payload = _extract_json_object(response)
    except ValueError:
        partial = _salvage_partial_semantic_judge_response(response)
        if partial is not None:
            return partial
        if _looks_like_json_response(response):
            raise
        reason = response.strip()
        semantic_label = _normalize_semantic_label(reason)
        return {
            "semantic_label": semantic_label,
            "grounded_label": "not_evaluated",
            "semantic_score": _score(None, semantic_label),
            "grounded_score": 0.0,
            "reason": reason,
            "judge_error": None,
        }

    reason = str(payload.get("reason") or "").strip()
    semantic_label = _normalize_semantic_label(
        _first_payload_value(
            payload,
            (
                "semantic_label",
                "semantic_correctness",
                "semantic",
                "correctness",
                "answer_correctness",
            ),
            fallback=reason,
        )
    )
    semantic_label = _reconcile_semantic_label(
        semantic_label,
        raw_score=payload.get("semantic_score"),
        reason=reason,
    )
    return {
        "semantic_label": semantic_label,
        "grounded_label": "not_evaluated",
        "semantic_score": _score(payload.get("semantic_score"), semantic_label),
        "grounded_score": 0.0,
        "reason": reason,
        "judge_error": None,
    }


def parse_judge_response(response: str) -> dict[str, Any]:
    if not response.strip():
        raise ValueError("Judge response was empty.")
    try:
        payload = _extract_json_object(response)
    except ValueError:
        partial = _salvage_partial_judge_response(response)
        if partial is not None:
            return partial
        if _looks_like_json_response(response):
            raise
        reason = response.strip()
        semantic_label = _normalize_semantic_label(reason)
        grounded_label = _normalize_grounded_label(reason)
        return {
            "semantic_label": semantic_label,
            "grounded_label": grounded_label,
            "semantic_score": _score(None, semantic_label),
            "grounded_score": _score(None, grounded_label),
            "reason": reason,
            "judge_error": None,
        }

    reason = str(payload.get("reason") or "").strip()
    semantic_label = _normalize_semantic_label(
        _first_payload_value(
            payload,
            (
                "semantic_label",
                "semantic_correctness",
                "semantic",
                "correctness",
                "answer_correctness",
            ),
            fallback=reason,
        )
    )
    semantic_label = _reconcile_semantic_label(
        semantic_label,
        raw_score=payload.get("semantic_score"),
        reason=reason,
    )
    grounded_label = _normalize_grounded_label(
        _first_payload_value(
            payload,
            (
                "grounded_label",
                "groundedness",
                "grounding",
                "support_label",
                "supported_label",
                "evidence_support",
            ),
            fallback=reason,
        )
    )
    grounded_label = _reconcile_grounded_label(
        grounded_label,
        raw_score=payload.get("grounded_score"),
        reason=reason,
    )
    return {
        "semantic_label": semantic_label,
        "grounded_label": grounded_label,
        "semantic_score": _score(payload.get("semantic_score"), semantic_label),
        "grounded_score": _score(payload.get("grounded_score"), grounded_label),
        "reason": reason,
        "judge_error": None,
    }


def _salvage_partial_judge_response(response: str) -> dict[str, Any] | None:
    if not _looks_like_json_response(response):
        return None
    semantic_value = _partial_json_value(response, "semantic_label")
    grounded_value = _partial_json_value(response, "grounded_label")
    semantic_score_value = _partial_json_value(response, "semantic_score")
    grounded_score_value = _partial_json_value(response, "grounded_score")
    reason = _partial_json_value(response, "reason")
    if semantic_value in (None, "") or grounded_value in (None, ""):
        return None
    semantic_label = _normalize_semantic_label(semantic_value)
    grounded_label = _normalize_grounded_label(grounded_value)
    return {
        "semantic_label": semantic_label,
        "grounded_label": grounded_label,
        "semantic_score": _score(semantic_score_value, semantic_label),
        "grounded_score": _score(grounded_score_value, grounded_label),
        "reason": str(reason or "").strip(),
        "judge_error": None,
    }


def _salvage_partial_semantic_judge_response(response: str) -> dict[str, Any] | None:
    if not _looks_like_json_response(response):
        return None
    semantic_value = _partial_json_value(response, "semantic_label")
    semantic_score_value = _partial_json_value(response, "semantic_score")
    reason = _partial_json_value(response, "reason")
    if semantic_value in (None, ""):
        return None
    semantic_label = _normalize_semantic_label(semantic_value)
    return {
        "semantic_label": semantic_label,
        "grounded_label": "not_evaluated",
        "semantic_score": _score(semantic_score_value, semantic_label),
        "grounded_score": 0.0,
        "reason": str(reason or "").strip(),
        "judge_error": None,
    }


def _partial_json_value(response: str, key: str) -> Any:
    pattern = re.compile(
        rf'"{re.escape(key)}"\s*:\s*("(?P<quoted>[^"]*)"|(?P<bare>true|false|null|-?\d+(?:\.\d+)?|yes|no))',
        flags=re.IGNORECASE,
    )
    match = pattern.search(response)
    if not match:
        return None
    quoted = match.group("quoted")
    if quoted is not None:
        return quoted
    bare = (match.group("bare") or "").lower()
    if bare == "true":
        return True
    if bare == "false":
        return False
    if bare == "null":
        return None
    try:
        return float(bare)
    except ValueError:
        return bare


def diagnose_case(
    case: dict[str, Any],
    judgment: dict[str, Any],
    text_evidence: list[dict[str, str]],
    *,
    real_extraction: bool,
) -> str:
    if judgment.get("semantic_label") == "correct":
        return "ok"

    citation_recall = float(case.get("citation_recall", 0.0) or 0.0)
    required = [str(item) for item in case.get("required_citations", [])]
    citations = [str(item) for item in case.get("citations", [])]
    missing_required = bool(set(required) - set(citations))
    if citation_recall < 1.0 or missing_required:
        return "extraction_gap" if real_extraction else "retrieval_gap"

    answer = str(case.get("answer") or case.get("system_answer") or "")
    gold = str(case.get("gold_answer") or case.get("answer_gold") or "")
    if REFUSAL_RE.search(answer):
        if gold and not _gold_supported_by_text(gold, text_evidence):
            return "data_or_judge_issue"
        return "answerer_refusal"

    if gold and text_evidence and not _gold_supported_by_text(gold, text_evidence):
        return "data_or_judge_issue"
    return "answerer_refusal"


def diagnose_semantic_only_case(
    case: dict[str, Any],
    judgment: dict[str, Any],
) -> str:
    if judgment.get("semantic_label") == "correct":
        return "ok"
    answer = str(case.get("answer") or case.get("system_answer") or "")
    if REFUSAL_RE.search(answer):
        return "answerer_refusal"
    return "semantic_mismatch"


def summarize_judgments(
    judgments: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    total = len(judgments)
    semantic_correct = sum(1 for item in judgments if item["semantic_label"] == "correct")
    semantic_partial = sum(1 for item in judgments if item["semantic_label"] == "partial")
    semantic_incorrect = sum(1 for item in judgments if item["semantic_label"] == "incorrect")
    grounded = sum(1 for item in judgments if item["grounded_label"] == "grounded")
    partially_grounded = sum(
        1 for item in judgments if item["grounded_label"] == "partially_grounded"
    )
    ungrounded = sum(1 for item in judgments if item["grounded_label"] == "ungrounded")
    grounded_not_evaluated = sum(
        1 for item in judgments if item["grounded_label"] == "not_evaluated"
    )
    diagnosis_counts = {}
    for item in judgments:
        diagnosis = str(item.get("diagnosis_label") or "unknown")
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    return {
        "total": total,
        "semantic_correct": semantic_correct,
        "semantic_partial": semantic_partial,
        "semantic_incorrect": semantic_incorrect,
        "semantic_accuracy": _ratio(semantic_correct, total),
        "semantic_correct_or_partial": semantic_correct + semantic_partial,
        "semantic_correct_or_partial_rate": _ratio(semantic_correct + semantic_partial, total),
        "grounded": grounded,
        "partially_grounded": partially_grounded,
        "ungrounded": ungrounded,
        "grounded_not_evaluated": grounded_not_evaluated,
        "grounded_rate": _ratio(grounded, total),
        "grounded_or_partial_rate": _ratio(grounded + partially_grounded, total),
        "semantically_correct_and_grounded": sum(
            1
            for item in judgments
            if item["semantic_label"] == "correct" and item["grounded_label"] == "grounded"
        ),
        "average_semantic_score": _mean([item["semantic_score"] for item in judgments]),
        "average_grounded_score": _mean([item["grounded_score"] for item in judgments]),
        "judge_errors": sum(1 for item in judgments if item.get("judge_error")),
        "judge_unresolved": sum(1 for item in judgments if _is_judge_unresolved(item)),
        "diagnosis_counts": diagnosis_counts,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def extract_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("cases"), list):
        return list(report["cases"])
    for key in ("real_llm_smoke", "offline_qa"):
        section = report.get(key)
        if isinstance(section, dict) and isinstance(section.get("cases"), list):
            return list(section["cases"])
    raise ValueError("Input report does not contain top-level, real_llm_smoke, or offline_qa cases.")


def load_case_text_evidence(
    case: dict[str, Any],
    *,
    max_chars_per_chunk: int,
) -> list[dict[str, str]]:
    citations = [str(item) for item in case.get("citations", [])]
    index_path_value = case.get("case_index_path") or case.get("index_path")
    if not citations or not index_path_value:
        return []
    text_units_path = Path(str(index_path_value)) / "text_units.parquet"
    if not text_units_path.exists():
        return []
    try:
        frame = pd.read_parquet(text_units_path)
    except Exception:
        return []
    by_chunk = {
        str(row["chunk_id"]): str(row.get("text", ""))
        for _, row in frame.iterrows()
        if str(row.get("chunk_id", ""))
    }
    evidence = []
    for chunk_id in citations:
        text = by_chunk.get(chunk_id)
        if text is None:
            continue
        evidence.append(
            {
                "chunk_id": chunk_id,
                "text": " ".join(text.split())[:max_chars_per_chunk],
            }
        )
    return evidence


def format_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# LLM QA Judge Report",
        "",
        f"- Judge: `{report['judge']['provider']}` / `{report['judge']['model']}`",
        f"- Prompt version: `{report['judge']['prompt_version']}`",
        f"- Input report: `{report['judge']['input_report']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "total",
        "semantic_correct",
        "semantic_partial",
        "semantic_incorrect",
        "semantic_accuracy",
        "grounded",
        "partially_grounded",
        "ungrounded",
        "grounded_rate",
        "semantically_correct_and_grounded",
        "average_semantic_score",
        "average_grounded_score",
        "judge_errors",
        "judge_unresolved",
        "elapsed_seconds",
    ]:
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(["", "## Diagnosis", "", "| Label | Count |", "| --- | ---: |"])
    for label, count in sorted(summary.get("diagnosis_counts", {}).items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Judgments",
            "",
            "| # | Semantic | Grounded | Diagnosis | Question | Reason |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["judgments"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["ordinal"]),
                    str(item["semantic_label"]),
                    str(item["grounded_label"]),
                    str(item["diagnosis_label"]),
                    _md_cell(str(item["question"])),
                    _md_cell(str(item["reason"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def create_real_llm_client(config_path: Path):
    config = load_config(config_path)
    model_config = config.get_language_model_config(DEFAULT_CHAT_MODEL_ID)
    return create_chat_provider(
        provider=config.extraction.llm_provider,
        model_config=model_config,
        require_real=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge QA report answers with a real LLM.")
    parser.add_argument("--input-report", required=True, help="QA report JSON path.")
    parser.add_argument(
        "--config",
        default=None,
        help="Settings YAML for real LLM provider. Not required for offline reparse/merge.",
    )
    parser.add_argument("--output-json", default=None, help="Output judge JSON path.")
    parser.add_argument("--output-md", default=None, help="Output judge Markdown path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit.")
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=1200,
        help="Maximum chars per retrieved evidence chunk in the judge prompt.",
    )
    parser.add_argument(
        "--retry-evidence-chars",
        default=None,
        help="Comma-separated per-chunk evidence char limits for judge retries, e.g. 600,300,80.",
    )
    parser.add_argument(
        "--checkpoint-jsonl",
        default=None,
        help="Per-case checkpoint JSONL path. Defaults to <output-json>.checkpoint.jsonl for real judge runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint and rerun only missing or judge_unresolved cases.",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=None,
        help="1-based inclusive case ordinal to start judging.",
    )
    parser.add_argument(
        "--slice-end",
        type=int,
        default=None,
        help="1-based inclusive case ordinal to stop judging.",
    )
    parser.add_argument(
        "--reparse-only",
        action="store_true",
        help="Offline mode: reparse stored judge_raw_response values and refresh summary.",
    )
    parser.add_argument(
        "--merge-report",
        action="append",
        default=[],
        help="Offline mode: merge one retry/reparse judge JSON report. Repeatable.",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Judge only question/gold/system-answer semantic equivalence; do not load evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_report = Path(args.input_report)
    output_json = Path(args.output_json) if args.output_json else input_report.with_name(
        input_report.stem + "-llm-judge.json"
    )
    output_md = Path(args.output_md) if args.output_md else input_report.with_name(
        input_report.stem + "-llm-judge.md"
    )
    try:
        if args.reparse_only or args.merge_report:
            report = json.loads(input_report.read_text(encoding="utf-8"))
            if args.reparse_only:
                report = reparse_judge_report(report)
            if args.merge_report:
                patch_reports = [
                    json.loads(Path(path).read_text(encoding="utf-8"))
                    for path in args.merge_report
                ]
                report = merge_judge_reports(
                    report,
                    patch_reports,
                    merge_sources=[str(Path(path)) for path in args.merge_report],
                )
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(format_markdown_report(report), encoding="utf-8")
        else:
            if not args.config:
                raise ValueError("--config is required unless --reparse-only or --merge-report is used.")
            llm_client = create_real_llm_client(Path(args.config))
            checkpoint_jsonl = (
                Path(args.checkpoint_jsonl)
                if args.checkpoint_jsonl
                else output_json.with_suffix(".checkpoint.jsonl")
            )
            report = run_judge(
                input_report_path=input_report,
                output_json_path=output_json,
                output_markdown_path=output_md,
                llm_client=llm_client,
                limit=args.limit,
                max_evidence_chars=args.max_evidence_chars,
                retry_evidence_chars=_parse_int_csv(args.retry_evidence_chars),
                checkpoint_jsonl_path=checkpoint_jsonl,
                resume=args.resume,
                slice_start=args.slice_start,
                slice_end=args.slice_end,
                semantic_only=args.semantic_only,
            )
    except Exception as exc:
        print(f"LLM QA judge failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["judge_errors"] == 0 else 1


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response did not contain a JSON object.")
    decoded = json.loads(stripped[start : end + 1])
    if not isinstance(decoded, dict):
        raise ValueError("Judge response JSON must be an object.")
    return decoded


def _looks_like_json_response(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.startswith("{") or stripped.startswith("```")


def _normalize_label(value: Any, *, allowed: set[str], default: str) -> str:
    label = str(value or "").strip().lower().replace("-", "_")
    if label in allowed:
        return label
    return default


def _first_payload_value(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    fallback: str,
) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return fallback


def _normalize_semantic_label(value: Any) -> str:
    label = _normalize_label(value, allowed=SEMANTIC_LABELS, default="")
    if label:
        return label
    text = _label_text(value)
    if not text:
        return "incorrect"
    if text in {"1", "yes", "true"}:
        return "correct"
    if any(
        phrase in text
        for phrase in (
            "incorrect",
            "not correct",
            "not semantically correct",
            "semantically incorrect",
            "wrong",
            "does not match",
            "doesn't match",
        )
    ):
        return "incorrect"
    if "partial" in text or "partially" in text:
        return "partial"
    if any(phrase in text for phrase in ("correct", "equivalent", "matches", "same answer")):
        return "correct"
    return "incorrect"


def _normalize_grounded_label(value: Any) -> str:
    label = _normalize_label(value, allowed=GROUNDED_LABELS, default="")
    if label:
        return label
    text = _label_text(value)
    if not text:
        return "ungrounded"
    if text in {"1", "yes", "true", "correct"}:
        return "grounded"
    if any(
        phrase in text
        for phrase in (
            "ungrounded",
            "not grounded",
            "unsupported",
            "not supported",
            "does not support",
            "doesn't support",
        )
    ):
        return "ungrounded"
    if "partial" in text or "partially" in text:
        return "partially_grounded"
    if any(
        phrase in text
        for phrase in (
            "grounded",
            "supported",
            "supporting evidence",
            "directly support",
            "directly supported",
        )
    ):
        return "grounded"
    return "ungrounded"


def _reconcile_semantic_label(label: str, *, raw_score: Any, reason: str) -> str:
    if label != "incorrect":
        return label
    if _raw_score(raw_score) < 0.9:
        return label
    text = _label_text(reason)
    if _has_negative_semantic_cue(text):
        return label
    if any(
        phrase in text
        for phrase in (
            "semantically correct",
            "matches the gold",
            "matches gold",
            "matches the answer",
            "matches exactly",
            "exactly matches",
            "semantically equivalent",
        )
    ):
        return "correct"
    return label


def _reconcile_grounded_label(label: str, *, raw_score: Any, reason: str) -> str:
    if label != "ungrounded":
        return label
    if _raw_score(raw_score) < 0.9:
        return label
    text = _label_text(reason)
    if _has_negative_grounded_cue(text):
        return label
    if any(
        phrase in text
        for phrase in (
            "fully grounded",
            "grounded in the evidence",
            "fully supported",
            "directly supported",
            "evidence supports",
            "supported by the retrieved evidence",
            "supported by the evidence",
        )
    ):
        return "grounded"
    return label


def _has_negative_semantic_cue(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "semantically incorrect",
            "not semantically correct",
            "incorrect",
            "does not match",
            "doesn't match",
            "wrong",
        )
    )


def _has_negative_grounded_cue(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "ungrounded",
            "not grounded",
            "not supported",
            "unsupported",
            "does not support",
            "doesn't support",
        )
    )


def _label_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _raw_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return -1.0
    if score > 1.0 and score <= 10.0:
        score = score / 10.0
    return score


def _score(value: Any, label: str) -> float:
    try:
        score = _raw_score(value)
    except (TypeError, ValueError):
        if label == "correct" or label == "grounded":
            return 1.0
        if label == "partial" or label == "partially_grounded":
            return 0.5
        return 0.0
    if score < 0.0:
        if label == "correct" or label == "grounded":
            return 1.0
        if label == "partial" or label == "partially_grounded":
            return 0.5
        return 0.0
    return round(min(max(score, 0.0), 1.0), 4)


def _client_stats(llm_client: Any) -> dict[str, Any]:
    get_stats = getattr(llm_client, "get_stats", None)
    if callable(get_stats):
        stats = get_stats()
        if isinstance(stats, dict):
            return stats
    return {}


def _parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parsed = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed.append(int(stripped))
    return parsed


def _is_real_extraction_report(report: dict[str, Any]) -> bool:
    benchmark = report.get("benchmark", {})
    if not isinstance(benchmark, dict):
        return False
    explicit = benchmark.get("real_extraction")
    if explicit is not None:
        return bool(explicit)
    index_mode = str(benchmark.get("index_mode") or "").lower()
    if "gold" in index_mode:
        return False
    if "extract" in index_mode:
        return True
    name = str(benchmark.get("name") or "").lower()
    return "extract" in name and "real" in name


def _gold_supported_by_text(gold: str, text_evidence: list[dict[str, str]]) -> bool:
    normalized_gold = _normalize_text(gold)
    if not normalized_gold:
        return True
    joined = _normalize_text(" ".join(item.get("text", "") for item in text_evidence))
    return normalized_gold in joined


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:160]


if __name__ == "__main__":
    sys.exit(main())
