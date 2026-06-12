"""Answer generation for graph-grounded QA."""

from __future__ import annotations

import json
import re
from typing import Any

from graphrag_v2.qa.models import (
    CandidateAnswer,
    CommunityEvidence,
    GraphEvidence,
    TextEvidence,
)
from graphrag_v2.qa.prompts import build_prompt

INSUFFICIENT_EVIDENCE_ANSWER = "证据不足，无法回答该问题。"


class MockAnswerer:
    """Deterministic answerer for tests and offline demos."""

    def answer(
        self,
        question: str,
        route: str,
        graph_evidence: GraphEvidence,
        community_evidence: list[CommunityEvidence],
        text_evidence: list[TextEvidence],
    ) -> str:
        if (
            not graph_evidence.linked_entities
            and not graph_evidence.relationships
            and not community_evidence
            and not text_evidence
        ):
            return INSUFFICIENT_EVIDENCE_ANSWER

        if route == "global":
            return self._answer_global(question, graph_evidence, community_evidence, text_evidence)
        return self._answer_local(question, graph_evidence, community_evidence, text_evidence)

    def _answer_local(
        self,
        question: str,
        graph_evidence: GraphEvidence,
        community_evidence: list[CommunityEvidence],
        text_evidence: list[TextEvidence],
    ) -> str:
        lines = [f"基于图谱证据，回答“{question}”:"]
        if graph_evidence.linked_entities:
            lines.append(
                "识别到实体: "
                + ", ".join(entity.name for entity in graph_evidence.linked_entities[:3])
            )
        if graph_evidence.relationships:
            top = graph_evidence.relationships[0]
            lines.append(
                f"最相关关系: {top.source_name} {top.relation} {top.target_name}。"
            )
        if text_evidence:
            lines.append(f"证据文本: {text_evidence[0].text[:160]}")
        if community_evidence:
            lines.append(
                f"补充社区报告: {community_evidence[0].title} - {community_evidence[0].summary}"
            )
        return "\n".join(lines)

    def _answer_global(
        self,
        question: str,
        graph_evidence: GraphEvidence,
        community_evidence: list[CommunityEvidence],
        text_evidence: list[TextEvidence],
    ) -> str:
        lines = [f"基于社区报告和图谱证据，回答“{question}”:"]
        if community_evidence:
            top = community_evidence[0]
            lines.append(f"主要社区: {top.title}。")
            if top.summary:
                lines.append(f"社区摘要: {top.summary}")
            if top.findings:
                lines.append("关键发现: " + "；".join(top.findings[:3]))
        if graph_evidence.relationships:
            top_relationship = graph_evidence.relationships[0]
            lines.append(
                f"支持关系: {top_relationship.source_name} {top_relationship.relation} {top_relationship.target_name}。"
            )
        if text_evidence:
            snippets = [
                item.text[:160]
                for item in text_evidence[:3]
                if item.text
            ]
            if snippets:
                lines.append("证据文本: " + " ".join(snippets))
        return "\n".join(lines)


class LLMAnswerer:
    """LLM-backed answerer for explicit production QA requests."""

    def __init__(
        self,
        llm_client: Any | None = None,
        candidate_extractor: "CandidateAnswerExtractor | None" = None,
    ):
        self.llm_client = llm_client
        self.candidate_extractor = candidate_extractor or CandidateAnswerExtractor()

    def answer(
        self,
        question: str,
        route: str,
        graph_evidence: GraphEvidence,
        community_evidence: list[CommunityEvidence],
        text_evidence: list[TextEvidence],
    ) -> str:
        if self.llm_client is None or getattr(self.llm_client, "mock_mode", False):
            raise ValueError("LLM answerer requires a configured real LLM client.")

        candidates = self.candidate_extractor.extract(
            question=question,
            graph_evidence=graph_evidence,
            community_evidence=community_evidence,
            text_evidence=text_evidence,
        )
        if not candidates:
            return INSUFFICIENT_EVIDENCE_ANSWER

        prompt = build_prompt(
            question=question,
            route=route,
            graph_evidence=graph_evidence,
            community_evidence=community_evidence,
            text_evidence=text_evidence,
        )
        prompt = prompt + "\n\nCandidate Answers:\n" + _format_candidates(candidates)
        errors_before = int(getattr(self.llm_client, "total_errors", 0) or 0)
        request_kwargs: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a graph-grounded QA assistant. Use only the "
                        "provided graph evidence, community evidence, and text "
                        "evidence. You must choose from the provided Candidate "
                        "Answers only. If none of the candidates answers the "
                        "question, mark supported=false. Return only JSON with "
                        "keys: candidate_id, answer_text, supported, reason. "
                        "You must select exactly one candidate_id from Candidate "
                        "Answers when the evidence supports an answer. Do not "
                        "invent a new answer_text. Prefer the candidate whose "
                        "source_chunk_id directly contains the final answer span "
                        "required by the question. For chain questions, follow "
                        "the relationship chain in the question before selecting "
                        "the candidate. "
                        "Do not include hidden reasoning or <think> blocks."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        if bool(getattr(self.llm_client, "supports_response_format_json", False)):
            request_kwargs["response_format_json"] = True
        response = self.llm_client.chat_completion(**request_kwargs)
        errors_after = int(getattr(self.llm_client, "total_errors", 0) or 0)
        if errors_after > errors_before:
            raise ValueError("LLM answer generation failed after retry attempts.")
        if not isinstance(response, str):
            raise ValueError("LLM answer generation returned a non-string response.")
        cleaned_response = _clean_llm_answer(response)
        if not cleaned_response:
            raise ValueError("LLM answer generation returned an empty response.")
        if _response_marks_unsupported(cleaned_response):
            retry_answer = self._retry_candidate_selection(
                question=question,
                candidates=candidates,
                text_evidence=text_evidence,
            )
            if retry_answer and retry_answer != INSUFFICIENT_EVIDENCE_ANSWER:
                return retry_answer
        return _select_candidate_answer(cleaned_response, candidates)

    def _retry_candidate_selection(
        self,
        *,
        question: str,
        candidates: list[CandidateAnswer],
        text_evidence: list[TextEvidence],
    ) -> str | None:
        shortlist = _shortlist_reselection_candidates(candidates)
        if not shortlist:
            return None
        errors_before = int(getattr(self.llm_client, "total_errors", 0) or 0)
        request_kwargs: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are selecting a final graph-grounded QA answer. "
                        "Use only the provided candidate answers and snippets. "
                        "If a candidate directly answers the question, select it. "
                        "Return only JSON with keys: candidate_id, answer_text, "
                        "supported, reason."
                    ),
                },
                {
                    "role": "user",
                    "content": _format_reselection_prompt(
                        question=question,
                        candidates=shortlist,
                        text_evidence=text_evidence,
                    ),
                },
            ],
            "temperature": 0.0,
        }
        if bool(getattr(self.llm_client, "supports_response_format_json", False)):
            request_kwargs["response_format_json"] = True
        response = self.llm_client.chat_completion(**request_kwargs)
        errors_after = int(getattr(self.llm_client, "total_errors", 0) or 0)
        if errors_after > errors_before:
            raise ValueError("LLM answer generation failed after retry attempts.")
        if not isinstance(response, str):
            raise ValueError("LLM answer generation returned a non-string response.")
        cleaned_response = _clean_llm_answer(response)
        if not cleaned_response:
            return None
        return _select_candidate_answer(cleaned_response, candidates)


class CandidateAnswerExtractor:
    """Extract answer candidates from retrieved source text."""

    def extract(
        self,
        *,
        question: str,
        graph_evidence: GraphEvidence,
        community_evidence: list[CommunityEvidence],
        text_evidence: list[TextEvidence],
    ) -> list[CandidateAnswer]:
        del graph_evidence, community_evidence
        raw_candidates: list[tuple[str, str, str, list[int], float]] = (
            _extract_question_option_candidates(question, text_evidence)
        )
        for item in text_evidence:
            raw_candidates.extend(_extract_chain_answer_candidates(question, item))
        for item in text_evidence:
            raw_candidates.extend(_extract_text_candidates(question, item))

        candidates: list[CandidateAnswer] = []
        seen: set[tuple[str, str]] = set()
        for text, answer_type, chunk_id, source_span, confidence in raw_candidates:
            key = (text.casefold(), chunk_id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CandidateAnswer(
                    id=f"cand_{len(candidates) + 1}",
                    answer_text=text,
                    answer_type=answer_type,
                    source_chunk_id=chunk_id,
                    source_span=source_span,
                    confidence=round(confidence, 4),
                )
            )
        candidates.sort(key=lambda item: (-item.confidence, item.source_chunk_id, item.source_span))
        return [
            CandidateAnswer(
                id=f"cand_{index}",
                answer_text=item.answer_text,
                answer_type=item.answer_type,
                source_chunk_id=item.source_chunk_id,
                source_span=item.source_span,
                confidence=item.confidence,
                source=item.source,
            )
            for index, item in enumerate(candidates[:48], start=1)
        ]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_UPPER_LETTER_CHARS = "A-ZÀ-ÖØ-ÞĀĂĄĆĈĊČĎĐĒĔĖĘĚĜĞĠĢĤĦĨĪĬĮİĲĴĶĹĻĽĿŁŃŅŇŊŌŎŐŒŔŖŘŚŜŞŠŢŤŦŨŪŬŮŰŲŴŶŸŹŻŽ"
_NAME_CHARS = r"A-Za-zÀ-ɏ.'’ʻ\-"
_SPAN_CHARS = _NAME_CHARS + r",\s"
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
    r"|\b(?:18|19|20)\d{2}\b",
    flags=re.IGNORECASE,
)
_EARLY_YEAR_RE = re.compile(r"\b(?:c\.\s*)?([1-9]\d{2,3})(?=\s*(?:[–-]|-|AD|BC|\)|,|\.|$))")
_BORN_PLACE_RE = re.compile(rf"\b(?:born|birthplace|birth place)\s+(?:in|at|near)\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})")
_BORN_COMPLEX_PLACE_RE = re.compile(rf"\bborn\b[^.;\n]{{0,90}}?\s+in\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})", flags=re.IGNORECASE)
_BORN_NAMED_PLACE_RE = re.compile(
    rf"\bborn\s+[{_UPPER_LETTER_CHARS}][^.;\n]{{0,90}}?\s+in\s+([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})(?=\s+(?:on|in|,|\.|;)|[.;\n])"
)
_DIED_PLACE_RE = re.compile(rf"\b(?:died|death place|place of death)\b[^.;\n]{{0,90}}?\s+(?:in|at|near)\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})", flags=re.IGNORECASE)
_BURIAL_PLACE_RE = re.compile(rf"\b(?:buried|interred|place of burial|burial place)\b[^.;\n]{{0,60}}?\s+(?:in|at|near)\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})", flags=re.IGNORECASE)
_EVENT_PLACE_RE = re.compile(rf"\b(?:hosted|located|set|based)\s+(?:in|at|near)\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})")
_EDUCATION_ORG_RE = re.compile(rf"\b(?:studied|educated|graduated)\s+(?:at|from|in)\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})", flags=re.IGNORECASE)
_WORK_ORG_RE = re.compile(rf"\b(?:worked|served)\s+(?:at|for|with)\s+(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}})", flags=re.IGNORECASE)
_POLITICAL_ENTITY_RE = re.compile(
    rf"\b(?:unified\s+)?((?:Kingdom|Republic|Empire|Colony|State|Province|Duchy|Principality|County|Sultanate|Caliphate)\s+of\s+[{_SPAN_CHARS}]{{2,80}}|[{_UPPER_LETTER_CHARS}][{_NAME_CHARS}]+(?:\s+[{_UPPER_LETTER_CHARS}][{_NAME_CHARS}]+){{0,3}}\s+(?:Empire|Kingdom|Republic|Federation|Confederation|Sultanate|Caliphate))",
)
_RELATION_TARGET_RE = re.compile(
    r"\b(?:directed by|founded by|created by|written by|composed by|produced by|starring|married to|married|spouse is|father is|mother is)\s+([A-ZÀ-ÖØ-Þ][^.,;()\n]{1,80})"
)
_INSTRUMENT_OBJECT_RE = re.compile(
    r"\b(?:used|played|performed on|featured)\s+(?:a|an|the)?\s*([a-z][a-z-]+)(?=\s+(?:in|section|at|with)\b)"
)
_QUOTE_RE = re.compile(r'"([^"\n]{2,80})"|“([^”\n]{2,80})”')
_ESTATE_RE = re.compile(rf"\b(?:former\s+)?([{_UPPER_LETTER_CHARS}][{_NAME_CHARS}]+)\s+(?:Estate|gharana)\b")
_CAPITALIZED_SPAN_RE = re.compile(
    rf"\b[{_UPPER_LETTER_CHARS}][{_NAME_CHARS}]*(?<!\.)(?:\s+(?:of|the|and|de|la|van|von|[{_UPPER_LETTER_CHARS}][{_NAME_CHARS}]*(?<!\.))){{0,5}}"
)
_WHICH_OPTION_RE = re.compile(r",\s*(?P<left>[^?,]+?)\s+or\s+(?P<right>[^?]+)\?")


def _clean_llm_answer(response: str) -> str:
    cleaned = response.strip()
    cleaned = _THINK_BLOCK_RE.sub("", cleaned).strip()
    closing_index = cleaned.lower().rfind("</think>")
    if closing_index != -1:
        cleaned = cleaned[closing_index + len("</think>"):].strip()
    return cleaned


def _extract_chain_answer_candidates(
    question: str,
    evidence: TextEvidence,
) -> list[tuple[str, str, str, list[int], float]]:
    text = evidence.text
    normalized_question = question.lower()
    patterns: list[tuple[re.Pattern[str], str, float]] = []
    if "born" in normalized_question or "birth" in normalized_question:
        patterns.append(
            (
                re.compile(
                    rf"\bwas born in (?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}}?)(?=\s+(?:and|but|in\s+\d{{4}})|,|\.)"
                ),
                "location",
                0.99,
            )
        )
    if "die" in normalized_question or "died" in normalized_question:
        patterns.append(
            (
                re.compile(
                    rf"\bdied in (?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}}?)(?=\s+in\s+\d{{4}}|,|\.)"
                ),
                "location",
                0.99,
            )
        )
    if "sibling-in-law" in normalized_question or "sibling in law" in normalized_question:
        patterns.append(
            (
                re.compile(
                    rf"\bmarried\s+(?:(?:his|her|their)\s+)?(?:(?:first|second|third)\s+)?(?:wife|husband|spouse)?\s*(?:the\s+)?([{_UPPER_LETTER_CHARS}][{_SPAN_CHARS}]{{1,80}}?)(?=\.|,|\s+(?:in|on|whose)\b)"
                ),
                "entity",
                0.99,
            )
        )

    candidates: list[tuple[str, str, str, list[int], float]] = []
    for pattern, answer_type, confidence in patterns:
        for match in pattern.finditer(text):
            answer = _trim_candidate(match.group(1))
            if _valid_candidate(answer):
                candidates.append(
                    (
                        answer,
                        answer_type,
                        evidence.chunk_id,
                        [match.start(1), match.start(1) + len(answer)],
                        confidence,
                    )
                )
    return candidates


def _extract_text_candidates(
    question: str,
    evidence: TextEvidence,
) -> list[tuple[str, str, str, list[int], float]]:
    text = evidence.text
    candidates: list[tuple[str, str, str, list[int], float]] = []
    for regex, answer_type in (
        (_BORN_PLACE_RE, "location"),
        (_BORN_COMPLEX_PLACE_RE, "location"),
        (_BORN_NAMED_PLACE_RE, "location"),
        (_DIED_PLACE_RE, "location"),
        (_BURIAL_PLACE_RE, "location"),
        (_EVENT_PLACE_RE, "location"),
        (_EDUCATION_ORG_RE, "organization"),
        (_WORK_ORG_RE, "organization"),
        (_POLITICAL_ENTITY_RE, "nationality"),
        (_RELATION_TARGET_RE, "entity"),
        (_INSTRUMENT_OBJECT_RE, "object"),
        (_ESTATE_RE, "location"),
    ):
        for match in regex.finditer(text):
            span = match.span(1)
            answer = _trim_candidate(match.group(1))
            confidence = _pattern_candidate_confidence(question, answer_type)
            for variant, variant_span, variant_confidence in _candidate_variants(
                answer,
                span,
                confidence,
            ):
                if _valid_candidate(variant):
                    candidates.append(
                        (
                            variant,
                            answer_type,
                            evidence.chunk_id,
                            variant_span,
                            variant_confidence,
                        )
                    )

    for match in _DATE_RE.finditer(text):
        answer = _trim_candidate(match.group(0))
        confidence = 0.93 if _question_prefers_date(question) else 0.45
        candidates.append((answer, "date", evidence.chunk_id, [match.start(), match.end()], confidence))

    for match in _EARLY_YEAR_RE.finditer(text):
        answer = _trim_candidate(match.group(1))
        confidence = 0.93 if _question_prefers_date(question) else 0.42
        candidates.append((answer, "date", evidence.chunk_id, [match.start(1), match.end(1)], confidence))

    for match in _QUOTE_RE.finditer(text):
        answer = _trim_candidate(match.group(1) or match.group(2) or "")
        if _valid_candidate(answer):
            candidates.append((answer, "entity", evidence.chunk_id, [match.start(), match.end()], 0.91))

    if _asks_yes_no(question):
        for value in ("yes", "no"):
            candidates.append((value, "boolean", evidence.chunk_id, [0, 0], 0.55))

    for match in _CAPITALIZED_SPAN_RE.finditer(text):
        answer = _trim_candidate(match.group(0))
        if not _valid_candidate(answer):
            continue
        if answer.lower() in {"the", "a", "an", "he", "she", "it", "no"}:
            continue
        confidence = _capitalized_span_confidence(question, answer)
        candidates.append((answer, "entity", evidence.chunk_id, [match.start(), match.end()], confidence))
    return candidates


def _extract_question_option_candidates(
    question: str,
    text_evidence: list[TextEvidence],
) -> list[tuple[str, str, str, list[int], float]]:
    match = _WHICH_OPTION_RE.search(question)
    if not match:
        return []
    candidates = []
    for option in (
        _trim_candidate(match.group("left"), split_sentence=False),
        _trim_candidate(match.group("right"), split_sentence=False),
    ):
        if not _valid_candidate(option):
            continue
        source_chunk_id = ""
        source_span = [0, 0]
        for evidence in text_evidence:
            index = evidence.text.casefold().find(option.casefold())
            if index != -1:
                source_chunk_id = evidence.chunk_id
                source_span = [index, index + len(option)]
                break
        if source_chunk_id:
            candidates.append((option, "entity", source_chunk_id, source_span, 0.98))
    return candidates


def _format_candidates(candidates: list[CandidateAnswer]) -> str:
    payload = [
        {
            "candidate_id": candidate.id,
            "answer_text": candidate.answer_text,
            "answer_type": candidate.answer_type,
            "source_chunk_id": candidate.source_chunk_id,
            "source_span": candidate.source_span,
            "confidence": candidate.confidence,
        }
        for candidate in candidates
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_reselection_prompt(
    *,
    question: str,
    candidates: list[CandidateAnswer],
    text_evidence: list[TextEvidence],
) -> str:
    snippets = []
    by_chunk_id = {item.chunk_id: item for item in text_evidence}
    for candidate in candidates:
        evidence = by_chunk_id.get(candidate.source_chunk_id)
        snippet = ""
        if evidence:
            start, end = candidate.source_span
            window_start = max(0, start - 140)
            window_end = min(len(evidence.text), max(end, start) + 140)
            snippet = evidence.text[window_start:window_end]
        snippets.append(
            {
                "candidate_id": candidate.id,
                "answer_text": candidate.answer_text,
                "answer_type": candidate.answer_type,
                "source_chunk_id": candidate.source_chunk_id,
                "confidence": candidate.confidence,
                "snippet": snippet,
            }
        )
    return json.dumps(
        {
            "question": question,
            "instruction": (
                "Select the candidate that directly answers the question. "
                "Use supported=false only when none of these candidates answer it."
            ),
            "candidates": snippets,
        },
        ensure_ascii=False,
        indent=2,
    )


def _shortlist_reselection_candidates(candidates: list[CandidateAnswer]) -> list[CandidateAnswer]:
    strong = [candidate for candidate in candidates if candidate.confidence >= 0.9]
    return (strong or candidates)[:12]


def _response_marks_unsupported(response: str) -> bool:
    try:
        payload = _extract_json_object(response)
    except (ValueError, json.JSONDecodeError):
        return False
    return payload.get("supported") is False


def _select_candidate_answer(response: str, candidates: list[CandidateAnswer]) -> str:
    by_id = {candidate.id: candidate for candidate in candidates}
    by_text = {candidate.answer_text.casefold(): candidate for candidate in candidates}
    try:
        payload = _extract_json_object(response)
    except ValueError:
        matched = [
            candidate
            for candidate in candidates
            if re.search(rf"\b{re.escape(candidate.answer_text)}\b", response, flags=re.IGNORECASE)
        ]
        if len(matched) == 1:
            return matched[0].answer_text
        raise ValueError("LLM answer generation returned text outside extracted candidate answers.")

    supported = bool(payload.get("supported", True))
    if not supported:
        return INSUFFICIENT_EVIDENCE_ANSWER
    candidate_id = str(payload.get("candidate_id") or payload.get("id") or "").strip()
    if candidate_id in by_id:
        return by_id[candidate_id].answer_text
    answer_text = str(payload.get("answer_text") or "").strip()
    if answer_text.casefold() in by_text:
        return by_text[answer_text.casefold()].answer_text
    raise ValueError("LLM answer generation returned an answer outside extracted candidate answers.")


def _extract_json_object(response: str) -> dict[str, Any]:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM answer response did not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM answer response JSON must be an object.")
    return payload


def _trim_candidate(value: str, *, split_sentence: bool = True) -> str:
    trimmed = value.strip()
    if split_sentence:
        trimmed = re.split(r"(?:\.\s+|\n+)", trimmed, maxsplit=1)[0]
        trimmed = re.split(r"\s+(?:who|which|that)\b", trimmed, maxsplit=1, flags=re.IGNORECASE)[0]
    trimmed = trimmed.strip(" \t\r\n,.;:()[]{}\"'")
    trimmed = re.sub(r"\s+(?:on|in|at|near|from|with|by|to)$", "", trimmed, flags=re.IGNORECASE)
    return trimmed.strip(" \t\r\n,.;:()[]{}\"'")


def _valid_candidate(value: str) -> bool:
    if not value:
        return False
    if len(value) > 80:
        return False
    if not any(char.isalnum() for char in value):
        return False
    if len(value) == 1 and not value.isdigit():
        return False
    return True


def _asks_yes_no(question: str) -> bool:
    return question.strip().lower().split(" ", 1)[0] in {"is", "are", "was", "were", "do", "does", "did", "has", "have"}


def _question_prefers_date(question: str) -> bool:
    normalized = question.lower()
    return any(
        cue in normalized
        for cue in (
            "date of birth",
            "date of death",
            "birth date",
            "death date",
            "what year",
            "which year",
        )
    ) or normalized.startswith("when")


def _question_prefers_named_entity(question: str) -> bool:
    normalized = question.lower()
    return any(word in normalized for word in ("who", "where", "which", "what", "whose"))


def _pattern_candidate_confidence(question: str, answer_type: str) -> float:
    if answer_type == "object":
        return 0.93
    if answer_type == "date":
        return 0.93 if _question_prefers_date(question) else 0.45
    if answer_type == "nationality":
        return 0.97 if "nationality" in question.lower() else 0.9
    if answer_type == "organization":
        return 0.97 if any(cue in question.lower() for cue in ("graduate", "studied", "educated")) else 0.92
    if answer_type == "location":
        return 0.97 if any(cue in question.lower() for cue in ("where", "place")) else 0.94
    return 0.96


def _capitalized_span_confidence(question: str, answer: str) -> float:
    normalized_answer = answer.lower()
    if normalized_answer in {
        "american",
        "british",
        "indian",
        "french",
        "german",
        "japanese",
        "spanish",
        "russian",
        "italian",
        "danish",
    }:
        return 0.86 if "nationality" in question.lower() else 0.52
    if len(answer) <= 2:
        return 0.35
    return 0.72 if _question_prefers_named_entity(question) else 0.58


def _candidate_variants(
    answer: str,
    span: tuple[int, int],
    confidence: float,
) -> list[tuple[str, list[int], float]]:
    trimmed = _trim_candidate(answer)
    if not trimmed:
        return []
    variants = [(trimmed, [span[0], span[0] + len(trimmed)], round(confidence, 4))]
    if "," in trimmed:
        head = _trim_candidate(trimmed.split(",", 1)[0])
        if head and head != trimmed:
            variants.append((head, [span[0], span[0] + len(head)], round(confidence - 0.01, 4)))
    return variants
