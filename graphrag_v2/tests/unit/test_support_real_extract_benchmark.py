"""Tests for support-only real-extraction benchmark adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_benchmark_module():
    module_path = REPO_ROOT / "scripts" / "benchmark_support_real_extract.py"
    spec = importlib.util.spec_from_file_location("benchmark_support_real_extract", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_2wiki_support_only_dataset_writes_docs_questions_and_gold_index(
    tmp_path: Path,
):
    benchmark = _load_benchmark_module()
    input_path = tmp_path / "2wiki.json"
    output_dir = tmp_path / "2wiki-real-extract"
    input_path.write_text(json.dumps([_two_wiki_record()]), encoding="utf-8")

    prepared = benchmark.prepare_support_only_dataset(
        input_path=input_path,
        output_dir=output_dir,
        benchmark_name="2Wiki",
        sample_size=1,
        seed=7,
    )

    assert prepared.sample_size == 1
    assert prepared.docs_path == output_dir / "adapter" / "docs"
    assert prepared.questions_path == output_dir / "adapter" / "questions.jsonl"
    assert (prepared.docs_path / "case_1.md").exists()
    doc_text = (prepared.docs_path / "case_1.md").read_text(encoding="utf-8")
    assert "## Les films du losange" in doc_text
    assert "## Éric Rohmer" in doc_text
    assert "Distractor Page" not in doc_text

    questions = [
        json.loads(line)
        for line in prepared.questions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert questions == [
        {
            "id": "2wiki_case_1",
            "type": "2wiki_compositional",
            "question": "Where does the founder of Les Films Du Losange work at?",
            "expected_route": "local",
            "expected_refused": False,
            "answer": "Cahiers du cinéma",
            "required_entities": [
                {"name": "Les films du losange"},
                {"name": "Éric Rohmer"},
            ],
            "required_relationships": [
                {
                    "source": {"name": "Les films du losange"},
                    "relation": {"name": "supports_answer"},
                    "target": {"name": "Éric Rohmer"},
                }
            ],
            "required_citations": [
                benchmark.stable_id("2wiki_chunk", "case_1", "Les films du losange"),
                benchmark.stable_id("2wiki_chunk", "case_1", "Éric Rohmer"),
            ],
            "answer_terms": ["Cahiers du cinéma"],
            "metadata": {
                "benchmark": "2Wiki",
                "record_id": "case_1",
                "level": "unknown",
                "supporting_titles": ["Les films du losange", "Éric Rohmer"],
            },
        }
    ]

    metadata = json.loads((prepared.gold_index_path / "index_metadata.json").read_text())
    assert metadata["benchmark"] == "2Wiki Support-Only"
    assert metadata["graph_store_provider"] == "json"
    assert metadata["sample_seed"] == 7


def test_prepare_hotpotqa_columnar_support_only_dataset_writes_support_pages(
    tmp_path: Path,
):
    benchmark = _load_benchmark_module()
    input_path = tmp_path / "hotpotqa.json"
    output_dir = tmp_path / "hotpotqa-real-extract"
    input_path.write_text(json.dumps([_hotpotqa_columnar_record()]), encoding="utf-8")

    prepared = benchmark.prepare_support_only_dataset(
        input_path=input_path,
        output_dir=output_dir,
        benchmark_name="HotpotQA",
        sample_size=1,
        seed=7,
    )

    doc_text = (prepared.docs_path / "hotpot_case.md").read_text(encoding="utf-8")
    assert "## Ted Kooshian" in doc_text
    assert "## Marvin Hamlisch" in doc_text
    assert "## Distractor Page" not in doc_text

    questions = [
        json.loads(line)
        for line in prepared.questions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert questions[0]["id"] == "hotpotqa_hotpot_case"
    assert questions[0]["required_entities"] == [
        {"name": "Ted Kooshian"},
        {"name": "Marvin Hamlisch"},
    ]
    assert questions[0]["required_citations"] == [
        benchmark.stable_id("hotpotqa_chunk", "hotpot_case", "Ted Kooshian"),
        benchmark.stable_id("hotpotqa_chunk", "hotpot_case", "Marvin Hamlisch"),
    ]


def test_support_only_sample_manifest_uses_seeded_prefixes(tmp_path: Path):
    benchmark = _load_benchmark_module()
    input_path = tmp_path / "2wiki.json"
    input_path.write_text(
        json.dumps(
            [
                _two_wiki_record("case_1"),
                _two_wiki_record("case_2"),
                _two_wiki_record("case_3"),
            ]
        ),
        encoding="utf-8",
    )

    records = benchmark.load_records(input_path)
    manifest = benchmark.build_sample_manifest(
        records,
        benchmark_name="2Wiki",
        sample_sizes=[1, 2],
        seed=7,
        source_file=input_path,
    )

    assert manifest["benchmark"] == "2Wiki"
    assert manifest["selection"] == "seeded_shuffle_prefix"
    assert manifest["samples"]["dev1"]["ids"] == manifest["ordered_ids"][:1]
    assert manifest["samples"]["dev2"]["ids"] == manifest["ordered_ids"][:2]


def test_support_real_extract_runner_tolerates_failed_chunks_by_default(
    tmp_path: Path,
    monkeypatch,
):
    benchmark = _load_benchmark_module()
    input_path = tmp_path / "2wiki.json"
    output_dir = tmp_path / "2wiki-real-extract"
    config_path = tmp_path / "settings.yaml"
    input_path.write_text(json.dumps([_two_wiki_record()]), encoding="utf-8")
    config_path.write_text("models: {}\n", encoding="utf-8")
    seen = {}

    from graphrag_v2.config import GraphRagConfig

    async def fake_index_fusion_only(**kwargs):
        seen["fail_on_invalid_chunk"] = kwargs["config"].extraction.fail_on_invalid_chunk
        index_path = Path(kwargs["output_path"])
        prepared = benchmark.prepare_support_only_dataset(
            input_path=input_path,
            output_dir=tmp_path / "prepared-copy",
            benchmark_name="2Wiki",
            sample_size=1,
            seed=7,
        )
        for source in prepared.gold_index_path.iterdir():
            target = index_path / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        return {"run_status": "succeeded", "extraction_failed_chunks": 1}

    monkeypatch.setattr(benchmark, "load_config", lambda _path: GraphRagConfig())
    monkeypatch.setattr(benchmark, "index_fusion_only", fake_index_fusion_only)

    report = benchmark.run_support_real_extract_benchmark(
        input_path=input_path,
        output_dir=output_dir,
        benchmark_name="2Wiki",
        sample_size=1,
        seed=7,
        config_path=config_path,
        answerer_name="mock",
        run_real_extraction=True,
    )

    assert seen["fail_on_invalid_chunk"] is False
    assert report["real_index_metadata"]["extraction_failed_chunks"] == 1


def test_remap_required_citations_to_real_index_chunks(tmp_path: Path):
    benchmark = _load_benchmark_module()
    index_path = tmp_path / "index"
    index_path.mkdir()
    import pandas as pd

    pd.DataFrame(
        [
            {
                "chunk_id": "chunk_real_1",
                "source_path": str(tmp_path / "docs" / "case_1.md"),
                "chunk_index": 0,
                "text": "# Case\n\n## Les films du losange\nFounded by Éric Rohmer.\n\n## Éric Rohmer\nEdited Cahiers.",
            },
            {
                "chunk_id": "chunk_other",
                "source_path": str(tmp_path / "docs" / "case_2.md"),
                "chunk_index": 0,
                "text": "Unrelated.",
            },
        ]
    ).to_parquet(index_path / "text_units.parquet")
    questions = [
        {
            "id": "2wiki_case_1",
            "required_citations": ["2wiki_chunk_old_a", "2wiki_chunk_old_b"],
            "metadata": {
                "record_id": "case_1",
                "supporting_titles": ["Les films du losange", "Éric Rohmer"],
            },
        }
    ]

    remapped = benchmark.remap_required_citations_to_real_index_chunks(
        questions,
        index_path,
    )

    assert remapped[0]["required_citations"] == ["chunk_real_1"]
    assert remapped[0]["metadata"]["gold_required_citations"] == [
        "2wiki_chunk_old_a",
        "2wiki_chunk_old_b",
    ]
    assert remapped[0]["metadata"]["required_citation_mapping"] == {
        "Les films du losange": ["chunk_real_1"],
        "Éric Rohmer": ["chunk_real_1"],
    }


def _two_wiki_record(case_id: str = "case_1") -> dict:
    return {
        "_id": case_id,
        "type": "compositional",
        "question": "Where does the founder of Les Films Du Losange work at?",
        "answer": "Cahiers du cinéma",
        "supporting_facts": [["Les films du losange", 0], ["Éric Rohmer", 2]],
        "evidences": [
            ["Les films du losange", "founded by", "Éric Rohmer"],
            ["Éric Rohmer", "employer", "Cahiers du cinéma"],
        ],
        "context": [
            [
                "Les films du losange",
                ["Les films du losange is a film production company founded by Éric Rohmer."],
            ],
            [
                "Éric Rohmer",
                ['Rohmer edited the influential film journal, "Cahiers du cinéma".'],
            ],
            ["Distractor Page", ["Distractor Page is unrelated."]],
        ],
    }


def _hotpotqa_columnar_record() -> dict:
    return {
        "id": "hotpot_case",
        "type": "bridge",
        "level": "hard",
        "question": (
            "What award won by only twelve people has a man who Ted Kooshian "
            "has performed with won?"
        ),
        "answer": "EGOT",
        "supporting_facts": {
            "title": ["Ted Kooshian", "Marvin Hamlisch", "Marvin Hamlisch"],
            "sent_id": [0, 1, 2],
        },
        "context": {
            "title": ["Ted Kooshian", "Marvin Hamlisch", "Distractor Page"],
            "sentences": [
                [
                    "Ted Kooshian is an American jazz pianist.",
                    "He has performed with Marvin Hamlisch.",
                ],
                [
                    "Marvin Hamlisch won Emmy, Grammy, Oscar, and Tony awards.",
                    "Only a small group of people have won all four.",
                ],
                ["This page is unrelated."],
            ],
        },
    }
