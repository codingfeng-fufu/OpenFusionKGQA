#!/usr/bin/env python
"""Export graph fusion decisions for human review."""

from __future__ import annotations

import argparse
import json

from graphrag_v2.graph_fusion import export_review_queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export accepted/rejected graph fusion decisions to JSONL.",
    )
    parser.add_argument("--index", required=True, help="Artifact/index directory.")
    parser.add_argument("--output", required=True, help="Output .jsonl path.")
    args = parser.parse_args()

    summary = export_review_queue(index_path=args.index, output_path=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
