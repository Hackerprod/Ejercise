"""Generate and independently verify Fase C pointer-chasing datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import DEFAULT_COUNTS, generate_pointer_examples, write_jsonl  # noqa: E402


def solve(example) -> int:
    sources = [int(value) for value in str(example.metadata["memory_sources"]).split(",")]
    destinations = [int(value) for value in str(example.metadata["memory_destinations"]).split(",")]
    mapping = dict(zip(sources, destinations))
    current = int(example.metadata["start_key"])
    for _ in range(int(example.metadata["hop_count"])):
        current = mapping[current]
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "pointer_chasing")
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, count in DEFAULT_COUNTS.items():
        examples = generate_pointer_examples(split, count, args.seed)
        errors = [index for index, example in enumerate(examples) if solve(example) != example.target]
        if errors:
            raise AssertionError(f"pointer solver mismatch at {split}: {errors[:3]}")
        write_jsonl(args.output_dir / f"{split}.jsonl", examples)
        summaries[split] = {
            "count": count,
            "solver_accuracy": 1.0,
            "hop_counts": dict(sorted(Counter(int(row.metadata["hop_count"]) for row in examples).items())),
            "target_cardinality": 256,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
