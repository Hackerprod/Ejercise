"""Verify generated T1-B dataset counts, ranges, and baseline distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import (  # noqa: E402
    DEFAULT_COUNTS,
    OUTPUT_CARDINALITIES,
    SPLIT_NAMES,
    TASK_NAMES,
    TokenVocabulary,
    dataset_summary,
    load_jsonl,
)


def verify(output_dir: Path) -> dict[str, object]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    vocabulary = TokenVocabulary()
    if manifest["token_vocabulary_size"] != len(vocabulary):
        raise AssertionError("Vocabulary size differs from manifest")

    report: dict[str, object] = {}
    for task in TASK_NAMES:
        task_report: dict[str, object] = {}
        cardinality = OUTPUT_CARDINALITIES[task]
        for split in SPLIT_NAMES:
            path = output_dir / task / f"{split}.jsonl"
            examples = load_jsonl(path)
            expected_count = DEFAULT_COUNTS[split]
            if len(examples) != expected_count:
                raise AssertionError(f"{path}: expected {expected_count}, got {len(examples)}")
            for example in examples:
                if example.task != task or example.split != split:
                    raise AssertionError(f"{path}: task/split mismatch")
                if len(example.tokens) > 64:
                    raise AssertionError(f"{path}: sequence exceeds max length 64")
                vocabulary.encode(example.tokens)
                vocabulary.id_for(example.query_token)
                if example.output_cardinality != cardinality:
                    raise AssertionError(f"{path}: output cardinality mismatch")
                if not 0 <= example.target < cardinality:
                    raise AssertionError(f"{path}: target outside output range")
                if task == "length_generalization":
                    hop_count = int(example.metadata["hop_count"])
                    expected = range(4, 7) if split == "test" else range(1, 4)
                    if hop_count not in expected:
                        raise AssertionError(f"{path}: invalid generalization hop count {hop_count}")
                if task == "sequential_update" and example.metadata["result_in_range"] != 1:
                    raise AssertionError(f"{path}: sequential result escaped [0, 31]")
            summary = dataset_summary(examples)
            task_report[split] = summary
        report[task] = task_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    report = verify(args.output_dir)
    print("Dataset verification: PASS")
    for task in TASK_NAMES:
        task_report = report[task]
        train_summary = task_report["train"]  # type: ignore[index]
        cardinality = train_summary["output_cardinality"]  # type: ignore[index]
        baseline = train_summary["random_baseline_accuracy"]  # type: ignore[index]
        expected = {split: DEFAULT_COUNTS[split] / cardinality for split in SPLIT_NAMES}
        counts = ", ".join(f"{split}={task_report[split]['count']}" for split in SPLIT_NAMES)  # type: ignore[index]
        print(
            f"{task}: {counts}; cardinality={cardinality}; "
            f"random_baseline={baseline:.6f}; expected_random_hits={expected}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
