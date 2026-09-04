"""Generate deterministic T1-B JSONL datasets and a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import (  # noqa: E402
    DEFAULT_COUNTS,
    SPLIT_NAMES,
    TASK_NAMES,
    TokenVocabulary,
    dataset_summary,
    generate_all_datasets,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = generate_all_datasets(seed=args.seed)
    vocabulary = TokenVocabulary()
    vocabulary.to_json(args.output_dir / "vocabulary.json")

    manifest: dict[str, object] = {
        "format_version": 1,
        "seed": args.seed,
        "counts": DEFAULT_COUNTS,
        "token_vocabulary_size": len(vocabulary),
        "tasks": {},
    }
    task_manifest: dict[str, object] = manifest["tasks"]  # type: ignore[assignment]
    for task in TASK_NAMES:
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        split_manifest: dict[str, object] = {}
        for split in SPLIT_NAMES:
            examples = datasets[task][split]
            write_jsonl(task_dir / f"{split}.jsonl", examples)
            split_manifest[split] = dataset_summary(examples)
        task_manifest[task] = split_manifest

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated datasets in {args.output_dir}")
    for task in TASK_NAMES:
        summaries = task_manifest[task]
        random_baseline = summaries["train"]["random_baseline_accuracy"]  # type: ignore[index]
        print(
            f"{task}: train={summaries['train']['count']} "  # type: ignore[index]
            f"val={summaries['val']['count']} test={summaries['test']['count']} "  # type: ignore[index]
            f"random_baseline={random_baseline:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
