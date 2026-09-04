"""Regenerate only datasets whose generators were corrected after audit."""

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
    TokenVocabulary,
    dataset_summary,
    generate_examples,
    write_jsonl,
)


AFFECTED_TASKS = ("multi_hop", "length_generalization", "variable_binding")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(manifest["seed"])
    for task in AFFECTED_TASKS:
        task_dir = args.output_dir / task
        split_manifest: dict[str, object] = {}
        for split in SPLIT_NAMES:
            examples = generate_examples(task, split, DEFAULT_COUNTS[split], seed)
            write_jsonl(task_dir / f"{split}.jsonl", examples)
            split_manifest[split] = dataset_summary(examples)
        manifest["tasks"][task] = split_manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Regenerated only: {', '.join(AFFECTED_TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
