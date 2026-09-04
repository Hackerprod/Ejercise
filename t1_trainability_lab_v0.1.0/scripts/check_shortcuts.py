"""Check positional shortcut heuristics on corrected task datasets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import OUTPUT_CARDINALITIES, SPLIT_NAMES, load_jsonl  # noqa: E402


AFFECTED_TASKS = ("multi_hop", "length_generalization", "variable_binding")


def shortcut(example: object) -> tuple[bool, bool, int, float, str]:
    tokens = example.tokens  # type: ignore[attr-defined]
    target = example.target  # type: ignore[attr-defined]
    task = example.task  # type: ignore[attr-defined]
    if task in {"multi_hop", "length_generalization"}:
        destinations = [
            int(tokens[index + 2].split(":")[1])
            for index, token in enumerate(tokens)
            if token == "REL"
        ]
        prediction = destinations[-1]
        hop_count = int(example.metadata["hop_count"])  # type: ignore[attr-defined]
        target_position = destinations.index(target)
        return prediction == target, target_position == len(destinations) - 1, hop_count, 1 / hop_count, "last_relation_destination"
    attributes: list[tuple[int, int]] = []
    for index, token in enumerate(tokens):
        if token == "ATTR" and index + 3 < len(tokens):
            attributes.append(
                (int(tokens[index + 1].split(":")[1]), int(tokens[index + 3].split(":")[1]))
            )
    assignment_index = tokens.index("ASSIGN")
    target_object = int(tokens[assignment_index + 2].split(":")[1])
    distractors = int(example.metadata["distractor_count"])  # type: ignore[attr-defined]
    target_color = next(color for object_index, color in attributes if object_index == target_object)
    prediction = attributes[0][1]
    attribute_count = distractors + 1
    # Color collisions make prediction accuracy higher than positional chance;
    # object identity is the actual shortcut signal being audited.
    collision_adjusted_null = 1 / attribute_count + (1 - 1 / attribute_count) / 8
    return prediction == target, attributes[0][0] == target_object, attribute_count, collision_adjusted_null, "first_attribute"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    if args.sample_size < 1:
        raise ValueError("sample-size must be positive")

    failures = 0
    for task in AFFECTED_TASKS:
        class_baseline = 1.0 / OUTPUT_CARDINALITIES[task]
        print(f"{task}: class_random_baseline={class_baseline:.6f}")
        for split in SPLIT_NAMES:
            examples = load_jsonl(args.data_dir / task / f"{split}.jsonl")[: args.sample_size]
            hits = 0
            groups: dict[int, list[bool]] = defaultdict(list)
            prediction_groups: dict[int, list[bool]] = defaultdict(list)
            for example in examples:
                hit, positional_hit, group_size, expected_prediction, _ = shortcut(example)
                hits += int(hit)
                groups[group_size].append(positional_hit)
                prediction_groups[group_size].append(hit)
            accuracy = hits / len(examples)
            group_text = ", ".join(
                f"{group}:position={sum(values)/len(values):.3f} (expected={1/group:.3f}),"
                f"prediction={sum(prediction_groups[group])/len(values):.3f}"
                for group, values in sorted(groups.items())
            )
            print(f"  {split}: heuristic_accuracy={accuracy:.3f}; positional_null={group_text}")
            # A fixed-order leak would make this heuristic near-perfect. The
            # per-group positional null is the correct comparison because the
            # task deliberately exposes one of N valid relation/attribute facts.
            if any(abs(sum(values) / len(values) - 1 / group) > 0.12 for group, values in groups.items() if len(values) >= 20):
                failures += 1
    if failures:
        print(f"Shortcut verification: FAIL ({failures} group(s) differ from positional null)")
        return 1
    print("Shortcut verification: PASS (no fixed-position shortcut detected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
