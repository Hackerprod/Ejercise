"""Audit elementary ALU transition coverage in a canonical sequential split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from train_u0a import OPCODE_IDS, VALUE_BASE, build_canonical_data, save_json


NAMES = {OPCODE_IDS["ALU_ADD"]: "ADD", OPCODE_IDS["ALU_SUB"]: "SUB", OPCODE_IDS["ALU_MUL"]: "MUL"}


def audit(examples: list[object]) -> dict[str, object]:
    seen: set[tuple[int, int, int]] = set()
    counts = Counter()
    for example in examples:
        current = int(example.initial_ids[1] - VALUE_BASE)
        for index in range(example.hop_count):
            opcode = example.opcodes[index]
            operand = int(example.immediates[index] - VALUE_BASE)
            if opcode not in NAMES:
                continue
            seen.add((opcode, current, operand))
            counts[NAMES[opcode]] += 1
            if opcode == OPCODE_IDS["ALU_ADD"]:
                current = (current + operand) % 32
            elif opcode == OPCODE_IDS["ALU_SUB"]:
                current = (current - operand) % 32
            else:
                current = (current * operand) % 32
    unique_by_operation = {name: len({transition for transition in seen if NAMES[transition[0]] == name}) for name in NAMES.values()}
    return {
        "examples": len(examples),
        "hop_counts": dict(Counter(int(example.hop_count) for example in examples)),
        "elementary_transition_instances": sum(counts.values()),
        "unique_elementary_transitions": len(seen),
        "unique_by_operation": unique_by_operation,
        "coverage_total": len(seen) / 3072,
        "coverage_by_operation": {name: value / 1024 for name, value in unique_by_operation.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = build_canonical_data(args.campaign_dir)
    result = {"campaign_dir": str(args.campaign_dir), "split": args.split, **audit(datasets["sequential_update"][args.split])}
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
