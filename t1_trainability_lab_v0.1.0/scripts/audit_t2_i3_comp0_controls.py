"""COMP-0 G1/G2 controls using frozen raw Writer slots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import evaluate_t2_i2_r1_controls as g1_base
import audit_t2_i2_r3_cardinality as g2_base
import t2_i2_r3_semantic_writer as writer_mod
from train_t2_i2_r3 import checkpoint_for_seed


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--gate", choices=("G1", "G2"), required=True)
    args = parser.parse_args()
    base = g1_base if args.gate == "G1" else g2_base
    original_encode = base.encode
    output = CAMPAIGN / f"t2_i3_comp0_{args.gate.lower()}_seed{args.seed}"
    base.CompetitiveSemanticWriter = writer_mod.CompetitiveSemanticWriter
    base.checkpoint_for_seed = lambda _seed: checkpoint_for_seed(6301)
    base.output_for_seed = lambda _seed: output
    if args.gate == "G2":
        panel_path = CAMPAIGN / "t2_i2_r3_seed6301" / "panel_manifest.json"
        base.load_panels = lambda _seed: (json.loads(panel_path.read_text(encoding="utf-8")), __import__("hashlib").sha256(panel_path.read_bytes()).hexdigest())
    base.encode = original_encode
    argv = sys.argv
    sys.argv = [argv[0], "--seed", str(args.seed)]
    try:
        base.main()
    finally:
        sys.argv = argv
    source = output / ("controls" if args.gate == "G1" else "cardinality") / "results.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    result["task"] = "T2-I3-COMP-0"
    result["gate"] = args.gate
    target = CAMPAIGN / f"t2_i3_comp0_seed{args.seed}" / args.gate.lower() / "results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
