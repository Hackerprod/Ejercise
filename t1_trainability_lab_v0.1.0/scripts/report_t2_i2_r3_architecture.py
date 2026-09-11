"""Write T2-I2-R3 architecture contract before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, architecture_report

ROOT = Path(__file__).resolve().parents[1]; DEFAULT_OUTPUT = ROOT / "campaign" / "t2_i2_r3_architecture_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(); writer = CompetitiveSemanticWriter(); report = architecture_report(writer) | {"task": "T2-I2-R3", "writer_reused_from": "T2-I2-R2 parameters, additive pooling", "lambda_card": 1.0, "loss": "L_main + 1.0 * L_card", "panel": "R2 data-derived panels, unchanged", "heldout": "(1,1) absent; R1 evaluator logic reused", "seeds": [6301, 6302, 6303], "output_pattern": "campaign/t2_i2_r3_seed{seed}/", "collision_patterns": ["campaign/t2_i2_seed{seed}/", "campaign/t2_i2_r1_seed{seed}/", "campaign/t2_i2_r2_seed{seed}/", "campaign/t2_i2_r3_seed{seed}/"]}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(args.output), "report": report}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
