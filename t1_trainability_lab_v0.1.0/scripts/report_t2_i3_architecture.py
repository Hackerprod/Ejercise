"""Write T2-I3 architecture and calibration contract."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from t2_i3_common import MANIFEST_PATH, WRITER_CHECKPOINT, build_calibration_manifest
from t2_i3_think0 import Think0, architecture_report
from train_t2_i3 import output_for_seed

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "campaign" / "t2_i3_architecture_report.json"); args = parser.parse_args(); manifest = build_calibration_manifest(MANIFEST_PATH); report = architecture_report(Think0()) | {"writer_checkpoint": str(WRITER_CHECKPOINT), "calibration_manifest_sha256": manifest["sha256"], "seeds": [6401, 6402, 6403], "output_pattern": str(output_for_seed(6401).parent / "t2_i3_think0_seed{seed}"), "gates": ["G0", "G1", "G2", "G3", "G4", "G5"]}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(args.output), "report": report}, indent=2, sort_keys=True))

if __name__ == "__main__": main()
