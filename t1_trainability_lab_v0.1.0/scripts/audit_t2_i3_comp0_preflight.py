"""Mandatory zero-correction COMP-0 equivalence preflight."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, tensorize
from t2_i3_common import MANIFEST_PATH, build_calibration_manifest, load_writer
from t2_i3_comp0 import Composition0, parameter_count


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        return writer(token_ids, lengths)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    comp = Composition0()
    writer = load_writer()
    manifest = build_calibration_manifest(MANIFEST_PATH)
    max_abs = 0.0
    with torch.no_grad():
        for pair in manifest["calibration"]:
            raw = encode(writer, f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}").unsqueeze(0)
            max_abs = max(max_abs, float((comp(raw) - raw.sum(dim=1)).abs().max().item()))
    if parameter_count(comp) != 1072 or max_abs != 0.0 or any(parameter.requires_grad is False for parameter in comp.parameters()):
        raise RuntimeError(f"COMP-0 zero preflight failed: params={parameter_count(comp)} max_abs={max_abs}")
    script = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(script)
    subprocess.run([sys.executable, str(script / "audit_t2_i3_comp0_controls.py"), "--seed", str(args.seed), "--gate", "G1"], check=True, env=environment)
    subprocess.run([sys.executable, str(script / "audit_t2_i3_comp0_controls.py"), "--seed", str(args.seed), "--gate", "G2"], check=True, env=environment)
    subprocess.run([sys.executable, str(script / "audit_t2_i3_comp0_g3.py"), "--seed", str(args.seed)], check=True, env=environment)
    g1 = json.loads((CAMPAIGN / f"t2_i3_comp0_seed{args.seed}" / "g1" / "results.json").read_text(encoding="utf-8"))
    g2 = json.loads((CAMPAIGN / f"t2_i3_comp0_seed{args.seed}" / "g2" / "results.json").read_text(encoding="utf-8"))
    g3 = json.loads((CAMPAIGN / f"t2_i3_comp0_g3_seed{args.seed}" / "results.json").read_text(encoding="utf-8"))
    if g1["status"] != "passed" or g2["matching"]["success"] != 386 or g3["matching"]["success"] != 129:
        raise RuntimeError(f"COMP-0 zero behavioral preflight failed: G1={g1['status']} G2={g2['matching']['success']} G3={g3['matching']['success']}")
    result = {"status": "passed", "task": "T2-I3-COMP-0", "phase": "zero_correction_preflight", "training": False, "parameter_count": parameter_count(comp), "w_out_zero": True, "max_abs_correction": max_abs, "calibration_manifest_sha256": manifest["sha256"], "behavior": {"g1": "passed", "g2": "386/386", "g3": "129/139"}}
    output = CAMPAIGN / f"t2_i3_comp0_preflight_seed{args.seed}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
