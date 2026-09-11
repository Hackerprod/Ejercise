"""T2-I3 smoke: THINK shape, G0, parameter count, manifest, K0 regression."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH, WRITER_CHECKPOINT, build_calibration_manifest, load_writer
from t2_i3_think0 import Think0, parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6401); parser.parse_args()
    forbidden = ("constraints", "lower", "forbidden", "floor", "avoid", "tokens", "clauses", "segments", "split")
    parameters = inspect.signature(Think0.forward).parameters
    if tuple(parameters) != ("self", "workspace") or any(name in parameters for name in forbidden): raise RuntimeError("G0 oracle isolation failed")
    manifest = build_calibration_manifest(MANIFEST_PATH); writer = load_writer(); think = Think0(); token_ids, lengths = tensorize(["NOOP VALUE_0", "AT_LEAST VALUE_12 AND AVOID VALUE_23"])
    with torch.no_grad():
        raw = writer(token_ids, lengths); k0 = think.run_rounds(raw, 0); k1 = think.run_rounds(raw, 1); k2 = think(raw)
    raw_diff = float((k0 - raw).abs().max()); reference_path = Path(__file__).resolve().parents[1] / "campaign" / "t2_i2_r3_seed6301" / "controls" / "results.json"; reference = json.loads(reference_path.read_text(encoding="utf-8")); expected = {"position_invariance": (128, 128), "early_memory": (128, 128), "NONE": (2048, 2048), "AT_LEAST": (2048, 2048), "MINIMUM": (2048, 2048), "AVOID": (2048, 2048), "EXCLUDE": (2048, 2048)}; actual = {"position_invariance": (reference["summary"]["position_invariance"]["samples"], reference["summary"]["position_invariance"]["success"]), "early_memory": (reference["summary"]["early_memory"]["samples"], reference["summary"]["early_memory"]["success"]), **{name: (reference["summary"]["noop_exhaustive"][name]["samples"], reference["summary"]["noop_exhaustive"][name]["success"]) for name in ("NONE", "AT_LEAST", "MINIMUM", "AVOID", "EXCLUDE")}}
    if raw_diff != 0.0 or actual != expected: raise RuntimeError(f"K0_REGRESSION: raw_diff={raw_diff}, actual={actual}")
    result = {"status": "passed", "task": "T2-I3", "phase": "implementation_smoke", "training": False, "g0": "passed", "k0_regression": {"raw_roundtrip_max_abs_diff": raw_diff, "status": "passed", "reference_results_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(), "counts": actual}, "k1_shape": list(k1.shape), "k2_shape": list(k2.shape), "think_parameters": parameter_count(think), "calibration_manifest": {"path": str(MANIFEST_PATH), "sha256": manifest["sha256"], "candidate_count": manifest["candidate_count"], "calibration_count": manifest["calibration_count"], "heldout_count": manifest["heldout_count"], "salt": manifest["salt"], "bucket_rule": manifest["bucket_rule"]}, "writer_checkpoint": str(WRITER_CHECKPOINT)}; output = Path(__file__).resolve().parents[1] / "campaign" / "t2_i3_smoke.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
