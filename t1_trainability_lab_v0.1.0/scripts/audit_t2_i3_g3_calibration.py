"""G3 real calibration-pair matching evaluator."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import audit_t2_i2_r1_matching as base
import t2_i2_r3_semantic_writer as writer_mod
from t2_i3_common import build_calibration_manifest
from t2_i3_think0 import Think0
from train_t2_i2_r3 import checkpoint_for_seed as writer_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--think-checkpoint", type=Path, required=True); args = parser.parse_args(); think = Think0(); think.load_state_dict(torch.load(args.think_checkpoint, weights_only=False)["think"], strict=True); manifest = build_calibration_manifest(); original_encode = base.encode; base.REALIZATIONS = (("CALIBRATION", "AT_LEAST", "AVOID"),); base.ORDERS = ("NORMAL",); base.real_cases = lambda: [("calibration", 10, pair["lower"], pair["forbidden"]) for pair in manifest["calibration"]]
    def encode(writer, text):
        with torch.no_grad(): return think( original_encode(writer, text).unsqueeze(0))[0]
    base.CompetitiveSemanticWriter = writer_mod.CompetitiveSemanticWriter; base.checkpoint_for_seed = lambda _seed: writer_checkpoint(6301); base.output_for_seed = lambda _seed: Path(__file__).resolve().parents[1] / "campaign" / f"t2_i3_g3_seed{args.seed}"; base.encode = encode; argv = sys.argv; sys.argv = [argv[0], "--seed", str(args.seed)]
    try: base.main()
    finally: sys.argv = argv
    source = base.output_for_seed(args.seed) / "matching" / "results.json"; result = json.loads(source.read_text(encoding="utf-8")); count = len(manifest["calibration"]); matching = result["results"]["CALIBRATION_NORMAL"]["matching"]; result["task"] = "T2-I3"; result["gate"] = "G3"; result["calibration_manifest_sha256"] = manifest["sha256"]; result["status"] = "passed" if matching["success"] == count else "failed"; matching["samples"] = count; matching["fraction"] = matching["success"] / count; output = Path(__file__).resolve().parents[1] / "campaign" / f"t2_i3_seed{args.seed}" / "g3" / "results.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
