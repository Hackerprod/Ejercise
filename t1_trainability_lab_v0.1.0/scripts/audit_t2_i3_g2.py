"""G2 real post-THINK cardinality evaluator for K=0/K=1/K=2."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import audit_t2_i2_r3_cardinality as base
import t2_i2_r3_semantic_writer as writer_mod
from t2_i3_think1 import Think1
from train_t2_i2_r3 import checkpoint_for_seed as writer_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--think-checkpoint", type=Path, required=True); args = parser.parse_args(); think = Think1(); think.load_state_dict(torch.load(args.think_checkpoint, weights_only=False)["think"], strict=True); original_encode = base.encode; outputs = {}
    for k in (0, 1, 2):
        def encode(writer, text, round_count=k):
            with torch.no_grad(): return think.run_rounds(original_encode(writer, text).unsqueeze(0), round_count)[0]
        base.CompetitiveSemanticWriter = writer_mod.CompetitiveSemanticWriter; base.checkpoint_for_seed = lambda _seed: writer_checkpoint(6301); base.output_for_seed = lambda _seed, round_count=k: Path(__file__).resolve().parents[1] / "campaign" / f"t2_i3_g2_k{round_count}_seed{args.seed}"; base.load_panels = lambda _seed: (json.loads((Path(__file__).resolve().parents[1] / "campaign" / "t2_i2_r3_seed6301" / "panel_manifest.json").read_text(encoding="utf-8")), __import__("hashlib").sha256((Path(__file__).resolve().parents[1] / "campaign" / "t2_i2_r3_seed6301" / "panel_manifest.json").read_bytes()).hexdigest()); base.encode = encode; argv = sys.argv; sys.argv = [argv[0], "--seed", str(args.seed)]
        try: base.main()
        finally: sys.argv = argv
        outputs[str(k)] = json.loads((base.output_for_seed(args.seed) / "cardinality" / "results.json").read_text(encoding="utf-8"))
    path = Path(__file__).resolve().parents[1] / "campaign" / f"t2_i3_seed{args.seed}" / "g2"; path.mkdir(parents=True, exist_ok=True); result = {"status": "passed" if outputs["2"]["matching"]["success"] == 386 else "failed", "task": "T2-I3", "gate": "G2", "diagnostic_rounds": outputs}; (path / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
