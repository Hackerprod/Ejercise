"""G1 real regression evaluator for THINK-0, with K0/K1/K2 diagnostics."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import evaluate_t2_i2_r1_controls as base
import t2_i2_r3_semantic_writer as writer_mod
from t2_i3_think1 import Think1
from train_t2_i2_r3 import checkpoint_for_seed as writer_checkpoint
from train_t2_i3 import checkpoint_for_seed as think_checkpoint

def run(k: int, seed: int, checkpoint: Path | None) -> dict:
    think = Think1()
    if checkpoint is not None: think.load_state_dict(torch.load(checkpoint, weights_only=False)["think"], strict=True)
    original_encode = base.encode
    def encode(writer, text):
        with torch.no_grad(): return think.run_rounds(original_encode(writer, text).unsqueeze(0), k)[0]
    base.CompetitiveSemanticWriter = writer_mod.CompetitiveSemanticWriter; base.checkpoint_for_seed = lambda _seed: writer_checkpoint(6301); base.output_for_seed = lambda _seed: Path(__file__).resolve().parents[1] / "campaign" / f"t2_i3_g1_k{k}_seed{seed}"; base.encode = encode; argv = sys.argv; sys.argv = [argv[0], "--seed", str(seed)]
    try: base.main()
    finally: sys.argv = argv
    result_path = base.output_for_seed(seed) / "controls" / "results.json"; return json.loads(result_path.read_text(encoding="utf-8"))

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--think-checkpoint", type=Path, default=None); args = parser.parse_args(); outputs = {str(k): run(k, args.seed, args.think_checkpoint if k else None) for k in (0, 1, 2)}; path = Path(__file__).resolve().parents[1] / "campaign" / f"t2_i3_seed{args.seed}" / "g1"; path.mkdir(parents=True, exist_ok=True); result = {"status": "passed" if outputs["2"]["status"] == "passed" else "failed", "task": "T2-I3", "gate": "G1", "diagnostic_rounds": outputs}; (path / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
