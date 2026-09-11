"""G5 real exhaustive evaluator entry point using the frozen R3 heldout battery."""
from __future__ import annotations
import argparse, sys
import torch
import evaluate_t2_i2_r1_heldout as base
import t2_i2_r3_semantic_writer as writer_mod
from t2_i3_think0 import Think0
from train_t2_i2_r3 import checkpoint_for_seed as writer_checkpoint, output_for_seed

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--think-checkpoint", required=True); args = parser.parse_args(); think = Think0(); think.load_state_dict(torch.load(args.think_checkpoint, weights_only=False)["think"], strict=True); original_encode = base.encode
    def encode(writer, text):
        with torch.no_grad(): return think(original_encode(writer, text).unsqueeze(0))[0]
    base.CompetitiveSemanticWriter = writer_mod.CompetitiveSemanticWriter; base.checkpoint_for_seed = lambda _seed: writer_checkpoint(6301); base.output_for_seed = output_for_seed; base.encode = encode; argv = sys.argv; sys.argv = [argv[0], "--seed", str(args.seed)]
    try: base.main()
    finally: sys.argv = argv

if __name__ == "__main__": main()
