"""R3 controls wrapper; delegates byte-identical R1 controls logic."""

from __future__ import annotations

import evaluate_t2_i2_r1_controls as _base
import t2_i2_r3_semantic_writer as _r3w
from train_t2_i2_r3 import checkpoint_for_seed, output_for_seed

_base.CompetitiveSemanticWriter = _r3w.CompetitiveSemanticWriter
_base.checkpoint_for_seed = checkpoint_for_seed
_base.output_for_seed = output_for_seed

if __name__ == "__main__": _base.main()
