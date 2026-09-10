"""R2 controls wrapper; delegates byte-identical R1 evaluator logic."""

from __future__ import annotations

import evaluate_t2_i2_r1_controls as _base
from train_t2_i2_r2 import checkpoint_for_seed, output_for_seed

_base.checkpoint_for_seed = checkpoint_for_seed
_base.output_for_seed = output_for_seed


if __name__ == "__main__":
    _base.main()
