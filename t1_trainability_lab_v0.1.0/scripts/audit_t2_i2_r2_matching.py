"""R2 matching wrapper; delegates byte-identical R1 matching audit logic."""

from __future__ import annotations

import audit_t2_i2_r1_matching as _base
from train_t2_i2_r2 import checkpoint_for_seed, output_for_seed

_base.checkpoint_for_seed = checkpoint_for_seed
_base.output_for_seed = output_for_seed


if __name__ == "__main__":
    _base.main()
