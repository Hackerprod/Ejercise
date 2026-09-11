"""R3 diagnostic wrapper; reuses frozen cross-talk logic with R3 writer."""

from __future__ import annotations

import audit_t2_i2_r2_alg_crosstalk as _base
import t2_i2_r3_semantic_writer as _r3w
from train_t2_i2_r3 import checkpoint_for_seed, output_for_seed

_base.CompetitiveSemanticWriter = _r3w.CompetitiveSemanticWriter
_base.checkpoint_for_seed = checkpoint_for_seed
_base.output_for_seed = output_for_seed

if __name__ == "__main__": _base.main()
