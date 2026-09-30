"""OMEGA V2-2A CUDA conformance/trainability preflight package."""

import os
from pathlib import Path
import sys

# Must be set before any CUDA context or cuBLAS handle is initialized.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

V2_0_ROOT = Path(__file__).resolve().parents[1] / "omega_v2_0_conformance"
if str(V2_0_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_0_ROOT))
