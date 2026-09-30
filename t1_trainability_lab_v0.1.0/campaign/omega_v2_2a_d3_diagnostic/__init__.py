"""OMEGA V2-2A D3 calibration diagnostic; separate from V2-2A-r1."""

import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

_CAMPAIGN_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
V20_ROOT = _CAMPAIGN_ROOT / "omega_v2_0_conformance"
V22A_ROOT = _CAMPAIGN_ROOT / "omega_v2_2a_cuda_preflight"
ATTEMPT00_ROOT = V22A_ROOT / "results" / "attempt_00"
V22A_R1_RESULT = V22A_ROOT / "results" / "omega_v2_2a_local_preflight" / "preflight_result.json"
for _dependency in (V20_ROOT, V22A_ROOT):
    if str(_dependency) not in sys.path:
        sys.path.insert(0, str(_dependency))
