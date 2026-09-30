"""OMEGA V2-2A D3Q held-out validation package."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PACKAGE_ROOT = Path(__file__).resolve().parent
CAMPAIGN_ROOT = PACKAGE_ROOT.parent
V20_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
V22A_ROOT = CAMPAIGN_ROOT / "omega_v2_2a_cuda_preflight"
D3_DIAGNOSTIC_ROOT = CAMPAIGN_ROOT / "omega_v2_2a_d3_diagnostic"

for _dependency in (V20_ROOT, V22A_ROOT, CAMPAIGN_ROOT):
    if str(_dependency) not in sys.path:
        sys.path.insert(0, str(_dependency))
