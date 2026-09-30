"""D3Q CLI; set the deterministic cuBLAS workspace before importing torch."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from .runner import main

raise SystemExit(main())
