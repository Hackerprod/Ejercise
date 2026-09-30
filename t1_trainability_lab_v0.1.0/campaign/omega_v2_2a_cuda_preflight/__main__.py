"""V2-2A package entry point; establish cuBLAS determinism before torch import."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from .runner import main

raise SystemExit(main())
