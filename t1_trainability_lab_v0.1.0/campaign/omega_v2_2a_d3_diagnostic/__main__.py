"""Diagnostic package CLI; set cuBLAS determinism before importing torch."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from .runner import main

raise SystemExit(main())
