"""OMEGA V2-2B pilot CLI; set the deterministic cuBLAS workspace first."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
