#!/usr/bin/env python3
"""Validate and export a NumPy matrix as MRDL raw little-endian float32."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("numpy is required") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy matrix [tokens, dimension]")
    parser.add_argument("output", type=Path, help="raw .f32 output")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    args = parser.parse_args()

    matrix = np.load(args.input, allow_pickle=False)
    if matrix.shape != (args.rows, args.dimension):
        raise SystemExit(f"shape {matrix.shape} != {(args.rows, args.dimension)}")
    matrix = np.asarray(matrix, dtype="<f4", order="C")
    if not np.isfinite(matrix).all():
        raise SystemExit("matrix contains NaN or infinity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    matrix.tofile(temporary)
    temporary.replace(args.output)
    print(f"wrote {args.output} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
