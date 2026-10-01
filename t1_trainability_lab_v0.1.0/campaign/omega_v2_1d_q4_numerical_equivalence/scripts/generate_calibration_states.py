from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
from typing import Any

import torch


DIMS = (512, 640)
SLOTS = (1, 4, 8, 16)
FAMILIES = ("HISTORICAL_FORMULA", "CAL_RANDN")


def historical_state_bytes(d: int, m: int) -> bytes:
    count = d * m
    return b"".join(
        struct.pack("<f", (((i * 37 + d + m) % 127) - 63) / 256.0)
        for i in range(count)
    )


def cal_randn_state_bytes(d: int, m: int) -> tuple[bytes, int]:
    seed = 20260930 + d + m
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.randn((m, d), generator=generator, dtype=torch.float32)
    tensor = tensor.contiguous()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tuple(tensor.shape) != (m, d):
        raise RuntimeError("CAL_RANDN tensor contract mismatch")
    if not tensor.is_contiguous():
        raise RuntimeError("CAL_RANDN tensor is not contiguous")
    return tensor.numpy().tobytes(order="C"), seed


def get_file_hash(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "''")
    command = f"(Get-FileHash -Algorithm SHA256 -LiteralPath '{escaped}').Hash"
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = process.stdout.strip().upper()
    if len(digest) != 64:
        raise RuntimeError(f"Get-FileHash returned an invalid SHA-256 for {path}")
    return digest


def generate(output_root: Path) -> dict[str, Any]:
    if sys.byteorder != "little":
        raise RuntimeError("raw FP32 input persistence requires the frozen little-endian host")
    if output_root.exists():
        raise FileExistsError(f"calibration input directory is immutable: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for d in DIMS:
        for m in SLOTS:
            for family in FAMILIES:
                seed: int | None = None
                if family == "HISTORICAL_FORMULA":
                    raw = historical_state_bytes(d, m)
                else:
                    raw, seed = cal_randn_state_bytes(d, m)
                expected = d * m * 4
                if len(raw) != expected:
                    raise RuntimeError(f"wrong state byte count for d={d}, m={m}, family={family}")
                path = output_root / f"d{d}_m{m}_{family}.f32"
                path.write_bytes(raw)
                records.append(
                    {
                        "d": d,
                        "m": m,
                        "shape": [m, d],
                        "family": family,
                        "seed": seed,
                        "dtype": "float32",
                        "device": "cpu",
                        "byte_order": "little-endian",
                        "layout": "contiguous-row-major",
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": get_file_hash(path),
                    }
                )
    manifest = {
        "schema": "omega-v2-1d-calibration-state-inputs-v1",
        "classification": "CALIBRATION_INPUTS_PREPARED_NO_NATIVE_EXECUTION",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_available": torch.cuda.is_available(),
        "historical_state_formula": "(((i*37+d+m) mod 127)-63)/256.0f",
        "cal_randn_seed_formula": "20260930+d+m",
        "native_rng_used": False,
        "states": records,
    }
    manifest_path = output_root / "state_inputs_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = generate(args.output_root)
    print(json.dumps({"state_count": len(result["states"]), "torch_version": result["torch_version"], "native_execution": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
