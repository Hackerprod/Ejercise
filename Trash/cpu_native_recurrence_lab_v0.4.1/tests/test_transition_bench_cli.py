#!/usr/bin/env python3
"""Validate transition-benchmark chain accounting and reset semantics."""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(binary: Path, chain_length: int) -> dict[str, str]:
    completed = subprocess.run(
        [
            str(binary), "--D", "64", "--S", "4", "--transition", "fixed",
            "--cpus", "0", "--rows", "64", "--warmup", "0",
            "--repetitions", "3", "--chain-length", str(chain_length),
            "--allow-affinity-failure",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"transition bench failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one transition CSV row, got {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    args = parser.parse_args()

    collected: list[dict[str, str]] = []
    for chain_length in (1, 8):
        row = invoke(args.bench, chain_length)
        collected.append(row)
        expected = 64 * 4 * 3 * chain_length
        if int(row["cell_updates"]) != expected:
            raise RuntimeError(f"cell_updates mismatch for chain={chain_length}")
        if int(row["updated_cells"]) != expected:
            raise RuntimeError(f"updated_cells mismatch for chain={chain_length}")
        if row["chain_length"] != str(chain_length):
            raise RuntimeError("chain_length was not recorded")
        if row["state_reset_between_chains"] != "true":
            raise RuntimeError("state reset contract was not recorded")
        if row["synthetic_output_reused"] != "true":
            raise RuntimeError("synthetic output reuse contract was not recorded")
        if row["projection_shift"] != "14":
            raise RuntimeError("fixed-point transition benchmark default must be shift 14")
        if row["valid"] != "true":
            raise RuntimeError(f"transition row invalid: {row.get('error', '')}")
        clipped = int(row["clipped_cells"])
        if clipped < 0 or clipped > expected:
            raise RuntimeError("clipping accounting is outside valid range")

    with tempfile.TemporaryDirectory(prefix="cnrl-transition-") as directory:
        raw = Path(directory) / "transitions.csv"
        fieldnames = list(collected[0].keys())
        with raw.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(collected)
        accepted = subprocess.run(
            [sys.executable, str(args.analyzer), str(raw), "--strict"],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if accepted.returncode != 0:
            raise RuntimeError(f"transition analyzer rejected valid rows\n{accepted.stdout}\n{accepted.stderr}")
        collected[0]["cell_updates"] = str(int(collected[0]["cell_updates"]) * 8)
        tampered = Path(directory) / "tampered.csv"
        with tampered.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(collected)
        rejected = subprocess.run(
            [sys.executable, str(args.analyzer), str(tampered), "--strict"],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if rejected.returncode == 0 or "cell_updates mismatch" not in rejected.stdout:
            raise RuntimeError("transition analyzer failed to reject tampered accounting")

    print("PASS transition benchmark chain/reset/accounting/analyzer contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
