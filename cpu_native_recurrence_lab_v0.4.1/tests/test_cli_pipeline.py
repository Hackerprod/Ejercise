#!/usr/bin/env python3
"""End-to-end gate -> CSV -> strict analyzer test, including tamper rejection."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(gate: Path, arguments: list[str]) -> tuple[str, str]:
    completed = subprocess.run(
        [str(gate), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"gate failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise RuntimeError(
            f"gate emitted {len(lines)} non-empty lines instead of header+row: {arguments}"
        )
    return lines[0], lines[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    args = parser.parse_args()

    common = [
        "--D", "64", "--R", "2", "--cpus", "0", "--rows", "64",
        "--warmup", "0", "--repetitions", "1", "--seed", "777",
        "--slot-tile", "4",
    ]
    invocations: list[list[str]] = []
    for variant in ("shared", "clone"):
        invocations.append([
            "--gate", "t0r", "--S", "1", "--kernel", "fused",
            "--variant", variant, *common,
        ])
    for slots in (1, 4):
        for variant in ("shared", "clone"):
            for kernel in ("repeat", "fused"):
                invocations.append([
                    "--gate", "t0m", "--S", str(slots), "--kernel", kernel,
                    "--variant", variant, *common,
                ])
    for transition in ("fixed", "group-rms", "global-rms"):
        for variant in ("shared", "clone"):
            transition_arguments = [
                "--gate", "t0rm", "--S", "4", "--kernel", "fused",
                "--variant", variant, "--transition", transition,
            ]
            # Fixed-point deliberately exercises the CLI default (14); RMS paths
            # keep an explicit test scale so parser/default behavior is separated.
            if transition != "fixed":
                transition_arguments += ["--projection-shift", "8"]
            invocations.append([*transition_arguments, *common])

    header: str | None = None
    rows: list[str] = []
    for invocation in invocations:
        current_header, row = invoke(args.gate, invocation)
        if header is None:
            header = current_header
        elif current_header != header:
            raise RuntimeError("CSV header changed between gate invocations")
        rows.append(row)
    assert header is not None

    with tempfile.TemporaryDirectory(prefix="cnrl-pipeline-") as directory:
        directory_path = Path(directory)
        raw = directory_path / "gates.csv"
        raw.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        accepted = subprocess.run(
            [sys.executable, str(args.analyzer), str(raw), "--strict-structure"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if accepted.returncode != 0:
            raise RuntimeError(
                f"strict analyzer rejected generated CSV\n{accepted.stdout}\n{accepted.stderr}"
            )

        with raw.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
            fieldnames = list(parsed[0].keys())
        fixed_rows = [row for row in parsed if row["gate"] == "t0rm" and row["transition"] == "fixed-point"]
        if not fixed_rows or any(row["projection_shift"] != "14" for row in fixed_rows):
            raise RuntimeError("fixed-point CLI default is not projection_shift=14")
        parsed[0]["mac_total"] = str(int(parsed[0]["mac_total"]) * 8)
        tampered = directory_path / "tampered.csv"
        with tampered.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(parsed)
        rejected = subprocess.run(
            [sys.executable, str(args.analyzer), str(tampered), "--strict-structure"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if rejected.returncode == 0 or "mac_total" not in rejected.stdout:
            raise RuntimeError("strict analyzer failed to reject tampered MAC accounting")

    print("PASS CLI gate/CSV/analyzer pipeline and tamper rejection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
