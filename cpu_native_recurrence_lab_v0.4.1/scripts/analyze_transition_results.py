#!/usr/bin/env python3
"""Audit and summarize CNRL transition microbenchmark CSVs."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED = {
    "project_version", "seed", "require_affinity", "allow_smt_siblings",
    "D", "S", "transition", "warmup_chains", "chains", "chain_length",
    "state_reset_between_chains", "synthetic_output_reused", "worker_count",
    "cpus", "physical_cores", "rows", "projection_shift",
    "state_multiplier", "output_multiplier", "final_shift", "target_rms", "epsilon",
    "elapsed_seconds", "cell_updates", "updated_cells",
    "cell_updates_per_second", "ns_per_cell", "clipped_cells",
    "clipping_rate", "all_affinity_succeeded", "affinity_error",
    "state_checksum", "valid", "error",
}


def semicolon_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(";") if item]


def close(actual: float, expected: float, tolerance: float = 2e-8) -> bool:
    scale = max(1.0, abs(actual), abs(expected))
    return math.isfinite(actual) and abs(actual - expected) <= tolerance * scale


def load(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
                continue
            for row in reader:
                row["source"] = str(path)
                rows.append(row)
    return rows


def audit(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    versions = {row.get("project_version", "") for row in rows}
    if len(versions) != 1 or "" in versions:
        errors.append(f"mixed or missing project versions: {sorted(versions)}")
    for index, row in enumerate(rows, 1):
        try:
            D = int(row["D"]); S = int(row["S"])
            chains = int(row["chains"]); chain_length = int(row["chain_length"])
            workers = int(row["worker_count"])
            elapsed = float(row["elapsed_seconds"])
            if D <= 0 or S <= 0 or chains <= 0 or chain_length <= 0 or workers <= 0:
                errors.append(f"row {index}: non-positive shape/count")
            expected = D * S * chains * chain_length
            updated = int(row["updated_cells"])
            clipped = int(row["clipped_cells"])
            if int(row["cell_updates"]) != expected:
                errors.append(f"row {index}: cell_updates mismatch")
            if updated != expected:
                errors.append(f"row {index}: updated_cells mismatch")
            if clipped < 0 or clipped > expected:
                errors.append(f"row {index}: clipped_cells outside valid range")
            if elapsed <= 0 or not math.isfinite(elapsed):
                errors.append(f"row {index}: invalid elapsed_seconds")
            else:
                if not close(float(row["cell_updates_per_second"]), expected / elapsed):
                    errors.append(f"row {index}: cell_updates_per_second mismatch")
                if not close(float(row["ns_per_cell"]), elapsed * 1e9 / expected):
                    errors.append(f"row {index}: ns_per_cell mismatch")
            expected_clipping = clipped / expected if expected else 0.0
            if not close(float(row["clipping_rate"]), expected_clipping):
                errors.append(f"row {index}: clipping_rate mismatch")
            cpus = semicolon_ints(row["cpus"])
            physical = semicolon_ints(row["physical_cores"])
            shard_rows = semicolon_ints(row["rows"])
            if len(cpus) != workers or len(physical) != workers or len(shard_rows) != workers:
                errors.append(f"row {index}: worker/list cardinality mismatch")
            if row["allow_smt_siblings"] == "false" and len(set(physical)) != len(physical):
                errors.append(f"row {index}: SMT siblings present without opt-in")
            if sum(shard_rows) != D:
                errors.append(f"row {index}: shard rows do not sum to D")
            if row["state_reset_between_chains"] != "true":
                errors.append(f"row {index}: state was not reset between chains")
            if row["require_affinity"] not in {"true", "false"}:
                errors.append(f"row {index}: invalid require_affinity flag")
            if row["allow_smt_siblings"] not in {"true", "false"}:
                errors.append(f"row {index}: invalid allow_smt_siblings flag")
            if row["synthetic_output_reused"] != "true":
                errors.append(f"row {index}: synthetic-output semantics changed")
            if row["require_affinity"] == "true" and row["all_affinity_succeeded"] != "true":
                errors.append(f"row {index}: required affinity failed")
            if row["valid"] != "true":
                errors.append(f"row {index}: invalid benchmark ({row.get('error', '')})")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            errors.append(f"row {index}: parse/audit failure ({error})")
    return errors


def report(rows: list[dict[str, str]], errors: list[str]) -> str:
    lines = ["# CNRL transition benchmark analysis", "",
             f"Rows loaded: **{len(rows)}**.", "", "## Structural audit", ""]
    if errors:
        lines.append(f"**REJECTED:** {len(errors)} inconsistency(ies).")
        lines.extend(f"- {error}" for error in errors[:100])
    else:
        lines.append("**PASS:** chain accounting, reset semantics, affinity, sharding and clipping rates are consistent.")
    groups: dict[tuple[object, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["D"]), int(row["S"]), row["transition"],
            int(row["chain_length"]), int(row["projection_shift"]),
            int(row["state_multiplier"]), int(row["output_multiplier"]),
            int(row["final_shift"]), float(row["target_rms"]),
            float(row["epsilon"]), row["cpus"], row["rows"], row["seed"],
        )
        groups[key].append(row)
    lines += ["", "## Results", "",
              "| D | S | transition | chain | shift | state/output/final | target RMS | median ns/cell | clipping | numeric reading |",
              "|---:|---:|---|---:|---:|---|---:|---:|---:|---|"]
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        samples = groups[key]
        ns = statistics.median(float(row["ns_per_cell"]) for row in samples)
        clipping = statistics.median(float(row["clipping_rate"]) for row in samples)
        reading = "OK" if clipping <= 0.01 else "ADVERTENCIA" if clipping <= 0.05 else "SATURACIÓN"
        scale_text = f"{key[5]}/{key[6]}/{key[7]}"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | "
            f"{scale_text} | {key[8]:g} | {ns:.3f} | {clipping:.3%} | {reading} |"
        )
    lines += ["", "`chain_length=1` mide coste aislado; una cadena R solo caracteriza deriva bajo el output sintético declarado. La validez autoritativa de una transición sigue siendo su clipping dentro de T0-RM real.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    expanded: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.csv")))
        else:
            expanded.append(path)
    rows = load(expanded)
    if not rows:
        print("No compatible transition CSV rows found", file=sys.stderr)
        return 2
    errors = audit(rows)
    text = report(rows, errors)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if errors and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
