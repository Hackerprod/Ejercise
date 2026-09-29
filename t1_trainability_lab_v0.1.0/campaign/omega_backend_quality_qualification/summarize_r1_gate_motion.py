"""Aggregate saved gate-logit/sigmoid motion; reads JSON only, never checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "r1_state_causal_diagnostics_20260926T181530" / "state_causal_diagnostics_report.json"


def canonical_hash(value: Any) -> str:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize an empty gate distribution")
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(float(row[field]) for row in rows)
    mean = sum(values) / len(values)
    median = _percentile(values, 0.5)
    return {
        "count": len(values),
        "mean": mean,
        "median": median,
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "min": values[0],
        "max": values[-1],
    }


def _direction_counts(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    up = sum(value > 0.0 for value in values)
    down = sum(value < 0.0 for value in values)
    unchanged = sum(value == 0.0 for value in values)
    total = len(values)
    return {
        "up": up,
        "down": down,
        "unchanged": unchanged,
        "proportion_up": up / total,
        "proportion_down": down / total,
        "proportion_unchanged": unchanged / total,
        "direction_field": field,
        "zero_rule": "exact floating-point equality to 0.0",
    }


def summarize(input_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"gate-motion summary is immutable: {output_path}")
    report = json.loads(input_path.read_text(encoding="utf-8"))
    signature = report.get("report_self_sha256")
    unsigned = dict(report)
    unsigned.pop("report_self_sha256", None)
    if not signature or signature != canonical_hash(unsigned):
        raise ValueError("source state-causal diagnostics report self-hash mismatch")
    if (
        report.get("status") != "COMPLETE_DIAGNOSTIC_NO_PASS_FAIL_GATE"
        or report.get("training_updates") != 0
        or report.get("optimizer_restored_or_loaded") is not False
        or report.get("teacher_or_hidden_cache_used") is not False
        or report.get("test_split_loaded") is not False
    ):
        raise ValueError("source report violates read-only diagnostics contract")

    expected_groups = {
        (seed, rounds, backend, round_index)
        for seed in (20260913, 20260914, 20260915, 20260916, 20260917)
        for rounds in (1, 4)
        for backend in ("pytorch", "native")
        for round_index in range(rounds)
    }
    observed_groups: set[tuple[int, int, str, int]] = set()
    gate_rows: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    for route in report["checkpoint_gate_motion"]:
        seed, rounds, backend = int(route["seed"]), int(route["K"]), str(route["backend"])
        values = route.get("gate_values_by_round_index", [])
        by_round: dict[int, list[dict[str, Any]]] = {round_index: [] for round_index in range(rounds)}
        for gate in values:
            round_index, index = int(gate["round"]), int(gate["index"])
            key = (seed, rounds, backend, round_index)
            if key not in expected_groups:
                raise ValueError(f"unexpected gate round identity: {key}")
            if (round_index, index) in {(int(x["round"]), int(x["index"])) for x in values if x is not gate}:
                raise ValueError(f"duplicate gate index in route seed={seed},K={rounds},{backend}")
            if not all(math.isfinite(float(gate[field])) for field in (
                "logit_initial", "logit_update2000", "delta_logit",
                "sigmoid_initial", "sigmoid_update2000", "delta_sigmoid",
            )):
                raise FloatingPointError(f"non-finite gate motion at seed={seed},K={rounds},{backend},round={round_index},index={index}")
            expected_delta_logit = struct.unpack("f", struct.pack("f", float(gate["logit_update2000"]) - float(gate["logit_initial"])))[0]
            expected_delta_sigmoid = struct.unpack("f", struct.pack("f", float(gate["sigmoid_update2000"]) - float(gate["sigmoid_initial"])))[0]
            if float(gate["delta_logit"]) != expected_delta_logit or float(gate["delta_sigmoid"]) != expected_delta_sigmoid:
                raise ValueError(f"stored gate delta does not equal endpoint subtraction at seed={seed},K={rounds},round={round_index},index={index}")
            observed_groups.add(key)
            row = {
                "seed": seed,
                "K": rounds,
                "backend": backend,
                "round": round_index,
                "index": index,
                "logit_initial": float(gate["logit_initial"]),
                "logit_update2000": float(gate["logit_update2000"]),
                "delta_logit": float(gate["delta_logit"]),
                "g_initial": float(gate["sigmoid_initial"]),
                "g_update2000": float(gate["sigmoid_update2000"]),
                "delta_g": float(gate["delta_sigmoid"]),
            }
            gate_rows.append(row)
            by_round[round_index].append(row)
        expected_indices = {(round_index, index) for round_index in range(rounds) for index in range(128)}
        observed_indices = {(int(gate["round"]), int(gate["index"])) for gate in values}
        if observed_indices != expected_indices:
            raise ValueError(f"gate index coverage mismatch for seed={seed},K={rounds},{backend}")
        for round_index, round_values in by_round.items():
            round_summaries.append({
                "seed": seed,
                "K": rounds,
                "backend": backend,
                "round": round_index,
                "gate_count": len(round_values),
                "distributions": {
                    field: _distribution(round_values, field)
                    for field in ("logit_initial", "logit_update2000", "delta_logit", "g_initial", "g_update2000", "delta_g")
                },
                "g_direction_counts": _direction_counts(round_values, "delta_g"),
                "logit_direction_counts": _direction_counts(round_values, "delta_logit"),
            })

    if observed_groups != expected_groups or len(gate_rows) != 6400 or len(round_summaries) != 50:
        raise ValueError("gate-motion coverage must be 5 seeds ×2 backends ×(K1+K4 rounds) ×128 indices")

    run_summaries: list[dict[str, Any]] = []
    for seed in (20260913, 20260914, 20260915, 20260916, 20260917):
        for rounds in (1, 4):
            for backend in ("pytorch", "native"):
                rows = [row for row in gate_rows if row["seed"] == seed and row["K"] == rounds and row["backend"] == backend]
                run_summaries.append({
                    "seed": seed,
                    "K": rounds,
                    "backend": backend,
                    "gate_count": len(rows),
                    "distributions": {
                        field: _distribution(rows, field)
                        for field in ("logit_initial", "logit_update2000", "delta_logit", "g_initial", "g_update2000", "delta_g")
                    },
                    "g_direction_counts": _direction_counts(rows, "delta_g"),
                    "logit_direction_counts": _direction_counts(rows, "delta_logit"),
                })

    summary = {
        "schema": "omega-r1-gate-motion-summary-v1",
        "source_report_path": str(input_path.resolve()),
        "source_report_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_report_self_sha256": signature,
        "source_status": report["status"],
        "optimizer_restored_or_loaded": False,
        "training_updates": 0,
        "checkpoint_count": 20,
        "gate_value_count": len(gate_rows),
        "round_summary_count": len(round_summaries),
        "round_summaries": round_summaries,
        "seed_backend_K_summaries": run_summaries,
        "all_gate_values_and_endpoint_deltas": gate_rows,
        "percentile_method": "linear interpolation at position (n-1)*q (Type 7)",
        "direction_definition": "up if delta_g > 0, down if < 0, unchanged if exactly 0.0",
        "causal_interpretation": "descriptive mechanism only; reset/shuffle NLL ablations are causal evidence; no pass/fail gate",
    }
    summary["summary_self_sha256"] = canonical_hash(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve() if args.output is not None else source.parent / "gate_motion_distribution_summary.json"
    report = summarize(source, output)
    print(json.dumps({
        "output": str(output),
        "summary_sha256": report["summary_self_sha256"],
        "source_report_sha256": report["source_report_sha256"],
        "gate_value_count": report["gate_value_count"],
        "round_summary_count": report["round_summary_count"],
        "optimizer_restored_or_loaded": False,
        "training_updates": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
