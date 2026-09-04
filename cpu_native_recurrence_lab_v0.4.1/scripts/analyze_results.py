#!/usr/bin/env python3
"""Analyze raw CNRL gate CSVs without inferring DRAM traffic from GMAC/s.

The script treats one_pass_weight_gb_per_second as the S-corrected weight-stream
metric. logical_weight_load_gb_per_second includes repeated slot-tile loads and
is intentionally reported separately.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

NUMERIC = {
    "D": int, "S": int, "R": int, "total_rows": int, "slot_tile": int,
    "seed": int, "projection_shift": int, "state_multiplier": int,
    "output_multiplier": int, "final_shift": int, "target_rms": float,
    "epsilon": float, "warmup_repetitions": int, "timed_repetitions": int,
    "batch_repeat": int, "variant_order": int,
    "sequences_per_repetition": int, "worker_count": int,
    "elapsed_seconds": float, "mac_total": int, "mac_per_second": float,
    "base_weight_bytes": int, "allocated_weight_bytes": int,
    "logical_weight_load_bytes": int, "one_pass_weight_bytes": int,
    "distinct_weight_storage_bytes": int,
    "one_pass_weight_gb_per_second": float,
    "logical_weight_load_gb_per_second": float,
    "clipped_cells": int, "transition_cells": int,
}

REQUIRED_COLUMNS = {
    "project_version", "gate", "D", "S", "R", "total_rows", "kernel",
    "variant", "transition", "timing_scope", "slot_tile", "seed",
    "phase_profile", "require_affinity", "projection_shift",
    "state_multiplier", "output_multiplier", "final_shift", "target_rms",
    "epsilon", "warmup_repetitions", "timed_repetitions",
    "sequences_per_repetition", "worker_count", "allow_smt_siblings",
    "cpus", "physical_cores", "rows", "base_weight_bytes",
    "allocated_weight_bytes", "logical_weight_load_bytes",
    "one_pass_weight_bytes", "distinct_weight_storage_bytes",
    "elapsed_seconds", "mac_total", "mac_per_second",
    "logical_weight_load_gb_per_second", "one_pass_weight_gb_per_second",
    "output_checksum", "state_checksum", "round_sink",
    "weight_hash_signature", "clone_hashes_equal",
    "clone_addresses_distinct", "all_affinity_succeeded",
    "clipped_cells", "transition_cells", "valid", "error",
}


def load_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                continue
            for raw in reader:
                row: dict[str, object] = dict(raw)
                row["source"] = str(path)
                try:
                    for key, converter in NUMERIC.items():
                        if key in raw and raw[key] != "":
                            row[key] = converter(raw[key])
                except (TypeError, ValueError) as error:
                    row["_parse_error"] = str(error)
                rows.append(row)
    return rows


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def condition_key(row: dict[str, object], omit: set[str] | None = None) -> tuple[object, ...]:
    omit = omit or set()
    keys = ["gate", "D", "S", "R", "total_rows", "kernel", "transition",
            "timing_scope", "slot_tile", "rows", "cpus", "physical_cores",
            "seed", "phase_profile", "require_affinity", "projection_shift",
            "state_multiplier", "output_multiplier", "final_shift", "target_rms",
            "epsilon", "warmup_repetitions", "timed_repetitions",
            "sequences_per_repetition", "allow_smt_siblings", "project_version"]
    return tuple(row.get(key) for key in keys if key not in omit)


EQUIVALENCE_FIELDS = [
    "project_version", "gate", "D", "S", "R", "total_rows", "kernel",
    "variant", "transition", "timing_scope", "slot_tile", "seed", "phase_profile",
    "require_affinity", "projection_shift", "state_multiplier",
    "output_multiplier", "final_shift", "target_rms", "epsilon",
    "warmup_repetitions", "timed_repetitions", "sequences_per_repetition",
    "worker_count", "allow_smt_siblings", "cpus", "physical_cores", "rows",
    "batch_repeat",
]


def equivalence_key(row: dict[str, object], omit: set[str] | None = None) -> tuple[object, ...]:
    omit = omit or set()
    return tuple(row.get(field) for field in EQUIVALENCE_FIELDS if field not in omit)


def relative_close(actual: float, expected: float, tolerance: float = 2.0e-8) -> bool:
    scale = max(1.0, abs(actual), abs(expected))
    return abs(actual - expected) <= tolerance * scale


def fused_passes(slots: int, tile: int) -> int:
    passes = 0
    left = slots
    while left > 0:
        if tile >= 8 and left >= 8:
            left -= 8
        elif tile >= 4 and left >= 4:
            left -= 4
        elif tile >= 2 and left >= 2:
            left -= 2
        else:
            left -= 1
        passes += 1
    return passes


def parse_semicolon_ints(value: object) -> list[int]:
    return [int(item) for item in str(value).split(";") if item]


def accounting_errors(row: dict[str, object], index: int) -> list[str]:
    errors: list[str] = []
    try:
        D = int(row["D"]); S = int(row["S"]); R = int(row["R"])
        total_rows = int(row["total_rows"])
        repetitions = int(row["timed_repetitions"])
        sequences = int(row["sequences_per_repetition"])
        elapsed = float(row["elapsed_seconds"])
        base = total_rows * D
        expected_mac = base * S * R * sequences * repetitions
        expected_one_pass = base * R * sequences * repetitions
        passes = (fused_passes(S, int(row["slot_tile"]))
                  if row.get("kernel") == "avx2-fused" else S)
        expected_logical = base * passes * R * sequences * repetitions
        rows_by_worker = parse_semicolon_ints(row.get("rows", ""))
        block_count = R if row.get("variant") in {"clone", "untied"} else 1
        expected_allocated = sum(((worker_rows * D + 63) // 64) * 64 * block_count
                                 for worker_rows in rows_by_worker)
        checks = {
            "mac_total": expected_mac,
            "base_weight_bytes": base,
            "one_pass_weight_bytes": expected_one_pass,
            "logical_weight_load_bytes": expected_logical,
            "allocated_weight_bytes": expected_allocated,
            "distinct_weight_storage_bytes": expected_allocated,
        }
        for field, expected in checks.items():
            if int(row.get(field, -1)) != expected:
                errors.append(f"row {index}: {field}={row.get(field)} expected {expected}")
        expected_transition_cells = (D * S * R * sequences * repetitions
                                     if row.get("gate") == "t0rm" else 0)
        if int(row.get("transition_cells", -1)) != expected_transition_cells:
            errors.append(
                f"row {index}: transition_cells={row.get('transition_cells')} "
                f"expected {expected_transition_cells}"
            )
        clipped = int(row.get("clipped_cells", -1))
        if clipped < 0 or clipped > expected_transition_cells:
            errors.append(
                f"row {index}: clipped_cells={clipped} outside [0,{expected_transition_cells}]"
            )
        if elapsed <= 0:
            errors.append(f"row {index}: non-positive elapsed time")
        else:
            rates = {
                "mac_per_second": expected_mac / elapsed,
                "one_pass_weight_gb_per_second": expected_one_pass / elapsed / 1e9,
                "logical_weight_load_gb_per_second": expected_logical / elapsed / 1e9,
            }
            for field, expected in rates.items():
                if not relative_close(float(row.get(field, math.nan)), expected):
                    errors.append(f"row {index}: {field} disagrees with raw counts/time")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(f"row {index}: accounting audit could not parse row ({error})")
    return errors


def structural_checks(rows: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    versions = {str(row.get("project_version", "")).strip() for row in rows
                if str(row.get("project_version", "")).strip()}
    if len(versions) > 1:
        errors.append(f"mixed project versions in one analysis: {sorted(versions)}")
    for index, row in enumerate(rows, 1):
        if row.get("_parse_error"):
            errors.append(f"row {index}: CSV numeric parse failed ({row['_parse_error']})")
            continue
        if row.get("valid") != "true":
            errors.append(f"row {index}: invalid run ({row.get('error', '')})")
        if row.get("all_affinity_succeeded") != "true":
            errors.append(f"row {index}: affinity failed")

        gate = row.get("gate")
        transition = row.get("transition")
        slots = int(row.get("S", 0))
        total_rows = int(row.get("total_rows", 0))
        dimension = int(row.get("D", 0))
        if gate == "t0r" and (slots != 1 or transition != "frozen"):
            errors.append(f"row {index}: T0-R contract requires S=1 and frozen transition")
        if gate == "t0m" and transition != "frozen":
            errors.append(f"row {index}: T0-M contract requires frozen transition")
        if gate == "t0rm" and (transition == "frozen" or total_rows != dimension):
            errors.append(f"row {index}: T0-RM contract requires real transition and total_rows=D")
        if not str(row.get("project_version", "")).strip():
            errors.append(f"row {index}: missing project_version")
        for field in ("elapsed_seconds", "mac_per_second",
                      "one_pass_weight_gb_per_second",
                      "logical_weight_load_gb_per_second", "target_rms", "epsilon"):
            try:
                if not math.isfinite(float(row.get(field, math.nan))):
                    errors.append(f"row {index}: non-finite {field}")
            except (TypeError, ValueError):
                errors.append(f"row {index}: invalid numeric field {field}")

        if row.get("variant") == "clone":
            if row.get("clone_hashes_equal") != "true":
                errors.append(f"row {index}: Bclone hashes are not equal")
            if int(row.get("R", 1)) > 1 and row.get("clone_addresses_distinct") != "true":
                errors.append(f"row {index}: Bclone addresses are not distinct")

        try:
            physical = parse_semicolon_ints(row.get("physical_cores", ""))
            cpus = parse_semicolon_ints(row.get("cpus", ""))
            rows_by_worker = parse_semicolon_ints(row.get("rows", ""))
            worker_count = int(row.get("worker_count", 0))
            if len(physical) != worker_count or len(cpus) != worker_count or len(rows_by_worker) != worker_count:
                errors.append(
                    f"row {index}: worker_count disagrees with CPUs/physical cores/row shards"
                )
            if sum(rows_by_worker) != total_rows:
                errors.append(f"row {index}: rows do not sum to total_rows")
            if row.get("allow_smt_siblings") != "true" and len(physical) != len(set(physical)):
                errors.append(f"row {index}: selected logical CPUs share a physical core")
        except (TypeError, ValueError) as error:
            errors.append(f"row {index}: malformed CPU/core/row list ({error})")
        errors.extend(accounting_errors(row, index))

    paired: dict[tuple[object, ...], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("variant") in {"shared", "clone"}:
            paired[equivalence_key(row, {"variant"})][str(row["variant"])].append(row)
    for key, variants in paired.items():
        if "shared" not in variants or "clone" not in variants:
            errors.append(f"missing shared/Bclone pair for condition: {key}")
            continue
        if len(variants["shared"]) != len(variants["clone"]):
            errors.append(
                f"unbalanced shared/Bclone repetitions for condition: {key} "
                f"({len(variants['shared'])} vs {len(variants['clone'])})"
            )
        for field in ["output_checksum", "state_checksum", "round_sink", "weight_hash_signature",
                      "clipped_cells", "transition_cells"]:
            shared_values = {str(row.get(field)) for row in variants["shared"]}
            clone_values = {str(row.get(field)) for row in variants["clone"]}
            if shared_values != clone_values:
                errors.append(f"shared/Bclone mismatch for {field}: {key}")

    kernel_pairs: dict[tuple[object, ...], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("gate") == "t0m" and row.get("kernel") in {"avx2-repeat", "avx2-fused"}:
            kernel_pairs[equivalence_key(row, {"kernel"})][str(row["kernel"])].append(row)
    for key, kernels in kernel_pairs.items():
        if "avx2-repeat" not in kernels or "avx2-fused" not in kernels:
            errors.append(f"missing repeat/fused T0-M pair for condition: {key}")
            continue
        if len(kernels["avx2-repeat"]) != len(kernels["avx2-fused"]):
            errors.append(
                f"unbalanced repeat/fused repetitions for condition: {key} "
                f"({len(kernels['avx2-repeat'])} vs {len(kernels['avx2-fused'])})"
            )
        for field in ["output_checksum", "state_checksum", "round_sink", "weight_hash_signature",
                      "clipped_cells", "transition_cells"]:
            repeat_values = {str(row.get(field)) for row in kernels["avx2-repeat"]}
            fused_values = {str(row.get(field)) for row in kernels["avx2-fused"]}
            if repeat_values != fused_values:
                errors.append(f"repeat/fused mismatch for {field}: {key}")
    return errors


def t0r_table(rows: list[dict[str, object]]) -> list[str]:
    source = [r for r in rows if r.get("gate") == "t0r" and r.get("variant") in {"shared", "clone"}]
    groups: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    onepass: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in source:
        key = condition_key(row, {"gate"})
        groups[key][str(row["variant"])].append(float(row["mac_per_second"]))
        onepass[key][str(row["variant"])].append(float(row.get("one_pass_weight_gb_per_second", math.nan)))
    lines = ["## T0-R: residencia por profundidad", "",
             "| D | S | R | filas | kernel | A/B mediana | min(A)>max(B) | B one-pass GB/s | lectura |",
             "|---:|---:|---:|---:|---|---:|---|---:|---|"]
    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        values = groups[key]
        if "shared" not in values or "clone" not in values:
            continue
        # key omits gate, but preserves D,S,R,total_rows,kernel,transition,timing_scope,tile,rows
        D, S, R, total_rows, kernel, transition, timing, tile, row_layout, *_ = key
        a, b = values["shared"], values["clone"]
        ratio = median(a) / median(b)
        separated = min(a) > max(b)
        label = "PASS_STRONG" if ratio >= 3 else "PASS" if ratio >= 2 else "WEAK" if ratio >= 1.2 else "NO"
        lines.append(f"| {D} | {S} | {R} | {total_rows} | {kernel} | {ratio:.3f}× | {'sí' if separated else 'no'} | {median(onepass[key]['clone']):.2f} | {label} |")
    lines.append("")
    return lines


def t0m_table(rows: list[dict[str, object]]) -> list[str]:
    source = [r for r in rows if r.get("gate") == "t0m"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in source:
        key = (
            row.get("D"), row.get("R"), row.get("total_rows"), row.get("variant"),
            row.get("transition"), row.get("timing_scope"), row.get("slot_tile"),
            row.get("rows"), row.get("cpus"), row.get("physical_cores"),
            row.get("seed"), row.get("phase_profile"), row.get("require_affinity"),
            row.get("allow_smt_siblings"), row.get("projection_shift"),
            row.get("state_multiplier"), row.get("output_multiplier"),
            row.get("final_shift"), row.get("target_rms"), row.get("epsilon"),
            row.get("warmup_repetitions"), row.get("timed_repetitions"),
            row.get("sequences_per_repetition"),
        )
        groups[key].append(row)
    lines = ["## T0-M: matrixización por slots", "",
             "| D | filas | R | variante | S | G(S)=fused/fused(S=1) | F(S)=fused/repeat | lectura |",
             "|---:|---:|---:|---|---:|---:|---:|---|"]
    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        items = groups[key]
        fused_by_s: dict[int, list[float]] = defaultdict(list)
        repeat_by_s: dict[int, list[float]] = defaultdict(list)
        for row in items:
            target = fused_by_s if row.get("kernel") == "avx2-fused" else repeat_by_s
            target[int(row["S"])].append(float(row["mac_per_second"]))
        if 1 not in fused_by_s:
            continue
        base = median(fused_by_s[1])
        for S in sorted(fused_by_s):
            g = median(fused_by_s[S]) / base
            f = median(fused_by_s[S]) / median(repeat_by_s[S]) if S in repeat_by_s else math.nan
            label = "PASS_STRONG" if g >= 2 else "PASS" if g >= 1.5 else "WEAK" if g >= 1.2 else "NO"
            f_text = f"{f:.3f}×" if math.isfinite(f) else "—"
            lines.append(f"| {key[0]} | {key[2]} | {key[1]} | {key[3]} | {S} | {g:.3f}× | {f_text} | {label} |")
    lines.append("")
    return lines


def t0rm_table(rows: list[dict[str, object]]) -> list[str]:
    source = [r for r in rows if r.get("gate") == "t0rm" and r.get("variant") in {"shared", "clone"}]
    groups: dict[tuple[object, ...], dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in source:
        groups[condition_key(row, {"gate"})][str(row["variant"])].append(row)
    lines = ["## T0-RM: recurrencia real", "",
             "| D | S | R | transición | shift | A/B mediana | clipping mediano | validez numérica | lectura física |",
             "|---:|---:|---:|---|---:|---:|---:|---|---|"]
    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        values = groups[key]
        if "shared" not in values or "clone" not in values:
            continue
        D, S, R, total_rows, kernel, transition, timing, tile, row_layout, *_ = key
        projection_shift = int(values["shared"][0].get("projection_shift", 0))
        shared_rates = [float(row["mac_per_second"]) for row in values["shared"]]
        clone_rates = [float(row["mac_per_second"]) for row in values["clone"]]
        ratio = median(shared_rates) / median(clone_rates)
        clipping_rates: list[float] = []
        for row in values["shared"] + values["clone"]:
            cells = int(row.get("transition_cells", 0))
            clipping_rates.append(int(row.get("clipped_cells", 0)) / cells if cells else 0.0)
        clipping = median(clipping_rates)
        numeric = "OK" if clipping <= 0.01 else "ADVERTENCIA" if clipping <= 0.05 else "NO VÁLIDO NUMÉRICAMENTE"
        if ratio >= 2:
            label = "separación fuerte"
        elif ratio >= 1.2:
            label = "separación"
        elif int(S) > 1:
            label = "paridad compatible con alta reutilización por slots"
        else:
            label = "sin separación material"
        lines.append(
            f"| {D} | {S} | {R} | {transition} | {projection_shift} | {ratio:.3f}× | "
            f"{clipping:.3%} | {numeric} | {label} |"
        )
    lines.append("")
    lines.append("El clipping del microbenchmark de transición no sustituye esta columna: "
                 "la validez del gate recurrente se juzga sobre la trayectoria T0-RM de R rondas.")
    lines.append("")
    return lines


def constant_work_table(rows: list[dict[str, object]]) -> list[str]:
    source = [r for r in rows if r.get("gate") == "t0m" and
              r.get("kernel") == "avx2-fused" and r.get("transition") == "frozen"]
    groups: dict[tuple[object, ...], dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in source:
        work = int(row["S"]) * int(row["R"])
        key = (
            row.get("D"), row.get("total_rows"), row.get("variant"), work,
            row.get("slot_tile"), row.get("rows"), row.get("cpus"),
            row.get("physical_cores"), row.get("seed"), row.get("phase_profile"),
            row.get("require_affinity"), row.get("allow_smt_siblings"),
            row.get("timing_scope"), row.get("warmup_repetitions"),
            row.get("timed_repetitions"), row.get("sequences_per_repetition"),
        )
        groups[key][(int(row["R"]), int(row["S"]))].append(float(row["elapsed_seconds"]))
    lines = ["## T0-M: intercambio profundidad/slots a R×S constante", "",
             "| D | filas | variante | R×S | pares medidos (R,S: mediana ms) | más rápido |",
             "|---:|---:|---|---:|---|---|"]
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        pairs = groups[key]
        if len(pairs) < 2:
            continue
        values = {pair: median(samples) for pair, samples in pairs.items()}
        fastest = min(values, key=values.get)
        description = "; ".join(
            f"{pair[0]},{pair[1]}: {values[pair]*1000:.3f}"
            for pair in sorted(values))
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {description} | R={fastest[0]}, S={fastest[1]} |")
    lines.append("")
    return lines


def recurrent_retention_table(rows: list[dict[str, object]]) -> list[str]:
    static: dict[tuple[object, ...], list[float]] = defaultdict(list)
    recurrent: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            row.get("D"), row.get("S"), row.get("R"), row.get("total_rows"),
            row.get("kernel"), row.get("variant"), row.get("slot_tile"),
            row.get("rows"), row.get("cpus"), row.get("physical_cores"),
            row.get("seed"), row.get("phase_profile"), row.get("require_affinity"),
            row.get("allow_smt_siblings"), row.get("timing_scope"),
            row.get("warmup_repetitions"), row.get("timed_repetitions"),
            row.get("sequences_per_repetition"),
        )
        if row.get("gate") == "t0m" and row.get("transition") == "frozen" and int(row.get("total_rows", 0)) == int(row.get("D", -1)):
            static[key].append(float(row["mac_per_second"]))
        elif row.get("gate") == "t0rm":
            recurrent[key][str(row.get("transition"))].append(float(row["mac_per_second"]))
    lines = ["## T0-RM: throughput retenido frente al puente frozen", "",
             "| D | S | R | variante | transición | recurrente/estático |",
             "|---:|---:|---:|---|---|---:|"]
    for key in sorted(recurrent, key=lambda item: tuple(str(value) for value in item)):
        if key not in static:
            continue
        baseline = median(static[key])
        for transition, samples in sorted(recurrent[key].items()):
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[5]} | {transition} | {median(samples)/baseline:.3f}× |")
    lines.append("")
    return lines


def variability_warnings(rows: list[dict[str, object]]) -> list[str]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = condition_key(row) + (row.get("variant"),)
        groups[key].append(float(row["mac_per_second"]))
    warnings: list[str] = []
    for key, values in groups.items():
        if len(values) < 3:
            continue
        mean = statistics.fmean(values)
        if mean <= 0:
            continue
        cv = statistics.pstdev(values) / mean
        if cv > 0.10:
            warnings.append(f"CV={cv:.1%}: {key}")
    lines = ["## Variabilidad externa", ""]
    if warnings:
        lines.append(f"**ALERTA:** {len(warnings)} condición(es) superan 10% de CV.")
        lines.extend(f"- {warning}" for warning in warnings[:50])
    else:
        lines.append("No se detectaron condiciones con al menos tres muestras y CV superior a 10%.")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-structure", action="store_true")
    args = parser.parse_args()
    expanded: list[Path] = []
    for path in args.paths:
        if path.is_dir(): expanded.extend(sorted(path.glob("*.csv")))
        else: expanded.append(path)
    rows = load_rows(expanded)
    if not rows:
        print("No compatible gate CSV rows found", file=sys.stderr)
        return 2
    errors = structural_checks(rows)
    report = ["# CNRL gate analysis", "",
              f"Rows loaded: **{len(rows)}** from **{len(expanded)}** file(s).", "",
              "## Structural audit", ""]
    if errors:
        report.append(f"**REJECTED:** {len(errors)} structural inconsistency(ies).")
        report.extend(f"- {error}" for error in errors[:100])
    else:
        report.append("**PASS:** no invalid rows, affinity failures, Bclone invariant failures, or shared/Bclone checksum divergences.")
    report += [""] + t0r_table(rows) + t0m_table(rows) + constant_work_table(rows)
    report += t0rm_table(rows) + recurrent_retention_table(rows) + variability_warnings(rows)
    report += ["## Metric warning", "",
               "`mac_per_second / S` equals the one-pass int8 weight-stream rate only for a square/rectangular dot-product that loads each weight once per slot group. The authoritative CSV field is `one_pass_weight_gb_per_second`; do not equate raw GMAC/s with DRAM GB/s when S>1.", ""]
    text = "\n".join(report)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if errors and args.strict_structure else 0

if __name__ == "__main__":
    raise SystemExit(main())
