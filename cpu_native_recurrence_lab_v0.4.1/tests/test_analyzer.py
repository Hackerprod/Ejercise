#!/usr/bin/env python3
"""Unit tests for strict CSV accounting and pairing rules."""
from __future__ import annotations

import importlib.util
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cnrl_analyzer", ROOT / "scripts" / "analyze_results.py"
)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def row(variant: str, kernel: str) -> dict[str, object]:
    D, S, R, total_rows, repetitions, sequences, tile = 64, 4, 2, 64, 3, 2, 4
    base = D * total_rows
    mac = base * S * R * repetitions * sequences
    one_pass = base * R * repetitions * sequences
    logical_passes = S if kernel == "avx2-repeat" else analyzer.fused_passes(S, tile)
    logical = base * logical_passes * R * repetitions * sequences
    block_count = R if variant == "clone" else 1
    allocated = 2 * (((32 * D + 63) // 64) * 64 * block_count)
    elapsed = 1.0
    return {
        "project_version": "0.4.1",
        "gate": "t0m",
        "D": D,
        "S": S,
        "R": R,
        "total_rows": total_rows,
        "kernel": kernel,
        "variant": variant,
        "transition": "frozen",
        "timing_scope": "full-repetition",
        "slot_tile": tile,
        "seed": 123,
        "phase_profile": "false",
        "require_affinity": "true",
        "projection_shift": 12,
        "state_multiplier": 1,
        "output_multiplier": 1,
        "final_shift": 0,
        "target_rms": 32.0,
        "epsilon": 1.0e-6,
        "warmup_repetitions": 2,
        "timed_repetitions": repetitions,
        "sequences_per_repetition": sequences,
        "worker_count": 2,
        "allow_smt_siblings": "false",
        "cpus": "0;1",
        "physical_cores": "0;1",
        "rows": "32;32",
        "base_weight_bytes": base,
        "allocated_weight_bytes": allocated,
        "logical_weight_load_bytes": logical,
        "one_pass_weight_bytes": one_pass,
        "distinct_weight_storage_bytes": allocated,
        "elapsed_seconds": elapsed,
        "mac_total": mac,
        "mac_per_second": mac / elapsed,
        "logical_weight_load_gb_per_second": logical / elapsed / 1.0e9,
        "one_pass_weight_gb_per_second": one_pass / elapsed / 1.0e9,
        "output_checksum": 111,
        "state_checksum": 222,
        "round_sink": 333,
        "weight_hash_signature": 444,
        "clone_hashes_equal": "true" if variant == "clone" else "false",
        "clone_addresses_distinct": "true" if variant == "clone" else "false",
        "all_affinity_succeeded": "true",
        "clipped_cells": 0,
        "transition_cells": 0,
        "valid": "true",
        "error": "",
        "batch_repeat": 1,
        "variant_order": 1,
    }


def main() -> int:
    rows = [
        row("shared", "avx2-repeat"),
        row("shared", "avx2-fused"),
        row("clone", "avx2-repeat"),
        row("clone", "avx2-fused"),
    ]
    errors = analyzer.structural_checks(rows)
    if errors:
        raise AssertionError(f"valid fixture rejected: {errors}")

    tampered = deepcopy(rows)
    tampered[0]["mac_total"] = int(tampered[0]["mac_total"]) * 8
    errors = analyzer.structural_checks(tampered)
    if not any("mac_total" in error for error in errors):
        raise AssertionError("inflated MAC count was not rejected")

    incomplete = [entry for entry in rows if entry["variant"] == "shared"]
    errors = analyzer.structural_checks(incomplete)
    if not any("missing shared/Bclone pair" in error for error in errors):
        raise AssertionError("missing Bclone pair was not rejected")

    divergent = deepcopy(rows)
    divergent[-1]["output_checksum"] = 999
    errors = analyzer.structural_checks(divergent)
    if not any("shared/Bclone mismatch for output_checksum" in error for error in errors):
        raise AssertionError("Bclone checksum divergence was not rejected")

    kernel_divergent = deepcopy(rows)
    kernel_divergent[1]["round_sink"] = 999
    errors = analyzer.structural_checks(kernel_divergent)
    if not any("repeat/fused mismatch for round_sink" in error for error in errors):
        raise AssertionError("repeat/fused divergence was not rejected")


    unbalanced = deepcopy(rows)
    unbalanced.append(deepcopy(rows[0]))
    errors = analyzer.structural_checks(unbalanced)
    if not any("unbalanced shared/Bclone repetitions" in error for error in errors):
        raise AssertionError("unequal A/B repeat counts were not rejected")

    kernel_unbalanced = deepcopy(rows)
    kernel_unbalanced.append(deepcopy(rows[0]))
    errors = analyzer.structural_checks(kernel_unbalanced)
    if not any("unbalanced repeat/fused repetitions" in error for error in errors):
        raise AssertionError("unequal repeat/fused counts were not rejected")

    transition_tampered = deepcopy(rows)
    transition_tampered[0]["transition_cells"] = 1
    errors = analyzer.structural_checks(transition_tampered)
    if not any("transition_cells" in error for error in errors):
        raise AssertionError("invalid transition cell count was not rejected")

    malformed = deepcopy(rows)
    malformed[0]["rows"] = "32;not-an-integer"
    errors = analyzer.structural_checks(malformed)
    if not any("malformed CPU/core/row list" in error or "could not parse" in error
               for error in errors):
        raise AssertionError("malformed shard list was not rejected cleanly")

    mixed_versions = deepcopy(rows)
    mixed_versions[-1]["project_version"] = "9.9.9"
    errors = analyzer.structural_checks(mixed_versions)
    if not any("mixed project versions" in error for error in errors):
        raise AssertionError("mixed project versions were not rejected")

    clipping_divergent = deepcopy(rows)
    clipping_divergent[-1]["clipped_cells"] = 1
    errors = analyzer.structural_checks(clipping_divergent)
    if not any("shared/Bclone mismatch for clipped_cells" in error for error in errors):
        raise AssertionError("Bclone clipping divergence was not rejected")

    parse_failed = deepcopy(rows)
    parse_failed[0]["_parse_error"] = "invalid literal"
    errors = analyzer.structural_checks(parse_failed)
    if not any("CSV numeric parse failed" in error for error in errors):
        raise AssertionError("CSV conversion failure was not reported structurally")

    print("PASS analyzer accounting, pairing, malformed-input, and tamper rejection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
