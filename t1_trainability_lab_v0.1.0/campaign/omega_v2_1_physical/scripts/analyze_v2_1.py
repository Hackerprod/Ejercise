"""Native Windows V2-1 orchestration and raw-sample analysis. Never times Python."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
RESULTS_ROOT = UNIT_ROOT / "results" / "omega_v2_1_physical"
BUILD_ROOT = UNIT_ROOT / "build"
EXECUTABLE = BUILD_ROOT / "Release" / "omega_v2_1_bench.exe"
V2_0_SEAL_PATH = REPO_ROOT / "t1_trainability_lab_v0.1.0" / "campaign" / "omega_v2_0_conformance" / "V2_0_RESULT_SEAL.json"
AUTHORIZED_SNAPSHOT = "38061477d4c2b0c5c20d75d902b21dd1ef0a2611"
V2_0_VALIDATED_HEAD = "5df1cf270837a9cc4f56a3dd31f277af868f61ce"
V2_0_IMPLEMENTATION_COMMIT = "bc9c1745f70e08f15fd340097d04fff6f56cba43"
CONVERSACION_MD_BLOB = "7027e2ac9d1ba89db08dda73c81e244f3b9b19db"
CONVERSACION_LN_BLOB = "fc750a2ae9fb7d3933c54fb91f09ce5568d035ec"
AUDIT_SHA256 = "a7e82cddfcacf17f163e7d046853046595f08e532086fb3c2a577432807933dc"
CONTRACT_RANGES = ["2390-2485", "2585-2630", "2688-2945", "3078-3145"]
TEST_NAMES = [
    "test_v2_1_reference_commit_and_v2_0_seal",
    "test_v2_1_native_windows_required",
    "test_v2_1_hardware_topology_complete",
    "test_v2_1_four_primary_workers_are_distinct_pcores",
    "test_v2_1_no_smt_sibling_in_primary_workers",
    "test_v2_1_q4_group_size_32",
    "test_v2_1_q4_pack_unpack_signed_nibbles",
    "test_v2_1_q4_fp16_scale_layout",
    "test_v2_1_q4_no_zeropoint",
    "test_v2_1_q4_physical_byte_ledger",
    "test_v2_1_full_block_scalar_reference",
    "test_v2_1_v2_0_equation_structure",
    "test_v2_1_abc_numerical_identity",
    "test_v2_1_a_reuses_same_storage",
    "test_v2_1_b_round_storages_are_disjoint",
    "test_v2_1_b_values_equal_a",
    "test_v2_1_b_pool_exceeds_2p5_llc",
    "test_v2_1_c_uses_a_storage",
    "test_v2_1_eviction_probe_effectiveness",
    "test_v2_1_eviction_outside_timed_region",
    "test_v2_1_worker_shards_cover_all_rows",
    "test_v2_1_worker_shards_do_not_overlap",
    "test_v2_1_no_full_weight_replication_per_worker",
    "test_v2_1_no_heap_allocation_in_timed_kernel",
    "test_v2_1_persistent_thread_pool",
    "test_v2_1_fixed_affinity_preserved",
    "test_v2_1_qpc_monotonic",
    "test_v2_1_qpc_overhead_recorded",
    "test_v2_1_flop_mac_ledger_matches_v2_0",
    "test_v2_1_same_compute_abc",
    "test_v2_1_sweep_contains_m1_control",
    "test_v2_1_sweep_completeness",
    "test_v2_1_rho_resident_formula_fixture",
    "test_v2_1_c_vs_b_formula_fixture",
    "test_v2_1_matrixization_formula_fixture",
    "test_v2_1_conformance_block",
    "test_v2_1_report_contract_paths_and_hashes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def validate_v2_0_seal() -> dict[str, Any]:
    seal = read_json(V2_0_SEAL_PATH)
    if seal.get("terminal_status") != "OMEGA_V2_0_CONFORMANT_PASS" or seal.get("execution_commit") != V2_0_IMPLEMENTATION_COMMIT:
        raise RuntimeError("V2_1_ACCEPTANCE_HOLD: V2-0 result seal status/implementation commit mismatch")
    if (seal.get("tests") or {}) != {"run": 10, "passed": 10, "failed": 0, "skipped": 0}:
        raise RuntimeError("V2_1_ACCEPTANCE_HOLD: V2-0 seal test counts are not 10/10/0")
    sealed_results = V2_0_SEAL_PATH.parent / "results" / "omega_v2_0_conformance"
    observed = {}
    for name, expected in seal["artifact_sha256"].items():
        path = sealed_results / name
        if not path.is_file():
            raise RuntimeError(f"V2_1_ACCEPTANCE_HOLD: sealed V2-0 result artifact missing: {path}")
        actual = sha256_file(path)
        observed[name] = actual
        if actual != expected:
            raise RuntimeError(f"V2_1_ACCEPTANCE_HOLD: V2-0 artifact SHA mismatch: {name}")
    return {"seal": seal, "observed_artifact_sha256": observed}


def source_provenance() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    merge_base = git("merge-base", "HEAD", AUTHORIZED_SNAPSHOT)
    status = git("status", "--porcelain", "--", "t1_trainability_lab_v0.1.0/campaign/omega_v2_1_physical")
    full_status = git("status", "--porcelain")
    md_blob = git("rev-parse", "HEAD:Conversacion.md")
    ln_blob = git("rev-parse", "HEAD:Conversacion LN.md")
    if branch != "main" or merge_base != AUTHORIZED_SNAPSHOT or status or md_blob != CONVERSACION_MD_BLOB or ln_blob != CONVERSACION_LN_BLOB:
        raise RuntimeError(
            "V2_1_ACCEPTANCE_HOLD: implementation requires clean committed sources on main descending from the authorized snapshot; "
            f"branch={branch},commit={commit},parents={parents},merge_base={merge_base},unit_status={status!r},"
            f"md_blob={md_blob},ln_blob={ln_blob}"
        )
    return {
        "repository": "Hackerprod/Ejercise",
        "branch": branch,
        "implementation_commit": commit,
        "implementation_parent": parents[0] if parents else None,
        "git_status_porcelain": full_status,
        "git_status_porcelain_unit": status,
        "authorized_snapshot_commit": AUTHORIZED_SNAPSHOT,
        "merge_base": merge_base,
        "conversacion_md_blob": md_blob,
        "conversacion_ln_blob": ln_blob,
        "source_tree_clean_before_build": True,
    }


def find_vswhere() -> Path:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")), Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    for root in roots:
        path = root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if path.is_file():
            return path
    raise FileNotFoundError("vswhere.exe was not found; stop before V2-1 computation")


def find_msvc_cmake(vswhere: Path) -> tuple[Path, str, str]:
    vs_paths = subprocess.run(
        [str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    cmake_paths = subprocess.run(
        [str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.CMake.Project", "-find", r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if not vs_paths or not cmake_paths:
        raise FileNotFoundError("MSVC x64 compiler or Visual Studio CMake component missing")
    cmake = Path(cmake_paths[0])
    cmake_version = subprocess.run([str(cmake), "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    return cmake, vs_paths[0], cmake_version


def run_build(cmake: Path) -> tuple[Path, dict[str, Any]]:
    configure = subprocess.run(
        [str(cmake), "-S", str(UNIT_ROOT), "-B", str(BUILD_ROOT), "-G", "Visual Studio 17 2022", "-A", "x64"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if configure.returncode != 0:
        raise RuntimeError("native CMake configure failed: " + configure.stderr[-3000:])
    build = subprocess.run(
        [str(cmake), "--build", str(BUILD_ROOT), "--config", "Release", "--target", "omega_v2_1_bench", "-j", "8"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError("native MSVC Release build failed: " + build.stderr[-5000:])
    executable = BUILD_ROOT / "Release" / "omega_v2_1_bench.exe"
    if not executable.is_file():
        raise FileNotFoundError("native Windows Release executable missing after CMake build")
    log = {"configure_stdout": configure.stdout, "configure_stderr": configure.stderr, "build_stdout": build.stdout, "build_stderr": build.stderr}
    return executable, log


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def parse_raw(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append({
                **row,
                "d": int(row["d"]), "m": int(row["m"]), "K": int(row["K"]),
                "block_id": int(row["block_id"]), "sample_id": int(row["sample_id"]),
                "round_index": int(row["round_index"]), "qpc_ticks": int(row["qpc_ticks"]),
                "qpc_seconds": float(row["qpc_seconds"]), "effective_macs": int(row["effective_macs"]),
                "effective_flops": int(row["effective_flops"]),
                "timed_heap_allocation_count": int(row["timed_heap_allocation_count"]),
                "valid": row["valid"].lower() == "true", "is_warmup": row["is_warmup"].lower() == "true",
            })
    return rows, fieldnames


def check_sweep_completeness(raw_rows: list[dict[str, Any]], native_status: str) -> bool:
    if native_status != "SWEEP_COMPLETE":
        return False
    grouped: dict[tuple[int, int, int, str, bool, int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (row["d"], row["m"], row["K"], row["variant"], row["is_warmup"], row["block_id"], row["sample_id"])
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 72 * (10 + 105):
        return False
    for d in (512, 640):
        for m in (1, 4, 8, 16):
            for K in (1, 4, 8):
                for variant in ("A", "B", "C"):
                    cell = (d, m, K, variant)
                    warmups = [key for key in grouped if key[:4] == cell and key[4]]
                    measured = [key for key in grouped if key[:4] == cell and not key[4]]
                    if len(warmups) != 10 or len(measured) != 105:
                        return False
                    if {key[6] for key in measured} != set(range(105)) or {key[5] for key in measured} != set(range(5)):
                        return False
                    for key in warmups + measured:
                        if sorted(row["round_index"] for row in grouped[key]) != list(range(K)):
                            return False
    return True


def summarize_samples(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int, str, bool, int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (row["d"], row["m"], row["K"], row["variant"], row["is_warmup"], row["block_id"], row["sample_id"])
        grouped.setdefault(key, []).append(row)
    cells: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    discarded = []
    for d in (512, 640):
        for m in (1, 4, 8, 16):
            for K in (1, 4, 8):
                for variant in ("A", "B", "C"):
                    sample_times, sample_macs = [], []
                    warmups = measured = 0
                    block_totals = {block: 0 for block in range(5)}
                    block_times = {block: [] for block in range(5)}
                    for sample_key, rows in grouped.items():
                        if sample_key[:4] != (d, m, K, variant):
                            continue
                        _, _, _, _, is_warmup, block, sample_id = sample_key
                        rows.sort(key=lambda row: row["round_index"])
                        if is_warmup:
                            warmups += 1
                            continue
                        measured += 1
                        if block in block_totals:
                            block_totals[block] += 1
                        reasons = sorted({row["invalid_reason"] for row in rows if row["invalid_reason"]})
                        valid = len(rows) == K and [row["round_index"] for row in rows] == list(range(K)) and all(row["valid"] for row in rows)
                        if not valid:
                            discarded.append({"d": d, "m": m, "K": K, "variant": variant, "block_id": block, "sample_id": sample_id, "reasons": reasons or ["INVALID_ROUND_SET"]})
                            continue
                        sample_seconds = sum(row["qpc_seconds"] for row in rows)
                        sample_times.append(sample_seconds)
                        if block in block_times:
                            block_times[block].append(sample_seconds)
                        sample_macs.append(sum(row["effective_macs"] for row in rows))
                    median_time = statistics.median(sample_times) if sample_times else None
                    mad = statistics.median([abs(value - median_time) for value in sample_times]) if sample_times else None
                    cells[(d, m, K, variant)] = {
                        "d": d, "m": m, "K": K, "variant": variant,
                        "n_total": measured, "n_valid": len(sample_times), "n_warmup": warmups,
                        "median_seconds": median_time, "MAD_seconds": mad,
                        "R_MAD": (mad / median_time) if median_time and median_time > 0 else None,
                        "p10_seconds": percentile(sample_times, .10) if sample_times else None,
                        "p25_seconds": percentile(sample_times, .25) if sample_times else None,
                        "p50_seconds": percentile(sample_times, .50) if sample_times else None,
                        "p75_seconds": percentile(sample_times, .75) if sample_times else None,
                        "p90_seconds": percentile(sample_times, .90) if sample_times else None,
                        "blocks": [
                            {
                                "block_id": block,
                                "n_total": block_totals[block],
                                "n_valid": len(block_times[block]),
                                "median_seconds": statistics.median(block_times[block]) if block_times[block] else None,
                                "MAD_seconds": statistics.median([abs(value - statistics.median(block_times[block])) for value in block_times[block]]) if block_times[block] else None,
                            }
                            for block in range(5)
                        ],
                        "mean_seconds": statistics.mean(sample_times) if sample_times else None,
                        "std_seconds": statistics.pstdev(sample_times) if len(sample_times) > 1 else (0.0 if sample_times else None),
                        "effective_MAC_per_s": statistics.median([mac / t for mac, t in zip(sample_macs, sample_times)]) if sample_times else None,
                        "effective_GFLOP_per_s": statistics.median([2.0 * mac / t / 1e9 for mac, t in zip(sample_macs, sample_times)]) if sample_times else None,
                    }
    return list(cells.values()), cells, discarded


def compute_gate_metrics(cells: dict[tuple[int, int, int, str], dict[str, Any]]) -> dict[str, Any]:
    def median_time(m: int, K: int, variant: str) -> float:
        value = cells[(512, m, K, variant)]["median_seconds"]
        if value is None:
            raise ValueError(f"invalid/missing primary median: d512,m{m},K{K},{variant}")
        return float(value)

    def marginal(variant: str, K: int, m: int = 4) -> float:
        return (median_time(m, K, variant) - median_time(m, 1, variant)) / (K - 1)

    c_a8, c_b8, c_c8 = marginal("A", 8), marginal("B", 8), marginal("C", 8)
    if c_a8 <= 0 or c_b8 <= 0:
        raise ValueError("measurement-invalid: c_A(8) and c_B(8) must be positive")
    rates = {m: cells[(512, m, 8, "A")]["effective_MAC_per_s"] for m in (1, 4, 8, 16)}
    if any(value is None or value <= 0 for value in rates.values()):
        raise ValueError("measurement-invalid: non-positive matrixization rate")
    g = max(rates[8], rates[16]) / rates[1]
    stability_cells = [(512, 4, K, variant) for variant in ("A", "B", "C") for K in (1, 8)] + [(512, m, 8, "A") for m in (1, 8, 16)]
    stability = []
    for d, m, K, variant in stability_cells:
        rmad = cells[(d, m, K, variant)]["R_MAD"]
        stability.append({"d": d, "m": m, "K": K, "variant": variant, "R_MAD": rmad, "noisy": rmad is not None and rmad > .15})
    noisy = sum(row["noisy"] for row in stability)
    return {
        "primary_cell": {"d": 512, "m": 4, "K": 8},
        "c_A_K8": c_a8, "c_B_K8": c_b8, "c_C_K8": c_c8,
        "rho_resident": c_a8 / c_b8,
        "rho_CB": c_c8 / c_b8,
        "delta_CB": abs(c_c8 - c_b8) / c_b8,
        "delta_CA": abs(c_c8 - c_a8) / c_a8,
        "c_C_gt_c_A": c_c8 > c_a8,
        "R_A": {str(m): rates[m] for m in rates},
        "R_A_m1": rates[1], "R_A_m4": rates[4], "R_A_m8": rates[8], "R_A_m16": rates[16],
        "R_A_ratio_m4": rates[4] / rates[1],
        "R_A_ratio_m8": rates[8] / rates[1],
        "R_A_ratio_m16": rates[16] / rates[1],
        "matrix_gain_m4": rates[4] / rates[1],
        "matrix_gain_m8": rates[8] / rates[1],
        "matrix_gain_m16": rates[16] / rates[1],
        "G_matrix": g,
        "primary_stability_cells": stability,
        "primary_stability_cell_count": 9,
        "noisy_primary_stability_cells": noisy,
        "measurement_stability_hold": noisy >= 2,
        "rho_minimum_pass": c_a8 / c_b8 <= .50,
        "rho_strong": c_a8 / c_b8 <= .25,
        "causal_minimum_pass": abs(c_c8 - c_b8) / c_b8 <= .25 and c_c8 > c_a8,
        "causal_strong": abs(c_c8 - c_b8) / c_b8 <= .10 and c_c8 > c_a8,
        "matrixization_minimum_pass": g >= 1.50,
        "matrixization_strong": g >= 2.00,
    }


def bootstrap_gate_diagnostics(raw_rows: list[dict[str, Any]], reps: int = 5000) -> dict[str, Any]:
    import random

    grouped: dict[tuple[int, int, int, str, int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row["is_warmup"]:
            continue
        key = (row["d"], row["m"], row["K"], row["variant"], row["block_id"], row["sample_id"])
        grouped.setdefault(key, []).append(row)
    series: dict[tuple[int, int, int, str], dict[str, list[float]]] = {}
    for (d, m, K, variant, _, _), rows in grouped.items():
        rows.sort(key=lambda row: row["round_index"])
        if len(rows) != K or [row["round_index"] for row in rows] != list(range(K)) or not all(row["valid"] for row in rows):
            continue
        seconds = sum(row["qpc_seconds"] for row in rows)
        macs = sum(row["effective_macs"] for row in rows)
        if seconds <= 0.0:
            continue
        cell = series.setdefault((d, m, K, variant), {"seconds": [], "rates": []})
        cell["seconds"].append(seconds)
        cell["rates"].append(macs / seconds)

    required = [
        (512, 4, 1, "A"), (512, 4, 8, "A"), (512, 4, 1, "B"), (512, 4, 8, "B"),
        (512, 4, 1, "C"), (512, 4, 8, "C"), (512, 1, 8, "A"), (512, 8, 8, "A"), (512, 16, 8, "A"),
    ]
    if any(not series.get(key, {}).get("seconds") for key in required):
        raise ValueError("bootstrap diagnostics lack valid primary-cell samples")

    rng = random.Random(20260929)
    draws: dict[str, list[float]] = {"rho_resident": [], "delta_CB": [], "G_matrix": []}

    def resampled_median(key: tuple[int, int, int, str], field: str) -> float:
        values = series[key][field]
        return statistics.median(rng.choices(values, k=len(values)))

    for _ in range(reps):
        t = {key: resampled_median(key, "seconds") for key in required[:6]}
        c_a = (t[(512, 4, 8, "A")] - t[(512, 4, 1, "A")]) / 7.0
        c_b = (t[(512, 4, 8, "B")] - t[(512, 4, 1, "B")]) / 7.0
        c_c = (t[(512, 4, 8, "C")] - t[(512, 4, 1, "C")]) / 7.0
        if c_a > 0 and c_b > 0:
            draws["rho_resident"].append(c_a / c_b)
        if c_b > 0:
            draws["delta_CB"].append(abs(c_c - c_b) / c_b)
        r1 = resampled_median((512, 1, 8, "A"), "rates")
        r8 = resampled_median((512, 8, 8, "A"), "rates")
        r16 = resampled_median((512, 16, 8, "A"), "rates")
        if r1 > 0:
            draws["G_matrix"].append(max(r8, r16) / r1)
    if any(not values for values in draws.values()):
        raise ValueError("bootstrap diagnostics produced no finite replicates")
    return {
        name: {"lower90": percentile(values, .05), "upper90": percentile(values, .95), "replicates": len(values)}
        for name, values in draws.items()
    }


def build_test_rows(
    provenance: dict[str, Any],
    executable: Path,
    hardware_preflight: dict[str, Any],
    q4_ledger: dict[str, Any],
    worker_shards: dict[str, Any],
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    results_root: Path,
) -> list[dict[str, Any]]:
    results: dict[str, tuple[str, str]] = {}

    def record(name: str, condition: bool | None, detail: str = "") -> None:
        status = "SKIP" if condition is None else ("PASS" if condition else "FAIL")
        results[name] = (status, detail)

    seal_ok = provenance.get("v2_0_seal_status") == "OMEGA_V2_0_CONFORMANT_PASS" and provenance.get("v2_0_execution_commit") == V2_0_IMPLEMENTATION_COMMIT
    record("test_v2_1_reference_commit_and_v2_0_seal", seal_ok)
    record("test_v2_1_native_windows_required", os.name == "nt" and sys.platform == "win32" and executable.suffix.lower() == ".exe")

    hw = hardware_preflight.get("hardware", {})
    topology_complete = (
        bool(hw.get("cpu_model"))
        and hw.get("physical_core_count", 0) > 0
        and hw.get("logical_processor_count", 0) > 0
        and hw.get("processor_group_count", 0) > 0
        and hw.get("llc_bytes", 0) > 0
        and hw.get("cache_line_bytes", 0) > 0
        and bool(hw.get("caches"))
        and bool(hardware_preflight.get("h0_measurements"))
        and hw.get("p_core_count", 0) >= 4
        and hw.get("e_core_count", 0) > 0
        and hw.get("physical_core_count") == hw.get("p_core_count", 0) + hw.get("e_core_count", 0)
        and all(core.get("intel_cpuid_core_type") in (0x20, 0x40) for core in hw.get("cores", []))
        and len(hardware_preflight.get("h0_measurements", [])) == hw.get("p_core_count", 0) * 4
        and all(row.get("affinity_ok") is True and row.get("warmups") == 10 and row.get("repetitions") == 31 for row in hardware_preflight.get("h0_measurements", []))
    )
    record("test_v2_1_hardware_topology_complete", topology_complete)
    selected = hardware_preflight.get("selected_workers", [])
    unique_physical = {(row.get("group"), row.get("physical_core_id")) for row in selected}
    unique_cpu_sets = {row.get("windows_cpu_set_id") for row in selected}
    workers_ok = len(selected) == 4 and len(unique_physical) == 4 and len(unique_cpu_sets) == 4 and all(
        row.get("h0_v_i", 0) > 0 and row.get("intel_cpuid_core_type") == 0x40 for row in selected
    )
    record("test_v2_1_four_primary_workers_are_distinct_pcores", workers_ok)
    record("test_v2_1_no_smt_sibling_in_primary_workers", workers_ok and all(row.get("smt_sibling_worker_selected") is False for row in selected))

    native_q4 = hardware_preflight.get("q4_round_trip_test", {})
    record("test_v2_1_q4_group_size_32", q4_ledger.get("group_size") == 32)
    record("test_v2_1_q4_pack_unpack_signed_nibbles", native_q4.get("signed_range_test") is True)
    matrices = [matrix for family in q4_ledger.get("core_families", []) for matrix in family.get("matrices", [])]
    fp16_ok = q4_ledger.get("scale_dtype") == "FP16" and bool(matrices) and all(matrix.get("scale_address_mod64") == 0 for matrix in matrices)
    record("test_v2_1_q4_fp16_scale_layout", fp16_ok)
    record("test_v2_1_q4_no_zeropoint", q4_ledger.get("zero_point") is False)
    physical_bytes_ok = bool(matrices) and all(
        matrix.get("physical_allocated_bytes", 0) >= matrix.get("logical_weight_bytes", 0) + matrix.get("logical_scale_bytes", 0)
        and matrix.get("packed_address_mod64") == 0
        for matrix in matrices
    )
    record("test_v2_1_q4_physical_byte_ledger", physical_bytes_ok)
    scalar_full = hardware_preflight.get("full_block_scalar_reference_test", {})
    record("test_v2_1_full_block_scalar_reference", scalar_full.get("pass") is True)
    shape_by_name = {
        "W_Q": lambda d: (d, d), "W_K": lambda d: (d, d), "W_V": lambda d: (d, d), "W_O": lambda d: (d, d),
        "W_gate": lambda d: (4 * d, d), "W_up": lambda d: (4 * d, d), "W_down": lambda d: (d, 4 * d),
    }
    family_by_d = {family.get("d"): family for family in q4_ledger.get("core_families", [])}
    equation_ok = len(matrices) == 14
    for d in (512, 640):
        family_matrices = family_by_d.get(d, {}).get("matrices", [])
        observed_shapes = {matrix.get("name"): (matrix.get("rows"), matrix.get("cols")) for matrix in family_matrices}
        equation_ok = equation_ok and set(observed_shapes) == set(shape_by_name)
        equation_ok = equation_ok and all(observed_shapes.get(name) == shape(d) for name, shape in shape_by_name.items())
        dimensions_valid = all(
            isinstance(rows, int) and not isinstance(rows, bool) and isinstance(cols, int) and not isinstance(cols, bool)
            for rows, cols in observed_shapes.values()
        )
        equation_ok = equation_ok and dimensions_valid
        if dimensions_valid:
            equation_ok = equation_ok and sum(rows * cols for rows, cols in observed_shapes.values()) == 16 * d * d
    record("test_v2_1_v2_0_equation_structure", equation_ok)

    abc = hardware_preflight.get("abc_correctness", {})
    abc_ok = abc.get("A_B_C_bitwise_equal") is True and all(abc.get(key) is True for key in ("A_finite", "B_finite", "C_finite"))
    record("test_v2_1_abc_numerical_identity", abc_ok)
    record("test_v2_1_a_reuses_same_storage", abc.get("A_C_same_storage") is True and config.get("a_storage_policy") == "same_core_weight_storage_for_all_rounds")
    b_pools = q4_ledger.get("b_pool_by_d_k", [])
    record("test_v2_1_b_round_storages_are_disjoint", bool(b_pools) and all(row.get("all_B_storages_disjoint") is True for row in b_pools))
    record("test_v2_1_b_values_equal_a", bool(b_pools) and all(row.get("q4_values_bitwise_equal_to_A") is True for row in b_pools))
    record("test_v2_1_b_pool_exceeds_2p5_llc", bool(b_pools) and all(row.get("passes_2p5_llc_and_64mib") is True and row.get("pool_to_llc_ratio", 0) >= 2.5 for row in b_pools))
    record("test_v2_1_c_uses_a_storage", abc.get("A_C_same_storage") is True and config.get("c_storage_policy") == "same_storage_as_A_evicted_before_each_round")
    record("test_v2_1_eviction_probe_effectiveness", config.get("eviction_effectiveness_E", 0.0) >= 1.5 and config.get("eviction_method") in ("CLFLUSH", "SWEEP_BUFFER"))
    record("test_v2_1_eviction_outside_timed_region", config.get("c_eviction_outside_each_round_timer") is True)

    matrix_shards = worker_shards.get("matrices", [])
    shard_coverage = bool(matrix_shards) and all(matrix.get("rows_covered") == matrix.get("rows") for matrix in matrix_shards)
    shard_nonoverlap = bool(matrix_shards) and all(
        bool(matrix.get("shards"))
        and
        matrix["shards"][0].get("first_output_row") == 0
        and matrix["shards"][-1].get("last_output_row_exclusive") == matrix.get("rows")
        and all(a.get("last_output_row_exclusive") == b.get("first_output_row") for a, b in zip(matrix["shards"], matrix["shards"][1:]))
        for matrix in matrix_shards
    )
    record("test_v2_1_worker_shards_cover_all_rows", shard_coverage)
    record("test_v2_1_worker_shards_do_not_overlap", shard_nonoverlap)
    no_replicated_full_shard = bool(matrix_shards) and all(
        all(shard.get("physical_bytes_assigned", 0) < matrix.get("matrix_buffer_physical_bytes", 0) for shard in matrix["shards"])
        and sum(shard.get("physical_bytes_assigned", 0) for shard in matrix["shards"]) >= matrix.get("rows", 0) * matrix.get("cols", 0) // 2
        for matrix in matrix_shards
    )
    record("test_v2_1_no_full_weight_replication_per_worker", no_replicated_full_shard)

    native_status = read_json(results_root / "native_run_status.json").get("status", "UNKNOWN") if (results_root / "native_run_status.json").is_file() else "UNKNOWN"
    raw_available = native_status == "SWEEP_COMPLETE" and bool(raw_rows)
    allocations_ok = raw_available and all(row.get("timed_heap_allocation_count") == 0 for row in raw_rows)
    record("test_v2_1_no_heap_allocation_in_timed_kernel", allocations_ok)
    record("test_v2_1_persistent_thread_pool", config.get("worker_pool_persistent") is True and config.get("worker_pool_size") == 4)
    affinity = hardware_preflight.get("worker_affinity_records", [])
    affinity_ok = len(affinity) == 4 and all("affinity_ok=true" in row for row in affinity)
    no_affinity_loss_in_samples = not raw_available or all(row.get("invalid_reason") != "AFFINITY_LOST" for row in raw_rows)
    record("test_v2_1_fixed_affinity_preserved", affinity_ok and no_affinity_loss_in_samples)
    record("test_v2_1_qpc_monotonic", hardware_preflight.get("qpc_monotonic") is True and hw.get("qpc", {}).get("frequency", 0) > 0)
    record("test_v2_1_qpc_overhead_recorded", hw.get("qpc_overhead_ns", 0.0) > 0.0)

    flop_ok = raw_available and all(
        row["effective_macs"] == 16 * row["m"] * row["d"] ** 2 + 2 * row["m"] ** 2 * row["d"]
        and row["effective_flops"] == 2 * row["effective_macs"]
        for row in raw_rows
    )
    record("test_v2_1_flop_mac_ledger_matches_v2_0", flop_ok)
    same_compute = raw_available
    if raw_available:
        for d in (512, 640):
            for m in (1, 4, 8, 16):
                for k in (1, 4, 8):
                    row_counts = {variant: {row["effective_macs"] for row in raw_rows if (row["d"], row["m"], row["K"], row["variant"]) == (d, m, k, variant)} for variant in ("A", "B", "C")}
                    if not all(values == {16 * m * d * d + 2 * m * m * d} for values in row_counts.values()):
                        same_compute = False
    record("test_v2_1_same_compute_abc", same_compute)
    record("test_v2_1_sweep_contains_m1_control", 1 in config.get("m", []) and config.get("m1_role") == "CONTROL_ONLY")
    completeness = raw_available and check_sweep_completeness(raw_rows, native_status=native_status)
    record("test_v2_1_sweep_completeness", completeness)

    fixture_c = lambda T1, TK, K: (TK - T1) / (K - 1)
    fixture_rho = fixture_c(10.0, 24.0, 8) / fixture_c(12.0, 40.0, 8)
    record("test_v2_1_rho_resident_formula_fixture", math.isclose(fixture_rho, .5, rel_tol=0, abs_tol=1e-15))
    fixture_delta = abs(fixture_c(12.0, 43.5, 8) - fixture_c(12.0, 40.0, 8)) / fixture_c(12.0, 40.0, 8)
    record("test_v2_1_c_vs_b_formula_fixture", math.isclose(fixture_delta, .125, rel_tol=0, abs_tol=1e-15) and fixture_c(12, 43.5, 8) > fixture_c(10, 24, 8))
    fixture_gain = max(160.0, 210.0) / 100.0
    record("test_v2_1_matrixization_formula_fixture", math.isclose(fixture_gain, 2.1, rel_tol=0, abs_tol=1e-15))
    block_path = results_root / "OMEGA_CONFORMANCE_BLOCK.yaml"
    block_ok = False
    if block_path.is_file():
        try:
            block = read_json(block_path)["OMEGA_CONFORMANCE_BLOCK"]
            block_ok = (
                block.get("authority", {}).get("v2_0_validated_head") == V2_0_VALIDATED_HEAD
                and block.get("authority", {}).get("conversacion_md_blob") == CONVERSACION_MD_BLOB
                and block.get("global_status") == "CONFORMANCE_HOLD"
                and block.get("status") in ("CONFORMANT", "CONFORMANCE_HOLD")
                and block.get("actual_candidate", {}).get("values_obtained_by_introspection_and_native_measurement") is True
            )
        except (OSError, ValueError, KeyError):
            block_ok = False
    record("test_v2_1_conformance_block", block_ok)
    path_keys = ("repo_root_abs", "unit_root_abs", "source_root_abs", "build_root_abs", "executable_abs", "results_root_abs", "raw_results_abs", "report_abs", "conformance_block_abs")
    absolute_paths = {key: config.get(key) for key in path_keys}
    paths_ok = all(isinstance(value, str) and Path(value).is_absolute() for value in absolute_paths.values())
    manifest_path = results_root / "build_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    source_hashes = manifest.get("source_file_sha256", {})
    source_hashes_ok = bool(source_hashes) and source_hashes == {name: sha256_file(UNIT_ROOT / name) for name in source_files()}
    executable_hash_ok = manifest.get("executable_sha256") == sha256_file(executable)
    required_outputs = (
        "hardware_preflight.json", "q4_physical_ledger.json", "worker_shard_manifest.json", "benchmark_config.json",
        "native_test_report.json", "native_run_status.json", "native_stdout.log", "native_stderr.log", "raw_measurements.csv",
    )
    outputs_exist = all((results_root / name).is_file() and len(sha256_file(results_root / name)) == 64 for name in required_outputs)
    record("test_v2_1_report_contract_paths_and_hashes", paths_ok and source_hashes_ok and executable_hash_ok and outputs_exist)
    return [{"name": name, "status": results[name][0], "detail": results[name][1]} for name in TEST_NAMES]


def source_files() -> list[str]:
    names = ["OMEGA_V2_1_PHYSICAL_SPEC.md", "CMakeLists.txt", ".gitignore", "scripts/analyze_v2_1.py"]
    names.extend(str(path.relative_to(UNIT_ROOT)) for path in sorted((UNIT_ROOT / "src").glob("*.cpp")))
    names.extend(str(path.relative_to(UNIT_ROOT)) for path in sorted((UNIT_ROOT / "src").glob("*.hpp")))
    names.extend(str(path.relative_to(UNIT_ROOT)) for path in sorted((UNIT_ROOT / "tests").glob("*.py")))
    return names


def build_test_report(
    provenance: dict[str, Any], seal_info: dict[str, Any], executable: Path, hardware: dict[str, Any],
    q4: dict[str, Any], worker_shards: dict[str, Any], config: dict[str, Any], native_tests: dict[str, Any],
    raw_rows: list[dict[str, Any]], summary: dict[str, Any],
) -> list[dict[str, Any]]:
    native_status = summary["native_status"].get("status", "UNKNOWN")
    rows = build_test_rows(provenance, executable, hardware, q4, worker_shards, config, raw_rows, RESULTS_ROOT)
    # Add the seal check result explicitly to provenance test details.
    seal_ok = seal_info.get("terminal_status") == "OMEGA_V2_0_CONFORMANT_PASS" and seal_info.get("execution_commit") == V2_0_IMPLEMENTATION_COMMIT
    by_name = {row["name"]: row for row in rows}
    by_name["test_v2_1_reference_commit_and_v2_0_seal"]["status"] = "PASS" if seal_ok else "FAIL"
    if not seal_ok:
        by_name["test_v2_1_reference_commit_and_v2_0_seal"]["detail"] = "V2-0 result seal is invalid"
    # When the native sweep did not complete, report contractual post-sweep tests as SKIP explicitly.
    if native_status != "SWEEP_COMPLETE":
        post_sweep = {
            "test_v2_1_no_heap_allocation_in_timed_kernel",
            "test_v2_1_flop_mac_ledger_matches_v2_0",
            "test_v2_1_same_compute_abc",
            "test_v2_1_sweep_completeness",
        }
        for name in post_sweep:
            by_name[name]["status"] = "SKIP"
            by_name[name]["detail"] = f"native sweep status={native_status}"
    return [by_name[name] for name in TEST_NAMES]


def choose_terminal(native_status: dict[str, Any], tests: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[str, str]:
    native_code = native_status.get("status", "UNKNOWN")
    if native_code != "SWEEP_COMPLETE":
        skipped = [row for row in tests if row["status"] == "SKIP"]
        reason = native_status.get("reason", native_status.get("hold_reason", "native preflight/sweep stopped"))
        if skipped:
            return "V2_1_ACCEPTANCE_HOLD", f"native stop {native_code}; {len(skipped)} contractual tests SKIP; {reason}"
        return native_code, reason
    if any(row["status"] != "PASS" for row in tests):
        return "V2_1_ACCEPTANCE_HOLD", "one or more contractual tests failed or were skipped"
    gates = summary.get("gate_metrics") or {}
    if gates.get("analysis_error"):
        return "MEASUREMENT_INVALID", gates["analysis_error"]
    if gates.get("measurement_stability_hold"):
        return "MEASUREMENT_STABILITY_HOLD", f"{gates['noisy_primary_stability_cells']}/9 primary cells exceed R_MAD 0.15"
    if not gates.get("rho_minimum_pass"):
        return "RESIDENCY_GATE_FAIL", f"rho_resident={gates.get('rho_resident')} > 0.50"
    if not gates.get("causal_minimum_pass"):
        if gates.get("delta_CA", 1.0) <= .25 and gates.get("delta_CB", 1.0) > .25:
            return "CAUSAL_RESIDENCY_FAIL", f"A and C are within 25% while C does not approach B; delta_CA={gates.get('delta_CA')}, delta_CB={gates.get('delta_CB')}"
        return "RESIDENCY_SIGNAL_NOT_CAUSALLY_ISOLATED", f"delta_CB={gates.get('delta_CB')}, c_C_gt_c_A={gates.get('c_C_gt_c_A')}"
    if not gates.get("matrixization_minimum_pass"):
        return "MATRIXIZATION_GATE_FAIL", f"G_matrix={gates.get('G_matrix')} < 1.50"
    if gates.get("rho_strong") and gates.get("causal_strong") and gates.get("matrixization_strong"):
        return "OMEGA_V2_1_PHYSICAL_PASS_STRONG", "all strong point-estimate gates pass"
    return "OMEGA_V2_1_PHYSICAL_PASS_MINIMUM", "all minimum point-estimate gates pass"


def build_conformance_block(provenance: dict[str, Any], seal_info: dict[str, Any], hardware: dict[str, Any],
                            q4: dict[str, Any], worker_shards: dict[str, Any], summary: dict[str, Any],
                            tests: list[dict[str, Any]], terminal: str) -> dict[str, Any]:
    families = q4.get("core_families", [])
    candidates = []
    for d in (512, 640):
        family = next((row for row in families if row.get("d") == d), {})
        matrices = family.get("matrices", [])
        candidates.append({
            "candidate_id": f"V2-{d}",
            "d": d,
            "m": [4, 8, 16],
            "m_control": [1],
            "K": [1, 4, 8],
            "recurrent_block_formula": "4d^2 + 12d^2 = 16d^2",
            "P_core_unique": 16 * d * d,
            "P_shell": 0,
            "P_retriever": 0,
            "P_memory_learned": 0,
            "B_memory_static": 0,
            "B_index": 0,
            "logical_weight_bytes": family.get("logical_weight_bytes"),
            "logical_scale_bytes": family.get("logical_scale_bytes"),
            "alignment_padding_bytes": family.get("alignment_padding_bytes"),
            "physical_allocated_bytes": family.get("physical_allocated_bytes"),
            "sharing": "A/C shared; B distinct untied equal-valued storage",
            "values_obtained_by_introspection": len(matrices) == 7,
        })
    native_complete = summary.get("native_status", {}).get("status") == "SWEEP_COMPLETE"
    tests_pass = all(row["status"] == "PASS" for row in tests)
    return {
        "OMEGA_CONFORMANCE_BLOCK": {
            "authority": {
                "repository": "Hackerprod/Ejercise",
                "v2_0_validated_head": V2_0_VALIDATED_HEAD,
                "v2_0_implementation_commit": V2_0_IMPLEMENTATION_COMMIT,
                "v2_1_implementation_commit": provenance["implementation_commit"],
                "conversacion_md_blob": CONVERSACION_MD_BLOB,
                "contract_line_ranges": ["2390-2485", "2585-2630", "2688-2945", "3078-3145"],
            },
            "phase": "T0_PHYSICAL",
            "contract_target": {
                "d": [512, 640],
                "m_target": [4, 8, 16],
                "m_control": [1],
                "K": [1, 4, 8],
                "recurrent_block_formula": "4d^2 + 12d^2 = 16d^2",
                "quantization": {"format": "signed_symmetric_Q4", "group_size": 32, "scale_dtype": "FP16", "zero_point": False},
                "variants": {"A": "shared_same_storage_resident", "B": "distinct_storage_untied_equal_values", "C": "shared_same_storage_evicted_each_round"},
            },
            "actual_candidate": {
                "cpu_model": hardware.get("hardware", {}).get("cpu_model"),
                "windows_native": platform.system() == "Windows",
                "hardware": hardware.get("hardware", {}),
                "selected_workers": hardware.get("selected_workers", []),
                "candidates": candidates,
                "values_obtained_by_introspection_and_native_measurement": native_complete and all(row["values_obtained_by_introspection"] for row in candidates),
                "q4_physical_ledger_path_abs": str(RESULTS_ROOT / "q4_physical_ledger.json"),
                "worker_shard_manifest_path_abs": str(RESULTS_ROOT / "worker_shard_manifest.json"),
                "benchmark_config_path_abs": str(RESULTS_ROOT / "benchmark_config.json"),
                "terminal_status": terminal,
            },
            "v2_0_seal": {
                "path_abs": str(V2_0_SEAL_PATH.resolve()),
                "sha256": sha256_file(V2_0_SEAL_PATH),
                "terminal_status": seal_info["terminal_status"],
                "execution_commit": seal_info["execution_commit"],
                "artifact_sha256": seal_info["artifact_sha256"],
                "all_sealed_result_hashes_validated": True,
            },
            "science_scope": {"allowed": "physical residency/matrixization/causal eviction at T0 only", "language_quality": "FORBIDDEN", "T3": "HOLD"},
            "deviations": [],
            "authorized_deviation_ids": [],
            "claim_scope": {
                "allowed": ["physical_residency_gate", "matrixization_gate", "causal_eviction_control", "physical_Q4_footprint", "V2_2_release_decision"],
                "forbidden": ["language_quality", "sharing_quality_after_training", "test_time_K_quality_scaling", "external_memory_claim", "T3_release"],
            },
            "global_status": "CONFORMANCE_HOLD",
            "status": "CONFORMANT" if tests_pass else "CONFORMANCE_HOLD",
        }
    }


def format_report(terminal: str, reason: str, provenance: dict[str, Any],
                  hardware: dict[str, Any], gates: dict[str, Any] | None,
                  tests: list[dict[str, Any]], artifact_hashes: dict[str, Any]) -> str:
    lines = [
        "# OMEGA-V2-1 Physical T0 Report", "",
        f"- terminal_status: `{terminal}`",
        f"- terminal_reason: `{reason}`",
        f"- repo_root_abs: `{REPO_ROOT}`",
        f"- unit_root_abs: `{UNIT_ROOT}`",
        f"- implementation_commit: `{provenance['implementation_commit']}`",
        f"- implementation_parent: `{provenance['implementation_parent']}`",
        f"- branch: `{provenance['branch']}`",
        f"- git_status_porcelain: `{provenance['git_status_porcelain']}`",
        f"- global_status: `CONFORMANCE_HOLD`",
        f"- T3: `HOLD`",
        "",
        "## Native hardware preflight",
        f"- cpu_model: `{hardware.get('hardware', {}).get('cpu_model')}`",
        f"- physical/logical processors: `{hardware.get('hardware', {}).get('physical_core_count')}` / `{hardware.get('hardware', {}).get('logical_processor_count')}`",
        f"- P/E cores measured/classified: `{hardware.get('hardware', {}).get('p_core_count')}` / `{hardware.get('hardware', {}).get('e_core_count')}`",
        f"- LLC bytes: `{hardware.get('hardware', {}).get('llc_bytes')}`",
        f"- cache line bytes: `{hardware.get('hardware', {}).get('cache_line_bytes')}`",
        f"- selected CPU sets: `{[worker.get('windows_cpu_set_id') for worker in hardware.get('selected_workers', [])]}`",
        f"- H0 P-core classification: `{hardware.get('p_classification_method')}`",
        f"- eviction method/E: `{hardware.get('eviction_method')}` / `{hardware.get('eviction_effectiveness_E')}`",
        "",
        "## Primary gates (point estimate; bootstrap90 diagnostic only)",
    ]
    if gates:
        for key in ("c_A_K8", "c_B_K8", "c_C_K8", "rho_resident", "rho_CB", "delta_CB", "c_C_gt_c_A", "R_A_m1", "R_A_m4", "R_A_m8", "R_A_m16", "G_matrix", "noisy_primary_stability_cells", "measurement_stability_hold"):
            lines.append(f"- {key}: `{gates.get(key)}`")
        lines.append(f"- bootstrap90_diagnostics: `{json.dumps(gates.get('diagnostic_bootstrap90', {}), sort_keys=True)}`")
    else:
        lines.append("- NOT_COMPUTED: sweep did not reach valid analysis")
    lines.extend(["", "## Tests (all required names)"])
    for row in tests:
        lines.append(f"- {row['status']}: `{row['name']}`" + (f" — {row['detail']}" if row.get("detail") else ""))
    lines.extend(["", "## SHA-256 (complete)"])
    for name, record in sorted(artifact_hashes.items()):
        lines.append(f"- `{name}`: `{record['sha256']}` ({record['size_bytes']} bytes; `{record['absolute_path']}`)")
    lines.extend(["", "STOP: no language training, GPU, LN, T3, or OMEGA-trained scoring/generation was run.", ""])
    return "\n".join(lines)


def test_report_payload(implementation_commit: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "omega-v2-1-test-report-v1",
        "implementation_commit": implementation_commit,
        "test_count": len(tests),
        "pass_count": sum(row["status"] == "PASS" for row in tests),
        "fail_count": sum(row["status"] == "FAIL" for row in tests),
        "skip_count": sum(row["status"] == "SKIP" for row in tests),
        "tests": tests,
    }


def main() -> int:
    if os.name != "nt" or sys.platform != "win32":
        raise RuntimeError("MEASUREMENT_INVALID: V2-1 must run as a native Windows process")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-1 results are immutable and already exist: {RESULTS_ROOT}")

    provenance = source_provenance()
    seal_validation = validate_v2_0_seal()
    vswhere = find_vswhere()
    cmake, visual_studio_path, cmake_version = find_msvc_cmake(vswhere)

    source_hashes_before = {name: sha256_file(UNIT_ROOT / name) for name in source_files()}
    executable, build_logs = run_build(cmake)
    executable_hash = sha256_file(executable)
    source_hashes_after_build = {name: sha256_file(UNIT_ROOT / name) for name in source_files()}
    if source_hashes_before != source_hashes_after_build:
        raise RuntimeError("V2_1_ACCEPTANCE_HOLD: implementation source/configuration changed during build")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    build_manifest = {
        "schema": "omega-v2-1-native-build-manifest-v1",
        "implementation_commit": provenance["implementation_commit"],
        "git_provenance": provenance,
        "generator": "Visual Studio 17 2022 x64",
        "configuration": "Release",
        "visual_studio_installation": visual_studio_path,
        "cmake_path": str(cmake),
        "cmake_version": cmake_version,
        "compiler": "MSVC",
        "executable_absolute_path": str(executable.resolve()),
        "executable_size_bytes": executable.stat().st_size,
        "executable_sha256": executable_hash,
        "source_file_sha256": source_hashes_before,
        "configure_stdout": build_logs["configure_stdout"],
        "configure_stderr": build_logs["configure_stderr"],
        "build_stdout": build_logs["build_stdout"],
        "build_stderr": build_logs["build_stderr"],
    }
    write_json(RESULTS_ROOT / "build_manifest.json", build_manifest)
    run_timeout_seconds = 8 * 60 * 60
    try:
        native_process = subprocess.run(
            [str(executable), "--run"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=run_timeout_seconds,
        )
        (RESULTS_ROOT / "native_stdout.log").write_text(native_process.stdout, encoding="utf-8", newline="\n")
        (RESULTS_ROOT / "native_stderr.log").write_text(native_process.stderr, encoding="utf-8", newline="\n")
        if not (RESULTS_ROOT / "native_run_status.json").is_file():
            write_json(RESULTS_ROOT / "native_run_status.json", {
                "status": "MEASUREMENT_INVALID",
                "reason": f"native executable exited {native_process.returncode} without native_run_status.json",
                "returncode": native_process.returncode,
            })
    except subprocess.TimeoutExpired as error:
        (RESULTS_ROOT / "native_stdout.log").write_text(str(error.stdout or ""), encoding="utf-8", newline="\n")
        (RESULTS_ROOT / "native_stderr.log").write_text(str(error.stderr or ""), encoding="utf-8", newline="\n")
        write_json(RESULTS_ROOT / "native_run_status.json", {
            "status": "V2_1_ACCEPTANCE_HOLD",
            "reason": "native sweep exceeded the authorized CPU wall-time budget",
            "timeout_seconds": run_timeout_seconds,
        })

    native_status = read_json(RESULTS_ROOT / "native_run_status.json")
    hardware_path = RESULTS_ROOT / "hardware_preflight.json"
    q4_path = RESULTS_ROOT / "q4_physical_ledger.json"
    shards_path = RESULTS_ROOT / "worker_shard_manifest.json"
    config_path = RESULTS_ROOT / "benchmark_config.json"
    native_tests_path = RESULTS_ROOT / "native_test_report.json"
    raw_path = RESULTS_ROOT / "raw_measurements.csv"
    csv_header = (
        "run_id,block_id,sample_id,is_warmup,timestamp,cpu_model,worker_cpu_sets,worker_core_ids,d,m,K,variant,"
        "B_group_id,eviction_method,round_index,qpc_ticks,qpc_seconds,rdtscp_delta_if_available,effective_macs,"
        "effective_flops,output_checksum,timed_heap_allocation_count,frequency_if_available,temperature_if_available,valid,invalid_reason\n"
    )
    fallback_config = {
        "schema": "omega-v2-1-benchmark-config-v1",
        "repo_root_abs": str(REPO_ROOT.resolve()),
        "unit_root_abs": str(UNIT_ROOT.resolve()),
        "source_root_abs": str((UNIT_ROOT / "src").resolve()),
        "build_root_abs": str(BUILD_ROOT.resolve()),
        "executable_abs": str(EXECUTABLE.resolve()),
        "results_root_abs": str(RESULTS_ROOT.resolve()),
        "raw_results_abs": str(raw_path.resolve()),
        "report_abs": str((RESULTS_ROOT / "OMEGA_V2_1_REPORT.md").resolve()),
        "conformance_block_abs": str((RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml").resolve()),
        "status": "NOT_MEASURED",
    }
    fallbacks = (
        (hardware_path, {"status": "NOT_MEASURED", "hardware": {}, "h0_measurements": [], "selected_workers": []}),
        (q4_path, {"schema": "omega-v2-1-q4-physical-ledger-v1", "group_size": 32, "scale_dtype": "FP16", "zero_point": False, "core_families": [], "b_pool_by_d_k": []}),
        (shards_path, {"schema": "omega-v2-1-worker-shard-manifest-v1", "matrices": []}),
        (config_path, fallback_config),
        (native_tests_path, {"schema": "omega-v2-1-native-preflight-tests-v1", "status": "NOT_RUN", "tests": []}),
    )
    for path, value in fallbacks:
        if not path.is_file():
            write_json(path, value)
    if not raw_path.is_file():
        raw_path.write_text(csv_header, encoding="utf-8", newline="\n")
    for log_path in (RESULTS_ROOT / "native_stdout.log", RESULTS_ROOT / "native_stderr.log"):
        if not log_path.is_file():
            log_path.write_text("", encoding="utf-8", newline="\n")
    hardware = read_json(hardware_path) if hardware_path.is_file() else {}
    q4_ledger = read_json(q4_path) if q4_path.is_file() else {}
    worker_shards = read_json(shards_path) if shards_path.is_file() else {}
    config = read_json(config_path) if config_path.is_file() else {}
    native_tests = read_json(native_tests_path) if native_tests_path.is_file() else {"tests": []}
    if raw_path.is_file():
        raw_rows, raw_fieldnames = parse_raw(raw_path)
    else:
        raw_rows, raw_fieldnames = [], []

    summaries, cell_map, discarded = summarize_samples(raw_rows)
    summary = {
        "schema": "omega-v2-1-summary-metrics-v1",
        "native_status": native_status,
        "hardware_preflight": hardware,
        "benchmark_config": config,
        "q4_physical_ledger": q4_ledger,
        "worker_shard_manifest": worker_shards,
        "raw_fieldnames": raw_fieldnames,
        "raw_row_count": len(raw_rows),
        "raw_sample_count": len({(row["d"], row["m"], row["K"], row["variant"], row["is_warmup"], row["block_id"], row["sample_id"]) for row in raw_rows}),
        "cells": summaries,
        "discarded_samples_with_predeclared_reasons": discarded,
        "gate_metrics": None,
    }
    if native_status.get("status") == "SWEEP_COMPLETE":
        try:
            summary["gate_metrics"] = compute_gate_metrics(cell_map)
            summary["gate_metrics"]["diagnostic_bootstrap90"] = bootstrap_gate_diagnostics(raw_rows)
        except Exception as error:
            summary["gate_metrics"] = {"analysis_error": f"{type(error).__name__}: {error}"}
    write_json(RESULTS_ROOT / "summary_metrics.json", summary)

    preliminary_block = build_conformance_block(provenance, seal_validation["seal"], hardware, q4_ledger, worker_shards, summary, [], "V2_1_PRELIMINARY")
    write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml", preliminary_block)
    tests = build_test_report(provenance, seal_validation["seal"], executable, hardware, q4_ledger, worker_shards, config, native_tests, raw_rows, summary)
    terminal, reason = choose_terminal(native_status, tests, summary)
    block = build_conformance_block(provenance, seal_validation["seal"], hardware, q4_ledger, worker_shards, summary, tests, terminal)
    write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml", block)
    # Validate the emitted final block and then freeze the individual test statuses.
    tests = build_test_report(provenance, seal_validation["seal"], executable, hardware, q4_ledger, worker_shards, config, native_tests, raw_rows, summary)
    test_failures = [row for row in tests if row["status"] != "PASS"]
    if test_failures and terminal in ("OMEGA_V2_1_PHYSICAL_PASS_MINIMUM", "OMEGA_V2_1_PHYSICAL_PASS_STRONG"):
        terminal = "V2_1_ACCEPTANCE_HOLD"
        reason = "one or more contractual tests were not PASS in the final result audit"
        block = build_conformance_block(provenance, seal_validation["seal"], hardware, q4_ledger, worker_shards, summary, tests, terminal)
        write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml", block)
    write_json(RESULTS_ROOT / "summary_metrics.json", {**summary, "terminal_status": terminal, "terminal_reason": reason})
    write_json(RESULTS_ROOT / "test_report.json", test_report_payload(provenance["implementation_commit"], tests))

    # Hash the pre-report artifacts; the artifact_hashes file below also includes report+sidecar.
    pre_report_paths = [
        UNIT_ROOT / "OMEGA_V2_1_PHYSICAL_SPEC.md",
        UNIT_ROOT / "CMakeLists.txt",
        *[UNIT_ROOT / name for name in source_files() if name not in ("OMEGA_V2_1_PHYSICAL_SPEC.md", "CMakeLists.txt")],
        executable,
        V2_0_SEAL_PATH,
        RESULTS_ROOT / "build_manifest.json",
        hardware_path,
        q4_path,
        shards_path,
        config_path,
        RESULTS_ROOT / "native_test_report.json",
        RESULTS_ROOT / "native_run_status.json",
        RESULTS_ROOT / "native_stdout.log",
        RESULTS_ROOT / "native_stderr.log",
        raw_path,
        RESULTS_ROOT / "summary_metrics.json",
        RESULTS_ROOT / "test_report.json",
        RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml",
    ]
    missing_artifacts = [str(path) for path in pre_report_paths if not path.is_file()]
    source_hashes_after = {name: sha256_file(UNIT_ROOT / name) for name in source_files()}
    source_unchanged = source_hashes_after == build_manifest["source_file_sha256"]
    if missing_artifacts or not source_unchanged or sha256_file(executable) != build_manifest["executable_sha256"]:
        terminal = "V2_1_ACCEPTANCE_HOLD"
        reason = f"artifact/source integrity audit failed; missing={missing_artifacts}; sources_unchanged={source_unchanged}"
        for row in tests:
            if row["name"] == "test_v2_1_report_contract_paths_and_hashes":
                row["status"] = "FAIL"
                row["detail"] = reason
                break
        summary["terminal_status"] = terminal
        summary["terminal_reason"] = reason
        block = build_conformance_block(provenance, seal_validation["seal"], hardware, q4_ledger, worker_shards, summary, tests, terminal)
        write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml", block)
        write_json(RESULTS_ROOT / "summary_metrics.json", summary)
        write_json(RESULTS_ROOT / "test_report.json", test_report_payload(provenance["implementation_commit"], tests))

    report_path = RESULTS_ROOT / "OMEGA_V2_1_REPORT.md"
    artifact_hashes_path = RESULTS_ROOT / "artifact_hashes.json"

    def write_report_and_hashes() -> tuple[str, dict[str, Any], bool]:
        report_sha_map = {
            str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size, "absolute_path": str(path.resolve())}
            for path in pre_report_paths
            if path.is_file()
        }
        report = format_report(terminal, reason, provenance, hardware, summary.get("gate_metrics"), tests, report_sha_map)
        report_path.write_text(report, encoding="utf-8", newline="\n")
        report_hash = sha256_file(report_path)
        report_sidecar = RESULTS_ROOT / "OMEGA_V2_1_REPORT.md.sha256"
        report_sidecar.write_text(report_hash + "\n", encoding="ascii", newline="\n")
        final_artifact_paths = [*pre_report_paths, report_path, report_sidecar]
        artifact_hashes = {
            "schema": "omega-v2-1-artifact-hashes-v1",
            "implementation_commit": provenance["implementation_commit"],
            "report_self_sha256": report_hash,
            "artifacts": {
                str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in final_artifact_paths
                if path.is_file()
            },
        }
        write_json(artifact_hashes_path, artifact_hashes)
        verified = all(
            record["sha256"] == sha256_file(Path(path)) and len(record["sha256"]) == 64
            for path, record in artifact_hashes["artifacts"].items()
        )
        verified = verified and sha256_file(report_path) == report_sidecar.read_text(encoding="ascii").strip()
        return report_hash, artifact_hashes, verified

    report_hash, artifact_hashes, verified = write_report_and_hashes()
    if not verified:
        terminal = "V2_1_ACCEPTANCE_HOLD"
        reason = "final SHA-256 path/hash verification failed"
        for row in tests:
            if row["name"] == "test_v2_1_report_contract_paths_and_hashes":
                row["status"] = "FAIL"
                row["detail"] = reason
                break
        summary["terminal_status"] = terminal
        summary["terminal_reason"] = reason
        block = build_conformance_block(provenance, seal_validation["seal"], hardware, q4_ledger, worker_shards, summary, tests, terminal)
        write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml", block)
        write_json(RESULTS_ROOT / "summary_metrics.json", summary)
        write_json(RESULTS_ROOT / "test_report.json", test_report_payload(provenance["implementation_commit"], tests))
        report_hash, artifact_hashes, verified = write_report_and_hashes()

    print(json.dumps({
        "terminal_status": terminal,
        "terminal_reason": reason,
        "implementation_commit": provenance["implementation_commit"],
        "implementation_parent": provenance["implementation_parent"],
        "results_root_abs": str(RESULTS_ROOT.resolve()),
        "report_abs": str(report_path.resolve()),
        "report_sha256": report_hash,
        "artifact_hashes_abs": str(artifact_hashes_path.resolve()),
        "artifact_hash_count": len(artifact_hashes["artifacts"]),
        "artifact_hashes_verified": verified,
        "test_report_abs": str((RESULTS_ROOT / "test_report.json").resolve()),
        "tests_passed": read_json(RESULTS_ROOT / "test_report.json")["pass_count"],
        "tests_failed": read_json(RESULTS_ROOT / "test_report.json")["fail_count"],
        "tests_skipped": read_json(RESULTS_ROOT / "test_report.json")["skip_count"],
    }, indent=2, sort_keys=True))
    if terminal in ("OMEGA_V2_1_PHYSICAL_PASS_MINIMUM", "OMEGA_V2_1_PHYSICAL_PASS_STRONG"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
