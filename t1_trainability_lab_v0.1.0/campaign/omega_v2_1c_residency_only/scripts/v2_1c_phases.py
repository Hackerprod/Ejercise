"""Ordered, sealed V2-1c preflight stages; no sweep is started by these stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import run_v2_1c as u

KQ_BINDING_CPU_SET_IDS = [266, 264, 258, 270]
KQ_BINDING_V_I = [511664633.51857996, 510535857.6587815, 500940450.12757146, 500577604.74870884]
SEALED_EVIDENCE_ROOT = u.UNIT_ROOT / "results" / "v2_1c_sealed"


def prepare_companion() -> dict[str, Any]:
    """Compile once and run no-timing checks before the binding notice/window."""
    u.PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    attempt_manifest, _, attempt_sha = u.validate_attempt02()
    c2_manifest, c2_summary, c2_build, c2_hash = u.validate_candidate02()
    u.kq_contract.validate_v2_0_seal()
    cmake, vs_install, cmake_version = u.find_msvc_cmake(u.find_vswhere())
    source_before = u.source_hashes()
    physical_before = {str(path.resolve()): u.sha256_file(path) for path in u.physical_dependency_paths()}
    kernel_before = {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()}
    build_logs = u.build_native(cmake)
    python_tests = u.run_python_contract_tests()
    kernel_binding = u.validate_kq_kernel_binding(c2_build)
    if source_before != u.source_hashes() or physical_before != {str(path.resolve()): u.sha256_file(path) for path in u.physical_dependency_paths()} or kernel_before != {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()}:
        raise RuntimeError("V2_1C_SOURCE_HOLD: sources changed during static companion build")
    record = {
        "schema": "omega-v2-1c-companion-preparation-v1", "stage": "NO_TIMING_PREPARATION_ONLY",
        "implementation_commit": u.git("rev-parse", "HEAD"), "branch": u.git("branch", "--show-current"),
        "git_status_unit": u.git("status", "--porcelain", "--", str(u.UNIT_ROOT.relative_to(u.REPO_ROOT))),
        "attempt02_manifest_sha256": attempt_sha, "attempt02_artifact_count": len(attempt_manifest["artifacts"]),
        "candidate02_kq_manifest_sha256": u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "candidate02_kq_exe_sha256": c2_hash, "candidate02_kernel_binding": kernel_binding,
        "v2_1c_source_sha256": source_before, "physical_dependency_sha256": physical_before,
        "candidate02_kernel_tu_sha256": kernel_before,
        "companion_bench_exe_abs": str(u.BENCH_EXE.resolve()), "companion_bench_exe_sha256": u.sha256_file(u.BENCH_EXE),
        "companion_correctness_exe_abs": str(u.CORRECTNESS_EXE.resolve()), "companion_correctness_exe_sha256": u.sha256_file(u.CORRECTNESS_EXE),
        "build_logs": build_logs, "python_contract_tests": python_tests,
        "cmake_path": str(cmake), "cmake_version": cmake_version, "visual_studio_installation": vs_install,
        "timed_binding_started": False, "core_selection_started": False, "correctness_started": False, "sweep_started": False,
    }
    u.write_json(u.PREFLIGHT_ROOT / "companion_preparation.json", record)
    return record


def seal_stage(root: Path, record: dict[str, Any], tests: dict[str, Any], artifact_paths: list[Path], report_name: str, title: str) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"V2-1c preflight stage is immutable: {root}")
    root.mkdir(parents=True, exist_ok=False)
    write = u.write_json
    record_path = root / "stage_record.json"
    test_path = root / "test_report.json"
    write(record_path, record)
    write(test_path, tests)
    unique = {str(path.resolve()): path for path in [*artifact_paths, record_path, test_path] if path.is_file()}
    hashes = {absolute: {"sha256": u.sha256_file(path), "size_bytes": path.stat().st_size, "absolute_path": absolute} for absolute, path in unique.items()}
    lines = [title, "", f"- stage_status: `{record['stage_status']}`", f"- implementation_commit: `{record['implementation_commit']}`", f"- tests PASS/FAIL/SKIP: `{tests['pass_count']}/{tests['fail_count']}/{tests['skip_count']}`", "", "## Tests"]
    lines.extend(f"- {row['status']}: `{row['name']}`" for row in tests["tests"])
    lines.extend(["", "## SHA-256"])
    lines.extend(f"- `{path}`: `{item['sha256']}` ({item['size_bytes']} bytes)" for path, item in sorted(hashes.items()))
    report_path = root / report_name
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    report_sha = u.sha256_file(report_path)
    sidecar = root / (report_name + ".sha256")
    sidecar.write_text(report_sha + "\n", encoding="ascii", newline="\n")
    final_paths = [*unique.values(), report_path, sidecar]
    artifact_manifest = {
        "schema": "omega-v2-1c-stage-artifact-hashes-v1",
        "stage": record.get("phase"),
        "implementation_commit": record["implementation_commit"],
        "report_self_sha256": report_sha,
        "artifacts": {str(path.resolve()): {"sha256": u.sha256_file(path), "size_bytes": path.stat().st_size} for path in final_paths if path.is_file()},
    }
    manifest_path = root / "artifact_hashes.json"
    write(manifest_path, artifact_manifest)
    verified = all(Path(path).is_file() and u.sha256_file(Path(path)) == row["sha256"] for path, row in artifact_manifest["artifacts"].items())
    verified = verified and u.sha256_file(report_path) == sidecar.read_text(encoding="ascii").strip()
    write(root / "artifact_hashes_verified.json", {"verified": verified, "artifact_count": len(artifact_manifest["artifacts"])})
    if not verified:
        raise RuntimeError(f"V2_1C_STAGE_HASH_FAILURE: {root}")
    return {"root": root, "record": record, "tests": tests, "report_path": report_path, "report_sha256": report_sha, "manifest_path": manifest_path, "manifest_sha256": u.sha256_file(manifest_path), "manifest": artifact_manifest, "verified": verified}


def verify_stage(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = u.verify_artifact_manifest(root / "artifact_hashes.json")
    record = u.read_json(root / "stage_record.json")
    tests = u.read_json(root / "test_report.json")
    if not (root / "artifact_hashes_verified.json").is_file() or u.read_json(root / "artifact_hashes_verified.json").get("verified") is not True:
        raise RuntimeError(f"V2_1C_PREFLIGHT_HOLD: stage hash sidecar not true: {root}")
    if tests["fail_count"] or tests["skip_count"]:
        raise RuntimeError(f"V2_1C_PREFLIGHT_HOLD: stage has failed/skipped tests: {root}")
    return record, manifest


def binding_stage() -> dict[str, Any]:
    root = u.BINDING_PREFLIGHT_RESULTS
    if root.exists():
        raise FileExistsError(f"binding preflight is immutable: {root}")
    preparation_path = u.PREFLIGHT_ROOT / "companion_preparation.json"
    if not preparation_path.is_file():
        raise RuntimeError("V2_1C_PREPARE_REQUIRED: build and static checks must finish before notifying the judge for binding")
    preparation = u.read_json(preparation_path)
    attempt_manifest, attempt_config, attempt_sha = u.validate_attempt02()
    binding_ids = [int(row["windows_cpu_set_id"]) for row in attempt_config["selected_workers"]]
    binding_weights = [float(row["h0_v_i"]) for row in attempt_config["selected_workers"]]
    if binding_ids != KQ_BINDING_CPU_SET_IDS or binding_weights != KQ_BINDING_V_I:
        raise RuntimeError(f"V2_1C_BINDING_CONFIG_HOLD: attempt_02/KQ binding config differs from MD/313: ids={binding_ids}, v_i={binding_weights}")
    c2_manifest, c2_summary, c2_build, c2_exe_hash = u.validate_candidate02()
    v2_0 = u.kq_contract.validate_v2_0_seal()
    provenance = u.verify_provenance(attempt_sha, u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"))
    cmake = Path(preparation["cmake_path"])
    vs_install = preparation["visual_studio_installation"]
    cmake_version = preparation["cmake_version"]
    source_before = u.source_hashes()
    physical_before = {str(path.resolve()): u.sha256_file(path) for path in u.physical_dependency_paths()}
    kernel_before = {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()}
    if preparation["v2_1c_source_sha256"] != source_before or preparation["physical_dependency_sha256"] != physical_before or preparation["candidate02_kernel_tu_sha256"] != kernel_before:
        raise RuntimeError("V2_1C_PREPARE_HOLD: source hashes changed after companion preparation")
    build_logs = preparation["build_logs"]
    python_tests = preparation["python_contract_tests"]
    binding_identity = u.validate_kq_kernel_binding(c2_build)
    companion_hash_before = u.sha256_file(u.BENCH_EXE)
    binding_identity["companion_executable_sha256"] = companion_hash_before
    if source_before != u.source_hashes() or physical_before != {str(path.resolve()): u.sha256_file(path) for path in u.physical_dependency_paths()} or kernel_before != {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()}:
        raise RuntimeError("V2_1C_SOURCE_HOLD: sources changed during companion build")

    payload, source_weights = u.kq_contract.v2_0_fp32_weight_payload()
    if source_weights["source_weight_value_sha256"] != u.read_json(u.KQ2_NATIVE)["source_weight_sha256"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: binding Q4 weights differ from sealed candidate_02 KQ source")
    u.PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    weight_file = u.PREFLIGHT_ROOT / "v2_0_fp32_source_weights.tmp"
    weight_file.write_bytes(payload)
    binding_json = u.PREFLIGHT_ROOT / "binding_companion.json"
    timing_json = u.PREFLIGHT_ROOT / "binding_timing_sanity.json"
    # Default CPU-set IDs and v_i are read from immutable attempt_02 config by the utility.
    binding = u.execute_equivalence("--binding-preflight", weight_file, u.KQ2_OUTPUTS, binding_json, KQ_BINDING_CPU_SET_IDS, KQ_BINDING_V_I)
    kq_native = u.read_json(u.KQ2_NATIVE)
    timing = u.compare_binding_timing(binding, kq_native) if binding.get("cells") else {"authorized_pre_sweep_binding_only": True, "allowed_ratio_band": [0.90, 1.10], "cells": {}, "no_timed_allocations": False, "pass": False}
    u.write_json(timing_json, timing)
    companion_hash_after = u.sha256_file(u.BENCH_EXE)
    same_exe = companion_hash_before == companion_hash_after
    passed = binding.get("pass") is True and timing["pass"] is True and same_exe
    tests = [
        ("test_v2_1c_binding_same_companion_exe_as_sweep", same_exe, {"exe_sha256_before": companion_hash_before, "exe_sha256_after": companion_hash_after}),
        ("test_v2_1c_candidate02_compute_identity_before_binding", binding_identity["kernel_translation_units_byte_identical"] and binding_identity["compute_defines_match"] and binding_identity["compute_options_match"] and binding_identity["compute_link_options_match"], binding_identity),
        ("test_v2_1c_binding_three_sealed_cells_bit_exact", binding.get("pass") is True, binding.get("sealed_output_comparisons", {})),
        ("test_v2_1c_binding_timing_0p90_1p10", timing["pass"], timing),
        ("test_v2_1c_binding_10_warmups_31_samples_no_allocations", binding.get("timing_protocol", {}).get("warmups") == 10 and binding.get("timing_protocol", {}).get("samples") == 31 and binding.get("timing_protocol", {}).get("no_timed_allocations") is True, binding.get("timing_protocol", {})),
        ("test_v2_1c_binding_exact_kq_cpu_sets_and_shard_weights", binding.get("worker_cpu_set_ids") == KQ_BINDING_CPU_SET_IDS and binding_ids == KQ_BINDING_CPU_SET_IDS and binding_weights == KQ_BINDING_V_I, {"cpu_set_ids": binding_ids, "shard_weights": binding_weights}),
        ("test_v2_1c_v2_0_attempt02_inputs_verified", v2_0["seal"]["terminal_status"] == "OMEGA_V2_0_CONFORMANT_PASS" and attempt_sha == u.sha256_file(u.ATTEMPT02_ROOT / "artifact_hashes.json"), {"attempt02_manifest_sha256": attempt_sha, "v2_0_seal_sha256": u.sha256_file(u.V2_0_ROOT / "V2_0_RESULT_SEAL.json")}),
        ("test_v2_1c_python_contract_tests", True, python_tests),
    ]
    test_rows = [{"name": name, "status": "PASS" if ok else "FAIL", "detail": detail} for name, ok, detail in tests]
    test_report = {"schema": "omega-v2-1c-binding-test-report-v1", "test_count": len(test_rows), "pass_count": sum(row["status"] == "PASS" for row in test_rows), "fail_count": sum(row["status"] == "FAIL" for row in test_rows), "skip_count": 0, "tests": test_rows}
    record = {
        "phase": "COMPANION_BINDING_PREFLIGHT", "implementation_commit": provenance["implementation_commit"],
        "source_provenance": provenance, "stage_status": "PASS" if passed and not test_report["fail_count"] else "COMPANION_BINDING_INVALID",
        "candidate02_kq_exe_sha256": u.FROZEN_CANDIDATE02_EXE_SHA256,
        "candidate02_kq_artifact_manifest_sha256": u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "candidate02_kernel_binding": binding_identity, "candidate02_build_manifest": c2_build,
        "candidate02_kernel_tu_sha256": kernel_before, "physical_dependency_sha256": physical_before, "v2_1c_source_sha256": source_before,
        "companion_exe_abs": str(u.BENCH_EXE.resolve()), "companion_exe_sha256_before_binding": companion_hash_before,
        "companion_exe_sha256_frozen": companion_hash_after, "attempt02_manifest_sha256": attempt_sha,
        "attempt02_worker_cpu_set_ids": [int(row["windows_cpu_set_id"]) for row in attempt_config["selected_workers"]],
        "attempt02_shard_weights": [float(row["h0_v_i"]) for row in attempt_config["selected_workers"]],
        "exact_kq_binding_cpu_set_ids": KQ_BINDING_CPU_SET_IDS, "exact_kq_binding_shard_weights": KQ_BINDING_V_I,
        "binding_report": binding, "binding_timing_ratios": timing,
        "binding_test_report": test_report, "build_logs": build_logs,
        "cmake_path": str(cmake), "cmake_version": cmake_version, "visual_studio_installation": vs_install,
        "timed_72_cell_sweep_started": False, "core_selection_started": False,
    }
    artifacts = [*(u.UNIT_ROOT / name for name in u.source_files()), *u.physical_dependency_paths(), *u.candidate02_kernel_paths(),
        Path(c2_build["executable_absolute_path"]), u.V2_0_ROOT / "V2_0_RESULT_SEAL.json", u.ATTEMPT02_ROOT / "artifact_hashes.json",
        u.ATTEMPT02_ROOT / "benchmark_config.json", u.ATTEMPT02_ROOT / "hardware_preflight.json", u.ATTEMPT02_ROOT / "q4_physical_ledger.json",
        u.KQ2_ROOT_RESULTS / "artifact_hashes.json", u.KQ2_NATIVE, u.KQ2_ROOT_RESULTS / "build_manifest.json", u.KQ2_ROOT_RESULTS / "summary_metrics.json",
        *(u.KQ2_OUTPUTS / name for name in ("candidate2_full_m4_k1.bin", "candidate2_full_m16_k1.bin", "candidate2_full_m8_k4.bin")),
        u.BENCH_EXE, u.CORRECTNESS_EXE, weight_file, binding_json, timing_json]
    report_lines = ["# V2-1c COMPANION_BINDING_PREFLIGHT", "", f"- status: `{record['stage_status']}`", f"- same companion exe hash, before/after binding: `{companion_hash_before}`", f"- frozen candidate_02 KQ exe SHA-256: `{u.FROZEN_CANDIDATE02_EXE_SHA256}`", "- This is the exact companion EXE later used for core selection and the physical sweep; no rebuild/relink is permitted after this stage.", "- Binding cells: d512,m4,K1; d512,m16,K1; d512,m8,K4; 10 warmups/31 samples; ratio band [0.90,1.10].", "- No H0 v_i measurement or 72-cell sweep started.", "", "## Binding ratios"]
    report_lines.extend(f"- {name}: `{row['ratio']:.9g}`; pass `{row['within_sanity_band']}`" for name, row in timing["cells"].items())
    stage = seal_stage(u.BINDING_PREFLIGHT_RESULTS, record, test_report, artifacts, "V2_1C_BINDING_PREFLIGHT.md", "# V2-1c COMPANION_BINDING_PREFLIGHT")
    if record["stage_status"] != "PASS" or not stage["verified"]:
        raise RuntimeError("V2_1C_BINDING_HOLD: companion binding preflight did not pass")
    return stage


def _core_selection_tests(report: dict[str, Any], attempt_config: dict[str, Any], attempt_manifest_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = report["h0_measurement"]["rows"]
    by_core: dict[int, dict[int, dict[str, Any]]] = {}
    valid_rows = True
    for row in rows:
        core_id = int(row["physical_core_id"])
        m = int(row["m"])
        by_core.setdefault(core_id, {})[m] = row
        valid_rows = valid_rows and int(row["warmups"]) == 10 and int(row["samples"]) == 31
        valid_rows = valid_rows and len(row["sample_macs_per_second"]) == 31 and row["affinity_ok"] is True
    eight_cores = len(by_core) == 8 and all(set(values) == {1, 4, 8, 16} for values in by_core.values())
    computed_vi: dict[int, float] = {}
    formula_ok = eight_cores
    for core_id, measurements in by_core.items():
        medians = {m: statistics.median(measurements[m]["sample_macs_per_second"]) for m in (4, 8, 16)}
        vi = math.pow(medians[4] * medians[8] * medians[16], 1.0 / 3.0)
        computed_vi[core_id] = vi
        expected = next((float(item["v_i"]) for item in report["p_core_ranking"] if int(item["physical_core_id"]) == core_id), math.nan)
        formula_ok = formula_ok and math.isclose(vi, expected, rel_tol=1e-12, abs_tol=1e-6)
        formula_ok = formula_ok and int(measurements[1]["samples"]) == 31  # m1 is measured/report-only, never ranked.
    ranking = report["p_core_ranking"]
    expected_ranking = sorted(ranking, key=lambda row: (-float(row["v_i"]), int(row["cpu_set_id"])))
    rank_ok = [int(row["cpu_set_id"]) for row in ranking] == [int(row["cpu_set_id"]) for row in expected_ranking]
    selected = report["selected_workers"]
    top4 = expected_ranking[:4]
    selected_ok = len(selected) == 4 and [int(row["cpu_set_id"]) for row in selected] == [int(row["cpu_set_id"]) for row in top4]
    selected_ok = selected_ok and all(math.isclose(float(a["v_i"]), float(b["v_i"]), rel_tol=0.0, abs_tol=0.0) for a, b in zip(selected, top4))
    qpc_ok = report.get("qpc_monotonic") is True and int(report.get("qpc_frequency", 0)) == int(attempt_config["qpc_frequency"]) and float(report.get("qpc_overhead_ns", 0)) > 0
    topology_ok = report.get("cpu_model", "").find("i7-13700F") >= 0 and report.get("h0_measurement", {}).get("threads_per_core") == 1 and report.get("h0_measurement", {}).get("smt_sibling_active") is False
    worker_ids = [int(row["cpu_set_id"]) for row in selected]
    worker_weights = [float(row["v_i"]) for row in selected]
    attempt_ids = [int(row["windows_cpu_set_id"]) for row in attempt_config["selected_workers"]]
    attempt_weights = [float(row["h0_v_i"]) for row in attempt_config["selected_workers"]]
    old_shards = u.read_json(u.ATTEMPT02_ROOT / "worker_shard_manifest.json")
    old_map = _attempt02_shard_map(old_shards)
    new_map = {(int(matrix["d"]), matrix["matrix"]): [(int(shard["cpu_set_id"]), int(shard["first_output_row"]), int(shard["last_output_row_exclusive"])) for shard in matrix["shards"]] for matrix in report["worker_row_shards"]}
    shards_equal = old_map == new_map
    differs = worker_ids != attempt_ids or not shards_equal
    pairing_status = "DIAGNOSTIC_ONLY / NOT_PAIRED_KERNEL_COMPARISON" if differs else "PAIRED_KERNEL_COMPARISON"
    tests = [
        ("test_v2_1c_core_selection_h0_all_8_p_cores", eight_cores and valid_rows, {"physical_p_core_count": len(by_core), "rows": len(rows), "expected_rows": 32}),
        ("test_v2_1c_core_selection_v_i_formula_m1_excluded", formula_ok, {"formula": report.get("v_i_formula"), "computed_v_i_by_physical_core": computed_vi}),
        ("test_v2_1c_core_selection_top4_tie_break_by_cpu_set_id", rank_ok and selected_ok, {"ranked_cpu_set_ids": [int(row["cpu_set_id"]) for row in ranking], "selected_cpu_set_ids": worker_ids}),
        ("test_v2_1c_core_selection_tile4_shards_cover_all_rows", _validate_row_shards(report["worker_row_shards"]), {"matrix_count": len(report["worker_row_shards"]), "row_tile": 4}),
        ("test_v2_1c_core_selection_inherited_qpc_and_topology", qpc_ok and topology_ok, {"qpc_monotonic": report.get("qpc_monotonic"), "qpc_frequency": report.get("qpc_frequency"), "qpc_overhead_ns": report.get("qpc_overhead_ns"), "cpu_model": report.get("cpu_model")}),
        ("test_v2_1c_core_selection_attempt02_reselection_pairing_label", (not differs) or pairing_status == "DIAGNOSTIC_ONLY / NOT_PAIRED_KERNEL_COMPARISON", {"attempt02_cpu_set_ids": attempt_ids, "selected_cpu_set_ids": worker_ids, "attempt02_shards_equal": shards_equal, "performance_comparison_status": pairing_status, "attempt02_manifest_sha256": attempt_manifest_sha}),
    ]
    report_obj = {"schema": "omega-v2-1c-core-selection-test-report-v1", "test_count": len(tests), "pass_count": sum(ok for _, ok, _ in tests), "fail_count": sum(not ok for _, ok, _ in tests), "skip_count": 0, "tests": [{"name": name, "status": "PASS" if ok else "FAIL", "detail": detail} for name, ok, detail in tests]}
    selection = {"selected_workers": selected, "selected_cpu_set_ids": worker_ids, "selected_v_i": worker_weights, "p_core_ranking": ranking, "h0_measurement": report["h0_measurement"], "worker_row_shards": report["worker_row_shards"], "v_i_formula": report["v_i_formula"], "m1_role": report["m1_role"], "qpc_frequency": report["qpc_frequency"], "qpc_overhead_ns": report["qpc_overhead_ns"], "qpc_monotonic": report["qpc_monotonic"], "attempt02_pairing_status": pairing_status, "attempt02_shards_equal": shards_equal}
    return report_obj, selection


def _validate_row_shards(matrices: list[dict[str, Any]]) -> bool:
    if len(matrices) != 14:
        return False
    for matrix in matrices:
        rows = int(matrix["rows"])
        shards = matrix["shards"]
        if int(matrix["row_tile"]) != 4 or len(shards) != 4 or int(matrix["rows_covered"]) != rows:
            return False
        cursor = 0
        for index, shard in enumerate(shards):
            first = int(shard["first_output_row"])
            end = int(shard["last_output_row_exclusive"])
            if int(shard["worker_id"]) != index or first != cursor or end < first or (index < 3 and (first % 4 or end % 4)):
                return False
            cursor = end
        if cursor != rows:
            return False
    return True


def _attempt02_shard_cpu_set_id(shard: dict[str, Any]) -> int:
    """Accept both the sealed attempt_02 key and its legacy report spelling."""
    if "cpu_set_id" in shard:
        return int(shard["cpu_set_id"])
    if "windows_cpu_set_id" in shard:
        return int(shard["windows_cpu_set_id"])
    raise KeyError("attempt_02 shard lacks cpu_set_id/windows_cpu_set_id")


def _attempt02_shard_map(manifest: dict[str, Any]) -> dict[tuple[int, str], list[tuple[int, int, int]]]:
    return {
        (int(matrix["d"]), matrix["matrix"]): [
            (_attempt02_shard_cpu_set_id(shard), int(shard["first_output_row"]), int(shard["last_output_row_exclusive"]))
            for shard in matrix["shards"]
        ]
        for matrix in manifest["matrices"]
    }


def core_selection_stage() -> dict[str, Any]:
    root = u.CORE_SELECTION_RESULTS
    if root.exists():
        raise FileExistsError(f"core-selection preflight is immutable: {root}")
    binding_record, _ = verify_stage(u.BINDING_PREFLIGHT_RESULTS)
    attempt_manifest, attempt_config, attempt_sha = u.validate_attempt02()
    _, _, c2_build, _ = u.validate_candidate02()
    if binding_record["stage_status"] != "PASS" or binding_record["attempt02_manifest_sha256"] != attempt_sha:
        raise RuntimeError("V2_1C_CORE_SELECTION_HOLD: binding stage or attempt_02 seal changed")
    if u.sha256_file(u.BENCH_EXE) != binding_record["companion_exe_sha256_frozen"]:
        raise RuntimeError("V2_1C_COMPANION_FROZEN_HASH_MISMATCH before H0 core selection")
    if u.source_hashes() != binding_record["v2_1c_source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source files changed after binding executable freeze")
    raw_path = u.PREFLIGHT_ROOT / "core_selection_preflight.json"
    process = subprocess.run([str(u.BENCH_EXE), "--core-selection-preflight", str(raw_path)], cwd=u.REPO_ROOT, capture_output=True, text=True, timeout=60 * 60)
    (u.PREFLIGHT_ROOT / "core_selection_stdout.log").write_text(process.stdout, encoding="utf-8", newline="\n")
    (u.PREFLIGHT_ROOT / "core_selection_stderr.log").write_text(process.stderr, encoding="utf-8", newline="\n")
    if process.returncode != 0 or not raw_path.is_file():
        raise RuntimeError(f"V2_1C_CORE_SELECTION_INVALID: companion H0 selection returned {process.returncode}: {process.stderr}")
    report = u.read_json(raw_path)
    if report.get("pass") is not True or report.get("status") != "PASS":
        raise RuntimeError("V2_1C_CORE_SELECTION_INVALID: native H0 report status is not PASS")
    test_report, selection = _core_selection_tests(report, attempt_config, attempt_sha)
    bench_hash_after = u.sha256_file(u.BENCH_EXE)
    same_companion = bench_hash_after == binding_record["companion_exe_sha256_frozen"]
    test_report["tests"].append({"name": "test_v2_1c_core_selection_same_frozen_companion_exe", "status": "PASS" if same_companion else "FAIL", "detail": {"frozen_sha256": binding_record["companion_exe_sha256_frozen"], "observed_sha256": bench_hash_after}})
    test_report["test_count"] += 1
    test_report["pass_count"] += int(same_companion)
    test_report["fail_count"] += int(not same_companion)
    source_hashes = u.source_hashes()
    record = {
        "phase": "CANDIDATE02_CORE_SELECTION_PREFLIGHT", "implementation_commit": binding_record["implementation_commit"],
        "stage_status": "PASS" if test_report["fail_count"] == 0 and test_report["skip_count"] == 0 else "CORE_SELECTION_INVALID",
        "binding_stage_artifact_manifest_sha256": u.sha256_file(u.BINDING_PREFLIGHT_RESULTS / "artifact_hashes.json"),
        "attempt02_manifest_sha256": attempt_sha, "attempt02_config_sha256": u.sha256_file(u.ATTEMPT02_ROOT / "benchmark_config.json"),
        "candidate02_kq_artifact_manifest_sha256": u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "companion_exe_abs": str(u.BENCH_EXE.resolve()), "companion_exe_sha256_frozen": binding_record["companion_exe_sha256_frozen"],
        "companion_exe_sha256_after_core_selection": bench_hash_after, "v2_1c_source_sha256": source_hashes,
        "candidate02_kernel_tu_sha256": {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()},
        "physical_dependency_sha256": {str(path.resolve()): u.sha256_file(path) for path in u.physical_dependency_paths()},
        "raw_core_selection_report": report, "core_selection_preflight": selection, "test_report": test_report,
        "timed_72_cell_sweep_started": False, "reselected_after_this_stage": False,
    }
    artifacts = [u.BENCH_EXE, c2_build and Path(c2_build["executable_absolute_path"]), raw_path, u.PREFLIGHT_ROOT / "core_selection_stdout.log", u.PREFLIGHT_ROOT / "core_selection_stderr.log",
        *(u.UNIT_ROOT / name for name in u.source_files()), *u.physical_dependency_paths(), *u.candidate02_kernel_paths(),
        u.BINDING_PREFLIGHT_RESULTS / "artifact_hashes.json", u.BINDING_PREFLIGHT_RESULTS / "V2_1C_BINDING_PREFLIGHT.md", u.ATTEMPT02_ROOT / "artifact_hashes.json", u.ATTEMPT02_ROOT / "benchmark_config.json", u.ATTEMPT02_ROOT / "worker_shard_manifest.json"]
    stage = seal_stage(root, record, test_report, artifacts, "V2_1C_CORE_SELECTION_PREFLIGHT.md", "# V2-1c CANDIDATE_02_CORE_SELECTION_PREFLIGHT")
    if record["stage_status"] != "PASS" or not stage["verified"]:
        raise RuntimeError("V2_1C_CORE_SELECTION_HOLD: core-selection preflight did not pass")
    return stage


def correctness_stage() -> dict[str, Any]:
    root = u.CORRECTNESS_PREFLIGHT_RESULTS
    if root.exists():
        raise FileExistsError(f"correctness preflight is immutable: {root}")
    binding, _ = verify_stage(u.BINDING_PREFLIGHT_RESULTS)
    selection_record, _ = verify_stage(u.CORE_SELECTION_RESULTS)
    selection = selection_record["core_selection_preflight"]
    ids = selection["selected_cpu_set_ids"]
    weights = selection["selected_v_i"]
    attempt_manifest, attempt_config, attempt_sha = u.validate_attempt02()
    _, c2_summary, c2_build, c2_hash = u.validate_candidate02()
    if binding["stage_status"] != "PASS" or selection_record["stage_status"] != "PASS":
        raise RuntimeError("V2_1C_CORRECTNESS_HOLD: binding/core-selection stage has not passed")
    if attempt_sha != binding["attempt02_manifest_sha256"] or attempt_sha != selection_record["attempt02_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest changed before correctness preflight")
    frozen_hash = binding["companion_exe_sha256_frozen"]
    if u.sha256_file(u.BENCH_EXE) != frozen_hash:
        raise RuntimeError("V2_1C_COMPANION_FROZEN_HASH_MISMATCH before correctness preflight")
    if u.source_hashes() != selection_record["v2_1c_source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source files changed after core-selection seal")
    payload, source_weights = u.kq_contract.v2_0_fp32_weight_payload()
    weight_file = u.PREFLIGHT_ROOT / "v2_0_fp32_source_weights_correctness.tmp"
    weight_file.write_bytes(payload)
    eq_path = u.PREFLIGHT_ROOT / "candidate02_correctness_preflight.json"
    equivalence = u.execute_equivalence("--correctness-preflight", weight_file, u.KQ2_OUTPUTS, eq_path, ids, weights)
    physical = u.execute_physical_correctness(ids, weights)
    expected_rows = [int(row["cpu_set_id"]) for row in selection["selected_workers"]]
    report_ids = [int(value) for value in physical.get("selected_cpu_set_ids", [])]
    report_weights = [float(value) for value in physical.get("selected_v_i", [])]
    physical_selection_ok = report_ids == expected_rows and len(report_weights) == 4 and all(math.isclose(a, b, rel_tol=0.0, abs_tol=0.0) for a, b in zip(report_weights, weights))
    binding_hash_ok = u.sha256_file(u.BENCH_EXE) == frozen_hash
    test_rows = [
        ("test_v2_1c_correctness_uses_new_core_selection", physical_selection_ok and equivalence.get("worker_cpu_set_ids") == expected_rows, {"selected_cpu_set_ids": expected_rows, "candidate02_equivalence_worker_ids": equivalence.get("worker_cpu_set_ids"), "physical_correctness_worker_ids": report_ids}),
        ("test_v2_1c_candidate02_kq_six_cell_scalar_and_repeat_correctness", equivalence.get("pass") is True and equivalence.get("six_cell_scalar_and_determinism_pass") is True, equivalence),
        ("test_v2_1c_candidate02_available_sealed_outputs_bit_exact", equivalence.get("kq_sealed_output_comparisons", {}).get("all_available_sealed_outputs_bit_exact") is True, equivalence.get("kq_sealed_output_comparisons", {})),
        ("test_v2_1c_physical_d512_d640_scalar_and_abc_correctness", physical.get("pass") is True and physical.get("all_d512_d640_scalar_pass") is True and physical.get("all_72_cell_abc_bit_exact") is True and physical.get("all_72_cell_repeats_bit_exact") is True, {key: physical.get(key) for key in ("pass", "all_d512_d640_scalar_pass", "all_72_cell_abc_bit_exact", "all_72_cell_repeats_bit_exact", "all_B_values_equal_A", "all_B_storage_disjoint", "worker_affinity_ok", "toy_recurrence_d32_m4")}),
        ("test_v2_1c_correctness_does_not_rebuild_frozen_companion", binding_hash_ok, {"frozen_sha256": frozen_hash, "observed_sha256": u.sha256_file(u.BENCH_EXE)}),
        ("test_v2_1c_attempt02_and_candidate02_kq_artifacts_unchanged", attempt_sha == binding["attempt02_manifest_sha256"] and c2_hash == u.FROZEN_CANDIDATE02_EXE_SHA256 and source_weights["source_weight_value_sha256"] == u.read_json(u.KQ2_NATIVE)["source_weight_sha256"], {"attempt02_manifest_sha256": attempt_sha, "candidate02_kq_exe_sha256": c2_hash, "source_weight_sha256": source_weights["source_weight_value_sha256"]}),
    ]
    tests = {"schema": "omega-v2-1c-correctness-test-report-v1", "test_count": len(test_rows), "pass_count": sum(ok for _, ok, _ in test_rows), "fail_count": sum(not ok for _, ok, _ in test_rows), "skip_count": 0, "tests": [{"name": name, "status": "PASS" if ok else "FAIL", "detail": detail} for name, ok, detail in test_rows]}
    status = "PASS" if tests["fail_count"] == 0 else "CORRECTNESS_INVALID"
    record = {
        "phase": "CORRECTNESS_PREFLIGHT_AFTER_NEW_CORE_SELECTION", "implementation_commit": binding["implementation_commit"], "stage_status": status,
        "attempt02_manifest_sha256": attempt_sha, "attempt02_config_sha256": u.sha256_file(u.ATTEMPT02_ROOT / "benchmark_config.json"),
        "candidate02_kq_exe_sha256": c2_hash, "candidate02_kq_artifact_manifest_sha256": u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "candidate02_build_manifest": c2_build, "candidate02_summary": c2_summary, "selected_cpu_set_ids": ids, "selected_v_i": weights,
        "candidate02_kq_correctness_preflight": equivalence, "physical_abc_correctness_preflight": physical,
        "source_weight_payload_sha256": source_weights["source_weight_value_sha256"], "companion_exe_sha256_frozen": frozen_hash,
        "test_report": tests, "timing_sanity_performed": False, "timed_72_cell_sweep_started": False,
    }
    artifacts = [eq_path, u.PREFLIGHT_ROOT / "correctness-preflight_stdout.log", u.PREFLIGHT_ROOT / "correctness-preflight_stderr.log",
        u.PREFLIGHT_ROOT / "physical_correctness_stdout.log", u.PREFLIGHT_ROOT / "physical_correctness_stderr.log", weight_file,
        u.BENCH_EXE, u.CORRECTNESS_EXE, Path(c2_build["executable_absolute_path"]), *(u.UNIT_ROOT / name for name in u.source_files()),
        *u.physical_dependency_paths(), *u.candidate02_kernel_paths(), u.BINDING_PREFLIGHT_RESULTS / "artifact_hashes.json",
        u.CORE_SELECTION_RESULTS / "artifact_hashes.json", u.CORE_SELECTION_RESULTS / "V2_1C_CORE_SELECTION_PREFLIGHT.md",
        u.ATTEMPT02_ROOT / "artifact_hashes.json", u.ATTEMPT02_ROOT / "benchmark_config.json", u.KQ2_ROOT_RESULTS / "artifact_hashes.json"]
    stage = seal_stage(root, record, tests, artifacts, "V2_1C_CORRECTNESS_PREFLIGHT.md", "# V2-1c CORRECTNESS_PREFLIGHT_AFTER_NEW_CORE_SELECTION")
    if status != "PASS" or not stage["verified"]:
        raise RuntimeError("V2_1C_CORRECTNESS_HOLD: correctness preflight did not pass")
    return stage


def seal_all_preflights() -> dict[str, Any]:
    root = u.PREFLIGHT_RESULTS
    if root.exists():
        raise FileExistsError(f"final V2-1c preflight is immutable: {root}")
    binding, binding_manifest = verify_stage(u.BINDING_PREFLIGHT_RESULTS)
    selection_record, selection_manifest = verify_stage(u.CORE_SELECTION_RESULTS)
    correctness, correctness_manifest = verify_stage(u.CORRECTNESS_PREFLIGHT_RESULTS)
    attempt_manifest, attempt_config, attempt_sha = u.validate_attempt02()
    c2_manifest, c2_summary, c2_build, c2_exe_hash = u.validate_candidate02()
    if not (binding["stage_status"] == selection_record["stage_status"] == correctness["stage_status"] == "PASS"):
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: one or more ordered stages did not pass")
    if attempt_sha != binding["attempt02_manifest_sha256"] or attempt_sha != selection_record["attempt02_manifest_sha256"] or attempt_sha != correctness["attempt02_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest differs across ordered preflight stages")
    selected = selection_record["core_selection_preflight"]
    ids = selected["selected_cpu_set_ids"]
    weights = selected["selected_v_i"]
    frozen_companion = binding["companion_exe_sha256_frozen"]
    if selection_record["companion_exe_sha256_after_core_selection"] != frozen_companion or correctness["companion_exe_sha256_frozen"] != frozen_companion or u.sha256_file(u.BENCH_EXE) != frozen_companion:
        raise RuntimeError("V2_1C_COMPANION_FROZEN_HASH_MISMATCH across ordered preflight stages")
    source_hashes = u.source_hashes()
    if source_hashes != binding["v2_1c_source_sha256"] or source_hashes != selection_record["v2_1c_source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source hashes differ across ordered preflight stages")
    physical_hashes = {str(path.resolve()): u.sha256_file(path) for path in u.physical_dependency_paths()}
    kernel_hashes = {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()}
    if physical_hashes != binding["physical_dependency_sha256"] or kernel_hashes != binding["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: frozen physical dependencies or candidate_02 compute changed")
    qpc_frequency_ok = int(selected["qpc_frequency"]) == int(attempt_config["qpc_frequency"])
    all_tests = [row for stage in (binding, selection_record, correctness) for row in stage["test_report"]["tests"]]
    all_tests.append({"name": "test_v2_1c_sealed_preflight_same_qpc_frequency_as_attempt02", "status": "PASS" if qpc_frequency_ok else "FAIL", "detail": {"attempt02_qpc_frequency": attempt_config["qpc_frequency"], "core_selection_qpc_frequency": selected["qpc_frequency"]}})
    tests = {"schema": "omega-v2-1c-ordered-preflight-test-report-v1", "test_count": len(all_tests), "pass_count": sum(row["status"] == "PASS" for row in all_tests), "fail_count": sum(row["status"] == "FAIL" for row in all_tests), "skip_count": sum(row["status"] == "SKIP" for row in all_tests), "tests": all_tests}
    provenance = binding["source_provenance"]
    record = {
        "phase": "V2_1C_PREFLIGHT_STEPS_1_TO_5_SEALED", "implementation_commit": binding["implementation_commit"],
        "source_provenance": provenance, "preflight_status": "PASS" if tests["fail_count"] == 0 and tests["skip_count"] == 0 else "PREFLIGHT_INVALID",
        "source_sha256": source_hashes, "v2_1c_source_sha256": source_hashes, "physical_dependency_sha256": physical_hashes,
        "candidate02_kernel_tu_sha256": kernel_hashes, "candidate02_artifact_manifest_sha256": u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "candidate02_build_manifest": c2_build, "candidate02_kernel_binding": binding["candidate02_kernel_binding"],
        "candidate02_exe_sha256": c2_exe_hash, "candidate02_manifest_sha256": u.sha256_file(u.KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "candidate02_kq_correctness_preflight": correctness["candidate02_kq_correctness_preflight"],
        "physical_abc_correctness_preflight": correctness["physical_abc_correctness_preflight"],
        "companion_binding_preflight": binding["binding_report"], "companion_binding_timing_sanity": binding["binding_timing_ratios"],
        "attempt02_artifact_manifest_sha256": attempt_sha, "attempt02_config": attempt_config,
        "attempt02_preservation": {"artifact_count": len(attempt_manifest["artifacts"]), "artifact_hashes_verified": True},
        "core_selection_preflight": selected, "selected_worker_cpu_set_ids": ids, "selected_v_i": weights,
        "performance_comparison_status": selected["attempt02_pairing_status"],
        "companion_bench_exe_sha256": frozen_companion, "companion_equivalence_exe_sha256": frozen_companion,
        "companion_correctness_exe_sha256": u.sha256_file(u.CORRECTNESS_EXE),
        "candidate02_kq_correctness_report_path_abs": str((u.PREFLIGHT_ROOT / "candidate02_correctness_preflight.json").resolve()),
        "physical_correctness_report_path_abs": str((u.PREFLIGHT_ROOT / "physical_correctness_preflight.json").resolve()),
        "companion_binding_report_path_abs": str((u.PREFLIGHT_ROOT / "binding_companion.json").resolve()),
        "companion_binding_timing_report_path_abs": str((u.PREFLIGHT_ROOT / "binding_timing_sanity.json").resolve()),
        "binding_stage_artifact_manifest_sha256": u.sha256_file(u.BINDING_PREFLIGHT_RESULTS / "artifact_hashes.json"),
        "core_selection_stage_artifact_manifest_sha256": u.sha256_file(u.CORE_SELECTION_RESULTS / "artifact_hashes.json"),
        "correctness_stage_artifact_manifest_sha256": u.sha256_file(u.CORRECTNESS_PREFLIGHT_RESULTS / "artifact_hashes.json"),
        "stage_manifests": {"binding": binding_manifest, "core_selection": selection_manifest, "correctness": correctness_manifest},
        "test_report": tests, "build_logs": binding["build_logs"], "cmake_path": binding["cmake_path"],
        "cmake_version": binding["cmake_version"], "visual_studio_installation": binding["visual_studio_installation"],
        "timed_72_cell_sweep_started": False, "sweep_go_received": False,
    }
    if tests["fail_count"] or tests["skip_count"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: at least one step 1-5 check did not pass")
    u.write_json(u.PREFLIGHT_ROOT / "sealed_preflight_record.json", record)
    u.write_json(u.PREFLIGHT_ROOT / "preflight_tests.json", tests)
    artifacts = [u.PREFLIGHT_ROOT / name for name in ("sealed_preflight_record.json", "preflight_tests.json", "binding_companion.json", "binding_timing_sanity.json", "candidate02_correctness_preflight.json", "physical_correctness_preflight.json", "core_selection_preflight.json" )]
    artifacts.extend([u.BENCH_EXE, u.CORRECTNESS_EXE, Path(c2_build["executable_absolute_path"]), u.V2_0_ROOT / "V2_0_RESULT_SEAL.json",
        u.ATTEMPT02_ROOT / "artifact_hashes.json", u.ATTEMPT02_ROOT / "benchmark_config.json", u.ATTEMPT02_ROOT / "hardware_preflight.json", u.ATTEMPT02_ROOT / "q4_physical_ledger.json", u.ATTEMPT02_ROOT / "worker_shard_manifest.json",
        u.KQ2_ROOT_RESULTS / "artifact_hashes.json", u.KQ2_NATIVE, u.KQ2_ROOT_RESULTS / "build_manifest.json", u.KQ2_ROOT_RESULTS / "summary_metrics.json",
        *(u.UNIT_ROOT / name for name in u.source_files()), *u.physical_dependency_paths(), *u.candidate02_kernel_paths(),
        u.BINDING_PREFLIGHT_RESULTS / "artifact_hashes.json", u.CORE_SELECTION_RESULTS / "artifact_hashes.json", u.CORRECTNESS_PREFLIGHT_RESULTS / "artifact_hashes.json"])
    record["source_provenance"]["v2_1c_source_sha256"] = source_hashes
    stage = seal_stage(root, record, tests, artifacts, "V2_1C_PREFLIGHT_REPORT.md", "# V2-1c PREFLIGHT STEPS 1–5")
    if record["preflight_status"] != "PASS" or not stage["verified"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: final steps 1-5 seal did not verify")
    return stage


def load_sealed_preflight() -> dict[str, Any]:
    record, manifest = verify_stage(u.PREFLIGHT_RESULTS)
    if record.get("preflight_status") != "PASS" or record.get("timed_72_cell_sweep_started") is not False:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: final preflight is not a passing no-sweep seal")
    if record.get("sweep_go_received") is not False:
        raise RuntimeError("V2_1C_STOP: GO is not encoded in the preflight seal")
    if u.source_hashes() != record["source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source changed after ordered preflight seal")
    if u.sha256_file(u.BENCH_EXE) != record["companion_bench_exe_sha256"] or u.sha256_file(u.CORRECTNESS_EXE) != record["companion_correctness_exe_sha256"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: companion executable hash changed after preflight")
    if u.sha256_file(Path(record["candidate02_build_manifest"]["executable_absolute_path"])) != u.FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 KQ executable changed")
    attempt_manifest, _, attempt_sha = u.validate_attempt02()
    if attempt_sha != record["attempt02_artifact_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 artifacts changed after preflight")
    return record


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def verify_offline_seals() -> dict[str, Any]:
    """Replay binding/H0 parsing and semantic checks using sealed JSON only; never launches a process."""
    output_root = SEALED_EVIDENCE_ROOT / "OFFLINE_SEAL_RECOVERY_VERIFICATION"
    if output_root.exists():
        if output_root.is_dir() and not any(output_root.iterdir()):
            output_root.rmdir()
        else:
            raise FileExistsError(f"offline seal verification is immutable: {output_root}")

    copied_preflight = SEALED_EVIDENCE_ROOT / "build_preflight"
    binding_input = copied_preflight / "binding_companion.json"
    core_input = copied_preflight / "core_selection_preflight.json"
    native_binding = u.read_json(binding_input)
    native_core_selection = u.read_json(core_input)
    kq_native = u.read_json(u.KQ2_NATIVE)
    attempt_config = u.read_json(u.ATTEMPT02_ROOT / "benchmark_config.json")
    attempt_sha = u.sha256_file(u.ATTEMPT02_ROOT / "artifact_hashes.json")

    binding_stage_root = SEALED_EVIDENCE_ROOT / "binding_preflight"
    binding_record = u.read_json(binding_stage_root / "stage_record.json")
    binding_tests = u.read_json(binding_stage_root / "test_report.json")
    binding_report_path = binding_stage_root / "V2_1C_BINDING_PREFLIGHT.md"
    binding_manifest_path = binding_stage_root / "artifact_hashes.json"
    binding_report_sha = u.sha256_file(binding_report_path)
    binding_manifest_sha = u.sha256_file(binding_manifest_path)
    binding_ratios = u.compare_binding_timing(native_binding, kq_native)
    binding_value_sha = _semantic_sha256(binding_ratios)
    sealed_binding_value_sha = _semantic_sha256(binding_record["binding_timing_ratios"])
    binding_inputs_match = (
        u.sha256_file(binding_input) == "e7637df1a32ef5a33af3309abdaa816ef3222e9b1da91d0b0871877c651e522a"
        and u.sha256_file(u.KQ2_NATIVE) == "230dce1ccae58696f38610077459e2c6781b749330298b4617df9b3561da7b6c"
    )
    binding_hashes_match = (
        binding_report_sha == "478f985238f42babb9bfd33d64f6578a4441e9989b01957ceb7c173ad9759191"
        and binding_manifest_sha == "1ffc581ba5e7c3c49d3e3fe4aad22bbdb2257beb2b5afcbe92d8b92e5221a74d"
    )
    binding_semantics_match = (
        binding_value_sha == sealed_binding_value_sha
        and binding_ratios == binding_record["binding_timing_ratios"]
        and native_binding["pass"] is True
        and binding_ratios["pass"] is True
        and binding_tests["fail_count"] == 0
        and binding_tests["skip_count"] == 0
    )

    core_stage_root = SEALED_EVIDENCE_ROOT / "core_selection_preflight"
    core_record = u.read_json(core_stage_root / "stage_record.json")
    core_tests = u.read_json(core_stage_root / "test_report.json")
    core_report_path = core_stage_root / "V2_1C_CORE_SELECTION_PREFLIGHT.md"
    core_manifest_path = core_stage_root / "artifact_hashes.json"
    core_report_sha = u.sha256_file(core_report_path)
    core_manifest_sha = u.sha256_file(core_manifest_path)
    attempt_manifest = u.verify_artifact_manifest(u.ATTEMPT02_ROOT / "artifact_hashes.json")
    recomputed_core_tests, recomputed_selection = _core_selection_tests(native_core_selection, attempt_config, attempt_sha)
    frozen_companion_sha = u.sha256_file(u.BENCH_EXE)
    same_companion = frozen_companion_sha == core_record["companion_exe_sha256_frozen"]
    recomputed_core_tests["tests"].append({"name": "test_v2_1c_core_selection_same_frozen_companion_exe", "status": "PASS" if same_companion else "FAIL", "detail": {"frozen_sha256": core_record["companion_exe_sha256_frozen"], "observed_sha256": frozen_companion_sha}})
    recomputed_core_tests["test_count"] += 1
    recomputed_core_tests["pass_count"] += int(same_companion)
    recomputed_core_tests["fail_count"] += int(not same_companion)
    recomputed_selection = _json_normalize(recomputed_selection)
    recomputed_core_tests = _json_normalize(recomputed_core_tests)
    core_value_sha = _semantic_sha256({"selection": recomputed_selection, "tests": recomputed_core_tests})
    sealed_core_value_sha = _semantic_sha256({"selection": core_record["core_selection_preflight"], "tests": core_tests})
    core_inputs_match = (
        u.sha256_file(core_input) == "e0ca7b7bffae9242ade1b5b5981e82d5b26c68c0be5f18e9b341bae04e5a08a4"
        and attempt_sha == "d17d5b79d43fe72eba644069e72b8539ce6990032579313d953190a7313f5287"
    )
    core_hashes_match = (
        core_report_sha == "09faba6e82a632bd6e2ac4a6e78c2d75de5cbc520709868828f464df1673ca02"
        and core_manifest_sha == "2dbf73d67eeb0e897fb301a21251275da28495a8533321279d50353bc653a7b7"
    )
    core_semantics_match = (
        core_value_sha == sealed_core_value_sha
        and recomputed_selection == core_record["core_selection_preflight"]
        and recomputed_core_tests == core_tests
        and native_core_selection["pass"] is True
        and core_tests["fail_count"] == 0
        and core_tests["skip_count"] == 0
    )

    verification = {
        "schema": "omega-v2-1c-offline-seal-recovery-verification-v1",
        "status": "PASS" if all((binding_inputs_match, binding_hashes_match, binding_semantics_match, core_inputs_match, core_hashes_match, core_semantics_match)) else "FAIL",
        "binding": {
            "input_json_sha256": u.sha256_file(binding_input),
            "kq_baseline_json_sha256": u.sha256_file(u.KQ2_NATIVE),
            "recomputed_timing_ratios": binding_ratios,
            "recomputed_semantic_values_sha256": binding_value_sha,
            "sealed_semantic_values_sha256": sealed_binding_value_sha,
            "semantic_values_match": binding_semantics_match,
            "sealed_report_sha256_expected": "478f985238f42babb9bfd33d64f6578a4441e9989b01957ceb7c173ad9759191",
            "sealed_report_sha256_observed": binding_report_sha,
            "sealed_manifest_sha256_expected": "1ffc581ba5e7c3c49d3e3fe4aad22bbdb2257beb2b5afcbe92d8b92e5221a74d",
            "sealed_manifest_sha256_observed": binding_manifest_sha,
            "input_hashes_match": binding_inputs_match,
            "seal_hashes_match": binding_hashes_match,
        },
        "core_selection": {
            "input_json_sha256": u.sha256_file(core_input),
            "attempt02_worker_shard_manifest_sha256": u.sha256_file(u.ATTEMPT02_ROOT / "worker_shard_manifest.json"),
            "recomputed_selected_cpu_set_ids": recomputed_selection["selected_cpu_set_ids"],
            "recomputed_selected_v_i": recomputed_selection["selected_v_i"],
            "recomputed_semantic_values_sha256": core_value_sha,
            "sealed_semantic_values_sha256": sealed_core_value_sha,
            "semantic_values_match": core_semantics_match,
            "sealed_report_sha256_expected": "09faba6e82a632bd6e2ac4a6e78c2d75de5cbc520709868828f464df1673ca02",
            "sealed_report_sha256_observed": core_report_sha,
            "sealed_manifest_sha256_expected": "2dbf73d67eeb0e897fb301a21251275da28495a8533321279d50353bc653a7b7",
            "sealed_manifest_sha256_observed": core_manifest_sha,
            "input_hashes_match": core_inputs_match,
            "seal_hashes_match": core_hashes_match,
        },
        "exe_invocations": 0,
        "measurements_repeated": False,
        "original_seals_modified": False,
        "attempt02_artifact_manifest_sha256": u.sha256_file(u.ATTEMPT02_ROOT / "artifact_hashes.json"),
        "frozen_companion_exe_sha256_observed_by_hash_only": frozen_companion_sha,
        "attempt02_artifact_count": len(attempt_manifest["artifacts"]),
        "correctness_failure_terminal_classification": "V2_1C_INVALID_PREFLIGHT",
    }
    output_root.mkdir(parents=True)
    verification_path = output_root / "verification.json"
    u.write_json(verification_path, verification)
    lines = ["# OFFLINE_SEAL_RECOVERY_VERIFICATION", "", f"- status: `{verification['status']}`", "- executable invocations: `0`", "- measurements repeated: `false`", "- original seals modified: `false`", "", "## Binding replay", f"- semantic values match: `{binding_semantics_match}`", f"- original report SHA-256 match: `{binding_report_sha == verification['binding']['sealed_report_sha256_expected']}`", f"- original manifest SHA-256 match: `{binding_manifest_sha == verification['binding']['sealed_manifest_sha256_expected']}`", "", "## Core-selection replay", f"- semantic values match: `{core_semantics_match}`", f"- original report SHA-256 match: `{core_report_sha == verification['core_selection']['sealed_report_sha256_expected']}`", f"- original manifest SHA-256 match: `{core_manifest_sha == verification['core_selection']['sealed_manifest_sha256_expected']}`", f"- selected CPU-set IDs: `{recomputed_selection['selected_cpu_set_ids']}`", ""]
    report_path = output_root / "OFFLINE_SEAL_RECOVERY_VERIFICATION.md"
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    report_sha = u.sha256_file(report_path)
    sidecar = output_root / "OFFLINE_SEAL_RECOVERY_VERIFICATION.md.sha256"
    sidecar.write_text(report_sha + "\n", encoding="ascii", newline="\n")
    inputs = [binding_input, core_input, u.KQ2_NATIVE, u.ATTEMPT02_ROOT / "benchmark_config.json", u.ATTEMPT02_ROOT / "worker_shard_manifest.json",
        binding_stage_root / "stage_record.json", binding_stage_root / "test_report.json", binding_report_path, binding_manifest_path,
        core_stage_root / "stage_record.json", core_stage_root / "test_report.json", core_report_path, core_manifest_path,
        u.BENCH_EXE, u.CORRECTNESS_EXE, *(u.UNIT_ROOT / path for path in binding_record["v2_1c_source_sha256"]),
        *(Path(path) for path in binding_record["physical_dependency_sha256"]), *(Path(path) for path in binding_record["candidate02_kernel_tu_sha256"])]
    unique = {str(path.resolve()): path for path in [*inputs, verification_path, report_path, sidecar] if path.is_file()}
    artifact_manifest = {"schema": "omega-v2-1c-offline-seal-recovery-artifact-hashes-v1", "verification_status": verification["status"], "report_self_sha256": report_sha, "artifacts": {name: {"sha256": u.sha256_file(path), "size_bytes": path.stat().st_size} for name, path in unique.items()}}
    artifact_manifest_path = output_root / "artifact_hashes.json"
    u.write_json(artifact_manifest_path, artifact_manifest)
    hashes_verified = all(Path(name).is_file() and u.sha256_file(Path(name)) == row["sha256"] for name, row in artifact_manifest["artifacts"].items()) and u.sha256_file(report_path) == sidecar.read_text(encoding="ascii").strip()
    u.write_json(output_root / "artifact_hashes_verified.json", {"verified": hashes_verified, "artifact_count": len(artifact_manifest["artifacts"])})
    if not hashes_verified or verification["status"] != "PASS":
        raise RuntimeError("V2_1C_OFFLINE_SEAL_RECOVERY_VERIFICATION_FAILED")
    return {"verification": verification, "report_path": report_path, "report_sha256": report_sha, "artifact_manifest_path": artifact_manifest_path, "artifact_manifest_sha256": u.sha256_file(artifact_manifest_path), "artifact_count": len(artifact_manifest["artifacts"]), "hashes_verified": hashes_verified}


def print_stage(stage: dict[str, Any], label: str) -> None:
    record = stage["record"]
    print(json.dumps({"phase": label, "stage_status": record.get("stage_status", record.get("preflight_status")), "report_abs": str(stage["report_path"].resolve()), "report_sha256": stage["report_sha256"], "artifact_manifest_abs": str(stage["manifest_path"].resolve()), "artifact_manifest_sha256": stage["manifest_sha256"], "artifact_count": len(stage["manifest"]["artifacts"]), "hashes_verified": stage["verified"], "timed_72_cell_sweep_started": False}, indent=2, sort_keys=True))


def main() -> int:
    parser = u.argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="build the companion and run static checks without timing")
    mode.add_argument("--binding-preflight-only", action="store_true", help="run the authorized 3-cell companion binding preflight with the prepared executable, then freeze its hash")
    mode.add_argument("--core-selection-preflight-only", action="store_true", help="run the authorized one-time candidate_02 H0 v_i selection after binding is frozen")
    mode.add_argument("--correctness-preflight-only", action="store_true", help="run no-timing correctness only after new candidate_02 core selection is sealed")
    mode.add_argument("--seal-preflight-only", action="store_true", help="verify and seal steps 1-5; no timed 72-cell sweep")
    mode.add_argument("--verify-offline-seals", action="store_true", help="replay binding/H0 parsers over sealed JSON copies without invoking executables")
    mode.add_argument("--run-sweep", action="store_true", help="run 72 cells only with explicit judge GO medicion")
    parser.add_argument("--go-medicion", action="store_true", help="explicit acknowledgement of judge GO medicion; required with --run-sweep")
    args = parser.parse_args()
    if u.os.name != "nt" or u.sys.platform != "win32":
        raise RuntimeError("V2_1C_MEASUREMENT_INVALID: native Windows is required")
    if args.verify_offline_seals:
        result = verify_offline_seals()
        print(json.dumps({"phase": "OFFLINE_SEAL_RECOVERY_VERIFICATION", "status": result["verification"]["status"], "report_abs": str(result["report_path"].resolve()), "report_sha256": result["report_sha256"], "artifact_manifest_abs": str(result["artifact_manifest_path"].resolve()), "artifact_manifest_sha256": result["artifact_manifest_sha256"], "artifact_count": result["artifact_count"], "hashes_verified": result["hashes_verified"], "exe_invocations": 0, "measurements_repeated": False}, indent=2, sort_keys=True))
        return 0
    if args.prepare_only:
        record = prepare_companion()
        print(json.dumps({"phase": "NO_TIMING_PREPARATION_ONLY", "companion_bench_exe_abs": record["companion_bench_exe_abs"], "companion_bench_exe_sha256": record["companion_bench_exe_sha256"], "companion_correctness_exe_sha256": record["companion_correctness_exe_sha256"], "static_contract_tests": "PASS", "timed_binding_started": False, "core_selection_started": False, "sweep_started": False, "branch": record["branch"], "git_status_unit": record["git_status_unit"]}, indent=2, sort_keys=True))
        return 0
    if args.binding_preflight_only:
        stage = binding_stage()
        print_stage(stage, "COMPANION_BINDING_PREFLIGHT")
        return 0
    if args.core_selection_preflight_only:
        stage = core_selection_stage()
        print_stage(stage, "CANDIDATE02_CORE_SELECTION_PREFLIGHT")
        return 0
    if args.correctness_preflight_only:
        stage = correctness_stage()
        print_stage(stage, "CORRECTNESS_PREFLIGHT_AFTER_NEW_CORE_SELECTION")
        return 0
    if args.seal_preflight_only:
        stage = seal_all_preflights()
        print_stage(stage, "V2_1C_PREFLIGHT_STEPS_1_TO_5")
        return 0
    if not args.go_medicion:
        raise RuntimeError("V2_1C_STOP: --run-sweep requires the judge's explicit GO medicion")
    if u.RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-1c attempt_01 result slot already exists and is immutable: {u.RESULTS_ROOT}")
    record = load_sealed_preflight()
    attempt_manifest, attempt_config, attempt_sha = u.validate_attempt02()
    attempt_before = {"artifact_manifest_path_abs": str((u.ATTEMPT02_ROOT / "artifact_hashes.json").resolve()), "artifact_manifest_sha256": attempt_sha, "artifact_count": len(attempt_manifest["artifacts"]), "artifact_hashes_verified_before_v2_1c": True, "frozen_worker_cpu_set_ids": [row["windows_cpu_set_id"] for row in attempt_config["selected_workers"]], "frozen_h0_shard_weights": [row["h0_v_i"] for row in attempt_config["selected_workers"]], "frozen_worker_logical_ids": [row["logical_processor_id"] for row in attempt_config["selected_workers"]]}
    preflight = {"preflight_record": record, "source_provenance": record["source_provenance"], "source_sha256": record["source_sha256"], "physical_dependency_sha256": record["physical_dependency_sha256"], "candidate02_kernel_tu_sha256": record["candidate02_kernel_tu_sha256"], "candidate02_manifest_sha256": record["candidate02_artifact_manifest_sha256"], "candidate02_build_manifest": record["candidate02_build_manifest"], "bench_executable_sha256": record["companion_bench_exe_sha256"], "equivalence_executable_sha256": record["companion_equivalence_exe_sha256"], "correctness_executable_sha256": record["companion_correctness_exe_sha256"], "build_logs": record["build_logs"], "cmake_path": record["cmake_path"], "cmake_version": record["cmake_version"], "visual_studio_installation": record["visual_studio_installation"], "candidate02_manifest_sha256": record["candidate02_artifact_manifest_sha256"]}
    u.run_native_sweep(preflight, attempt_before)
    attempt_after = u.attempt02_preservation_after(attempt_before)
    if u.source_hashes() != record["source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: V2-1c source changed during timed physical sweep")
    if u.sha256_file(Path(record["candidate02_build_manifest"]["executable_absolute_path"])) != u.FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_STOP: candidate_02 KQ executable hash changed during sweep")
    if {str(path.resolve()): u.sha256_file(path) for path in u.candidate02_kernel_paths()} != record["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_STOP: candidate_02 compute TUs changed during sweep")
    summary = u.finalize_results(preflight, attempt_before)
    summary["sweep_go_acknowledged"] = True
    u.write_json(u.RESULTS_ROOT / "summary_metrics.json", summary)
    seal = u.seal_results(summary)
    print(json.dumps({"unit": "OMEGA-V2-1c-RESIDENCY-ONLY", "terminal_status": summary["terminal_status"], "gate_metrics": summary.get("gate_metrics"), "test_count": summary["test_report"]["test_count"], "test_pass_count": summary["test_report"]["pass_count"], "test_fail_count": summary["test_report"]["fail_count"], "test_skip_count": summary["test_report"]["skip_count"], "results_root_abs": str(u.RESULTS_ROOT.resolve()), "report_sha256": seal["report_sha256"], "artifact_manifest_sha256": seal["artifact_manifest_sha256"], "artifact_count": seal["artifact_count"], "artifact_hashes_verified": seal["verified"], "attempt03_run": False, "pytorch_s_native_gate": "EXCLUDED"}, indent=2, sort_keys=True))
    return 0 if summary["terminal_status"] == "OMEGA_V2_1C_RESIDENCY_ONLY_PASS" and seal["verified"] else 1
