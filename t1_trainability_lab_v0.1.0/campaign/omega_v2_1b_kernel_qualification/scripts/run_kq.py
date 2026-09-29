"""Build and run the isolated V2-1b KQ candidate; never runs A/B/C or attempt_03."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
CAMPAIGN_ROOT = UNIT_ROOT.parent
V2_1_ROOT = CAMPAIGN_ROOT / "omega_v2_1_physical"
V2_0_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
RESULTS_BASE = UNIT_ROOT / "results" / "omega_v2_1b_kernel_qualification"
BUILD_ROOT = UNIT_ROOT / "build"
EXECUTABLE = BUILD_ROOT / "Release" / "omega_v2_1b_kq.exe"
V2_0_SEAL = V2_0_ROOT / "V2_0_RESULT_SEAL.json"
ATTEMPT02_ROOT = V2_1_ROOT / "results" / "omega_v2_1_physical" / "attempt_02"
AUTHORIZED_SNAPSHOT = "38061477d4c2b0c5c20d75d902b21dd1ef0a2611"
V2_0_VALIDATED_HEAD = "5df1cf270837a9cc4f56a3dd31f277af868f61ce"
V2_0_IMPLEMENTATION_COMMIT = "bc9c1745f70e08f15fd340097d04fff6f56cba43"
CONVERSACION_MD_BLOB = "7027e2ac9d1ba89db08dda73c81e244f3b9b19db"
CONVERSACION_LN_BLOB = "fc750a2ae9fb7d3933c54fb91f09ce5568d035ec"
FROZEN_CPU_SET_IDS = [266, 264, 258, 270]
MATRIX_FAMILIES = ("W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down")
KQ_TEST_NAMES = [
    "test_v2_1b_vectorized_q4_matches_scalar",
    "test_v2_1b_dequant_tile_reused_across_slots",
    "test_v2_1b_no_full_predequantized_weight_copy",
    "test_v2_1b_scratch_per_worker_le_64k",
    "test_v2_1b_fma_peak_fixture",
    "test_v2_1b_h0_efficiency_formula",
    "test_v2_1b_full_efficiency_formula",
    "test_v2_1b_kernel_quality_thresholds",
    "test_v2_1b_pytorch_reference_same_equations",
    "test_v2_1b_pytorch_speedup_formula",
    "test_v2_1b_candidate_limit_two",
    "test_v2_1b_no_abc_before_kernel_freeze",
    "test_v2_1b_attempt02_preserved",
    "test_v2_1b_attempt03_same_72_cells",
    "test_v2_1b_m1_residency_diagnostic",
    "test_v2_1b_machine_balance_diagnostic",
    "test_v2_1b_conformance_block",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def classify_candidate1(*, correctness: bool, e_q4: float, e_full4: float, e_full16: float, s_native: float) -> dict[str, Any]:
    scientific = correctness and e_q4 >= 0.25 and e_full4 >= 0.60 and e_full16 >= 0.60
    project_speed = s_native >= 1.20
    all_five = scientific and project_speed
    if all_five:
        status = "FREEZE_CANDIDATE_1_ATTEMPT_03_ALLOWED"
        candidate2_allowed = False
    elif scientific and not project_speed:
        status = "KERNEL_SCIENTIFICALLY_QUALIFIED_PROJECT_NATIVE_SPEED_GATE_FAIL_CANDIDATE_2_ALLOWED"
        candidate2_allowed = True
    else:
        status = "CANDIDATE_1_NOT_QUALIFIED_CANDIDATE_2_ALLOWED"
        candidate2_allowed = True
    return {
        "scientifically_qualified": scientific,
        "project_speed_pass": project_speed,
        "all_five_gates_pass": all_five,
        "candidate2_allowed": candidate2_allowed,
        "terminal_status": status,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def source_files() -> list[str]:
    files = ["OMEGA_V2_1B_KQ_SPEC.md", "CMakeLists.txt", ".gitignore"]
    files.extend(f"src/{path.name}" for path in sorted((UNIT_ROOT / "src").glob("*.cpp")))
    files.extend(f"src/{path.name}" for path in sorted((UNIT_ROOT / "src").glob("*.hpp")))
    files.extend(f"scripts/{path.name}" for path in sorted((UNIT_ROOT / "scripts").glob("*.py")))
    files.extend(f"tests/{path.name}" for path in sorted((UNIT_ROOT / "tests").glob("*.py")))
    return files


def dependency_files() -> list[Path]:
    old_src = V2_1_ROOT / "src"
    return [old_src / name for name in (
        "v2_1.hpp", "hardware_topology.cpp", "q4_layout.cpp", "v2_full_block.cpp",
        "worker_pool.cpp", "cache_controls.cpp", "allocation_guard.cpp",
    )]


def source_provenance() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    merge_base = git("merge-base", "HEAD", AUTHORIZED_SNAPSHOT)
    unit_status = git("status", "--porcelain", "--", str(UNIT_ROOT.relative_to(REPO_ROOT)))
    previous_unit_status = git("status", "--porcelain", "--", str(V2_1_ROOT.relative_to(REPO_ROOT)))
    md_blob = git("rev-parse", "HEAD:Conversacion.md")
    ln_blob = git("rev-parse", "HEAD:Conversacion LN.md")
    if branch != "main" or merge_base != AUTHORIZED_SNAPSHOT or unit_status or previous_unit_status:
        raise RuntimeError(
            f"KQ_SOURCE_HOLD: expected clean KQ/attempt02 sources on main; branch={branch},"
            f"merge_base={merge_base},kq_status={unit_status!r},attempt02_unit_status={previous_unit_status!r}"
        )
    if md_blob != CONVERSACION_MD_BLOB or ln_blob != CONVERSACION_LN_BLOB:
        raise RuntimeError("KQ_SOURCE_HOLD: authoritative conversation blobs differ from the sealed attempt")
    return {
        "repository": "Hackerprod/Ejercise",
        "branch": branch,
        "implementation_commit": commit,
        "implementation_parent": parents[0] if parents else None,
        "git_status_porcelain": git("status", "--porcelain"),
        "kq_unit_status": unit_status,
        "attempt02_unit_status": previous_unit_status,
        "authorized_snapshot_commit": AUTHORIZED_SNAPSHOT,
        "merge_base": merge_base,
        "conversacion_md_blob": md_blob,
        "conversacion_ln_blob": ln_blob,
    }


def validate_v2_0_seal() -> dict[str, Any]:
    seal = json.loads(V2_0_SEAL.read_text(encoding="utf-8"))
    if seal.get("terminal_status") != "OMEGA_V2_0_CONFORMANT_PASS" or seal.get("execution_commit") != V2_0_IMPLEMENTATION_COMMIT:
        raise RuntimeError("KQ_SOURCE_HOLD: validated V2-0 seal status/commit mismatch")
    sealed_results = V2_0_ROOT / "results" / "omega_v2_0_conformance"
    observed = {}
    for name, expected in seal["artifact_sha256"].items():
        path = sealed_results / name
        actual = sha256_file(path)
        observed[name] = actual
        if actual != expected:
            raise RuntimeError(f"KQ_SOURCE_HOLD: V2-0 sealed artifact hash mismatch: {name}")
    return {"seal": seal, "observed_artifact_sha256": observed}


def validate_attempt02_preservation() -> dict[str, Any]:
    manifest_path = ATTEMPT02_ROOT / "artifact_hashes.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for absolute_path, record in manifest["artifacts"].items():
        path = Path(absolute_path)
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            mismatches.append(absolute_path)
    if mismatches:
        raise RuntimeError(f"KQ_STOP: attempt_02 immutable artifact hash mismatch: {mismatches}")
    # Read only the worker-affinity configuration, not attempt_02 scientific metrics/rho.
    config = json.loads((ATTEMPT02_ROOT / "benchmark_config.json").read_text(encoding="utf-8"))
    workers = config["selected_workers"]
    ids = [int(row["windows_cpu_set_id"]) for row in workers]
    if ids != FROZEN_CPU_SET_IDS:
        raise RuntimeError(f"KQ_STOP: attempt_02 CPU-set order changed: {ids}")
    weights = [float(row["h0_v_i"]) for row in workers]
    logical_ids = [int(row["logical_processor_id"]) for row in workers]
    return {
        "artifact_manifest_path_abs": str(manifest_path.resolve()),
        "artifact_manifest_sha256": manifest_hash,
        "artifact_hashes_verified_before_kq": True,
        "artifact_count": len(manifest["artifacts"]),
        "implementation_commit": manifest["implementation_commit"],
        "frozen_worker_cpu_set_ids": ids,
        "frozen_worker_logical_ids": logical_ids,
        "frozen_h0_shard_weights": weights,
    }


def find_vswhere() -> Path:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")), Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    for root in roots:
        candidate = root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("vswhere.exe not found")


def find_msvc_cmake(vswhere: Path) -> tuple[Path, str, str]:
    vs_paths = subprocess.run(
        [str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    cmake_paths = subprocess.run(
        [str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.CMake.Project", "-find", r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    if not vs_paths or not cmake_paths:
        raise FileNotFoundError("MSVC x64 compiler or Visual Studio CMake component missing")
    cmake = Path(cmake_paths[0])
    version = subprocess.run([str(cmake), "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return cmake, vs_paths[0], version


def build_native(cmake: Path) -> tuple[Path, dict[str, str]]:
    configure = subprocess.run(
        [str(cmake), "-S", str(UNIT_ROOT), "-B", str(BUILD_ROOT), "-G", "Visual Studio 17 2022", "-A", "x64"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if configure.returncode:
        raise RuntimeError("KQ CMake configure failed: " + configure.stderr[-4000:])
    build = subprocess.run(
        [str(cmake), "--build", str(BUILD_ROOT), "--config", "Release", "--target", "omega_v2_1b_kq", "-j", "8"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if build.returncode:
        raise RuntimeError("KQ MSVC Release build failed: " + build.stderr[-5000:])
    if not EXECUTABLE.is_file():
        raise FileNotFoundError("KQ Release executable missing")
    return EXECUTABLE, {
        "configure_stdout": configure.stdout, "configure_stderr": configure.stderr,
        "build_stdout": build.stdout, "build_stderr": build.stderr,
    }


def file_hashes() -> dict[str, str]:
    return {name: sha256_file(UNIT_ROOT / name) for name in source_files()}


def v2_0_fp32_weight_payload() -> tuple[bytes, dict[str, Any]]:
    sys.path.insert(0, str(V2_0_ROOT))
    import torch
    from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed = 20260929
    block = ContractualCoreBlock(512, dtype=torch.float32, seed=seed)
    payload_parts = []
    value_digest = hashlib.sha256()
    matrix_metadata = []
    for name in MATRIX_FAMILIES:
        tensor = getattr(block, name).detach().contiguous().cpu()
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        payload_parts.append(raw)
        value_digest.update(name.encode("utf-8"))
        value_digest.update(b"\0")
        value_digest.update(raw)
        matrix_metadata.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": len(raw)})
    payload = b"".join(payload_parts)
    return payload, {
        "source": "V2-0 omega_v2.core.ContractualCoreBlock FP32 tensors before Q4",
        "source_path_abs": str((V2_0_ROOT / "omega_v2" / "core.py").resolve()),
        "seed": seed,
        "d": 512,
        "matrix_families": matrix_metadata,
        "source_weight_value_sha256": value_digest.hexdigest(),
        "source_weight_stream_sha256": sha256_bytes(payload),
        "source_weight_stream_bytes": len(payload),
    }


def sample_groups(raw_path: Path, *, d: int, m: int, variant: str) -> dict[int, list[float]]:
    grouped: dict[tuple[int, int, int, bool], list[dict[str, str]]] = {}
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["d"]) != d or int(row["m"]) != m or row["variant"] != variant:
                continue
            if row["is_warmup"].lower() == "true":
                continue
            key = (int(row["K"]), int(row["block_id"]), int(row["sample_id"]), row["valid"].lower() == "true")
            grouped.setdefault(key, []).append(row)
    result: dict[int, list[float]] = {1: [], 8: []}
    for (k, _block, _sample, valid), rows in grouped.items():
        if not valid or k not in result or len(rows) != k:
            continue
        result[k].append(sum(float(row["qpc_seconds"]) for row in rows))
    return result


def attempt02_m1_diagnostics() -> dict[str, Any]:
    raw_path = ATTEMPT02_ROOT / "raw_measurements.csv"
    rows = {}
    for d in (512, 640):
        by_variant = {}
        for variant in ("A", "B", "C"):
            samples = sample_groups(raw_path, d=d, m=1, variant=variant)
            if len(samples[1]) != 105 or len(samples[8]) != 105:
                raise RuntimeError(f"attempt_02 m1 diagnostic is incomplete for d{d}/{variant}")
            t1, t8 = statistics.median(samples[1]), statistics.median(samples[8])
            by_variant[variant] = {"T1_median_seconds": t1, "T8_median_seconds": t8, "c8_seconds_per_round": (t8 - t1) / 7.0}
        a, b, c = (by_variant[key]["c8_seconds_per_round"] for key in ("A", "B", "C"))
        rows[str(d)] = {
            "variants": by_variant,
            "rho_m1_A_over_B_diagnostic_only": a / b,
            "delta_CB_m1_diagnostic_only": abs(c - b) / b,
            "c_C_gt_c_A_diagnostic_only": c > a,
            "gate": False,
            "source": "attempt_02 raw_measurements.csv; no m4/rho_resident fields read or calculated",
        }
    return rows


def make_test_rows(
    native: dict[str, Any], pytorch: dict[str, Any], q4_gate: bool, full4_gate: bool, full16_gate: bool,
    speed_gate: bool, attempt02_before: dict[str, Any], attempt02_after: dict[str, Any],
    m1_diag: dict[str, Any], source: dict[str, Any], candidate_id: str,
) -> list[dict[str, Any]]:
    q4_scalar_pass = native["correctness"]["toy_d32_m4_k2"]["pass"] is True
    tile = native["dequant_tile_reuse"]["Q_W_m16"]
    no_copy = native["persistent_weight_format"] == "signed symmetric Q4 + FP16 group32 scales only" and native["predequantized_weight_copy_persistent"] is False
    scratch_pass = native["scratch_per_worker_bytes"] <= 64 * 1024
    fma_pass = native["fma_l1"]["series"]["median_macs_per_second"] > 0 and native["fma_l1"]["tile_fits_l1d"]
    h0_formula_pass = abs(native["efficiencies"]["E_Q4"] - native["h0_q4"]["m16"]["median_macs_per_second"] / native["fma_l1"]["series"]["median_macs_per_second"]) < 1e-12
    full_formula_pass = all(
        abs(native["efficiencies"][f"E_FULL_{m}"] - native["full_resident"][f"m{m}_k1"]["median_macs_per_second"] / native["h0_q4"][f"m{m}"]["median_macs_per_second"]) < 1e-12
        for m in (4, 16)
    )
    source_equal = pytorch["source"]["source_weight_value_sha256"] == source["weight_state_sha256"]
    native_k4 = native["full_resident"]["m8_k4_for_s_native"]["median_seconds"]
    pytorch_k4 = pytorch["S_native_cells"]["d512_m8_K4"]["pytorch_fp32_original_median_seconds"]
    computed_speed = pytorch_k4 / native_k4
    candidate_decision = classify_candidate1(
        correctness=q4_scalar_pass,
        e_q4=float(native["efficiencies"]["E_Q4"]),
        e_full4=float(native["efficiencies"]["E_FULL_4"]),
        e_full16=float(native["efficiencies"]["E_FULL_16"]),
        s_native=computed_speed,
    )
    speed_formula_pass = abs(computed_speed - pytorch["S_native_cells"]["d512_m8_K4"]["S_native"]) < 1e-12
    candidate_limit_pass = candidate_id in ("candidate_01", "candidate_02") and len([candidate_id]) <= 2
    attempt02_pass = attempt02_before["artifact_hashes_verified_before_kq"] and attempt02_before["artifact_manifest_sha256"] == attempt02_after["artifact_manifest_sha256"] and attempt02_after["artifact_hashes_verified_after_kq"]
    m1_pass = all(str(d) in m1_diag and m1_diag[str(d)]["gate"] is False for d in (512, 640))
    machine = native["machine_balance_diagnostic"]
    machine_pass = machine["dram_working_set_bytes"] >= 4 * native["hardware"]["llc_bytes"] and machine["affinity_ok"] is True
    conformance = {
        "phase": "T0_PHYSICAL_KERNEL_QUALIFICATION_AND_REPLICATION",
        "prior_evidence": {
            "attempt_01": {"classification": "MEASUREMENT_INVALID", "scientific_result": None, "immutable": True},
            "attempt_02": {"classification": "RESIDENCY_GATE_FAIL", "validity": "VALID", "immutable": True},
        },
        "architecture": {"d": [512, 640], "m_target": [4, 8, 16], "m_control": [1], "K": [1, 4, 8], "block_formula": "16d^2", "quantization": "Q4_GROUP32_FP16_SCALE", "persistent_predequantized_copy": False},
        "kernel_qualification": {
            "candidate_id": candidate_id,
            "max_candidates": 2,
            "fma_peak": {"measured": True, "backend": "AVX2_FP32_FMA_L1", "workers": "four frozen attempt_02 P-cores", "P_FMA_MAC_per_s": native["fma_l1"]["series"]["median_macs_per_second"]},
            "h0_q4": {"P_H0_4_MAC_per_s": native["h0_q4"]["m4"]["median_macs_per_second"], "P_H0_16_MAC_per_s": native["h0_q4"]["m16"]["median_macs_per_second"], "E_Q4": native["efficiencies"]["E_Q4"], "minimum": 0.25, "pass": q4_gate},
            "full_block": {"P_FULL_4_MAC_per_s": native["full_resident"]["m4_k1"]["median_macs_per_second"], "P_FULL_16_MAC_per_s": native["full_resident"]["m16_k1"]["median_macs_per_second"], "E_FULL_4": native["efficiencies"]["E_FULL_4"], "E_FULL_16": native["efficiencies"]["E_FULL_16"], "minimum": 0.60, "pass": full4_gate and full16_gate},
            "project_speed": {"reference": "V2-0 PyTorch FP32 ContractualCoreBlock pre-Q4", "cell": "d512_m8_K4", "S_native": computed_speed, "minimum": 1.20, "pass": speed_gate},
            "scientifically_qualified": candidate_decision["scientifically_qualified"],
            "all_freeze_gates_pass": candidate_decision["all_five_gates_pass"],
        },
        "scientific_attempt": {"id": "attempt_03", "allowed_runs": 1, "cell_count": 72, "executed_during_kq": False},
        "scientific_gates_unchanged": {"rho_resident_max": 0.50, "delta_CB_max": 0.25, "matrixization_gain_min": 1.50},
        "diagnostics": {"residency_m1": m1_diag, "machine_balance": machine},
        "global_status": "CONFORMANCE_HOLD", "gpu": "HOLD", "T3": "HOLD",
        "candidate_decision": candidate_decision["terminal_status"],
        "status": "CONFORMANCE_HOLD",
    }
    rows = [
        ("test_v2_1b_vectorized_q4_matches_scalar", q4_scalar_pass, native["correctness"]["toy_d32_m4_k2"]),
        ("test_v2_1b_dequant_tile_reused_across_slots", tile["pass"] is True and tile["slots_reusing_each_row_tile"] == 16, tile),
        ("test_v2_1b_no_full_predequantized_weight_copy", no_copy, {"persistent_weight_format": native["persistent_weight_format"], "persistent_copy": native["predequantized_weight_copy_persistent"]}),
        ("test_v2_1b_scratch_per_worker_le_64k", scratch_pass, {"scratch_bytes": native["scratch_per_worker_bytes"], "limit_bytes": 65536}),
        ("test_v2_1b_fma_peak_fixture", fma_pass, native["fma_l1"]),
        ("test_v2_1b_h0_efficiency_formula", h0_formula_pass, {"E_Q4": native["efficiencies"]["E_Q4"]}),
        ("test_v2_1b_full_efficiency_formula", full_formula_pass, {"E_FULL_4": native["efficiencies"]["E_FULL_4"], "E_FULL_16": native["efficiencies"]["E_FULL_16"]}),
        ("test_v2_1b_kernel_quality_thresholds", q4_gate and full4_gate and full16_gate and q4_scalar_pass, {"E_Q4_pass": q4_gate, "E_FULL_4_pass": full4_gate, "E_FULL_16_pass": full16_gate, "correctness_pass": q4_scalar_pass}),
        ("test_v2_1b_pytorch_reference_same_equations", source_equal and pytorch["threads"] == 4 and pytorch["interop_threads"] == 1 and pytorch["eval"] and not pytorch["grad_enabled"], {"source_weight_hash_match": source_equal, "torch_version": pytorch["torch_version"], "threads": pytorch["threads"], "interop_threads": pytorch["interop_threads"], "eval": pytorch["eval"], "grad_enabled": pytorch["grad_enabled"]}),
        ("test_v2_1b_pytorch_speedup_formula", speed_formula_pass, {"S_native": computed_speed, "formula": "median(T_PyTorch_FP32_pre-Q4)/median(T_native_Q4)"}),
        ("test_v2_1b_candidate_limit_two", candidate_limit_pass, {"candidate_id": candidate_id, "max_candidates": 2, "measured_candidate_count": 1}),
        ("test_v2_1b_no_abc_before_kernel_freeze", native["kq_scope"]["a_b_c_executed"] is False and native["kq_scope"]["attempt03_executed"] is False, native["kq_scope"]),
        ("test_v2_1b_attempt02_preserved", attempt02_pass, {"before": attempt02_before, "after": attempt02_after}),
        ("test_v2_1b_attempt03_same_72_cells", conformance["scientific_attempt"]["cell_count"] == 72 and not conformance["scientific_attempt"]["executed_during_kq"], conformance["scientific_attempt"]),
        ("test_v2_1b_m1_residency_diagnostic", m1_pass, m1_diag),
        ("test_v2_1b_machine_balance_diagnostic", machine_pass, machine),
        ("test_v2_1b_conformance_block", conformance["status"] == "CONFORMANCE_HOLD" and conformance["gpu"] == "HOLD" and conformance["T3"] == "HOLD", conformance),
    ]
    return [{"name": name, "status": "PASS" if passed else "FAIL", "detail": detail} for name, passed, detail in rows], conformance


def format_kq_report(summary: dict[str, Any], results_root: Path, executable: Path, executable_hash: str,
                     build_manifest: dict[str, Any], artifact_hashes: dict[str, str]) -> str:
    native = summary["native"]
    gates = summary["gates"]
    tests = summary["test_report"]["tests"]
    m1 = summary["m1_residency_diagnostic_no_gate"]
    machine = summary["machine_balance_diagnostic"]
    lines = [
        "# OMEGA-V2-1b Kernel Qualification (KQ only)", "",
        f"- candidate: `{summary['candidate_id']}`",
        f"- terminal_status: `{summary['terminal_status']}`",
        f"- implementation_commit: `{summary['source_provenance']['implementation_commit']}`",
        f"- implementation_parent: `{summary['source_provenance']['implementation_parent']}`",
        f"- results_root_abs: `{results_root.resolve()}`",
        f"- frozen_attempt02: `immutable; manifest_sha256={summary['attempt02_preservation']['artifact_manifest_sha256']}`",
        f"- executable_abs: `{executable.resolve()}`",
        f"- executable_sha256: `{executable_hash}`",
        f"- attempt03_executed: `false`",
        f"- global_status: `CONFORMANCE_HOLD`; GPU/T3: `HOLD`",
        "",
        "## Correctness and KQ gates",
        f"- vectorized_vs_scalar_d32_m4_k2: `{native['correctness']['toy_d32_m4_k2']}`",
        f"- d512_full_finite_and_checksums_stable: `{native['correctness']['full_d512_outputs_finite']}` / `{native['correctness']['full_checksum_stable']}`",
        f"- P_FMA: `{native['fma_l1']['series']['median_macs_per_second']:.9g}` MAC/s; L1 tile: `{native['fma_l1']['working_set_bytes_per_worker']}` / `{native['fma_l1']['l1d_bytes_per_worker']}` bytes",
        f"- P_H0(4/16): `{native['h0_q4']['m4']['median_macs_per_second']:.9g}` / `{native['h0_q4']['m16']['median_macs_per_second']:.9g}` MAC/s",
        f"- P_FULL(4/16): `{native['full_resident']['m4_k1']['median_macs_per_second']:.9g}` / `{native['full_resident']['m16_k1']['median_macs_per_second']:.9g}` MAC/s",
        f"- E_Q4: `{gates['E_Q4']['value']:.6f}` (minimum 0.25): `{gates['E_Q4']['pass']}`",
        f"- E_FULL(4/16): `{gates['E_FULL_4']['value']:.6f}` / `{gates['E_FULL_16']['value']:.6f}` (each minimum 0.60): `{gates['E_FULL_4']['pass']}` / `{gates['E_FULL_16']['pass']}`",
        f"- S_native: `{gates['S_native']['value']:.6f}` (minimum 1.20): `{gates['S_native']['pass']}`",
        f"- scratch_per_worker_bytes: `{gates['scratch_per_worker_bytes']['value']}` (limit 65,536)",
        "",
        "## PyTorch source and correctness-side errors",
        f"- source: `{summary['source_weight_manifest']['source']}`",
        f"- seed: `{summary['source_weight_manifest']['seed']}`; source FP32 value SHA-256: `{summary['source_weight_manifest']['weight_state_sha256']}`",
        f"- same source tensors in native-Q4 packer and PyTorch control: `{summary['pytorch']['source']['source_weight_value_sha256'] == summary['source_weight_manifest']['weight_state_sha256']}`",
    ]
    dequant_delta = summary["pytorch"]["S_native_cells"]["d512_m8_K4"]["dequant_vs_original_median_delta_fraction"]
    lines.append(f"- dequantized-FP32 vs original-FP32 timing delta (diagnostic): `{dequant_delta:.6%}`; >3%: `{dequant_delta > 0.03}`")
    for cell, detail in summary["pytorch"]["correctness_side_checks"].items():
        lines.append(f"- {cell}: kernel error vs dequant-FP32 `{detail['kernel_error_vs_dequant_fp32']}`; quantization error vs original-FP32 `{detail['quantization_error_vs_original_fp32']}`")
    lines.extend([
        "", "## Non-gate diagnostics",
        f"- machine balance: BW_DRAM `{machine['BW_DRAM_bytes_per_second']}` bytes/s; working set `{machine['dram_working_set_bytes']}` bytes; M_machine `{machine['M_machine_MAC_per_byte']}` MAC/byte; AI_m4 `{machine['AI_m4_MAC_per_byte']}` MAC/byte; rho_optimistic `{machine['rho_optimistic']}`",
        f"- attempt_02 m1 d512 diagnostic rho_A/B `{m1['512']['rho_m1_A_over_B_diagnostic_only']}`; delta_CB `{m1['512']['delta_CB_m1_diagnostic_only']}`; c_C>c_A `{m1['512']['c_C_gt_c_A_diagnostic_only']}`",
        f"- attempt_02 m1 d640 diagnostic rho_A/B `{m1['640']['rho_m1_A_over_B_diagnostic_only']}`; delta_CB `{m1['640']['delta_CB_m1_diagnostic_only']}`; c_C>c_A `{m1['640']['c_C_gt_c_A_diagnostic_only']}`",
        "- m1 diagnostics are non-gate and were not used to select/score the kernel candidate.",
        "", "## Contractual KQ tests (17)",
    ])
    for row in tests:
        lines.append(f"- {row['status']}: `{row['name']}`")
    lines.extend(["", "## Source/build/artifact SHA-256"])
    for path, digest in sorted(artifact_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.extend([
        "", "All SHA-256 values are complete 64-hex digests. `artifact_hashes.json` also records byte sizes; attempt_01/attempt_02 artifacts were read-only.",
        "", "STOP: this report covers kernel qualification only. No A/B/C sweep, attempt_03, GPU, training, language scoring, LN, or T3 ran.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", choices=("candidate_01", "candidate_02"), default="candidate_01")
    args = parser.parse_args()
    if os.name != "nt" or sys.platform != "win32":
        raise RuntimeError("KQ_MEASUREMENT_INVALID: candidate qualification must run as native Windows")
    if args.candidate_id != "candidate_01":
        raise RuntimeError("candidate_02 is not implemented yet; do not run until candidate_01 outcome authorizes it")

    provenance = source_provenance()
    seal_validation = validate_v2_0_seal()
    attempt02_before = validate_attempt02_preservation()
    attempt02_m1 = attempt02_m1_diagnostics()
    vswhere = find_vswhere()
    cmake, vs_install, cmake_version = find_msvc_cmake(vswhere)
    if (RESULTS_BASE / args.candidate_id).exists():
        raise FileExistsError(f"immutable KQ result directory already exists: {RESULTS_BASE / args.candidate_id}")

    source_hashes_before = file_hashes()
    dependency_hashes = {str(path.resolve()): sha256_file(path) for path in dependency_files()}
    executable, build_logs = build_native(cmake)
    exe_hash = sha256_file(executable)
    source_hashes_after = file_hashes()
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("KQ_SOURCE_HOLD: KQ source/configuration changed during build")

    results_root = RESULTS_BASE / args.candidate_id
    results_root.mkdir(parents=True, exist_ok=False)
    build_manifest = {
        "schema": "omega-v2-1b-kq-build-manifest-v1",
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "implementation_commit": provenance["implementation_commit"],
        "implementation_parent": provenance["implementation_parent"],
        "git_provenance": provenance,
        "generator": "Visual Studio 17 2022 x64",
        "configuration": "Release",
        "visual_studio_installation": vs_install,
        "cmake_path": str(cmake),
        "cmake_version": cmake_version,
        "compiler": "MSVC 19.44.35229.0",
        "compile_flags": ["/O2", "/GL", "/arch:AVX2", "/fp:precise", "/W4", "/EHsc", "/LTCG"],
        "executable_absolute_path": str(executable.resolve()),
        "executable_size_bytes": executable.stat().st_size,
        "executable_sha256": exe_hash,
        "source_sha256": source_hashes_before,
        "shared_attempt02_dependency_sha256": dependency_hashes,
        "configure_stdout": build_logs["configure_stdout"], "configure_stderr": build_logs["configure_stderr"],
        "build_stdout": build_logs["build_stdout"], "build_stderr": build_logs["build_stderr"],
    }
    write_json(results_root / "build_manifest.json", build_manifest)

    import torch
    from pathlib import PurePosixPath
    sys.path.insert(0, str(V2_0_ROOT))
    from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_block = ContractualCoreBlock(512, dtype=torch.float32, seed=20260929)
    weight_digest = hashlib.sha256()
    payload_parts = []
    for name in MATRIX_FAMILIES:
        value = getattr(source_block, name).detach().contiguous().cpu()
        raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
        payload_parts.append(raw)
        weight_digest.update(name.encode("utf-8")); weight_digest.update(b"\0"); weight_digest.update(raw)
    source_weight_sha = weight_digest.hexdigest()
    weight_payload = b"".join(payload_parts)
    source_weight_info = {
        "source": "V2-0 omega_v2.core.ContractualCoreBlock FP32 source tensors, pre-Q4",
        "source_path_abs": str((V2_0_ROOT / "omega_v2" / "core.py").resolve()),
        "seed": 20260929,
        "d": 512,
        "weight_state_sha256": source_weight_sha,
        "weight_stream_sha256": sha256_bytes(weight_payload),
        "weight_stream_bytes": len(weight_payload),
        "matrix_families": [{"name": name, "shape": list(getattr(source_block, name).shape)} for name in MATRIX_FAMILIES],
    }
    temporary_owner = tempfile.TemporaryDirectory(prefix="omega_v2_1b_kq_")
    temporary = Path(temporary_owner.name)
    weights_path = temporary / "v2_0_fp32_source_weights.tmp"
    weights_path.write_bytes(weight_payload)
    dequant_path = temporary / "q4_dequantized_weights.tmp"
    native_output_dir = results_root / "native_correctness_outputs"
    native_output_dir.mkdir(parents=True, exist_ok=False)
    native_env = os.environ.copy()
    native_env.update({
        "OMEGA_V2_1B_RESULTS_ROOT": str(results_root.resolve()),
        "OMEGA_V2_1B_DEQUANT_OUTPUT": str(dequant_path.resolve()),
        "OMEGA_V2_1B_NATIVE_OUTPUT_DIR": str(native_output_dir.resolve()),
        "OMEGA_V2_1B_SHARD_WEIGHTS": ",".join(str(value) for value in attempt02_before["frozen_h0_shard_weights"]),
        "OMEGA_V2_1B_SOURCE_WEIGHT_SHA256": source_weight_sha,
        "OMEGA_V2_1B_ATTEMPT02_MANIFEST_SHA256": attempt02_before["artifact_manifest_sha256"],
    })
    native_process = subprocess.run(
        [str(executable), "--run-kq", str(weights_path)], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=4 * 60 * 60, env=native_env,
    )
    (results_root / "native_kq_stdout.log").write_text(native_process.stdout, encoding="utf-8", newline="\n")
    (results_root / "native_kq_stderr.log").write_text(native_process.stderr, encoding="utf-8", newline="\n")
    if native_process.returncode != 0:
        raise RuntimeError(f"KQ native candidate failed before analysis, exit={native_process.returncode}: {native_process.stderr[-3000:]}")
    native_path = results_root / "native_kq_candidate_01.json"
    native = read_json(native_path)
    if native.get("source_weight_sha256") != source_weight_sha:
        raise RuntimeError("KQ native candidate source weight hash differs from V2-0 PyTorch FP32 weights")

    pytorch_path = results_root / "pytorch_control.json"
    pytorch_process = subprocess.run(
        [sys.executable, str(UNIT_ROOT / "scripts" / "pytorch_control.py"),
         "--logical-processor-ids", ",".join(str(x) for x in attempt02_before["frozen_worker_logical_ids"]),
         "--dequantized-weights", str(dequant_path.resolve()),
         "--native-output-m4", str(native_output_dir / "candidate1_full_m4_k1.bin"),
         "--native-output-m16", str(native_output_dir / "candidate1_full_m16_k1.bin"),
         "--native-output-m8-k4", str(native_output_dir / "candidate1_full_m8_k4.bin"),
         "--output-json", str(pytorch_path.resolve())], cwd=REPO_ROOT, capture_output=True, text=True, timeout=4 * 60 * 60,
    )
    (results_root / "pytorch_control_stdout.log").write_text(pytorch_process.stdout, encoding="utf-8", newline="\n")
    (results_root / "pytorch_control_stderr.log").write_text(pytorch_process.stderr, encoding="utf-8", newline="\n")
    if pytorch_process.returncode != 0:
        raise RuntimeError(f"KQ PyTorch control failed, exit={pytorch_process.returncode}: {pytorch_process.stderr[-3000:]}")
    pytorch = read_json(pytorch_path)
    temporary_owner.cleanup()

    native_k4 = float(native["full_resident"]["m8_k4_for_s_native"]["median_seconds"])
    pytorch_k4 = float(pytorch["S_native_cells"]["d512_m8_K4"]["pytorch_fp32_original_median_seconds"])
    speed_ratio = pytorch_k4 / native_k4
    pytorch["S_native_cells"]["d512_m8_K4"]["native_median_seconds"] = native_k4
    pytorch["S_native_cells"]["d512_m8_K4"]["S_native"] = speed_ratio
    pytorch["S_native_cells"]["d512_m8_K4"]["S_native_minimum"] = 1.20
    write_json(pytorch_path, pytorch)

    scratch_bytes = int(native["scratch_per_worker_bytes"])
    e_q4 = float(native["efficiencies"]["E_Q4"])
    e_full4 = float(native["efficiencies"]["E_FULL_4"])
    e_full16 = float(native["efficiencies"]["E_FULL_16"])
    correctness_pass = native["correctness"]["toy_d32_m4_k2"].get("pass") is True and native["correctness"]["full_d512_outputs_finite"] is True and native["correctness"]["full_checksum_stable"] is True
    q4_gate = e_q4 >= 0.25
    full4_gate = e_full4 >= 0.60
    full16_gate = e_full16 >= 0.60
    speed_gate = speed_ratio >= 1.20
    candidate_decision = classify_candidate1(
        correctness=correctness_pass, e_q4=e_q4, e_full4=e_full4, e_full16=e_full16, s_native=speed_ratio,
    )
    all_freeze_gates = candidate_decision["all_five_gates_pass"]

    attempt02_after = validate_attempt02_preservation()
    m1_diag = attempt02_m1_diagnostics()
    tests, conformance = make_test_rows(native, pytorch, q4_gate, full4_gate, full16_gate, speed_gate, attempt02_before, attempt02_after, m1_diag, source_weight_info, args.candidate_id)
    if [row["name"] for row in tests] != KQ_TEST_NAMES:
        raise RuntimeError("KQ_REPORT_INVALID: the 17 contractual tests are missing or out of order")
    test_report = {
        "schema": "omega-v2-1b-kq-test-report-v1",
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "test_count": len(tests), "pass_count": sum(row["status"] == "PASS" for row in tests),
        "fail_count": sum(row["status"] == "FAIL" for row in tests), "skip_count": sum(row["status"] == "SKIP" for row in tests),
        "tests": tests,
    }
    write_json(results_root / "test_report.json", test_report)
    write_json(results_root / "OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml", {"OMEGA_CONFORMANCE_BLOCK": conformance})
    write_json(results_root / "attempt02_preservation.json", attempt02_after)
    write_json(results_root / "source_weight_manifest.json", source_weight_info)

    terminal = candidate_decision["terminal_status"]
    kq_summary = {
        "schema": "omega-v2-1b-kq-summary-v1",
        "terminal_status": terminal,
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "source_provenance": provenance,
        "attempt02_preservation": attempt02_after,
        "source_weight_manifest": source_weight_info,
        "native": native,
        "pytorch": pytorch,
        "m1_residency_diagnostic_no_gate": m1_diag,
        "gates": {
            "correctness": correctness_pass,
            "E_Q4": {"value": e_q4, "minimum": 0.25, "pass": q4_gate},
            "E_FULL_4": {"value": e_full4, "minimum": 0.60, "pass": full4_gate},
            "E_FULL_16": {"value": e_full16, "minimum": 0.60, "pass": full16_gate},
            "S_native": {"value": speed_ratio, "minimum": 1.20, "pass": speed_gate},
            "scratch_per_worker_bytes": {"value": scratch_bytes, "maximum": 65536, "pass": scratch_bytes <= 65536},
        },
        "candidate_state": {
            "max_candidates": 2,
            "candidate01_measured": True,
            "candidate02_measured": False,
            "candidate02_allowed": not all_freeze_gates,
            "candidate02_prohibited_if_candidate01_passes_all_five": all_freeze_gates,
            "attempt03_executed": False,
            "attempt03_allowed_after_freeze": all_freeze_gates,
        },
        "machine_balance_diagnostic": native["machine_balance_diagnostic"],
        "test_report": test_report,
        "global_status": "CONFORMANCE_HOLD",
        "gpu": "HOLD",
        "T3": "HOLD",
    }
    write_json(results_root / "summary_metrics.json", kq_summary)
    freeze_manifest_path = results_root / "candidate_freeze_manifest.json"
    if all_freeze_gates:
        write_json(freeze_manifest_path, {
            "schema": "omega-v2-1b-candidate-freeze-v1",
            "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
            "frozen": True,
            "attempt03_allowed_exactly_once": True,
            "implementation_commit": provenance["implementation_commit"],
            "implementation_parent": provenance["implementation_parent"],
            "executable_absolute_path": str(executable.resolve()),
            "executable_sha256": exe_hash,
            "source_sha256": build_manifest["source_sha256"],
            "shared_attempt02_dependency_sha256": dependency_hashes,
            "compiler_flags": build_manifest["compile_flags"],
            "selected_worker_cpu_set_ids": attempt02_before["frozen_worker_cpu_set_ids"],
            "selected_worker_shard_weights": attempt02_before["frozen_h0_shard_weights"],
            "weight_source": source_weight_info,
            "qualification_gates": kq_summary["gates"],
        })
    source_hashes_final = file_hashes()
    if source_hashes_final != source_hashes_before:
        raise RuntimeError("KQ_SOURCE_HOLD: source/configuration changed during qualification")
    if attempt02_after != validate_attempt02_preservation():
        raise RuntimeError("KQ_STOP: attempt_02 artifacts changed during KQ")
    report_path = results_root / "OMEGA_V2_1B_KQ_REPORT.md"
    report_sidecar = results_root / "OMEGA_V2_1B_KQ_REPORT.md.sha256"
    pre_report_paths = [
        *(UNIT_ROOT / name for name in source_files()),
        *dependency_files(), executable, V2_0_SEAL, Path(attempt02_before["artifact_manifest_path_abs"]),
        results_root / "build_manifest.json", results_root / "native_kq_candidate_01.json",
        results_root / "pytorch_control.json", results_root / "source_weight_manifest.json",
        results_root / "attempt02_preservation.json", results_root / "test_report.json",
        results_root / "summary_metrics.json", results_root / "OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml",
        results_root / "OMEGA_V2_1B_KQ_REPORT.md", results_root / "native_kq_stdout.log",
        results_root / "native_kq_stderr.log", results_root / "pytorch_control_stdout.log", results_root / "pytorch_control_stderr.log",
        *(native_output_dir / name for name in ("candidate1_full_m4_k1.bin", "candidate1_full_m16_k1.bin", "candidate1_full_m8_k4.bin")),
    ]
    if freeze_manifest_path.is_file():
        pre_report_paths.append(freeze_manifest_path)
    report_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in pre_report_paths if path.is_file()
    }
    report = format_kq_report(kq_summary, results_root, executable, exe_hash, build_manifest, report_hashes)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    report_digest = sha256_file(report_path)
    report_sidecar.write_text(report_digest + "\n", encoding="ascii", newline="\n")
    result_paths = [*pre_report_paths, report_path, report_sidecar]
    artifact_hashes = {
        "schema": "omega-v2-1b-kq-artifact-hashes-v1",
        "implementation_commit": provenance["implementation_commit"],
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "artifacts": {
            str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in result_paths if path.is_file()
        },
    }
    write_json(results_root / "artifact_hashes.json", artifact_hashes)
    verified = all(rec["sha256"] == sha256_file(Path(path)) for path, rec in artifact_hashes["artifacts"].items())
    verified = verified and sha256_file(report_path) == report_sidecar.read_text(encoding="ascii").strip()
    (results_root / "artifact_hashes_verified.json").write_text(json.dumps({"verified": verified, "artifact_count": len(artifact_hashes["artifacts"])}, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not verified:
        raise RuntimeError("KQ_RESULT_HASH_FAILURE")

    print(json.dumps({
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE", "terminal_status": terminal,
        "implementation_commit": provenance["implementation_commit"],
        "results_root_abs": str(results_root.resolve()),
        "report_abs": str((results_root / "OMEGA_V2_1B_KQ_REPORT.md").resolve()),
        "artifact_hashes_abs": str((results_root / "artifact_hashes.json").resolve()),
        "artifact_count": len(artifact_hashes["artifacts"]), "hashes_verified": verified,
        "tests_passed": test_report["pass_count"], "tests_failed": test_report["fail_count"], "tests_skipped": test_report["skip_count"],
        "E_Q4": e_q4, "E_FULL_4": e_full4, "E_FULL_16": e_full16, "S_native": speed_ratio,
    }, indent=2, sort_keys=True))
    return 0 if all_freeze_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
