"""Build and run the single isolated OMEGA-V2-1b candidate_02 KQ qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
CAMPAIGN_ROOT = UNIT_ROOT.parent
KQ_UNIT = CAMPAIGN_ROOT / "omega_v2_1b_kernel_qualification"
V2_1_ROOT = CAMPAIGN_ROOT / "omega_v2_1_physical"
V2_0_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
RESULTS_BASE = KQ_UNIT / "results" / "omega_v2_1b_kernel_qualification"
RESULTS_ROOT = RESULTS_BASE / "candidate_02" / "run_01"
BUILD_ROOT = UNIT_ROOT / "build"
EXECUTABLE = BUILD_ROOT / "Release" / "omega_v2_1b_candidate_02.exe"
CORRECTNESS_EXE = BUILD_ROOT / "Release" / "omega_v2_1b_candidate_02_correctness.exe"
ATTEMPT02_ROOT = V2_1_ROOT / "results" / "omega_v2_1_physical" / "attempt_02"
V2_0_SEAL = V2_0_ROOT / "V2_0_RESULT_SEAL.json"
CANDIDATE01_ROOT = RESULTS_BASE / "candidate_01" / "run_02"
FROZEN_CPU_SET_IDS = [266, 264, 258, 270]

sys.path.insert(0, str(KQ_UNIT))
from scripts import run_kq as contract  # noqa: E402


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


def candidate_source_files() -> list[str]:
    files = ["CMakeLists.txt", "OMEGA_V2_1B_CANDIDATE_02_SPEC.md", ".gitignore"]
    files.extend(f"src/{path.name}" for path in sorted((UNIT_ROOT / "src").glob("*.cpp")))
    files.extend(f"src/{path.name}" for path in sorted((UNIT_ROOT / "src").glob("*.hpp")))
    files.extend(f"scripts/{path.name}" for path in sorted((UNIT_ROOT / "scripts").glob("*.py")))
    files.extend(f"tests/{path.name}" for path in sorted((UNIT_ROOT / "tests").glob("*.py")))
    return files


def candidate_source_hashes() -> dict[str, str]:
    return {name: sha256_file(UNIT_ROOT / name) for name in candidate_source_files()}


def shared_dependencies() -> list[Path]:
    source = V2_1_ROOT / "src"
    return [source / name for name in (
        "v2_1.hpp", "hardware_topology.cpp", "q4_layout.cpp", "worker_pool.cpp", "allocation_guard.cpp",
    )]


def validate_candidate01() -> tuple[dict[str, Any], dict[str, Any], str]:
    hashes_path = CANDIDATE01_ROOT / "artifact_hashes.json"
    manifest = read_json(hashes_path)
    failures = []
    for absolute_path, record in manifest["artifacts"].items():
        artifact = Path(absolute_path)
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            failures.append(absolute_path)
    if failures:
        raise RuntimeError(f"KQ_STOP: candidate_01 immutable artifacts failed hash checks: {failures}")
    native = read_json(CANDIDATE01_ROOT / "native_kq_candidate_01.json")
    summary = read_json(CANDIDATE01_ROOT / "summary_metrics.json")
    if summary["candidate_state"].get("candidate02_allowed") is not True:
        raise RuntimeError("KQ_STOP: candidate_01 does not authorize candidate_02")
    if summary["candidate_state"].get("attempt03_executed") is not False:
        raise RuntimeError("KQ_STOP: candidate_01 unexpectedly records an attempt_03 execution")
    if native["candidate_id"] != "KQ1_DEQUANT_ROW_REUSE":
        raise RuntimeError("KQ_STOP: candidate_01 native baseline identity mismatch")
    return native, summary, sha256_file(hashes_path)


def source_manifest(provenance: dict[str, Any], attempt02: dict[str, Any], candidate01_sha: str) -> dict[str, Any]:
    candidate_status = git("status", "--porcelain", "--", str(UNIT_ROOT.relative_to(REPO_ROOT)))
    if candidate_status:
        raise RuntimeError(f"KQ_SOURCE_HOLD: candidate_02 sources must be committed before measurement: {candidate_status!r}")
    return {
        "repository": "Hackerprod/Ejercise",
        "branch": git("branch", "--show-current"),
        "implementation_commit": git("rev-parse", "HEAD"),
        "git_provenance": provenance,
        "candidate02_unit_status": candidate_status,
        "candidate02_source_sha256": candidate_source_hashes(),
        "shared_attempt02_dependency_sha256": {str(path.resolve()): sha256_file(path) for path in shared_dependencies()},
        "attempt02_manifest_sha256": attempt02["artifact_manifest_sha256"],
        "candidate01_artifact_manifest_sha256": candidate01_sha,
        "v2_0_seal_sha256": sha256_file(V2_0_SEAL),
    }


def find_vswhere() -> Path:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")), Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    for root in roots:
        path = root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if path.is_file():
            return path
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


def build_native(cmake: Path) -> tuple[dict[str, str], str]:
    configure = subprocess.run(
        [str(cmake), "-S", str(UNIT_ROOT), "-B", str(BUILD_ROOT), "-G", "Visual Studio 17 2022", "-A", "x64"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if configure.returncode:
        raise RuntimeError("candidate_02 CMake configure failed: " + configure.stderr[-4000:])
    build = subprocess.run(
        [str(cmake), "--build", str(BUILD_ROOT), "--config", "Release", "--target", "omega_v2_1b_candidate_02", "omega_v2_1b_candidate_02_correctness", "-j", "8"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if build.returncode:
        raise RuntimeError("candidate_02 MSVC Release build failed: " + build.stderr[-5000:])
    if not EXECUTABLE.is_file() or not CORRECTNESS_EXE.is_file():
        raise FileNotFoundError("candidate_02 Release executable(s) missing")
    return {
        "configure_stdout": configure.stdout,
        "configure_stderr": configure.stderr,
        "build_stdout": build.stdout,
        "build_stderr": build.stderr,
    }, configure.stdout.splitlines()[0] if configure.stdout else "Visual Studio 17 2022 x64"


def run_offline_correctness() -> dict[str, Any]:
    result = subprocess.run([str(CORRECTNESS_EXE)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20 * 60)
    if result.returncode:
        raise RuntimeError(f"candidate_02 offline correctness failed ({result.returncode}): {result.stdout}\n{result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("candidate_02 offline correctness result is incomplete")
    correctness = json.loads(lines[0])
    affinity = json.loads(lines[1])
    if correctness.get("pass") is not True or affinity.get("worker_affinity_ok") is not True:
        raise RuntimeError("candidate_02 offline correctness/affinity check did not pass")
    return {"report": correctness, "worker_affinity_ok": affinity["worker_affinity_ok"], "stdout": result.stdout}


def verify_static_scope() -> dict[str, Any]:
    sources = {path.name: path.read_text(encoding="utf-8") for path in (UNIT_ROOT / "src").glob("*.cpp")}
    forbidden = ("run_full_sweep", "full_block_abc_correctness", "probe_eviction", "evict_weights")
    present = [symbol for text in sources.values() for symbol in forbidden if symbol in text]
    if present:
        raise RuntimeError(f"KQ_SOURCE_HOLD: candidate_02 includes A/B/C or eviction helpers: {present}")
    return {"forbidden_scientific_helpers_absent": True, "source_files_checked": sorted(sources)}


def read_source_weights() -> tuple[bytes, dict[str, Any]]:
    payload, raw = contract.v2_0_fp32_weight_payload()
    return payload, {
        "source": raw["source"],
        "source_path_abs": raw["source_path_abs"],
        "seed": raw["seed"],
        "d": raw["d"],
        "weight_state_sha256": raw["source_weight_value_sha256"],
        "weight_stream_sha256": raw["source_weight_stream_sha256"],
        "weight_stream_bytes": raw["source_weight_stream_bytes"],
        "matrix_families": raw["matrix_families"],
    }


def run_pytorch_control(results_root: Path, logical_ids: list[int], dequant_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    output_path = results_root / "pytorch_control.json"
    process = subprocess.run(
        [sys.executable, str(KQ_UNIT / "scripts" / "pytorch_control.py"),
         "--logical-processor-ids", ",".join(str(value) for value in logical_ids),
         "--dequantized-weights", str(dequant_path.resolve()),
         "--native-output-m4", str(output_dir / "candidate2_full_m4_k1.bin"),
         "--native-output-m16", str(output_dir / "candidate2_full_m16_k1.bin"),
         "--native-output-m8-k4", str(output_dir / "candidate2_full_m8_k4.bin"),
         "--output-json", str(output_path.resolve())],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=4 * 60 * 60,
    )
    (results_root / "pytorch_control_stdout.log").write_text(process.stdout, encoding="utf-8", newline="\n")
    (results_root / "pytorch_control_stderr.log").write_text(process.stderr, encoding="utf-8", newline="\n")
    if process.returncode:
        raise RuntimeError(f"candidate_02 PyTorch control failed ({process.returncode}): {process.stderr[-3000:]}")
    return read_json(output_path), {"stdout": process.stdout, "stderr": process.stderr}


def classify_candidate02(correctness: bool, e_q4: float, e_full4: float, e_full16: float, s_native: float) -> dict[str, Any]:
    scientific = correctness and e_q4 >= 0.25 and e_full4 >= 0.60 and e_full16 >= 0.60
    speed = s_native >= 1.20
    if scientific and speed:
        status = "FREEZE_CANDIDATE_2_ATTEMPT_03_ALLOWED"
    elif scientific:
        status = "PROJECT_NATIVE_SPEED_GATE_FAIL"
    else:
        status = "V2_1B_KERNEL_QUALIFICATION_FAIL"
    return {"scientifically_qualified": scientific, "project_speed_pass": speed, "all_five_gates_pass": scientific and speed, "terminal_status": status}


def make_candidate02_tests(
    native: dict[str, Any], pytorch: dict[str, Any], source_weights: dict[str, Any],
    candidate01_native: dict[str, Any], candidate01_summary: dict[str, Any],
    attempt_before: dict[str, Any], attempt_after: dict[str, Any], m1: dict[str, Any],
    decision: dict[str, Any], s_native: float, static_scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qkernel = (UNIT_ROOT / "src" / "q4_kernel_candidate2.cpp").read_text(encoding="utf-8")
    h0m16 = native["h0_q4"]["m16"]
    full = native["full_resident"]
    fma = native["fma_l1"]
    probe = native["dequant_tile_reuse"]["Q_W_m16"]
    dequant_pos = qkernel.find("dequantize_row(*job->matrices[matrix],")
    slot_pos = qkernel.find("for (int slot_base = 0; slot_base < token_rows")
    reuse_static = dequant_pos >= 0 and slot_pos > dequant_pos
    no_full_copy = (
        native["persistent_weight_format"] == "signed symmetric Q4 + FP16 group32 scales only"
        and native["predequantized_weight_copy_persistent"] is False
        and "alignas(64) float dequantized[kOutputRowTile][kMaxColumns]" in qkernel
        and "static float dequantized" not in qkernel
    )
    scratch_ok = native["scratch_per_worker_bytes"] == 49152 and native["scratch_per_worker_bytes"] <= 65536
    fma_ok = (
        fma["series"]["median_macs_per_second"] == candidate01_native["fma_l1"]["series"]["median_macs_per_second"]
        and fma["series"]["repetitions"] == 31
        and all(value == 0 for value in fma["series"]["timed_heap_allocations"])
        and fma["tile_fits_l1d"] is True and fma["affinity_ok"] is True
    )
    h0_alloc_ok = all(value == 0 for m in ("m1", "m4", "m16") for value in native["h0_q4"][m]["timed_heap_allocations"])
    h0_formula_ok = h0_alloc_ok and abs(native["efficiencies"]["E_Q4"] - h0m16["median_macs_per_second"] / fma["series"]["median_macs_per_second"]) < 1e-12
    full_formula_ok = all(
        abs(native["efficiencies"][f"E_FULL_{m}"] - full[f"m{m}_k1"]["median_macs_per_second"] / native["h0_q4"][f"m{m}"]["median_macs_per_second"]) < 1e-12
        for m in (4, 16)
    ) and all(value == 0 for name in ("m4_k1", "m16_k1", "m8_k4_for_s_native") for value in full[name]["timed_heap_allocations"])
    source_equal = pytorch["source"]["source_weight_value_sha256"] == source_weights["weight_state_sha256"]
    affinity_equal = pytorch["process_affinity"]["logical_processor_ids"] == attempt_before["frozen_worker_logical_ids"]
    pytorch_ok = source_equal and affinity_equal and pytorch["threads"] == 4 and pytorch["interop_threads"] == 1 and pytorch["eval"] is True and pytorch["grad_enabled"] is False
    pytorch_k4 = pytorch["S_native_cells"]["d512_m8_K4"]["pytorch_fp32_original_median_seconds"]
    native_k4 = full["m8_k4_for_s_native"]["median_seconds"]
    speed_formula_ok = abs(pytorch_k4 / native_k4 - s_native) < 1e-12
    q4_gate = native["efficiencies"]["E_Q4"] >= 0.25
    full4_gate = native["efficiencies"]["E_FULL_4"] >= 0.60
    full16_gate = native["efficiencies"]["E_FULL_16"] >= 0.60
    candidate_dirs = [path for path in RESULTS_BASE.glob("candidate_*") if any(path.rglob("native_kq_candidate_*.json"))]
    candidate_limit = len(candidate_dirs) <= 2 and "candidate_02" in [path.name for path in candidate_dirs]
    scope = native["kq_scope"]
    no_abc = static_scope["forbidden_scientific_helpers_absent"] and scope["a_b_c_executed"] is False and scope["attempt03_executed"] is False
    attempt_ok = attempt_before["artifact_hashes_verified_before_kq"] and attempt_after["artifact_hashes_verified_after_kq"] and attempt_before["artifact_manifest_sha256"] == attempt_after["artifact_manifest_sha256"]
    m1_ok = all(str(d) in m1 and m1[str(d)]["gate"] is False for d in (512, 640))
    machine = native["machine_balance_diagnostic"]
    machine_ok = machine["dram_working_set_bytes"] >= 4 * native["hardware"]["llc_bytes"] and machine["affinity_ok"] is True and all(value == 0 for value in machine["dram_stream"]["timed_heap_allocations"])
    correctness = native["correctness"]["toy_d32_m4_k2"].get("pass") is True and native["correctness"]["full_d512_outputs_finite"] is True and native["correctness"]["full_checksum_stable"] is True

    conformance = {
        "phase": "T0_PHYSICAL_KERNEL_QUALIFICATION_AND_REPLICATION",
        "prior_evidence": {
            "attempt_01": {"classification": "MEASUREMENT_INVALID", "scientific_result": None, "immutable": True},
            "attempt_02": {"classification": "RESIDENCY_GATE_FAIL", "validity": "VALID", "immutable": True},
        },
        "architecture": {"d": [512, 640], "m_target": [4, 8, 16], "m_control": [1], "K": [1, 4, 8], "block_formula": "16d^2", "quantization": "Q4_GROUP32_FP16_SCALE", "persistent_predequantized_copy": False},
        "kernel_qualification": {
            "candidate_id": "KQ2_ROW_TILE4_SLOT_TILE2_FUSED",
            "max_candidates": 2,
            "fma_peak": {"measured": True, "baseline_candidate": "candidate_01", "backend": "AVX2_FP32_FMA_L1", "workers": "four frozen attempt_02 P-cores", "P_FMA_MAC_per_s": fma["series"]["median_macs_per_second"]},
            "h0_q4": {"P_H0_4_MAC_per_s": native["h0_q4"]["m4"]["median_macs_per_second"], "P_H0_16_MAC_per_s": h0m16["median_macs_per_second"], "E_Q4": native["efficiencies"]["E_Q4"], "minimum": 0.25, "pass": q4_gate},
            "full_block": {"P_FULL_4_MAC_per_s": full["m4_k1"]["median_macs_per_second"], "P_FULL_16_MAC_per_s": full["m16_k1"]["median_macs_per_second"], "E_FULL_4": native["efficiencies"]["E_FULL_4"], "E_FULL_16": native["efficiencies"]["E_FULL_16"], "minimum": 0.60, "pass": full4_gate and full16_gate},
            "project_speed": {"reference": "V2-0 PyTorch FP32 ContractualCoreBlock pre-Q4", "cell": "d512_m8_K4", "S_native": s_native, "minimum": 1.20, "pass": s_native >= 1.20},
            "scientifically_qualified": decision["scientifically_qualified"],
            "all_freeze_gates_pass": decision["all_five_gates_pass"],
        },
        "scientific_attempt": {"id": "attempt_03", "allowed_runs": 1, "cell_count": 72, "executed_during_kq": False},
        "scientific_gates_unchanged": {"rho_resident_max": 0.50, "delta_CB_max": 0.25, "matrixization_gain_min": 1.50},
        "diagnostics": {"residency_m1": m1, "machine_balance": machine},
        "global_status": "CONFORMANCE_HOLD", "gpu": "HOLD", "T3": "HOLD",
        "candidate_decision": decision["terminal_status"], "status": "CONFORMANCE_HOLD",
    }
    tests = [
        ("test_v2_1b_vectorized_q4_matches_scalar", correctness, native["correctness"]["toy_d32_m4_k2"]),
        ("test_v2_1b_dequant_tile_reused_across_slots", probe["pass"] is True and probe["slots_reusing_each_row_tile"] == 16 and reuse_static, {**probe, "dequantization_outside_slot_loop": reuse_static}),
        ("test_v2_1b_no_full_predequantized_weight_copy", no_full_copy, {"persistent_weight_format": native["persistent_weight_format"], "persistent_copy": native["predequantized_weight_copy_persistent"]}),
        ("test_v2_1b_scratch_per_worker_le_64k", scratch_ok, {"scratch_bytes": native["scratch_per_worker_bytes"], "limit_bytes": 65536}),
        ("test_v2_1b_fma_peak_fixture", fma_ok, {"baseline_source": "candidate_01 immutable native KQ", "P_FMA_MAC_per_s": fma["series"]["median_macs_per_second"], "no_timed_allocations": all(value == 0 for value in fma["series"]["timed_heap_allocations"])}),
        ("test_v2_1b_h0_efficiency_formula", h0_formula_ok, {"E_Q4": native["efficiencies"]["E_Q4"]}),
        ("test_v2_1b_full_efficiency_formula", full_formula_ok, {"E_FULL_4": native["efficiencies"]["E_FULL_4"], "E_FULL_16": native["efficiencies"]["E_FULL_16"]}),
        ("test_v2_1b_kernel_quality_thresholds", correctness and q4_gate and full4_gate and full16_gate, {"correctness": correctness, "E_Q4_pass": q4_gate, "E_FULL_4_pass": full4_gate, "E_FULL_16_pass": full16_gate}),
        ("test_v2_1b_pytorch_reference_same_equations", pytorch_ok, {"source_weight_hash_match": source_equal, "same_four_p_core_logical_ids": affinity_equal, "threads": pytorch["threads"], "interop_threads": pytorch["interop_threads"], "eval": pytorch["eval"], "grad_enabled": pytorch["grad_enabled"]}),
        ("test_v2_1b_pytorch_speedup_formula", speed_formula_ok, {"S_native": s_native, "formula": "median(T_PyTorch_FP32_pre-Q4)/median(T_native_Q4)"}),
        ("test_v2_1b_candidate_limit_two", candidate_limit, {"max_candidates": 2, "measured_candidate_count": len(candidate_dirs), "candidate_id": native["candidate_id"]}),
        ("test_v2_1b_no_abc_before_kernel_freeze", no_abc, {**scope, **static_scope}),
        ("test_v2_1b_attempt02_preserved", attempt_ok, {"before": attempt_before, "after": attempt_after}),
        ("test_v2_1b_attempt03_same_72_cells", conformance["scientific_attempt"]["cell_count"] == 72 and not conformance["scientific_attempt"]["executed_during_kq"], conformance["scientific_attempt"]),
        ("test_v2_1b_m1_residency_diagnostic", m1_ok, m1),
        ("test_v2_1b_machine_balance_diagnostic", machine_ok, machine),
        ("test_v2_1b_conformance_block", conformance["status"] == "CONFORMANCE_HOLD" and conformance["gpu"] == "HOLD" and conformance["T3"] == "HOLD", conformance),
    ]
    if [name for name, _, _ in tests] != contract.KQ_TEST_NAMES:
        raise RuntimeError("candidate_02 KQ report does not match the frozen 17-test inventory")
    rows = [{"name": name, "status": "PASS" if passed else "FAIL", "detail": detail} for name, passed, detail in tests]
    return rows, conformance


def format_report(summary: dict[str, Any], results_root: Path, hashes: dict[str, str]) -> str:
    native, pytorch, gates = summary["native"], summary["pytorch"], summary["gates"]
    c1 = summary["candidate01_comparison"]
    lines = [
        "# OMEGA-V2-1b Candidate 02 Kernel Qualification (KQ only)", "",
        f"- candidate: `{native['candidate_id']}`",
        f"- terminal_status: `{summary['terminal_status']}`",
        f"- implementation_commit: `{summary['source_provenance']['implementation_commit']}`",
        f"- results_root_abs: `{results_root.resolve()}`",
        f"- attempt_02 immutable manifest SHA-256: `{summary['attempt02_preservation']['artifact_manifest_sha256']}`",
        f"- candidate_01 immutable artifact manifest SHA-256: `{summary['candidate01_artifact_manifest_sha256']}`",
        f"- executable_abs: `{summary['build_manifest']['executable_absolute_path']}`",
        f"- executable_sha256: `{summary['build_manifest']['executable_sha256']}`",
        "- attempt_03 executed: `false`; A/B/C executed: `false`; GPU/T3: `HOLD`", "",
        "## Gate results",
        f"- correctness: `{gates['correctness']['pass']}`; toy d32,m4,K2 max_abs `{native['correctness']['toy_d32_m4_k2']['max_abs_error']:.9g}`, max_rel `{native['correctness']['toy_d32_m4_k2']['max_rel_error']:.9g}`",
        f"- exp approximation: dense 16,001-point grid on `[-80,80]`, observed max relative error `{native['correctness']['toy_d32_m4_k2']['exp_approx_measured_max_rel']:.9g}`, bound `2e-6`",
        f"- P_FMA reused from candidate_01: `{native['fma_l1']['series']['median_macs_per_second']:.9g}` MAC/s",
        f"- P_H0(4/16): `{native['h0_q4']['m4']['median_macs_per_second']:.9g}` / `{native['h0_q4']['m16']['median_macs_per_second']:.9g}` MAC/s",
        f"- P_FULL(4/16): `{native['full_resident']['m4_k1']['median_macs_per_second']:.9g}` / `{native['full_resident']['m16_k1']['median_macs_per_second']:.9g}` MAC/s",
        f"- E_Q4 `{gates['E_Q4']['value']:.6f}` (min 0.25): `{gates['E_Q4']['pass']}`",
        f"- E_FULL(4/16) `{gates['E_FULL_4']['value']:.6f}` / `{gates['E_FULL_16']['value']:.6f}` (each min 0.60): `{gates['E_FULL_4']['pass']}` / `{gates['E_FULL_16']['pass']}`",
        f"- S_native `{gates['S_native']['value']:.6f}` (min 1.20): `{gates['S_native']['pass']}`",
        f"- scratch per worker `{gates['scratch_per_worker_bytes']['value']}` bytes (limit 65,536)", "",
        "## Candidate 01 → Candidate 02",
        f"- Candidate 01: E_Q4 `{c1['E_Q4']:.6f}`, E_FULL(4) `{c1['E_FULL_4']:.6f}`, E_FULL(16) `{c1['E_FULL_16']:.6f}`, S_native `{c1['S_native']:.6f}`; candidate_02 authorized: `{c1['candidate02_allowed']}`.",
        f"- Candidate 02: E_Q4 `{gates['E_Q4']['value']:.6f}`, E_FULL(4) `{gates['E_FULL_4']['value']:.6f}`, E_FULL(16) `{gates['E_FULL_16']['value']:.6f}`, S_native `{gates['S_native']['value']:.6f}`.",
        "- Candidate 02 kernel: output-row tile 4 × slot tile 2; fused QKV and gate/up; vectorized AVX2 attention and SwiGLU; Q4+FP16 scales remain the only persistent weight form.", "",
        "## PyTorch control and quantization error",
        f"- source FP32 value SHA-256: `{summary['source_weight_manifest']['weight_state_sha256']}`; same source tensors: `{summary['pytorch']['source']['source_weight_value_sha256'] == summary['source_weight_manifest']['weight_state_sha256']}`.",
        f"- PyTorch: `{pytorch['torch_version']}`, threads `{pytorch['threads']}`, interop `{pytorch['interop_threads']}`, four-core affinity `{pytorch['process_affinity']['logical_processor_ids']}`.",
        f"- S_native cell: PyTorch original median `{pytorch['S_native_cells']['d512_m8_K4']['pytorch_fp32_original_median_seconds']:.9g}` s, native Q4 median `{pytorch['S_native_cells']['d512_m8_K4']['native_median_seconds']:.9g}` s.",
    ]
    for cell, values in pytorch["correctness_side_checks"].items():
        lines.append(f"- {cell}: kernel error vs dequant-FP32 `{values['kernel_error_vs_dequant_fp32']}`; Q4 quantization error vs original FP32 `{values['quantization_error_vs_original_fp32']}`")
    machine = summary["machine_balance_diagnostic"]
    m1 = summary["m1_residency_diagnostic_no_gate"]
    lines.extend([
        "", "## Non-gate diagnostics",
        f"- machine balance reuses candidate_01's independent same-host/four-P-core DRAM stream measurement: BW_DRAM `{machine['BW_DRAM_bytes_per_second']:.9g}` bytes/s; working set `{machine['dram_working_set_bytes']}` bytes; M_machine `{machine['M_machine_MAC_per_byte']:.9g}`; AI_m4 `{machine['AI_m4_MAC_per_byte']:.9g}`; rho_optimistic `{machine['rho_optimistic']:.9g}`.",
        f"- attempt_02 m1 diagnostics (non-gate, evaluated after candidate selection): d512 rho_A/B `{m1['512']['rho_m1_A_over_B_diagnostic_only']:.9g}`, delta_CB `{m1['512']['delta_CB_m1_diagnostic_only']:.9g}`; d640 rho_A/B `{m1['640']['rho_m1_A_over_B_diagnostic_only']:.9g}`, delta_CB `{m1['640']['delta_CB_m1_diagnostic_only']:.9g}.",
        "- No A/B/C run, attempt_03 run, residency-gate evaluation, or GPU work occurred.", "", "## Contractual KQ tests (17)",
    ])
    lines.extend(f"- {row['status']}: `{row['name']}`" for row in summary["test_report"]["tests"])
    lines.extend(["", "## Source, build, and artifact SHA-256"])
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in sorted(hashes.items()))
    lines.extend(["", "STOP: candidate_02 is the final permitted kernel candidate. attempt_03 is permitted only if all five frozen gates pass.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="validate sealed inputs, build Release, and run offline correctness without KQ timing")
    args = parser.parse_args()
    if os.name != "nt" or sys.platform != "win32":
        raise RuntimeError("KQ_MEASUREMENT_INVALID: candidate_02 qualification must run as native Windows")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"candidate_02 run slot already exists and is immutable: {RESULTS_ROOT}")

    provenance = contract.source_provenance()
    seal_validation = contract.validate_v2_0_seal()
    attempt_before = contract.validate_attempt02_preservation()
    if attempt_before["frozen_worker_cpu_set_ids"] != FROZEN_CPU_SET_IDS:
        raise RuntimeError("KQ_STOP: attempt_02 CPU-set order differs from the frozen candidate_02 configuration")
    candidate01_native, candidate01_summary, candidate01_sha = validate_candidate01()
    if candidate01_summary["candidate_state"].get("all_five_gates_pass") is True:
        raise RuntimeError("KQ_STOP: candidate_01 passed all gates; candidate_02 is prohibited")
    candidate_provenance = source_manifest(provenance, attempt_before, candidate01_sha)
    static_scope = verify_static_scope()
    source_hashes_before = candidate_provenance["candidate02_source_sha256"]
    deps_before = {str(path.resolve()): sha256_file(path) for path in shared_dependencies()}
    vswhere = find_vswhere()
    cmake, vs_install, cmake_version = find_msvc_cmake(vswhere)
    build_logs, _ = build_native(cmake)
    offline_correctness = run_offline_correctness()
    executable_hash = sha256_file(EXECUTABLE)
    correctness_exe_hash = sha256_file(CORRECTNESS_EXE)
    if candidate_source_hashes() != source_hashes_before:
        raise RuntimeError("KQ_SOURCE_HOLD: candidate_02 source changed during build/offline correctness")
    if {str(path.resolve()): sha256_file(path) for path in shared_dependencies()} != deps_before:
        raise RuntimeError("KQ_SOURCE_HOLD: shared physical-kernel dependency changed during build")

    weight_payload, source_weight_info = read_source_weights()
    if source_weight_info["weight_state_sha256"] != candidate01_native["source_weight_sha256"]:
        raise RuntimeError("KQ_STOP: V2-0 source tensors differ from candidate_01's sealed KQ source")
    if args.preflight_only:
        print(json.dumps({
            "preflight_only": True,
            "measurement_started": False,
            "candidate02_sources_clean": True,
            "candidate01_hashes_verified": True,
            "attempt02_hashes_verified": True,
            "v2_0_seal_validated": True,
            "offline_correctness_passed": offline_correctness["report"]["pass"],
            "offline_correctness_executable_sha256": correctness_exe_hash,
            "measurement_executable_sha256": executable_hash,
            "candidate02_results_slot_free": not RESULTS_ROOT.exists(),
        }, indent=2, sort_keys=True))
        return 0
    logical_ids = attempt_before["frozen_worker_logical_ids"]
    build_manifest = {
        "schema": "omega-v2-1b-candidate02-kq-build-manifest-v1",
        "candidate_id": "KQ2_ROW_TILE4_SLOT_TILE2_FUSED",
        "measurement_implementation_commit": provenance["implementation_commit"],
        "git_provenance": provenance,
        "candidate02_source_provenance": candidate_provenance,
        "candidate02_source_sha256": source_hashes_before,
        "shared_dependency_sha256": deps_before,
        "candidate01_artifact_manifest_sha256": candidate01_sha,
        "attempt02_artifact_manifest_sha256": attempt_before["artifact_manifest_sha256"],
        "v2_0_seal": seal_validation["seal"],
        "generator": "Visual Studio 17 2022 x64",
        "configuration": "Release",
        "visual_studio_installation": vs_install,
        "cmake_path": str(cmake),
        "cmake_version": cmake_version,
        "compiler": "MSVC 19.44.35229.0",
        "compile_flags": ["/O2", "/GL", "/arch:AVX2", "/fp:precise", "/W4", "/EHsc", "/LTCG"],
        "executable_absolute_path": str(EXECUTABLE.resolve()),
        "executable_size_bytes": EXECUTABLE.stat().st_size,
        "executable_sha256": executable_hash,
        "offline_correctness_executable_absolute_path": str(CORRECTNESS_EXE.resolve()),
        "offline_correctness_executable_sha256": correctness_exe_hash,
        "offline_correctness_report": offline_correctness["report"],
        "offline_correctness_worker_affinity_ok": offline_correctness["worker_affinity_ok"],
        "build_logs": build_logs,
        "candidate01_p_fma_source": str((CANDIDATE01_ROOT / "native_kq_candidate_01.json").resolve()),
        "candidate01_p_fma_source_sha256": sha256_file(CANDIDATE01_ROOT / "native_kq_candidate_01.json"),
        "attempt02_frozen_worker_cpu_set_ids": attempt_before["frozen_worker_cpu_set_ids"],
        "attempt02_frozen_worker_logical_ids": logical_ids,
        "attempt02_frozen_h0_shard_weights": attempt_before["frozen_h0_shard_weights"],
    }

    # Creating run_01 seals the one candidate_02 measurement slot; it is never reused.
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    write_json(RESULTS_ROOT / "build_manifest.json", build_manifest)
    output_dir = RESULTS_ROOT / "native_correctness_outputs"
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="omega_v2_1b_candidate02_kq_") as temporary:
        temporary_root = Path(temporary)
        weights_path = temporary_root / "v2_0_fp32_source_weights.tmp"
        weights_path.write_bytes(weight_payload)
        dequant_path = temporary_root / "q4_dequantized_weights.tmp"
        native_env = os.environ.copy()
        native_env.update({
            "OMEGA_V2_1B_KQ2_RESULTS_ROOT": str(RESULTS_ROOT.resolve()),
            "OMEGA_V2_1B_KQ2_SHARD_WEIGHTS": ",".join(str(value) for value in attempt_before["frozen_h0_shard_weights"]),
            "OMEGA_V2_1B_KQ2_SOURCE_WEIGHT_SHA256": source_weight_info["weight_state_sha256"],
            "OMEGA_V2_1B_KQ2_ATTEMPT02_MANIFEST_SHA256": attempt_before["artifact_manifest_sha256"],
            "OMEGA_V2_1B_KQ2_CANDIDATE01_MANIFEST_SHA256": candidate01_sha,
            "OMEGA_V2_1B_KQ2_CANDIDATE01_NATIVE_JSON": str((CANDIDATE01_ROOT / "native_kq_candidate_01.json").resolve()),
            "OMEGA_V2_1B_KQ2_DEQUANT_PATH": str(dequant_path.resolve()),
            "OMEGA_V2_1B_KQ2_OUTPUTS_DIR": str(output_dir.resolve()),
        })
        native_process = subprocess.run(
            [str(EXECUTABLE), "--run-kq2", str(weights_path)], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=4 * 60 * 60, env=native_env,
        )
        (RESULTS_ROOT / "native_kq_stdout.log").write_text(native_process.stdout, encoding="utf-8", newline="\n")
        (RESULTS_ROOT / "native_kq_stderr.log").write_text(native_process.stderr, encoding="utf-8", newline="\n")
        if native_process.returncode:
            raise RuntimeError(f"candidate_02 native KQ failed ({native_process.returncode}): {native_process.stderr[-4000:]}")
        native = read_json(RESULTS_ROOT / "native_kq_candidate_02.json")
        if native.get("source_weight_sha256") != source_weight_info["weight_state_sha256"]:
            raise RuntimeError("candidate_02 native source-weight identity does not match V2-0 FP32 source")
        pytorch, _ = run_pytorch_control(RESULTS_ROOT, logical_ids, dequant_path, output_dir)

    if candidate_source_hashes() != source_hashes_before:
        raise RuntimeError("KQ_SOURCE_HOLD: candidate_02 source changed during measurement")
    if {str(path.resolve()): sha256_file(path) for path in shared_dependencies()} != deps_before:
        raise RuntimeError("KQ_SOURCE_HOLD: shared dependency changed during measurement")
    attempt_after_raw = contract.validate_attempt02_preservation()
    if attempt_after_raw["artifact_manifest_sha256"] != attempt_before["artifact_manifest_sha256"]:
        raise RuntimeError("KQ_STOP: immutable attempt_02 hash manifest changed during candidate_02 KQ")
    candidate01_native_after, _, candidate01_sha_after = validate_candidate01()
    if candidate01_sha_after != candidate01_sha or candidate01_native_after != candidate01_native:
        raise RuntimeError("KQ_STOP: candidate_01 KQ baseline changed during candidate_02 measurement")

    native_k4 = native["full_resident"]["m8_k4_for_s_native"]["median_seconds"]
    pytorch_k4 = pytorch["S_native_cells"]["d512_m8_K4"]["pytorch_fp32_original_median_seconds"]
    s_native = pytorch_k4 / native_k4
    pytorch["S_native_cells"]["d512_m8_K4"].update({"native_median_seconds": native_k4, "S_native": s_native, "S_native_minimum": 1.20})
    e_q4 = float(native["efficiencies"]["E_Q4"])
    e_full4 = float(native["efficiencies"]["E_FULL_4"])
    e_full16 = float(native["efficiencies"]["E_FULL_16"])
    correctness = native["correctness"]["toy_d32_m4_k2"].get("pass") is True and native["correctness"]["full_d512_outputs_finite"] is True and native["correctness"]["full_checksum_stable"] is True
    decision = classify_candidate02(correctness, e_q4, e_full4, e_full16, s_native)
    attempt_after = {**attempt_after_raw, "artifact_hashes_verified_after_kq": True}
    m1 = contract.attempt02_m1_diagnostics()
    rows, conformance = make_candidate02_tests(
        native, pytorch, source_weight_info, candidate01_native, candidate01_summary,
        {**attempt_before, "artifact_hashes_verified_before_kq": True}, attempt_after,
        m1, decision, s_native, static_scope,
    )
    test_report = {
        "schema": "omega-v2-1b-kq-test-report-v1",
        "candidate_id": native["candidate_id"],
        "test_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "skip_count": 0,
        "tests": rows,
    }
    correctness_info = {
        "pass": correctness,
        "toy_d32_m4_k2": native["correctness"]["toy_d32_m4_k2"],
        "full_d512_outputs_finite": native["correctness"]["full_d512_outputs_finite"],
        "full_checksum_stable": native["correctness"]["full_checksum_stable"],
    }
    metrics = {
        "correctness": {"pass": correctness},
        "E_Q4": {"value": e_q4, "minimum": 0.25, "pass": e_q4 >= 0.25},
        "E_FULL_4": {"value": e_full4, "minimum": 0.60, "pass": e_full4 >= 0.60},
        "E_FULL_16": {"value": e_full16, "minimum": 0.60, "pass": e_full16 >= 0.60},
        "S_native": {"value": s_native, "minimum": 1.20, "pass": s_native >= 1.20},
        "scratch_per_worker_bytes": {"value": native["scratch_per_worker_bytes"], "maximum": 65536, "pass": native["scratch_per_worker_bytes"] <= 65536},
    }
    summary = {
        "schema": "omega-v2-1b-candidate02-kq-summary-v1",
        "terminal_status": decision["terminal_status"],
        "candidate_id": native["candidate_id"],
        "source_provenance": provenance,
        "measurement_implementation_commit": provenance["implementation_commit"],
        "measurement_source_manifest_sha256": source_hashes_before,
        "build_manifest": build_manifest,
        "candidate01_artifact_manifest_sha256": candidate01_sha,
        "candidate01_comparison": {
            "candidate_id": candidate01_native["candidate_id"],
            "E_Q4": candidate01_summary["gates"]["E_Q4"]["value"],
            "E_FULL_4": candidate01_summary["gates"]["E_FULL_4"]["value"],
            "E_FULL_16": candidate01_summary["gates"]["E_FULL_16"]["value"],
            "S_native": candidate01_summary["gates"]["S_native"]["value"],
            "candidate02_allowed": candidate01_summary["candidate_state"]["candidate02_allowed"],
        },
        "attempt02_preservation": attempt_after,
        "source_weight_manifest": source_weight_info,
        "offline_correctness": offline_correctness,
        "native": native,
        "pytorch": pytorch,
        "correctness": correctness_info,
        "gates": metrics,
        "candidate_state": {
            "max_candidates": 2, "candidate01_measured": True, "candidate02_measured": True,
            "candidate02_allowed": False, "attempt03_executed": False,
            "attempt03_allowed_after_freeze": decision["all_five_gates_pass"],
        },
        "m1_residency_diagnostic_no_gate": m1,
        "machine_balance_diagnostic": native["machine_balance_diagnostic"],
        "test_report": test_report,
        "global_status": "CONFORMANCE_HOLD", "gpu": "HOLD", "T3": "HOLD",
    }
    write_json(RESULTS_ROOT / "source_weight_manifest.json", source_weight_info)
    write_json(RESULTS_ROOT / "attempt02_preservation.json", attempt_after)
    write_json(RESULTS_ROOT / "test_report.json", test_report)
    write_json(RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml", {"OMEGA_CONFORMANCE_BLOCK": conformance})
    write_json(RESULTS_ROOT / "summary_metrics.json", summary)

    result_paths = [
        *(UNIT_ROOT / name for name in candidate_source_files()), *shared_dependencies(),
        EXECUTABLE, CORRECTNESS_EXE, V2_0_SEAL,
        Path(attempt_before["artifact_manifest_path_abs"]), CANDIDATE01_ROOT / "artifact_hashes.json",
        CANDIDATE01_ROOT / "native_kq_candidate_01.json", CANDIDATE01_ROOT / "summary_metrics.json",
        RESULTS_ROOT / "build_manifest.json", RESULTS_ROOT / "native_kq_candidate_02.json",
        RESULTS_ROOT / "pytorch_control.json", RESULTS_ROOT / "source_weight_manifest.json",
        RESULTS_ROOT / "attempt02_preservation.json", RESULTS_ROOT / "test_report.json",
        RESULTS_ROOT / "summary_metrics.json", RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml",
        RESULTS_ROOT / "native_kq_stdout.log", RESULTS_ROOT / "native_kq_stderr.log",
        RESULTS_ROOT / "pytorch_control_stdout.log", RESULTS_ROOT / "pytorch_control_stderr.log",
        *(output_dir / name for name in ("candidate2_full_m4_k1.bin", "candidate2_full_m16_k1.bin", "candidate2_full_m8_k4.bin")),
    ]
    hashes = {str(path.resolve()): sha256_file(path) for path in result_paths if path.is_file()}
    report_path = RESULTS_ROOT / "OMEGA_V2_1B_CANDIDATE_02_KQ_REPORT.md"
    report_path.write_text(format_report(summary, RESULTS_ROOT, hashes), encoding="utf-8", newline="\n")
    report_sha = sha256_file(report_path)
    (RESULTS_ROOT / "OMEGA_V2_1B_CANDIDATE_02_KQ_REPORT.md.sha256").write_text(report_sha + "\n", encoding="ascii", newline="\n")
    hash_paths = [*result_paths, report_path, RESULTS_ROOT / "OMEGA_V2_1B_CANDIDATE_02_KQ_REPORT.md.sha256"]
    artifact_manifest = {
        "schema": "omega-v2-1b-kq-artifact-hashes-v1",
        "measurement_implementation_commit": provenance["implementation_commit"],
        "candidate_id": native["candidate_id"],
        "report_self_sha256": report_sha,
        "artifacts": {str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in hash_paths if path.is_file()},
    }
    write_json(RESULTS_ROOT / "artifact_hashes.json", artifact_manifest)
    verified = all(record["sha256"] == sha256_file(Path(path)) for path, record in artifact_manifest["artifacts"].items())
    verified = verified and sha256_file(report_path) == (RESULTS_ROOT / "OMEGA_V2_1B_CANDIDATE_02_KQ_REPORT.md.sha256").read_text(encoding="ascii").strip()
    write_json(RESULTS_ROOT / "artifact_hashes_verified.json", {"verified": verified, "artifact_count": len(artifact_manifest["artifacts"])})
    if not verified:
        raise RuntimeError("candidate_02 KQ result hash verification failed")
    print(json.dumps({
        "candidate_id": native["candidate_id"], "terminal_status": decision["terminal_status"],
        "measurement_implementation_commit": provenance["implementation_commit"],
        "results_root_abs": str(RESULTS_ROOT.resolve()), "report_abs": str(report_path.resolve()),
        "report_sha256": report_sha, "artifact_hashes_abs": str((RESULTS_ROOT / "artifact_hashes.json").resolve()),
        "artifact_hashes_verified": verified, "tests_passed": test_report["pass_count"],
        "tests_failed": test_report["fail_count"], "tests_skipped": test_report["skip_count"],
        "E_Q4": e_q4, "E_FULL_4": e_full4, "E_FULL_16": e_full16, "S_native": s_native,
    }, indent=2, sort_keys=True))
    return 0 if decision["all_five_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
