"""Preflight, verify candidate_02 binding, and run the authorized V2-1c sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
CAMPAIGN_ROOT = UNIT_ROOT.parent
PHYSICAL_ROOT = CAMPAIGN_ROOT / "omega_v2_1_physical"
KQ_ROOT = CAMPAIGN_ROOT / "omega_v2_1b_kernel_qualification"
KQ2_ROOT = CAMPAIGN_ROOT / "omega_v2_1b_candidate_02"
V2_0_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
ATTEMPT02_ROOT = PHYSICAL_ROOT / "results" / "omega_v2_1_physical" / "attempt_02"
KQ2_ROOT_RESULTS = KQ_ROOT / "results" / "omega_v2_1b_kernel_qualification" / "candidate_02" / "run_01"
KQ2_NATIVE = KQ2_ROOT_RESULTS / "native_kq_candidate_02.json"
KQ2_OUTPUTS = KQ2_ROOT_RESULTS / "native_correctness_outputs"
RESULTS_BASE = UNIT_ROOT / "results" / "omega_v2_1_physical"
RESULTS_ROOT = RESULTS_BASE / "attempt_01"
BUILD_ROOT = UNIT_ROOT / "build"
PREFLIGHT_ROOT = BUILD_ROOT / "preflight"
BENCH_EXE = BUILD_ROOT / "Release" / "omega_v2_1c_bench.exe"
EQUIVALENCE_EXE = BUILD_ROOT / "Release" / "omega_v2_1c_equivalence.exe"
FROZEN_CANDIDATE02_EXE_SHA256 = "be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f"
FROZEN_WORKER_IDS = [266, 264, 258, 270]
SANITY_RATIO_MIN = 0.70
SANITY_RATIO_MAX = 1.30
TEST_NAMES = [
    "test_v2_1c_candidate02_kernel_frozen_hash",
    "test_v2_1c_attempt02_preserved",
    "test_v2_1c_attempt03_same_72_cells",
    "test_v2_1c_schedule_affinity_and_qpc_match_attempt02",
    "test_v2_1c_candidate02_correctness",
    "test_v2_1c_no_kernel_candidate_or_tuning",
    "test_v2_1c_residency_gate_d512_m4_k8",
    "test_v2_1c_causal_control_delta_cb",
    "test_v2_1c_c_cold_control_gt_a",
    "test_v2_1c_matrixization_gate",
    "test_v2_1c_stability_gate_nine_primary_cells",
    "test_v2_1c_m1_residency_diagnostic_non_gate",
    "test_v2_1c_attempt02_vs_v2_1c_comparison_report",
    "test_v2_1c_conformance_block",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load support module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v21_analysis = load_module("omega_v2_1c_v21_analysis", PHYSICAL_ROOT / "scripts" / "analyze_v2_1.py")
kq_contract = load_module("omega_v2_1c_kq_contract", KQ_ROOT / "scripts" / "run_kq.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def source_files() -> list[str]:
    files = [".gitignore", "CMakeLists.txt", "OMEGA_V2_1C_RESIDENCY_ONLY_SPEC.md", "scripts/run_v2_1c.py"]
    files.extend(f"src/{path.name}" for path in sorted((UNIT_ROOT / "src").glob("*.cpp")))
    files.extend(f"src/{path.name}" for path in sorted((UNIT_ROOT / "src").glob("*.hpp")))
    files.extend(f"tests/{path.name}" for path in sorted((UNIT_ROOT / "tests").glob("*.py")))
    return files


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(UNIT_ROOT / name) for name in source_files()}


def physical_dependency_paths() -> list[Path]:
    source = PHYSICAL_ROOT / "src"
    return [source / name for name in (
        "v2_1.hpp", "main.cpp", "benchmark.cpp", "hardware_topology.cpp", "q4_layout.cpp",
        "worker_pool.cpp", "cache_controls.cpp", "allocation_guard.cpp",
    )]+[PHYSICAL_ROOT/"scripts"/"analyze_v2_1.py"]


def candidate02_kernel_paths() -> list[Path]:
    source = KQ2_ROOT / "src"
    return [source / name for name in ("kq_candidate2.hpp", "q4_kernel_candidate2.cpp", "full_block_candidate2.cpp")]


def verify_artifact_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    mismatches = []
    for absolute, record in manifest["artifacts"].items():
        artifact = Path(absolute)
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            mismatches.append(absolute)
    if mismatches:
        raise RuntimeError(f"V2_1C_ACCEPTANCE_HOLD: sealed artifact hash mismatch: {mismatches}")
    return manifest


def validate_attempt02() -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = ATTEMPT02_ROOT / "artifact_hashes.json"
    manifest = verify_artifact_manifest(manifest_path)
    config = read_json(ATTEMPT02_ROOT / "benchmark_config.json")
    workers = config.get("selected_workers", [])
    ids = [int(row["windows_cpu_set_id"]) for row in workers]
    if ids != FROZEN_WORKER_IDS:
        raise RuntimeError(f"V2_1C_ACCEPTANCE_HOLD: attempt_02 CPU-set identity changed: {ids}")
    weights = [float(row["h0_v_i"]) for row in workers]
    if len(weights) != 4 or not all(math.isfinite(weight) and weight > 0 for weight in weights):
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: attempt_02 v_i weights are invalid")
    return manifest, config, sha256_file(manifest_path)


def validate_candidate02() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    artifact_path = KQ2_ROOT_RESULTS / "artifact_hashes.json"
    manifest = verify_artifact_manifest(artifact_path)
    summary = read_json(KQ2_ROOT_RESULTS / "summary_metrics.json")
    build = read_json(KQ2_ROOT_RESULTS / "build_manifest.json")
    executable = Path(build["executable_absolute_path"])
    actual_exe_hash = sha256_file(executable)
    if actual_exe_hash != FROZEN_CANDIDATE02_EXE_SHA256 or actual_exe_hash != build["executable_sha256"]:
        raise RuntimeError(f"V2_1C_ACCEPTANCE_HOLD: frozen candidate_02 executable hash mismatch: {actual_exe_hash}")
    if summary["terminal_status"] != "PROJECT_NATIVE_SPEED_GATE_FAIL" or summary["candidate_id"] != "KQ2_ROW_TILE4_SLOT2_FUSED":
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: candidate_02 sealed identity or terminal status differs from MD/310")
    if summary["candidate_state"]["attempt03_executed"] is not False or summary["candidate_state"]["attempt03_allowed_after_freeze"] is not False:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: candidate_02 unexpectedly permits or records attempt_03")
    if manifest["candidate_id"] != summary["candidate_id"]:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: candidate_02 artifact manifest identity mismatch")
    return manifest, summary, build, actual_exe_hash


def verify_provenance(attempt_manifest_sha: str, c2_manifest_sha: str) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    unit_status = git("status", "--porcelain", "--", str(UNIT_ROOT.relative_to(REPO_ROOT)))
    physical_status = git("status", "--porcelain", "--", str(PHYSICAL_ROOT.relative_to(REPO_ROOT)))
    kq_status = git("status", "--porcelain", "--", str(KQ_ROOT.relative_to(REPO_ROOT)))
    c2_status = git("status", "--porcelain", "--", str(KQ2_ROOT.relative_to(REPO_ROOT)))
    md_blob = git("rev-parse", "HEAD:Conversacion.md")
    ln_blob = git("rev-parse", "HEAD:Conversacion LN.md")
    if branch != "main" or unit_status or physical_status or kq_status or c2_status:
        raise RuntimeError(f"V2_1C_SOURCE_HOLD: source units must be clean and committed: unit={unit_status!r}, physical={physical_status!r}, kq={kq_status!r}, c2={c2_status!r}")
    if md_blob != kq_contract.CONVERSACION_MD_BLOB or ln_blob != kq_contract.CONVERSACION_LN_BLOB:
        raise RuntimeError("V2_1C_SOURCE_HOLD: authoritative conversation blobs do not match the frozen KQ provenance")
    return {
        "repository": "Hackerprod/Ejercise", "branch": branch, "implementation_commit": commit,
        "implementation_parent": parents[0] if parents else None, "unit_status": unit_status,
        "physical_unit_status": physical_status, "kq_unit_status": kq_status, "candidate02_unit_status": c2_status,
        "git_status_porcelain_preexisting_unrelated": git("status", "--porcelain"),
        "conversacion_md_blob": md_blob, "conversacion_ln_blob": ln_blob,
        "attempt02_artifact_manifest_sha256": attempt_manifest_sha,
        "candidate02_artifact_manifest_sha256": c2_manifest_sha,
    }


def find_vswhere() -> Path:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")), Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    for root in roots:
        candidate = root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("vswhere.exe not found")


def find_msvc_cmake(vswhere: Path) -> tuple[Path, str, str]:
    vs_paths = subprocess.run([str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    cmake_paths = subprocess.run([str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.CMake.Project", "-find", r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if not vs_paths or not cmake_paths:
        raise FileNotFoundError("MSVC x64 compiler or Visual Studio CMake component missing")
    cmake = Path(cmake_paths[0])
    version = subprocess.run([str(cmake), "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return cmake, vs_paths[0], version


def build_native(cmake: Path) -> dict[str, str]:
    configure = subprocess.run([str(cmake), "-S", str(UNIT_ROOT), "-B", str(BUILD_ROOT), "-G", "Visual Studio 17 2022", "-A", "x64"], cwd=REPO_ROOT, capture_output=True, text=True)
    if configure.returncode:
        raise RuntimeError("V2-1c CMake configure failed: " + configure.stderr[-4000:])
    build = subprocess.run([str(cmake), "--build", str(BUILD_ROOT), "--config", "Release", "--target", "omega_v2_1c_bench", "omega_v2_1c_equivalence", "-j", "8"], cwd=REPO_ROOT, capture_output=True, text=True)
    if build.returncode:
        raise RuntimeError("V2-1c MSVC Release build failed: " + build.stderr[-5000:])
    if not BENCH_EXE.is_file() or not EQUIVALENCE_EXE.is_file():
        raise FileNotFoundError("V2-1c Release executable(s) were not generated")
    return {"configure_stdout": configure.stdout, "configure_stderr": configure.stderr, "build_stdout": build.stdout, "build_stderr": build.stderr}


def run_python_contract_tests() -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(UNIT_ROOT / "tests"), "-v"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10 * 60)
    if result.returncode:
        raise RuntimeError("V2-1c static contract tests failed:\n" + result.stdout + "\n" + result.stderr)
    return {"stdout": result.stdout, "stderr": result.stderr}


def validate_kq_kernel_binding(kq2_build: dict[str, Any]) -> dict[str, Any]:
    expected = kq2_build["candidate02_source_sha256"]
    observed = {path.name: sha256_file(path) for path in candidate02_kernel_paths()}
    binding = {
        "frozen_candidate02_exe_sha256": sha256_file(Path(kq2_build["executable_absolute_path"])),
        "expected_candidate02_exe_sha256": FROZEN_CANDIDATE02_EXE_SHA256,
        "candidate02_kernel_tu_and_header_sha256": observed,
        "candidate02_kq_recorded_sha256": {name: expected[f"src/{name}"] for name in ("kq_candidate2.hpp", "q4_kernel_candidate2.cpp", "full_block_candidate2.cpp")},
        "compile_flags_candidate02_core": kq2_build["compile_flags"],
        "compile_flags_v2_1c_candidate02_core": ["/O2", "/GL", "/arch:AVX2", "/fp:precise", "/W4", "/EHsc", "/LTCG"],
        "compiler_candidate02": kq2_build["compiler"],
        "compiler_v2_1c": "MSVC 19.44.35229.0",
        "toolset_candidate02": kq2_build["visual_studio_installation"],
        "kernel_translation_units_byte_identical": all(observed[name] == expected[f"src/{name}"] for name in observed),
        "frozen_kq_executable_invoked_for_sweep": False,
        "companion_harness_links_same_kernel_translation_units": True,
        "authorized_companion_harness_executable_hash_separate": True,
    }
    if binding["frozen_candidate02_exe_sha256"] != FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_SOURCE_HOLD: frozen candidate_02 executable SHA-256 changed")
    if not binding["kernel_translation_units_byte_identical"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: companion harness candidate_02 kernel TU/header hash mismatch")
    if kq2_build["compiler"] != binding["compiler_v2_1c"] or kq2_build["compile_flags"] != binding["compile_flags_v2_1c_candidate02_core"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: candidate_02 kernel compiler/toolset/flags do not match frozen build identity")
    return binding


def execute_equivalence(weights_path: Path, golden_dir: Path, output_path: Path) -> dict[str, Any]:
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("candidate_02 equivalence output already exists and is immutable")
    environment = os.environ.copy()
    manifest, attempt_config, _ = validate_attempt02()
    environment["OMEGA_V2_1C_SHARD_WEIGHTS"] = ",".join(str(row["h0_v_i"]) for row in attempt_config["selected_workers"])
    process = subprocess.run([str(EQUIVALENCE_EXE), "--verify-frozen-candidate02", str(weights_path), str(golden_dir), str(output_path)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30*60, env=environment)
    (PREFLIGHT_ROOT / "equivalence_stdout.log").write_text(process.stdout, encoding="utf-8", newline="\n")
    (PREFLIGHT_ROOT / "equivalence_stderr.log").write_text(process.stderr, encoding="utf-8", newline="\n")
    if process.returncode:
        raise RuntimeError(f"V2-1c frozen-kernel equivalence failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
    report = read_json(output_path)
    report["attempt02_artifact_manifest_sha256"] = sha256_file(ATTEMPT02_ROOT / "artifact_hashes.json")
    report["candidate02_kq_artifact_manifest_sha256"] = sha256_file(KQ2_ROOT_RESULTS / "artifact_hashes.json")
    report["candidate02_source_weight_state_sha256"] = read_json(KQ2_ROOT_RESULTS / "source_weight_manifest.json")["weight_state_sha256"]
    write_json(output_path, report)
    return report


def compare_kq_timing_sanity(eq: dict[str, Any], kq_native: dict[str, Any]) -> dict[str, Any]:
    timings = eq["timing_sanity"]
    series = {
        "H0_m1": (timings["h0_m1"]["median_seconds"], kq_native["h0_q4"]["m1"]["median_seconds"]),
        "H0_m4": (timings["h0_m4"]["median_seconds"], kq_native["h0_q4"]["m4"]["median_seconds"]),
        "H0_m16": (timings["h0_m16"]["median_seconds"], kq_native["h0_q4"]["m16"]["median_seconds"]),
        "FULL_m1_k1": (timings["full_m1_k1_seconds"], kq_native["full_resident"]["m1_k1"]["median_seconds"]),
        "FULL_m4_k1": (timings["full_m4_k1_seconds"], kq_native["full_resident"]["m4_k1"]["median_seconds"]),
        "FULL_m16_k1": (timings["full_m16_k1_seconds"], kq_native["full_resident"]["m16_k1"]["median_seconds"]),
        "FULL_m8_k4": (timings["full_m8_k4_seconds"], kq_native["full_resident"]["m8_k4_for_s_native"]["median_seconds"]),
    }
    ratios = {name: {"companion_seconds": new, "candidate02_kq_seconds": old, "ratio": new / old, "within_sanity_band": SANITY_RATIO_MIN <= new / old <= SANITY_RATIO_MAX} for name,(new,old) in series.items()}
    return {"allowed_ratio_band": [SANITY_RATIO_MIN,SANITY_RATIO_MAX], "cells": ratios, "pass": all(row["within_sanity_band"] for row in ratios.values()) and timings["no_timed_allocations"] is True}


def run_preflight(cmake: Path, vs_install: str, cmake_version: str, *, force_build: bool) -> dict[str, Any]:
    if not force_build and BENCH_EXE.is_file() and EQUIVALENCE_EXE.is_file():
        build_logs = {"configure_stdout": "REUSED_COMMITTED_BUILD", "configure_stderr": "", "build_stdout": "REUSED_RELEASE_EXECUTABLES", "build_stderr": ""}
    else:
        build_logs = build_native(cmake)
    tests = run_python_contract_tests()
    kq_native = read_json(KQ2_NATIVE)
    attempt_manifest, attempt_config, attempt_sha = validate_attempt02()
    c2_manifest, c2_summary, c2_build, c2_exe_hash = validate_candidate02()
    seal = kq_contract.validate_v2_0_seal()
    provenance = verify_provenance(attempt_sha,sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"))
    binding = validate_kq_kernel_binding(c2_build)
    static_before = source_hashes()
    physical_deps = {str(path.resolve()): sha256_file(path) for path in physical_dependency_paths()}
    candidate2_tus = {str(path.resolve()): sha256_file(path) for path in candidate02_kernel_paths()}
    if static_before!=source_hashes():
        raise RuntimeError("V2_1C_SOURCE_HOLD: source hashes changed before V2-1c preflight")
    payload, weight_info = kq_contract.v2_0_fp32_weight_payload()
    if weight_info["source_weight_value_sha256"] != kq_native["source_weight_sha256"]:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: equivalence source weights differ from candidate_02 KQ")
    preflight_dir = PREFLIGHT_ROOT
    preflight_dir.mkdir(parents=True, exist_ok=True)
    weight_file = preflight_dir / "v2_0_fp32_source_weights.tmp"
    if not weight_file.exists():
        weight_file.write_bytes(payload)
    elif sha256_file(weight_file) != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: preflight weight stream differs from V2-0 source tensors")
    kernel_key=hashlib.sha256(json.dumps(candidate2_tus,sort_keys=True).encode("utf-8")).hexdigest()[:16]
    equivalence_path=preflight_dir/f"candidate02_equivalence_{kernel_key}.json"
    if not equivalence_path.exists():
        equivalence=execute_equivalence(weight_file,KQ2_OUTPUTS,equivalence_path)
    else:
        equivalence=read_json(equivalence_path)
    timing_sanity = compare_kq_timing_sanity(equivalence,kq_native)
    if not timing_sanity["pass"]:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: companion-kernel H0/FULL timing sanity differs materially from sealed candidate_02 KQ")
    if equivalence["pass"] is not True or equivalence["six_cell_scalar_and_determinism_pass"] is not True or not equivalence["kq_sealed_output_comparisons"]["all_available_sealed_outputs_bit_exact"]:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: candidate_02 equivalence/determinism checks failed")
    if source_hashes() != static_before:
        raise RuntimeError("V2_1C_SOURCE_HOLD: V2-1c sources changed during preflight")
    if {str(path.resolve()): sha256_file(path) for path in physical_dependency_paths()} != physical_deps:
        raise RuntimeError("V2_1C_SOURCE_HOLD: physical harness dependencies changed during preflight")
    if {str(path.resolve()): sha256_file(path) for path in candidate02_kernel_paths()} != candidate2_tus:
        raise RuntimeError("V2_1C_SOURCE_HOLD: candidate_02 kernel TUs changed during preflight")
    if c2_build["cmake_version"]!=cmake_version or c2_build["visual_studio_installation"]!=vs_install:
        raise RuntimeError("V2_1C_SOURCE_HOLD: candidate_02 CMake/MSVC installation differs from the companion build")
    preflight_record={
        "v2_1c_source_sha256":static_before,
        "physical_dependency_sha256":physical_deps,
        "attempt02_config":attempt_config,
        "candidate02_exe_sha256":c2_exe_hash,
        "candidate02_artifact_manifest_sha256":sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"),
        "attempt02_artifact_manifest_sha256":attempt_sha,
        "candidate02_build_manifest":c2_build,
        "candidate02_kernel_tu_sha256":candidate2_tus,
        "kernel_binding":binding,
        "candidate02_equivalence":equivalence,
        "candidate02_equivalence_path_abs":str(equivalence_path.resolve()),
        "timing_sanity":timing_sanity,
        "source_weight_manifest":{"source":weight_info["source"],"seed":weight_info["seed"],"weight_state_sha256":weight_info["source_weight_value_sha256"],"weight_stream_sha256":weight_info["source_weight_stream_sha256"],"weight_stream_bytes":weight_info["source_weight_stream_bytes"]},
        "python_contract_tests_status":"PASS",
        "v2_0_seal_status":seal["seal"]["terminal_status"],
    }
    return {
        "source_provenance":provenance,
        "source_sha256":static_before,
        "physical_dependency_sha256":physical_deps,
        "candidate02_kernel_tu_sha256":candidate2_tus,
        "attempt02_preservation":{"artifact_manifest_sha256":attempt_sha,"artifact_count":len(attempt_manifest["artifacts"]),"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in attempt_config["selected_workers"]],"frozen_v_i":[row["h0_v_i"] for row in attempt_config["selected_workers"]]},
        "candidate02_manifest_sha256":sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"),
        "preflight_record":preflight_record,
        "v2_0_seal":seal["seal"],
        "v2_1c_source_sha256": static_before,
        "candidate02_exe_sha256": c2_exe_hash,
        "candidate02_artifact_manifest_sha256": sha256_file(KQ2_ROOT_RESULTS / "artifact_hashes.json"),
        "attempt02_artifact_manifest_sha256": attempt_sha,
        "attempt02_artifact_manifest": attempt_manifest,
        "attempt02_config": attempt_config,
        "candidate02_summary": c2_summary,
        "candidate02_build_manifest": c2_build,
        "kernel_binding": binding,
        "candidate02_equivalence": equivalence,
        "candidate02_equivalence_path":str(equivalence_path.resolve()),
        "timing_sanity": timing_sanity,
        "source_weight_manifest": {"source": weight_info["source"], "seed": weight_info["seed"], "weight_state_sha256": weight_info["source_weight_value_sha256"], "weight_stream_sha256": weight_info["source_weight_stream_sha256"], "weight_stream_bytes": weight_info["source_weight_stream_bytes"]},
        "build_logs": build_logs,
        "visual_studio_installation": vs_install,
        "cmake_version": cmake_version,
        "bench_executable_sha256": sha256_file(BENCH_EXE),
        "equivalence_executable_sha256": sha256_file(EQUIVALENCE_EXE),
        "equivalence_executable_abs": str(EQUIVALENCE_EXE.resolve()),
        "bench_executable_abs": str(BENCH_EXE.resolve()),
        "cmake_path":str(cmake),
        "python_contract_tests": {"status": "PASS", **tests},
    }


def attempt02_preservation_after(expected_sha: str) -> dict[str, Any]:
    manifest,config,observed_sha=validate_attempt02()
    if observed_sha!=expected_sha:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest hash changed during V2-1c")
    return {"artifact_manifest_sha256": observed_sha,"artifact_count":len(manifest["artifacts"]),"artifact_hashes_verified_after_v2_1c":True,"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in config["selected_workers"]],"frozen_v_i":[row["h0_v_i"] for row in config["selected_workers"]]}


def make_v2_1c_tests(preflight: dict[str,Any], native_status: dict[str,Any], hardware: dict[str,Any], config: dict[str,Any],
                     raw_rows: list[dict[str,Any]], complete: bool, gates: dict[str,Any] | None,
                     attempt_after: dict[str,Any], comparison: dict[str,Any], conformance: dict[str,Any]) -> list[dict[str,Any]]:
    binding=preflight["kernel_binding"]
    eq=preflight["candidate02_equivalence"]
    selected=hardware.get("selected_workers",[])
    frozen_ids=[int(row["windows_cpu_set_id"]) for row in preflight["attempt02_config"]["selected_workers"]]
    frozen_vi=[float(row["h0_v_i"]) for row in preflight["attempt02_config"]["selected_workers"]]
    config_ids=[int(row["windows_cpu_set_id"]) for row in config.get("selected_workers",[])]
    config_vi=[float(row["h0_v_i"]) for row in config.get("selected_workers",[])]
    observed_qpc=hardware.get("hardware",{}).get("qpc",{}).get("frequency")
    attempt02_qpc=preflight["attempt02_config"].get("qpc_frequency")
    schedule_ok=(config_ids==frozen_ids and config_vi==frozen_vi and config.get("schedule_seed")==20260929 and config.get("measurement_blocks")==5 and config.get("samples_per_block")==21 and config.get("warmups_per_cell_variant")==10 and config.get("cell_count")==72 and config.get("qpc_frequency")==attempt02_qpc and observed_qpc==attempt02_qpc)
    status=native_status.get("status")
    q4_scalar=hardware.get("q4_scalar_reference_test",{}).get("pass") is True
    full_scalar=hardware.get("full_block_scalar_reference_test",{}).get("pass") is True
    abc=hardware.get("abc_correctness",{})
    abc_correct=abc.get("A_B_C_bitwise_equal") is True and all(abc.get(key) is True for key in ("A_finite","B_finite","C_finite","B_values_equal_A","B_round_storages_disjoint"))
    candidate_correct=(eq.get("pass") is True and q4_scalar and full_scalar and abc_correct and hardware.get("native_test_report",{}).get("status")=="READY")
    no_tune=binding["kernel_translation_units_byte_identical"] and len(preflight["candidate02_kernel_tu_sha256"])==3 and source_hashes()==preflight["v2_1c_source_sha256"]
    rho_pass=bool(gates) and gates["rho_minimum_pass"] is True
    causal_pass=bool(gates) and gates["causal_minimum_pass"] is True
    c_greater=bool(gates) and gates["c_C_gt_c_A"] is True
    matrix_pass=bool(gates) and gates["matrixization_minimum_pass"] is True
    stability_pass=bool(gates) and gates["measurement_stability_hold"] is False
    m1_ok=bool(gates) and all(key in gates["m1_diagnostic_non_gate"] for key in ("512","640"))
    compare_ok=all(key in comparison for key in ("attempt02_gate_metrics","v2_1c_gate_metrics","attempt02_vs_v2_1c_non_gate"))
    report_expected=conformance.get("phase")=="PHYSICAL_T0" and conformance.get("claim_scope")=="physical residency only; no runtime/backend/trainability verdict"
    test_rows=[
        (TEST_NAMES[0],binding["frozen_candidate02_exe_sha256"]==FROZEN_CANDIDATE02_EXE_SHA256 and binding["kernel_translation_units_byte_identical"],binding),
        (TEST_NAMES[1],attempt_after["artifact_hashes_verified_after_v2_1c"] and attempt_after["artifact_manifest_sha256"]==preflight["attempt02_artifact_manifest_sha256"],attempt_after),
        (TEST_NAMES[2],complete and config.get("cell_count")==72,{"cell_count":72,"native_status":status,"complete":complete}),
        (TEST_NAMES[3],schedule_ok,{"cpu_set_ids":config_ids,"v_i":config_vi,"schedule_seed":config.get("schedule_seed"),"qpc_frequency":hardware.get("qpc_frequency"),"matches_attempt02":schedule_ok}),
        (TEST_NAMES[4],candidate_correct,{"equivalence_pass":eq.get("pass"),"six_cell_scalar_and_determinism_pass":eq.get("six_cell_scalar_and_determinism_pass"),"native_preflight_status":hardware.get("status"),"q4_scalar_pass":q4_scalar,"full_scalar_pass":full_scalar,"abc_correctness":abc}),
        (TEST_NAMES[5],no_tune,{"candidate02_kernel_tu_hashes_unchanged":binding["kernel_translation_units_byte_identical"],"v2_1c_source_unchanged":source_hashes()==preflight["v2_1c_source_sha256"]}),
        (TEST_NAMES[6],rho_pass,{"rho_resident":gates.get("rho_resident") if gates else None,"threshold":0.50}),
        (TEST_NAMES[7],causal_pass,{"delta_CB":gates.get("delta_CB") if gates else None,"threshold":0.25,"c_C_gt_c_A":gates.get("c_C_gt_c_A") if gates else None}),
        (TEST_NAMES[8],c_greater,{"c_C_gt_c_A":gates.get("c_C_gt_c_A") if gates else None}),
        (TEST_NAMES[9],matrix_pass,{"G_matrix":gates.get("G_matrix") if gates else None,"threshold":1.50}),
        (TEST_NAMES[10],stability_pass,{"noisy_primary_stability_cells":gates.get("noisy_primary_stability_cells") if gates else None,"hold_if_noisy_cells_ge":2}),
        (TEST_NAMES[11],m1_ok,gates.get("m1_diagnostic_non_gate") if gates else {}),
        (TEST_NAMES[12],compare_ok,comparison),
        (TEST_NAMES[13],report_expected,conformance),
    ]
    if [name for name,_,_ in test_rows]!=TEST_NAMES:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: test inventory differs from approved V2-1c names")
    return [{"name":name,"status":"PASS" if condition else "FAIL","detail":detail} for name,condition,detail in test_rows]


def build_comparison(attempt02: dict[str,Any],current: dict[str,Any],cells: dict[tuple[int,int,int,str],dict[str,Any]])->dict[str,Any]:
    old_gates=attempt02.get("gate_metrics",{})
    def marginal(metric_cells: dict, variant:str, k:int, m:int=4)->float:
        return (metric_cells[(512,m,k,variant)]["median_seconds"]-metric_cells[(512,m,1,variant)]["median_seconds"])/(k-1)
    new_c={name:marginal(cells,name,8) for name in "ABC"}
    new_rho=new_c["A"]/new_c["B"]
    new_delta=abs(new_c["C"]-new_c["B"])/new_c["B"]
    return {
        "attempt02_gate_metrics":{key:old_gates.get(key) for key in ("rho_resident","delta_CB","G_matrix","c_A_K8","c_B_K8","c_C_K8")},
        "v2_1c_gate_metrics":{key:current.get(key) for key in ("rho_resident","delta_CB","G_matrix","c_A_K8","c_B_K8","c_C_K8")},
        "attempt02_vs_v2_1c_non_gate":{
            "attempt02_terminal_status":attempt02.get("terminal_status"),
            "v2_1c_status":"SWEEP_COMPLETE",
            "rho_resident_delta":current.get("rho_resident")-old_gates["rho_resident"] if old_gates.get("rho_resident") is not None and current.get("rho_resident") is not None else None,
            "delta_CB_delta":current.get("delta_CB")-old_gates["delta_CB"] if old_gates.get("delta_CB") is not None and current.get("delta_CB") is not None else None,
            "G_matrix_delta":current.get("G_matrix")-old_gates["G_matrix"] if old_gates.get("G_matrix") is not None and current.get("G_matrix") is not None else None,
            "A_d512_m4_K8_latency_ratio_attempt02_over_v2_1c":old_gates.get("primary_cell_median_times",{}).get("A_m4_K8"),
        },
    }


def classify_v2_1c(native_status: dict[str,Any],complete: bool,gates: dict[str,Any]|None,tests: list[dict[str,Any]])->tuple[str,str]:
    status=native_status.get("status","UNKNOWN")
    if status!="SWEEP_COMPLETE":
        return "MEASUREMENT_INVALID",native_status.get("reason",native_status.get("hold_reason",f"native status={status}"))
    if not complete:
        return "MEASUREMENT_INVALID","the 72-cell raw sweep is incomplete or contains missing round groups"
    if gates is None:
        return "MEASUREMENT_INVALID","primary gate calculations could not be computed"
    if gates["measurement_stability_hold"]:
        return "MEASUREMENT_STABILITY_HOLD",f"{gates['noisy_primary_stability_cells']}/9 primary cells exceed R_MAD 0.15"
    if not gates["rho_minimum_pass"]:
        return "RESIDENCY_NOT_SUPPORTED_ON_THIS_HOST_REGIME",f"rho_resident={gates['rho_resident']} > 0.50"
    if not gates["causal_minimum_pass"]:
        return "V2_1C_CAUSAL_CONTROL_FAIL",f"delta_CB={gates['delta_CB']}; c_C_gt_c_A={gates['c_C_gt_c_A']}"
    if not gates["matrixization_minimum_pass"]:
        return "V2_1C_MATRIXIZATION_GATE_FAIL",f"G_matrix={gates['G_matrix']} < 1.50"
    failures=[row["name"] for row in tests if row["status"]!="PASS"]
    if failures:
        return "V2_1C_ACCEPTANCE_HOLD",f"contractual test failures: {failures}"
    return "OMEGA_V2_1C_RESIDENCY_ONLY_PASS","all three primary point-estimate gates and supporting tests pass"


def build_conformance(native: dict[str,Any],preflight: dict[str,Any],attempt_before: dict[str,Any],attempt_after: dict[str,Any],
                      gates: dict[str,Any]|None,terminal: str,tests: list[dict[str,Any]])->dict[str,Any]:
    return {
        "OMEGA_CONFORMANCE_BLOCK":{
            "unit":"OMEGA-V2-1c-RESIDENCY-ONLY",
            "phase":"PHYSICAL_T0",
            "authority":{"md310":"GO","candidate02_kq_commit":"2e8ac1207e742e659aab34ef0ab4bf51d43dbe55","v2_1c_implementation_commit":git("rev-parse","HEAD")},
            "prior_evidence":{"attempt_02":{"status":"VALID_MEASUREMENT","terminal":"RESIDENCY_GATE_FAIL","immutable":True,"artifact_manifest_sha256":attempt_before["artifact_manifest_sha256"]},"candidate02_kq":{"status":"KERNEL_SCIENTIFICALLY_QUALIFIED+PROJECT_NATIVE_SPEED_GATE_FAIL","immutable":True,"exe_sha256":FROZEN_CANDIDATE02_EXE_SHA256}},
            "kernel_binding":{"candidate_id":"KQ2_ROW_TILE4_SLOT2_FUSED","frozen_kq_executable_sha256":FROZEN_CANDIDATE02_EXE_SHA256,"companion_physical_harness_executable_sha256":preflight["bench_executable_sha256"],"candidate02_kernel_tus_byte_identical":preflight["kernel_binding"]["kernel_translation_units_byte_identical"],"candidate02_compile_flags_and_toolset_match":True,"no_kernel_source_changes":True},
            "protocol":{"cells":72,"d":[512,640],"m":[1,4,8,16],"m1_role":"CONTROL_ONLY","K":[1,4,8],"variants":["A","B","C"],"blocks":5,"samples_per_block":21,"warmups":10,"schedule_seed":20260929,"workers_cpu_set_ids":FROZEN_WORKER_IDS,"v_i_source":"attempt_02 benchmark_config.json; frozen, not reselected"},
            "gates":{"primary_cell":{"d":512,"m":4,"K":8},"rho_max":0.50,"delta_CB_max":0.25,"require_c_C_gt_c_A":True,"G_matrix_min":1.50,"pytorch_S_native_gate":"EXCLUDED","m1":"diagnostic only"},
            "gate_results":gates,
            "attempt02_preservation":{"manifest_sha256_before":attempt_before["artifact_manifest_sha256"],"manifest_sha256_after":attempt_after["artifact_manifest_sha256"],"preserved":attempt_before["artifact_manifest_sha256"]==attempt_after["artifact_manifest_sha256"]},
            "candidate_decision":terminal,
            "claim_scope":"physical residency only; no runtime/backend/trainability verdict",
            "forbidden_claims":["Q4 backend competitiveness","PyTorch speed parity","trainability","language quality","GPU release","T3 release"],
            "deviations":["Companion V2-1c harness executable is distinct from the frozen KQ exe; it links byte-identical candidate_02 kernel TUs/header with matching toolset/flags. The frozen KQ exe remains unchanged. Direct exact-output comparison is limited to its three sealed output files; all six required cells are independently checked against scalar oracle and deterministic repeat."],
            "global_status":"CONFORMANCE_HOLD","gpu":"HOLD","T3":"HOLD","status":"CONFORMANCE_HOLD",
        }
    }


def make_attempt02_comparison(attempt02_summary: dict[str,Any],attempt02_cells: dict[tuple[int,int,int,str],dict[str,Any]],
                              current_gates: dict[str,Any],current_cells: dict[tuple[int,int,int,str],dict[str,Any]])->dict[str,Any]:
    old=attempt02_summary.get("gate_metrics",{})
    def time(cells:dict[tuple[int,int,int,str],dict[str,Any]],m:int,k:int,v:str)->float:
        value=cells[(512,m,k,v)]["median_seconds"]
        if value is None: raise ValueError(f"missing comparison cell d512/m{m}/K{k}/{v}")
        return float(value)
    metrics={
        "attempt02":{key:old.get(key) for key in ("rho_resident","delta_CB","G_matrix","c_A_K8","c_B_K8","c_C_K8")},
        "v2_1c":{key:current_gates.get(key) for key in ("rho_resident","delta_CB","G_matrix","c_A_K8","c_B_K8","c_C_K8")},
        "non_gate_comparisons":{
            "A_d512_m4_K8_latency_ratio_attempt02_over_v2_1c":time(attempt02_cells,4,8,"A")/time(current_cells,4,8,"A"),
            "A_d512_m16_K8_latency_ratio_attempt02_over_v2_1c":time(attempt02_cells,16,8,"A")/time(current_cells,16,8,"A"),
            "A_d512_m4_K1_throughput_attempt02":attempt02_cells[(512,4,1,"A")]["effective_MAC_per_s"],
            "A_d512_m4_K1_throughput_v2_1c":current_cells[(512,4,1,"A")]["effective_MAC_per_s"],
            "attempt02_status":attempt02_summary.get("terminal_status"),
            "v2_1c_scope":"physical residency only",
        },
    }
    return metrics


def load_attempt02_cells() -> tuple[dict[str,Any],dict[tuple[int,int,int,str],dict[str,Any]]]:
    summary=read_json(ATTEMPT02_ROOT/"summary_metrics.json")
    cells={(row["d"],row["m"],row["K"],row["variant"]):row for row in summary["cells"]}
    return summary,cells


def result_paths_for_hashing(preflight:dict[str,Any])->list[Path]:
    paths=[*(UNIT_ROOT/name for name in source_files()),*physical_dependency_paths(),*candidate02_kernel_paths(),
        Path(preflight["candidate02_build_manifest"]["executable_absolute_path"]),V2_0_ROOT/"V2_0_RESULT_SEAL.json",
        ATTEMPT02_ROOT/"artifact_hashes.json",ATTEMPT02_ROOT/"benchmark_config.json",KQ2_ROOT_RESULTS/"artifact_hashes.json",KQ2_NATIVE,
        KQ2_ROOT_RESULTS/"summary_metrics.json",KQ2_ROOT_RESULTS/"build_manifest.json",KQ2_ROOT_RESULTS/"source_weight_manifest.json",KQ2_ROOT_RESULTS/"test_report.json",
        *(KQ2_OUTPUTS/name for name in ("candidate2_full_m4_k1.bin","candidate2_full_m16_k1.bin","candidate2_full_m8_k4.bin")),
        BENCH_EXE,EQUIVALENCE_EXE,Path(preflight["candidate02_equivalence_path_abs"]),
        RESULTS_ROOT/"build_manifest.json",RESULTS_ROOT/"hardware_preflight.json",RESULTS_ROOT/"q4_physical_ledger.json",
        RESULTS_ROOT/"worker_shard_manifest.json",RESULTS_ROOT/"benchmark_config.json",RESULTS_ROOT/"native_test_report.json",
        RESULTS_ROOT/"native_run_status.json",RESULTS_ROOT/"raw_measurements.csv",RESULTS_ROOT/"summary_metrics.json",
        RESULTS_ROOT/"test_report.json",RESULTS_ROOT/"OMEGA_CONFORMANCE_BLOCK.yaml",RESULTS_ROOT/"attempt02_preservation.json",
        RESULTS_ROOT/"candidate02_kernel_binding.json",RESULTS_ROOT/"candidate02_equivalence.json",RESULTS_ROOT/"attempt02_vs_v2_1c_comparison.json",
        RESULTS_ROOT/"preflight_record.json",
        RESULTS_ROOT/"native_stdout.log",RESULTS_ROOT/"native_stderr.log"]
    return [path for path in paths if path.is_file()]


def format_report(summary:dict[str,Any],hashes:dict[str,dict[str,Any]])->str:
    gates=summary.get("gate_metrics")
    lines=[
        "# OMEGA-V2-1c-RESIDENCY-ONLY — PHYSICAL_T0", "",
        f"- terminal_status: `{summary['terminal_status']}`",
        f"- implementation_commit: `{summary['source_provenance']['implementation_commit']}`",
        f"- results_root_abs: `{RESULTS_ROOT.resolve()}`",
        f"- frozen candidate_02 KQ exe SHA-256: `{FROZEN_CANDIDATE02_EXE_SHA256}`",
        f"- companion physical harness SHA-256: `{summary['build_manifest']['bench_executable_sha256']}`",
        "- candidate_02 kernel TUs/header: byte-identical to the frozen KQ build; same compiler/toolset/flags.",
        "- attempt_02: immutable and hash-verified before/after.",
        "- attempt_03 (V2-1b): `NOT_RUN`; A/B/C were run only as the authorized V2-1c 72-cell physical replication.",
        "- Scope: physical residency on the measured i7-13700F only. No backend-quality, PyTorch-speed, trainability, language, GPU-release, or T3 verdict.", "",
        "## Frozen-kernel equivalence and timing sanity",
        f"- exact KQ saved output comparisons: `{summary['equivalence']['kq_sealed_output_comparisons']}`",
        f"- six d512 cells scalar-oracle max errors/tolerances and deterministic repeats: see `candidate02_equivalence.json`; overall `{summary['equivalence']['six_cell_scalar_and_determinism_pass']}`.",
        f"- pre-sweep H0/FULL timing sanity against KQ run_01 (0.70–1.30 ratio band): `{summary['preflight']['timing_sanity']}`.",
        "- Limitation: candidate_02 KQ sealed files contained exact outputs only for d512,m4,K1; d512,m16,K1; and d512,m8,K4. No sealed outputs existed for m1,K1 or K4 at m={1,4,16}; those six requested cells were checked against scalar oracle and repeated-run determinism instead.", "",
        "## Protocol",
        "- Cells: d={512,640}, m={1,4,8,16}, K={1,4,8}, A/B/C = 72; 10 warmups, 5 blocks × 21 samples; seed 20260929.",
        f"- v_i and fixed CPU-set workers reused from attempt_02: `{summary['attempt02_preservation']['frozen_worker_cpu_set_ids']}` with v_i `{summary['attempt02_preservation']['frozen_v_i']}`.",
        f"- native status: `{summary['native_status']}`; valid raw rows: `{summary['raw_row_count']}`; complete 72-cell sweep: `{summary['sweep_complete']}`.", "",
        "## Primary gates (point estimates; unchanged MD/304–305 thresholds)",
    ]
    if gates:
        lines.extend([
            f"- c_A(8): `{gates['c_A_K8']:.9g}` s/round; c_B(8): `{gates['c_B_K8']:.9g}`; c_C(8): `{gates['c_C_K8']:.9g}`",
            f"- rho_resident = c_A/c_B: `{gates['rho_resident']:.9g}`; maximum 0.50: `{gates['rho_minimum_pass']}`",
            f"- delta_CB: `{gates['delta_CB']:.9g}`; maximum 0.25 and c_C>c_A: `{gates['causal_minimum_pass']}` / `{gates['c_C_gt_c_A']}`",
            f"- G_matrix: `{gates['G_matrix']:.9g}`; minimum 1.50: `{gates['matrixization_minimum_pass']}`",
            f"- stability: `{gates['noisy_primary_stability_cells']}/9` noisy; HOLD threshold ≥2: `{gates['measurement_stability_hold']}`",
            f"- diagnostic only, m1: `{gates['m1_diagnostic_non_gate']}`",
        ])
    else:
        lines.append("- Gate metrics not computed because the native sweep/analysis did not complete.")
    lines.extend(["", "## Attempt_02 vs V2-1c (non-gate comparison)", f"- `{summary.get('attempt02_vs_v2_1c_comparison')}`", "", "## Tests"])
    lines.extend(f"- {row['status']}: `{row['name']}`" for row in summary["test_report"]["tests"])
    lines.extend(["", "## SHA-256 (complete)"])
    lines.extend(f"- `{name}`: `{record['sha256']}` ({record['size_bytes']} bytes; `{record['absolute_path']}`)" for name,record in sorted(hashes.items()))
    lines.extend(["", "STOP: V2-1c is residency-only. No GPU, Phase B, language scoring, training, or T3 action occurred.", ""])
    return "\n".join(lines)


def finalize_results(preflight:dict[str,Any],attempt_before:dict[str,Any])->dict[str,Any]:
    native_status=read_json(RESULTS_ROOT/"native_run_status.json")
    hardware=read_json(RESULTS_ROOT/"hardware_preflight.json")
    hardware["native_test_report"]=read_json(RESULTS_ROOT/"native_test_report.json")
    q4=read_json(RESULTS_ROOT/"q4_physical_ledger.json")
    shards=read_json(RESULTS_ROOT/"worker_shard_manifest.json")
    config=read_json(RESULTS_ROOT/"benchmark_config.json")
    raw_rows,raw_fields=v21_analysis.parse_raw(RESULTS_ROOT/"raw_measurements.csv")
    complete=v21_analysis.check_sweep_completeness(raw_rows,native_status.get("status","UNKNOWN"))
    cells_list,cells,discarded=v21_analysis.summarize_samples(raw_rows)
    complete=complete and len(cells_list)==72 and not discarded and all(cell["n_valid"]==105 for cell in cells_list)
    gates=None
    analysis_error=None
    if complete:
        try:
            gates=v21_analysis.compute_gate_metrics(cells)
            gates["diagnostic_bootstrap90"]=v21_analysis.bootstrap_gate_diagnostics(raw_rows)
            for d in (512,640):
                def marginal_m1(variant:str)->float:
                    return (cells[(d,1,8,variant)]["median_seconds"]-cells[(d,1,1,variant)]["median_seconds"])/7.0
                c_a,c_b,c_c=(marginal_m1(v) for v in "ABC")
                gates.setdefault("m1_diagnostic_non_gate",{})[str(d)]={
                    "c_A_K8":c_a,"c_B_K8":c_b,"c_C_K8":c_c,
                    "rho_A_over_B":c_a/c_b if c_b>0 else None,
                    "delta_CB":abs(c_c-c_b)/c_b if c_b>0 else None,
                    "c_C_gt_c_A":c_c>c_a,"gate":False,
                }
        except Exception as error:
            analysis_error=f"{type(error).__name__}: {error}"
            gates=None
    summary={
        "schema":"omega-v2-1c-residency-only-summary-v1",
        "native_status":native_status,"hardware_preflight":hardware,"q4_physical_ledger":q4,
        "worker_shard_manifest":shards,"benchmark_config":config,"raw_fieldnames":raw_fields,
        "raw_row_count":len(raw_rows),
        "raw_sample_count":len({(r["d"],r["m"],r["K"],r["variant"],r["is_warmup"],r["block_id"],r["sample_id"]) for r in raw_rows}),
        "cells":cells_list,"discarded_samples_with_predeclared_reasons":discarded,
        "sweep_complete":complete,"analysis_error":analysis_error,"gate_metrics":gates,
        "source_provenance":preflight["source_provenance"],"build_manifest":read_json(RESULTS_ROOT/"build_manifest.json"),
        "preflight":preflight["preflight_record"],"equivalence":preflight["preflight_record"]["candidate02_equivalence"],
        "attempt02_preservation":{**attempt_before,"artifact_hashes_verified_before_v2_1c":True},
        "attempt02_vs_v2_1c_comparison":None,
        "global_status":"CONFORMANCE_HOLD","gpu":"HOLD","T3":"HOLD",
    }
    attempt02_summary,attempt02_cells=load_attempt02_cells()
    comparison=make_attempt02_comparison(attempt02_summary,attempt02_cells,gates or {},cells) if gates else {"attempt02_gate_metrics":attempt02_summary.get("gate_metrics",{}),"v2_1c_gate_metrics":None,"attempt02_vs_v2_1c_non_gate":{}}
    summary["attempt02_vs_v2_1c_comparison"]=comparison
    provisional_conformance=build_conformance(native_status,preflight["preflight_record"],attempt_before,attempt_before,gates,None,[])
    tests=make_v2_1c_tests(preflight["preflight_record"],native_status,hardware,config,raw_rows,complete,gates,attempt_before,comparison,provisional_conformance)
    terminal,reason=classify_v2_1c(native_status,complete,gates,tests)
    conformance=build_conformance(native_status,preflight["preflight_record"],attempt_before,attempt02_preservation_after(attempt_before),gates,terminal,tests)
    test_report={"schema":"omega-v2-1c-residency-only-test-report-v1","test_count":len(tests),"pass_count":sum(r["status"]=="PASS" for r in tests),"fail_count":sum(r["status"]=="FAIL" for r in tests),"skip_count":sum(r["status"]=="SKIP" for r in tests),"tests":tests}
    summary["terminal_status"]=terminal
    summary["terminal_reason"]=reason
    summary["test_report"]=test_report
    summary["attempt02_preservation"]=attempt02_preservation_after(attempt_before)
    write_json(RESULTS_ROOT/"summary_metrics.json",summary)
    write_json(RESULTS_ROOT/"test_report.json",test_report)
    write_json(RESULTS_ROOT/"OMEGA_CONFORMANCE_BLOCK.yaml",conformance)
    write_json(RESULTS_ROOT/"attempt02_preservation.json",summary["attempt02_preservation"])
    write_json(RESULTS_ROOT/"attempt02_vs_v2_1c_comparison.json",comparison)
    return summary


def attempt02_preservation_after(before:dict[str,Any])->dict[str,Any]:
    manifest,config,manifest_sha=validate_attempt02()
    if manifest_sha!=before["artifact_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest changed during V2-1c")
    return {"artifact_manifest_path_abs":str((ATTEMPT02_ROOT/"artifact_hashes.json").resolve()),"artifact_manifest_sha256":manifest_sha,"artifact_count":len(manifest["artifacts"]),"artifact_hashes_verified_before_v2_1c":True,"artifact_hashes_verified_after_v2_1c":True,"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in config["selected_workers"]],"frozen_v_i":[row["h0_v_i"] for row in config["selected_workers"]]}


def seal_results(summary:dict[str,Any])->dict[str,Any]:
    report_path=RESULTS_ROOT/"OMEGA_V2_1C_RESIDENCY_ONLY_REPORT.md"
    source_hash_map={**summary["build_manifest"]["source_sha256"],**summary["build_manifest"]["physical_dependency_sha256"],**summary["build_manifest"]["candidate02_kernel_tu_sha256"]}
    paths=result_paths_for_hashing(summary["preflight"]["preflight_record"])
    hashes={str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size,"absolute_path":str(path.resolve())} for path in paths}
    write_json(RESULTS_ROOT/"source_hashes.json",source_hash_map)
    hashes[str((RESULTS_ROOT/"source_hashes.json").resolve())]={"sha256":sha256_file(RESULTS_ROOT/"source_hashes.json"),"size_bytes":(RESULTS_ROOT/"source_hashes.json").stat().st_size,"absolute_path":str((RESULTS_ROOT/"source_hashes.json").resolve())}
    (report_path).write_text(format_report(summary,hashes),encoding="utf-8",newline="\n")
    report_sha=sha256_file(report_path)
    sidecar=RESULTS_ROOT/"OMEGA_V2_1C_RESIDENCY_ONLY_REPORT.md.sha256"
    sidecar.write_text(report_sha+"\n",encoding="ascii",newline="\n")
    final_paths=[*paths,RESULTS_ROOT/"source_hashes.json",report_path,sidecar]
    artifact_manifest={"schema":"omega-v2-1c-artifact-hashes-v1","implementation_commit":summary["source_provenance"]["implementation_commit"],"candidate02_kq_manifest_sha256":summary["preflight"]["preflight_record"]["candidate02_artifact_manifest_sha256"],"attempt02_manifest_sha256":summary["attempt02_preservation"]["artifact_manifest_sha256"],"report_self_sha256":report_sha,"artifacts":{str(p.resolve()):{"sha256":sha256_file(p),"size_bytes":p.stat().st_size} for p in final_paths if p.is_file()}}
    write_json(RESULTS_ROOT/"artifact_hashes.json",artifact_manifest)
    verified=all(Path(path).is_file() and sha256_file(Path(path))==rec["sha256"] for path,rec in artifact_manifest["artifacts"].items()) and sha256_file(report_path)==sidecar.read_text(encoding="ascii").strip()
    write_json(RESULTS_ROOT/"artifact_hashes_verified.json",{"verified":verified,"artifact_count":len(artifact_manifest["artifacts"])})
    if not verified: raise RuntimeError("V2_1C_RESULT_HASH_FAILURE")
    return {"report_sha256":report_sha,"artifact_manifest_sha256":sha256_file(RESULTS_ROOT/"artifact_hashes.json"),"artifact_count":len(artifact_manifest["artifacts"]),"verified":verified}


def run_native_sweep(preflight:dict[str,Any],attempt_before:dict[str,Any])->dict[str,Any]:
    if RESULTS_ROOT.exists(): raise FileExistsError(f"V2-1c attempt_01 result slot already exists: {RESULTS_ROOT}")
    RESULTS_ROOT.mkdir(parents=True,exist_ok=False)
    build_manifest={
        "schema":"omega-v2-1c-native-build-manifest-v1",
        "implementation_commit":preflight["source_provenance"]["implementation_commit"],
        "source_provenance":preflight["source_provenance"],
        "candidate02_frozen_binding":preflight["preflight_record"]["kernel_binding"],
        "candidate02_equivalence":preflight["preflight_record"]["candidate02_equivalence"],
        "candidate02_timing_sanity":preflight["preflight_record"]["timing_sanity"],
        "source_sha256":preflight["source_sha256"],"physical_dependency_sha256":preflight["physical_dependency_sha256"],
        "candidate02_kernel_tu_sha256":preflight["candidate02_kernel_tu_sha256"],
        "generator":"Visual Studio 17 2022 x64","configuration":"Release","compiler":"MSVC 19.44.35229.0",
        "cmake_path":preflight["cmake_path"],"cmake_version":preflight["cmake_version"],"visual_studio_installation":preflight["visual_studio_installation"],
        "kernel_compile_flags":["/O2","/GL","/arch:AVX2","/fp:precise","/W4","/EHsc","/LTCG"],
        "companion_bench_executable_abs":str(BENCH_EXE.resolve()),"companion_bench_executable_sha256":sha256_file(BENCH_EXE),
        "equivalence_executable_abs":str(EQUIVALENCE_EXE.resolve()),"equivalence_executable_sha256":sha256_file(EQUIVALENCE_EXE),
        "attempt02_manifest_sha256":attempt_before["artifact_manifest_sha256"],
        "candidate02_kq_manifest_sha256":preflight["candidate02_manifest_sha256"],
        "frozen_candidate02_exe_sha256":FROZEN_CANDIDATE02_EXE_SHA256,
        "build_logs":preflight["build_logs"],
        "equivalence_report":preflight["preflight_record"]["candidate02_equivalence"],
    }
    write_json(RESULTS_ROOT/"build_manifest.json",build_manifest)
    write_json(RESULTS_ROOT/"preflight_record.json",preflight["preflight_record"])
    equivalence_path=Path(preflight["preflight_record"]["candidate02_equivalence_path_abs"])
    (RESULTS_ROOT/"candidate02_equivalence.json").write_bytes(equivalence_path.read_bytes())
    write_json(RESULTS_ROOT/"candidate02_kernel_binding.json",preflight["preflight_record"]["kernel_binding"])
    attempt02_after=attempt02_preservation_after(attempt_before)
    write_json(RESULTS_ROOT/"attempt02_preservation.json",attempt02_after)
    native_env=os.environ.copy()
    native_env["OMEGA_V2_1_RESULTS_ROOT"]=str(RESULTS_ROOT.resolve())
    native_env["OMEGA_V2_1C_ATTEMPT02_CPU_SET_IDS"]=",".join(str(value) for value in FROZEN_WORKER_IDS)
    native_env["OMEGA_V2_1C_ATTEMPT02_V_I"]=",".join(str(value) for value in attempt_before["frozen_h0_shard_weights"])
    try:
        native_process=subprocess.run([str(BENCH_EXE),"--run"],cwd=REPO_ROOT,capture_output=True,text=True,timeout=8*60*60,env=native_env)
        stdout,stderr,returncode=native_process.stdout,native_process.stderr,native_process.returncode
    except subprocess.TimeoutExpired as error:
        stdout=str(error.stdout or "");stderr=str(error.stderr or "");returncode=124
        write_json(RESULTS_ROOT/"native_run_status.json",{"status":"MEASUREMENT_INVALID","reason":"V2-1c physical harness exceeded 8h CPU timeout","returncode":returncode,"sweep_started":True})
    (RESULTS_ROOT/"native_stdout.log").write_text(stdout,encoding="utf-8",newline="\n")
    (RESULTS_ROOT/"native_stderr.log").write_text(stderr,encoding="utf-8",newline="\n")
    if not (RESULTS_ROOT/"native_run_status.json").is_file():
        write_json(RESULTS_ROOT/"native_run_status.json",{"status":"MEASUREMENT_INVALID","reason":f"native harness exited {returncode} without native_run_status.json","returncode":returncode})
    return {"native_returncode":returncode,"stdout":stdout,"stderr":stderr}


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only",action="store_true",help="build and verify frozen candidate_02 equivalence/timing sanity without opening the V2-1c sweep result slot")
    args=parser.parse_args()
    if os.name!="nt" or sys.platform!="win32":
        raise RuntimeError("V2_1C_MEASUREMENT_INVALID: unit requires native Windows")
    if not args.preflight_only and RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-1c attempt_01 is immutable and already exists: {RESULTS_ROOT}")

    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    c2_manifest,c2_summary,c2_build,c2_exe_sha=validate_candidate02()
    provenance=verify_provenance(attempt_sha,sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"))
    vswhere=find_vswhere()
    cmake,vs_install,cmake_version=find_msvc_cmake(vswhere)
    force_build=args.preflight_only or not (BENCH_EXE.is_file() and EQUIVALENCE_EXE.is_file())
    preflight=run_preflight(cmake,vs_install,cmake_version,force_build=force_build)
    preflight["source_provenance"]=provenance
    preflight_record={
        "candidate02_exe_sha256":preflight["candidate02_exe_sha256"],
        "candidate02_artifact_manifest_sha256":preflight["candidate02_artifact_manifest_sha256"],
        "attempt02_artifact_manifest_sha256":preflight["attempt02_artifact_manifest_sha256"],
        "candidate02_build_manifest":preflight["candidate02_build_manifest"],
        "candidate02_kernel_tu_sha256":preflight["candidate02_kernel_tu_sha256"],
        "kernel_binding":preflight["kernel_binding"],
        "candidate02_equivalence":preflight["candidate02_equivalence"],
        "candidate02_equivalence_path_abs":preflight["candidate02_equivalence_path_abs"],
        "timing_sanity":preflight["timing_sanity"],
        "v2_1c_source_sha256":preflight["source_sha256"],
        "physical_dependency_sha256":preflight["physical_dependency_sha256"],
        "attempt02_config":preflight["attempt02_config"],
        "candidate02_kq_summary":preflight["candidate02_summary"],
        "preflight_tests":preflight["python_contract_tests"],
    }
    preflight["preflight_record"]=preflight_record
    if args.preflight_only:
        print(json.dumps({"preflight_only":True,"measurement_started":False,"v2_1c_source_commit":provenance["implementation_commit"],"frozen_candidate02_exe_sha256":c2_exe_sha,"candidate02_kernel_tus_byte_identical":preflight["kernel_binding"]["kernel_translation_units_byte_identical"],"equivalence_pass":preflight["candidate02_equivalence"]["pass"],"six_cell_scalar_and_determinism_pass":preflight["candidate02_equivalence"]["six_cell_scalar_and_determinism_pass"],"all_available_sealed_outputs_bit_exact":preflight["candidate02_equivalence"]["kq_sealed_output_comparisons"]["all_available_sealed_outputs_bit_exact"],"timing_sanity_pass":preflight["timing_sanity"]["pass"],"attempt02_hash_verified":True,"attempt03_run":False,"results_slot_free":not RESULTS_ROOT.exists()},indent=2,sort_keys=True))
        return 0

    attempt_before={"artifact_manifest_path_abs":str((ATTEMPT02_ROOT/"artifact_hashes.json").resolve()),"artifact_manifest_sha256":attempt_sha,"artifact_count":len(attempt_manifest["artifacts"]),"artifact_hashes_verified_before_v2_1c":True,"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in attempt_config["selected_workers"]],"frozen_h0_shard_weights":[row["h0_v_i"] for row in attempt_config["selected_workers"]],"frozen_worker_logical_ids":[row["logical_processor_id"] for row in attempt_config["selected_workers"]]}
    run_native_sweep(preflight,attempt_before)
    attempt_after=attempt02_preservation_after(attempt_before)
    if source_hashes()!=preflight["source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: unit source changed during timed physical sweep")
    if sha256_file(Path(c2_build["executable_absolute_path"]))!=FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_STOP: candidate_02 frozen executable changed during V2-1c")
    if {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=preflight["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 kernel TUs changed during sweep")
    summary=finalize_results(preflight,attempt_before)
    seal=seal_results(summary)
    print(json.dumps({"unit":"OMEGA-V2-1c-RESIDENCY-ONLY","terminal_status":summary["terminal_status"],"gate_metrics":summary.get("gate_metrics"),"test_count":summary["test_report"]["test_count"],"test_pass_count":summary["test_report"]["pass_count"],"test_fail_count":summary["test_report"]["fail_count"],"test_skip_count":summary["test_report"]["skip_count"],"results_root_abs":str(RESULTS_ROOT.resolve()),"report_sha256":seal["report_sha256"],"artifact_manifest_sha256":seal["artifact_manifest_sha256"],"artifact_count":seal["artifact_count"],"artifact_hashes_verified":seal["verified"],"attempt03_run":False,"pytorch_s_native_gate":"EXCLUDED"},indent=2,sort_keys=True))
    return 0 if summary["terminal_status"]=="OMEGA_V2_1C_RESIDENCY_ONLY_PASS" and seal["verified"] else 1


if __name__=="__main__":
    raise SystemExit(main())
