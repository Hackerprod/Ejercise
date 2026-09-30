"""Preflight, verify candidate_02 binding, and run the authorized V2-1c sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
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
BINDING_PREFLIGHT_RESULTS = UNIT_ROOT / "results" / "binding_preflight"
CORE_SELECTION_RESULTS = UNIT_ROOT / "results" / "core_selection_preflight"
CORRECTNESS_PREFLIGHT_RESULTS = UNIT_ROOT / "results" / "correctness_preflight"
PREFLIGHT_RESULTS = UNIT_ROOT / "results" / "sealed_preflight"
BUILD_ROOT = UNIT_ROOT / "build"
PREFLIGHT_ROOT = BUILD_ROOT / "preflight"
BENCH_EXE = BUILD_ROOT / "Release" / "omega_v2_1c_bench.exe"
EQUIVALENCE_EXE = BENCH_EXE
CORRECTNESS_EXE = BUILD_ROOT / "Release" / "omega_v2_1c_correctness.exe"
FROZEN_CANDIDATE02_EXE_SHA256 = "be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f"
FROZEN_WORKER_IDS = [266, 264, 258, 270]
SANITY_RATIO_MIN = 0.90
SANITY_RATIO_MAX = 1.10
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
    files = [".gitignore", "CMakeLists.txt", "OMEGA_V2_1C_RESIDENCY_ONLY_SPEC.md", "scripts/run_v2_1c.py", "scripts/v2_1c_phases.py"]
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
    build = subprocess.run([str(cmake), "--build", str(BUILD_ROOT), "--config", "Release", "--target", "omega_v2_1c_bench", "omega_v2_1c_correctness", "-j", "8"], cwd=REPO_ROOT, capture_output=True, text=True)
    if build.returncode:
        raise RuntimeError("V2-1c MSVC Release build failed: " + build.stderr[-5000:])
    if not BENCH_EXE.is_file() or not CORRECTNESS_EXE.is_file():
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
    c2_cmake=(KQ2_ROOT/"CMakeLists.txt").read_text(encoding="utf-8")
    c1c_cmake=(UNIT_ROOT/"CMakeLists.txt").read_text(encoding="utf-8")
    def target_values(content:str,setting:str,target:str)->list[str]:
        match=re.search(rf"{setting}\({re.escape(target)}\s+(?:PUBLIC|PRIVATE)\s+([^)]+)\)",content)
        if not match:raise RuntimeError(f"V2_1C_SOURCE_HOLD: missing {setting} for {target}")
        return match.group(1).split()
    c2_defs=target_values(c2_cmake,"target_compile_definitions","omega_v2_1b_candidate_02_core")
    c1c_defs=target_values(c1c_cmake,"target_compile_definitions","omega_v2_1c_candidate02_core")
    c2_opts=target_values(c2_cmake,"target_compile_options","omega_v2_1b_candidate_02_core")
    c1c_opts=target_values(c1c_cmake,"target_compile_options","omega_v2_1c_candidate02_core")
    c2_link=target_values(c2_cmake,"target_link_options","omega_v2_1b_candidate_02_core")
    c1c_link=target_values(c1c_cmake,"target_link_options","omega_v2_1c_candidate02_core")
    binding = {
        "frozen_candidate02_exe_sha256": sha256_file(Path(kq2_build["executable_absolute_path"])),
        "expected_candidate02_exe_sha256": FROZEN_CANDIDATE02_EXE_SHA256,
        "candidate02_kernel_tu_and_header_sha256": observed,
        "candidate02_kq_recorded_sha256": {name: expected[f"src/{name}"] for name in ("kq_candidate2.hpp", "q4_kernel_candidate2.cpp", "full_block_candidate2.cpp")},
        "compile_flags_candidate02_core": kq2_build["compile_flags"],
        "compile_flags_v2_1c_candidate02_core": ["/O2", "/GL", "/arch:AVX2", "/fp:precise", "/W4", "/EHsc", "/LTCG"],
        "compile_defines_candidate02_core":c2_defs,"compile_defines_v2_1c_candidate02_core":c1c_defs,
        "compile_options_candidate02_core":c2_opts,"compile_options_v2_1c_candidate02_core":c1c_opts,
        "link_options_candidate02_core":c2_link,"link_options_v2_1c_candidate02_core":c1c_link,
        "compiler_candidate02": kq2_build["compiler"],
        "compiler_v2_1c": "MSVC 19.44.35229.0",
        "toolset_candidate02": kq2_build["visual_studio_installation"],
        "kernel_translation_units_byte_identical": all(observed[name] == expected[f"src/{name}"] for name in observed),
        "frozen_kq_executable_invoked_for_sweep": False,
        "companion_harness_links_same_kernel_translation_units": True,
        "authorized_companion_harness_executable_hash_separate": True,
        "compute_defines_match":c2_defs==c1c_defs,
        "compute_options_match":c2_opts==c1c_opts,
        "compute_link_options_match":c2_link==c1c_link,
    }
    if binding["frozen_candidate02_exe_sha256"] != FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_SOURCE_HOLD: frozen candidate_02 executable SHA-256 changed")
    if not binding["kernel_translation_units_byte_identical"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: companion harness candidate_02 kernel TU/header hash mismatch")
    if not binding["compute_defines_match"] or not binding["compute_options_match"] or not binding["compute_link_options_match"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: candidate_02 compute compile/link defines or options differ in the companion")
    if kq2_build["compiler"] != binding["compiler_v2_1c"] or kq2_build["compile_flags"] != binding["compile_flags_v2_1c_candidate02_core"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: candidate_02 kernel compiler/toolset/flags do not match frozen build identity")
    return binding


def execute_equivalence(mode: str, weights_path: Path, golden_dir: Path, output_path: Path,
                        worker_cpu_set_ids:list[int]|None=None,shard_weights:list[float]|None=None) -> dict[str, Any]:
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"candidate_02 {mode} output already exists and is immutable: {output_path}")
    if mode not in ("--correctness-preflight","--binding-preflight"):
        raise ValueError(f"unsupported candidate_02 equivalence mode: {mode}")
    environment = os.environ.copy()
    manifest, attempt_config, _ = validate_attempt02()
    ids=worker_cpu_set_ids if worker_cpu_set_ids is not None else [int(row["windows_cpu_set_id"]) for row in attempt_config["selected_workers"]]
    weights=shard_weights if shard_weights is not None else [float(row["h0_v_i"]) for row in attempt_config["selected_workers"]]
    environment["OMEGA_V2_1C_WORKER_CPU_SET_IDS"]=",".join(str(value) for value in ids)
    environment["OMEGA_V2_1C_SHARD_WEIGHTS"] = ",".join(str(value) for value in weights)
    process = subprocess.run([str(EQUIVALENCE_EXE), mode, str(weights_path), str(golden_dir), str(output_path)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30*60, env=environment)
    (PREFLIGHT_ROOT / f"{mode[2:]}_stdout.log").write_text(process.stdout, encoding="utf-8", newline="\n")
    (PREFLIGHT_ROOT / f"{mode[2:]}_stderr.log").write_text(process.stderr, encoding="utf-8", newline="\n")
    if output_path.is_file():
        report=read_json(output_path)
    else:
        report={"schema":"omega-v2-1c-equivalence-process-failure-v1","pass":False,"reason":"companion equivalence executable did not produce JSON","stdout":process.stdout,"stderr":process.stderr}
    report["process_returncode"]=process.returncode
    report["attempt02_artifact_manifest_sha256"] = sha256_file(ATTEMPT02_ROOT / "artifact_hashes.json")
    report["candidate02_kq_artifact_manifest_sha256"] = sha256_file(KQ2_ROOT_RESULTS / "artifact_hashes.json")
    report["candidate02_source_weight_state_sha256"] = read_json(KQ2_ROOT_RESULTS / "source_weight_manifest.json")["weight_state_sha256"]
    write_json(output_path, report)
    return report


def execute_physical_correctness(worker_cpu_set_ids: list[int], shard_weights: list[float]) -> dict[str, Any]:
    environment=os.environ.copy()
    if len(worker_cpu_set_ids) != 4 or len(shard_weights) != 4:
        raise ValueError("V2-1c physical correctness requires four newly selected CPU-set IDs and v_i weights")
    environment["OMEGA_V2_1C_SELECTED_CPU_SET_IDS"] = ",".join(str(value) for value in worker_cpu_set_ids)
    environment["OMEGA_V2_1C_SELECTED_V_I"] = ",".join(str(value) for value in shard_weights)
    _,config,_=validate_attempt02()
    environment["OMEGA_V2_1C_SHARD_WEIGHTS"]=",".join(str(row["h0_v_i"]) for row in config["selected_workers"])
    result=subprocess.run([str(CORRECTNESS_EXE)],cwd=REPO_ROOT,capture_output=True,text=True,timeout=60*60,env=environment)
    (PREFLIGHT_ROOT/"physical_correctness_stdout.log").write_text(result.stdout,encoding="utf-8",newline="\n")
    (PREFLIGHT_ROOT/"physical_correctness_stderr.log").write_text(result.stderr,encoding="utf-8",newline="\n")
    if result.returncode:
        raise RuntimeError(f"V2-1c no-timing correctness preflight failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    report = json.loads(result.stdout)
    write_json(PREFLIGHT_ROOT / "physical_correctness_preflight.json", report)
    return report


def compare_binding_timing(binding: dict[str, Any], kq_native: dict[str, Any]) -> dict[str, Any]:
    rows={row["cell"]:row["median_seconds"] for row in binding["cells"]}
    series={
        "d512_m4_K1":(rows["d512_m4_K1"],kq_native["full_resident"]["m4_k1"]["median_seconds"]),
        "d512_m16_K1":(rows["d512_m16_K1"],kq_native["full_resident"]["m16_k1"]["median_seconds"]),
        "d512_m8_K4":(rows["d512_m8_K4"],kq_native["full_resident"]["m8_k4_for_s_native"]["median_seconds"]),
    }
    ratios={name:{"companion_seconds":new,"candidate02_kq_seconds":old,"ratio":new/old,"within_sanity_band":SANITY_RATIO_MIN<=new/old<=SANITY_RATIO_MAX} for name,(new,old) in series.items()}
    return {"authorized_pre_sweep_binding_only":True,"allowed_ratio_band":[SANITY_RATIO_MIN,SANITY_RATIO_MAX],"cells":ratios,"no_timed_allocations":binding["no_timed_allocations"],"pass":all(row["within_sanity_band"] for row in ratios.values()) and binding["no_timed_allocations"] is True}


def create_preflight(cmake:Path,vs_install:str,cmake_version:str)->dict[str,Any]:
    if CORRECTNESS_PREFLIGHT_RESULTS.exists() or PREFLIGHT_RESULTS.exists():
        raise FileExistsError("V2-1c preflight result directory already exists and is immutable")
    build_logs=build_native(cmake)
    python_tests=run_python_contract_tests()
    kq_native=read_json(KQ2_NATIVE)
    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    attempt02_hardware=read_json(ATTEMPT02_ROOT/"hardware_preflight.json")
    attempt02_ledger=read_json(ATTEMPT02_ROOT/"q4_physical_ledger.json")
    c2_manifest,c2_summary,c2_build,c2_exe_hash=validate_candidate02()
    seal=kq_contract.validate_v2_0_seal()
    provenance=verify_provenance(attempt_sha,sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"))
    binding=validate_kq_kernel_binding(c2_build)
    static_before=source_hashes()
    physical_deps={str(path.resolve()):sha256_file(path) for path in physical_dependency_paths()}
    candidate2_tus={str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}
    payload,weight_info=kq_contract.v2_0_fp32_weight_payload()
    if weight_info["source_weight_value_sha256"]!=kq_native["source_weight_sha256"]:
        raise RuntimeError("V2_1C_ACCEPTANCE_HOLD: source stream differs from candidate_02 KQ weights")
    PREFLIGHT_ROOT.mkdir(parents=True,exist_ok=True)
    weight_file=PREFLIGHT_ROOT/"v2_0_fp32_source_weights.tmp"
    weight_file.write_bytes(payload)

    c2_correctness_path=PREFLIGHT_ROOT/"candidate02_correctness_preflight.json"
    c2_correctness=execute_equivalence("--correctness-preflight",weight_file,KQ2_OUTPUTS,c2_correctness_path)
    c2_correctness_ok=(c2_correctness.get("pass") is True and c2_correctness.get("six_cell_scalar_and_determinism_pass") is True and c2_correctness.get("kq_sealed_output_comparisons",{}).get("all_available_sealed_outputs_bit_exact") is True)

    env=os.environ.copy()
    env["OMEGA_V2_1C_SHARD_WEIGHTS"]=",".join(str(row["h0_v_i"]) for row in attempt_config["selected_workers"])
    physical_correctness_process=subprocess.run([str(CORRECTNESS_EXE)],cwd=REPO_ROOT,capture_output=True,text=True,timeout=60*60,env=env)
    (PREFLIGHT_ROOT/"physical_correctness_stdout.log").write_text(physical_correctness_process.stdout,encoding="utf-8",newline="\n")
    (PREFLIGHT_ROOT/"physical_correctness_stderr.log").write_text(physical_correctness_process.stderr,encoding="utf-8",newline="\n")
    try:
        physical_correctness=json.loads(physical_correctness_process.stdout)
    except (json.JSONDecodeError,TypeError):
        physical_correctness={"schema":"omega-v2-1c-physical-correctness-preflight-v1","pass":False,"reason":"correctness executable produced no valid JSON","stdout":physical_correctness_process.stdout,"stderr":physical_correctness_process.stderr}
    physical_correctness["process_returncode"]=physical_correctness_process.returncode
    physical_correctness_ok=(physical_correctness_process.returncode==0 and physical_correctness.get("pass") is True)

    if source_hashes()!=static_before or {str(path.resolve()):sha256_file(path) for path in physical_dependency_paths()}!=physical_deps or {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=candidate2_tus:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source/TU hashes changed during no-timing correctness preflight")
    correctness_tests=[
        ("test_v2_1c_python_contract_tests",True),
        ("test_v2_1c_candidate02_kernel_frozen_hash",binding["kernel_translation_units_byte_identical"] and c2_exe_hash==FROZEN_CANDIDATE02_EXE_SHA256),
        ("test_v2_1c_candidate02_kq_correctness_preflight",c2_correctness.get("pass") is True),
        ("test_v2_1c_d512_d640_scalar_and_72_cell_abc_correctness",physical_correctness_ok),
        ("test_v2_1c_attempt02_preserved",attempt_sha==sha256_file(ATTEMPT02_ROOT/"artifact_hashes.json")),
    ]
    correctness_test_report={"schema":"omega-v2-1c-correctness-preflight-tests-v1","test_count":len(correctness_tests),"pass_count":sum(passed for _,passed in correctness_tests),"fail_count":sum(not passed for _,passed in correctness_tests),"skip_count":0,"tests":[{"name":name,"status":"PASS" if passed else "FAIL"} for name,passed in correctness_tests]}
    correctness_record={
        "schema":"omega-v2-1c-correctness-preflight-v1","phase":"correctness-only","timing_executed":False,"timed_72_cell_sweep_started":False,
        "source_provenance":provenance,"v2_1c_source_sha256":static_before,"physical_dependency_sha256":physical_deps,
        "attempt02_artifact_manifest_sha256":attempt_sha,"attempt02_config":attempt_config,
        "candidate02_artifact_manifest_sha256":sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"),
        "candidate02_build_manifest":c2_build,"candidate02_exe_sha256":c2_exe_hash,"candidate02_kernel_tu_sha256":candidate2_tus,
        "candidate02_kernel_binding":binding,"candidate02_kq_correctness":c2_correctness,
        "physical_abc_correctness":physical_correctness,"source_weight_manifest":{"source":weight_info["source"],"seed":weight_info["seed"],"weight_state_sha256":weight_info["source_weight_value_sha256"],"weight_stream_sha256":weight_info["source_weight_stream_sha256"],"weight_stream_bytes":weight_info["source_weight_stream_bytes"]},
        "compiler":"MSVC 19.44.35229.0","visual_studio_installation":vs_install,"cmake_path":str(cmake),"cmake_version":cmake_version,
        "candidate02_compute_compile_flags":c2_build["compile_flags"],"candidate02_compute_defines":["WIN32_LEAN_AND_MEAN","NOMINMAX","_WIN32_WINNT=0x0A00","OMEGA_V2_1_AVX2=1","OMEGA_V2_1B_CANDIDATE_ID=2"],
        "companion_bench_exe_abs":str(BENCH_EXE.resolve()),"companion_bench_exe_sha256":sha256_file(BENCH_EXE),
        "equivalence_exe_abs":str(EQUIVALENCE_EXE.resolve()),"equivalence_exe_sha256":sha256_file(EQUIVALENCE_EXE),
        "correctness_exe_abs":str(CORRECTNESS_EXE.resolve()),"correctness_exe_sha256":sha256_file(CORRECTNESS_EXE),
        "build_logs":build_logs,"python_contract_tests":{"status":"PASS","stdout":python_tests["stdout"],"stderr":python_tests["stderr"]},
    }
    correctness_status="PASS" if correctness_test_report["fail_count"]==0 else "CORRECTNESS_PREFLIGHT_HOLD"
    correctness_record["preflight_status"]=correctness_status
    CORRECTNESS_PREFLIGHT_RESULTS.mkdir(parents=True,exist_ok=False)
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"correctness_preflight.json",correctness_record)
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"test_report.json",correctness_test_report)
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"candidate02_kq_correctness.json",c2_correctness)
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"physical_abc_correctness.json",physical_correctness)
    attempt_preservation={"artifact_manifest_sha256":attempt_sha,"artifact_count":len(attempt_manifest["artifacts"]),"verified":True,"selected_workers":attempt_config["selected_workers"]}
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"attempt02_preservation.json",attempt_preservation)
    correctness_paths=[*(UNIT_ROOT/name for name in source_files()),*physical_dependency_paths(),*candidate02_kernel_paths(),
        V2_0_ROOT/"V2_0_RESULT_SEAL.json",ATTEMPT02_ROOT/"artifact_hashes.json",ATTEMPT02_ROOT/"benchmark_config.json",ATTEMPT02_ROOT/"hardware_preflight.json",ATTEMPT02_ROOT/"q4_physical_ledger.json",ATTEMPT02_ROOT/"raw_measurements.csv",
        KQ2_ROOT_RESULTS/"artifact_hashes.json",KQ2_ROOT_RESULTS/"native_kq_candidate_02.json",KQ2_ROOT_RESULTS/"build_manifest.json",KQ2_ROOT_RESULTS/"summary_metrics.json",KQ2_ROOT_RESULTS/"source_weight_manifest.json",
        *(KQ2_OUTPUTS/name for name in ("candidate2_full_m4_k1.bin","candidate2_full_m16_k1.bin","candidate2_full_m8_k4.bin")),
        Path(c2_build["executable_absolute_path"]),BENCH_EXE,EQUIVALENCE_EXE,CORRECTNESS_EXE,
        CORRECTNESS_PREFLIGHT_RESULTS/"correctness_preflight.json",CORRECTNESS_PREFLIGHT_RESULTS/"test_report.json",CORRECTNESS_PREFLIGHT_RESULTS/"candidate02_kq_correctness.json",CORRECTNESS_PREFLIGHT_RESULTS/"physical_abc_correctness.json",CORRECTNESS_PREFLIGHT_RESULTS/"attempt02_preservation.json"]
    correctness_hashes={str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in correctness_paths if path.is_file()}
    correctness_report=["# V2-1c correctness preflight (no timing)","",f"- status: `{correctness_status}`",f"- implementation commit: `{provenance['implementation_commit']}`",f"- frozen candidate_02 exe SHA-256: `{c2_exe_hash}`",f"- candidate_02 TUs byte-identical: `{binding['kernel_translation_units_byte_identical']}`",f"- candidate_02 KQ six-cell scalar/golden-output checks: `{c2_correctness.get('pass')}`",f"- d512/d640 scalar K1 + toy K1/4/8 + all 72-cell A/B/C repeat/equivalence checks: `{physical_correctness.get('pass')}`","- binding timing not started; 72-cell timed sweep not started.","","## Tests"]
    correctness_report.extend(f"- {row['status']}: `{row['name']}`" for row in correctness_test_report["tests"])
    correctness_report.extend(["","## SHA-256"])
    correctness_report.extend(f"- `{path}`: `{record['sha256']}` ({record['size_bytes']} bytes)" for path,record in sorted(correctness_hashes.items()))
    correctness_report_path=CORRECTNESS_PREFLIGHT_RESULTS/"V2_1C_CORRECTNESS_PREFLIGHT.md"
    correctness_report_path.write_text("\n".join(correctness_report)+"\n",encoding="utf-8",newline="\n")
    correctness_report_sha=sha256_file(correctness_report_path)
    correctness_sidecar=CORRECTNESS_PREFLIGHT_RESULTS/"V2_1C_CORRECTNESS_PREFLIGHT.md.sha256"
    correctness_sidecar.write_text(correctness_report_sha+"\n",encoding="ascii",newline="\n")
    correctness_paths.extend([correctness_report_path,correctness_sidecar])
    correctness_manifest={"schema":"omega-v2-1c-correctness-preflight-artifact-hashes-v1","implementation_commit":provenance["implementation_commit"],"report_self_sha256":correctness_report_sha,"artifacts":{str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in correctness_paths if path.is_file()}}
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes.json",correctness_manifest)
    correctness_verified=all(Path(path).is_file() and sha256_file(Path(path))==row["sha256"] for path,row in correctness_manifest["artifacts"].items()) and sha256_file(correctness_report_path)==correctness_sidecar.read_text(encoding="ascii").strip()
    write_json(CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes_verified.json",{"verified":correctness_verified,"artifact_count":len(correctness_manifest["artifacts"])})
    if not correctness_verified:raise RuntimeError("V2_1C_CORRECTNESS_PREFLIGHT_HASH_FAILURE")
    return {"phase":"correctness-only","preflight_status":correctness_status,"correctness_report_abs":str(correctness_report_path.resolve()),"correctness_report_sha256":correctness_report_sha,"artifact_manifest_abs":str((CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes.json").resolve()),"artifact_manifest_sha256":sha256_file(CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes.json"),"artifact_count":len(correctness_manifest["artifacts"]),"hashes_verified":correctness_verified,"correctness_record":correctness_record}

    binding_path=PREFLIGHT_ROOT/"companion_binding_preflight.json"
    if c2_correctness_ok and physical_correctness_ok:
        binding_report=execute_equivalence("--binding-preflight",weight_file,KQ2_OUTPUTS,binding_path)
        binding_sanity=compare_binding_timing(binding_report,kq_native) if binding_report.get("cells") else {"authorized_pre_sweep_binding_only":True,"allowed_ratio_band":[SANITY_RATIO_MIN,SANITY_RATIO_MAX],"cells":{},"no_timed_allocations":False,"pass":False,"reason":"binding executable produced no cell samples"}
    else:
        binding_report={"schema":"omega-v2-1c-binding-preflight-v1","pass":False,"skipped":True,"reason":"correctness preflight failed; binding timing was not started"}
        binding_sanity={"authorized_pre_sweep_binding_only":True,"allowed_ratio_band":[SANITY_RATIO_MIN,SANITY_RATIO_MAX],"cells":{},"no_timed_allocations":False,"pass":False,"skipped":True}
    binding_pass=(binding_report.get("pass") is True and binding_report.get("timing_protocol",{}).get("warmups")==10 and binding_report.get("timing_protocol",{}).get("samples")==31 and binding_sanity.get("pass") is True)
    if source_hashes()!=static_before or {str(path.resolve()):sha256_file(path) for path in physical_dependency_paths()}!=physical_deps or {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=candidate2_tus:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source/TU hashes changed during preflight")

    preflight_checks=[
        ("test_v2_1c_python_contract_tests",True),
        ("test_v2_1c_candidate02_kernel_frozen_hash",binding["kernel_translation_units_byte_identical"] and c2_exe_hash==FROZEN_CANDIDATE02_EXE_SHA256),
        ("test_v2_1c_candidate02_kq_correctness_preflight",c2_correctness_ok),
        ("test_v2_1c_d512_d640_scalar_and_72_cell_abc_correctness",physical_correctness_ok),
        ("test_v2_1c_companion_binding_three_sealed_cells",binding_pass),
    ]
    preflight_tests={"schema":"omega-v2-1c-preflight-test-report-v1","test_count":len(preflight_checks),"pass_count":sum(ok for _,ok in preflight_checks),"fail_count":sum(not ok for _,ok in preflight_checks),"skip_count":sum(bool(binding_report.get("skipped")) and name=="test_v2_1c_companion_binding_three_sealed_cells" for name,_ in preflight_checks),"tests":[{"name":name,"status":"PASS" if ok else ("SKIP" if binding_report.get("skipped") and name=="test_v2_1c_companion_binding_three_sealed_cells" else "FAIL")} for name,ok in preflight_checks]}
    preflight_record={
        "schema":"omega-v2-1c-preflight-v1",
        "source_provenance":provenance,
        "v2_1c_source_sha256":static_before,
        "physical_dependency_sha256":physical_deps,
        "attempt02_config":attempt_config,
        "attempt02_artifact_manifest_sha256":attempt_sha,
        "attempt02_benchmark_config_sha256":sha256_file(ATTEMPT02_ROOT/"benchmark_config.json"),
        "attempt02_hardware_preflight_sha256":sha256_file(ATTEMPT02_ROOT/"hardware_preflight.json"),
        "attempt02_q4_physical_ledger_sha256":sha256_file(ATTEMPT02_ROOT/"q4_physical_ledger.json"),
        "attempt02_hardware_summary":{"cpu_model":attempt02_hardware.get("hardware",{}).get("cpu_model"),"qpc_frequency":attempt02_hardware.get("hardware",{}).get("qpc",{}).get("frequency"),"coordinator_cpu_set_id":attempt02_hardware.get("hardware",{}).get("coordinator_cpu_set_id")},
        "attempt02_protocol":{"cell_count":attempt_config["cell_count"],"d":attempt_config["d"],"m":attempt_config["m"],"K":attempt_config["K"],"variants":attempt_config["variants"],"warmups":attempt_config["warmups_per_cell_variant"],"blocks":attempt_config["measurement_blocks"],"samples_per_block":attempt_config["samples_per_block"],"schedule_seed":attempt_config["schedule_seed"],"qpc_frequency":attempt_config["qpc_frequency"],"eviction_method":attempt_config["eviction_method"],"eviction_effectiveness_E":attempt_config["eviction_effectiveness_E"],"B_pool_by_d_k":attempt02_ledger.get("b_pool_by_d_k",[]),"required_b_pool_bytes":attempt02_ledger.get("required_b_pool_bytes"),"llc_bytes":attempt02_ledger.get("hardware_llc_bytes")},
        "attempt02_worker_v_i":attempt_config["selected_workers"],
        "candidate02_artifact_manifest_sha256":sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"),
        "candidate02_build_manifest":c2_build,
        "candidate02_exe_sha256":c2_exe_hash,
        "candidate02_kernel_tu_sha256":candidate2_tus,
        "candidate02_kernel_binding":binding,
        "candidate02_kq_correctness_preflight":c2_correctness,
        "candidate02_kq_correctness_report_path_abs":str(c2_correctness_path.resolve()),
        "physical_abc_correctness_preflight":physical_correctness,
        "physical_correctness_preflight_pass":physical_correctness_ok,
        "physical_correctness_executable_abs":str(CORRECTNESS_EXE.resolve()),
        "physical_correctness_report_path_abs":str((PREFLIGHT_RESULTS/"physical_correctness_preflight.json").resolve()),
        "companion_binding_preflight":binding_report,
        "companion_binding_report_path_abs":str(binding_path.resolve()),
        "companion_binding_timing_sanity":binding_sanity,
        "companion_binding_preflight_pass":binding_pass,
        "companion_binding_timing_report_path_abs":str((PREFLIGHT_RESULTS/"companion_binding_timing_sanity.json").resolve()),
        "source_weight_manifest":{"source":weight_info["source"],"seed":weight_info["seed"],"weight_state_sha256":weight_info["source_weight_value_sha256"],"weight_stream_sha256":weight_info["source_weight_stream_sha256"],"weight_stream_bytes":weight_info["source_weight_stream_bytes"]},
        "compiler":"MSVC 19.44.35229.0","visual_studio_installation":vs_install,"cmake_path":str(cmake),"cmake_version":cmake_version,
        "compile_flags_compute":["/O2","/GL","/arch:AVX2","/fp:precise","/W4","/EHsc","/LTCG"],
        "compute_defines":["WIN32_LEAN_AND_MEAN","NOMINMAX","_WIN32_WINNT=0x0A00","OMEGA_V2_1_AVX2=1","OMEGA_V2_1B_CANDIDATE_ID=2"],
        "companion_only_defines":["OMEGA_V2_1C_RESIDENCY_ONLY=1","select_p_cores_by_h0=select_p_cores_by_h0_attempt02"],
        "candidate02_exe_sha256":c2_exe_hash,
        "candidate02_exe_abs":str(Path(c2_build["executable_absolute_path"]).resolve()),
        "companion_bench_exe_abs":str(BENCH_EXE.resolve()),
        "companion_bench_exe_sha256":sha256_file(BENCH_EXE),
        "companion_equivalence_exe_abs":str(EQUIVALENCE_EXE.resolve()),
        "companion_equivalence_exe_sha256":sha256_file(EQUIVALENCE_EXE),
        "companion_correctness_exe_abs":str(CORRECTNESS_EXE.resolve()),
        "companion_correctness_exe_sha256":sha256_file(CORRECTNESS_EXE),
        "build_logs":build_logs,
        "python_static_contract_tests":{"status":"PASS","stdout":python_tests["stdout"],"stderr":python_tests["stderr"]},
        "pre_sweep_timing_scope":"binding preflight only: exactly three KQ sealed cells, 10 warmups,31 samples; no H0/FULL timing sanity sweep",
        "timed_72_cell_sweep_started":False,
        "attempt03_run":False,
        "pytorch_s_native_gate":"EXCLUDED",
        "preflight_status":"PASS" if preflight_tests["fail_count"]==0 and preflight_tests["skip_count"]==0 else "PREFLIGHT_HOLD",
        "preflight_tests":preflight_tests,
    }
    PREFLIGHT_RESULTS.mkdir(parents=True,exist_ok=False)
    write_json(PREFLIGHT_RESULTS/"preflight_record.json",preflight_record)
    write_json(PREFLIGHT_RESULTS/"preflight_tests.json",preflight_tests)
    write_json(PREFLIGHT_RESULTS/"candidate02_correctness_preflight.json",c2_correctness)
    write_json(PREFLIGHT_RESULTS/"physical_correctness_preflight.json",physical_correctness)
    write_json(PREFLIGHT_RESULTS/"companion_binding_preflight.json",binding_report)
    write_json(PREFLIGHT_RESULTS/"companion_binding_timing_sanity.json",binding_sanity)
    write_json(PREFLIGHT_RESULTS/"attempt02_preservation.json",{"artifact_manifest_sha256":attempt_sha,"artifact_count":len(attempt_manifest["artifacts"]),"verified":True,"selected_workers":attempt_config["selected_workers"]})
    pre_paths=[*(UNIT_ROOT/name for name in source_files()),*physical_dependency_paths(),*candidate02_kernel_paths(),
        Path(c2_build["executable_absolute_path"]),V2_0_ROOT/"V2_0_RESULT_SEAL.json",ATTEMPT02_ROOT/"artifact_hashes.json",ATTEMPT02_ROOT/"benchmark_config.json",ATTEMPT02_ROOT/"hardware_preflight.json",ATTEMPT02_ROOT/"q4_physical_ledger.json",ATTEMPT02_ROOT/"worker_shard_manifest.json",ATTEMPT02_ROOT/"summary_metrics.json",ATTEMPT02_ROOT/"raw_measurements.csv",ATTEMPT02_ROOT/"OMEGA_V2_1_REPORT.md",
        KQ2_ROOT_RESULTS/"artifact_hashes.json",KQ2_ROOT_RESULTS/"native_kq_candidate_02.json",KQ2_ROOT_RESULTS/"build_manifest.json",KQ2_ROOT_RESULTS/"summary_metrics.json",
        *(KQ2_OUTPUTS/name for name in ("candidate2_full_m4_k1.bin","candidate2_full_m16_k1.bin","candidate2_full_m8_k4.bin")),
        BENCH_EXE,EQUIVALENCE_EXE,CORRECTNESS_EXE,
        PREFLIGHT_RESULTS/"preflight_record.json",PREFLIGHT_RESULTS/"preflight_tests.json",PREFLIGHT_RESULTS/"candidate02_correctness_preflight.json",PREFLIGHT_RESULTS/"physical_correctness_preflight.json",
        PREFLIGHT_RESULTS/"companion_binding_preflight.json",PREFLIGHT_RESULTS/"companion_binding_timing_sanity.json",PREFLIGHT_RESULTS/"attempt02_preservation.json"]
    pre_hashes={str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size,"absolute_path":str(path.resolve())} for path in pre_paths if path.is_file()}
    pre_report=["# V2-1c candidate_02 preflight seal","",f"- preflight_status: `{preflight_record['preflight_status']}`",f"- tests: `{preflight_tests['pass_count']}/{preflight_tests['test_count']}` PASS; `{preflight_tests['fail_count']}` FAIL; `{preflight_tests['skip_count']}` SKIP",f"- v2_1c_source_commit: `{provenance['implementation_commit']}`",f"- frozen_candidate02_exe_sha256: `{c2_exe_hash}`",f"- candidate02_kernel_TU_binding: `{binding['kernel_translation_units_byte_identical']}`",f"- no-timing correctness preflight: `{physical_correctness['pass']}`",f"- candidate02 KQ-output/scalar correctness preflight: `{c2_correctness['pass']}`",f"- companion binding preflight: `{binding_report['pass']}`",f"- 3-cell timing band [0.90,1.10]: `{binding_sanity['pass']}`","- 72-cell timed sweep started: `false`","","## Binding ratios"]
    pre_report.extend(f"- {name}: ratio `{row['ratio']:.9g}`; pass `{row['within_sanity_band']}`" for name,row in binding_sanity["cells"].items())
    pre_report.extend(["","## SHA-256"])
    pre_report.extend(f"- `{path}`: `{record['sha256']}` ({record['size_bytes']} bytes)" for path,record in sorted(pre_hashes.items()))
    report_path=PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md"
    report_path.write_text("\n".join(pre_report)+"\n",encoding="utf-8",newline="\n")
    report_sha=sha256_file(report_path)
    sidecar=PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md.sha256"
    sidecar.write_text(report_sha+"\n",encoding="ascii",newline="\n")
    pre_paths.extend([report_path,sidecar])
    pre_manifest={"schema":"omega-v2-1c-preflight-artifact-hashes-v1","implementation_commit":provenance["implementation_commit"],"candidate02_exe_sha256":c2_exe_hash,"report_self_sha256":report_sha,"artifacts":{str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in pre_paths if path.is_file()}}
    write_json(PREFLIGHT_RESULTS/"artifact_hashes.json",pre_manifest)
    verified=all(Path(path).is_file() and sha256_file(Path(path))==record["sha256"] for path,record in pre_manifest["artifacts"].items()) and sha256_file(report_path)==sidecar.read_text(encoding="ascii").strip()
    write_json(PREFLIGHT_RESULTS/"artifact_hashes_verified.json",{"verified":verified,"artifact_count":len(pre_manifest["artifacts"])})
    if not verified: raise RuntimeError("V2_1C_PREFLIGHT_HASH_FAILURE")
    return {"preflight_record":preflight_record,"report_path":report_path,"report_sha256":report_sha,"artifact_manifest_path":PREFLIGHT_RESULTS/"artifact_hashes.json","artifact_manifest_sha256":sha256_file(PREFLIGHT_RESULTS/"artifact_hashes.json"),"artifact_count":len(pre_manifest["artifacts"]),"verified":verified}


def create_binding_preflight()->dict[str,Any]:
    if PREFLIGHT_RESULTS.exists():
        raise FileExistsError(f"V2-1c binding preflight is immutable: {PREFLIGHT_RESULTS}")
    correctness_manifest=verify_artifact_manifest(CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes.json")
    correctness_record=read_json(CORRECTNESS_PREFLIGHT_RESULTS/"correctness_preflight.json")
    correctness_tests=read_json(CORRECTNESS_PREFLIGHT_RESULTS/"test_report.json")
    if correctness_record["preflight_status"]!="PASS" or correctness_tests["fail_count"] or correctness_tests["skip_count"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: no-timing correctness preflight did not pass")
    if source_hashes()!=correctness_record["v2_1c_source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: V2-1c source changed between correctness and binding stages")
    if {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=correctness_record["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_STOP: candidate_02 compute TUs changed between preflight stages")
    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    if attempt_sha!=correctness_record["attempt02_artifact_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 artifact manifest changed between preflight stages")
    c2_manifest,c2_summary,c2_build,c2_exe_sha=validate_candidate02()
    if c2_exe_sha!=FROZEN_CANDIDATE02_EXE_SHA256 or c2_exe_sha!=correctness_record["candidate02_exe_sha256"]:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 executable changed between preflight stages")
    payload,source_info=kq_contract.v2_0_fp32_weight_payload()
    weight_file=PREFLIGHT_ROOT/"v2_0_fp32_source_weights.tmp"
    if not weight_file.is_file() or sha256_file(weight_file)!=hashlib.sha256(payload).hexdigest() or source_info["source_weight_value_sha256"]!=correctness_record["source_weight_manifest"]["weight_state_sha256"]:
        raise RuntimeError("V2_1C_STOP: preflight Q4 input stream is missing/changed")

    # This is the only timed preparation step: exactly the three sealed KQ binding cells.
    binding_path=PREFLIGHT_ROOT/"companion_binding_preflight.json"
    binding_report=execute_equivalence("--binding-preflight",weight_file,KQ2_OUTPUTS,binding_path)
    kq_native=read_json(KQ2_NATIVE)
    binding_timing=compare_binding_timing(binding_report,kq_native) if binding_report.get("cells") else {"authorized_pre_sweep_binding_only":True,"allowed_ratio_band":[SANITY_RATIO_MIN,SANITY_RATIO_MAX],"cells":{},"no_timed_allocations":False,"pass":False,"reason":"no three-cell timing samples were produced"}
    binding_pass=binding_report.get("pass") is True and binding_timing["pass"] is True and binding_report.get("timing_protocol",{}).get("warmups")==10 and binding_report.get("timing_protocol",{}).get("samples")==31
    current_sources=source_hashes()
    if current_sources!=correctness_record["source_sha256"] or {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=correctness_record["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: sources changed during binding preflight")

    preflight_record={
        **correctness_record,
        "phase":"correctness_then_binding_preflights",
        "companion_binding_preflight":binding_report,
        "companion_binding_timing_sanity":binding_timing,
        "companion_binding_preflight_pass":binding_pass,
        "correctness_preflight_manifest_sha256":sha256_file(CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes.json"),
        "binding_preflight_started_after_correctness_pass":True,
        "timed_72_cell_sweep_started":False,
        "preflight_status":"PASS" if binding_pass else "COMPANION_BINDING_INVALID",
    }
    binding_tests=[
        {"name":"test_v2_1c_candidate02_correctness_preflight","status":"PASS","detail":"sealed no-timing correctness report passed"},
        {"name":"test_v2_1c_companion_binding_three_kq_cells_bit_exact","status":"PASS" if binding_report.get("sealed_output_comparisons",{}).get("all_available_sealed_outputs_bit_exact") else "FAIL","detail":binding_report.get("sealed_output_comparisons",{})},
        {"name":"test_v2_1c_companion_binding_warmups_samples_and_no_allocations","status":"PASS" if binding_report.get("timing_protocol",{}).get("warmups")==10 and binding_report.get("timing_protocol",{}).get("samples")==31 and binding_report.get("timing_protocol",{}).get("no_timed_allocations") is True else "FAIL","detail":binding_report.get("timing_protocol",{})},
        {"name":"test_v2_1c_companion_binding_timing_ratio_0p90_1p10","status":"PASS" if binding_timing["pass"] else "FAIL","detail":binding_timing},
        {"name":"test_v2_1c_attempt02_preserved","status":"PASS","detail":{"artifact_manifest_sha256":attempt_sha,"artifact_hashes_verified":True}},
        {"name":"test_v2_1c_candidate02_kernel_frozen_hash","status":"PASS" if c2_exe_sha==FROZEN_CANDIDATE02_EXE_SHA256 and correctness_record["candidate02_kernel_binding"]["kernel_translation_units_byte_identical"] else "FAIL","detail":correctness_record["candidate02_kernel_binding"]},
    ]
    preflight_tests={"schema":"omega-v2-1c-preflight-test-report-v1","test_count":len(binding_tests),"pass_count":sum(row["status"]=="PASS" for row in binding_tests),"fail_count":sum(row["status"]=="FAIL" for row in binding_tests),"skip_count":0,"tests":binding_tests}
    preflight_record["binding_preflight_test_report"]=preflight_tests
    PREFLIGHT_RESULTS.mkdir(parents=True,exist_ok=False)
    write_json(PREFLIGHT_RESULTS/"preflight_record.json",preflight_record)
    write_json(PREFLIGHT_RESULTS/"test_report.json",preflight_tests)
    write_json(PREFLIGHT_RESULTS/"companion_binding_preflight.json",binding_report)
    write_json(PREFLIGHT_RESULTS/"companion_binding_timing_sanity.json",binding_timing)
    write_json(PREFLIGHT_RESULTS/"correctness_preflight_artifact_hashes.json",correctness_manifest)
    write_json(PREFLIGHT_RESULTS/"correctness_preflight_record.json",correctness_record)

    inputs=[*(UNIT_ROOT/name for name in source_files()),*physical_dependency_paths(),*candidate02_kernel_paths(),
        Path(c2_build["executable_absolute_path"]),BENCH_EXE,EQUIVALENCE_EXE,CORRECTNESS_EXE,V2_0_ROOT/"V2_0_RESULT_SEAL.json",
        ATTEMPT02_ROOT/"artifact_hashes.json",ATTEMPT02_ROOT/"benchmark_config.json",ATTEMPT02_ROOT/"hardware_preflight.json",ATTEMPT02_ROOT/"q4_physical_ledger.json",ATTEMPT02_ROOT/"summary_metrics.json",ATTEMPT02_ROOT/"raw_measurements.csv",
        KQ2_ROOT_RESULTS/"artifact_hashes.json",KQ2_NATIVE,KQ2_ROOT_RESULTS/"summary_metrics.json",KQ2_ROOT_RESULTS/"build_manifest.json",
        *(KQ2_OUTPUTS/name for name in ("candidate2_full_m4_k1.bin","candidate2_full_m16_k1.bin","candidate2_full_m8_k4.bin")),
        CORRECTNESS_PREFLIGHT_RESULTS/"artifact_hashes.json",CORRECTNESS_PREFLIGHT_RESULTS/"V2_1C_CORRECTNESS_PREFLIGHT.md",CORRECTNESS_PREFLIGHT_RESULTS/"V2_1C_CORRECTNESS_PREFLIGHT.md.sha256"]
    pre_hashes={str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in inputs if path.is_file()}
    lines=["# V2-1c two-stage preflight seal","",f"- implementation commit: `{correctness_record['source_provenance']['implementation_commit']}`",f"- correctness-only preflight: `{correctness_record['preflight_status']}`",f"- binding preflight: `{preflight_record['preflight_status']}`",f"- frozen candidate_02 exe SHA-256: `{FROZEN_CANDIDATE02_EXE_SHA256}`",f"- source TU identity: `{correctness_record['candidate02_kernel_binding']['kernel_translation_units_byte_identical']}`","- correctness stage: no timer; scalar/oracle, A/B/C equivalence, deterministic repeat.","- binding stage: only three sealed KQ cells; 10 warmups/31 samples; ratio band 0.90–1.10.","- 72-cell sweep started: `false`.","","## Binding ratios"]
    lines.extend(f"- {name}: ratio `{row['ratio']:.9g}`; pass `{row['within_sanity_band']}`" for name,row in binding_timing["cells"].items())
    lines.extend(["","## SHA-256"])
    lines.extend(f"- `{path}`: `{record['sha256']}` ({record['size_bytes']} bytes)" for path,record in sorted(pre_hashes.items()))
    report_path=PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md"
    report_path.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    report_sha=sha256_file(report_path)
    sidecar=PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md.sha256"
    sidecar.write_text(report_sha+"\n",encoding="ascii",newline="\n")
    final_paths=[*inputs,PREFLIGHT_RESULTS/"preflight_record.json",PREFLIGHT_RESULTS/"test_report.json",PREFLIGHT_RESULTS/"companion_binding_preflight.json",PREFLIGHT_RESULTS/"companion_binding_timing_sanity.json",PREFLIGHT_RESULTS/"correctness_preflight_artifact_hashes.json",PREFLIGHT_RESULTS/"correctness_preflight_record.json",report_path,sidecar]
    manifest={"schema":"omega-v2-1c-final-preflight-artifact-hashes-v1","implementation_commit":correctness_record["source_provenance"]["implementation_commit"],"report_self_sha256":report_sha,"candidate02_exe_sha256":FROZEN_CANDIDATE02_EXE_SHA256,"artifacts":{str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in final_paths if path.is_file()}}
    manifest_path=PREFLIGHT_RESULTS/"artifact_hashes.json"
    write_json(manifest_path,manifest)
    verified=all(Path(path).is_file() and sha256_file(Path(path))==rec["sha256"] for path,rec in manifest["artifacts"].items()) and sha256_file(report_path)==sidecar.read_text(encoding="ascii").strip()
    write_json(PREFLIGHT_RESULTS/"artifact_hashes_verified.json",{"verified":verified,"artifact_count":len(manifest["artifacts"])})
    if not verified:raise RuntimeError("V2_1C_PREFLIGHT_HASH_FAILURE")
    return {"preflight_status":preflight_record["preflight_status"],"correctness_preflight_status":correctness_record["preflight_status"],"binding_preflight_status":preflight_record["preflight_status"],"preflight_report_abs":str(report_path.resolve()),"preflight_report_sha256":report_sha,"artifact_manifest_abs":str(manifest_path.resolve()),"artifact_manifest_sha256":sha256_file(manifest_path),"artifact_count":len(manifest["artifacts"]),"verified":verified,"record":preflight_record}
def seal_preflight_stage(root:Path,record:dict[str,Any],artifact_paths:list[Path],report_name:str,report_title:str)->dict[str,Any]:
    if root.exists():raise FileExistsError(f"preflight stage is immutable: {root}")
    root.mkdir(parents=True,exist_ok=False)
    record_path=root/"preflight_record.json";write_json(record_path,record)
    test_path=root/"test_report.json";write_json(test_path,record["test_report"])
    paths=[*artifact_paths,record_path,test_path]
    unique={str(path.resolve()):path for path in paths if path.is_file()}
    hashes={name:{"sha256":sha256_file(path),"size_bytes":path.stat().st_size,"absolute_path":name} for name,path in unique.items()}
    lines=[report_title,"",f"- status: `{record['preflight_status']}`",f"- implementation_commit: `{record['source_provenance']['implementation_commit']}`"]
    lines.extend(record.get("report_summary_lines",[]))
    lines.extend(["","## Contractual preflight tests"])
    lines.extend(f"- {row['status']}: `{row['name']}`" for row in record["test_report"]["tests"])
    lines.extend(["","## SHA-256"])
    lines.extend(f"- `{name}`: `{item['sha256']}` ({item['size_bytes']} bytes)" for name,item in sorted(hashes.items()))
    report_path=root/report_name;report_path.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    report_sha=sha256_file(report_path)
    sidecar=root/(report_name+".sha256");sidecar.write_text(report_sha+"\n",encoding="ascii",newline="\n")
    final_paths=[*unique.values(),report_path,sidecar]
    manifest={"schema":"omega-v2-1c-preflight-stage-artifact-hashes-v1","implementation_commit":record["source_provenance"]["implementation_commit"],"report_self_sha256":report_sha,"artifacts":{str(path.resolve()):{"sha256":sha256_file(path),"size_bytes":path.stat().st_size} for path in final_paths if path.is_file()}}
    manifest_path=root/"artifact_hashes.json";write_json(manifest_path,manifest)
    verified=all(Path(path).is_file() and sha256_file(Path(path))==item["sha256"] for path,item in manifest["artifacts"].items()) and sha256_file(report_path)==sidecar.read_text(encoding="ascii").strip()
    write_json(root/"artifact_hashes_verified.json",{"verified":verified,"artifact_count":len(manifest["artifacts"])})
    if not verified:raise RuntimeError(f"preflight artifact hash verification failed: {root}")
    return {"root":root,"report_path":report_path,"report_sha256":report_sha,"manifest_path":manifest_path,"manifest_sha256":sha256_file(manifest_path),"artifact_count":len(manifest["artifacts"]),"verified":verified,"manifest":manifest}


def run_binding_stage()->dict[str,Any]:
    if BINDING_PREFLIGHT_RESULTS.exists():raise FileExistsError(f"binding preflight is immutable: {BINDING_PREFLIGHT_RESULTS}")
    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    c2_manifest,c2_summary,c2_build,c2_exe_sha=validate_candidate02()
    seal=kq_contract.validate_v2_0_seal()
    provenance=verify_provenance(attempt_sha,sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"))
    vswhere=find_vswhere();cmake,vs_install,cmake_version=find_msvc_cmake(vswhere)
    source_before=source_hashes();physical_before={str(p.resolve()):sha256_file(p) for p in physical_dependency_paths()};kernel_before={str(p.resolve()):sha256_file(p) for p in candidate02_kernel_paths()}
    build_logs=build_native(cmake)
    python_tests=run_python_contract_tests()
    binding=validate_kq_kernel_binding(c2_build)
    companion_hash_before=sha256_file(BENCH_EXE)
    if source_hashes()!=source_before or {str(p.resolve()):sha256_file(p) for p in physical_dependency_paths()}!=physical_before or {str(p.resolve()):sha256_file(p) for p in candidate02_kernel_paths()}!=kernel_before:
        raise RuntimeError("V2_1C_SOURCE_HOLD: sources changed during companion build")
    payload,source_info=kq_contract.v2_0_fp32_weight_payload()
    if source_info["source_weight_value_sha256"]!=read_json(KQ2_NATIVE)["source_weight_sha256"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: Q4 binding weights differ from sealed candidate_02 KQ source")
    PREFLIGHT_ROOT.mkdir(parents=True,exist_ok=True)
    weights_path=PREFLIGHT_ROOT/"v2_0_fp32_source_weights.tmp";weights_path.write_bytes(payload)
    output_path=PREFLIGHT_ROOT/"binding_companion.json"
    binding_report=execute_equivalence("--binding-preflight",weights_path,KQ2_OUTPUTS,output_path)
    kq_native=read_json(KQ2_NATIVE)
    ratios=compare_binding_timing(binding_report,kq_native) if binding_report.get("cells") else {"authorized_pre_sweep_binding_only":True,"allowed_ratio_band":[SANITY_RATIO_MIN,SANITY_RATIO_MAX],"cells":{},"no_timed_allocations":False,"pass":False}
    companion_hash_after=sha256_file(BENCH_EXE)
    binding_pass=binding_report.get("pass") is True and ratios["pass"] is True and companion_hash_before==companion_hash_after
    record={
        "schema":"omega-v2-1c-binding-preflight-v1","phase":"1_BINDING_BEFORE_CORE_SELECTION",
        "source_provenance":provenance,"candidate02_exe_sha256":FROZEN_CANDIDATE02_EXE_SHA256,
        "candidate02_artifact_manifest_sha256":sha256_file(KQ2_ROOT_RESULTS/"artifact_hashes.json"),
        "attempt02_artifact_manifest_sha256":attempt_sha,"attempt02_config":attempt_config,
        "candidate02_build_manifest":c2_build,"candidate02_kernel_binding":binding,
        "candidate02_kernel_tu_sha256":kernel_before,"physical_dependency_sha256":physical_before,"v2_1c_source_sha256":source_before,
        "companion_executable_abs":str(BENCH_EXE.resolve()),"companion_exe_sha256_before_binding":companion_hash_before,"companion_exe_sha256_frozen":companion_hash_after,
        "companion_binding_preflight":binding_report,"companion_binding_timing_sanity":ratios,
        "attempt02_protocol":{"d":attempt_config["d"],"m":attempt_config["m"],"K":attempt_config["K"],"variants":attempt_config["variants"],"blocks":attempt_config["measurement_blocks"],"samples_per_block":attempt_config["samples_per_block"],"warmups":attempt_config["warmups_per_cell_variant"],"schedule_seed":attempt_config["schedule_seed"],"cpu_set_ids":[row["windows_cpu_set_id"] for row in attempt_config["selected_workers"]],"v_i":[row["h0_v_i"] for row in attempt_config["selected_workers"]]},
        "compiler":c2_build["compiler"],"visual_studio_installation":vs_install,"cmake_path":str(cmake),"cmake_version":cmake_version,"build_logs":build_logs,"python_contract_tests":python_tests,
        "preflight_status":"PASS" if binding_pass else "COMPANION_BINDING_INVALID","core_selection_started":False,"correctness_with_new_selection_started":False,"timed_72_cell_sweep_started":False,
    }
    tests=[
        {"name":"test_v2_1c_binding_same_companion_exe_as_sweep","status":"PASS" if companion_hash_before==companion_hash_after and companion_hash_after==binding["companion_executable_sha256"] else "FAIL","detail":{"companion_exe_sha256":companion_hash_after,"frozen_after_binding":True}},
        {"name":"test_v2_1c_candidate02_compute_identity_before_binding","status":"PASS" if binding["kernel_translation_units_byte_identical"] and binding["compute_defines_match"] and binding["compute_options_match"] and binding["compute_link_options_match"] else "FAIL","detail":binding},
        {"name":"test_v2_1c_binding_three_sealed_cells_bit_exact","status":"PASS" if binding_report.get("pass") else "FAIL","detail":binding_report.get("sealed_output_comparisons",{})},
        {"name":"test_v2_1c_binding_timing_0p90_1p10","status":"PASS" if ratios["pass"] else "FAIL","detail":ratios},
        {"name":"test_v2_1c_binding_10_warmups_31_samples_no_allocations","status":"PASS" if binding_report.get("timing_protocol",{}).get("warmups")==10 and binding_report.get("timing_protocol",{}).get("samples")==31 and binding_report.get("timing_protocol",{}).get("no_timed_allocations") is True else "FAIL","detail":binding_report.get("timing_protocol",{})},
    ]
    record["test_report"]={"test_count":len(tests),"pass_count":sum(row["status"]=="PASS" for row in tests),"fail_count":sum(row["status"]=="FAIL" for row in tests),"skip_count":0,"tests":tests}
    artifacts=[*(UNIT_ROOT/name for name in source_files()),*physical_dependency_paths(),*candidate02_kernel_paths(),
        Path(c2_build["executable_absolute_path"]),V2_0_ROOT/"V2_0_RESULT_SEAL.json",ATTEMPT02_ROOT/"artifact_hashes.json",ATTEMPT02_ROOT/"benchmark_config.json",ATTEMPT02_ROOT/"hardware_preflight.json",ATTEMPT02_ROOT/"q4_physical_ledger.json",ATTEMPT02_ROOT/"raw_measurements.csv",
        KQ2_ROOT_RESULTS/"artifact_hashes.json",KQ2_NATIVE,KQ2_ROOT_RESULTS/"build_manifest.json",KQ2_ROOT_RESULTS/"summary_metrics.json",
        *(KQ2_OUTPUTS/name for name in ("candidate2_full_m4_k1.bin","candidate2_full_m16_k1.bin","candidate2_full_m8_k4.bin")),BENCH_EXE,EQUIVALENCE_EXE,CORRECTNESS_EXE,
        output_path]
    report_lines=["# V2-1c COMPANION_BINDING_PREFLIGHT","",f"- status: `{record['preflight_status']}`",f"- same companion exe later used for core selection/correctness/sweep: `{companion_hash_after}`",f"- frozen candidate_02 exe: `{FROZEN_CANDIDATE02_EXE_SHA256}`",f"- candidate_02 source/toolset/flags/defines identity matches: `{binding['kernel_translation_units_byte_identical'] and binding['compute_defines_match'] and binding['compute_options_match'] and binding['compute_link_options_match']}`","- core-selection/H0 preflight has not started; 72-cell sweep has not started.","","## Timing ratios (companion/KQ; required 0.90–1.10)"]
    report_lines.extend(f"- {name}: `{row['ratio']:.9g}` ({row['within_sanity_band']})" for name,row in ratios["cells"].items())
    stage=seal_preflight_stage(BINDING_PREFLIGHT_RESULTS,record,artifacts,"V2_1C_BINDING_PREFLIGHT.md",report_lines)
    return {"stage":stage,"record":record}
    manifest_path=PREFLIGHT_RESULTS/"artifact_hashes.json"
    manifest=read_json(manifest_path)
    mismatches=[]
    for absolute,record in manifest["artifacts"].items():
        path=Path(absolute)
        if not path.is_file() or sha256_file(path)!=record["sha256"]:mismatches.append(absolute)
    if mismatches:raise RuntimeError(f"V2_1C_PREFLIGHT_SOURCE_HOLD: preflight seal mismatch: {mismatches}")
    record=read_json(PREFLIGHT_RESULTS/"preflight_record.json")
    tests=read_json(PREFLIGHT_RESULTS/"preflight_tests.json")
    if record["timed_72_cell_sweep_started"] is not False or record["preflight_status"]!="PASS" or tests["fail_count"] or tests["skip_count"] or not record["companion_binding_preflight"]["pass"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: preflight manifest does not authorize sweep readiness")
    if source_hashes()!=record["v2_1c_source_sha256"]:raise RuntimeError("V2_1C_SOURCE_HOLD: V2-1c implementation changed after preflight seal")
    if sha256_file(Path(record["candidate02_build_manifest"]["executable_absolute_path"]))!=FROZEN_CANDIDATE02_EXE_SHA256:raise RuntimeError("V2_1C_STOP: frozen candidate_02 executable changed")
    for absolute,expected in record["candidate02_kernel_tu_sha256"].items():
        if sha256_file(Path(absolute))!=expected:raise RuntimeError("V2_1C_STOP: frozen candidate_02 kernel TU changed")
    if sha256_file(BENCH_EXE)!=record["companion_bench_exe_sha256"] or sha256_file(EQUIVALENCE_EXE)!=record["companion_equivalence_exe_sha256"] or sha256_file(CORRECTNESS_EXE)!=record["companion_correctness_exe_sha256"]:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: companion executable changed after binding preflight")
    if record["companion_binding_preflight"].get("pass") is not True or record["companion_binding_timing_sanity"].get("pass") is not True:
        raise RuntimeError("V2_1C_PREFLIGHT_HOLD: companion binding preflight did not fully pass")
    return record


def attempt02_preservation_after(expected_sha: str) -> dict[str, Any]:
    manifest,config,observed_sha=validate_attempt02()
    if observed_sha!=expected_sha:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest hash changed during V2-1c")
    return {"artifact_manifest_sha256": observed_sha,"artifact_count":len(manifest["artifacts"]),"artifact_hashes_verified_after_v2_1c":True,"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in config["selected_workers"]],"frozen_v_i":[row["h0_v_i"] for row in config["selected_workers"]]}


def make_v2_1c_tests(preflight: dict[str,Any], native_status: dict[str,Any], hardware: dict[str,Any], config: dict[str,Any],
                     raw_rows: list[dict[str,Any]], complete: bool, gates: dict[str,Any] | None,
                     attempt_after: dict[str,Any], comparison: dict[str,Any], conformance: dict[str,Any]) -> list[dict[str,Any]]:
    binding=preflight["candidate02_kernel_binding"]
    eq=preflight["candidate02_equivalence"]
    selected=hardware.get("selected_workers",[])
    selection=preflight["core_selection_preflight"]
    frozen_ids=[int(value) for value in selection["selected_cpu_set_ids"]]
    frozen_vi=[float(value) for value in selection["selected_v_i"]]
    config_ids=[int(row["windows_cpu_set_id"]) for row in config.get("selected_workers",[])]
    config_vi=[float(row["h0_v_i"]) for row in config.get("selected_workers",[])]
    observed_qpc=hardware.get("hardware",{}).get("qpc",{}).get("frequency")
    attempt02_qpc=selection.get("qpc_frequency")
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
            "candidate_02_identity":"SOURCE_AND_BEHAVIOR_BOUND_COMPANION",
            "original_KQ_executable":{"sha256":FROZEN_CANDIDATE02_EXE_SHA256,"immutable":True,"not_used_for_V2_1c":True},
            "scientific_kernel":"no_compute_source_changes",
            "kernel_binding":{"candidate_id":"KQ2_ROW_TILE4_SLOT2_FUSED","frozen_kq_executable_sha256":FROZEN_CANDIDATE02_EXE_SHA256,"companion_physical_harness_executable_sha256":preflight["companion_bench_exe_sha256"],"candidate02_kernel_tus_byte_identical":preflight["candidate02_kernel_binding"]["kernel_translation_units_byte_identical"],"candidate02_compile_flags_and_toolset_match":True,"no_kernel_source_changes":True},
            "protocol":{"cells":72,"d":[512,640],"m":[1,4,8,16],"m1_role":"CONTROL_ONLY","K":[1,4,8],"variants":["A","B","C"],"blocks":5,"samples_per_block":21,"warmups":10,"schedule_seed":20260929,"workers_cpu_set_ids":preflight["core_selection_preflight"]["selected_cpu_set_ids"],"v_i":preflight["core_selection_preflight"]["selected_v_i"],"v_i_source":"fresh candidate_02 H0 core-selection preflight; sealed, not reselected during sweep","attempt02_performance_comparison_status":preflight["core_selection_preflight"]["attempt02_pairing_status"]},
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
                              current_gates: dict[str,Any],current_cells: dict[tuple[int,int,int,str],dict[str,Any]],
                              pairing_status: str)->dict[str,Any]:
    old=attempt02_summary.get("gate_metrics",{})
    def time(cells:dict[tuple[int,int,int,str],dict[str,Any]],m:int,k:int,v:str)->float:
        value=cells[(512,m,k,v)]["median_seconds"]
        if value is None: raise ValueError(f"missing comparison cell d512/m{m}/K{k}/{v}")
        return float(value)
    metrics={
        "attempt02":{key:old.get(key) for key in ("rho_resident","delta_CB","G_matrix","c_A_K8","c_B_K8","c_C_K8")},
        "v2_1c":{key:current_gates.get(key) for key in ("rho_resident","delta_CB","G_matrix","c_A_K8","c_B_K8","c_C_K8")},
        "non_gate_comparisons":{
            "performance_comparison_status":pairing_status,
            "performance_comparison_interpretation":"paired kernel comparison" if pairing_status=="PAIRED_KERNEL_COMPARISON" else "DIAGNOSTIC_ONLY / NOT_PAIRED_KERNEL_COMPARISON",
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
        ATTEMPT02_ROOT/"artifact_hashes.json",ATTEMPT02_ROOT/"benchmark_config.json",ATTEMPT02_ROOT/"hardware_preflight.json",ATTEMPT02_ROOT/"q4_physical_ledger.json",ATTEMPT02_ROOT/"worker_shard_manifest.json",ATTEMPT02_ROOT/"summary_metrics.json",ATTEMPT02_ROOT/"raw_measurements.csv",ATTEMPT02_ROOT/"OMEGA_V2_1_REPORT.md",KQ2_ROOT_RESULTS/"artifact_hashes.json",KQ2_NATIVE,
        KQ2_ROOT_RESULTS/"summary_metrics.json",KQ2_ROOT_RESULTS/"build_manifest.json",KQ2_ROOT_RESULTS/"source_weight_manifest.json",KQ2_ROOT_RESULTS/"test_report.json",
        *(KQ2_OUTPUTS/name for name in ("candidate2_full_m4_k1.bin","candidate2_full_m16_k1.bin","candidate2_full_m8_k4.bin")),
        BENCH_EXE,EQUIVALENCE_EXE,CORRECTNESS_EXE,Path(preflight["candidate02_kq_correctness_report_path_abs"]),Path(preflight["physical_correctness_report_path_abs"]),Path(preflight["companion_binding_report_path_abs"]),Path(preflight["companion_binding_timing_report_path_abs"]),
        PREFLIGHT_RESULTS/"artifact_hashes.json",PREFLIGHT_RESULTS/"artifact_hashes_verified.json",PREFLIGHT_RESULTS/"preflight_record.json",PREFLIGHT_RESULTS/"preflight_tests.json",PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md",PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md.sha256",
        RESULTS_ROOT/"build_manifest.json",RESULTS_ROOT/"hardware_preflight.json",RESULTS_ROOT/"q4_physical_ledger.json",
        RESULTS_ROOT/"worker_shard_manifest.json",RESULTS_ROOT/"benchmark_config.json",RESULTS_ROOT/"native_test_report.json",
        RESULTS_ROOT/"native_run_status.json",RESULTS_ROOT/"raw_measurements.csv",RESULTS_ROOT/"summary_metrics.json",
        RESULTS_ROOT/"test_report.json",RESULTS_ROOT/"OMEGA_CONFORMANCE_BLOCK.yaml",RESULTS_ROOT/"attempt02_preservation.json",
        RESULTS_ROOT/"candidate02_kernel_binding.json",RESULTS_ROOT/"candidate02_equivalence.json",RESULTS_ROOT/"physical_correctness_preflight.json",RESULTS_ROOT/"companion_binding_preflight.json",RESULTS_ROOT/"companion_binding_timing_sanity.json",RESULTS_ROOT/"attempt02_vs_v2_1c_comparison.json",
        RESULTS_ROOT/"preflight_record.json",
        PREFLIGHT_RESULTS/"artifact_hashes.json",PREFLIGHT_RESULTS/"artifact_hashes_verified.json",PREFLIGHT_RESULTS/"preflight_record.json",PREFLIGHT_RESULTS/"test_report.json",PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md",PREFLIGHT_RESULTS/"V2_1C_PREFLIGHT_REPORT.md.sha256",PREFLIGHT_RESULTS/"candidate02_correctness_preflight.json",PREFLIGHT_RESULTS/"physical_correctness_preflight.json",PREFLIGHT_RESULTS/"companion_binding_preflight.json",PREFLIGHT_RESULTS/"companion_binding_timing_sanity.json",PREFLIGHT_RESULTS/"attempt02_preservation.json",
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
        "## Correctness and companion binding preflights",
        f"- candidate_02 KQ-source correctness/oracle and all available sealed output checks: `{summary['preflight']['candidate02_kq_correctness_preflight']['pass']}`.",
        f"- d512/d640 full-size K1 scalar oracle, toy recurrence K={1,4,8}, and all 72 A/B/C cell equality/determinism checks: `{summary['preflight']['physical_abc_correctness_preflight']['pass']}`.",
        f"- binding-preflight (only d512,m4,K1; d512,m16,K1; d512,m8,K4; 10 warmups/31 samples; allowed companion/KQ ratio 0.90–1.10): `{summary['preflight']['companion_binding_timing_sanity']}`.",
        "- Limitation: candidate_02 KQ sealed files contain exact outputs only for d512,m4,K1; d512,m16,K1; and d512,m8,K4. No sealed output exists for m1,K1 or K4 at m={1,4,16}; those cells were checked against the scalar oracle and repeated-run determinism. The original KQ executable was not rerun or modified.", "",
        "## Protocol",
        "- Cells: d={512,640}, m={1,4,8,16}, K={1,4,8}, A/B/C = 72; 10 warmups, 5 blocks × 21 samples; seed 20260929.",
        f"- v_i and fixed CPU-set workers from the fresh candidate_02 core-selection preflight: `{summary['preflight']['core_selection_preflight']['selected_cpu_set_ids']}` with v_i `{summary['preflight']['core_selection_preflight']['selected_v_i']}`.",
        f"- attempt_02 performance comparison status: `{summary['preflight']['core_selection_preflight']['attempt02_pairing_status']}`.",
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
        "preflight":preflight["preflight_record"],"equivalence":preflight["preflight_record"]["candidate02_kq_correctness_preflight"],
        "attempt02_preservation":{**attempt_before,"artifact_hashes_verified_before_v2_1c":True},
        "attempt02_vs_v2_1c_comparison":None,
        "global_status":"CONFORMANCE_HOLD","gpu":"HOLD","T3":"HOLD",
    }
    attempt02_summary,attempt02_cells=load_attempt02_cells()
    pairing_status=preflight["preflight_record"]["core_selection_preflight"]["attempt02_pairing_status"]
    comparison=make_attempt02_comparison(attempt02_summary,attempt02_cells,gates or {},cells,pairing_status) if gates else {"attempt02_gate_metrics":attempt02_summary.get("gate_metrics",{}),"v2_1c_gate_metrics":None,"attempt02_vs_v2_1c_non_gate":{"performance_comparison_status":pairing_status,"performance_comparison_interpretation":"paired kernel comparison" if pairing_status=="PAIRED_KERNEL_COMPARISON" else "DIAGNOSTIC_ONLY / NOT_PAIRED_KERNEL_COMPARISON"}}
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
        "candidate02_frozen_binding":preflight["preflight_record"]["candidate02_kernel_binding"],
        "candidate02_correctness_preflight":preflight["preflight_record"]["candidate02_kq_correctness_preflight"],
        "physical_correctness_preflight":preflight["preflight_record"]["physical_abc_correctness_preflight"],
        "companion_binding_preflight":preflight["preflight_record"]["companion_binding_preflight"],
        "candidate02_binding_timing":preflight["preflight_record"]["companion_binding_timing_sanity"],
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
        "candidate02_kq_correctness_report":preflight["preflight_record"]["candidate02_kq_correctness_preflight"],
    }
    write_json(RESULTS_ROOT/"build_manifest.json",build_manifest)
    write_json(RESULTS_ROOT/"preflight_record.json",preflight["preflight_record"])
    equivalence_path=Path(preflight["preflight_record"]["candidate02_kq_correctness_report_path_abs"])
    (RESULTS_ROOT/"candidate02_equivalence.json").write_bytes(equivalence_path.read_bytes())
    physical_path=Path(preflight["preflight_record"]["physical_correctness_report_path_abs"])
    (RESULTS_ROOT/"physical_correctness_preflight.json").write_bytes(physical_path.read_bytes())
    (RESULTS_ROOT/"companion_binding_preflight.json").write_bytes(Path(preflight["preflight_record"]["companion_binding_report_path_abs"]).read_bytes())
    (RESULTS_ROOT/"companion_binding_timing_sanity.json").write_bytes(Path(preflight["preflight_record"]["companion_binding_timing_report_path_abs"]).read_bytes())
    write_json(RESULTS_ROOT/"candidate02_kernel_binding.json",preflight["preflight_record"]["candidate02_kernel_binding"])
    attempt02_after=attempt02_preservation_after(attempt_before)
    write_json(RESULTS_ROOT/"attempt02_preservation.json",attempt02_after)
    native_env=os.environ.copy()
    native_env["OMEGA_V2_1_RESULTS_ROOT"]=str(RESULTS_ROOT.resolve())
    selected=preflight["preflight_record"]["core_selection_preflight"]["selected_workers"]
    native_env["OMEGA_V2_1C_SELECTED_CPU_SET_IDS"]=",".join(str(row["cpu_set_id"]) for row in selected)
    native_env["OMEGA_V2_1C_SELECTED_V_I"]=",".join(str(row["v_i"]) for row in selected)
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


def main_legacy_do_not_call()->int:
    raise RuntimeError("Disabled legacy combined-preflight/sweep entrypoint; use the staged V2-1c CLI below.")
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only",action="store_true",help="run no-timing correctness checks followed by the authorized 3-cell binding preflight; do not run 72 cells")
    mode.add_argument("--run-sweep",action="store_true",help="run the 72-cell sweep only after a verified sealed preflight and explicit GO medicion")
    parser.add_argument("--go-medicion",action="store_true",help="required acknowledgement of the judge's explicit post-preflight GO medicion")
    args=parser.parse_args()
    if os.name!="nt" or sys.platform!="win32":
        raise RuntimeError("V2_1C_MEASUREMENT_INVALID: unit requires native Windows")
    if args.preflight_only:
        cmake,vs_install,cmake_version=find_msvc_cmake(find_vswhere())
        result=create_preflight_seal(cmake,vs_install,cmake_version)
        print(json.dumps({"preflight_only":True,"preflight_status":result["preflight_record"]["preflight_status"],"measurement_started":False,"v2_1c_implementation_commit":result["preflight_record"]["source_provenance"]["implementation_commit"],"frozen_candidate02_exe_sha256":result["preflight_record"]["candidate02_exe_sha256"],"candidate02_kernel_tus_byte_identical":result["preflight_record"]["candidate02_kernel_binding"]["kernel_translation_units_byte_identical"],"candidate02_correctness_pass":result["preflight_record"]["candidate02_kq_correctness_preflight"].get("pass"),"physical_abc_correctness_pass":result["preflight_record"]["physical_abc_correctness_preflight"].get("pass"),"binding_preflight_pass":result["preflight_record"]["companion_binding_preflight"].get("pass"),"binding_timing_ratios":result["preflight_record"]["companion_binding_timing_sanity"],"preflight_report_abs":str(result["report_path"].resolve()),"preflight_report_sha256":result["report_sha256"],"artifact_manifest_abs":str(result["artifact_manifest_path"].resolve()),"artifact_manifest_sha256":result["artifact_manifest_sha256"],"preflight_hashes_verified":result["verified"],"timed_72_cell_sweep_started":False,"attempt02_hash_verified":True},indent=2,sort_keys=True))
        return 0 if result["preflight_record"]["preflight_status"]=="PASS" and result["verified"] else 1

    if not args.go_medicion:
        raise RuntimeError("V2_1C_STOP: --run-sweep requires the judge's explicit GO medicion")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-1c attempt_01 result slot is immutable and already exists: {RESULTS_ROOT}")
    preflight_record=load_sealed_preflight()
    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    if attempt_sha!=preflight_record["attempt02_artifact_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest differs from sealed preflight")
    c2_manifest,c2_summary,c2_build,c2_exe_sha=validate_candidate02()
    if c2_exe_sha!=preflight_record["candidate02_exe_sha256"] or c2_exe_sha!=FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 KQ executable does not match preflight")
    preflight={
        "preflight_record":preflight_record,
        "source_provenance":preflight_record["source_provenance"],
        "source_sha256":preflight_record["v2_1c_source_sha256"],
        "physical_dependency_sha256":preflight_record["physical_dependency_sha256"],
        "candidate02_kernel_tu_sha256":preflight_record["candidate02_kernel_tu_sha256"],
        "candidate02_manifest_sha256":preflight_record["candidate02_artifact_manifest_sha256"],
        "candidate02_build_manifest":preflight_record["candidate02_build_manifest"],
        "bench_executable_sha256":preflight_record["companion_bench_exe_sha256"],
        "equivalence_executable_sha256":preflight_record["companion_equivalence_exe_sha256"],
        "correctness_executable_sha256":preflight_record["companion_correctness_exe_sha256"],
        "build_logs":preflight_record["build_logs"],
    }
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


def _legacy_main_do_not_call()->int:
    raise RuntimeError("Disabled legacy combined-preflight entrypoint; use the staged V2-1c CLI below.")
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--correctness-preflight-only",action="store_true",help="no-timing candidate02 scalar/oracle and all-cell A/B/C correctness preflight")
    mode.add_argument("--binding-preflight-only",action="store_true",help="authorized three-cell KQ binding timing preflight after correctness passes")
    mode.add_argument("--run-sweep",action="store_true",help="72-cell V2-1c sweep only after sealed preflights and explicit judge GO medicion")
    parser.add_argument("--go-medicion",action="store_true",help="acknowledge the judge's explicit GO medicion; required with --run-sweep")
    args=parser.parse_args()
    if os.name!="nt" or sys.platform!="win32":
        raise RuntimeError("V2_1C_MEASUREMENT_INVALID: unit requires native Windows")

    if args.correctness_preflight_only:
        cmake,vs_install,cmake_version=find_msvc_cmake(find_vswhere())
        result=create_preflight(cmake,vs_install,cmake_version)
        print(json.dumps({"phase":"correctness-only","preflight_status":result["preflight_status"],"timing_started":False,"sweep_started":False,"report_abs":result["correctness_report_abs"],"report_sha256":result["correctness_report_sha256"],"artifact_manifest_abs":result["artifact_manifest_abs"],"artifact_manifest_sha256":result["artifact_manifest_sha256"],"artifact_count":result["artifact_count"],"hashes_verified":result["hashes_verified"],"candidate02_kq_scalar_pass":result["correctness_record"]["candidate02_kq_correctness_preflight"].get("pass"),"physical_correctness_pass":result["correctness_record"]["physical_abc_correctness_preflight"].get("pass")},indent=2,sort_keys=True))
        return 0 if result["preflight_status"]=="PASS" and result["hashes_verified"] else 1

    if args.binding_preflight_only:
        result=create_binding_preflight()
        print(json.dumps({"phase":"binding-only","preflight_status":result["preflight_status"],"timed_72_cell_sweep_started":False,"report_abs":str(result["report_path"].resolve()),"report_sha256":result["report_sha256"],"artifact_manifest_abs":str(result["artifact_manifest_path"].resolve()),"artifact_manifest_sha256":result["artifact_manifest_sha256"],"artifact_count":result["artifact_count"],"artifact_hashes_verified":result["verified"],"binding_timing_ratios":result["record"]["companion_binding_timing_sanity"]},indent=2,sort_keys=True))
        return 0 if result["preflight_status"]=="PASS" and result["verified"] else 1

    if not args.go_medicion:
        raise RuntimeError("V2_1C_STOP: --run-sweep requires explicit judge GO medicion")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-1c attempt_01 is immutable and already exists: {RESULTS_ROOT}")
    preflight_record=load_sealed_preflight()
    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    if attempt_sha!=preflight_record["attempt02_artifact_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 manifest differs from sealed preflight")
    c2_manifest,c2_summary,c2_build,c2_exe_sha=validate_candidate02()
    if c2_exe_sha!=FROZEN_CANDIDATE02_EXE_SHA256 or c2_exe_sha!=preflight_record["candidate02_exe_sha256"]:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 executable differs from preflight")
    preflight={
        "preflight_record":preflight_record,
        "source_provenance":preflight_record["source_provenance"],
        "source_sha256":preflight_record["source_sha256"],
        "physical_dependency_sha256":preflight_record["physical_dependency_sha256"],
        "candidate02_kernel_tu_sha256":preflight_record["candidate02_kernel_tu_sha256"],
        "candidate02_manifest_sha256":preflight_record["candidate02_artifact_manifest_sha256"],
        "candidate02_build_manifest":preflight_record["candidate02_build_manifest"],
        "build_logs":preflight_record["build_logs"],
    }
    attempt_before={"artifact_manifest_path_abs":str((ATTEMPT02_ROOT/"artifact_hashes.json").resolve()),"artifact_manifest_sha256":attempt_sha,"artifact_count":len(attempt_manifest["artifacts"]),"artifact_hashes_verified_before_v2_1c":True,"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in attempt_config["selected_workers"]],"frozen_h0_shard_weights":[row["h0_v_i"] for row in attempt_config["selected_workers"]],"frozen_worker_logical_ids":[row["logical_processor_id"] for row in attempt_config["selected_workers"]]}
    run_native_sweep(preflight,attempt_before)
    attempt02_preservation_after(attempt_before)
    if source_hashes()!=preflight["source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: source changed during timed physical sweep")
    if sha256_file(Path(c2_build["executable_absolute_path"]))!=FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 KQ executable hash changed during sweep")
    if {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=preflight["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_STOP: candidate_02 kernel TUs changed during sweep")
    summary=finalize_results(preflight,attempt_before)
    summary["sweep_go_acknowledged"]=True
    write_json(RESULTS_ROOT/"summary_metrics.json",summary)
    seal=seal_results(summary)
    print(json.dumps({"unit":"OMEGA-V2-1c-RESIDENCY-ONLY","terminal_status":summary["terminal_status"],"gate_metrics":summary.get("gate_metrics"),"test_count":summary["test_report"]["test_count"],"test_pass_count":summary["test_report"]["pass_count"],"test_fail_count":summary["test_report"]["fail_count"],"test_skip_count":summary["test_report"]["skip_count"],"results_root_abs":str(RESULTS_ROOT.resolve()),"report_sha256":seal["report_sha256"],"artifact_manifest_sha256":seal["artifact_manifest_sha256"],"artifact_count":seal["artifact_count"],"artifact_hashes_verified":seal["verified"],"attempt03_run":False,"pytorch_s_native_gate":"EXCLUDED"},indent=2,sort_keys=True))
    return 0 if summary["terminal_status"]=="OMEGA_V2_1C_RESIDENCY_ONLY_PASS" and seal["verified"] else 1


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--correctness-preflight-only",action="store_true",help="run no-timing correctness checks only")
    mode.add_argument("--binding-preflight-only",action="store_true",help="run the authorized three-cell binding timing preflight after correctness passes")
    mode.add_argument("--run-sweep",action="store_true",help="run the 72-cell sweep only after a sealed preflight and explicit judge GO medicion")
    parser.add_argument("--go-medicion",action="store_true",help="acknowledge explicit judge GO medicion; required for --run-sweep")
    args=parser.parse_args()
    if os.name!="nt" or sys.platform!="win32":
        raise RuntimeError("V2_1C_MEASUREMENT_INVALID: native Windows is required")
    if args.correctness_preflight_only:
        cmake,vs_install,cmake_version=find_msvc_cmake(find_vswhere())
        result=create_preflight(cmake,vs_install,cmake_version)
        print(json.dumps({"phase":"correctness-only","preflight_status":result["preflight_status"],"timing_started":False,"sweep_started":False,"correctness_report_abs":result["correctness_report_abs"],"correctness_report_sha256":result["correctness_report_sha256"],"artifact_manifest_abs":result["artifact_manifest_abs"],"artifact_manifest_sha256":result["artifact_manifest_sha256"],"artifact_count":result["artifact_count"],"hashes_verified":result["hashes_verified"],"candidate02_kq_scalar_pass":result["correctness_record"]["candidate02_kq_correctness_preflight"].get("pass"),"physical_abc_correctness_pass":result["correctness_record"]["physical_abc_correctness_preflight"].get("pass")},indent=2,sort_keys=True))
        return 0 if result["preflight_status"]=="PASS" and result["hashes_verified"] else 1
    if args.binding_preflight_only:
        result=create_binding_preflight()
        print(json.dumps({"phase":"binding-only","preflight_status":result["preflight_status"],"timed_72_cell_sweep_started":False,"report_abs":str(result["report_path"].resolve()),"report_sha256":result["report_sha256"],"artifact_manifest_abs":str(result["artifact_manifest_path"].resolve()),"artifact_manifest_sha256":result["artifact_manifest_sha256"],"artifact_count":result["artifact_count"],"artifact_hashes_verified":result["verified"],"binding_timing_ratios":result["record"]["companion_binding_timing_sanity"]},indent=2,sort_keys=True))
        return 0 if result["preflight_status"]=="PASS" and result["verified"] else 1
    if not args.go_medicion:
        raise RuntimeError("V2_1C_STOP: --run-sweep requires explicit GO medicion")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-1c attempt_01 result slot already exists: {RESULTS_ROOT}")
    preflight_record=load_sealed_preflight()
    attempt_manifest,attempt_config,attempt_sha=validate_attempt02()
    if attempt_sha!=preflight_record["attempt02_artifact_manifest_sha256"]:
        raise RuntimeError("V2_1C_STOP: attempt_02 hash differs from sealed preflight")
    c2_manifest,c2_summary,c2_build,c2_exe_sha=validate_candidate02()
    if c2_exe_sha!=FROZEN_CANDIDATE02_EXE_SHA256 or c2_exe_sha!=preflight_record["candidate02_exe_sha256"]:
        raise RuntimeError("V2_1C_STOP: frozen candidate_02 executable changed")
    preflight={
        "preflight_record":preflight_record,
        "source_provenance":preflight_record["source_provenance"],
        "source_sha256":preflight_record["source_sha256"],
        "physical_dependency_sha256":preflight_record["physical_dependency_sha256"],
        "candidate02_kernel_tu_sha256":preflight_record["candidate02_kernel_tu_sha256"],
        "candidate02_manifest_sha256":preflight_record["candidate02_artifact_manifest_sha256"],
        "candidate02_build_manifest":preflight_record["candidate02_build_manifest"],
        "bench_executable_sha256":preflight_record["companion_bench_exe_sha256"],
        "equivalence_executable_sha256":preflight_record["companion_equivalence_exe_sha256"],
        "correctness_executable_sha256":preflight_record["companion_correctness_exe_sha256"],
        "build_logs":preflight_record["build_logs"],
    }
    attempt_before={"artifact_manifest_path_abs":str((ATTEMPT02_ROOT/"artifact_hashes.json").resolve()),"artifact_manifest_sha256":attempt_sha,"artifact_count":len(attempt_manifest["artifacts"]),"artifact_hashes_verified_before_v2_1c":True,"frozen_worker_cpu_set_ids":[row["windows_cpu_set_id"] for row in attempt_config["selected_workers"]],"frozen_h0_shard_weights":[row["h0_v_i"] for row in attempt_config["selected_workers"]],"frozen_worker_logical_ids":[row["logical_processor_id"] for row in attempt_config["selected_workers"]]}
    run_native_sweep(preflight,attempt_before)
    attempt02_preservation_after(attempt_before)
    if source_hashes()!=preflight["source_sha256"]:
        raise RuntimeError("V2_1C_SOURCE_HOLD: V2-1c source changed during timed sweep")
    if sha256_file(Path(c2_build["executable_absolute_path"]))!=FROZEN_CANDIDATE02_EXE_SHA256:
        raise RuntimeError("V2_1C_STOP: candidate_02 KQ executable hash changed")
    if {str(path.resolve()):sha256_file(path) for path in candidate02_kernel_paths()}!=preflight["candidate02_kernel_tu_sha256"]:
        raise RuntimeError("V2_1C_STOP: candidate_02 compute TUs changed during sweep")
    summary=finalize_results(preflight,attempt_before)
    summary["sweep_go_acknowledged"]=True
    write_json(RESULTS_ROOT/"summary_metrics.json",summary)
    seal=seal_results(summary)
    print(json.dumps({"unit":"OMEGA-V2-1c-RESIDENCY-ONLY","terminal_status":summary["terminal_status"],"gate_metrics":summary.get("gate_metrics"),"test_count":summary["test_report"]["test_count"],"test_pass_count":summary["test_report"]["pass_count"],"test_fail_count":summary["test_report"]["fail_count"],"test_skip_count":summary["test_report"]["skip_count"],"results_root_abs":str(RESULTS_ROOT.resolve()),"report_sha256":seal["report_sha256"],"artifact_manifest_sha256":seal["artifact_manifest_sha256"],"artifact_count":seal["artifact_count"],"artifact_hashes_verified":seal["verified"],"attempt03_run":False,"pytorch_s_native_gate":"EXCLUDED"},indent=2,sort_keys=True))
    return 0 if summary["terminal_status"]=="OMEGA_V2_1C_RESIDENCY_ONLY_PASS" and seal["verified"] else 1


def main()->int:
    from v2_1c_phases import main as phases_main
    return phases_main()


if __name__=="__main__":
    raise SystemExit(main())
