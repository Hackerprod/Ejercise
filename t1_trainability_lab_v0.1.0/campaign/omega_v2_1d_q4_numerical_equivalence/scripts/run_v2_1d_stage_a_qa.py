from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = UNIT_ROOT.parent
KQ_ROOT = CAMPAIGN_ROOT / "omega_v2_1b_candidate_02"
PHYSICAL_ROOT = CAMPAIGN_ROOT / "omega_v2_1_physical"
KQ_RESULTS = CAMPAIGN_ROOT / "omega_v2_1b_kernel_qualification" / "results" / "omega_v2_1b_kernel_qualification" / "candidate_02" / "run_01"
V1C_WEIGHT_STREAM = CAMPAIGN_ROOT / "omega_v2_1c_residency_only" / "build" / "preflight" / "v2_0_fp32_source_weights_correctness.tmp"
KQ_OUTPUTS = KQ_RESULTS / "native_correctness_outputs"
BUILD_ROOT = UNIT_ROOT / "build_qa"
PREFLIGHT_ROOT = BUILD_ROOT / "preflight"
COMPANION_EXE = BUILD_ROOT / "Release" / "omega_v2_1d_stage_a.exe"
TRACKABLE_BUILD_MANIFEST = UNIT_ROOT / "OMEGA_V2_1D_COMPANION_BUILD_MANIFEST.json"
FROZEN_SPEC = UNIT_ROOT / "OMEGA_V2_1D_STAGE_A_SPEC_FROZEN.md"

EXPECTED_KQ_EXE_SHA256 = "be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f"
EXPECTED_FP32_STREAM_SHA256 = "65b9179e13d09513765eba3f56f0368480c03237eb1bbbac48c0f383b56ecbb8"
EXPECTED_CANDIDATE_SOURCE_HASHES = {
    "src/full_block_candidate2.cpp": "68cfdbca0856ecfcc015f76fdc2613a2581636a6e76234c3897425f47772fc0b",
    "src/q4_kernel_candidate2.cpp": "c15e1618a25888a9b77dacee85ecc373215cdc03c764ca9445de2ade5a90594a",
    "src/kq_candidate2.hpp": "77c6e0305a5eb87408b1838392376955fcbd563d6e8390ee1d6006d808f48bfd",
}
EXPECTED_SHARED_HASHES = {
    "q4_layout.cpp": "b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7",
    "v2_1.hpp": "96c77cfaf3e6b8e1a1661401411d8641e0e96a258e637c02615b232723bbb594",
    "worker_pool.cpp": "3549c67d69765beafbaabd821e49b94dbab47681d33485e2f3f10ecf2f2e6270",
}


def powershell_hash(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "''")
    command = f"(Get-FileHash -Algorithm SHA256 -LiteralPath '{escaped}').Hash"
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    )
    digest = process.stdout.strip().upper()
    if len(digest) != 64:
        raise RuntimeError(f"invalid Get-FileHash output for {path}")
    return digest


def run(command: list[str], *, cwd: Path = UNIT_ROOT, timeout: int = 60 * 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def write_logs(prefix: str, process: subprocess.CompletedProcess[str]) -> None:
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT_ROOT / f"{prefix}_stdout.log").write_text(process.stdout, encoding="utf-8", newline="\n")
    (PREFLIGHT_ROOT / f"{prefix}_stderr.log").write_text(process.stderr, encoding="utf-8", newline="\n")


def verify_frozen_inputs() -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_CANDIDATE_SOURCE_HASHES.items():
        path = KQ_ROOT / relative
        observed = powershell_hash(path)
        source_hashes[str(path.resolve())] = observed
        if observed.lower() != expected:
            raise RuntimeError(f"frozen candidate source hash mismatch: {path}")
    for name, expected in EXPECTED_SHARED_HASHES.items():
        path = PHYSICAL_ROOT / "src" / name
        observed = powershell_hash(path)
        source_hashes[str(path.resolve())] = observed
        if observed.lower() != expected:
            raise RuntimeError(f"frozen shared dependency hash mismatch: {path}")
    kq_build = json.loads((KQ_RESULTS / "build_manifest.json").read_text(encoding="utf-8"))
    candidate_exe = Path(kq_build["executable_absolute_path"])
    candidate_exe_hash = powershell_hash(candidate_exe)
    if candidate_exe_hash.lower() != EXPECTED_KQ_EXE_SHA256:
        raise RuntimeError("sealed KQ candidate executable hash mismatch")
    stream_hash = powershell_hash(V1C_WEIGHT_STREAM)
    if stream_hash.lower() != EXPECTED_FP32_STREAM_SHA256:
        raise RuntimeError("sealed V2-0 FP32 weight stream hash mismatch")
    return {
        "candidate_source_sha256": source_hashes,
        "sealed_kq_executable": str(candidate_exe.resolve()),
        "sealed_kq_executable_sha256": candidate_exe_hash,
        "d512_fp32_weight_stream": str(V1C_WEIGHT_STREAM.resolve()),
        "d512_fp32_weight_stream_sha256": stream_hash,
    }


def dry_run() -> None:
    cells = [
        {"d": d, "m": m, "K": k, "state_family": family}
        for d in (512,640)
        for m in (1,4,8,16)
        for k in (1,4)
        for family in ("HISTORICAL_FORMULA","CAL_RANDN")
    ]
    print(json.dumps({"mode":"DRY_RUN_ONLY","scientific_cells_planned":len(cells),"cells":cells,"native_executable_invoked":False,"official_launcher_invoked":False}, indent=2))


def run_unit_tests() -> None:
    process = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    write_logs("unit_tests", process)
    transcript=process.stdout+"\n"+process.stderr
    ran=re.findall(r"^Ran\s+(\d+)\s+tests?\s+in\b",transcript,re.MULTILINE)
    summary_ok=re.search(r"^OK\s*$",transcript,re.MULTILINE) is not None
    report={
        "schema":"omega-v2-1d-unit-static-test-report-v1",
        "status":"PASS" if process.returncode==0 and ran==["12"] and summary_ok else "FAIL",
        "test_count":int(ran[0]) if ran else 0,
        "expected_test_count":12,
        "unittest_ok_summary":summary_ok,
        "return_code":process.returncode,
        "stdout_log":"unit_tests_stdout.log",
        "stderr_log":"unit_tests_stderr.log",
    }
    PREFLIGHT_ROOT.mkdir(parents=True,exist_ok=True)
    (PREFLIGHT_ROOT/"unit_tests_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    if report["status"]!="PASS":
        raise RuntimeError(f"unit/static tests failed: {process.stderr}")
    print(json.dumps(report,indent=2))


def configure_and_build() -> None:
    kq_build = json.loads((KQ_RESULTS / "build_manifest.json").read_text(encoding="utf-8"))
    cmake_candidate = Path(kq_build.get("cmake_path", ""))
    cmake = str(cmake_candidate) if cmake_candidate.is_file() else (shutil.which("cmake") or "cmake")
    configure = run([cmake,"-S",str(UNIT_ROOT),"-B",str(BUILD_ROOT),"-G","Visual Studio 17 2022","-A","x64","-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"])
    write_logs("cmake_configure", configure)
    if configure.returncode:
        raise RuntimeError(f"CMake configure failed: {configure.stderr}")
    build = run([cmake,"--build",str(BUILD_ROOT),"--config","Release","--target","omega_v2_1d_stage_a","--clean-first","--verbose"])
    write_logs("cmake_build", build)
    if build.returncode or not COMPANION_EXE.is_file():
        raise RuntimeError(f"companion build failed: {build.stderr}")
    compiler_files=list((BUILD_ROOT/"CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(compiler_files)!=1:
        raise RuntimeError("cannot uniquely identify CMake compiler record")
    compiler_record=compiler_files[0].read_text(encoding="utf-8",errors="replace")
    compiler_match=re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)',compiler_record)
    compiler_version=compiler_match.group(1) if compiler_match else "UNKNOWN"
    source_paths = [
        UNIT_ROOT / "CMakeLists.txt",
        UNIT_ROOT / ".gitignore",
        *(UNIT_ROOT / p for p in ("src/stage_a_trace.hpp","src/stage_a_trace.cpp","src/stage_a_metrics.hpp","src/stage_a_metrics.cpp","src/stage_a_execution.hpp","src/stage_a_execution.cpp","src/stage_a_main.cpp")),
        *(UNIT_ROOT / p for p in ("scripts/run_v2_1d_stage_a_qa.py","scripts/run_v2_1d_stage_a.py","scripts/generate_calibration_states.py","scripts/create_source_seal.py","tests/test_stage_a_contract.py")),
        FROZEN_SPEC,
        *(KQ_ROOT / p for p in EXPECTED_CANDIDATE_SOURCE_HASHES),
        *(PHYSICAL_ROOT / "src" / p for p in ("q4_layout.cpp","v2_1.hpp","worker_pool.cpp","allocation_guard.cpp")),
    ]
    source_hashes = {str(path.resolve()): powershell_hash(path) for path in source_paths if path.is_file()}
    candidate_identities = verify_frozen_inputs()
    build_log = build.stdout + "\n" + build.stderr
    msbuild_match=re.search(r"MSBuild\s+([0-9][0-9A-Za-z.+-]*)",build_log,re.IGNORECASE)
    msbuild_version=msbuild_match.group(1) if msbuild_match else "UNKNOWN"
    command_lines = [line.strip() for line in build_log.splitlines() if "cl.exe" in line.lower() or "link.exe" in line.lower() or "CommandLine =" in line]
    manifest = {
        "schema":"omega-v2-1d-companion-build-manifest-v1",
        "build_classification":"QA_COMPANION_BUILD_ONLY_NO_STAGE_A_CELLS",
        "generator":"Visual Studio 17 2022 x64",
        "cmake_executable":cmake,
        "configuration":"Release",
        "compiler_version":compiler_version,
        "msbuild_version":msbuild_version,
        "compile_flags":["/O2","/GL","/arch:AVX2","/fp:precise","/W4","/EHsc"],
        "link_flags":["/LTCG"],
        "source_sha256":source_hashes,
        "candidate_source_identity":candidate_identities,
        "companion_executable":str(COMPANION_EXE.resolve()),
        "companion_executable_sha256":powershell_hash(COMPANION_EXE),
        "configure_command":[cmake,"-S",str(UNIT_ROOT),"-B",str(BUILD_ROOT),"-G","Visual Studio 17 2022","-A","x64","-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
        "build_command":[cmake,"--build",str(BUILD_ROOT),"--config","Release","--target","omega_v2_1d_stage_a","--clean-first","--verbose"],
        "compile_and_link_command_lines":command_lines,
        "build_log_stdout":str((PREFLIGHT_ROOT/"cmake_build_stdout.log").resolve()),
        "build_log_stderr":str((PREFLIGHT_ROOT/"cmake_build_stderr.log").resolve()),
        "sealed_kq_executable_sha256":candidate_identities["sealed_kq_executable_sha256"],
        "candidate_translation_units_byte_identical":True,
        "scientific_execution_performed":False,
    }
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_text=json.dumps(manifest,indent=2,sort_keys=True)+"\n"
    TRACKABLE_BUILD_MANIFEST.write_text(manifest_text,encoding="utf-8",newline="\n")
    (BUILD_ROOT/"companion_build_manifest.json").write_text(manifest_text,encoding="utf-8",newline="\n")
    print(json.dumps({"build":"PASS","compiler_version":compiler_version,"companion_sha256":manifest["companion_executable_sha256"],"manifest":str(TRACKABLE_BUILD_MANIFEST)},indent=2))


def invoke_qa(mode: str) -> dict[str, Any]:
    if not COMPANION_EXE.is_file():
        raise FileNotFoundError("run --prepare-only before native QA")
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        process = run([str(COMPANION_EXE),"--instrumentation-smoke"])
        write_logs("instrumentation_smoke",process)
        if process.returncode:
            raise RuntimeError(f"d32 instrumentation smoke failed: {process.stderr}")
        report = json.loads(process.stdout)
        report_path=PREFLIGHT_ROOT/"instrumentation_smoke.json"
    elif mode == "metrics":
        process = run([str(COMPANION_EXE),"--metrics-self-test"])
        write_logs("metrics_self_test",process)
        if process.returncode:
            raise RuntimeError(f"metric self-test failed: {process.stderr}")
        report=json.loads(process.stdout)
        report_path=PREFLIGHT_ROOT/"metrics_self_test.json"
    elif mode == "binding":
        identity=verify_frozen_inputs()
        process=run([str(COMPANION_EXE),"--companion-binding",str(V1C_WEIGHT_STREAM),str(KQ_OUTPUTS)])
        write_logs("companion_binding",process)
        if process.returncode:
            raise RuntimeError(f"COMPANION_BINDING_INVALID: {process.stderr}")
        report=json.loads(process.stdout)
        report["identity"]=identity
        report_path=PREFLIGHT_ROOT/"companion_binding.json"
    elif mode == "d640":
        canonical=PREFLIGHT_ROOT/"d640_q4_canonical_v1_retry10.bin"
        if canonical.exists():
            raise FileExistsError(f"d640 canonical binding artifact already exists: {canonical}")
        process=run([str(COMPANION_EXE),"--d640-weight-binding",str(canonical)])
        write_logs("d640_weight_binding",process)
        if process.returncode:
            raise RuntimeError(f"D640_WEIGHT_BINDING_HOLD: {process.stderr}")
        report=json.loads(process.stdout)
        report["canonical_sha256"]=powershell_hash(canonical)
        report["canonical_serialization_version"]="OMEGA-V2-1D-D640-Q4-CANONICAL-V1"
        report["canonical_serialization_status"]="RATIFIED_MD331"
        report_path=PREFLIGHT_ROOT/"d640_weight_binding_retry10.json"
    else:
        raise ValueError(mode)
    report["companion_executable_sha256"]=powershell_hash(COMPANION_EXE)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return report


def prepare_states() -> None:
    from generate_calibration_states import generate

    output_root=BUILD_ROOT/"calibration_inputs"
    if not output_root.parent.exists():
        raise FileNotFoundError(f"build parent missing: {output_root.parent}")
    manifest=generate(output_root)
    print(json.dumps({"state_generation":"PASS","state_count":len(manifest["states"]),"manifest":str((output_root/"state_inputs_manifest.json").resolve()),"native_execution":False},indent=2))


def main() -> int:
    parser=argparse.ArgumentParser(description="V2-1d authorized QA/preflight modes only; no scientific launcher is exposed")
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run",action="store_true")
    modes.add_argument("--unit-static",action="store_true")
    modes.add_argument("--prepare-only",action="store_true")
    modes.add_argument("--metrics-self-test",action="store_true")
    modes.add_argument("--smoke-only",action="store_true")
    modes.add_argument("--companion-binding-only",action="store_true")
    modes.add_argument("--d640-binding-only",action="store_true")
    modes.add_argument("--prepare-states-only",action="store_true")
    args=parser.parse_args()
    if args.dry_run: dry_run()
    elif args.unit_static: run_unit_tests()
    elif args.prepare_only: configure_and_build()
    elif args.metrics_self_test: invoke_qa("metrics")
    elif args.smoke_only: invoke_qa("smoke")
    elif args.companion_binding_only: invoke_qa("binding")
    elif args.d640_binding_only: invoke_qa("d640")
    elif args.prepare_states_only: prepare_states()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
