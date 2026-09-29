from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


DIAG_ROOT = Path(__file__).resolve().parent
KQ_ROOT = DIAG_ROOT.parents[1]
CAMPAIGN_ROOT = KQ_ROOT.parent
REPO_ROOT = KQ_ROOT.parents[2]
V2_0_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
V2_1_ROOT = CAMPAIGN_ROOT / "omega_v2_1_physical"
RESULTS_BASE = KQ_ROOT / "results" / "omega_v2_1b_kernel_qualification"
KQ1_RESULTS = RESULTS_BASE / "candidate_01" / "run_02"
ATTEMPT02_RESULTS = V2_1_ROOT / "results" / "omega_v2_1_physical" / "attempt_02"
RESULTS_BASE = RESULTS_BASE / "candidate_01" / "diagnostics"
RESULTS_ROOT = RESULTS_BASE / "run_02"
BUILD_ROOT = DIAG_ROOT / "build"
EXECUTABLE = BUILD_ROOT / "Release" / "omega_v2_1b_candidate01_diag.exe"
FROZEN_CPU_IDS = [266, 264, 258, 270]
MATRIX_FAMILIES = ("W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def source_files() -> list[Path]:
    return [
        DIAG_ROOT / ".gitignore",
        DIAG_ROOT / "CMakeLists.txt",
        DIAG_ROOT / "component_diagnostic.cpp",
        DIAG_ROOT / "run_candidate01_diagnostic.py",
    ]


def frozen_candidate01_files() -> list[Path]:
    paths = [
        KQ_ROOT / "CMakeLists.txt",
        KQ_ROOT / "OMEGA_V2_1B_KQ_SPEC.md",
        KQ_ROOT / "src" / "kq_candidate.hpp",
        KQ_ROOT / "src" / "q4_kernel_candidate1.cpp",
        KQ_ROOT / "src" / "main.cpp",
        KQ_ROOT / "scripts" / "run_kq.py",
        KQ_ROOT / "scripts" / "pytorch_control.py",
    ]
    return paths


def shared_attempt02_sources() -> list[Path]:
    old_src = V2_1_ROOT / "src"
    return [old_src / name for name in (
        "v2_1.hpp", "hardware_topology.cpp", "q4_layout.cpp", "worker_pool.cpp",
        "cache_controls.cpp", "allocation_guard.cpp",
    )]


def verify_preserved_evidence() -> dict[str, Any]:
    evidence = []
    for label, result_root in (("attempt_02", ATTEMPT02_RESULTS), ("candidate_01 KQ", KQ1_RESULTS)):
        manifest_path = result_root / "artifact_hashes.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            path for path, row in manifest["artifacts"].items()
            if not Path(path).is_file() or sha256_file(Path(path)) != row["sha256"]
        ]
        if mismatches:
            raise RuntimeError(f"DIAGNOSTIC_STOP: {label} evidence hash mismatch: {mismatches}")
        evidence.append({
            "label": label,
            "manifest_path_abs": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "manifested_artifact_count": len(manifest["artifacts"]),
            "all_artifact_hashes_verified": True,
        })
    kq1_build = json.loads((KQ1_RESULTS / "build_manifest.json").read_text(encoding="utf-8"))
    kq1_exe = Path(kq1_build["executable_absolute_path"])
    first_diagnostic = RESULTS_BASE / "candidate01_component_diagnostic.json"
    first_diag_sha = sha256_file(first_diagnostic) if first_diagnostic.is_file() else None
    return {
        "evidence": evidence,
        "previous_diagnostic_attempt_sha256": first_diag_sha,
        "candidate01_kq_executable_abs": str(kq1_exe.resolve()),
        "candidate01_kq_executable_sha256": sha256_file(kq1_exe),
    }


def find_cmake() -> tuple[Path, str]:
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")), Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    vswhere = next((root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe" for root in roots if (root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe").is_file()), None)
    if vswhere is None:
        raise FileNotFoundError("vswhere.exe not found")
    cmakes = subprocess.run(
        [str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.CMake.Project", "-find", r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    installs = subprocess.run(
        [str(vswhere), "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    if not cmakes or not installs:
        raise FileNotFoundError("native MSVC/CMake components missing")
    return Path(cmakes[0]), installs[0]


def source_weight_payload() -> tuple[bytes, str]:
    sys.path.insert(0, str(V2_0_ROOT))
    import torch
    from omega_v2.core import ContractualCoreBlock

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    block = ContractualCoreBlock(512, dtype=torch.float32, seed=20260929)
    payload = b"".join(
        getattr(block, name).detach().contiguous().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
        for name in MATRIX_FAMILIES
    )
    return payload, hashlib.sha256(payload).hexdigest()


def main() -> int:
    if os.name != "nt" or sys.platform != "win32":
        raise RuntimeError("candidate_01 component diagnostic is Windows-native only")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("candidate_01 diagnostic requires main")
    if git("status", "--porcelain", "--", str(DIAG_ROOT.relative_to(REPO_ROOT))):
        raise RuntimeError("candidate_01 diagnostic source must be committed before its separate build")
    if git("status", "--porcelain", "--", str(KQ_ROOT.relative_to(REPO_ROOT))):
        raise RuntimeError("candidate_01 KQ source tree must remain clean/frozen during the component diagnostic")
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"candidate_01 diagnostic result is immutable and already exists: {RESULTS_ROOT}")

    preserved_before = verify_preserved_evidence()
    v2_0_seal = json.loads((V2_0_ROOT / "V2_0_RESULT_SEAL.json").read_text(encoding="utf-8"))
    if v2_0_seal.get("terminal_status") != "OMEGA_V2_0_CONFORMANT_PASS":
        raise RuntimeError("diagnostic reference V2-0 seal does not pass")
    vs_cmake, vs_install = find_cmake()
    source_before = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files()}
    dep_before = {str(path.resolve()): sha256_file(path) for path in [*frozen_candidate01_files(), *shared_attempt02_sources()]}
    cmake_version = subprocess.run([str(vs_cmake), "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    configure = subprocess.run(
        [str(vs_cmake), "-S", str(DIAG_ROOT), "-B", str(BUILD_ROOT), "-G", "Visual Studio 17 2022", "-A", "x64"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if configure.returncode:
        raise RuntimeError("candidate_01 diagnostic configure failed: " + configure.stderr[-4000:])
    build = subprocess.run(
        [str(vs_cmake), "--build", str(BUILD_ROOT), "--config", "Release", "--target", "omega_v2_1b_candidate01_diag", "-j", "8"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if build.returncode:
        raise RuntimeError("candidate_01 diagnostic MSVC Release build failed: " + build.stderr[-5000:])
    if not EXECUTABLE.is_file():
        raise FileNotFoundError("candidate_01 diagnostic executable missing")
    source_after = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files()}
    if source_before != source_after:
        raise RuntimeError("candidate_01 diagnostic source changed during build")
    dep_after = {str(path.resolve()): sha256_file(path) for path in [*frozen_candidate01_files(), *shared_attempt02_sources()]}
    if dep_before != dep_after:
        raise RuntimeError("candidate_01 KQ/attempt_02 dependencies changed during diagnostic build")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    payload, weight_stream_sha = source_weight_payload()
    attempt02_config = json.loads((ATTEMPT02_RESULTS / "benchmark_config.json").read_text(encoding="utf-8"))
    workers = attempt02_config["selected_workers"]
    with tempfile.TemporaryDirectory(prefix="omega_v2_1b_candidate01_diag_") as temp:
        weights_path = Path(temp) / "v2_0_fp32_weights.tmp"
        weights_path.write_bytes(payload)
        env = os.environ.copy()
        env.update({
            "OMEGA_V2_1B_CANDIDATE01_DIAG_RESULTS_ROOT": str(RESULTS_ROOT.resolve()),
            "OMEGA_V2_1B_CANDIDATE01_DIAG_WEIGHTS": str(weights_path.resolve()),
            "OMEGA_V2_1B_CANDIDATE01_DIAG_SHARD_WEIGHTS": ",".join(str(worker["h0_v_i"]) for worker in workers),
        })
        native = subprocess.run(
            [str(EXECUTABLE), "--run-candidate01-component-diagnostic"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=2 * 60 * 60, env=env,
        )
    (RESULTS_ROOT / "native_stdout.log").write_text(native.stdout, encoding="utf-8", newline="\n")
    (RESULTS_ROOT / "native_stderr.log").write_text(native.stderr, encoding="utf-8", newline="\n")
    if native.returncode:
        raise RuntimeError("candidate_01 diagnostic process failed: " + native.stderr[-4000:])
    native_path = RESULTS_ROOT / "candidate01_component_diagnostic.json"
    native_report = json.loads(native_path.read_text(encoding="utf-8"))
    if native_report.get("attempt03_executed") is not False or native_report.get("a_b_c_executed") is not False or native_report.get("residency_gates_evaluated") is not False:
        raise RuntimeError("candidate_01 component diagnostic violated its KQ-only scope")

    preserved_after = verify_preserved_evidence()
    if preserved_after != preserved_before:
        raise RuntimeError("STOP: candidate_01 diagnostic changed attempt_02 or KQ1 evidence")
    build_manifest = {
        "schema": "omega-v2-1b-candidate01-diagnostic-build-v1",
        "diagnostic_only": True,
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "diagnostic_source_commit": git("rev-parse", "HEAD"),
        "candidate01_kq_implementation_commit": json.loads((KQ1_RESULTS / "build_manifest.json").read_text(encoding="utf-8"))["implementation_commit"],
        "visual_studio_installation": vs_install,
        "cmake_path": str(vs_cmake),
        "cmake_version": cmake_version,
        "generator": "Visual Studio 17 2022 x64",
        "configuration": "Release",
        "compiler": "MSVC 19.44.35229.0",
        "compiler_flags": ["/O2", "/GL", "/arch:AVX2", "/fp:precise", "/W4", "/EHsc", "/LTCG"],
        "diagnostic_executable_absolute_path": str(EXECUTABLE.resolve()),
        "diagnostic_executable_sha256": sha256_file(EXECUTABLE),
        "diagnostic_source_sha256": source_before,
        "frozen_candidate01_and_attempt02_dependencies_sha256": dep_before,
        "weight_source": "V2-0 ContractualCoreBlock FP32 seed=20260929, transient diagnostic input",
        "weight_stream_sha256": weight_stream_sha,
        "attempt02_preserved_manifest_sha256": preserved_before["evidence"][0]["manifest_sha256"],
        "candidate01_kq_preserved_manifest_sha256": preserved_before["evidence"][1]["manifest_sha256"],
        "configure_stdout": configure.stdout,
        "configure_stderr": configure.stderr,
        "build_stdout": build.stdout,
        "build_stderr": build.stderr,
    }
    write_json(RESULTS_ROOT / "build_manifest.json", build_manifest)
    report_payload = {
        "schema": "omega-v2-1b-candidate01-component-diagnostic-report-v1",
        "terminal_status": "CANDIDATE01_COMPONENT_DIAGNOSTIC_COMPLETE",
        "candidate_id": "KQ1_DEQUANT_ROW_REUSE",
        "scope": "diagnostic_only",
        "attempt03_executed": False,
        "a_b_c_executed": False,
        "rho_residency_gate_read_or_calculated": False,
        "attempt02_and_candidate01_kq_unchanged": True,
        "measurement_contract": {"hardware": "attempt_02 four selected P-cores", "d": 512, "m": 8, "K": 4, "warmups": 10, "samples": 31, "timer": "QPC"},
        "preserved_evidence": preserved_after,
        "native_component_breakdown": native_report,
    }
    write_json(RESULTS_ROOT / "summary_metrics.json", report_payload)
    markdown = [
        "# Candidate-01 Component Diagnostic (diagnostic only)", "",
        f"- terminal_status: `{report_payload['terminal_status']}`",
        f"- candidate_01_KQ_result_abs: `{KQ1_RESULTS.resolve()}`",
        f"- diagnostic_result_abs: `{RESULTS_ROOT.resolve()}`",
        "- candidate_01 executable: `not executed; diagnostic build used a separate executable`",
        "- attempt_02/candidate_01 artifacts: `hash-verified and unchanged`",
        "- A/B/C: `not run`; attempt_03: `not run`; rho: `not read or calculated`",
        "", "## Requested component timing breakdown", "",
        f"- dequant-only full K4, including checksum and 28 dispatches: `{native_report['dequant_only_full_K4_28_dispatches']['median_seconds_including_28_dispatches_and_row_checksum']:.9f} s`",
        f"- 28 empty dispatches: `{native_report['dispatch_overhead_28_empty']['median_seconds']:.9f} s`",
        f"- estimated dequant+checksum, subtracting empty dispatches: `{native_report['dequant_only_full_K4_28_dispatches']['estimated_dequant_plus_checksum_no_dispatch_seconds']:.9f} s`",
        f"- row-timed dequant wall estimate (dispatch/checksum excluded): `{native_report['dequant_only_full_K4_28_dispatches']['per_row_qpc_dequant_wall_estimate_seconds']:.9f} s`",
        f"- row-timed checksum wall estimate (reported separately): `{native_report['dequant_only_full_K4_28_dispatches']['per_row_qpc_checksum_wall_estimate_seconds']:.9f} s`",
        f"- dequant+FMA Q4 matmuls full K4, including 28 dispatches: `{native_report['q4_dequant_plus_fma_full_K4_28_dispatches']['median_seconds_including_dispatches']:.9f} s`",
        f"- estimated dequant+FMA, subtracting empty dispatches: `{native_report['q4_dequant_plus_fma_full_K4_28_dispatches']['estimated_dequant_plus_FMA_no_dispatch_seconds']:.9f} s`",
        f"- full-block QKVO: `{native_report['full_block_component_breakdown']['median_QKVO_seconds']:.9f} s`",
        f"- full-block SwiGLU: `{native_report['full_block_component_breakdown']['median_SwiGLU_seconds']:.9f} s`",
        f"- RMSNorm/residual: `{native_report['full_block_component_breakdown']['median_RMS_residual_seconds']:.9f} s`",
        f"- QKV matmuls / attention / W_O: `{native_report['full_block_component_breakdown']['median_QKV_matmul_seconds']:.9f}` / `{native_report['full_block_component_breakdown']['median_attention_seconds']:.9f}` / `{native_report['full_block_component_breakdown']['median_WO_seconds']:.9f} s`",
        f"- gate+up / SiLU+Hadamard / down: `{native_report['full_block_component_breakdown']['median_gate_up_seconds']:.9f}` / `{native_report['full_block_component_breakdown']['median_SiLU_hadamard_seconds']:.9f}` / `{native_report['full_block_component_breakdown']['median_down_seconds']:.9f} s`",
        "", "## Method note", "",
        "Dequant-only consumes each transient row tile into a checksum to keep decode writes observable. It reports the full wall interval, empty-dispatch-subtracted interval, and row-QPC dequant/checksum estimates separately. The q4_matmul batch includes dequant plus FMA for seven matrices over four rounds; its empty-dispatch-subtracted value is an estimate. Component timers are diagnostic, not gates.",
        "", "## SHA-256", "",
    ]
    report_path = RESULTS_ROOT / "CANDIDATE01_COMPONENT_DIAGNOSTIC.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    sidecar = RESULTS_ROOT / "CANDIDATE01_COMPONENT_DIAGNOSTIC.md.sha256"
    report_hash = sha256_file(report_path)
    sidecar.write_text(report_hash + "\n", encoding="ascii", newline="\n")
    artifact_paths = [
        *source_files(), *frozen_candidate01_files(), *shared_attempt02_sources(), EXECUTABLE,
        RESULTS_ROOT / "candidate01_component_diagnostic.json", RESULTS_ROOT / "build_manifest.json",
        RESULTS_ROOT / "summary_metrics.json", RESULTS_ROOT / "native_stdout.log", RESULTS_ROOT / "native_stderr.log",
        report_path, sidecar,
    ]
    artifact_manifest = {
        "schema": "omega-v2-1b-candidate01-diagnostic-artifact-hashes-v1",
        "report_self_sha256": report_hash,
        "candidate01_kq_manifest_sha256": preserved_after["evidence"][1]["manifest_sha256"],
        "attempt02_manifest_sha256": preserved_after["evidence"][0]["manifest_sha256"],
        "artifacts": {
            str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths if path.is_file()
        },
    }
    write_json(RESULTS_ROOT / "artifact_hashes.json", artifact_manifest)
    verified = all(record["sha256"] == sha256_file(Path(path)) for path, record in artifact_manifest["artifacts"].items())
    verified = verified and sha256_file(report_path) == sidecar.read_text(encoding="ascii").strip()
    if not verified:
        raise RuntimeError("candidate_01 diagnostic SHA verification failed")
    print(json.dumps({
        "terminal_status": report_payload["terminal_status"],
        "results_root_abs": str(RESULTS_ROOT.resolve()),
        "report_abs": str(report_path.resolve()),
        "report_sha256": report_hash,
        "artifact_hashes_abs": str((RESULTS_ROOT / "artifact_hashes.json").resolve()),
        "artifact_count": len(artifact_manifest["artifacts"]),
        "hashes_verified": verified,
        "diagnostic": native_report,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
