from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import create_source_seal

import create_source_seal


UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
BUILD_ROOT = UNIT_ROOT / "build_qa"
QA_ROOT = BUILD_ROOT / "preflight"
COMPANION_EXE = BUILD_ROOT / "Release" / "omega_v2_1d_stage_a.exe"
BUILD_MANIFEST = UNIT_ROOT / "OMEGA_V2_1D_COMPANION_BUILD_MANIFEST.json"
STATE_ROOT = BUILD_ROOT / "calibration_inputs"
STATE_MANIFEST = STATE_ROOT / "state_inputs_manifest.json"
COMPANION_BINDING = QA_ROOT / "companion_binding.json"
SMOKE_REPORT = QA_ROOT / "instrumentation_smoke.json"
UNIT_TEST_REPORT = QA_ROOT / "unit_tests_report.json"
D640_BINDING = QA_ROOT / "d640_weight_binding_retry10.json"
FROZEN_SPEC = UNIT_ROOT / "OMEGA_V2_1D_STAGE_A_SPEC_FROZEN.md"

EXPECTED_SPEC_SHA256 = "186e58b5b762c36791c27b5e8957ecd4e64af58d631a3477882218f10c19ef2d"
EXPECTED_KQ_EXE_SHA256 = "be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f"
EXPECTED_D512_STREAM_SHA256 = "65b9179e13d09513765eba3f56f0368480c03237eb1bbbac48c0f383b56ecbb8"
EXPECTED_D640_Q4_CHECKSUM = 15453147065333665836
GO_TOKEN = "STAGE_A_ONE_CALIBRATION_GO"
EXPECTED_UNIT_TEST_COUNT = 12


def powershell_hash(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "''")
    command = f"(Get-FileHash -Algorithm SHA256 -LiteralPath '{escaped}').Hash"
    process = subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",command],capture_output=True,text=True,check=True)
    digest=process.stdout.strip().lower()
    if len(digest)!=64:
        raise RuntimeError(f"Get-FileHash returned invalid SHA-256 for {path}")
    return digest


def git(*args: str) -> str:
    return subprocess.run(["git","-C",str(REPO_ROOT),*args],capture_output=True,text=True,check=True).stdout.strip()


def require_committed_clean(path: Path, label: str) -> None:
    relative=path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    git("ls-files","--error-unmatch",relative)
    if git("status","--porcelain","--",relative):
        raise RuntimeError(f"precondition FAIL: {label} is modified")


def verify_state_inputs() -> dict[str,Any]:
    manifest=json.loads(STATE_MANIFEST.read_text(encoding="utf-8"))
    states=manifest.get("states",[])
    if len(states)!=16 or manifest.get("native_rng_used") is not False:
        raise RuntimeError("state input manifest must bind 16 pre-generated states and use no native RNG")
    observed=[]
    for item in states:
        path=Path(item["path"])
        if path.stat().st_size!=int(item["size_bytes"]):
            raise RuntimeError(f"state input size mismatch: {path}")
        digest=powershell_hash(path)
        if digest!=str(item["sha256"]).lower():
            raise RuntimeError(f"state input SHA-256 mismatch: {path}")
        if item["dtype"]!="float32" or item["device"]!="cpu" or item["byte_order"]!="little-endian" or item["layout"]!="contiguous-row-major":
            raise RuntimeError(f"state input contract mismatch: {path}")
        observed.append({"d":int(item["d"]),"m":int(item["m"]),"family":item["family"],"seed":item.get("seed"),"shape":item["shape"],"path":str(path.resolve()),"sha256":digest})
    return {"state_count":len(states),"states":observed,"manifest_sha256":powershell_hash(STATE_MANIFEST),"torch_version":manifest.get("torch_version"),"python_version":manifest.get("python_version")}


def verify_preconditions(args: argparse.Namespace) -> dict[str,Any]:
    require_committed_clean(FROZEN_SPEC,"frozen spec")
    require_committed_clean(BUILD_MANIFEST,"source/build manifest")
    if git("status","--porcelain"):
        raise RuntimeError("precondition FAIL: working tree is not clean")
    spec_hash=powershell_hash(FROZEN_SPEC)
    if spec_hash!=EXPECTED_SPEC_SHA256:
        raise RuntimeError("precondition FAIL: frozen spec SHA-256 changed")
    seal_path=args.source_seal.resolve()
    if not seal_path.is_file():
        raise FileNotFoundError("precondition FAIL: source seal is missing")
    require_committed_clean(seal_path,"source seal")
    seal_hash=powershell_hash(seal_path)
    if seal_hash!=args.source_seal_sha256.lower():
        raise RuntimeError("precondition FAIL: source seal hash differs from supplied identity")
    seal=json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema")!="omega-v2-1d-stage-a-source-seal-v1":
        raise RuntimeError("precondition FAIL: source seal schema mismatch")
    sealed_head=str(seal.get("git_head_commit",""))
    current_head=git("rev-parse","HEAD")
    ancestry=subprocess.run(["git","-C",str(REPO_ROOT),"merge-base","--is-ancestor",sealed_head,current_head],capture_output=True,text=True)
    if ancestry.returncode!=0:
        raise RuntimeError("precondition FAIL: sealed source commit is not an ancestor of HEAD")
    sealed_artifacts={item["path"]:item for item in seal.get("artifacts",[])}
    expected_paths={path.resolve().relative_to(REPO_ROOT.resolve()).as_posix():path for path in create_source_seal.collect_artifacts()}
    if set(sealed_artifacts)!=set(expected_paths):
        raise RuntimeError("precondition FAIL: source seal artifact inventory differs from current preflight inventory")
    for relative,path in expected_paths.items():
        item=sealed_artifacts[relative]
        if powershell_hash(path)!=str(item.get("sha256","")).lower() or path.stat().st_size!=int(item.get("size_bytes",-1)):
            raise RuntimeError(f"precondition FAIL: sealed artifact drift at {relative}")

    build=json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    companion_hash=powershell_hash(COMPANION_EXE)
    if companion_hash!=build["companion_executable_sha256"]:
        raise RuntimeError("precondition FAIL: companion executable differs from build manifest")
    for source_path,expected_hash in build["source_sha256"].items():
        if powershell_hash(Path(source_path))!=str(expected_hash).lower():
            raise RuntimeError(f"precondition FAIL: source drift at {source_path}")
    if build["sealed_kq_executable_sha256"].lower()!=EXPECTED_KQ_EXE_SHA256 or not build.get("candidate_translation_units_byte_identical"):
        raise RuntimeError("precondition FAIL: companion candidate identity mismatch")

    kq_exe=Path(build["candidate_source_identity"]["sealed_kq_executable"])
    if powershell_hash(kq_exe)!=EXPECTED_KQ_EXE_SHA256:
        raise RuntimeError("precondition FAIL: sealed candidate_02 KQ executable hash mismatch")
    if build["candidate_source_identity"]["d512_fp32_weight_stream_sha256"].lower()!=EXPECTED_D512_STREAM_SHA256:
        raise RuntimeError("precondition FAIL: d512 FP32 source weight stream identity mismatch")
    d512_stream=Path(build["candidate_source_identity"]["d512_fp32_weight_stream"])
    if powershell_hash(d512_stream)!=EXPECTED_D512_STREAM_SHA256:
        raise RuntimeError("precondition FAIL: d512 FP32 source weight stream hash mismatch")

    binding=json.loads(COMPANION_BINDING.read_text(encoding="utf-8"))
    if binding.get("status")!="COMPANION_BINDING_PASS" or binding.get("pass") is not True or len(binding.get("cells",[]))!=3 or binding.get("companion_executable_sha256")!=companion_hash:
        raise RuntimeError("precondition FAIL: companion binding is not PASS")
    smoke=json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    if smoke.get("pass") is not True or smoke.get("classification")!="INSTRUMENTATION_SMOKE_NOT_SCIENTIFIC" or smoke.get("companion_executable_sha256")!=companion_hash:
        raise RuntimeError("precondition FAIL: d32 instrumentation smoke is not PASS")
    unit_report=json.loads(UNIT_TEST_REPORT.read_text(encoding="utf-8"))
    if unit_report.get("status")!="PASS" or unit_report.get("test_count")!=EXPECTED_UNIT_TEST_COUNT or unit_report.get("unittest_ok_summary") is not True or unit_report.get("return_code")!=0:
        raise RuntimeError("precondition FAIL: unit/static tests are not PASS")
    d640=json.loads(D640_BINDING.read_text(encoding="utf-8"))
    canonical=Path(d640["canonical_stream_path"])
    if d640.get("status")!="D640_Q4_BINDING_PASS" or int(d640.get("regenerated_q4_checksum",0))!=EXPECTED_D640_Q4_CHECKSUM or d640.get("companion_executable_sha256")!=companion_hash:
        raise RuntimeError("precondition FAIL: historical d640 Q4 checksum binding not PASS")
    if d640.get("serialization_version")!="OMEGA-V2-1D-D640-Q4-CANONICAL-V1" or powershell_hash(canonical)!=str(d640.get("canonical_sha256","")).lower():
        raise RuntimeError("precondition FAIL: canonical d640 Q4 SHA-256 binding mismatch")
    inputs=verify_state_inputs()
    return {
        "preconditions":"PASS",
        "spec_sha256":spec_hash,
        "source_seal_sha256":seal_hash,
        "source_seal_git_head_commit":sealed_head,
        "source_build_manifest_sha256":powershell_hash(BUILD_MANIFEST),
        "companion_executable_sha256":companion_hash,
        "sealed_kq_executable_sha256":EXPECTED_KQ_EXE_SHA256,
        "companion_binding":"COMPANION_BINDING_PASS",
        "d640_q4_checksum":EXPECTED_D640_Q4_CHECKSUM,
        "d640_canonical_sha256":d640["canonical_sha256"],
        "d640_canonical_stream":str(canonical.resolve()),
        "d512_weight_stream":str(d512_stream.resolve()),
        "state_inputs":inputs,
    }


def main() -> int:
    parser=argparse.ArgumentParser(description="Single guarded V2-1d Stage A calibration launcher; do not use before all §18 preconditions pass")
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--output-root",required=True,type=Path)
    parser.add_argument("--source-seal",required=True,type=Path)
    parser.add_argument("--source-seal-sha256",required=True)
    parser.add_argument("--go-calibration",required=True)
    args=parser.parse_args()
    args.output_root=args.output_root.resolve()
    args.source_seal=args.source_seal.resolve()
    if args.go_calibration!=GO_TOKEN:
        raise RuntimeError("Stage A launch HOLD: explicit GO token missing")
    if not re.fullmatch(r"[A-Za-z0-9._-]+",args.run_id):
        raise ValueError("run_id must contain only ASCII letters, digits, dot, underscore, or hyphen")
    if args.output_root.exists():
        raise FileExistsError("Stage A run ID/output root already exists; rerun is prohibited")
    preconditions=verify_preconditions(args)
    results_root=UNIT_ROOT/"results"
    results_root.mkdir(parents=True,exist_ok=True)
    run_id_registry=results_root/"stage_a_one_shot_ids"
    run_id_registry.mkdir(parents=True,exist_ok=True)
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    run_id_marker=run_id_registry/(args.run_id+".json")
    if run_id_marker.exists():
        raise FileExistsError("Stage A run ID was already used; no rerun is allowed")
    run_id_marker.write_text(json.dumps({"run_id":args.run_id,"status":"RESERVED_BEFORE_LAUNCH","preconditions":preconditions},indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    native_output=args.output_root/"scientific_output"
    command=[str(COMPANION_EXE),"--run-calibration",preconditions["d512_weight_stream"],str(STATE_ROOT),preconditions["d640_canonical_stream"],preconditions["d640_canonical_sha256"],str(native_output),GO_TOKEN]
    args.output_root.mkdir(parents=True,exist_ok=False)
    started_utc=datetime.now(timezone.utc).isoformat()
    started_ns=time.time_ns()
    wall_start=time.time_ns()
    process=subprocess.run(command,cwd=REPO_ROOT,capture_output=True,text=True)
    wall_ns=time.time_ns()-wall_start
    ended_ns=time.time_ns()
    ended_utc=datetime.now(timezone.utc).isoformat()
    (args.output_root/"stdout.log").write_text(process.stdout,encoding="utf-8",newline="\n")
    (args.output_root/"stderr.log").write_text(process.stderr,encoding="utf-8",newline="\n")
    artifacts={}
    if native_output.exists():
        for artifact in sorted(path for path in native_output.rglob("*") if path.is_file()):
            artifacts[str(artifact.resolve())]={"sha256":powershell_hash(artifact),"size_bytes":artifact.stat().st_size}
    for artifact in (args.output_root/"stdout.log",args.output_root/"stderr.log"):
        artifacts[str(artifact.resolve())]={"sha256":powershell_hash(artifact),"size_bytes":artifact.stat().st_size}
    artifact_manifest={"schema":"omega-v2-1d-stage-a-artifact-hashes-v1","run_id":args.run_id,"artifacts":artifacts}
    artifact_manifest_path=args.output_root/"artifact_hashes.json"
    artifact_manifest_path.write_text(json.dumps(artifact_manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    launch={
        "schema":"omega-v2-1d-external-launch-metadata-v1",
        "run_id":args.run_id,
        "command":command,
        "utc_start":started_utc,
        "utc_end":ended_utc,
        "start_time_ns":started_ns,
        "end_time_ns":ended_ns,
        "wall_clock_operational_only_ns":wall_ns,
        "return_code":process.returncode,
        "stdout_path":str((args.output_root/"stdout.log").resolve()),
        "stderr_path":str((args.output_root/"stderr.log").resolve()),
        "preconditions":preconditions,
        "artifact_manifest_path":str(artifact_manifest_path.resolve()),
        "artifact_manifest_sha256":powershell_hash(artifact_manifest_path),
        "qpc_used":False,
        "scientific_performance_timing":False,
    }
    (args.output_root/"launcher_metadata.json").write_text(json.dumps(launch,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    run_id_marker.write_text(json.dumps({"run_id":args.run_id,"status":"COMPLETED" if process.returncode==0 else "EXECUTION_FAILURE","launcher_metadata":str((args.output_root/"launcher_metadata.json").resolve())},indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    if process.returncode:
        raise RuntimeError(f"EXECUTION_FAILURE: preserved Stage A run at {args.output_root}")
    print(json.dumps({"status":"STAGE_A_DIAGNOSTIC_COMPLETE","run_id":args.run_id,"output_root":str(args.output_root.resolve()),"cells":32},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
