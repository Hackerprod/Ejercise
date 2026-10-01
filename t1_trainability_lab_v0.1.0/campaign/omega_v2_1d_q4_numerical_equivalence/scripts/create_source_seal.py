from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UNIT_ROOT.parents[2]
BUILD_ROOT = UNIT_ROOT / "build_qa"
QA_ROOT = BUILD_ROOT / "preflight"
STATE_ROOT = BUILD_ROOT / "calibration_inputs"
BUILD_MANIFEST = UNIT_ROOT / "OMEGA_V2_1D_COMPANION_BUILD_MANIFEST.json"
STATE_MANIFEST = STATE_ROOT / "state_inputs_manifest.json"
FROZEN_SPEC = UNIT_ROOT / "OMEGA_V2_1D_STAGE_A_SPEC_FROZEN.md"
KQ_ROOT = UNIT_ROOT.parent / "omega_v2_1b_candidate_02"
KQ_RESULTS = UNIT_ROOT.parent / "omega_v2_1b_kernel_qualification" / "results" / "omega_v2_1b_kernel_qualification" / "candidate_02" / "run_01"
SEAL_PATH = UNIT_ROOT / "SOURCE_SEAL.json"


def git(*args: str) -> str:
    return subprocess.run(["git","-C",str(REPO_ROOT),*args],capture_output=True,text=True,check=True).stdout.strip()


def get_file_hash(path: Path) -> str:
    escaped=str(path.resolve()).replace("'","''")
    command=f"(Get-FileHash -Algorithm SHA256 -LiteralPath '{escaped}').Hash"
    process=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",command],capture_output=True,text=True,check=True)
    digest=process.stdout.strip().lower()
    if len(digest)!=64:
        raise RuntimeError(f"Get-FileHash returned an invalid SHA-256 for {path}")
    return digest


def collect_artifacts() -> list[Path]:
    if not BUILD_MANIFEST.is_file() or not STATE_MANIFEST.is_file():
        raise FileNotFoundError("source/build manifest or state input manifest is missing")
    d640_report=QA_ROOT/"d640_weight_binding_retry10.json"
    if not d640_report.is_file():
        raise FileNotFoundError(d640_report)
    d640=json.loads(d640_report.read_text(encoding="utf-8"))
    build=json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    kq_build=json.loads((KQ_RESULTS/"build_manifest.json").read_text(encoding="utf-8"))
    paths=[
        UNIT_ROOT/".gitignore",
        UNIT_ROOT/"CMakeLists.txt",
        FROZEN_SPEC,
        UNIT_ROOT/"OMEGA_V2_1D_STAGE_A_SPEC_DRAFT_v2.md",
        BUILD_MANIFEST,
        Path(build["companion_executable"]),
        Path(kq_build["executable_absolute_path"]),
        KQ_ROOT/"src"/"full_block_candidate2.cpp",
        KQ_ROOT/"src"/"q4_kernel_candidate2.cpp",
        KQ_ROOT/"src"/"kq_candidate2.hpp",
        KQ_RESULTS/"artifact_hashes.json",
        KQ_RESULTS/"build_manifest.json",
        KQ_RESULTS/"source_weight_manifest.json",
        STATE_MANIFEST,
        Path(d640["canonical_stream_path"]),
    ]
    for subdir in ("src","scripts","tests"):
        paths.extend(path for path in (UNIT_ROOT/subdir).glob("*") if path.is_file())
    paths.extend(path for path in QA_ROOT.glob("*.json") if path.is_file())
    paths.extend(path for path in STATE_ROOT.glob("*.f32") if path.is_file())
    unique={str(path.resolve()):path for path in paths}
    missing=[path for path in unique.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source seal inputs missing: {missing}")
    return sorted(unique.values(),key=lambda path:path.resolve().relative_to(REPO_ROOT.resolve()).as_posix().lower())


def create_seal() -> dict[str,Any]:
    if SEAL_PATH.exists():
        raise FileExistsError("SOURCE_SEAL.json already exists; refusing overwrite")
    if git("status","--porcelain"):
        raise RuntimeError("SOURCE_SEAL can only be created after source/build/preflight commit with a clean working tree")
    head=git("rev-parse","HEAD")
    artifacts=[]
    for path in collect_artifacts():
        artifacts.append({
            "path":path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
            "sha256":get_file_hash(path),
            "size_bytes":path.stat().st_size,
        })
    seal={
        "schema":"omega-v2-1d-stage-a-source-seal-v1",
        "classification":"SOURCE_BUILD_AND_PREFLIGHT_SEAL_NO_SCIENTIFIC_CELLS",
        "git_head_commit":head,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "hash_algorithm":"SHA-256 via Get-FileHash",
        "artifact_count":len(artifacts),
        "artifacts":artifacts,
        "scientific_cells_executed":0,
    }
    SEAL_PATH.write_text(json.dumps(seal,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    return seal


def main() -> int:
    parser=argparse.ArgumentParser(description="Create versionable SOURCE_SEAL.json after committing sources/build/preflights")
    parser.add_argument("--output",type=Path,default=SEAL_PATH)
    args=parser.parse_args()
    if args.output.resolve()!=SEAL_PATH.resolve():
        raise ValueError("SOURCE_SEAL output path is fixed by contract")
    seal=create_seal()
    print(json.dumps({"status":"SOURCE_SEAL_CREATED","path":str(SEAL_PATH.resolve()),"git_head_commit":seal["git_head_commit"],"artifact_count":seal["artifact_count"],"scientific_cells_executed":0},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
