from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

UNIT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(UNIT_ROOT/"scripts"))
import run_v2_1d_stage_a as launcher


def main() -> int:
    parser=argparse.ArgumentParser(description="Post-commit real precondition QA; invokes verify_preconditions only, never native Stage A")
    parser.add_argument("--source-seal",type=Path,default=UNIT_ROOT/"SOURCE_SEAL.json")
    parser.add_argument("--source-seal-sha256")
    args=parser.parse_args()
    seal_path=args.source_seal.resolve()
    seal_sha=args.source_seal_sha256 or launcher.powershell_hash(seal_path)
    results_root=UNIT_ROOT/"results"
    marker_dir=results_root/"stage_a_one_shot_ids"
    if marker_dir.exists():
        raise RuntimeError("post-seal precondition QA requires no one-shot marker directory")
    invoked_native=[]
    original_run=launcher.subprocess.run

    def audited_run(command,*run_args,**run_kwargs):
        if any("--run-calibration" in str(part) for part in command):
            invoked_native.append(command)
            raise AssertionError("verify_preconditions attempted a native Stage A invocation")
        return original_run(command,*run_args,**run_kwargs)

    launcher.subprocess.run=audited_run
    try:
        result=launcher.verify_preconditions(argparse.Namespace(source_seal=seal_path,source_seal_sha256=seal_sha))
    finally:
        launcher.subprocess.run=original_run
    if marker_dir.exists() or invoked_native:
        raise AssertionError("precondition QA created marker or invoked native calibration")
    print(json.dumps({
        "status":"REAL_VERIFY_PRECONDITIONS_PASS",
        "preconditions":result,
        "scientific_cells_executed":0,
        "one_shot_marker_reserved":False,
        "native_run_calibration_invoked":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
