"""Run three fresh-process local M=8 FC2 dWeight cost diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
STEP_A_ROOT = HERE / "results" / "fc2_dweight_local_reduction_step_a"
CAPTURE_MANIFEST = STEP_A_ROOT / "capture_manifest.json"
DEFAULT_OUTPUT = HERE / "results" / "fc2_dweight_local_cost_diagnostic"
DEFAULT_EXE = HERE / "native" / "build-fc2-dweight-m8-candidate" / "Release" / "omega_fc2_dweight_local_cost_diagnostic.exe"
MAGIC = b"FC2M8D1\0"
EXPECTED_GROUPS = 4
MODES = ("full_cost", "accumulation_only")
FIXTURE_LABELS = ("A_update2_window0", "B_update3_window1")

PAYLOAD_NAMES = (
    "fc1_preactivation",
    "activated",
    "output_gradient",
    "dweight_before",
    "dweight_control_after",
    "dweight_candidate_after",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_certified_cases(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "CAPTURE_AND_LOCAL_EQUIVALENCE_PASS" or manifest.get("timing_performed") is not False:
        raise ValueError("Step-A capture manifest identity/status mismatch")
    cases = [group for fixture in manifest["fixtures"] for group in fixture["captured_groups"]]
    if len(cases) != EXPECTED_GROUPS:
        raise ValueError(f"expected {EXPECTED_GROUPS} certified groups, found {len(cases)}")
    input_root.mkdir()
    source_fixtures = []
    for fixture in manifest["fixtures"]:
        source = Path(fixture["fixture"])
        if not source.is_absolute():
            source = HERE / "results" / "recurrent_backward_replay_final" / "fixtures" / source.name
        if not source.is_file() or _file_sha256(source) != fixture["fixture_sha256"]:
            raise ValueError(f"certified fixture identity mismatch: {source}")
        source_fixtures.append({"path": source.as_posix(), "sha256": fixture["fixture_sha256"], "fixture": source.name})

    persisted: list[dict[str, Any]] = []
    for group in cases:
        npz_path = STEP_A_ROOT / group["capture_file"]
        if _file_sha256(npz_path) != group["capture_file_sha256"]:
            raise ValueError(f"captured group identity mismatch: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.ascontiguousarray(archive[name].astype("<f4", copy=False)) for name in PAYLOAD_NAMES}
        for name, array in arrays.items():
            descriptor = group["arrays"][name]
            if list(array.shape) != descriptor["shape"] or _sha256(array.tobytes(order="C")) != descriptor["sha256"]:
                raise ValueError(f"captured array changed: {group['capture_file']}:{name}")
        if not np.any(arrays["dweight_before"] != 0.0):
            raise ValueError(f"selected group has zero dW_old: {group['capture_file']}")
        if arrays["dweight_control_after"].tobytes() != arrays["dweight_candidate_after"].tobytes():
            raise ValueError(f"captured control/candidate are not bitwise equal: {group['capture_file']}")

        case_id = Path(group["capture_file"]).stem
        binary_path = input_root / f"{case_id}.bin"
        binary_payload = bytearray(MAGIC)
        for name in PAYLOAD_NAMES:
            binary_payload.extend(arrays[name].tobytes(order="C"))
        binary_path.write_bytes(binary_payload)
        persisted.append({
            "case_id": case_id,
            "fixture": group["fixture"],
            "fixture_id": group["fixture_id"],
            "fixture_sha256": group["fixture_sha256"],
            "update_index": group["update_index"],
            "window_index": group["window_index"],
            "position": group["position"],
            "round": group["round"],
            "global_batch": group["global_batch"],
            "worker_index": group["worker_index"],
            "slots": group["slots"],
            "payload_shapes": {name: list(arrays[name].shape) for name in PAYLOAD_NAMES},
            "capture_npz": group["capture_file"],
            "capture_npz_sha256": group["capture_file_sha256"],
            "input_binary": binary_path.name,
            "input_binary_bytes": len(binary_payload),
            "input_binary_sha256": _sha256(bytes(binary_payload)),
            "dweight_before_nonzero_elements": group["dweight_before_nonzero_elements"],
        })
    return persisted, source_fixtures


def _statistics(values: Sequence[int]) -> dict[str, float | int]:
    numbers = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": int(numbers.size),
        "mean_ns": float(np.mean(numbers)),
        "median_ns": float(np.median(numbers)),
        "stddev_ns_population": float(np.std(numbers, ddof=0)),
        "min_ns": int(np.min(numbers)),
        "max_ns": int(np.max(numbers)),
    }


def _aggregate(process_reports: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], exe_identity: Mapping[str, Any]) -> dict[str, Any]:
    if len(process_reports) != 3:
        raise ValueError("local diagnostic requires exactly three fresh process reports")
    per_fixture: dict[str, dict[str, dict[str, Any]]] = {}
    case_by_id = {str(case["case_id"]): case for case in cases}
    per_case_raw: dict[str, dict[str, dict[str, list[int]]]] = {
        case_id: {mode: {"control": [], "candidate": []} for mode in MODES} for case_id in case_by_id
    }
    for process_index, process in enumerate(process_reports):
        if process.get("status") != "LOCAL_COST_PROCESS_PASS" or process.get("process_index") != process_index:
            raise ValueError(f"local process report failed or out of order: index {process_index}")
        if process.get("worker_threads") != 1 or process.get("warmup_pairs_per_group_modality") != 5 or process.get("measured_pairs_per_group_modality") != 30:
            raise ValueError("process report did not satisfy local cost protocol")
        observed = {str(group["case_id"]): group for group in process["groups"]}
        if set(observed) != set(case_by_id):
            raise ValueError(f"process group IDs mismatch: {sorted(observed)}")
        for case_id, group in observed.items():
            if group.get("initial_dW_nonzero") is not True:
                raise ValueError(f"group dW_old is not nonzero: {case_id}")
            for mode in MODES:
                payload = group[mode]
                if mode == "full_cost" and payload.get("preparation_included") is not True:
                    raise ValueError("primary full-cost measurement omitted preparation")
                if mode == "accumulation_only" and not (
                    payload.get("diagnostic_only") is True and payload.get("excludes_operand_preparation") is True
                ):
                    raise ValueError("secondary accumulation-only labeling is incomplete")
                samples = payload.get("samples", [])
                if len(samples) != 30 or {sample.get("order") for sample in samples} != {
                    "control_then_candidate", "candidate_then_control"
                }:
                    raise ValueError(f"order/sample count mismatch for {case_id}/{mode}")
                group_index = int(group["group_index"])
                for index, sample in enumerate(samples):
                    control_first = ((index + group_index + process_index) & 1) == 0
                    expected_order = "control_then_candidate" if control_first else "candidate_then_control"
                    if sample.get("order") != expected_order or int(sample["control_ns"]) <= 0 or int(sample["candidate_ns"]) <= 0:
                        raise ValueError(f"invalid paired timing sample for {case_id}/{mode}")
                    if sample["control_output_hash"] != sample["candidate_output_hash"]:
                        raise ValueError(f"kernel outputs differ in timed repetition for {case_id}/{mode}")
                    per_case_raw[case_id][mode]["control"].append(int(sample["control_ns"]))
                    per_case_raw[case_id][mode]["candidate"].append(int(sample["candidate_ns"]))

    for fixture_label in FIXTURE_LABELS:
        groups = [case_id for case_id, case in case_by_id.items()
                  if ("update2_window0" in str(case["fixture"]) and fixture_label == "A_update2_window0") or
                     ("update3_window1" in str(case["fixture"]) and fixture_label == "B_update3_window1")]
        if len(groups) != 2:
            raise ValueError(f"expected two Step-A groups for fixture {fixture_label}, found {groups}")
        per_fixture[fixture_label] = {}
        for mode in MODES:
            control = [value for case_id in groups for value in per_case_raw[case_id][mode]["control"]]
            candidate = [value for case_id in groups for value in per_case_raw[case_id][mode]["candidate"]]
            paired_ratios = np.asarray([c / b for c, b in zip(candidate, control)], dtype=np.float64)
            per_fixture[fixture_label][mode] = {
                "preparation_included": mode == "full_cost",
                "diagnostic_only": mode == "accumulation_only",
                "excludes_operand_preparation": mode == "accumulation_only",
                "groups": groups,
                "control_raw_ns": control,
                "candidate_raw_ns": candidate,
                "control": _statistics(control),
                "candidate": _statistics(candidate),
                "ratio_candidate_over_control": float(sum(candidate) / sum(control)),
                "mean_paired_ratio": float(np.mean(paired_ratios)),
                "median_paired_ratio": float(np.median(paired_ratios)),
                "stddev_paired_ratio_population": float(np.std(paired_ratios, ddof=0)),
            }
            if len(control) != 180 or len(candidate) != 180:
                raise ValueError(f"expected 180 paired samples per fixture/mode, got {len(control)}")

    return {
        "schema": "omega-fc2-dweight-local-cost-aggregate-v1",
        "status": "LOCAL_COST_DIAGNOSTIC_PASS",
        "diagnostic_only": True,
        "process_count": 3,
        "worker_threads": 1,
        "pairs_per_group_per_modality_per_process": {"warmup": 5, "measured": 30},
        "process_binary": dict(exe_identity),
        "primary_definition": "direct elapsed time for GELU preparation + required A/D staging + current per-slot control or candidate per-slot staging/diagnostics + final dW accumulation/write",
        "secondary_definition": "diagnostic_only; excludes_operand_preparation; direct control/candidate accumulation on captured A/D/dW_old; not used as reconstructed full cost",
        "restoration": "dW_old memcpy before every individual timed evaluation; restoration excluded from both timers",
        "fixtures": per_fixture,
        "process_reports": [process["report_path"] for process in process_reports],
    }


def run_diagnostic(executable: Path, output_root: Path) -> dict[str, Any]:
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"output parent must exist: {output_root.parent}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite local diagnostic output: {output_root}")
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_root.mkdir()
    input_root = output_root / "cases"
    input_root.mkdir()
    cases, fixtures = _load_certified_cases(input_root)
    executable_identity = {"path": executable.resolve().as_posix(), "sha256": _file_sha256(executable)}
    source_paths = {
        "omega_recurrent.cpp": HERE / "native" / "omega_recurrent.cpp",
        "fc2_dweight_m8.h": HERE / "native" / "fc2_dweight_m8.h",
        "local_cost_diagnostic.cpp": HERE / "native" / "fc2_dweight_local_cost_diagnostic.cpp",
        "runner.py": Path(__file__).resolve(),
    }
    report: dict[str, Any] = {
        "schema": "omega-fc2-dweight-local-cost-run-v1",
        "status": "RUNNING",
        "diagnostic_only": True,
        "timing_performed": True,
        "torch_loaded": False,
        "executable": executable_identity,
        "compile_contract": "MSVC Release /O2 /fp:precise /arch:AVX2; one C++ executable invokes both actual helper paths",
        "source_sha256": {name: _file_sha256(path) for name, path in source_paths.items()},
        "step_a_manifest_sha256": _file_sha256(CAPTURE_MANIFEST),
        "fixtures": fixtures,
        "cases": cases,
        "protocol": {
            "fresh_processes": 3,
            "compute_threads": 1,
            "warmup_pairs_per_group_per_modality_per_process": 5,
            "measured_pairs_per_group_per_modality_per_process": 30,
            "alternating_pair_order": ["control_then_candidate", "candidate_then_control"],
            "dweight_restoration": "copy captured nonzero dW_old before each evaluation; outside timed interval for both routes",
            "timer": "steady_clock around local function only; output validation/hash outside timer",
            "primary_preparation": "GELU and candidate A/D staging are included in candidate time; control computes GELU but does not stage A/D",
            "secondary": "separable using extracted actual helpers; tagged diagnostic_only and excludes_operand_preparation",
            "secondary_bookkeeping": "retains route-specific per-element null diagnostics calls; operand preparation excluded",
            "no_pytorch_teacher_optimizer_pool_or_oracle": True,
        },
        "processes": [],
    }
    run_manifest_path = output_root / "diagnostic_manifest.json"
    _write_json(run_manifest_path, report)

    case_paths = [input_root / f"{case['case_id']}.bin" for case in cases]
    for process_index in range(3):
        command = [str(executable.resolve()), str(process_index), *(str(path) for path in case_paths)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        stdout_path = output_root / f"process{process_index}_stdout.json"
        stderr_path = output_root / f"process{process_index}_stderr.txt"
        status_path = output_root / f"process{process_index}_exit.json"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        try:
            process = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            process = {}
            report["status"] = "BLOCKED"
            report["failure"] = f"invalid process {process_index} JSON: {exc}"
        status = {
            "process_index": process_index,
            "exit_code": int(completed.returncode),
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "stdout_bytes": len(completed.stdout.encode("utf-8")),
            "stderr_bytes": len(completed.stderr.encode("utf-8")),
            "process_status": process.get("status"),
            "report_path": stdout_path.as_posix(),
            "status": "PASS" if completed.returncode == 0 and process.get("status") == "LOCAL_COST_PROCESS_PASS" and not completed.stderr else "FAIL",
        }
        _write_json(status_path, status)
        report["processes"].append(status)
        _write_json(run_manifest_path, report)
        if status["status"] != "PASS":
            report["status"] = "BLOCKED"
            report["failure"] = status
            _write_json(run_manifest_path, report)
            raise RuntimeError(f"local diagnostic process {process_index} failed; see {status_path}")

    process_reports = [json.loads(Path(item["report_path"]).read_text(encoding="utf-8")) for item in report["processes"]]
    per_fixture_aggregate = _aggregate(process_reports, cases, executable_identity)
    aggregate_path = output_root / "aggregate_report.json"
    _write_json(aggregate_path, per_fixture_aggregate)
    report["aggregate_path"] = aggregate_path.as_posix()
    report["aggregate_status"] = per_fixture_aggregate["status"]
    report["status"] = "LOCAL_COST_DIAGNOSTIC_PASS"
    _write_json(run_manifest_path, report)
    return {"manifest": report, "aggregate": per_fixture_aggregate}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_diagnostic(args.exe.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
