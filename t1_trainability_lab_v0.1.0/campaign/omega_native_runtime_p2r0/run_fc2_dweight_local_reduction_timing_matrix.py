"""Capture a clean six-process baseline/candidate backward timing matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import run_omega_recurrent_backward_replay_timing as timing


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKER = HERE / "run_fc2_dweight_local_reduction_timing.py"
FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
FIXTURE_NAMES = (
    "replay_K4_update2_window0.bin",
    "replay_K4_update3_window1.bin",
)
FIXTURE_LABELS = ("A_update2_window0", "B_update3_window1")
BASELINE_DLL_SHA256 = "6f38e3b1dcada32c98aef50e870e201510bab9c75b875438823915ba53b854d5"
VARIANT_ROUTES = {
    1: ("baseline", "candidate"),
    2: ("candidate", "baseline"),
    3: ("baseline", "candidate"),
}
ENV_VARS_REQUIRED_UNSET = timing.ENV_VARS_REQUIRED_UNSET
DEFAULT_RUN_ROOT = HERE / "results" / "fc2_dweight_local_reduction_step_b" / "timing_clean"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _process_plan() -> list[dict[str, Any]]:
    return [
        {"pair_id": pair_id, "position": position, "variant": variant}
        for pair_id, variants in VARIANT_ROUTES.items()
        for position, variant in enumerate(variants, start=1)
    ]


def _aggregate(run_root: Path, process_results: Sequence[Mapping[str, Any]], dlls: Mapping[str, Mapping[str, Any]], fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_order = [
        (pair_id, position, variant)
        for pair_id, variants in VARIANT_ROUTES.items()
        for position, variant in enumerate(variants, start=1)
    ]
    if len(process_results) != len(expected_order):
        raise ValueError("aggregate requires six process records")
    reports: list[dict[str, Any]] = []
    for status, expected in zip(process_results, expected_order):
        observed = (int(status["pair_id"]), int(status["position"]), str(status["variant"]))
        if observed != expected or status.get("status") != "PASS":
            raise ValueError(f"process plan/status mismatch: {observed} expected {expected}")
        report_path = Path(str(status["worker_report_path"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "TIMING_PROCESS_PASS" or report.get("timing_performed") is not True or
                report.get("route") != expected[2] or report.get("variant") != expected[2] or
                report.get("torch_loaded") is not False or
                report.get("native_library_sha256") != dlls[expected[2]]["sha256"] or
                report.get("expected_dll_sha256") != dlls[expected[2]]["sha256"]):
            raise ValueError(f"timing report identity/status mismatch: {report_path}")
        if int(report.get("warmup_backward_calls", -1)) != 2 or int(report.get("measured_backward_calls", -1)) != 8:
            raise ValueError(f"warmup/measured call count mismatch: {report_path}")
        expected_samples = [FIXTURE_LABELS[index % 2] for index in range(8)]
        samples = report.get("measured_samples", [])
        if len(samples) != 8 or [sample.get("fixture") for sample in samples] != expected_samples:
            raise ValueError(f"measured A/B schedule mismatch: {report_path}")
        if any(int(sample.get("elapsed_ns", 0)) <= 0 for sample in samples):
            raise ValueError(f"non-positive sample duration: {report_path}")
        reports.append(report)

    canonical_environment = reports[0]["environment"]
    if any(report["environment"] != canonical_environment for report in reports[1:]):
        raise ValueError("child processes did not share canonical inherited environment")

    variant_samples: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    for report in reports:
        variant_samples[str(report["variant"])].extend(report["measured_samples"])
    totals: dict[str, Any] = {}
    for variant, samples in variant_samples.items():
        by_fixture = {
            label: [int(sample["elapsed_ns"]) for sample in samples if sample["fixture"] == label]
            for label in FIXTURE_LABELS
        }
        if {label: len(values) for label, values in by_fixture.items()} != {label: 12 for label in FIXTURE_LABELS}:
            raise ValueError(f"measured sample balance mismatch: {variant}")
        totals[variant] = {
            "measured_backward_calls": len(samples),
            "sample_count_by_fixture": {label: len(values) for label, values in by_fixture.items()},
            "sum_ns_by_fixture": {label: sum(values) for label, values in by_fixture.items()},
            "median_ns_by_fixture": {label: float(np.median(np.asarray(values, dtype=np.float64))) for label, values in by_fixture.items()},
            "sum_ns_all_fixtures": sum(int(sample["elapsed_ns"]) for sample in samples),
            "mean_ns_all_fixtures": float(np.mean(np.asarray([int(sample["elapsed_ns"]) for sample in samples], dtype=np.float64))),
        }

    ratio_by_fixture = {
        label: totals["candidate"]["sum_ns_by_fixture"][label] / totals["baseline"]["sum_ns_by_fixture"][label]
        for label in FIXTURE_LABELS
    }
    pair_ratios: dict[str, Any] = {}
    for pair_id in (1, 2, 3):
        matching = [report for report in reports if int(report["pair_id"]) == pair_id]
        by_variant = {
            variant: sum(int(sample["elapsed_ns"]) for report in matching if report["variant"] == variant
                         for sample in report["measured_samples"])
            for variant in ("baseline", "candidate")
        }
        pair_ratios[str(pair_id)] = {
            "baseline_sum_ns": by_variant["baseline"],
            "candidate_sum_ns": by_variant["candidate"],
            "R_candidate_over_baseline": by_variant["candidate"] / by_variant["baseline"],
        }

    return {
        "schema": "omega-fc2-dweight-local-reduction-timing-aggregate-v1",
        "status": "TIMING_AGGREGATE_PASS",
        "timing_performed": True,
        "torch_loaded": False,
        "measurement": "only omega_runtime_backward C ABI call; forward/setup/checking outside timer",
        "process_order": [
            {"pair_id": pair_id, "position": position, "variant": variant}
            for pair_id, position, variant in expected_order
        ],
        "process_count": 6,
        "backward_calls_total": 60,
        "warmup_backward_calls_total": 12,
        "measured_backward_calls_total": 48,
        "measured_calls_per_variant": 24,
        "measured_calls_per_fixture_per_variant": 12,
        "dW_variants": dict(dlls),
        "fixtures": list(fixtures),
        "canonical_environment": canonical_environment,
        "totals": totals,
        "R_candidate_over_baseline": totals["candidate"]["sum_ns_all_fixtures"] / totals["baseline"]["sum_ns_all_fixtures"],
        "R_candidate_over_baseline_by_fixture": ratio_by_fixture,
        "R_candidate_over_baseline_by_pair": pair_ratios,
        "raw_process_reports": [str(status["worker_report_path"]) for status in process_results],
    }


def run_matrix(
    run_root: Path,
    baseline_dll: Path,
    candidate_dll: Path,
    candidate_sha256: str,
    timeout_seconds: int,
    baseline_sha256: str = BASELINE_DLL_SHA256,
) -> dict[str, Any]:
    if not run_root.parent.is_dir():
        raise FileNotFoundError(f"run-root parent must exist: {run_root.parent}")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite timing matrix: {run_root}")
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    for name in ENV_VARS_REQUIRED_UNSET:
        if os.environ.get(name) is not None:
            raise RuntimeError(f"canonical timing environment requires {name} to remain unset")
    for path in (*[FIXTURE_ROOT / name for name in FIXTURE_NAMES], baseline_dll, candidate_dll):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_sha256_actual = _sha256(baseline_dll.resolve())
    candidate_sha256_actual = _sha256(candidate_dll.resolve())
    if baseline_sha256_actual != baseline_sha256.lower():
        raise ValueError(f"baseline DLL identity mismatch: {baseline_sha256_actual}")
    if candidate_sha256_actual != candidate_sha256.lower():
        raise ValueError(f"candidate DLL identity mismatch: {candidate_sha256_actual}")

    replay, loaded_fixtures = timing._load_two_fixtures([FIXTURE_ROOT / name for name in FIXTURE_NAMES])
    del replay
    fixtures = [
        {"label": item["label"], "filename": Path(item["path"]).name, "sha256": item["sha256"]}
        for item in loaded_fixtures
    ]
    dlls = {
        "baseline": {"path": baseline_dll.resolve().as_posix(), "sha256": baseline_sha256_actual},
        "candidate": {"path": candidate_dll.resolve().as_posix(), "sha256": candidate_sha256_actual},
    }

    run_root.mkdir(parents=False)
    raw_root = run_root / "raw_process_io"
    raw_root.mkdir()
    manifest_path = run_root / "matrix_manifest.json"
    process_results: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema": "omega-fc2-dweight-local-reduction-timing-matrix-v1",
        "status": "RUNNING",
        "run_root": run_root.as_posix(),
        "candidate_dll_sha256": candidate_sha256_actual,
        "baseline_dll_sha256": baseline_sha256,
        "fixtures": fixtures,
        "process_order": _process_plan(),
        "protocol": {
            "fresh_processes": 6,
            "pairs": 3,
            "pair_orders": {"1": ["baseline", "candidate"], "2": ["candidate", "baseline"], "3": ["baseline", "candidate"]},
            "per_process_warmup_backward_calls": 2,
            "per_process_measured_backward_calls": 8,
            "warmup_fixture_sequence": ["A_update2_window0", "B_update3_window1"],
            "measured_fixture_sequence": ["A_update2_window0", "B_update3_window1"] * 4,
            "timer_scope": "omega_runtime_backward C ABI call only",
            "torch": "not imported in native workers",
        },
        "dlls": dlls,
        "child_environment": {name: None for name in ENV_VARS_REQUIRED_UNSET},
        "raw_process_io_directory": raw_root.as_posix(),
        "process_results": process_results,
    }
    _write_json(manifest_path, manifest)

    for item in _process_plan():
        pair_id = int(item["pair_id"])
        position = int(item["position"])
        variant = str(item["variant"])
        dll_path = baseline_dll if variant == "baseline" else candidate_dll
        expected_sha = dlls[variant]["sha256"]
        basename = f"{position:02d}_{variant}"
        pair_dir = run_root / f"pair{pair_id}"
        pair_dir.mkdir(exist_ok=True)
        raw_pair_dir = raw_root / f"pair{pair_id}"
        raw_pair_dir.mkdir(exist_ok=True)
        report_path = pair_dir / f"{basename}.json"
        stdout_path = raw_pair_dir / f"{basename}.stdout.txt"
        stderr_path = raw_pair_dir / f"{basename}.stderr.txt"
        exit_path = raw_pair_dir / f"{basename}.exit.json"
        argv = [
            sys.executable, "-B", str(WORKER),
            "--variant", variant,
            "--pair-id", str(pair_id),
            "--position", str(position),
            "--dll", str(dll_path.resolve()),
            "--expected-dll-sha256", expected_sha,
            "--fixture", str(FIXTURE_ROOT / FIXTURE_NAMES[0]),
            "--fixture", str(FIXTURE_ROOT / FIXTURE_NAMES[1]),
            "--output", str(report_path),
        ]
        child_environment = os.environ.copy()
        for name in ENV_VARS_REQUIRED_UNSET:
            child_environment.pop(name, None)
        child_environment["PYTHONUNBUFFERED"] = "1"

        stdout = ""
        stderr = ""
        exit_code: int | None = None
        timed_out = False
        try:
            completed = subprocess.run(
                argv,
                cwd=REPO_ROOT,
                env=child_environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        stdout_report: dict[str, Any] = {}
        try:
            stdout_report = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError:
            stdout_report = {}
        worker_report_exists = report_path.is_file()
        valid_stdout = (
            stdout_report.get("status") == "TIMING_PROCESS_PASS" and
            stdout_report.get("variant") == variant and
            stdout_report.get("native_library_sha256") == expected_sha and
            stdout_report.get("torch_loaded") is False
        )
        process_status = {
            "pair_id": pair_id,
            "position": position,
            "variant": variant,
            "dll_sha256": expected_sha,
            "argv": argv,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "stdout_json_valid": valid_stdout,
            "worker_report_path": report_path.as_posix(),
            "worker_report_exists": worker_report_exists,
            "exit_record_path": exit_path.as_posix(),
            "status": "PASS" if exit_code == 0 and not timed_out and valid_stdout and worker_report_exists else "FAIL",
        }
        _write_json(exit_path, process_status)
        process_results.append(process_status)
        manifest["status"] = "RUNNING" if process_status["status"] == "PASS" else "BLOCKED"
        _write_json(manifest_path, manifest)
        if process_status["status"] != "PASS":
            manifest["failure"] = process_status
            manifest["stop_reason"] = "child process identity, exit/stdout/stderr/report failed contract"
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"timing blocked at pair{pair_id}/{basename}; see {exit_path}")

    aggregate = _aggregate(run_root, process_results, dlls, fixtures)
    aggregate_path = run_root / "aggregate_report.json"
    _write_json(aggregate_path, aggregate)
    manifest["status"] = "TIMING_MATRIX_PASS"
    manifest["aggregate_report"] = aggregate_path.as_posix()
    manifest["aggregate_status"] = aggregate["status"]
    _write_json(manifest_path, manifest)
    return {"matrix_manifest": manifest, "aggregate": aggregate}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--baseline-dll", type=Path, default=Path(
        r"C:\Users\danil\AppData\Local\Temp\opencode\omega-backward-diagnostics-fastpath\python\omega_recurrent.dll"
    ))
    parser.add_argument("--candidate-dll", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--baseline-sha256", default=BASELINE_DLL_SHA256)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = run_matrix(
        args.run_root.resolve(), args.baseline_dll.resolve(), args.candidate_dll.resolve(),
        args.candidate_sha256, args.timeout_seconds, args.baseline_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
