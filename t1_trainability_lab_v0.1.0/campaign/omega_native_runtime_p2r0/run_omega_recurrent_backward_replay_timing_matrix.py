"""Launch the approved six-process matrix with durable child I/O capture."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TIMING_WORKER = HERE / "run_omega_recurrent_backward_replay_timing.py"
FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
DEFAULT_RUN_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "timing_clean_capture_2"
DEFAULT_DLL = Path(
    r"C:\Users\danil\AppData\Local\Temp\opencode\omega-backward-diagnostics-fastpath\python\omega_recurrent.dll"
)
PAIR_ROUTES = {
    1: ("pytorch", "native"),
    2: ("native", "pytorch"),
    3: ("pytorch", "native"),
}
FIXTURE_NAMES = (
    "replay_K4_update2_window0.bin",
    "replay_K4_update3_window1.bin",
)
ENV_VARS_REQUIRED_UNSET = (
    "KMP_BLOCKTIME",
    "KMP_LIBRARY",
    "OMP_WAIT_POLICY",
    "KMP_SETTINGS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _process_plan() -> list[dict[str, Any]]:
    return [
        {"pair_id": pair_id, "position": position, "route": route}
        for pair_id, routes in PAIR_ROUTES.items()
        for position, route in enumerate(routes, start=1)
    ]


def run_matrix(run_root: Path, fixture_root: Path, dll_path: Path, timeout_seconds: int) -> dict[str, Any]:
    parent = run_root.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"run-root parent must already exist: {parent}")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite timing matrix run directory: {run_root}")
    if not TIMING_WORKER.is_file():
        raise FileNotFoundError(TIMING_WORKER)
    fixture_paths = [fixture_root / name for name in FIXTURE_NAMES]
    missing = [path for path in fixture_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"verified fixture files missing: {missing}")
    for name in ENV_VARS_REQUIRED_UNSET:
        if os.environ.get(name) is not None:
            raise RuntimeError(f"canonical timing environment requires {name} to remain unset")

    run_root.mkdir(parents=False)
    raw_root = run_root / "raw_process_io"
    raw_root.mkdir()
    process_results: list[dict[str, Any]] = []
    plan = _process_plan()
    manifest: dict[str, Any] = {
        "schema": "omega-recurrent-backward-replay-timing-matrix-capture-v1",
        "status": "RUNNING",
        "run_root": run_root.as_posix(),
        "process_order": plan,
        "requested_backward_calls": {"total": 60, "warmup": 12, "measured": 48},
        "child_environment": {name: None for name in ENV_VARS_REQUIRED_UNSET},
        "raw_process_io_directory": raw_root.as_posix(),
        "process_results": process_results,
    }
    manifest_path = run_root / "matrix_manifest.json"
    _write_json(manifest_path, manifest)

    for item in plan:
        pair_id = int(item["pair_id"])
        position = int(item["position"])
        route = str(item["route"])
        basename = f"{position:02d}_{route}"
        pair_dir = run_root / f"pair{pair_id}"
        pair_dir.mkdir(exist_ok=True)
        raw_pair_dir = raw_root / f"pair{pair_id}"
        raw_pair_dir.mkdir(exist_ok=True)

        report_path = pair_dir / f"{basename}.json"
        stdout_path = raw_pair_dir / f"{basename}.stdout.txt"
        stderr_path = raw_pair_dir / f"{basename}.stderr.txt"
        status_path = raw_pair_dir / f"{basename}.exit.json"
        argv = [
            sys.executable,
            "-B",
            str(TIMING_WORKER),
            "--mode",
            "backward-timing",
            "--route",
            route,
            "--pair-id",
            str(pair_id),
            "--position",
            str(position),
            "--fixture",
            str(fixture_paths[0]),
            "--fixture",
            str(fixture_paths[1]),
            "--output",
            str(report_path),
        ]
        if route == "native":
            argv.extend(("--dll", str(dll_path)))

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
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)

        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        worker_report_exists = report_path.is_file()
        stdout_report_valid = False
        stdout_report_status: str | None = None
        if stdout.strip():
            try:
                parsed_stdout = json.loads(stdout)
                stdout_report_status = str(parsed_stdout.get("status"))
                stdout_report_valid = stdout_report_status == "TIMING_PROCESS_PASS"
            except json.JSONDecodeError:
                stdout_report_valid = False

        process_status = {
            "pair_id": pair_id,
            "position": position,
            "route": route,
            "argv": argv,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "stdout_json_status": stdout_report_status,
            "stdout_json_valid": stdout_report_valid,
            "worker_report_path": report_path.as_posix(),
            "worker_report_exists": worker_report_exists,
        }
        process_status["exit_record_path"] = status_path.as_posix()
        process_status["status"] = (
            "PASS"
            if exit_code == 0 and not timed_out and stdout_report_valid and worker_report_exists
            else "FAIL"
        )
        _write_json(status_path, process_status)
        process_results.append(process_status)
        manifest["status"] = "RUNNING" if process_status["status"] == "PASS" else "BLOCKED"
        _write_json(manifest_path, manifest)

        if process_status["status"] != "PASS":
            manifest["failure"] = process_status
            manifest["stop_reason"] = "child process exit/stdout/stderr/report did not satisfy capture contract"
            _write_json(manifest_path, manifest)
            raise RuntimeError(
                f"timing child blocked at pair{pair_id}/{basename}; see {status_path} and captured stdout/stderr"
            )

    timing_module_name = "run_omega_recurrent_backward_replay_timing"
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    timing_module = importlib.import_module(timing_module_name)
    ordered_reports = [Path(item["worker_report_path"]) for item in process_results]
    aggregate_path = run_root / "aggregate_report.json"
    aggregate = timing_module.aggregate_reports(ordered_reports, aggregate_path)
    manifest["status"] = "TIMING_MATRIX_PASS"
    manifest["aggregate_report"] = aggregate_path.as_posix()
    manifest["aggregate_status"] = aggregate["status"]
    _write_json(manifest_path, manifest)
    return {"matrix_manifest": manifest, "aggregate": aggregate}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    result = run_matrix(args.run_root.resolve(), args.fixture_root.resolve(), args.dll.resolve(), args.timeout_seconds)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
