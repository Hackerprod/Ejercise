"""Execute native replay markers inside an externally managed WPR session."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
TIMING_SCRIPT = HERE / "run_omega_recurrent_backward_replay_timing.py"
FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
ENV_VARS_REQUIRED_UNSET = (
    "KMP_BLOCKTIME",
    "KMP_LIBRARY",
    "OMP_WAIT_POLICY",
    "KMP_SETTINGS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
FIXTURE_FILENAMES = (
    "replay_K4_update2_window0.bin",
    "replay_K4_update3_window1.bin",
)
FIXTURE_LABELS = ("A_update2_window0", "B_update3_window1")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_environment() -> dict[str, Any]:
    inherited = {name: os.environ.get(name) for name in ENV_VARS_REQUIRED_UNSET}
    set_values = {name: value for name, value in inherited.items() if value is not None}
    if set_values:
        raise RuntimeError(f"native WPT process requires canonical inherited-unset env: {set_values}")
    return {"inherited_environment": inherited, "native_runtime_workers": 4, "torch_loaded": False}


def _load_timing_helpers() -> tuple[Any, Any]:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import run_omega_recurrent_backward_replay as replay
    import run_omega_recurrent_backward_replay_timing as timing
    return replay, timing


def _fixture_cases(replay: Any, timing: Any, fixture_root: Path) -> tuple[Any, list[dict[str, Any]]]:
    paths = [fixture_root / filename for filename in FIXTURE_FILENAMES]
    _replay_module, fixtures = timing._load_two_fixtures(paths)
    return replay, fixtures


def _qpc_frequency() -> int:
    frequency = ctypes.c_longlong()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    kernel32.QueryPerformanceFrequency.restype = ctypes.c_int
    if not kernel32.QueryPerformanceFrequency(ctypes.byref(frequency)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(frequency.value)


def _qpc_counter() -> int:
    counter = ctypes.c_longlong()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    kernel32.QueryPerformanceCounter.restype = ctypes.c_int
    if not kernel32.QueryPerformanceCounter(ctypes.byref(counter)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counter.value)


def run_native_wpt(
    *,
    fixture_root: Path,
    dll_path: Path,
    expected_dll_sha256: str,
    build_identity_path: Path,
    marker_output: Path,
) -> dict[str, Any]:
    environment = _assert_environment()
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("native WPT process must not load PyTorch")
    replay, timing = _load_timing_helpers()
    _replay, fixtures = _fixture_cases(replay, timing, fixture_root)

    resolved_dll = dll_path.resolve()
    dll_sha256 = _sha256_file(resolved_dll)
    if dll_sha256.lower() != expected_dll_sha256.lower():
        raise ValueError(f"diagnostic DLL hash mismatch: {dll_sha256}")
    build_identity = json.loads(build_identity_path.read_text(encoding="utf-8"))
    if str(build_identity.get("dll_sha256", "")).lower() != dll_sha256.lower():
        raise ValueError("diagnostic build manifest DLL SHA does not match loaded DLL")

    timing.ACCEPTED_DLL_SHA256 = dll_sha256
    library, runtime, observed_sha = timing._configure_native_library(replay, resolved_dll)
    if observed_sha.lower() != dll_sha256.lower():
        library.omega_runtime_destroy(runtime)
        raise RuntimeError("configured DLL identity drift")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        library.omega_runtime_destroy(runtime)
        raise RuntimeError("native WPT process loaded PyTorch unexpectedly")

    cases = [timing._make_native_case(replay, library, runtime, fixture) for fixture in fixtures]
    pid = os.getpid()
    main_tid = int(threading.get_native_id())
    frequency = _qpc_frequency()
    qpc_anchor = _qpc_counter()
    perf_anchor_ns = time.perf_counter_ns()
    utc_anchor_ns = time.time_ns()
    marker_events: list[dict[str, Any]] = []
    replays: list[dict[str, Any]] = []

    def marker_record(
        event_name: str,
        case: Mapping[str, Any],
        replay_number: int,
        phase: str,
        utc_ns: int,
        perf_ns: int,
        elapsed_ns: int | None = None,
    ) -> dict[str, Any]:
        qpc_estimate = qpc_anchor + int((perf_ns - perf_anchor_ns) * frequency / 1_000_000_000)
        record: dict[str, Any] = {
            "event": event_name,
            "fixture": case["label"],
            "replay": replay_number,
            "phase": phase,
            "pid": pid,
            "tid": main_tid,
            "utc_ns": utc_ns,
            "utc_iso": datetime.fromtimestamp(utc_ns / 1_000_000_000, tz=timezone.utc).isoformat(),
            "perf_counter_ns": perf_ns,
            "qpc_estimate_from_perf_counter_anchor": qpc_estimate,
            "utc_minus_perf_counter_ns": utc_ns - perf_ns,
        }
        if elapsed_ns is not None:
            record["clean_backward_call_elapsed_ns"] = elapsed_ns
        marker_events.append(record)
        return record

    def forward(case: Mapping[str, Any], replay_number: int, phase: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
        args = (
            runtime,
            case["config_ptr"],
            case["params_ptr"],
            case["token_ptr"],
            case["previous_ptr"],
            case["next_ptr"],
            case["readout_ptr"],
            case["workspace_ptr"],
            case["workspace_bytes"],
        )
        begin_utc = time.time_ns()
        begin_perf = time.perf_counter_ns()
        status = int(library.omega_runtime_forward(*args))
        end_perf = time.perf_counter_ns()
        end_utc = time.time_ns()
        if status != 0:
            raise RuntimeError(f"native forward failed outside backward range: status={status}")
        begin = marker_record(
            f"REPLAY_FORWARD_BEGIN|fixture={case['label']}|replay={replay_number}|phase={phase}|pid={pid}",
            case,
            replay_number,
            phase,
            begin_utc,
            begin_perf,
        )
        end = marker_record(
            f"REPLAY_FORWARD_END|fixture={case['label']}|replay={replay_number}|phase={phase}|pid={pid}",
            case,
            replay_number,
            phase,
            end_utc,
            end_perf,
            end_perf - begin_perf,
        )
        return status, begin, end

    def backward(case: Mapping[str, Any], replay_number: int, phase: str) -> dict[str, Any]:
        args = (
            runtime,
            case["config_ptr"],
            case["params_ptr"],
            case["token_ptr"],
            case["upstream_readout_ptr"],
            case["upstream_next_ptr"],
            case["workspace_ptr"],
            case["workspace_bytes"],
            case["grads_ptr"],
        )
        begin_utc = time.time_ns()
        begin_perf = time.perf_counter_ns()
        # This is exactly the clean benchmark timer's C-ABI call body. Do not
        # move checks, hashes, logging, or destination allocation inside it.
        status = int(library.omega_runtime_backward(*args))
        end_perf = time.perf_counter_ns()
        end_utc = time.time_ns()
        if status != 0:
            raise RuntimeError(f"native backward failed: status={status}")
        elapsed = end_perf - begin_perf
        begin = marker_record(
            f"REPLAY_BACKWARD_BEGIN|fixture={case['label']}|replay={replay_number}|phase={phase}|pid={pid}|tid={main_tid}",
            case,
            replay_number,
            phase,
            begin_utc,
            begin_perf,
        )
        end = marker_record(
            f"REPLAY_BACKWARD_END|fixture={case['label']}|replay={replay_number}|phase={phase}|pid={pid}|tid={main_tid}",
            case,
            replay_number,
            phase,
            end_utc,
            end_perf,
            elapsed,
        )
        return {
            "fixture": case["label"],
            "replay": replay_number,
            "phase": phase,
            "pid": pid,
            "tid": main_tid,
            "status": status,
            "elapsed_ns_clean_api_interval": elapsed,
            "begin_marker": begin,
            "end_marker": end,
        }

    replay_schedule = [
        (cases[0], 1, "warmup"),
        (cases[1], 2, "warmup"),
        (cases[0], 3, "analyzed"),
        (cases[1], 4, "analyzed"),
        (cases[0], 5, "analyzed"),
        (cases[1], 6, "analyzed"),
    ]
    try:
        for case, replay_number, phase in replay_schedule:
            _, forward_begin, forward_end = forward(case, replay_number, phase)
            backward_result = backward(case, replay_number, phase)
            replays.append({
                "fixture": case["label"],
                "replay": replay_number,
                "phase": phase,
                "forward_begin": forward_begin,
                "forward_end": forward_end,
                "backward": backward_result,
            })
    finally:
        library.omega_runtime_destroy(runtime)

    report = {
        "schema": "omega-recurrent-backward-replay-native-wpt-markers-v1",
        "status": "NATIVE_WPT_MARKER_RUN_PASS",
        "process_fresh": True,
        "route": "native",
        "pid": pid,
        "main_tid": main_tid,
        "torch_loaded": False,
        "runtime": {"workers": 4, "instrumentation": 0, "one_persistent_pool_per_process": True},
        "build_identity": build_identity,
        "loaded_dll": {
            "path": resolved_dll.as_posix(),
            "sha256": observed_sha,
            "pdb_path": build_identity.get("pdb_path"),
            "pdb_sha256": build_identity.get("pdb_sha256"),
        },
        "canonical_environment": environment,
        "timebase": {
            "perf_counter": "time.perf_counter_ns (Windows QPC-backed)",
            "qpc_frequency_hz": frequency,
            "qpc_anchor": qpc_anchor,
            "perf_counter_anchor_ns": perf_anchor_ns,
            "utc_anchor_ns": utc_anchor_ns,
            "correlation": "Per-marker UTC/perf_counter pairs; compare marker UTC to ETL start UTC and verify QPC/perf deltas before selecting ETL sample intervals.",
        },
        "schedule": {
            "warmups": ["A_update2_window0", "B_update3_window1"],
            "analyzed": ["A_update2_window0", "B_update3_window1", "A_update2_window0", "B_update3_window1"],
            "forward_calls": 6,
            "backward_calls": 6,
            "analyzed_backward_calls": 4,
        },
        "fixtures": [
            {
                "label": case["label"],
                "filename": Path(case["path"]).name,
                "sha256": case["sha256"],
                "update": int(case["header"]["metadata"]["update_index"]),
                "window": int(case["header"]["metadata"]["window_index"]),
            }
            for case in cases
        ],
        "replays": replays,
        "markers": marker_events,
        "wpr_etl_managed_by": "separate elevated PowerShell start/wait/stop helper",
    }
    _write_json(marker_output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--expected-dll-sha256", required=True)
    parser.add_argument("--build-identity", type=Path, required=True)
    parser.add_argument("--marker-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.marker_output.exists():
        raise FileExistsError(f"refusing to overwrite WPT marker report: {args.marker_output}")
    report = run_native_wpt(
        fixture_root=args.fixtures.resolve(),
        dll_path=args.dll.resolve(),
        expected_dll_sha256=args.expected_dll_sha256,
        build_identity_path=args.build_identity.resolve(),
        marker_output=args.marker_output.resolve(),
    )
    print(json.dumps({
        "status": report["status"],
        "pid": report["pid"],
        "main_tid": report["main_tid"],
        "dll_sha256": report["loaded_dll"]["sha256"],
        "backward_calls": report["schedule"]["backward_calls"],
        "analyzed_backward_calls": report["schedule"]["analyzed_backward_calls"],
        "markers": report["markers"],
        "marker_output": args.marker_output.as_posix(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
