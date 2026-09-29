"""Run fresh-process backward-only timing for certified replay fixtures.

Each invocation is one fresh process and measures exactly eight backwards after
two unmeasured warmups. Forward construction, fixture validation, environment
checks, and report writing stay outside measured intervals.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPLAY_SCRIPT = HERE / "run_omega_recurrent_backward_replay.py"
DEFAULT_FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
DEFAULT_DLL = Path(
    r"C:\Users\danil\AppData\Local\Temp\opencode\omega-backward-diagnostics-fastpath\python\omega_recurrent.dll"
)

ACCEPTED_DLL_SHA256 = "6f38e3b1dcada32c98aef50e870e201510bab9c75b875438823915ba53b854d5"
FIXTURE_FILENAMES = (
    "replay_K4_update2_window0.bin",
    "replay_K4_update3_window1.bin",
)
FIXTURE_LABELS = ("A_update2_window0", "B_update3_window1")
PAIR_ROUTES = {
    1: ("pytorch", "native"),
    2: ("native", "pytorch"),
    3: ("pytorch", "native"),
}
ENV_VARS_REQUIRED_UNSET = (
    "KMP_BLOCKTIME",
    "KMP_LIBRARY",
    "OMP_WAIT_POLICY",
    "KMP_SETTINGS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
EFFECTIVE_KMP_DEFAULTS = {
    "KMP_BLOCKTIME": "200ms",
    "KMP_LIBRARY": "throughput",
    "OMP_WAIT_POLICY": "PASSIVE",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _assert_canonical_process_environment() -> dict[str, Any]:
    observed = {name: os.environ.get(name) for name in ENV_VARS_REQUIRED_UNSET}
    set_values = {name: value for name, value in observed.items() if value is not None}
    if set_values:
        raise RuntimeError(f"canonical inherited KMP/thread environment must remain unset: {set_values}")
    return {
        "inherited_environment": observed,
        "effective_libiomp_defaults_from_separate_KMP_SETTINGS_probe": dict(EFFECTIVE_KMP_DEFAULTS),
        "probe_thread_policy": {"torch_intraop": 4, "torch_interop": 1},
        "KMP_SETTINGS_during_timing": "unset",
        "probe_note": "Separate preflight process used KMP_SETTINGS=TRUE; timing processes retain canonical inherited-unset environment.",
    }


def _load_replay_module() -> Any:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module("run_omega_recurrent_backward_replay")


def _load_two_fixtures(paths: Sequence[Path]) -> tuple[Any, list[dict[str, Any]]]:
    if len(paths) != 2:
        raise ValueError("exactly two fixtures (A then B) are required")
    replay = _load_replay_module()
    loaded: list[dict[str, Any]] = []
    for label, expected_name, path in zip(FIXTURE_LABELS, FIXTURE_FILENAMES, paths):
        if path.name != expected_name:
            raise ValueError(f"fixture order/name mismatch: expected {expected_name}, got {path.name}")
        header, arrays, file_sha = replay.read_fixture(path)
        metadata = header["metadata"]
        expected_update_window = (2, 0) if label.startswith("A_") else (3, 1)
        if (int(metadata["update_index"]), int(metadata["window_index"])) != expected_update_window:
            raise ValueError(f"fixture metadata mismatch for {path.name}")
        loaded.append({
            "label": label,
            "path": path,
            "header": header,
            "arrays": arrays,
            "sha256": file_sha,
        })
    return replay, loaded


def _make_report_base(pair_id: int, position: int, route: str, env_report: Mapping[str, Any], fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_route = PAIR_ROUTES[pair_id][position - 1]
    if route != expected_route:
        raise ValueError(f"pair {pair_id} position {position} must be {expected_route}, got {route}")
    return {
        "schema": "omega-recurrent-backward-replay-timing-process-v1",
        "status": "TIMING_PROCESS_PASS",
        "pair_id": pair_id,
        "position_in_pair": position,
        "route": route,
        "process_fresh": True,
        "route_order_for_pair": list(PAIR_ROUTES[pair_id]),
        "fixture_order": [str(item["label"]) for item in fixtures],
        "warmup_schedule": ["A_update2_window0", "B_update3_window1"],
        "measured_schedule": ["A_update2_window0", "B_update3_window1"] * 4,
        "warmup_backward_calls": 2,
        "measured_backward_calls": 8,
        "timing_performed": True,
        "measurement_contract": (
            "one perf_counter_ns interval around only the backward API call; all forward setup, "
            "gradient clearing, status checks, fixture hashes, and report work are outside"
        ),
        "measurement_api": "torch.autograd.backward" if route == "pytorch" else "omega_runtime_backward C ABI",
        "environment": dict(env_report),
        "fixtures": [
            {
                "label": str(item["label"]),
                "filename": Path(item["path"]).name,
                "sha256": str(item["sha256"]),
                "update": int(item["header"]["metadata"]["update_index"]),
                "window": int(item["header"]["metadata"]["window_index"]),
            }
            for item in fixtures
        ],
    }


def _torch_parameter_list(model: Any) -> list[Any]:
    block = model.blocks[0]
    return [
        model.prelude.weight,
        model.prelude_norm_weight,
        block.qkv.weight,
        block.qkv.bias,
        block.out.weight,
        block.out.bias,
        block.fc1.weight,
        block.fc1.bias,
        block.fc2.weight,
        block.fc2.bias,
        block.norm_weight,
        model.depth_embedding.weight,
        model.gate_logits,
    ]


def _run_pytorch_process(fixtures: Sequence[dict[str, Any]], pair_id: int, position: int, output: Path) -> dict[str, Any]:
    env_report = _assert_canonical_process_environment()
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("PyTorch route must own fresh process before torch import")
    replay = _load_replay_module()
    torch, (_p0, _ce, golden, fast_block_type, policy) = replay._torch_setup()
    if _assert_canonical_process_environment() != env_report:
        raise RuntimeError("PyTorch setup changed canonical process environment")
    if int(torch.get_num_threads()) != 4 or int(torch.get_num_interop_threads()) != 1:
        raise RuntimeError("canonical PyTorch thread policy drift: expected intraop=4, interop=1")

    cases: list[dict[str, Any]] = []
    for fixture in fixtures:
        model = replay._ReplayModel(fixture["arrays"], torch, fast_block_type)
        cases.append({
            **fixture,
            "model": model,
            "parameters": _torch_parameter_list(model),
            "upstream_readout": torch.from_numpy(np.array(fixture["arrays"]["G_R"], copy=True)),
            "upstream_next": torch.from_numpy(np.array(fixture["arrays"]["G_S"], copy=True)),
            "token_part": np.array(fixture["arrays"]["token_part"], copy=True),
            "previous_state": np.array(fixture["arrays"]["previous_state"], copy=True),
        })

    def evaluate(case: Mapping[str, Any], *, measure: bool) -> int | None:
        model = case["model"]
        for parameter in case["parameters"]:
            parameter.grad = None
        token_part = torch.from_numpy(case["token_part"].copy()).requires_grad_(True)
        previous_state = torch.from_numpy(case["previous_state"].copy()).requires_grad_(True)
        next_state, readout_states = golden._recurrent_forward_from_token_part(model, token_part, previous_state)
        backward_outputs = (next_state, readout_states)
        upstreams = (case["upstream_next"], case["upstream_readout"])
        if measure:
            started_ns = time.perf_counter_ns()
            torch.autograd.backward(backward_outputs, grad_tensors=upstreams, retain_graph=False)
            ended_ns = time.perf_counter_ns()
            return int(ended_ns - started_ns)
        torch.autograd.backward(backward_outputs, grad_tensors=upstreams, retain_graph=False)
        return None

    # Exactly one unmeasured backward for A and then B.
    for case in cases:
        evaluate(case, measure=False)

    measured_samples: list[dict[str, Any]] = []
    for sample_index in range(8):
        case = cases[sample_index % 2]
        elapsed_ns = evaluate(case, measure=True)
        if elapsed_ns is None or elapsed_ns <= 0:
            raise RuntimeError("PyTorch backward timer returned a non-positive duration")
        measured_samples.append({
            "sample_index": sample_index + 1,
            "fixture": case["label"],
            "elapsed_ns": elapsed_ns,
        })

    report = _make_report_base(pair_id, position, "pytorch", env_report, fixtures)
    report.update({
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "thread_policy": policy,
        "torch_parallel_info_outside_timer": torch.__config__.parallel_info(),
        "torch_loaded": True,
        "measured_samples": measured_samples,
        "measured_sum_ns_by_fixture": {
            case["label"]: sum(sample["elapsed_ns"] for sample in measured_samples if sample["fixture"] == case["label"])
            for case in cases
        },
        "warmup_fixture_sequence": [case["label"] for case in cases],
    })
    _write_json(output, report)
    return report


def _configure_native_library(replay: Any, dll_path: Path) -> tuple[Any, int, str]:
    resolved = dll_path.resolve()
    dll_sha = _sha256(resolved.read_bytes())
    if dll_sha != ACCEPTED_DLL_SHA256:
        raise ValueError(f"DLL SHA-256 differs from accepted candidate: {dll_sha}")
    library = ctypes.CDLL(str(resolved))
    required = (
        "omega_runtime_create",
        "omega_runtime_destroy",
        "omega_runtime_workspace_bytes",
        "omega_runtime_forward",
        "omega_runtime_backward",
    )
    missing = [name for name in required if not hasattr(library, name)]
    if missing:
        raise RuntimeError(f"accepted DLL lacks persistent runtime exports: {missing}")
    Config = replay._Config
    Params = replay._Params
    Grads = replay._Grads
    FloatPointer = replay._FLOAT_PTR
    SizePointer = replay._SIZE_T_PTR
    DoublePointer = replay._DOUBLE_PTR
    library.omega_runtime_create.argtypes = [ctypes.c_size_t]
    library.omega_runtime_create.restype = ctypes.c_void_p
    library.omega_runtime_destroy.argtypes = [ctypes.c_void_p]
    library.omega_runtime_destroy.restype = None
    library.omega_runtime_workspace_bytes.argtypes = [ctypes.c_void_p, Config]
    library.omega_runtime_workspace_bytes.restype = ctypes.c_size_t
    library.omega_runtime_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Config),
        ctypes.POINTER(Params),
        FloatPointer,
        FloatPointer,
        FloatPointer,
        FloatPointer,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.omega_runtime_forward.restype = ctypes.c_int
    library.omega_runtime_backward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Config),
        ctypes.POINTER(Params),
        FloatPointer,
        FloatPointer,
        FloatPointer,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(Grads),
    ]
    library.omega_runtime_backward.restype = ctypes.c_int
    runtime = library.omega_runtime_create(4)
    if not runtime:
        raise RuntimeError("omega_runtime_create(4) failed")
    return library, runtime, dll_sha


def _make_native_case(replay: Any, library: Any, runtime: Any, fixture: Mapping[str, Any]) -> dict[str, Any]:
    arrays = fixture["arrays"]
    state_storage = arrays["state_part_weight_storage"]
    state_matrix = state_storage[:, replay.DIMENSION:]
    expected_stride = (2 * replay.DIMENSION * 4, 4)
    if state_matrix.strides != expected_stride:
        raise ValueError(f"fused state view stride mismatch: {state_matrix.strides}")
    if not np.array_equal(state_matrix, arrays["state_part_weight"]):
        raise ValueError("fused state view data mismatch")

    config = replay._Config(
        replay.WINDOW_TOKENS,
        replay.BATCH,
        replay.SLOTS,
        replay.DIMENSION,
        replay.ROUNDS,
        1,
        0,
    )
    matrix_view = replay._MatrixView(
        replay._float_ptr(state_matrix),
        replay.SLOTS * replay.DIMENSION,
        replay.DIMENSION,
        2 * replay.DIMENSION,
    )
    params = replay._Params(
        matrix_view,
        *(
            replay._float_ptr(arrays[name])
            for name in (
                "prelude_norm_weight",
                "block_qkv_weight",
                "block_qkv_bias",
                "block_out_weight",
                "block_out_bias",
                "block_fc1_weight",
                "block_fc1_bias",
                "block_fc2_weight",
                "block_fc2_bias",
                "block_norm_weight",
                "depth_embedding_weight",
                "gate_logits",
            )
        ),
    )
    workspace_bytes = int(library.omega_runtime_workspace_bytes(runtime, config))
    if workspace_bytes <= 0:
        raise RuntimeError("native runtime returned invalid workspace size")
    workspace = ctypes.create_string_buffer(workspace_bytes)
    next_state = np.empty_like(arrays["previous_state"])
    readout_states = np.empty_like(arrays["readout_states"])
    gradients = {name: np.empty_like(arrays[name]) for name in replay.GRADIENT_NAMES}
    null_float = replay._FLOAT_PTR()
    null_size = replay._SIZE_T_PTR()
    null_double = replay._DOUBLE_PTR()
    native_grads = replay._Grads(
        *(replay._float_ptr(gradients[name]) for name in replay.GRADIENT_NAMES),
        *(null_float for _ in replay.GRADIENT_NAMES),
        *(null_size for _ in replay.GRADIENT_NAMES),
        null_double,
        null_size,
    )
    config_ptr = ctypes.byref(config)
    params_ptr = ctypes.byref(params)
    workspace_ptr = ctypes.cast(workspace, ctypes.c_void_p)
    grads_ptr = ctypes.byref(native_grads)
    token_ptr = replay._float_ptr(arrays["token_part"])
    previous_ptr = replay._float_ptr(arrays["previous_state"])
    next_ptr = replay._float_ptr(next_state)
    readout_ptr = replay._float_ptr(readout_states)
    upstream_readout_ptr = replay._float_ptr(arrays["G_R"])
    upstream_next_ptr = replay._float_ptr(arrays["G_S"])
    return {
        **fixture,
        "config": config,
        "config_ptr": config_ptr,
        "params_ptr": params_ptr,
        "workspace": workspace,
        "workspace_ptr": workspace_ptr,
        "workspace_bytes": workspace_bytes,
        "grads": native_grads,
        "grads_ptr": grads_ptr,
        # ctypes pointers expose raw addresses; keep every NumPy owner alive
        # for the complete sequence of forward/backward calls.
        "next_state_owner": next_state,
        "readout_states_owner": readout_states,
        "gradient_owners": gradients,
        "token_ptr": token_ptr,
        "previous_ptr": previous_ptr,
        "next_ptr": next_ptr,
        "readout_ptr": readout_ptr,
        "upstream_readout_ptr": upstream_readout_ptr,
        "upstream_next_ptr": upstream_next_ptr,
    }


def _run_native_process(
    fixtures: Sequence[dict[str, Any]],
    pair_id: int,
    position: int,
    dll_path: Path,
    output: Path,
    report_route: str = "native",
) -> dict[str, Any]:
    env_report = _assert_canonical_process_environment()
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("native timing process must not load PyTorch")
    replay = _load_replay_module()
    library, runtime, dll_sha = _configure_native_library(replay, dll_path)
    if _assert_canonical_process_environment() != env_report:
        library.omega_runtime_destroy(runtime)
        raise RuntimeError("native setup changed canonical process environment")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        library.omega_runtime_destroy(runtime)
        raise RuntimeError("native route loaded PyTorch unexpectedly")
    cases = [_make_native_case(replay, library, runtime, fixture) for fixture in fixtures]

    def evaluate(case: Mapping[str, Any], *, measure: bool) -> int | None:
        forward_status = int(
            library.omega_runtime_forward(
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
        )
        if forward_status != 0:
            raise RuntimeError(f"native forward failed outside timer with status {forward_status}")
        backward_args = (
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
        if measure:
            started_ns = time.perf_counter_ns()
            backward_status = int(library.omega_runtime_backward(*backward_args))
            ended_ns = time.perf_counter_ns()
            if backward_status != 0:
                raise RuntimeError(f"native backward failed after timer with status {backward_status}")
            return int(ended_ns - started_ns)
        backward_status = int(library.omega_runtime_backward(*backward_args))
        if backward_status != 0:
            raise RuntimeError(f"native warmup backward failed with status {backward_status}")
        return None

    try:
        for case in cases:
            evaluate(case, measure=False)
        measured_samples: list[dict[str, Any]] = []
        for sample_index in range(8):
            case = cases[sample_index % 2]
            elapsed_ns = evaluate(case, measure=True)
            if elapsed_ns is None or elapsed_ns <= 0:
                raise RuntimeError("native backward timer returned a non-positive duration")
            measured_samples.append({
                "sample_index": sample_index + 1,
                "fixture": case["label"],
                "elapsed_ns": elapsed_ns,
            })
    finally:
        library.omega_runtime_destroy(runtime)

    report = _make_report_base(pair_id, position, report_route, env_report, fixtures)
    report.update({
        "native_library_sha256": dll_sha,
        "native_library_bytes": int(dll_path.resolve().stat().st_size),
        "native_runtime_threads": 4,
        "torch_loaded": False,
        "workspace_mode": "clean production workspace (instrumentation=0)",
        "measured_samples": measured_samples,
        "measured_sum_ns_by_fixture": {
            case["label"]: sum(sample["elapsed_ns"] for sample in measured_samples if sample["fixture"] == case["label"])
            for case in cases
        },
        "warmup_fixture_sequence": [case["label"] for case in cases],
    })
    _write_json(output, report)
    return report


def aggregate_reports(process_reports: Sequence[Path], output: Path) -> dict[str, Any]:
    expected_order = [
        (1, 1, "pytorch"),
        (1, 2, "native"),
        (2, 1, "native"),
        (2, 2, "pytorch"),
        (3, 1, "pytorch"),
        (3, 2, "native"),
    ]
    if len(process_reports) != len(expected_order):
        raise ValueError("aggregate requires exactly six raw process reports")
    rows: list[dict[str, Any]] = []
    for path, expected in zip(process_reports, expected_order):
        report = json.loads(path.read_text(encoding="utf-8"))
        observed = (int(report["pair_id"]), int(report["position_in_pair"]), str(report["route"]))
        if observed != expected:
            raise ValueError(f"raw report order mismatch: {path.name}: {observed} != {expected}")
        if report.get("status") != "TIMING_PROCESS_PASS" or report.get("timing_performed") is not True:
            raise ValueError(f"raw process report did not pass: {path}")
        if int(report.get("warmup_backward_calls", -1)) != 2 or int(report.get("measured_backward_calls", -1)) != 8:
            raise ValueError(f"process count mismatch: {path}")
        samples = report.get("measured_samples")
        expected_sequence = [FIXTURE_LABELS[index % 2] for index in range(8)]
        if len(samples) != 8 or [item["fixture"] for item in samples] != expected_sequence:
            raise ValueError(f"measured A/B sequence mismatch: {path}")
        if any(int(item["elapsed_ns"]) <= 0 for item in samples):
            raise ValueError(f"non-positive measured duration: {path}")
        rows.append({"path": path.as_posix(), "report": report})

    canonical_environment = rows[0]["report"]["environment"]
    if any(row["report"]["environment"] != canonical_environment for row in rows[1:]):
        raise ValueError("processes did not preserve one canonical environment")

    route_samples: dict[str, list[dict[str, Any]]] = {"pytorch": [], "native": []}
    for row in rows:
        route_samples[row["report"]["route"]].extend(row["report"]["measured_samples"])
    expected_by_route = {route: {label: 12 for label in FIXTURE_LABELS} for route in route_samples}
    totals_by_route: dict[str, dict[str, Any]] = {}
    for route, samples in route_samples.items():
        by_fixture = {
            label: [int(item["elapsed_ns"]) for item in samples if item["fixture"] == label]
            for label in FIXTURE_LABELS
        }
        counts = {label: len(values) for label, values in by_fixture.items()}
        if counts != expected_by_route[route]:
            raise ValueError(f"fixture sample balance mismatch for {route}: {counts}")
        totals_by_route[route] = {
            "measured_backward_calls": len(samples),
            "sample_count_by_fixture": counts,
            "sum_ns_by_fixture": {label: sum(values) for label, values in by_fixture.items()},
            "mean_ns_by_fixture": {label: sum(values) / len(values) for label, values in by_fixture.items()},
            "median_ns_by_fixture": {
                label: float(np.median(np.asarray(values, dtype=np.float64))) for label, values in by_fixture.items()
            },
            "sum_ns_all_fixtures": sum(int(item["elapsed_ns"]) for item in samples),
            "mean_ns_all_fixtures": float(np.mean(np.asarray([item["elapsed_ns"] for item in samples], dtype=np.float64))),
        }

    native_total = int(totals_by_route["native"]["sum_ns_all_fixtures"])
    pytorch_total = int(totals_by_route["pytorch"]["sum_ns_all_fixtures"])
    ratio_by_fixture = {
        label: totals_by_route["native"]["sum_ns_by_fixture"][label]
        / totals_by_route["pytorch"]["sum_ns_by_fixture"][label]
        for label in FIXTURE_LABELS
    }
    pair_ratios: dict[str, Any] = {}
    for pair_id in (1, 2, 3):
        matching = [row["report"] for row in rows if int(row["report"]["pair_id"]) == pair_id]
        pytorch_sum = sum(int(sample["elapsed_ns"]) for report in matching if report["route"] == "pytorch" for sample in report["measured_samples"])
        native_sum = sum(int(sample["elapsed_ns"]) for report in matching if report["route"] == "native" for sample in report["measured_samples"])
        pair_ratios[str(pair_id)] = {
            "pytorch_sum_ns": pytorch_sum,
            "native_sum_ns": native_sum,
            "R_backward": native_sum / pytorch_sum,
        }

    aggregate = {
        "schema": "omega-recurrent-backward-replay-timing-aggregate-v1",
        "status": "TIMING_AGGREGATE_PASS",
        "timing_performed": True,
        "process_order": [
            {"pair_id": pair, "position": position, "route": route}
            for pair, position, route in expected_order
        ],
        "process_count": 6,
        "backward_calls_total": 60,
        "warmup_backward_calls_total": 12,
        "measured_backward_calls_total": 48,
        "measured_calls_per_route": 24,
        "measured_calls_per_fixture_per_route": 12,
        "canonical_environment": canonical_environment,
        "timing_definition": "R_backward = sum(T_native) / sum(T_pytorch), using measured backward-call nanoseconds only",
        "totals": totals_by_route,
        "R_backward": native_total / pytorch_total,
        "R_backward_by_fixture": ratio_by_fixture,
        "R_backward_by_pair": pair_ratios,
        "raw_process_reports": [row["path"] for row in rows],
    }
    _write_json(output, aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("backward-timing", "aggregate"), required=True)
    parser.add_argument("--route", choices=("pytorch", "native"))
    parser.add_argument("--pair-id", type=int, choices=(1, 2, 3))
    parser.add_argument("--position", type=int, choices=(1, 2))
    parser.add_argument("--fixture", type=Path, action="append", default=[])
    parser.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--process-report", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    if args.mode == "aggregate":
        if args.output is None:
            parser.error("--output is required for aggregate")
        report = aggregate_reports(args.process_report, args.output)
    else:
        if args.route is None or args.pair_id is None or args.position is None or args.output is None:
            parser.error("backward-timing requires --route, --pair-id, --position, and --output")
        expected_route = PAIR_ROUTES[args.pair_id][args.position - 1]
        if args.route != expected_route:
            parser.error(f"pair {args.pair_id} position {args.position} must be {expected_route}")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite timing process report: {args.output}")
        env_report = _assert_canonical_process_environment()
        replay, fixtures = _load_two_fixtures(args.fixture or [DEFAULT_FIXTURE_ROOT / name for name in FIXTURE_FILENAMES])
        del env_report
        if args.route == "pytorch":
            report = _run_pytorch_process(fixtures, args.pair_id, args.position, args.output)
        else:
            report = _run_native_process(fixtures, args.pair_id, args.position, args.dll, args.output)

    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
