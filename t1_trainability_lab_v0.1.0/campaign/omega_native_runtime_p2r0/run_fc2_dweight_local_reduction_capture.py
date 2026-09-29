"""Capture preselected M=8 FC2 dWeight groups from certified native replay."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import run_omega_recurrent_backward_replay as replay


HERE = Path(__file__).resolve().parent
FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
DEFAULT_OUTPUT = HERE / "results" / "fc2_dweight_local_reduction_step_a"
GROUP_SIZE = 8
HIDDEN = 512
OUTPUTS = 128
WORKERS = 4
BATCH = 8

# Fixed before any performance measurements. All groups have nonzero prior dW.
PRESELECTED_GROUPS = {
    "replay_K4_update2_window0.bin": (
        (200, 1, 0),
        (17, 3, 1),
    ),
    "replay_K4_update3_window1.bin": (
        (233, 0, 2),
        (64, 2, 7),
    ),
}


class _GroupIdentity(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_size_t),
        ("round", ctypes.c_size_t),
        ("batch", ctypes.c_size_t),
    ]


_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_CaptureCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    _FLOAT_PTR,
    _FLOAT_PTR,
    _FLOAT_PTR,
    _FLOAT_PTR,
    _FLOAT_PTR,
    _FLOAT_PTR,
    ctypes.c_void_p,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _array_from_pointer(pointer: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.ctypeslib.as_array(pointer, shape=(int(np.prod(shape)),)).copy().reshape(shape)


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return {
        "shape": list(canonical.shape),
        "dtype": "float32-le",
        "strides_bytes": list(canonical.strides),
        "sha256": _sha256(canonical.tobytes(order="C")),
    }


def _make_native_call(library: Any, arrays: dict[str, np.ndarray]) -> tuple[Any, Any, Any, Any, Any]:
    config = replay._Config(replay.WINDOW_TOKENS, replay.BATCH, replay.SLOTS, replay.DIMENSION, replay.ROUNDS, 1, 1)
    state_storage = arrays["state_part_weight_storage"]
    state_matrix = state_storage[:, replay.DIMENSION:]
    if state_matrix.strides != (2 * replay.DIMENSION * 4, 4):
        raise ValueError(f"state matrix stride mismatch: {state_matrix.strides}")
    if not np.array_equal(state_matrix, arrays["state_part_weight"]):
        raise ValueError("state matrix storage view differs from saved logical parameter")

    matrix_view = replay._MatrixView(
        replay._float_ptr(state_matrix), replay.SLOTS * replay.DIMENSION, replay.DIMENSION, 2 * replay.DIMENSION
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
    runtime = library.omega_runtime_create(WORKERS)
    if not runtime:
        raise RuntimeError("omega_runtime_create(4) failed")
    workspace_bytes = int(library.omega_runtime_workspace_bytes(runtime, config))
    if workspace_bytes <= 0:
        library.omega_runtime_destroy(runtime)
        raise RuntimeError("native runtime returned invalid workspace size")
    workspace = ctypes.create_string_buffer(workspace_bytes)
    return runtime, config, params, workspace, workspace_bytes


def _run_fixture(library: Any, fixture_path: Path, output_root: Path) -> dict[str, Any]:
    header, arrays, fixture_sha = replay.read_fixture(fixture_path)
    metadata = header["metadata"]
    if int(metadata["K"]) != replay.ROUNDS or tuple(arrays["token_part"].shape) != (
        BATCH,
        replay.WINDOW_TOKENS,
        replay.SLOTS * replay.DIMENSION,
    ):
        raise ValueError(f"unexpected fixture shape/config: {fixture_path}")
    targets_for_fixture = PRESELECTED_GROUPS.get(fixture_path.name)
    if not targets_for_fixture:
        raise ValueError(f"fixture was not in the preselected set: {fixture_path.name}")
    targets = (_GroupIdentity * len(targets_for_fixture))(
        *(_GroupIdentity(position, round_index, batch) for position, round_index, batch in targets_for_fixture)
    )
    captured: dict[tuple[int, int, int], dict[str, Any]] = {}
    callback_errors: list[str] = []

    @_CaptureCallback
    def on_group(
        worker_index: int,
        position: int,
        round_index: int,
        batch: int,
        preactivation: Any,
        activated: Any,
        output_gradient: Any,
        dweight_before: Any,
        dweight_control_after: Any,
        dweight_candidate_after: Any,
        _user_data: Any,
    ) -> None:
        try:
            identity = (int(position), int(round_index), int(batch))
            key = (identity[0], identity[1], identity[2])
            expected_worker = identity[2] // (BATCH // WORKERS)
            if int(worker_index) != expected_worker:
                raise AssertionError(f"group {identity} ran on worker {worker_index}, expected {expected_worker}")
            if key in captured:
                raise AssertionError(f"duplicate capture for group {identity}")
            captured[key] = {
                "worker_index": int(worker_index),
                "fc1_preactivation": _array_from_pointer(preactivation, (GROUP_SIZE, HIDDEN)),
                "activated": _array_from_pointer(activated, (GROUP_SIZE, HIDDEN)),
                "output_gradient": _array_from_pointer(output_gradient, (GROUP_SIZE, OUTPUTS)),
                "dweight_before": _array_from_pointer(dweight_before, (OUTPUTS, HIDDEN)),
                "dweight_control_after": _array_from_pointer(dweight_control_after, (OUTPUTS, HIDDEN)),
                "dweight_candidate_after": _array_from_pointer(dweight_candidate_after, (OUTPUTS, HIDDEN)),
            }
        except Exception as exc:  # ctypes callbacks cannot propagate exceptions to C++.
            callback_errors.append(f"{type(exc).__name__}: {exc}")

    library.omega_fc2_replay_capture_set_targets.argtypes = [
        ctypes.POINTER(_GroupIdentity),
        ctypes.c_size_t,
        _CaptureCallback,
        ctypes.c_void_p,
    ]
    library.omega_fc2_replay_capture_set_targets.restype = ctypes.c_int
    set_status = int(library.omega_fc2_replay_capture_set_targets(targets, len(targets_for_fixture), on_group, None))
    if set_status != 0:
        raise RuntimeError(f"capture target setup failed with status {set_status}")

    runtime, config, params, workspace, workspace_bytes = _make_native_call(library, arrays)
    try:
        next_state = np.empty_like(arrays["previous_state"])
        readout_states = np.empty_like(arrays["readout_states"])
        forward_status = int(
            library.omega_runtime_forward(
                runtime,
                ctypes.byref(config),
                ctypes.byref(params),
                replay._float_ptr(arrays["token_part"]),
                replay._float_ptr(arrays["previous_state"]),
                replay._float_ptr(next_state),
                replay._float_ptr(readout_states),
                ctypes.cast(workspace, ctypes.c_void_p),
                workspace_bytes,
            )
        )
        if forward_status != 0:
            raise RuntimeError(f"native forward failed with status {forward_status}")
        forward_checks = {}
        for name, actual in (("next_state", next_state), ("readout_states", readout_states)):
            expected = arrays[name]
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            forward_checks[name] = {
                "max_abs_error": float(difference.max(initial=0.0)),
                "failed_elements_atol_1e-5": int(np.count_nonzero(~np.isfinite(actual) | (difference > 1.0e-5))),
            }
        if any(check["failed_elements_atol_1e-5"] for check in forward_checks.values()):
            raise AssertionError(f"native forward diverged from frozen fixture: {forward_checks}")

        gradients = {name: np.empty_like(arrays[name]) for name in replay.GRADIENT_NAMES}
        sum_abs = {name: np.empty_like(arrays[name]) for name in replay.GRADIENT_NAMES}
        counts = {name: np.empty(arrays[name].shape, dtype=np.uintp) for name in replay.GRADIENT_NAMES}
        fp64_depth = np.empty_like(arrays["d_depth_embedding_weight"], dtype=np.float64)
        depth_max = np.empty_like(arrays["d_depth_embedding_weight"], dtype=np.uintp)
        grads = replay._Grads(
            *[replay._float_ptr(gradients[name]) for name in replay.GRADIENT_NAMES],
            *[replay._float_ptr(sum_abs[name]) for name in replay.GRADIENT_NAMES],
            *[counts[name].ctypes.data_as(replay._SIZE_T_PTR) for name in replay.GRADIENT_NAMES],
            fp64_depth.ctypes.data_as(replay._DOUBLE_PTR),
            depth_max.ctypes.data_as(replay._SIZE_T_PTR),
        )
        backward_status = int(
            library.omega_runtime_backward(
                runtime,
                ctypes.byref(config),
                ctypes.byref(params),
                replay._float_ptr(arrays["token_part"]),
                replay._float_ptr(arrays["G_R"]),
                replay._float_ptr(arrays["G_S"]),
                ctypes.cast(workspace, ctypes.c_void_p),
                workspace_bytes,
                ctypes.byref(grads),
            )
        )
        if backward_status != 0:
            raise RuntimeError(f"native backward failed with status {backward_status}")
        if callback_errors:
            raise AssertionError("capture callback failed: " + "; ".join(callback_errors))
        if set(captured) != set(targets_for_fixture):
            raise AssertionError(f"capture set mismatch; expected={targets_for_fixture}, got={sorted(captured)}")
    finally:
        library.omega_runtime_destroy(runtime)

    groups = []
    for identity in targets_for_fixture:
        position, round_index, batch = identity
        captured_group = captured[identity]
        before = captured_group["dweight_before"]
        control = captured_group["dweight_control_after"]
        candidate = captured_group["dweight_candidate_after"]
        if not np.any(before != 0.0):
            raise AssertionError(f"selected group {identity} has zero initial dW")
        if not np.isfinite(before).all() or not np.isfinite(control).all() or not np.isfinite(candidate).all():
            raise AssertionError(f"selected group {identity} has non-finite dW")
        bitwise_equal = control.tobytes(order="C") == candidate.tobytes(order="C")
        if not bitwise_equal:
            raise AssertionError(f"candidate changed FC2 accumulation bits for group {identity}")

        group_name = f"{fixture_path.stem}_p{position}_r{round_index}_b{batch}"
        group_path = output_root / f"{group_name}.npz"
        if group_path.exists():
            raise FileExistsError(f"refusing to overwrite capture: {group_path}")
        payload_arrays = {
            name: np.ascontiguousarray(captured_group[name].astype("<f4", copy=False))
            for name in (
                "fc1_preactivation",
                "activated",
                "output_gradient",
                "dweight_before",
                "dweight_control_after",
                "dweight_candidate_after",
            )
        }
        np.savez(group_path, **payload_arrays)
        groups.append({
            "fixture": fixture_path.name,
            "fixture_sha256": fixture_sha,
            "fixture_payload_sha256": header["payload_sha256"],
            "fixture_id": metadata["fixture_id"],
            "update_index": int(metadata["update_index"]),
            "window_index": int(metadata["window_index"]),
            "position": position,
            "round": round_index,
            "global_batch": batch,
            "worker_index": captured_group["worker_index"],
            "local_batch_index": batch % (BATCH // WORKERS),
            "slots": list(range(GROUP_SIZE)),
            "shapes": {
                "fc1_preactivation": [GROUP_SIZE, HIDDEN],
                "activated": [GROUP_SIZE, HIDDEN],
                "output_gradient": [GROUP_SIZE, OUTPUTS],
                "dweight_before": [OUTPUTS, HIDDEN],
            },
            "strides_elements": {
                "fc1_preactivation": [HIDDEN, 1],
                "activated": [HIDDEN, 1],
                "output_gradient": [OUTPUTS, 1],
                "dweight_before": [HIDDEN, 1],
            },
            "dweight_before_nonzero_elements": int(np.count_nonzero(before)),
            "control_candidate_bitwise_equal": bitwise_equal,
            "arrays": {name: _array_identity(array) for name, array in payload_arrays.items()},
            "capture_file": group_path.name,
            "capture_file_sha256": _file_sha256(group_path),
        })
    return {
        "fixture": fixture_path.name,
        "fixture_sha256": fixture_sha,
        "forward_checks": forward_checks,
        "captured_groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True, help="AVX2 capture DLL built with OMEGA_FC2_REPLAY_CAPTURE")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture", type=Path, action="append", default=[])
    args = parser.parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("capture harness must remain NumPy/ctypes only")
    fixtures = args.fixture or [FIXTURE_ROOT / name for name in PRESELECTED_GROUPS]
    for fixture in fixtures:
        if fixture.name not in PRESELECTED_GROUPS:
            raise ValueError(f"fixture is not preselected: {fixture}")
        if not fixture.is_file():
            raise FileNotFoundError(fixture)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite capture output directory: {args.output}")
    args.output.mkdir(parents=True)

    dll_path = args.dll.resolve()
    library = ctypes.CDLL(str(dll_path))
    for name in (
        "omega_runtime_create",
        "omega_runtime_destroy",
        "omega_runtime_workspace_bytes",
        "omega_runtime_forward",
        "omega_runtime_backward",
        "omega_fc2_replay_capture_set_targets",
    ):
        if not hasattr(library, name):
            raise RuntimeError(f"capture DLL missing export: {name}")
    library.omega_runtime_create.argtypes = [ctypes.c_size_t]
    library.omega_runtime_create.restype = ctypes.c_void_p
    library.omega_runtime_destroy.argtypes = [ctypes.c_void_p]
    library.omega_runtime_destroy.restype = None
    library.omega_runtime_workspace_bytes.argtypes = [ctypes.c_void_p, replay._Config]
    library.omega_runtime_workspace_bytes.restype = ctypes.c_size_t
    library.omega_runtime_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(replay._Config),
        ctypes.POINTER(replay._Params),
        _FLOAT_PTR,
        _FLOAT_PTR,
        _FLOAT_PTR,
        _FLOAT_PTR,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.omega_runtime_forward.restype = ctypes.c_int
    library.omega_runtime_backward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(replay._Config),
        ctypes.POINTER(replay._Params),
        _FLOAT_PTR,
        _FLOAT_PTR,
        _FLOAT_PTR,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(replay._Grads),
    ]
    library.omega_runtime_backward.restype = ctypes.c_int

    fixture_results = [_run_fixture(library, fixture, args.output) for fixture in fixtures]
    all_groups = [group for result in fixture_results for group in result["captured_groups"]]
    if len(all_groups) < 4 or len({group["fixture"] for group in all_groups}) < 2:
        raise AssertionError("selection must cover both fixtures and multiple local groups")
    if len({group["position"] for group in all_groups}) < 2 or len({group["round"] for group in all_groups}) < 2:
        raise AssertionError("selection must cover multiple positions and rounds")
    report = {
        "schema": "omega-fc2-dweight-local-reduction-step-a-v1",
        "status": "CAPTURE_AND_LOCAL_EQUIVALENCE_PASS",
        "timing_performed": False,
        "worker_count": WORKERS,
        "global_batch": BATCH,
        "local_batch_per_worker": BATCH // WORKERS,
        "group_size_m": GROUP_SIZE,
        "fc2_shapes": {"A": [GROUP_SIZE, HIDDEN], "D": [GROUP_SIZE, OUTPUTS], "dW": [OUTPUTS, HIDDEN]},
        "selection_policy": "fixed identities in PRESELECTED_GROUPS before any timing; both fixtures, multiple positions and rounds",
        "control_path": "actual omega_recurrent.cpp AVX2 per-slot FC2 dWeight accumulation in full native forward/backward replay",
        "candidate_path": "fc2_dweight_m8.h AVX2 retained-tile accumulation over captured production activations and output gradients",
        "candidate_order": "for each dW element: (((dW_old+c0)+c1)+...+c7)",
        "fixture_inputs": [
            {
                "path": path.as_posix(),
                "sha256": _file_sha256(path),
            }
            for path in fixtures
        ],
        "capture_dll": {"path": dll_path.as_posix(), "sha256": _file_sha256(dll_path)},
        "source_sha256": {
            "omega_recurrent.cpp": _file_sha256(HERE / "native" / "omega_recurrent.cpp"),
            "fc2_dweight_m8.h": _file_sha256(HERE / "native" / "fc2_dweight_m8.h"),
            "capture_harness.py": _file_sha256(Path(__file__).resolve()),
        },
        "fixtures": fixture_results,
    }
    report_path = args.output / "capture_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
