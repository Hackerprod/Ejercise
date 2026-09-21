"""Clean production ctypes/autograd bridge for the P2-R2 benchmark.

The R1 bridge remains certification-only. This module requests the native clean
workspace mode and passes null optional diagnostic pointers. Parameters are
borrowed through ctypes pointers; no parameter tensor is copied or cloned.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor


HERE = Path(__file__).resolve().parent
_LIBRARY: ctypes.CDLL | None = None
_LOADED_LIBRARY_PATH: Path | None = None
_DLL_DIRECTORY_HANDLES: list[object] = []
_POOL: dict[tuple[int, ...], list[Tensor]] = {}
_LOCK = threading.Lock()
_ALLOCATIONS = 0
_REUSES = 0
_FORWARD_C_ABI_SECONDS = 0.0
_BACKWARD_C_ABI_SECONDS = 0.0
_FORWARD_BOUNDARY_SECONDS = 0.0
_BACKWARD_BOUNDARY_SECONDS = 0.0


class _OmegaRecurrentConfig(ctypes.Structure):
    _fields_ = [
        ("sequence_length", ctypes.c_size_t),
        ("batch", ctypes.c_size_t),
        ("slots", ctypes.c_size_t),
        ("dimension", ctypes.c_size_t),
        ("rounds", ctypes.c_size_t),
        ("training", ctypes.c_int),
        ("instrumentation", ctypes.c_int),
    ]


class _OmegaRecurrentProfileDirection(ctypes.Structure):
    _fields_ = [
        ("calls", ctypes.c_uint64),
        ("qkv_projection_calls", ctypes.c_uint64),
        ("qkv_projection_seconds", ctypes.c_double),
        ("attention_scores_softmax_mixing_calls", ctypes.c_uint64),
        ("attention_scores_softmax_mixing_seconds", ctypes.c_double),
        ("out_projection_calls", ctypes.c_uint64),
        ("out_projection_seconds", ctypes.c_double),
        ("fc1_gelu_calls", ctypes.c_uint64),
        ("fc1_gelu_seconds", ctypes.c_double),
        ("fc2_calls", ctypes.c_uint64),
        ("fc2_seconds", ctypes.c_double),
        ("rmsnorm_gates_calls", ctypes.c_uint64),
        ("rmsnorm_gates_seconds", ctypes.c_double),
        ("state_prelude_calls", ctypes.c_uint64),
        ("state_prelude_seconds", ctypes.c_double),
        ("depth_embedding_pairwise_reduction_calls", ctypes.c_uint64),
        ("depth_embedding_pairwise_reduction_seconds", ctypes.c_double),
        ("depth_embedding_carry_calls", ctypes.c_uint64),
        ("depth_embedding_carry_seconds", ctypes.c_double),
        ("history_buffer_reads_writes_calls", ctypes.c_uint64),
        ("history_buffer_reads_writes_seconds", ctypes.c_double),
    ]


class _OmegaRecurrentProfileSnapshot(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("compiled", ctypes.c_int),
        ("forward", _OmegaRecurrentProfileDirection),
        ("backward", _OmegaRecurrentProfileDirection),
    ]


_FloatPointer = ctypes.POINTER(ctypes.c_float)


class _OmegaMatrixViewF32(ctypes.Structure):
    _fields_ = [
        ("data", _FloatPointer),
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
        ("row_stride", ctypes.c_ssize_t),
    ]


class _OmegaRecurrentParams(ctypes.Structure):
    _fields_ = [
        ("state_part_weight", _OmegaMatrixViewF32),
        ("prelude_norm_weight", _FloatPointer),
        ("block_qkv_weight", _FloatPointer),
        ("block_qkv_bias", _FloatPointer),
        ("block_out_weight", _FloatPointer),
        ("block_out_bias", _FloatPointer),
        ("block_fc1_weight", _FloatPointer),
        ("block_fc1_bias", _FloatPointer),
        ("block_fc2_weight", _FloatPointer),
        ("block_fc2_bias", _FloatPointer),
        ("block_norm_weight", _FloatPointer),
        ("depth_embedding", _FloatPointer),
        ("gate_logits", _FloatPointer),
    ]


class _OmegaRecurrentGrads(ctypes.Structure):
    _fields_ = [
        ("d_token_part", _FloatPointer),
        ("d_previous_state", _FloatPointer),
        ("d_state_part_weight", _FloatPointer),
        ("d_prelude_norm_weight", _FloatPointer),
        ("d_block_qkv_weight", _FloatPointer),
        ("d_block_qkv_bias", _FloatPointer),
        ("d_block_out_weight", _FloatPointer),
        ("d_block_out_bias", _FloatPointer),
        ("d_block_fc1_weight", _FloatPointer),
        ("d_block_fc1_bias", _FloatPointer),
        ("d_block_fc2_weight", _FloatPointer),
        ("d_block_fc2_bias", _FloatPointer),
        ("d_block_norm_weight", _FloatPointer),
        ("d_depth_embedding", _FloatPointer),
        ("d_gate_logits", _FloatPointer),
        ("sum_abs_d_token_part", _FloatPointer),
        ("sum_abs_d_previous_state", _FloatPointer),
        ("sum_abs_d_state_part_weight", _FloatPointer),
        ("sum_abs_d_prelude_norm_weight", _FloatPointer),
        ("sum_abs_d_block_qkv_weight", _FloatPointer),
        ("sum_abs_d_block_qkv_bias", _FloatPointer),
        ("sum_abs_d_block_out_weight", _FloatPointer),
        ("sum_abs_d_block_out_bias", _FloatPointer),
        ("sum_abs_d_block_fc1_weight", _FloatPointer),
        ("sum_abs_d_block_fc1_bias", _FloatPointer),
        ("sum_abs_d_block_fc2_weight", _FloatPointer),
        ("sum_abs_d_block_fc2_bias", _FloatPointer),
        ("sum_abs_d_block_norm_weight", _FloatPointer),
        ("sum_abs_d_depth_embedding", _FloatPointer),
        ("sum_abs_d_gate_logits", _FloatPointer),
        ("count_d_token_part", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_previous_state", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_state_part_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_prelude_norm_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_qkv_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_qkv_bias", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_out_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_out_bias", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_fc1_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_fc1_bias", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_fc2_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_fc2_bias", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_block_norm_weight", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_depth_embedding", ctypes.POINTER(ctypes.c_size_t)),
        ("count_d_gate_logits", ctypes.POINTER(ctypes.c_size_t)),
        ("fp64_d_depth_embedding", ctypes.POINTER(ctypes.c_double)),
        ("depth_max_level", ctypes.POINTER(ctypes.c_size_t)),
    ]


_PARAMETER_NAMES = (
    "state_part_weight",
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
    "depth_embedding",
    "gate_logits",
)


def _float_pointer(value: Tensor) -> _FloatPointer:
    return ctypes.cast(ctypes.c_void_p(int(value.data_ptr())), _FloatPointer)


def _candidate_library_paths() -> Iterable[Path]:
    override = os.environ.get("OMEGA_RECURRENT_DLL")
    if override:
        yield Path(override)
    yield HERE / "omega_recurrent.dll"
    build_root = HERE / "native" / "build"
    if build_root.is_dir():
        yield from sorted(build_root.rglob("omega_recurrent.dll"))


def _load_library(path: Path) -> ctypes.CDLL:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path.parent)))
    library = ctypes.CDLL(str(path))
    config_pointer = ctypes.POINTER(_OmegaRecurrentConfig)
    params_pointer = ctypes.POINTER(_OmegaRecurrentParams)
    grads_pointer = ctypes.POINTER(_OmegaRecurrentGrads)
    library.omega_recurrent_workspace_bytes.argtypes = [_OmegaRecurrentConfig]
    library.omega_recurrent_workspace_bytes.restype = ctypes.c_size_t
    library.omega_recurrent_forward.argtypes = [
        config_pointer, params_pointer, _FloatPointer, _FloatPointer,
        _FloatPointer, _FloatPointer, ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.omega_recurrent_forward.restype = ctypes.c_int
    library.omega_recurrent_backward.argtypes = [
        config_pointer, params_pointer, _FloatPointer, _FloatPointer,
        _FloatPointer, ctypes.c_void_p, ctypes.c_size_t, grads_pointer,
    ]
    library.omega_recurrent_backward.restype = ctypes.c_int
    try:
        profile_set_enabled = library.omega_recurrent_profile_set_enabled
        profile_reset = library.omega_recurrent_profile_reset
        profile_snapshot = library.omega_recurrent_profile_snapshot
    except AttributeError:
        library._omega_profile_available = False
    else:
        profile_set_enabled.argtypes = [ctypes.c_int]
        profile_set_enabled.restype = None
        profile_reset.argtypes = []
        profile_reset.restype = None
        profile_snapshot.argtypes = [ctypes.POINTER(_OmegaRecurrentProfileSnapshot)]
        profile_snapshot.restype = ctypes.c_int
        library._omega_profile_available = True
    return library


def configure_library(path: str | os.PathLike[str]) -> None:
    global _LIBRARY, _LOADED_LIBRARY_PATH
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    _LIBRARY = _load_library(resolved)
    _LOADED_LIBRARY_PATH = resolved


def loaded_library_path() -> Path:
    """Return exact DLL path loaded by this process, loading one if needed."""
    global _LOADED_LIBRARY_PATH
    if _LIBRARY is None:
        _library()
    if _LOADED_LIBRARY_PATH is None:
        raise RuntimeError("native library loaded without a path")
    return _LOADED_LIBRARY_PATH


def profile_abi_available() -> bool:
    return bool(getattr(_library(), "_omega_profile_available", False))


def _library() -> ctypes.CDLL:
    global _LIBRARY, _LOADED_LIBRARY_PATH
    if _LIBRARY is None:
        for candidate in _candidate_library_paths():
            if candidate.is_file():
                resolved = candidate.resolve()
                _LIBRARY = _load_library(resolved)
                _LOADED_LIBRARY_PATH = resolved
                break
    if _LIBRARY is None:
        searched = ", ".join(str(path) for path in _candidate_library_paths())
        raise FileNotFoundError(f"omega_recurrent.dll not found; searched: {searched}")
    return _LIBRARY


def _validate_tensor(value: Tensor, name: str, shape: Sequence[int], *, contiguous: bool = True) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a CPU float32 tensor")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous; production bridge will not copy it")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(shape)}")


def _validate_inputs(values: Sequence[Tensor], rounds: int) -> tuple[_OmegaRecurrentConfig, _OmegaRecurrentParams]:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    token_part, previous_state, *parameters = values
    batch, sequence, state_size = (int(value) for value in token_part.shape)
    if previous_state.ndim != 3:
        raise ValueError("previous_state must have rank 3")
    slots, dimension = (int(previous_state.shape[1]), int(previous_state.shape[2]))
    if state_size != slots * dimension:
        raise ValueError("token_part last dimension must equal slots * dimension")
    expected = (
        (batch, sequence, slots * dimension), (batch, slots, dimension),
        (slots * dimension, dimension), (slots * dimension,),
        (3 * dimension, dimension), (3 * dimension,), (dimension, dimension),
        (dimension,), (4 * dimension, dimension), (4 * dimension,),
        (dimension, 4 * dimension), (dimension,), (dimension,),
        (rounds, dimension), (rounds, dimension),
    )
    if len(parameters) != len(_PARAMETER_NAMES):
        raise ValueError("unexpected recurrent parameter count")
    for name, value, shape in zip(("token_part", "previous_state", *_PARAMETER_NAMES), values, expected):
        _validate_tensor(value, name, shape, contiguous=name != "state_part_weight")
    state_part_weight = parameters[0]
    if state_part_weight.stride(-1) != 1 or int(state_part_weight.stride(0)) < int(state_part_weight.shape[1]):
        raise ValueError("state_part_weight must have column stride 1 and row stride >= cols")
    state_view = _OmegaMatrixViewF32(
        _float_pointer(state_part_weight),
        int(state_part_weight.shape[0]),
        int(state_part_weight.shape[1]),
        int(state_part_weight.stride(0)),
    )
    config = _OmegaRecurrentConfig(sequence, batch, slots, dimension, rounds, 1, 0)
    params = _OmegaRecurrentParams(state_view, *(_float_pointer(value) for value in parameters[1:]))
    return config, params


def _workspace_key(config: _OmegaRecurrentConfig, workspace_bytes: int) -> tuple[int, ...]:
    return (
        int(config.sequence_length), int(config.batch), int(config.slots),
        int(config.dimension), int(config.rounds), int(config.training),
        int(config.instrumentation), workspace_bytes,
    )


def _acquire_workspace(key: tuple[int, ...], workspace_bytes: int) -> Tensor:
    global _ALLOCATIONS, _REUSES
    with _LOCK:
        available = _POOL.get(key)
        if available:
            _REUSES += 1
            return available.pop()
        _ALLOCATIONS += 1
    return torch.empty(workspace_bytes, dtype=torch.uint8, device="cpu")


def _release_workspace(key: tuple[int, ...], workspace: Tensor) -> None:
    with _LOCK:
        _POOL.setdefault(key, []).append(workspace)


def reset_runtime_stats(*, clear_pool: bool = False) -> None:
    global _ALLOCATIONS, _REUSES
    global _FORWARD_C_ABI_SECONDS, _BACKWARD_C_ABI_SECONDS
    global _FORWARD_BOUNDARY_SECONDS, _BACKWARD_BOUNDARY_SECONDS
    with _LOCK:
        _ALLOCATIONS = 0
        _REUSES = 0
        _FORWARD_C_ABI_SECONDS = 0.0
        _BACKWARD_C_ABI_SECONDS = 0.0
        _FORWARD_BOUNDARY_SECONDS = 0.0
        _BACKWARD_BOUNDARY_SECONDS = 0.0
        if clear_pool:
            _POOL.clear()


def runtime_stats() -> dict[str, float | int]:
    with _LOCK:
        return {
            "workspace_allocations": _ALLOCATIONS,
            "workspace_reuses": _REUSES,
            "workspace_pool_entries": sum(len(items) for items in _POOL.values()),
            "native_forward_c_abi_seconds": _FORWARD_C_ABI_SECONDS,
            "native_backward_c_abi_seconds": _BACKWARD_C_ABI_SECONDS,
            "native_forward_bridge_boundary_seconds": _FORWARD_BOUNDARY_SECONDS,
            "native_backward_bridge_boundary_seconds": _BACKWARD_BOUNDARY_SECONDS,
        }


_PROFILE_STAGES = (
    ("qkv_projection", "qkv_projection_calls", "qkv_projection_seconds"),
    ("attention_scores_softmax_mixing", "attention_scores_softmax_mixing_calls", "attention_scores_softmax_mixing_seconds"),
    ("out_projection", "out_projection_calls", "out_projection_seconds"),
    ("fc1_gelu", "fc1_gelu_calls", "fc1_gelu_seconds"),
    ("fc2", "fc2_calls", "fc2_seconds"),
    ("rmsnorm_gates", "rmsnorm_gates_calls", "rmsnorm_gates_seconds"),
    ("state_prelude", "state_prelude_calls", "state_prelude_seconds"),
    ("depth_embedding_pairwise_reduction", "depth_embedding_pairwise_reduction_calls", "depth_embedding_pairwise_reduction_seconds"),
    ("depth_embedding_carry", "depth_embedding_carry_calls", "depth_embedding_carry_seconds"),
    ("history_buffer_reads_writes", "history_buffer_reads_writes_calls", "history_buffer_reads_writes_seconds"),
)


def set_profile_enabled(enabled: bool) -> None:
    """Enable temporary native stage profiling explicitly for this process."""
    library = _library()
    if not getattr(library, "_omega_profile_available", False):
        raise RuntimeError("native library lacks temporary profile ABI; rebuild with OMEGA_PROFILE_INTERNAL=ON")
    library.omega_recurrent_profile_set_enabled(1 if enabled else 0)


def reset_profile() -> None:
    """Reset native profile totals without changing enabled state."""
    library = _library()
    if not getattr(library, "_omega_profile_available", False):
        raise RuntimeError("native library lacks temporary profile ABI; rebuild with OMEGA_PROFILE_INTERNAL=ON")
    library.omega_recurrent_profile_reset()


def _profile_direction_stats(value: _OmegaRecurrentProfileDirection) -> dict[str, object]:
    return {
        "calls": int(value.calls),
        "stages": {
            name: {"calls": int(getattr(value, calls_name)), "seconds": float(getattr(value, seconds_name))}
            for name, calls_name, seconds_name in _PROFILE_STAGES
        },
    }


def profile_stats() -> dict[str, object]:
    """Return native profile totals, including stage calls and monotonic seconds."""
    library = _library()
    if not getattr(library, "_omega_profile_available", False):
        raise RuntimeError("native library lacks temporary profile ABI; rebuild with OMEGA_PROFILE_INTERNAL=ON")
    snapshot = _OmegaRecurrentProfileSnapshot()
    status = int(library.omega_recurrent_profile_snapshot(ctypes.byref(snapshot)))
    if status != 0:
        raise RuntimeError(f"omega_recurrent_profile_snapshot failed with status {status}")
    return {
        "enabled": bool(snapshot.enabled),
        "compiled": bool(snapshot.compiled),
        "forward": _profile_direction_stats(snapshot.forward),
        "backward": _profile_direction_stats(snapshot.backward),
    }


def _raise_status(operation: str, status: int) -> None:
    if status != 0:
        raise RuntimeError(f"omega_recurrent_{operation} failed with status {status}")


class OmegaRecurrentFunction(torch.autograd.Function):
    """Autograd-only clean FFI boundary; native code owns recurrent equations."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, *args: object) -> tuple[Tensor, Tensor]:
        boundary_started = time.perf_counter()
        if len(args) != 16:
            raise TypeError("OmegaRecurrentFunction expects 15 tensors and rounds")
        tensor_args = tuple(args[:-1])
        rounds = args[-1]
        if not isinstance(rounds, int):
            raise TypeError("rounds must be an integer")
        config, params = _validate_inputs(tensor_args, rounds)  # type: ignore[arg-type]
        library = _library()
        workspace_bytes = int(library.omega_recurrent_workspace_bytes(config))
        if workspace_bytes <= 0:
            raise RuntimeError("omega_recurrent_workspace_bytes returned zero")
        key = _workspace_key(config, workspace_bytes)
        workspace = _acquire_workspace(key, workspace_bytes)
        token_part, previous_state = tensor_args[:2]  # type: ignore[assignment]
        next_state = torch.empty_like(previous_state)
        readout_states = torch.empty(
            (config.batch, config.sequence_length, config.slots, config.dimension),
            dtype=torch.float32,
            device="cpu",
        )
        started = time.perf_counter()
        try:
            status = library.omega_recurrent_forward(
                ctypes.byref(config), ctypes.byref(params), _float_pointer(token_part),
                _float_pointer(previous_state), _float_pointer(next_state),
                _float_pointer(readout_states), ctypes.c_void_p(int(workspace.data_ptr())),
                workspace_bytes,
            )
            _raise_status("forward", int(status))
        except BaseException:
            _release_workspace(key, workspace)
            raise
        global _FORWARD_C_ABI_SECONDS, _FORWARD_BOUNDARY_SECONDS
        with _LOCK:
            _FORWARD_C_ABI_SECONDS += time.perf_counter() - started
        ctx.save_for_backward(*tensor_args, workspace)  # type: ignore[arg-type]
        ctx.config = config
        # Finalization, not backward return, defines lease completion. This is
        # safe when callers retain the autograd graph for another backward.
        try:
            weakref.finalize(ctx, _release_workspace, key, workspace)
        except TypeError:
            # Older torch contexts may not support weak references. Retain the
            # lease rather than risk reusing workspace while graph is alive.
            ctx._workspace_lease = (key, workspace)
        with _LOCK:
            _FORWARD_BOUNDARY_SECONDS += time.perf_counter() - boundary_started
        return next_state, readout_states

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, d_next_state: Tensor | None, d_readout_states: Tensor | None) -> tuple[Tensor | None, ...]:
        boundary_started = time.perf_counter()
        saved = ctx.saved_tensors
        tensor_args = saved[:-1]
        workspace = saved[-1]
        config = ctx.config
        library = _library()
        token_part, previous_state = tensor_args[:2]
        parameters = tensor_args[2:]
        upstream_next = torch.zeros_like(previous_state) if d_next_state is None else d_next_state.contiguous()
        upstream_readout = torch.zeros(
            (config.batch, config.sequence_length, config.slots, config.dimension),
            dtype=torch.float32,
            device="cpu",
        ) if d_readout_states is None else d_readout_states.contiguous()
        gradients = tuple(torch.empty(tuple(value.shape), dtype=value.dtype, device=value.device) for value in tensor_args)
        null_float = _FloatPointer()
        null_count = ctypes.POINTER(ctypes.c_size_t)()
        grads = _OmegaRecurrentGrads(
            *(_float_pointer(value) for value in gradients),
            *(null_float for _ in range(15)),
            *(null_count for _ in range(15)),
            ctypes.POINTER(ctypes.c_double)(), null_count,
        )
        state_part_weight = parameters[0]
        params = _OmegaRecurrentParams(
            _OmegaMatrixViewF32(
                _float_pointer(state_part_weight),
                int(state_part_weight.shape[0]),
                int(state_part_weight.shape[1]),
                int(state_part_weight.stride(0)),
            ),
            *(_float_pointer(value) for value in parameters[1:]),
        )
        started = time.perf_counter()
        status = library.omega_recurrent_backward(
            ctypes.byref(config), ctypes.byref(params), _float_pointer(token_part),
            _float_pointer(upstream_readout), _float_pointer(upstream_next),
            ctypes.c_void_p(int(workspace.data_ptr())), int(workspace.numel()),
            ctypes.byref(grads),
        )
        _raise_status("backward", int(status))
        global _BACKWARD_C_ABI_SECONDS, _BACKWARD_BOUNDARY_SECONDS
        with _LOCK:
            _BACKWARD_C_ABI_SECONDS += time.perf_counter() - started
            _BACKWARD_BOUNDARY_SECONDS += time.perf_counter() - boundary_started
        return (*gradients, None)


def apply(
    token_part: Tensor,
    previous_state: Tensor,
    state_part_weight: Tensor,
    prelude_norm_weight: Tensor,
    block_qkv_weight: Tensor,
    block_qkv_bias: Tensor,
    block_out_weight: Tensor,
    block_out_bias: Tensor,
    block_fc1_weight: Tensor,
    block_fc1_bias: Tensor,
    block_fc2_weight: Tensor,
    block_fc2_bias: Tensor,
    block_norm_weight: Tensor,
    depth_embedding: Tensor,
    gate_logits: Tensor,
    rounds: int,
) -> tuple[Tensor, Tensor]:
    return OmegaRecurrentFunction.apply(
        token_part, previous_state, state_part_weight, prelude_norm_weight,
        block_qkv_weight, block_qkv_bias, block_out_weight, block_out_bias,
        block_fc1_weight, block_fc1_bias, block_fc2_weight, block_fc2_bias,
        block_norm_weight, depth_embedding, gate_logits, rounds,
    )
