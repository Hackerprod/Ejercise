"""Minimal torch.autograd.Function bridge for the P2-R0 recurrent DLL.

This module owns validation, storage, pointer marshalling, status translation,
and gradient return ordering.  Recurrent equations remain exclusively in the
native implementation.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor


HERE = Path(__file__).resolve().parent
_LIBRARY: ctypes.CDLL | None = None
_DLL_DIRECTORY_HANDLES: list[object] = []
_LAST_SUM_ABS_CONTRIBUTIONS: dict[str, Tensor] = {}
_LAST_CONTRIBUTION_COUNTS: dict[str, Tensor] = {}
_LAST_FP64_D_DEPTH_EMBEDDING: Tensor | None = None
_LAST_DEPTH_MAX_LEVEL: Tensor | None = None


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


_FloatPointer = ctypes.POINTER(ctypes.c_float)
_CountPointer = ctypes.POINTER(ctypes.c_size_t)
_DoublePointer = ctypes.POINTER(ctypes.c_double)


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
        ("count_d_token_part", _CountPointer),
        ("count_d_previous_state", _CountPointer),
        ("count_d_state_part_weight", _CountPointer),
        ("count_d_prelude_norm_weight", _CountPointer),
        ("count_d_block_qkv_weight", _CountPointer),
        ("count_d_block_qkv_bias", _CountPointer),
        ("count_d_block_out_weight", _CountPointer),
        ("count_d_block_out_bias", _CountPointer),
        ("count_d_block_fc1_weight", _CountPointer),
        ("count_d_block_fc1_bias", _CountPointer),
        ("count_d_block_fc2_weight", _CountPointer),
        ("count_d_block_fc2_bias", _CountPointer),
        ("count_d_block_norm_weight", _CountPointer),
        ("count_d_depth_embedding", _CountPointer),
        ("count_d_gate_logits", _CountPointer),
        ("fp64_d_depth_embedding", _DoublePointer),
        ("depth_max_level", _CountPointer),
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


def _count_pointer(value: Tensor) -> _CountPointer:
    return ctypes.cast(ctypes.c_void_p(int(value.data_ptr())), _CountPointer)


def _candidate_library_paths() -> Iterable[Path]:
    override = os.environ.get("OMEGA_RECURRENT_DLL")
    if override:
        yield Path(override)
    yield HERE / "omega_recurrent.dll"
    build_root = HERE / "native" / "build"
    if build_root.is_dir():
        yield from sorted(build_root.rglob("omega_recurrent.dll"))


def configure_library(path: str | os.PathLike[str]) -> None:
    """Set DLL path before the first Function invocation."""
    global _LIBRARY
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    _LIBRARY = _load_library(resolved)


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
        config_pointer,
        params_pointer,
        _FloatPointer,
        _FloatPointer,
        _FloatPointer,
        _FloatPointer,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.omega_recurrent_forward.restype = ctypes.c_int
    library.omega_recurrent_backward.argtypes = [
        config_pointer,
        params_pointer,
        _FloatPointer,
        _FloatPointer,
        _FloatPointer,
        ctypes.c_void_p,
        ctypes.c_size_t,
        grads_pointer,
    ]
    library.omega_recurrent_backward.restype = ctypes.c_int
    return library


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        for candidate in _candidate_library_paths():
            if candidate.is_file():
                _LIBRARY = _load_library(candidate.resolve())
                break
    if _LIBRARY is None:
        searched = ", ".join(str(path) for path in _candidate_library_paths())
        raise FileNotFoundError(f"omega_recurrent.dll not found; searched: {searched}")
    return _LIBRARY


def _validate_tensor(value: Tensor, name: str, shape: Sequence[int], *, contiguous: bool = True) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
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
        (batch, sequence, slots * dimension),
        (batch, slots, dimension),
        (slots * dimension, dimension),
        (slots * dimension,),
        (3 * dimension, dimension),
        (3 * dimension,),
        (dimension, dimension),
        (dimension,),
        (4 * dimension, dimension),
        (4 * dimension,),
        (dimension, 4 * dimension),
        (dimension,),
        (dimension,),
        (rounds, dimension),
        (rounds, dimension),
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
    config = _OmegaRecurrentConfig(sequence, batch, slots, dimension, rounds, 1, 1)
    params = _OmegaRecurrentParams(state_view, *(_float_pointer(value) for value in parameters[1:]))
    return config, params


def _raise_status(operation: str, status: int) -> None:
    if status != 0:
        raise RuntimeError(f"omega_recurrent_{operation} failed with status {status}")


def last_sum_abs_contributions() -> dict[str, Tensor]:
    """Return native per-element accumulation magnitudes from most recent backward."""
    return dict(_LAST_SUM_ABS_CONTRIBUTIONS)


def last_contribution_counts() -> dict[str, Tensor]:
    """Return native per-element contribution counts from most recent backward."""
    return dict(_LAST_CONTRIBUTION_COUNTS)


def last_fp64_d_depth_embedding() -> Tensor | None:
    """Return native FP64 reference accumulation from most recent backward."""
    return None if _LAST_FP64_D_DEPTH_EMBEDDING is None else _LAST_FP64_D_DEPTH_EMBEDDING.clone()


def last_depth_max_level() -> Tensor | None:
    """Return native maximum occupied carry level from most recent backward."""
    return None if _LAST_DEPTH_MAX_LEVEL is None else _LAST_DEPTH_MAX_LEVEL.clone()


class OmegaRecurrentFunction(torch.autograd.Function):
    """Autograd-only FFI boundary; native code owns all recurrent equations."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, *args: object) -> tuple[Tensor, Tensor]:
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
        token_part, previous_state = tensor_args[:2]  # type: ignore[assignment]
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cpu")
        next_state = torch.empty_like(previous_state)
        readout_states = torch.empty(
            (config.batch, config.sequence_length, config.slots, config.dimension),
            dtype=torch.float32,
            device="cpu",
        )
        status = library.omega_recurrent_forward(
            ctypes.byref(config),
            ctypes.byref(params),
            _float_pointer(token_part),
            _float_pointer(previous_state),
            _float_pointer(next_state),
            _float_pointer(readout_states),
            ctypes.c_void_p(int(workspace.data_ptr())),
            workspace_bytes,
        )
        _raise_status("forward", int(status))
        ctx.save_for_backward(*tensor_args, workspace)  # type: ignore[arg-type]
        ctx.config = config
        return next_state, readout_states

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, d_next_state: Tensor | None, d_readout_states: Tensor | None) -> tuple[Tensor | None, ...]:
        global _LAST_SUM_ABS_CONTRIBUTIONS, _LAST_CONTRIBUTION_COUNTS
        global _LAST_FP64_D_DEPTH_EMBEDDING, _LAST_DEPTH_MAX_LEVEL
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
        sum_abs_parameters = tuple(torch.empty(tuple(value.shape), dtype=value.dtype, device=value.device) for value in parameters)
        count_parameters = tuple(torch.empty(tuple(value.shape), dtype=torch.int64, device=value.device) for value in parameters)
        fp64_depth_embedding = torch.empty_like(parameters[-2], dtype=torch.float64)
        depth_max_level = torch.empty_like(parameters[-2], dtype=torch.int64)
        null_pointer = _FloatPointer()
        null_count_pointer = _CountPointer()
        grads = _OmegaRecurrentGrads(
            *(_float_pointer(value) for value in gradients),
            null_pointer,
            null_pointer,
            *(_float_pointer(value) for value in sum_abs_parameters),
            null_count_pointer,
            null_count_pointer,
            *(_count_pointer(value) for value in count_parameters),
            ctypes.cast(ctypes.c_void_p(int(fp64_depth_embedding.data_ptr())), _DoublePointer),
            ctypes.cast(ctypes.c_void_p(int(depth_max_level.data_ptr())), _CountPointer),
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
        status = library.omega_recurrent_backward(
            ctypes.byref(config),
            ctypes.byref(params),
            _float_pointer(token_part),
            _float_pointer(upstream_readout),
            _float_pointer(upstream_next),
            ctypes.c_void_p(int(workspace.data_ptr())),
            int(workspace.numel()),
            ctypes.byref(grads),
        )
        _raise_status("backward", int(status))
        _LAST_SUM_ABS_CONTRIBUTIONS = {
            name: value.detach().clone()
            for name, value in zip(_PARAMETER_NAMES, sum_abs_parameters)
        }
        _LAST_CONTRIBUTION_COUNTS = {
            name: value.detach().clone()
            for name, value in zip(_PARAMETER_NAMES, count_parameters)
        }
        _LAST_FP64_D_DEPTH_EMBEDDING = fp64_depth_embedding.detach().clone()
        _LAST_DEPTH_MAX_LEVEL = depth_max_level.detach().clone()
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
        token_part,
        previous_state,
        state_part_weight,
        prelude_norm_weight,
        block_qkv_weight,
        block_qkv_bias,
        block_out_weight,
        block_out_bias,
        block_fc1_weight,
        block_fc1_bias,
        block_fc2_weight,
        block_fc2_bias,
        block_norm_weight,
        depth_embedding,
        gate_logits,
        rounds,
    )
