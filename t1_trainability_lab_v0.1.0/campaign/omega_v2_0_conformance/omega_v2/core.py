"""Literal FP32/FP64 reference core for the OMEGA-V2-0 16d² contract."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


RMS_EPSILON = 1e-6
MATRIX_FAMILIES = ("W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down")


def configure_reference_execution() -> None:
    """Set the frozen CPU reference execution conditions."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def rms_norm(x: Tensor, *, eps: float = RMS_EPSILON) -> Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)


class ContractualCoreBlock(nn.Module):
    """One attention+SwiGLU block with exactly seven matrices and no other state."""

    def __init__(self, d: int, *, dtype: torch.dtype = torch.float32, seed: int = 20260929) -> None:
        super().__init__()
        if not isinstance(d, int) or isinstance(d, bool) or d < 1:
            raise ValueError("d must be a positive integer")
        self.d = d
        shapes = {
            "W_Q": (d, d),
            "W_K": (d, d),
            "W_V": (d, d),
            "W_O": (d, d),
            "W_gate": (4 * d, d),
            "W_up": (4 * d, d),
            "W_down": (d, 4 * d),
        }
        for name, shape in shapes.items():
            setattr(self, name, nn.Parameter(torch.empty(shape, dtype=dtype, device="cpu")))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for name in MATRIX_FAMILIES:
                nn.init.xavier_uniform_(getattr(self, name), generator=generator)

    def step(self, state: Tensor) -> Tensor:
        if state.ndim != 3 or int(state.shape[-1]) != self.d:
            raise ValueError(f"state must have shape [batch, m, {self.d}]")
        if state.device.type != "cpu":
            raise ValueError("V2-0 reference acceptance is CPU-only")

        normalized = rms_norm(state)
        query = normalized @ self.W_Q.transpose(0, 1)
        key = normalized @ self.W_K.transpose(0, 1)
        value = normalized @ self.W_V.transpose(0, 1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d)
        attention = torch.softmax(scores, dim=-1) @ value
        hidden = state + attention @ self.W_O.transpose(0, 1)

        mlp_input = rms_norm(hidden)
        gate = mlp_input @ self.W_gate.transpose(0, 1)
        up = mlp_input @ self.W_up.transpose(0, 1)
        mlp = (F.silu(gate) * up) @ self.W_down.transpose(0, 1)
        return hidden + mlp


class SharedRecurrentCore(nn.Module):
    """One block applied K times; K and m are runtime values, never parameters."""

    def __init__(self, block: ContractualCoreBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, state: Tensor, K: int, *, return_trace: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        if not isinstance(K, int) or isinstance(K, bool) or K < 1:
            raise ValueError("K must be a positive runtime integer")
        current = state
        trace: list[Tensor] = []
        for _ in range(K):
            current = self.block.step(current)
            if return_trace:
                trace.append(current.clone())
        return (current, trace) if return_trace else current


def make_seeded_core(d: int, seed: int, *, dtype: torch.dtype = torch.float32) -> ContractualCoreBlock:
    return ContractualCoreBlock(d, dtype=dtype, seed=seed)


def clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def state_dict_values_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    return left_state.keys() == right_state.keys() and all(torch.equal(left_state[key], right_state[key]) for key in left_state)


def tensor_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(int(dim) for dim in value.shape) for name, value in module.state_dict().items()}


def parameter_storage_identity(parameter: nn.Parameter) -> tuple[int, int]:
    storage = parameter.untyped_storage()
    return int(storage.data_ptr()), int(storage.nbytes())
