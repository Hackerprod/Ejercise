"""V2-0 contractual core adapters for CUDA; equations and parameterization stay in V2-0."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Callable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F
from torch import Tensor

V20_ROOT = Path(__file__).resolve().parents[1] / "omega_v2_0_conformance"
if str(V20_ROOT) not in sys.path:
    sys.path.insert(0, str(V20_ROOT))

from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES, configure_reference_execution, rms_norm  # noqa: E402


GPU_OPERATION_ORDER = (
    "rms_norm_attention_input",
    "query_projection",
    "key_projection",
    "value_projection",
    "attention_scores_qk",
    "attention_scores_scaled",
    "attention_softmax",
    "attention_av",
    "attention_output_projection",
    "attention_residual",
    "rms_norm_mlp_input",
    "gate_projection",
    "up_projection",
    "gate_silu",
    "swiglu_hadamard",
    "down_projection",
    "mlp_residual_output",
)


def configure_v2_2a_execution() -> None:
    """Apply MD/318 deterministic FP32 CUDA settings before device initialization."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was sealed")
    torch.use_deterministic_algorithms(True)
    torch.set_autocast_enabled("cuda", False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda_step(
    block: ContractualCoreBlock,
    state: Tensor,
    *,
    return_operations: bool = False,
) -> Tensor | tuple[Tensor, OrderedDict[str, Tensor]]:
    """Run the V2-0 step equations on CUDA without adding parameters or state."""
    if state.ndim != 3 or int(state.shape[-1]) != block.d:
        raise ValueError(f"state must have shape [batch, m, {block.d}]")
    if state.device.type != "cuda" or state.dtype != torch.float32:
        raise ValueError("V2-2A CUDA step requires CUDA FP32 state")
    operations: OrderedDict[str, Tensor] = OrderedDict()

    normalized = rms_norm(state)
    operations["rms_norm_attention_input"] = normalized
    query = normalized @ block.W_Q.transpose(0, 1)
    operations["query_projection"] = query
    key = normalized @ block.W_K.transpose(0, 1)
    operations["key_projection"] = key
    value = normalized @ block.W_V.transpose(0, 1)
    operations["value_projection"] = value
    scores = query @ key.transpose(-2, -1)
    operations["attention_scores_qk"] = scores
    scores = scores / math.sqrt(block.d)
    operations["attention_scores_scaled"] = scores
    attention_probabilities = torch.softmax(scores, dim=-1)
    operations["attention_softmax"] = attention_probabilities
    attention = attention_probabilities @ value
    operations["attention_av"] = attention
    attention_output = attention @ block.W_O.transpose(0, 1)
    operations["attention_output_projection"] = attention_output
    hidden = state + attention_output
    operations["attention_residual"] = hidden

    mlp_input = rms_norm(hidden)
    operations["rms_norm_mlp_input"] = mlp_input
    gate = mlp_input @ block.W_gate.transpose(0, 1)
    operations["gate_projection"] = gate
    up = mlp_input @ block.W_up.transpose(0, 1)
    operations["up_projection"] = up
    activated_gate = F.silu(gate)
    operations["gate_silu"] = activated_gate
    product = activated_gate * up
    operations["swiglu_hadamard"] = product
    down = product @ block.W_down.transpose(0, 1)
    operations["down_projection"] = down
    output = hidden + down
    operations["mlp_residual_output"] = output
    return (output, operations) if return_operations else output


def cuda_rounds(
    block_for_round: Callable[[int], ContractualCoreBlock],
    state: Tensor,
    K: int,
    *,
    return_trace: bool = False,
    return_operation_traces: bool = False,
):
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise ValueError("K must be a positive runtime integer")
    current = state
    trace: list[Tensor] = []
    operation_traces: list[OrderedDict[str, Tensor]] = []
    for round_index in range(K):
        block = block_for_round(round_index)
        if return_operation_traces:
            current, operations = cuda_step(block, current, return_operations=True)
            operation_traces.append(operations)
        else:
            current = cuda_step(block, current)
        if return_trace:
            trace.append(current.clone())
    if return_operation_traces:
        return current, trace, operation_traces
    return (current, trace) if return_trace else current


def shared_cuda_rounds(block: ContractualCoreBlock, state: Tensor, K: int, **kwargs):
    return cuda_rounds(lambda _round_index: block, state, K, **kwargs)


def cpu_reference_rounds(block: ContractualCoreBlock, state: Tensor, K: int, *, return_trace: bool = False):
    if state.device.type != "cpu" or state.dtype != torch.float32:
        raise ValueError("V2-2A CPU reference requires CPU FP32 state")
    current = state
    trace: list[Tensor] = []
    for _ in range(K):
        current = block.step(current)
        if return_trace:
            trace.append(current.clone())
    return (current, trace) if return_trace else current
