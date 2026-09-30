"""V2-0 R4/U4 construction and fixed CPU-generated V2-2A tensors."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES
from omega_v2.variants import R4Shared, U4Untied, clone_value_and_storage_report


MASTER_SEED = 20260930
WEIGHT_SEED = 20260930
INPUT_SEED = 20261938
LOSS_W_SEED = 20262938
TARGET_SEED = 20263938
D = 256
M = 8
BATCH = 8


def make_cpu_normal(shape: tuple[int, ...], seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")


def copy_fixed_tensor_to_cuda(cpu_tensor: Tensor) -> tuple[Tensor, bool]:
    if cpu_tensor.device.type != "cpu" or cpu_tensor.dtype != torch.float32:
        raise ValueError("fixed V2-2A source tensors must be CPU FP32")
    cuda_tensor = cpu_tensor.to(device="cuda", dtype=torch.float32)
    exact_copy = torch.equal(cpu_tensor, cuda_tensor.cpu())
    return cuda_tensor, exact_copy


def make_fixed_inputs() -> dict[str, Any]:
    shape = (BATCH, M, D)
    x_cpu = make_cpu_normal(shape, INPUT_SEED)
    w_cpu = make_cpu_normal(shape, LOSS_W_SEED)
    target_cpu = make_cpu_normal(shape, TARGET_SEED)
    x_cuda, x_copy_exact = copy_fixed_tensor_to_cuda(x_cpu)
    w_cuda, w_copy_exact = copy_fixed_tensor_to_cuda(w_cpu)
    target_cuda, target_copy_exact = copy_fixed_tensor_to_cuda(target_cpu)
    return {
        "x_cpu": x_cpu,
        "x_cuda": x_cuda,
        "loss_weights_cpu": w_cpu,
        "loss_weights_cuda": w_cuda,
        "target_cpu": target_cpu,
        "target_cuda": target_cuda,
        "cpu_to_cuda_copy_bitwise": {
            "input": x_copy_exact,
            "loss_weights": w_copy_exact,
            "target": target_copy_exact,
        },
    }


def build_initial_variants(*, device: str = "cuda") -> dict[str, Any]:
    """Create V2-0 seeded R4 on CPU, clone U4 before any forward, then copy to device."""
    if device not in ("cpu", "cuda"):
        raise ValueError("V2-2A variants support CPU construction and CUDA execution only")
    r4_cpu = R4Shared.seeded(D, WEIGHT_SEED, dtype=torch.float32)
    u4_cpu = U4Untied.from_shared(r4_cpu)
    initial_report = clone_value_and_storage_report(r4_cpu, u4_cpu)
    if initial_report["initial_values_bitwise_equal"] is not True:
        raise RuntimeError("U4 initialization did not clone R4 bitwise")
    if device == "cpu":
        r4 = r4_cpu
        u4 = u4_cpu
        weight_copies_exact = True
    else:
        r4 = copy.deepcopy(r4_cpu).to(device="cuda")
        u4 = copy.deepcopy(u4_cpu).to(device="cuda")
        r4_block = r4.recurrent.block
        u4_blocks = list(u4.blocks)
        weight_copies_exact = all(
            torch.equal(getattr(r4_cpu.recurrent.block, family), getattr(r4_block, family).detach().cpu())
            for family in MATRIX_FAMILIES
        ) and all(
            torch.equal(getattr(u4_cpu.blocks[index], family), getattr(u4_blocks[index], family).detach().cpu())
            for index in range(4)
            for family in MATRIX_FAMILIES
        )
    return {
        "r4_cpu": r4_cpu,
        "u4_cpu": u4_cpu,
        "r4": r4,
        "u4": u4,
        "initial_clone_report": initial_report,
        "cpu_to_cuda_weight_copies_bitwise_equal": weight_copies_exact,
    }


def r4_block(variant: R4Shared) -> ContractualCoreBlock:
    return variant.recurrent.block


def u4_block(variant: U4Untied, round_index: int) -> ContractualCoreBlock:
    return variant.block_for_round(round_index)


def family_parameters(block: ContractualCoreBlock) -> dict[str, torch.nn.Parameter]:
    return {name: getattr(block, name) for name in MATRIX_FAMILIES}


def weight_copy_report(r4_cpu: R4Shared, u4_cpu: U4Untied, r4_cuda: R4Shared, u4_cuda: U4Untied) -> dict[str, Any]:
    rows = []
    for family in MATRIX_FAMILIES:
        cpu_value = getattr(r4_cpu.recurrent.block, family)
        cuda_value = getattr(r4_cuda.recurrent.block, family)
        r4_exact = torch.equal(cpu_value, cuda_value.detach().cpu())
        u4_exact = all(
            torch.equal(getattr(u4_cpu.blocks[index], family), getattr(u4_cuda.blocks[index], family).detach().cpu())
            for index in range(4)
        )
        rows.append({"family": family, "R4_cpu_to_cuda_bitwise_equal": r4_exact, "U4_cpu_to_cuda_bitwise_equal": u4_exact})
    return {"families": rows, "all_bitwise_equal": all(row["R4_cpu_to_cuda_bitwise_equal"] and row["U4_cpu_to_cuda_bitwise_equal"] for row in rows)}
