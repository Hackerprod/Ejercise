"""D3 CUDA/CPU gradient extraction built on frozen V2-0/V2-2A equations."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor

CAMPAIGN_ROOT = Path(__file__).resolve().parent.parent
V20_ROOT = CAMPAIGN_ROOT / "omega_v2_0_conformance"
V22A_ROOT = CAMPAIGN_ROOT / "omega_v2_2a_cuda_preflight"
for _path in (V20_ROOT, V22A_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES, SharedRecurrentCore  # noqa: E402
from omega_v2.ledger import state_dict_sha256  # noqa: E402
from omega_v2.variants import R4Shared, U4Untied  # noqa: E402
from omega_v2_2a_cuda_preflight.core import configure_v2_2a_execution, cuda_rounds, shared_cuda_rounds  # noqa: E402
from omega_v2_2a_cuda_preflight.variants import (  # noqa: E402
    BATCH,
    D,
    INPUT_SEED,
    LOSS_W_SEED,
    M,
    MASTER_SEED,
    WEIGHT_SEED,
    build_initial_variants,
    make_cpu_normal,
    weight_copy_report,
)
from .metrics import sum_u4_variants


def configure_v2_2a_diagnostic_execution() -> None:
    configure_v2_2a_execution()


def create_cuda_d3_run(reference_run: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a fresh deterministic D3 graph from the V2-2A-r1 CPU initialization."""
    pair = build_initial_variants(device="cuda")
    x_cpu = make_cpu_normal((BATCH, M, D), INPUT_SEED)
    w_cpu = make_cpu_normal((BATCH, M, D), LOSS_W_SEED)
    initial_state_cpu = {
        "r4": {family: getattr(pair["r4_cpu"].recurrent.block, family).detach().clone() for family in MATRIX_FAMILIES},
        "u4": [
            {family: getattr(pair["u4_cpu"].blocks[round_index], family).detach().clone() for family in MATRIX_FAMILIES}
            for round_index in range(4)
        ],
    }
    comparison_to_run_01 = None
    if reference_run is not None:
        reference_state = reference_run["initial_state_cpu"]
        comparison_to_run_01 = {
            "R4_initial_weights_bitwise_equal": all(torch.equal(initial_state_cpu["r4"][family], reference_state["r4"][family]) for family in MATRIX_FAMILIES),
            "U4_initial_weights_bitwise_equal": all(
                torch.equal(initial_state_cpu["u4"][round_index][family], reference_state["u4"][round_index][family])
                for round_index in range(4)
                for family in MATRIX_FAMILIES
            ),
            "x_bitwise_equal": torch.equal(x_cpu, reference_run["x_cpu"]),
            "w_bitwise_equal": torch.equal(w_cpu, reference_run["w_cpu"]),
        }
        comparison_to_run_01["all_bitwise_equal"] = all(comparison_to_run_01.values())
        if not comparison_to_run_01["all_bitwise_equal"]:
            raise RuntimeError("run 2 initial weights or fixed tensors differ from run 1 before forward")
    x_cuda = x_cpu.to(device="cuda", dtype=torch.float32)
    w_cuda = w_cpu.to(device="cuda", dtype=torch.float32)
    input_copy_equal = torch.equal(x_cpu, x_cuda.cpu())
    loss_weight_copy_equal = torch.equal(w_cpu, w_cuda.cpu())
    weights_copy = weight_copy_report(pair["r4_cpu"], pair["u4_cpu"], pair["r4"], pair["u4"])

    r4 = pair["r4"]
    u4 = pair["u4"]
    r4.zero_grad(set_to_none=True)
    u4.zero_grad(set_to_none=True)
    r4_output = shared_cuda_rounds(r4.recurrent.block, x_cuda, 4)
    r4_loss = torch.sum(r4_output * w_cuda)
    r4_grad_values = torch.autograd.grad(r4_loss, tuple(getattr(r4.recurrent.block, family) for family in MATRIX_FAMILIES))

    u4_output = cuda_rounds(u4.block_for_round, x_cuda, 4)
    u4_loss = torch.sum(u4_output * w_cuda)
    u4_parameters = [getattr(block, family) for block in u4.blocks for family in MATRIX_FAMILIES]
    u4_grad_values = torch.autograd.grad(u4_loss, tuple(u4_parameters))

    families: dict[str, dict[str, Tensor]] = {}
    for family_index, family in enumerate(MATRIX_FAMILIES):
        grads_u = [u4_grad_values[round_index * len(MATRIX_FAMILIES) + family_index].detach().clone() for round_index in range(4)]
        sums = sum_u4_variants(grads_u)
        families[family] = {"gR": r4_grad_values[family_index].detach().clone(), **{f"gU{index}": value for index, value in enumerate(grads_u)}, **sums}

    cpu_bundle = {
        family: {name: tensor.detach().cpu().contiguous() for name, tensor in values.items()}
        for family, values in families.items()
    }
    cpu_bundle["__inputs__"] = {"x_fp32_cpu": x_cpu.detach().contiguous(), "w_fp32_cpu": w_cpu.detach().contiguous()}
    return {
        "families_cuda": families,
        "tensor_bundle_cpu": cpu_bundle,
        "initial_state_cpu": initial_state_cpu,
        "initial_comparison_to_run_01": comparison_to_run_01,
        "x_cpu": x_cpu,
        "w_cpu": w_cpu,
        "input_copy_bitwise_equal": input_copy_equal,
        "loss_weight_copy_bitwise_equal": loss_weight_copy_equal,
        "weight_copy_report": weights_copy,
        "initial_clone_report": pair["initial_clone_report"],
        "initial_weight_state_dict_sha256": state_dict_sha256(pair["r4_cpu"].recurrent.block),
        "fixed_seeds": {"WEIGHT_SEED": WEIGHT_SEED, "INPUT_SEED": INPUT_SEED, "LOSS_W_SEED": LOSS_W_SEED},
    }


def create_cpu_fp64_oracle(x_fp32: Tensor, w_fp32: Tensor) -> dict[str, Any]:
    """Promote the exact frozen FP32 tensors and evaluate the V2-0 CPU FP64 graph."""
    base = R4Shared.seeded(D, WEIGHT_SEED, dtype=torch.float32)
    base_u4 = U4Untied.from_shared(base)
    r4_block64 = ContractualCoreBlock(D, dtype=torch.float64, seed=WEIGHT_SEED)
    with torch.no_grad():
        for family in MATRIX_FAMILIES:
            getattr(r4_block64, family).copy_(getattr(base.recurrent.block, family).double())
    r4_64 = R4Shared(SharedRecurrentCore(r4_block64))
    u4_64 = U4Untied([ContractualCoreBlock(D, dtype=torch.float64, seed=WEIGHT_SEED) for _ in range(4)])
    with torch.no_grad():
        for round_index in range(4):
            for family in MATRIX_FAMILIES:
                getattr(u4_64.blocks[round_index], family).copy_(getattr(base_u4.blocks[round_index], family).double())

    r4_copy_exact = all(torch.equal(getattr(base.recurrent.block, family).double(), getattr(r4_64.recurrent.block, family)) for family in MATRIX_FAMILIES)
    u4_copy_exact = all(
        torch.equal(getattr(base_u4.blocks[round_index], family).double(), getattr(u4_64.blocks[round_index], family))
        for round_index in range(4)
        for family in MATRIX_FAMILIES
    )

    x64 = x_fp32.double()
    w64 = w_fp32.double()
    r4_output = x64
    for _ in range(4):
        r4_output = r4_64.recurrent.block.step(r4_output)
    r4_loss = torch.sum(r4_output * w64)
    r4_grad_values = torch.autograd.grad(r4_loss, tuple(getattr(r4_64.recurrent.block, family) for family in MATRIX_FAMILIES))

    u4_output = x64
    for block in u4_64.blocks:
        u4_output = block.step(u4_output)
    u4_loss = torch.sum(u4_output * w64)
    u4_parameters = [getattr(block, family) for block in u4_64.blocks for family in MATRIX_FAMILIES]
    u4_grad_values = torch.autograd.grad(u4_loss, tuple(u4_parameters))
    families: dict[str, dict[str, Tensor]] = {}
    for family_index, family in enumerate(MATRIX_FAMILIES):
        grads_u = [u4_grad_values[round_index * len(MATRIX_FAMILIES) + family_index].detach().clone() for round_index in range(4)]
        sum_u64 = (((grads_u[0] + grads_u[1]) + grads_u[2]) + grads_u[3])
        families[family] = {
            "gR64": r4_grad_values[family_index].detach().cpu().contiguous(),
            "gU0_64": grads_u[0].detach().cpu().contiguous(),
            "gU1_64": grads_u[1].detach().cpu().contiguous(),
            "gU2_64": grads_u[2].detach().cpu().contiguous(),
            "gU3_64": grads_u[3].detach().cpu().contiguous(),
            "S_fp64_cpu": sum_u64.detach().cpu().contiguous(),
        }
    families["__inputs__"] = {
        "x_fp64": x64.detach().cpu().contiguous(),
        "w_fp64": w64.detach().cpu().contiguous(),
    }
    return {
        "families": families,
        "x_fp64": x64.detach().cpu().contiguous(),
        "w_fp64": w64.detach().cpu().contiguous(),
        "fp32_to_fp64_initial_weights_bitwise_value_preserved": r4_copy_exact and u4_copy_exact,
        "fp32_to_fp64_x_exact_value_preserved": torch.equal(x_fp32.double(), x64),
        "fp32_to_fp64_w_exact_value_preserved": torch.equal(w_fp32.double(), w64),
    }
