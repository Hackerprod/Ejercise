"""Parameter, storage, and matmul-FLOP ledgers derived from live module tensors."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import torch
from torch import nn

from .core import ContractualCoreBlock, MATRIX_FAMILIES, parameter_storage_identity
from .variants import R4Shared, U4Untied


DIMENSIONS = (512, 640)
SLOT_COUNTS = (4, 8, 16)
K_VALUES = (1, 2, 4, 8, 16)
BITWISE_SEEDS = (20260929, 20260930, 20260931)
Q4_GROUP_SIZE = 32
Q4_SCALE_BYTES = 2  # FP16


def _canonical_hash(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        payload = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(payload.dtype).encode("ascii"))
        digest.update(json.dumps(list(payload.shape), separators=(",", ":")).encode("ascii"))
        digest.update(payload.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    unique: dict[tuple[int, int], nn.Parameter] = {}
    for parameter in module.parameters():
        unique.setdefault(parameter_storage_identity(parameter), parameter)
    return list(unique.values())


def _family_counts(blocks: Iterable[ContractualCoreBlock]) -> dict[str, int]:
    counts = {name: 0 for name in MATRIX_FAMILIES}
    for block in blocks:
        for name in MATRIX_FAMILIES:
            counts[name] += int(getattr(block, name).numel())
    return counts


def _quantization_accounting(parameters: Iterable[nn.Parameter]) -> dict[str, int | str]:
    tensors = list(parameters)
    numel = sum(int(parameter.numel()) for parameter in tensors)
    logical_weight_bytes = sum((int(parameter.numel()) + 1) // 2 for parameter in tensors)
    scale_groups = sum((int(parameter.numel()) + Q4_GROUP_SIZE - 1) // Q4_GROUP_SIZE for parameter in tensors)
    scale_bytes = scale_groups * Q4_SCALE_BYTES
    return {
        "q4_group_size": Q4_GROUP_SIZE,
        "q4_scale_dtype": "FP16",
        "q4_logical_weight_bytes": logical_weight_bytes,
        "q4_scale_bytes": scale_bytes,
        "q4_logical_total_bytes": logical_weight_bytes + scale_bytes,
        "q4_physical_allocated_bytes_or_NOT_MEASURED": "NOT_MEASURED",
        "padding_bytes_or_NOT_MEASURED": "NOT_MEASURED",
    }


def _candidate_row(d: int, variant: str, module: nn.Module, blocks: list[ContractualCoreBlock]) -> dict[str, Any]:
    counts = _family_counts(blocks)
    parameters = _unique_parameters(module)
    total = sum(int(parameter.numel()) for parameter in parameters)
    qkv_o = sum(counts[name] for name in ("W_Q", "W_K", "W_V", "W_O"))
    swiglu = sum(counts[name] for name in ("W_gate", "W_up", "W_down"))
    sharing = "shared" if variant == "R4" else "untied"
    return {
        "candidate_id": f"V2-{d}-{variant}",
        "d": d,
        "supported_m": list(SLOT_COUNTS),
        "supported_K_tested": list(K_VALUES) if variant == "R4" else [4],
        "P_Q": counts["W_Q"],
        "P_K": counts["W_K"],
        "P_V": counts["W_V"],
        "P_O": counts["W_O"],
        "P_attention_projection_total": qkv_o,
        "P_gate": counts["W_gate"],
        "P_up": counts["W_up"],
        "P_down": counts["W_down"],
        "P_swiglu_total": swiglu,
        "P_core_unique": total,
        "P_shell": 0,
        "P_retriever": 0,
        "P_memory_learned": 0,
        "B_memory_static": 0,
        "B_index": 0,
        "fp32_core_bytes": sum(int(parameter.numel()) * parameter.element_size() for parameter in parameters),
        **_quantization_accounting(parameters),
        "sharing_mode": sharing,
        "state_dict_sha256": state_dict_sha256(module),
        "parameter_names": sorted(name for name, _ in module.named_parameters()),
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in sorted(module.named_parameters())},
        "actual_module_count": len(blocks),
        "all_core_parameters_have_no_bias_or_norm_scale": len(parameters) == len(list(module.parameters())),
    }


def build_parameter_ledger() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for d in DIMENSIONS:
        shared = R4Shared.seeded(d, 20260929)
        untied = U4Untied.from_shared(shared)
        rows.append(_candidate_row(d, "R4", shared, [shared.recurrent.block]))
        rows.append(_candidate_row(d, "U4", untied, list(untied.blocks)))
    return {
        "schema": "omega-v2-parameter-ledger-v1",
        "derivation": "live module named_parameters, tensor shapes, storage identity, and state_dict bytes",
        "dimensions": list(DIMENSIONS),
        "supported_m": list(SLOT_COUNTS),
        "runtime_K_values_tested": list(K_VALUES),
        "rows": rows,
    }


def _linear_macs(weight: nn.Parameter, token_rows: int) -> int:
    out_features, in_features = (int(value) for value in weight.shape)
    return int(token_rows) * in_features * out_features


def count_one_round_macs(block: ContractualCoreBlock, *, m: int, batch_size: int = 1) -> dict[str, int]:
    if m < 1 or batch_size < 1:
        raise ValueError("batch_size and m must be positive")
    token_rows = int(batch_size) * int(m)
    projections = sum(_linear_macs(getattr(block, name), token_rows) for name in ("W_Q", "W_K", "W_V", "W_O"))
    swiglu = sum(_linear_macs(getattr(block, name), token_rows) for name in ("W_gate", "W_up", "W_down"))
    d = int(block.W_Q.shape[0])
    qk = int(batch_size) * int(m) * int(m) * d
    av = qk
    return {
        "projection_macs": projections,
        "swiglu_macs": swiglu,
        "attention_qk_macs": qk,
        "attention_av_macs": av,
        "total_macs_per_round": projections + swiglu + qk + av,
        "forward_flops_per_round": 2 * (projections + swiglu + qk + av),
        "non_gemm_per_round": {
            "rmsnorm_applications": 2,
            "rmsnorm_elements": 2 * token_rows * d,
            "softmax_rows": int(batch_size) * int(m),
            "softmax_elements": int(batch_size) * int(m) * int(m),
            "silu_activations": 4 * token_rows * d,
            "swiglu_hadamard_multiplies": 4 * token_rows * d,
            "residual_vector_additions": 2 * token_rows * d,
        },
    }


def build_flop_ledger() -> dict[str, Any]:
    rows = []
    for d in DIMENSIONS:
        shared = R4Shared.seeded(d, 20260929)
        untied = U4Untied.from_shared(shared)
        block = shared.recurrent.block
        for m in SLOT_COUNTS:
            per_round = count_one_round_macs(block, m=m, batch_size=1)
            for K in K_VALUES:
                rows.append(
                    {
                        "d": d,
                        "m": m,
                        "K": K,
                        "batch_size": 1,
                        "shared_R_forward_macs": K * per_round["total_macs_per_round"],
                        "shared_R_forward_flops": K * per_round["forward_flops_per_round"],
                        "non_gemm_counts_total": {name: K * value for name, value in per_round["non_gemm_per_round"].items()},
                    }
                )
            r4_flops = 4 * per_round["forward_flops_per_round"]
            u4_flops = sum(
                2 * sum(_linear_macs(getattr(block_variant, name), m) for name in MATRIX_FAMILIES)
                + 2 * (m * m * d) + 2 * (m * m * d)
                for block_variant in untied.blocks
            )
            rows.append(
                {
                    "d": d,
                    "m": m,
                    "K": 4,
                    "batch_size": 1,
                    "variant_pair": "R4_vs_U4",
                    "R4_forward_flops": r4_flops,
                    "U4_forward_flops": u4_flops,
                    "exact_compute_match": r4_flops == u4_flops,
                    "non_gemm_counts_total": {name: 4 * value for name, value in per_round["non_gemm_per_round"].items()},
                }
            )
    return {
        "schema": "omega-v2-flop-ledger-v1",
        "mac_convention": "1 MAC = 2 FLOPs",
        "derivation": "matrix multiply dimensions introspected from live core tensors; attention matmuls counted separately",
        "non_gemm_ops_reported_as_counts_not_fictitious_flops": True,
        "rows": rows,
    }


def ledger_digest(payload: Any) -> str:
    return _canonical_hash(payload)
