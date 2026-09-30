"""Analytic D512 FLOP ledger cross-check using the V2-0 counter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omega_v2.core import MATRIX_FAMILIES
from omega_v2.ledger import count_one_round_macs

from .config import (
    D,
    EXPECTED_D512_FLOPS,
    EXPECTED_FAMILY_COUNT,
    EXPECTED_R4_UNIQUE_PARAMS,
    EXPECTED_U4_UNIQUE_PARAMS,
    K4,
    M,
)


@dataclass(frozen=True)
class _ShapeOnlyTensor:
    shape: tuple[int, int]

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]


class _ShapeOnlyCore:
    def __init__(self, dimension: int):
        shapes = {
            "W_Q": (dimension, dimension),
            "W_K": (dimension, dimension),
            "W_V": (dimension, dimension),
            "W_O": (dimension, dimension),
            "W_gate": (4 * dimension, dimension),
            "W_up": (4 * dimension, dimension),
            "W_down": (dimension, 4 * dimension),
        }
        for family in MATRIX_FAMILIES:
            setattr(self, family, _ShapeOnlyTensor(shapes[family]))


def recompute_d512_ledger() -> dict[str, Any]:
    """Use V2-0's live-shape analytic counter without allocating or seeding weights."""
    r4 = _ShapeOnlyCore(D)
    u4 = [_ShapeOnlyCore(D) for _ in range(K4)]
    family_parameter_count = {family: getattr(r4, family).numel() for family in MATRIX_FAMILIES}
    r4_unique = sum(family_parameter_count.values())
    u4_unique = sum(getattr(block, family).numel() for block in u4 for family in MATRIX_FAMILIES)

    B1 = count_one_round_macs(r4, m=M, batch_size=1)
    B8 = count_one_round_macs(r4, m=M, batch_size=8)
    B128 = count_one_round_macs(r4, m=M, batch_size=128)
    values = {
        "per_round_B1": B1["forward_flops_per_round"],
        "K4_B1": K4 * B1["forward_flops_per_round"],
        "K4_B8": K4 * B8["forward_flops_per_round"],
        "K4_B128": K4 * B128["forward_flops_per_round"],
    }
    r4_non_gemm = {name: K4 * value for name, value in B8["non_gemm_per_round"].items()}
    u4_non_gemm = {
        name: sum(count_one_round_macs(block, m=M, batch_size=8)["non_gemm_per_round"][name] for block in u4)
        for name in B8["non_gemm_per_round"]
    }
    r4_flops_k4_b8 = K4 * B8["forward_flops_per_round"]
    u4_flops_k4_b8 = sum(count_one_round_macs(block, m=M, batch_size=8)["forward_flops_per_round"] for block in u4)
    comparisons = {
        **{name: values[name] == expected for name, expected in EXPECTED_D512_FLOPS.items()},
        "R4_U4_K4_B8_flops_equal": r4_flops_k4_b8 == u4_flops_k4_b8,
        "R4_U4_non_gemm_counts_equal": r4_non_gemm == u4_non_gemm,
        "R4_unique_parameter_count_equal": r4_unique == EXPECTED_R4_UNIQUE_PARAMS,
        "U4_unique_parameter_count_equal": u4_unique == EXPECTED_U4_UNIQUE_PARAMS,
        "seven_families": len(MATRIX_FAMILIES) == EXPECTED_FAMILY_COUNT,
    }
    return {
        "schema": "omega-v2-2b-d512-ledger-crosscheck-v1",
        "dimension": D,
        "m": M,
        "family_parameters": family_parameter_count,
        "R4_unique_parameters": r4_unique,
        "U4_unique_parameters": u4_unique,
        "expected_flops": dict(EXPECTED_D512_FLOPS),
        "actual_flops": values,
        "R4_U4_K4_B8_flops": {"R4": r4_flops_k4_b8, "U4": u4_flops_k4_b8},
        "R4_non_gemm_counts_K4_B8": r4_non_gemm,
        "U4_non_gemm_counts_K4_B8": u4_non_gemm,
        "comparisons": comparisons,
        "exact_match": all(comparisons.values()),
        "terminal_on_mismatch": "FLOP_LEDGER_PRESEAL_HOLD",
    }
