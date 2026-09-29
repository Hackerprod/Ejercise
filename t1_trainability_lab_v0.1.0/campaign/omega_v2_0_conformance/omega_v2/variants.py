"""Shared R4 and compute-matched untied U4 wrappers."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from .core import ContractualCoreBlock, SharedRecurrentCore, make_seeded_core, parameter_storage_identity


class R4Shared(nn.Module):
    """One physical block object reused for exactly four applications."""

    def __init__(self, recurrent: SharedRecurrentCore) -> None:
        super().__init__()
        self.recurrent = recurrent

    @classmethod
    def seeded(cls, d: int, seed: int, *, dtype: torch.dtype = torch.float32) -> "R4Shared":
        return cls(SharedRecurrentCore(make_seeded_core(d, seed, dtype=dtype)))

    def block_for_round(self, round_index: int) -> ContractualCoreBlock:
        if not 0 <= int(round_index) < 4:
            raise IndexError("R4 round index must be in [0,3]")
        return self.recurrent.block

    def forward(self, state: Tensor, *, return_trace: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        return self.recurrent(state, K=4, return_trace=return_trace)


class U4Untied(nn.Module):
    """Four independent blocks, initialized as bitwise clones of one R4 block."""

    def __init__(self, blocks: list[ContractualCoreBlock]) -> None:
        super().__init__()
        if len(blocks) != 4:
            raise ValueError("U4 must contain exactly four untied blocks")
        self.blocks = nn.ModuleList(blocks)

    @classmethod
    def from_shared(cls, shared: R4Shared) -> "U4Untied":
        return cls([copy.deepcopy(shared.recurrent.block) for _ in range(4)])

    @classmethod
    def seeded(cls, d: int, seed: int, *, dtype: torch.dtype = torch.float32) -> "U4Untied":
        return cls.from_shared(R4Shared.seeded(d, seed, dtype=dtype))

    def block_for_round(self, round_index: int) -> ContractualCoreBlock:
        return self.blocks[int(round_index)]

    def forward(self, state: Tensor, *, return_trace: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        current = state
        trace: list[Tensor] = []
        for block in self.blocks:
            current = block.step(current)
            if return_trace:
                trace.append(current.clone())
        return (current, trace) if return_trace else current


def clone_value_and_storage_report(shared: R4Shared, untied: U4Untied) -> dict[str, Any]:
    reference = shared.recurrent.block
    value_equal = all(
        torch.equal(getattr(reference, name), getattr(block, name))
        for block in untied.blocks
        for name in reference.state_dict()
    )
    shared_storages = {parameter_storage_identity(p) for p in reference.parameters()}
    untied_storages = [
        {parameter_storage_identity(p) for p in block.parameters()}
        for block in untied.blocks
    ]
    all_u4_disjoint = all(
        not (untied_storages[i] & untied_storages[j])
        for i in range(4)
        for j in range(i + 1, 4)
    )
    r4_reuses_one_object = len({id(shared.block_for_round(round_index)) for round_index in range(4)}) == 1
    u4_separate_objects = len({id(untied.block_for_round(round_index)) for round_index in range(4)}) == 4
    no_u4_storage_aliases_r4 = all(not (shared_storages & storages) for storages in untied_storages)
    return {
        "initial_values_bitwise_equal": value_equal,
        "r4_reuses_one_block_object": r4_reuses_one_object,
        "u4_has_four_block_objects": u4_separate_objects,
        "u4_block_storages_pairwise_disjoint": all_u4_disjoint,
        "u4_storage_disjoint_from_r4": no_u4_storage_aliases_r4,
    }


def fixed_input(d: int, m: int, seed: int, *, dtype: torch.dtype = torch.float32, batch: int = 1) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((batch, m, d), generator=generator, dtype=dtype, device="cpu")


def run_bitwise_acceptance(
    *,
    dimensions: tuple[int, ...] = (512, 640),
    slot_counts: tuple[int, ...] = (4, 8, 16),
    seeds: tuple[int, ...] = (20260929, 20260930, 20260931),
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows: list[dict[str, Any]] = []
    for d in dimensions:
        for seed in seeds:
            r4 = R4Shared.seeded(d, seed, dtype=torch.float32)
            u4 = U4Untied.from_shared(r4)
            clone_report = clone_value_and_storage_report(r4, u4)
            for m in slot_counts:
                sample = fixed_input(d, m, seed + 1000 + m, dtype=torch.float32)
                r4_output, r4_trace = r4(sample, return_trace=True)
                u4_output, u4_trace = u4(sample, return_trace=True)
                trace_equal = [torch.equal(left, right) for left, right in zip(r4_trace, u4_trace)]
                rows.append(
                    {
                        "d": d,
                        "m": m,
                        "K": 4,
                        "seed": seed,
                        **clone_report,
                        "round_trace_bitwise_equal": trace_equal,
                        "final_output_bitwise_equal": torch.equal(r4_output, u4_output),
                    }
                )
    checks = [
        "initial_values_bitwise_equal",
        "r4_reuses_one_block_object",
        "u4_has_four_block_objects",
        "u4_block_storages_pairwise_disjoint",
        "u4_storage_disjoint_from_r4",
        "final_output_bitwise_equal",
    ]
    all_pass = all(row[key] for row in rows for key in checks) and all(all(row["round_trace_bitwise_equal"]) for row in rows)
    return {
        "schema": "omega-v2-bitwise-report-v1",
        "backend": {"device": "CPU", "dtype": "FP32", "threads": 1, "deterministic_algorithms": True, "dropout": 0, "amp": False, "tf32": False},
        "dimensions": list(dimensions),
        "slot_counts": list(slot_counts),
        "seeds": list(seeds),
        "row_count": len(rows),
        "status": "PASS" if all_pass else "FAIL",
        "checks": rows,
    }
