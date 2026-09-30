"""Frozen D3Q dimensions and explicitly separated calibration/official seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


D = 256
M = 8
BATCH = 8
K = 4

CALIBRATION_SEED = 20260930
CALIBRATION_INPUT_SEED = CALIBRATION_SEED + 1008
CALIBRATION_LOSS_W_SEED = CALIBRATION_SEED + 2008

OFFICIAL_MASTER_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005)

SeedMode = Literal["qa", "calibration_smoke", "official"]


@dataclass(frozen=True)
class SeedPlan:
    mode: SeedMode
    master_seed: int
    weight_seed: int
    input_seed: int
    loss_w_seed: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def make_seed_plan(master_seed: int, *, mode: SeedMode) -> SeedPlan:
    """Validate mode before deriving any data/weight seed values."""
    if mode in ("qa", "calibration_smoke"):
        if master_seed != CALIBRATION_SEED:
            raise ValueError("D3Q_QA_SEED_REJECTED: QA/smoke accepts only calibration seed 20260930")
        return SeedPlan(
            mode=mode,
            master_seed=CALIBRATION_SEED,
            weight_seed=CALIBRATION_SEED,
            input_seed=CALIBRATION_INPUT_SEED,
            loss_w_seed=CALIBRATION_LOSS_W_SEED,
        )
    if mode != "official":
        raise ValueError(f"unsupported D3Q seed mode: {mode}")
    if master_seed == CALIBRATION_SEED:
        raise ValueError("D3Q_CALIBRATION_SEED_FORBIDDEN_FROM_OFFICIAL_VERDICT")
    if master_seed not in OFFICIAL_MASTER_SEEDS:
        raise ValueError("D3Q_OFFICIAL_SEED_NOT_IN_FROZEN_HELD_OUT_SET")
    return SeedPlan(
        mode="official",
        master_seed=master_seed,
        weight_seed=master_seed,
        input_seed=master_seed + 1008,
        loss_w_seed=master_seed + 2008,
    )


def official_seed_plans() -> tuple[SeedPlan, ...]:
    """Return the five official plans; call only after official GO and seal checks."""
    return tuple(make_seed_plan(seed, mode="official") for seed in OFFICIAL_MASTER_SEEDS)


@dataclass(frozen=True)
class DryRunSeedSlot:
    """Opaque control-flow slot used only by stubbed QA; carries no numeric seed."""

    case_id: str
