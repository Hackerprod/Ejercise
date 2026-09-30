"""Accepted MD/324–MD/325 constants; open operational choices are isolated here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from omega_v2_2a_d3q.metrics import GATE_A_LIMIT as D3_GATE_A_LIMIT
from omega_v2_2a_d3q.metrics import GATE_B_LIMIT as D3_GATE_B_LIMIT
from omega_v2_2a_d3q.metrics import GATE_C_ULPS as D3_GATE_C_ULPS
from omega_v2_2a_d3q.metrics import ORACLE_INF_LIMIT as D3_ORACLE_INF_LIMIT
from omega_v2_2a_d3q.metrics import ORACLE_L2_LIMIT as D3_ORACLE_L2_LIMIT


OFFICIAL_ID = "OMEGA-V2-2B-LOCAL-PILOT-01"
D = 512
M = 8
K4 = 4
CONFORMANCE_BATCH = 8
TRAINABILITY_BATCH = 128

CALIBRATION_SEED = 20260930
OFFICIAL_MASTER_SEEDS = (20261011, 20261012, 20261013)

# Q1 — accepted interpretation for the truncated D1 normwise expression.
D1_ELEMENTWISE_ABS_TOL = 1e-5
D1_ELEMENTWISE_REL_TOL = 1e-4
D1_ELEMENTWISE_FLOOR = 1e-6
D1_NORMWISE_E_L2_LIMIT = 1e-5
D1_NORMWISE_FLOOR = 1e-6
D1_K_VALUES = (1, 4)
D1_VARIANT = "R4"

# Q2 — immutable D3Q instruments.
D3_E_L2_LIMIT = D3_GATE_A_LIMIT
D3_E_INF_LIMIT = D3_GATE_B_LIMIT
D3_MAX_ABS_ULP_LIMIT = D3_GATE_C_ULPS
D3_ORACLE_E_L2_LIMIT = D3_ORACLE_L2_LIMIT
D3_ORACLE_E_INF_LIMIT = D3_ORACLE_INF_LIMIT

# Q3 — accepted D7 proposal.
D7_UPDATES = 20
D7_LR = 3e-4
D7_BETAS = (0.9, 0.999)
D7_EPS = 1e-8
D7_WEIGHT_DECAY = 0.0
D7_SCHEDULER = None
D7_GRADIENT_CLIPPING = None
D7_LOSS_REDUCTION = "mean"
D7_LOSS_FUNCTION = "mean((y-target)^2)"
D7_TARGET_SEED_OFFSET = 3008
D7_LOSS_W_USED = False
D7_FINITE_STATE_KEYS = ("output", "loss", "gradients", "parameters", "exp_avg", "exp_avg_sq")
D7_FINAL_LOSS_RATIO = 0.99

# Q4 — accepted D6 proposal.
D6_MASTER_SEED = OFFICIAL_MASTER_SEEDS[0]
D6_BATCH = CONFORMANCE_BATCH
D6_M = M
D6_FORWARD_K_VALUES = (1, 2, 4, 8, 16)
D6_BACKWARD_K_VALUES = (1, 4, 8, 16)
D6_LOSS_FUNCTION = "sum(y*w)"
D6_OPTIMIZER = None
D6_LOSS_W_OFFSET = 2008

# Q5 — accepted D8 per-cell and process limits.
D8_VRAM_BUDGET_BYTES = 3 * 1024**3
D8_WALL_LIMIT_SECONDS = 30 * 60
D8_CELLS = ("D1(seed,K)", "D2(seed)", "D3(seed)", "D6(mode,K)", "D7(seed,variant)")
D8_PRE_CELL_EMPTY_CACHE = True
D8_PRE_CELL_RESET_PEAK_STATS = True
D8_PRE_AND_POST_CELL_SYNCHRONIZE = True
D8_DESTROY_UNUSED_CUDA_OBJECTS = True
D8_GC_BEFORE_CELL = True

# Q6 — accepted terminal mapping.
TERMINAL_PASS = "OMEGA_V2_2B_LOCAL_PILOT_PASS"
TERMINAL_FAIL = "OMEGA_V2_2B_LOCAL_PILOT_FAIL"
TERMINAL_CAPACITY_HOLD = "OMEGA_V2_2B_LOCAL_CAPACITY_HOLD"
TERMINAL_CALIBRATION_QA_HOLD = "V2_2B_CALIBRATION_QA_HOLD"
TERMINAL_CALIBRATION_SMOKE_COMPLETE = "V2_2B_CALIBRATION_SMOKE_COMPLETE"
TERMINAL_CALIBRATION_QA_CAPACITY_HOLD = "V2_2B_CALIBRATION_QA_CAPACITY_HOLD"
TERMINAL_CALIBRATION_QA_SCIENTIFIC_HOLD = "V2_2B_CALIBRATION_QA_SCIENTIFIC_HOLD"
TERMINAL_CALIBRATION_QA_HARNESS_HOLD = "V2_2B_CALIBRATION_QA_HARNESS_HOLD"
TERMINAL_PRE_SCIENTIFIC_ABORT = "PRE_SCIENTIFIC_OPERATIONAL_ABORT"
CAPACITY_REASONS = ("VRAM_BUDGET", "OOM", "WALL_TIME")

# Q7 — exact analytic V2-0 ledger constants, never inferred from observed CUDA runs.
EXPECTED_D512_FLOPS = {
    "per_round_B1": 67_239_936,
    "K4_B1": 268_959_744,
    "K4_B8": 2_151_677_952,
    "K4_B128": 34_426_847_232,
}
EXPECTED_R4_UNIQUE_PARAMS = 4_194_304
EXPECTED_U4_UNIQUE_PARAMS = 16_777_216
EXPECTED_FAMILY_COUNT = 7

# Q8 — explicit proposed artifact inventory.
ARTIFACT_CONTRACT = {
    "D1": ("CPU_K1_output", "CUDA_K1_output", "CPU_K4_traces_1_4_final", "CUDA_K4_traces_1_4_final", "metrics"),
    "D2": ("initial_clone_hashes", "trace_equalities", "trace_raw_hashes", "final_raw_hashes"),
    "D3": (
        "gR_gU0_gU1_gU2_gU3_FP32_raw_pt",
        "raw_tensor_sha256",
        "A_B_C_metrics",
        "M32",
        "ULP_M32",
        "A_B_C_decisions",
        "old_max_rel_NON_GATE",
        "S_reverse_NON_GATE",
        "S64_recomputable_from_left_associated_FP64_sum_of_gU0_to_gU3",
    ),
    "D6": ("SCHEMA_SHA256_before_after", "VALUE_SHA256_before_after", "parameter_count_before_after"),
    "D7": ("L0", "per_step_finiteness", "L20", "final_parameter_hash", "final_optimizer_state_hash"),
    "D8": ("peak_allocated", "peak_reserved", "wall_seconds", "status"),
    "global": (
        "OMEGA_V2_2B_SPEC.md", "CALIBRATION_SOURCE_SNAPSHOT.json", "SOURCE_SEAL.json", "QA_REPORT.json", "OFFICIAL_RESULT.json",
        "OMEGA_V2_2B_REPORT.md", "OMEGA_V2_2B_REPORT.md.sha256",
        "OMEGA_V2_2B_CONFORMANCE_BLOCK.md", "artifact_hashes.json", "artifact_hashes_verified.json",
    ),
}

# Q9 — fixed official identity; cell labels are non-scientific implementation names.
OFFICIAL_RUN_ID = OFFICIAL_ID
OFFICIAL_RESULT_SLOT_NAME = "omega_v2_2b_d512_local_pilot_official_attempt01"
OFFICIAL_LAUNCH_LOG_DIR_NAME = "omega_v2_2b_d512_local_pilot_launch_logs_01"
INCIDENT_LOG_DIR_NAME = "omega_v2_2b_d512_local_pilot_incident_01"

SeedMode = Literal["qa", "smoke", "official"]


@dataclass(frozen=True)
class SeedPlan:
    mode: SeedMode
    master_seed: int
    weight_seed: int
    input_seed: int
    loss_w_seed: int
    target_seed: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def make_seed_plan(master_seed: int, *, mode: SeedMode) -> SeedPlan:
    """Validate before deriving any input/target/weight seed values."""
    if mode in ("qa", "smoke"):
        if master_seed != CALIBRATION_SEED:
            raise ValueError("V2_2B_QA_SEED_REJECTED: QA/smoke accepts only calibration seed 20260930")
        return SeedPlan(
            mode=mode,
            master_seed=CALIBRATION_SEED,
            weight_seed=CALIBRATION_SEED,
            input_seed=CALIBRATION_SEED + 1008,
            loss_w_seed=CALIBRATION_SEED + 2008,
            target_seed=CALIBRATION_SEED + D7_TARGET_SEED_OFFSET,
        )
    if mode != "official":
        raise ValueError(f"unsupported V2-2B seed mode: {mode}")
    if master_seed == CALIBRATION_SEED:
        raise ValueError("V2_2B_CALIBRATION_SEED_FORBIDDEN_FROM_OFFICIAL_VERDICT")
    if master_seed not in OFFICIAL_MASTER_SEEDS:
        raise ValueError("V2_2B_OFFICIAL_SEED_NOT_IN_FROZEN_SET")
    return SeedPlan(
        mode="official",
        master_seed=master_seed,
        weight_seed=master_seed,
        input_seed=master_seed + 1008,
        loss_w_seed=master_seed + 2008,
        target_seed=master_seed + D7_TARGET_SEED_OFFSET,
    )


def official_seed_plans() -> tuple[SeedPlan, ...]:
    """Call only after source seal and environment verification."""
    return tuple(make_seed_plan(seed, mode="official") for seed in OFFICIAL_MASTER_SEEDS)
