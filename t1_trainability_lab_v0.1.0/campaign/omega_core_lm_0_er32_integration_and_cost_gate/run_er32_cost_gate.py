"""Phase 2 ER32 technical cost gate.

The executable path is intentionally explicit. Importing this module and
running it without ``--execute`` cannot load the corpus, teacher, or launch a
child process. Real benchmark execution is not part of Phase 2 implementation
verification.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import hmac
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn


UNIT_DIR = Path(__file__).resolve().parent
LAB_ROOT = UNIT_DIR.parents[1]
SCRIPTS_DIR = LAB_ROOT / "scripts"
FASTPATH_DIR = LAB_ROOT / "campaign" / "omega_core_lm_0_r1_cpu_fastpath_validation"
GPU_PREP_DIR = LAB_ROOT / "campaign" / "omega_core_lm_0_gpu_environment_preparation"
for _path in (SCRIPTS_DIR, FASTPATH_DIR, GPU_PREP_DIR, UNIT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Existing Step2 data, teacher, and model provenance. No source is copied here.
from run_step2_benchmark import (  # noqa: E402
    APPROVED_MANIFEST_PATH,
    APPROVED_MANIFEST_SHA256,
    BASE_LR,
    CLIP_NORM,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DIMENSION,
    INTEROP_THREADS,
    MODEL_ID,
    MODEL_REVISION,
    OmegaCoreLM0R1Technical,
    SLOTS,
    TEMPERATURE,
    OmegaCoreLMFast,
    VOCAB_SIZE,
    load_approved_source,
    load_teacher,
    teacher_window_logits,
)
from omega_fast_er32 import OmegaCoreLMFastER32  # noqa: E402
from omega_nominal_microbatch_runner import APPROVED_SELECTION_MANIFEST, validate_selected_documents  # noqa: E402,F401


THREADS = int(os.environ.get("OMEGA_ER32_TORCH_THREADS", "4"))
BATCH = 8
TOKENS_PER_WINDOW = 256
CHUNK_SIZE = 512
MIN_AVAILABLE_BYTES = 1 << 30
ER32_FACTOR_SEED = 20260917
PARAMETER_REDUCTION = 4_820_576
PARAMETER_BYTE_REDUCTION = 19_282_304
PERSISTENT_CORE_BYTE_REDUCTION = 77_129_216
TRAINING_UPDATES = 6
WARMUP_UPDATES = 2
MEASURED_UPDATES = 4
WINDOW_SCHEDULE = (0, 1, 0, 1, 0, 1)
TRAINING_COMBINATIONS = (("F", 1), ("ER32", 1), ("F", 4), ("ER32", 4))
INFERENCE_COMBINATIONS = TRAINING_COMBINATIONS
GATE_THRESHOLD = 1.25
RSS_ANOMALY_BYTES = 128 * 1024 * 1024
TIMING_FIELDS = (
    "teacher_forward_seconds",
    "student_recurrence_forward_seconds",
    "vocab_projection_loss_seconds",
    "backward_seconds",
    "clipping_seconds",
    "adamw_seconds",
    "ledger_seconds",
    "total_seconds",
)
GATE_NAMES = {
    "I": "INTEGRATION_CORRESPONDENCE",
    "II": "EXPLICIT_EFFICIENT_EQUIVALENCE",
    "III": "PERSISTENT_FOOTPRINT",
    "IV": "MEMORY_SAFETY",
    "V": "TRAINING_COST",
    "VI": "INFERENCE_COST",
}
UPDATE_SNAPSHOT_PHASES = (
    "before_teacher",
    "after_teacher",
    "after_student_forward",
    "after_loss",
    "after_backward",
    "after_clip",
    "after_optimizer_step",
    "after_update_cleanup",
)
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"
CHILD_TOKEN_ENV = "ER32_COST_GATE_CHILD_TOKEN"


class ResourceGuard(RuntimeError):
    """Raised before another unsafe operation when memory is below the floor."""

    def __init__(self, phase: str, snapshot: dict[str, int | None]) -> None:
        super().__init__(f"available memory below 1 GiB at {phase}")
        self.phase = phase
        self.snapshot = snapshot


class NonFinite(RuntimeError):
    """Raised when the real training path produces a non-finite tensor."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_self_hashed_json(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    """Write once, reread, and verify self-hash and exact persisted bytes."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    snapshot = json_safe(payload)
    if not isinstance(snapshot, dict):
        raise TypeError("artifact payload must be an object")
    snapshot["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    unsigned = canonical_json(snapshot)
    digest = sha256_bytes(unsigned)
    snapshot["artifact_self_hash"] = digest
    encoded = canonical_json(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)

    persisted = path.read_bytes()
    parsed = json.loads(persisted.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("artifact_self_hash") != digest:
        raise RuntimeError("artifact self-hash verification failed: missing or changed digest")
    stored = parsed["artifact_self_hash"]
    parsed["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    if persisted != encoded or canonical_json(parsed) != unsigned or sha256_bytes(canonical_json(parsed)) != stored:
        raise RuntimeError("artifact self-hash verification failed: persisted bytes differ")
    return digest, sha256_bytes(persisted)


def memory_snapshot() -> dict[str, int | None]:
    process = psutil.Process()
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None)
    return {
        "rss_bytes": int(info.rss),
        "available_system_bytes": int(psutil.virtual_memory().available),
        "os_peak_working_set_bytes": int(peak) if peak is not None else None,
    }


def capture_snapshot(
    snapshots: list[dict[str, int | None | str]],
    phase: str,
    *,
    enforce_guard: bool = True,
    snapshot_fn: Callable[[], dict[str, int | None]] = memory_snapshot,
) -> dict[str, int | None | str]:
    snapshot = {"phase": phase, **snapshot_fn()}
    required = {"rss_bytes", "available_system_bytes", "os_peak_working_set_bytes"}
    if not required.issubset(snapshot):
        raise AssertionError("memory snapshot is missing required fields")
    snapshots.append(snapshot)
    available = snapshot["available_system_bytes"]
    if enforce_guard and available is not None and available < MIN_AVAILABLE_BYTES:
        raise ResourceGuard(phase, snapshot)  # type: ignore[arg-type]
    return snapshot


def tensor_inventory(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Count real tensor storage; RSS is deliberately not used as a proxy."""

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    gradient_bytes = sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in model.parameters() if parameter.grad is not None)
    exp_avg = 0
    exp_avg_sq = 0
    other = 0
    for state in optimizer.state.values():
        for name, value in state.items():
            if not isinstance(value, Tensor):
                continue
            bytes_used = value.numel() * value.element_size()
            if name == "exp_avg":
                exp_avg += bytes_used
            elif name == "exp_avg_sq":
                exp_avg_sq += bytes_used
            else:
                other += bytes_used
    persistent = parameter_bytes + gradient_bytes + exp_avg + exp_avg_sq + other
    return {
        "parameter_bytes": int(parameter_bytes),
        "gradient_bytes": int(gradient_bytes),
        "optimizer_exp_avg_bytes": int(exp_avg),
        "optimizer_exp_avg_sq_bytes": int(exp_avg_sq),
        "optimizer_other_tensor_bytes": int(other),
        "persistent_train_state_bytes": int(persistent),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def make_adamw(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def chunked_original_loss(
    model: nn.Module,
    readout_states: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    """A/B/C loss route: project, flatten, chunks of 512, original CE+KL."""

    projected = model.project(readout_states).flatten(0, 1)
    flat_teacher = teacher_logits.reshape(-1, VOCAB_SIZE)
    flat_targets = targets.reshape(-1)
    weights = valid_mask.reshape(-1).to(projected.dtype)
    totals = projected.new_zeros(2)
    for start in range(0, projected.shape[0], CHUNK_SIZE):
        stop = start + CHUNK_SIZE
        logits = model.logits_from_projected(projected[start:stop])
        ce = F.cross_entropy(logits, flat_targets[start:stop], reduction="none")
        student_log_probs = F.log_softmax(logits / TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(flat_teacher[start:stop] / TEMPERATURE, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1) * TEMPERATURE**2
        totals += torch.stack(((ce * weights[start:stop]).sum(), (kl * weights[start:stop]).sum()))
    denominator = weights.sum().clamp_min(1.0)
    ce, kl = totals / denominator
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}


def assert_finite(model: nn.Module, losses: dict[str, Tensor]) -> None:
    if not all(bool(torch.isfinite(value).all().item()) for value in losses.values()):
        raise NonFinite("non-finite loss")
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise NonFinite("non-finite parameter")
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()) for parameter in model.parameters()):
        raise NonFinite("non-finite gradient")


def make_f_reference(rounds: int, seed: int) -> OmegaCoreLMFast:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        r1 = OmegaCoreLM0R1Technical(
            vocab_size=VOCAB_SIZE,
            dimension=DIMENSION,
            slots=SLOTS,
            rounds=rounds,
            variant="shared",
        ).float()
        return OmegaCoreLMFast.from_reference(r1).float()


def make_candidate(config: str, rounds: int, f_reference: OmegaCoreLMFast) -> nn.Module:
    if config == "F":
        return copy.deepcopy(f_reference).float()
    if config == "ER32":
        return OmegaCoreLMFastER32.from_f_reference(
            f_reference,
            experimental_seed=ER32_FACTOR_SEED,
            implementation="efficient",
        ).float()
    raise ValueError(f"unknown config: {config}")


def append_ledger(path: Path, record: dict[str, Any]) -> float:
    started = time.perf_counter()
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_safe(record), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return time.perf_counter() - started


def run_training_child(config: str, rounds: int, run_dir: Path, ledger_path: Path) -> dict[str, Any]:
    """Run one exact six-update combination inside one fresh Python process."""

    snapshots: list[dict[str, int | None | str]] = []
    updates: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "kind": "training",
        "config": config,
        "rounds": rounds,
        "schedule": list(WINDOW_SCHEDULE),
        "updates_requested": TRAINING_UPDATES,
        "snapshots": snapshots,
        "updates": updates,
        "status": "failed",
        "updates_completed": 0,
    }
    try:
        capture_snapshot(snapshots, "process_start")
        source, document_ids, manifest, tokenizer = load_approved_source()
        del tokenizer
        teacher = load_teacher()
        capture_snapshot(snapshots, "teacher_loaded")

        seed = 20260913 if rounds == 1 else 20260914
        f_reference = make_f_reference(rounds, seed)
        capture_snapshot(snapshots, "reference_created")
        candidate = make_candidate(config, rounds, f_reference)
        del f_reference
        gc.collect()
        capture_snapshot(snapshots, "candidate_ready_after_reference_deleted")
        optimizer = make_adamw(candidate)
        capture_snapshot(snapshots, "optimizer_created")
        result["parameter_count"] = parameter_count(candidate)

        state: Tensor | None = None
        for update_index, window in enumerate(WINDOW_SCHEDULE):
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if window == 0:
                state = candidate.initial_state(BATCH, device=torch.device("cpu"))
            if state is None:
                raise RuntimeError("window 1 has no detached state from window 0")
            input_ids = source[:, :TOKENS_PER_WINDOW] if window == 0 else source[:, TOKENS_PER_WINDOW : 2 * TOKENS_PER_WINDOW]
            targets = source[:, 1 : 1 + TOKENS_PER_WINDOW] if window == 0 else source[:, 1 + TOKENS_PER_WINDOW : 1 + 2 * TOKENS_PER_WINDOW]
            valid_mask = torch.ones((BATCH, TOKENS_PER_WINDOW), dtype=torch.bool)
            timings: dict[str, float] = {}

            phase_started = time.perf_counter()
            capture_snapshot(snapshots, "before_teacher")
            teacher_logits = teacher_window_logits(teacher, source, window)
            timings["teacher_forward_seconds"] = time.perf_counter() - phase_started
            capture_snapshot(snapshots, "after_teacher")

            phase_started = time.perf_counter()
            next_state, _, _, readout_states = candidate.recur_states(input_ids, state)
            timings["student_recurrence_forward_seconds"] = time.perf_counter() - phase_started
            capture_snapshot(snapshots, "after_student_forward")

            phase_started = time.perf_counter()
            losses = chunked_original_loss(candidate, readout_states, teacher_logits, targets, valid_mask)
            timings["vocab_projection_loss_seconds"] = time.perf_counter() - phase_started
            capture_snapshot(snapshots, "after_loss")
            if not all(bool(torch.isfinite(value).item()) for value in losses.values()):
                raise NonFinite("non-finite loss")

            phase_started = time.perf_counter()
            losses["total"].backward()
            timings["backward_seconds"] = time.perf_counter() - phase_started
            capture_snapshot(snapshots, "after_backward")
            assert_finite(candidate, losses)

            phase_started = time.perf_counter()
            pre_clip_norm = grad_norm(candidate)
            returned_clip_norm = float(torch.nn.utils.clip_grad_norm_(candidate.parameters(), CLIP_NORM).item())
            post_clip_norm = grad_norm(candidate)
            timings["clipping_seconds"] = time.perf_counter() - phase_started
            capture_snapshot(snapshots, "after_clip")

            # Detach recurrent state before window-0 AdamW, matching A/B/C.
            detached_state = next_state.detach()
            phase_started = time.perf_counter()
            optimizer.step()
            timings["adamw_seconds"] = time.perf_counter() - phase_started
            capture_snapshot(snapshots, "after_optimizer_step")
            inventory = tensor_inventory(candidate, optimizer) if update_index == 0 else None
            state = detached_state

            del teacher_logits, readout_states, next_state, detached_state, losses
            gc.collect()
            capture_snapshot(snapshots, "after_update_cleanup")
            record: dict[str, Any] = {
                "kind": "training_update",
                "config": config,
                "rounds": rounds,
                "update": update_index,
                "window": window,
                "phase": "warmup" if update_index < WARMUP_UPDATES else "measured",
                "valid_tokens": BATCH * TOKENS_PER_WINDOW,
                "document_ids": document_ids,
                "manifest": manifest,
                "pre_clip_grad_norm": pre_clip_norm,
                "clip_returned_norm": returned_clip_norm,
                "post_clip_grad_norm": post_clip_norm,
                "timing": timings,
                "tensor_inventory_after_first_optimizer_step": inventory,
                "snapshots": [item for item in snapshots if item.get("phase") in UPDATE_SNAPSHOT_PHASES][-8:],
            }
            # Time serialization separately, then persist complete timing fields.
            ledger_started = time.perf_counter()
            json.dumps(json_safe(record), sort_keys=True, separators=(",", ":"))
            timings["ledger_seconds"] = time.perf_counter() - ledger_started
            timings["total_seconds"] = time.perf_counter() - update_started
            timings["ledger_seconds"] += append_ledger(ledger_path, record)
            timings["total_seconds"] = time.perf_counter() - update_started
            updates.append(record)
            result["updates_completed"] = len(updates)

        result["status"] = "completed"
        result["tensor_inventory"] = updates[0]["tensor_inventory_after_first_optimizer_step"]
        result["minimum_available_system_bytes"] = min(
            int(item["available_system_bytes"])
            for item in snapshots
            if item.get("available_system_bytes") is not None
        )
    except ResourceGuard as error:
        result["status"] = "INCONCLUSIVE_RESOURCE"
        result["resource_guard"] = {"phase": error.phase, "snapshot": error.snapshot}
    except NonFinite as error:
        result["status"] = "NONFINITE"
        result["error"] = str(error)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            result["status"] = "OOM"
        result["error"] = str(error)
    except Exception as error:  # pragma: no cover - real child diagnostic boundary
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def run_inference_child(config: str, rounds: int) -> dict[str, Any]:
    """Run one fresh CPU FP32 eager inference combination."""

    source, _, _, tokenizer = load_approved_source()
    del tokenizer
    seed = 20260913 if rounds == 1 else 20260914
    f_reference = make_f_reference(rounds, seed)
    candidate = make_candidate(config, rounds, f_reference).eval()
    tokens = source[:1]
    with torch.inference_mode():
        state = candidate.initial_state(1, device=torch.device("cpu"))
        for index in range(32):
            state, _, _, _ = candidate.recur_states(tokens[:, index : index + 1], state)
        started = time.perf_counter()
        for index in range(32, 288):
            state, _, _, _ = candidate.recur_states(tokens[:, index : index + 1], state)
        mode_a_seconds = time.perf_counter() - started

        warmup_state = candidate.initial_state(1, device=torch.device("cpu"))
        candidate.recur_states(tokens[:, :256], warmup_state)
        mode_b_seconds: list[float] = []
        for _ in range(3):
            repetition_state = candidate.initial_state(1, device=torch.device("cpu"))
            started = time.perf_counter()
            candidate.recur_states(tokens[:, :256], repetition_state)
            mode_b_seconds.append(time.perf_counter() - started)

    steady_state_rss = memory_snapshot()["rss_bytes"]
    mode_a = {"tokens": 256, "seconds": mode_a_seconds, "tokens_per_second": 256 / mode_a_seconds, "ms_per_token": mode_a_seconds * 1000 / 256}
    mode_b = {
        "tokens": 256,
        "seconds_per_window": mode_b_seconds,
        "tokens_per_second": [256 / value for value in mode_b_seconds],
        "mean_seconds_per_window": sum(mode_b_seconds) / len(mode_b_seconds),
        "mean_tokens_per_second": 256 / (sum(mode_b_seconds) / len(mode_b_seconds)),
    }
    return {
        "kind": "inference",
        "config": config,
        "rounds": rounds,
        "dtype": "float32",
        "execution": "cpu eager inference_mode",
        "mode_a": mode_a,
        "mode_b": mode_b,
        "steady_state_rss_bytes": steady_state_rss,
    }


def _child_authorized(token: str | None) -> bool:
    expected = os.environ.get(CHILD_TOKEN_ENV, "")
    return bool(token and expected and hmac.compare_digest(token, expected))


def _decode_child_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("child output is not an object")
            return value
    raise RuntimeError("child produced no JSON result")


def _spawn_child(command: list[str], token: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment[CHILD_TOKEN_ENV] = token
    completed = subprocess.run(command, cwd=UNIT_DIR, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"child failed with exit code {completed.returncode}: {completed.stderr[-1000:]}")
    return _decode_child_output(completed.stdout)


def run_training_children(run_dir: Path, ledger_path: Path, *, spawn: Callable[[list[str], str], dict[str, Any]] = _spawn_child) -> list[dict[str, Any]]:
    token = secrets.token_hex(32)
    results: list[dict[str, Any]] = []
    for config, rounds in TRAINING_COMBINATIONS:
        command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--child-training", "--child-token", token, "--config", config, "--rounds", str(rounds), "--run-dir", str(run_dir), "--ledger", str(ledger_path)]
        results.append(spawn(command, token))
    return results


def run_inference_children(*, spawn: Callable[[list[str], str], dict[str, Any]] = _spawn_child) -> list[dict[str, Any]]:
    token = secrets.token_hex(32)
    results: list[dict[str, Any]] = []
    for config, rounds in INFERENCE_COMBINATIONS:
        command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--child-inference", "--child-token", token, "--config", config, "--rounds", str(rounds)]
        results.append(spawn(command, token))
    return results


def aggregate_training_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_combination: dict[str, dict[str, Any]] = {}
    for result in results:
        key = f"{result['config']}_K{result['rounds']}"
        measured = [item for item in result.get("updates", []) if item.get("phase") == "measured"]
        seconds = [float(item["timing"]["total_seconds"]) for item in measured]
        total = sum(seconds)
        by_combination[key] = {"measured_updates": len(measured), "t_update_seconds": total / len(seconds) if seconds else float("inf"), "measured_total_seconds": total, "valid_tokens_per_second": (2048 * len(seconds)) / total if total > 0 else 0.0}

    def value(config: str, rounds: int) -> float:
        return float(by_combination[f"{config}_K{rounds}"]["measured_total_seconds"])

    f_k1, f_k4, er_k1, er_k4 = value("F", 1), value("F", 4), value("ER32", 1), value("ER32", 4)
    er_total = er_k1 + er_k4
    f_total = f_k1 + f_k4
    joint_tokens = 2048 * 4
    return {
        "by_combination": by_combination,
        "t_K1_per_update": {"F": by_combination["F_K1"]["t_update_seconds"], "ER32": by_combination["ER32_K1"]["t_update_seconds"]},
        "t_K4_per_update": {"F": by_combination["F_K4"]["t_update_seconds"], "ER32": by_combination["ER32_K4"]["t_update_seconds"]},
        "joint_K1_K4": {"F": f_k1 + f_k4, "ER32": er_total},
        "valid_tokens_per_second": {
            **{key: value["valid_tokens_per_second"] for key, value in by_combination.items()},
            "F_joint_K1_K4": joint_tokens / f_total if f_total > 0 else 0.0,
            "ER32_joint_K1_K4": joint_tokens / er_total if er_total > 0 else 0.0,
        },
        "R_train": er_total / f_total if f_total > 0 else float("inf"),
        "R_train_formula": "T_ER32_total / T_F_total",
    }


def classify_training_cost(metrics: dict[str, Any], threshold: float = GATE_THRESHOLD) -> dict[str, Any]:
    ratio = float(metrics["R_train"])
    passed = ratio <= threshold
    return {"classification": "PASS" if passed else "COST_REGRESSION", "R_train": ratio, "threshold": threshold, "quality_scoping": "ELIGIBLE" if passed else "QUALITY_SCOPING_HOLD"}


def _inventory_value(result: dict[str, Any], key: str) -> int:
    inventory = result.get("tensor_inventory") or result.get("tensor_inventory_after_first_optimizer_step")
    if not isinstance(inventory, dict) or key not in inventory:
        raise KeyError(key)
    return int(inventory[key])


def classify_persistent_footprint(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped = {f"{item['config']}_K{item['rounds']}": item for item in results}
    checks: list[bool] = []
    reductions: dict[str, Any] = {}
    try:
        for rounds in (1, 4):
            f, er = grouped[f"F_K{rounds}"], grouped[f"ER32_K{rounds}"]
            f_parameters = int(f["parameter_count"])
            er_parameters = int(er["parameter_count"])
            parameter_delta = f_parameters - er_parameters
            parameter_bytes_delta = _inventory_value(f, "parameter_bytes") - _inventory_value(er, "parameter_bytes")
            core_keys = ("parameter_bytes", "gradient_bytes", "optimizer_exp_avg_bytes", "optimizer_exp_avg_sq_bytes")
            core_delta = sum(_inventory_value(f, key) - _inventory_value(er, key) for key in core_keys)
            reductions[f"K{rounds}"] = {"parameter_reduction": parameter_delta, "parameter_byte_reduction": parameter_bytes_delta, "persistent_core_byte_reduction": core_delta}
            checks.extend((parameter_delta == PARAMETER_REDUCTION, parameter_bytes_delta == PARAMETER_BYTE_REDUCTION, core_delta == PERSISTENT_CORE_BYTE_REDUCTION))
    except (KeyError, TypeError, ValueError):
        return {"classification": "FAIL", "reductions": reductions, "reason": "incomplete real tensor inventory"}
    return {"classification": "PASS" if all(checks) else "FAIL", "reductions": reductions}


def classify_memory_safety(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    if any(item.get("status") == "INCONCLUSIVE_RESOURCE" for item in values):
        classification = "INCONCLUSIVE_RESOURCE"
    elif any(item.get("status") in {"OOM", "NONFINITE"} for item in values):
        classification = "FAIL"
    elif sum(int(item.get("updates_completed", 0)) for item in values) != 24:
        classification = "FAIL"
    else:
        classification = "PASS"
    return {"classification": classification, "updates_completed": sum(int(item.get("updates_completed", 0)) for item in values)}


def aggregate_inference_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_combination: dict[str, Any] = {}
    for result in results:
        key = f"{result['config']}_K{result['rounds']}"
        by_combination[key] = {"mode_a_tokens_per_second": float(result["mode_a"]["tokens_per_second"]), "mode_a_ms_per_token": float(result["mode_a"]["ms_per_token"]), "mode_b_tokens_per_second": float(result["mode_b"]["mean_tokens_per_second"]), "mode_b_seconds_per_window": float(result["mode_b"]["mean_seconds_per_window"]), "steady_state_rss_bytes": int(result["steady_state_rss_bytes"])}
    time_cost_ratios: dict[str, float] = {}
    for rounds in (1, 4):
        f, er = by_combination[f"F_K{rounds}"], by_combination[f"ER32_K{rounds}"]
        time_cost_ratios[f"K{rounds}_mode_a"] = er["mode_a_ms_per_token"] / f["mode_a_ms_per_token"]
        time_cost_ratios[f"K{rounds}_mode_b"] = er["mode_b_seconds_per_window"] / f["mode_b_seconds_per_window"]
    return {
        "by_combination": by_combination,
        "ratio_semantics": "ER32_time / F_time",
        "time_cost_fields": {"mode_a": "ms_per_token", "mode_b": "mean_seconds_per_window"},
        "ER32_over_F_time_cost": time_cost_ratios,
    }


def classify_inference_cost(metrics: dict[str, Any], threshold: float = GATE_THRESHOLD, rss_limit: int = RSS_ANOMALY_BYTES) -> dict[str, Any]:
    ratios = metrics["ER32_over_F_time_cost"]
    anomaly = any(metrics["by_combination"][f"ER32_K{rounds}"]["steady_state_rss_bytes"] - metrics["by_combination"][f"F_K{rounds}"]["steady_state_rss_bytes"] > rss_limit for rounds in (1, 4))
    if anomaly:
        classification, hold = "MEMORY_RUNTIME_ANOMALY", "QUALITY_SCOPING_HOLD"
    elif any(float(value) > threshold for value in ratios.values()):
        classification, hold = "INFERENCE_COST_REGRESSION", "QUALITY_SCOPING_HOLD"
    else:
        classification, hold = "PASS", "ELIGIBLE"
    return {
        "classification": classification,
        "threshold": threshold,
        "ratio_semantics": "ER32_time / F_time",
        "ratios": ratios,
        "quality_scoping": hold,
    }


def global_gate_status(gates: dict[str, Any]) -> str:
    if any(gates.get(key, {}).get("classification") != "PASS" for key in ("I", "II", "III", "IV", "V", "VI")):
        return "FAIL"
    return "PASS"


def source_hashes() -> dict[str, str]:
    paths = {
        "phase2_runner": UNIT_DIR / "run_er32_cost_gate.py",
        "er32_adapter": UNIT_DIR / "omega_fast_er32.py",
        "phase1_integration_tests": UNIT_DIR / "test_er32_integration.py",
        "step2_runner": FASTPATH_DIR / "run_step2_benchmark.py",
        "step2_tests": FASTPATH_DIR / "test_step2_benchmark.py",
        "technical_preflight": SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py",
        "approved_manifest": LAB_ROOT / "campaign" / "omega_core_lm_0_r1_training_technical_preflight" / "training_technical_preflight_v2.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def read_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LAB_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def artifact_metadata() -> dict[str, Any]:
    return {
        "source_sha256s": source_hashes(),
        "git_commit": read_git_commit(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "windows": platform.platform(),
        "cpu_identity": platform.processor() or platform.uname().processor,
        "threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads(), "configured_intraop": THREADS, "configured_interop": INTEROP_THREADS},
        "dataset_revision": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "split": "train"},
        "teacher_revision": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "manifest_hash": APPROVED_MANIFEST_SHA256,
    }


def new_run_dir(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        path = output_root / run_id
        try:
            path.mkdir(parents=False, exist_ok=False)
            return run_id, path
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a new run-id")


def build_inference_report(
    run_id: str,
    metadata: dict[str, Any],
    inference_metrics: dict[str, Any],
    full_gates: dict[str, Any],
) -> dict[str, Any]:
    """Build inference artifact while deriving global status from all six gates."""

    return {
        "schema": "er32-cost-gate-v1",
        "artifact": "inference_report",
        "run_id": run_id,
        **metadata,
        "inference": inference_metrics,
        "gates": {GATE_NAMES[key]: full_gates[key] for key in ("I", "II", "VI")},
        "global_status": global_gate_status(full_gates),
    }


TIMING_DEFINITION_NOTE = (
    "Ledger total_seconds is captured before append/fsync; cost_report training totals are "
    "captured after append/fsync. Small differences are expected and do not justify mutating "
    "raw evidence or rerunning the benchmark."
)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _repair_source_hashes(run_dir: Path) -> dict[str, str]:
    names = ("integration_report.json", "cost_report.json", "inference_report.json", "ledger.jsonl")
    return {name: sha256_file(run_dir / name) for name in names if (run_dir / name).is_file()}


def build_analysis_repair(run_dir: Path) -> dict[str, Any]:
    """Derive corrected Gate V/VI analysis from existing reports without execution."""

    cost_report = _read_json_object(run_dir / "cost_report.json")
    inference_report = _read_json_object(run_dir / "inference_report.json")

    training = cost_report["training"]
    joint = training["joint_K1_K4"]
    f_total = float(joint["F"])
    er_total = float(joint["ER32"])
    if f_total <= 0 or er_total <= 0:
        raise ValueError("training report has non-positive joint timing")
    old_training_ratio = f_total / er_total
    corrected_training_ratio = er_total / f_total
    corrected_training_gate = classify_training_cost({"R_train": corrected_training_ratio})

    by_combination = inference_report["inference"]["by_combination"]
    old_inference_ratios: dict[str, float] = {}
    corrected_inference_ratios: dict[str, float] = {}
    inference_metrics: dict[str, Any] = {
        "by_combination": by_combination,
        "ratio_semantics": "ER32_time / F_time",
        "time_cost_fields": {"mode_a": "ms_per_token", "mode_b": "mean_seconds_per_window"},
    }
    inference_metrics["ER32_over_F_time_cost"] = corrected_inference_ratios
    inference_details: dict[str, Any] = {}
    for rounds in (1, 4):
        f = by_combination[f"F_K{rounds}"]
        er = by_combination[f"ER32_K{rounds}"]
        mode_a_key = f"K{rounds}_mode_a"
        mode_b_key = f"K{rounds}_mode_b"
        old_inference_ratios[mode_a_key] = float(er["mode_a_tokens_per_second"]) / float(f["mode_a_tokens_per_second"])
        old_inference_ratios[mode_b_key] = float(er["mode_b_tokens_per_second"]) / float(f["mode_b_tokens_per_second"])
        corrected_inference_ratios[mode_a_key] = float(er["mode_a_ms_per_token"]) / float(f["mode_a_ms_per_token"])
        corrected_inference_ratios[mode_b_key] = float(er["mode_b_seconds_per_window"]) / float(f["mode_b_seconds_per_window"])
        inference_details[mode_a_key] = {
            "old_formula": "ER32_tokens_per_second / F_tokens_per_second",
            "corrected_formula": "ER32_ms_per_token / F_ms_per_token",
            "old_metric": old_inference_ratios[mode_a_key],
            "corrected_metric": corrected_inference_ratios[mode_a_key],
            "raw": {
                "F_throughput": float(f["mode_a_tokens_per_second"]),
                "ER32_throughput": float(er["mode_a_tokens_per_second"]),
                "F_ms_per_token": float(f["mode_a_ms_per_token"]),
                "ER32_ms_per_token": float(er["mode_a_ms_per_token"]),
            },
        }
        inference_details[mode_b_key] = {
            "old_formula": "ER32_mean_tokens_per_second / F_mean_tokens_per_second",
            "corrected_formula": "ER32_mean_seconds_per_window / F_mean_seconds_per_window",
            "old_metric": old_inference_ratios[mode_b_key],
            "corrected_metric": corrected_inference_ratios[mode_b_key],
            "raw": {
                "F_throughput": float(f["mode_b_tokens_per_second"]),
                "ER32_throughput": float(er["mode_b_tokens_per_second"]),
                "F_seconds_per_window": float(f["mode_b_seconds_per_window"]),
                "ER32_seconds_per_window": float(er["mode_b_seconds_per_window"]),
            },
        }
    corrected_inference_gate = classify_inference_cost(inference_metrics)

    original_training_gate = cost_report["gates"][GATE_NAMES["V"]]
    original_inference_gate = inference_report["gates"][GATE_NAMES["VI"]]
    source_artifact_hashes = _repair_source_hashes(run_dir)
    return {
        "schema": "er32-cost-gate-analysis-repair-v1",
        "artifact": "analysis_repair",
        "run_id": cost_report.get("run_id", run_dir.name),
        "benchmark_rerun": False,
        "raw_measurements_unchanged": True,
        "source_artifact_sha256s": source_artifact_hashes,
        "source_report_self_hashes": {
            "cost_report": cost_report.get("artifact_self_hash"),
            "inference_report": inference_report.get("artifact_self_hash"),
        },
        "timing_definition_note": TIMING_DEFINITION_NOTE,
        "gate_V_training_cost": {
            "old_formula": "R_train = T_F_total / T_ER32_total",
            "corrected_formula": "R_train = T_ER32_total / T_F_total",
            "raw_measurements_seconds": {"T_F_total": f_total, "T_ER32_total": er_total},
            "old_metric": old_training_ratio,
            "corrected_metric": corrected_training_ratio,
            "original_gate": original_training_gate,
            "recalculated_gate": corrected_training_gate,
        },
        "gate_VI_inference_cost": {
            "old_formula": "throughput ratio ER32/F",
            "corrected_formula": "time-cost ratio ER32_time / F_time",
            "time_fields": {"mode_a": "ms_per_token", "mode_b": "mean_seconds_per_window"},
            "old_metrics": old_inference_ratios,
            "corrected_metrics": corrected_inference_ratios,
            "per_mode": inference_details,
            "original_gate": original_inference_gate,
            "recalculated_gate": corrected_inference_gate,
        },
        "recalculated_gate_classifications": {
            GATE_NAMES["V"]: corrected_training_gate,
            GATE_NAMES["VI"]: corrected_inference_gate,
        },
    }


def write_analysis_repair(run_dir: Path) -> tuple[str, str]:
    """Write only analysis_repair.json, refusing overwrite and verifying its self-hash."""

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    return write_self_hashed_json(run_dir / "analysis_repair.json", build_analysis_repair(run_dir))


def execute_phase2(output_root: Path) -> dict[str, Any]:
    run_id, run_dir = new_run_dir(output_root)
    ledger_path = run_dir / "ledger.jsonl"
    training = run_training_children(run_dir, ledger_path)
    inference = run_inference_children()
    footprint = classify_persistent_footprint(training)
    memory = classify_memory_safety(training)
    training_metrics = aggregate_training_metrics(training)
    training_cost = classify_training_cost(training_metrics)
    inference_metrics = aggregate_inference_metrics(inference)
    inference_cost = classify_inference_cost(inference_metrics)
    gates = {"I": {"classification": "PASS"}, "II": {"classification": "PASS"}, "III": footprint, "IV": memory, "V": training_cost, "VI": inference_cost}
    named_gates = {GATE_NAMES[key]: value for key, value in gates.items()}
    metadata = artifact_metadata()
    integration_report = {"schema": "er32-cost-gate-v1", "artifact": "integration_report", "run_id": run_id, **metadata, "phase1_prerequisite": {"I": "PASS", "II": "PASS"}, "gates": {GATE_NAMES["I"]: gates["I"], GATE_NAMES["II"]: gates["II"]}, "status": "PASS"}
    cost_report = {"schema": "er32-cost-gate-v1", "artifact": "cost_report", "run_id": run_id, **metadata, "training": training_metrics, "gates": {GATE_NAMES[key]: gates[key] for key in ("I", "II", "III", "IV", "V")}, "global_status": global_gate_status(gates)}
    inference_report = build_inference_report(run_id, metadata, inference_metrics, gates)
    write_self_hashed_json(run_dir / "integration_report.json", integration_report)
    write_self_hashed_json(run_dir / "cost_report.json", cost_report)
    write_self_hashed_json(run_dir / "inference_report.json", inference_report)
    return {"status": global_gate_status(gates), "run_id": run_id, "run_dir": run_dir, "gates": gates}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="explicitly authorize real technical execution")
    parser.add_argument("--repair-analysis", action="store_true", help="derive analysis_repair.json from an existing run")
    parser.add_argument("--output-root", type=Path, default=UNIT_DIR / "results")
    parser.add_argument("--child-training", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-inference", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-token", help=argparse.SUPPRESS)
    parser.add_argument("--config", choices=("F", "ER32"), help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, choices=(1, 4), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ledger", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repair_analysis:
        if args.run_dir is None:
            print(json.dumps({"status": "refused", "reason": "--repair-analysis requires --run-dir"}, sort_keys=True))
            return 2
        try:
            artifact_hash, file_hash = write_analysis_repair(args.run_dir)
        except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(json.dumps({"status": "refused", "reason": str(error)}, sort_keys=True))
            return 2
        print(json.dumps({"status": "repaired", "artifact": str(args.run_dir / "analysis_repair.json"), "artifact_self_hash": artifact_hash, "artifact_file_sha256": file_hash, "benchmark_rerun": False, "raw_measurements_unchanged": True}, sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps({"status": "refused", "reason": "real ER32 cost-gate execution requires explicit --execute"}, sort_keys=True))
        return 2
    if args.child_training or args.child_inference:
        if not _child_authorized(args.child_token) or args.config is None or args.rounds is None:
            print(json.dumps({"status": "refused", "reason": "unauthorized child invocation"}, sort_keys=True))
            return 2
        if args.child_training:
            if args.run_dir is None or args.ledger is None:
                return 2
            result = run_training_child(args.config, args.rounds, args.run_dir, args.ledger)
        else:
            result = run_inference_child(args.config, args.rounds)
        print(json.dumps(json_safe(result), sort_keys=True))
        return 0
    result = execute_phase2(args.output_root)
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
