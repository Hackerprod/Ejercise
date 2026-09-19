"""OMEGA CE-only baseline.

This unit is intentionally isolated from historical campaign runners.  Real
Phase A execution is blocked unless explicitly authorized.  Smoke helpers use
synthetic tensors only; no teacher is constructed on the CE-only path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import psutil
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
REPO_ROOT = CAMPAIGN_ROOT.parent.parent
R1_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
F_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
R1_FULL_DIR = R1_DIR / "results" / "full_campaign"
PREP_DIR = REPO_ROOT / "t1_trainability_lab_v0.1.0" / "scripts"
GENERATION_AUDIT_RELATIVE = (
    "t1_trainability_lab_v0.1.0/campaign/"
    "omega_core_lm_0_autoregressive_generation_audit/results/"
    "autoregressive_generation_audit/generation_results.json"
)
HISTORICAL_GENERATION_COMMIT = "ca8f4ad"

if str(F_DIR) not in sys.path:
    sys.path.insert(0, str(F_DIR))
if str(R1_DIR) not in sys.path:
    sys.path.insert(0, str(R1_DIR))
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402


CAMPAIGN_ID = "OMEGA-CE-ONLY-BASELINE"
ADDENDUM = 220
COMMIT_REFERENCE = "b0e2909"
DIMENSION = 128
SLOTS = 8
TOKENIZER_VOCAB = 50257
SEEDS = (20260913, 20260914)
KS = (1, 4)
VARIANTS = ("shared_K1", "shared_K4")
TOTAL_UPDATES = 2000
BOUNDARIES = (0, 500, 1000, 1500, 2000)
PHYSICAL_BATCH = 8
WINDOW_TOKENS = 256
RETAINED_TOKENS = 513
CHUNK_SIZE = 512
BASE_LR = 3e-4
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
CPU_INTRAOP_THREADS = 4
CPU_INTEROP_THREADS = 1
NONINFERIOR_CEILING = 0.10
PHASE0_UPDATES_PER_COMBINATION = 6
PHASE0_WARMUP_UPDATES = 2
PHASE0_MEASURED_UPDATES = 4
PHASE0_COMBINATIONS = ("DISTILL-K1", "CE-only-K1", "DISTILL-K4", "CE-only-K4")
PHASE0_WORKER_FLAG = "--phase0-worker"
EOS_TOKEN_ID = 50256
GENERATION_PROMPT_LENGTH = 32
GENERATION_MAX_NEW_TOKENS = 64
EXPECTED_GENERATION_RECORDS = 32


class InitializationGateError(RuntimeError):
    """Raised when fresh CE initialization differs from frozen R1@0."""


class RealExecutionAuthorizationError(RuntimeError):
    """Raised when a real entry path lacks its explicit authorization flag."""


_CPU_RUNTIME_CONFIGURED = False


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_cpu_runtime() -> None:
    global _CPU_RUNTIME_CONFIGURED
    if _CPU_RUNTIME_CONFIGURED:
        return
    torch.set_num_threads(CPU_INTRAOP_THREADS)
    torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    _CPU_RUNTIME_CONFIGURED = True


def validate_policy() -> dict[str, Any]:
    configure_cpu_runtime()
    policy = {
        "device": "cpu",
        "dtype": "float32",
        "execution": "eager",
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": PHYSICAL_BATCH,
        "optimizer": "AdamW",
        "lr": BASE_LR,
        "betas": list(ADAMW_BETAS),
        "eps": ADAMW_EPS,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "intraop_threads": CPU_INTRAOP_THREADS,
        "interop_threads": CPU_INTEROP_THREADS,
    }
    if torch.get_num_threads() != CPU_INTRAOP_THREADS:
        raise AssertionError("intraop thread policy drift")
    return policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def state_dict_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def state_dict_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if list(left) != list(right):
        return False
    return all(torch.is_tensor(left[name]) and torch.equal(left[name], right[name]) for name in left)


def fresh_model(
    seed: int,
    k: int,
    *,
    vocab_size: int = TOKENIZER_VOCAB,
    dimension: int = DIMENSION,
    slots: int = SLOTS,
) -> OmegaCoreLMFast:
    if k not in KS:
        raise ValueError(f"unsupported K: {k}")
    configure_cpu_runtime()
    set_seed(seed)
    # R1@0 was initialized by reference construction followed by F conversion;
    # direct F construction consumes a different random stream for fused QKV.
    from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical

    reference = OmegaCoreLM0R1Technical(
        vocab_size=vocab_size,
        dimension=dimension,
        slots=slots,
        rounds=k,
        variant="shared",
    ).to(dtype=torch.float32)
    try:
        return OmegaCoreLMFast.from_reference(reference).to(dtype=torch.float32)
    finally:
        del reference


def oracle_checkpoint_path(seed: int, k: int) -> Path:
    return R1_FULL_DIR / "runs" / f"shared_K{k}_seed_{seed}" / "checkpoint_00000.pt"


def initialization_gate(
    *,
    oracle_paths: Mapping[tuple[int, int], Path] | None = None,
    model_factory: Callable[[int, int], torch.nn.Module] | None = None,
) -> dict[str, Any]:
    """Require exact fresh-model equality with every frozen R1 checkpoint @0."""
    factory = model_factory or fresh_model
    evidence: list[dict[str, Any]] = []
    for seed in SEEDS:
        for k in KS:
            path = (oracle_paths or {}).get((seed, k), oracle_checkpoint_path(seed, k))
            if not path.is_file():
                raise InitializationGateError(f"R1@0 oracle missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or int(payload.get("update", -1)) != 0:
                raise InitializationGateError(f"oracle is not checkpoint_00000: {path}")
            oracle = payload.get("model")
            if not isinstance(oracle, dict):
                raise InitializationGateError(f"oracle model state missing: {path}")
            model = factory(seed, k)
            actual = model.state_dict()
            oracle_hash = state_dict_hash(oracle)
            actual_hash = state_dict_hash(actual)
            equal = state_dict_equal(actual, oracle)
            if not equal or actual_hash != oracle_hash:
                raise InitializationGateError(
                    f"fresh CE model differs from R1@0 for seed={seed}, K={k}: "
                    f"actual={actual_hash} oracle={oracle_hash}"
                )
            evidence.append({
                "seed": seed,
                "K": k,
                "variant": f"shared_K{k}",
                "oracle_path": path.as_posix(),
                "oracle_checkpoint_sha256": file_hash(path),
                "state_dict_sha256": actual_hash,
                "torch_equal": True,
            })
            del model
    return {"passed": True, "oracle_update": 0, "combinations": evidence}


def frozen_manifest_paths() -> dict[str, Path]:
    return {
        "train": R1_FULL_DIR / "train_manifest.json",
        "validation": R1_FULL_DIR / "runs" / "shared_K1_seed_20260913" / "validation_manifest.json",
    }


def load_frozen_manifests() -> dict[str, Any]:
    """Load R1 manifests directly; never rebuild corpus order, pairs, or windows."""
    paths = frozen_manifest_paths()
    manifests = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    train = manifests["train"]
    validation = manifests["validation"]
    if train.get("document_count") != 602 or train.get("cyclic_pairs", {}).get("pair_count") != 1000:
        raise ValueError("frozen R1 train manifest identity mismatch")
    if validation.get("document_count") != 8:
        raise ValueError("frozen R1 validation manifest identity mismatch")
    if not train.get("manifest_sha256") or not validation.get("manifest_sha256"):
        raise ValueError("frozen R1 manifest hash missing")
    return {
        "train": train,
        "validation": validation,
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "direct_reuse": True,
        "reconstructed": False,
    }


def _masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    values = F.cross_entropy(logits, targets, reduction="none")
    return (values * weights).sum()


def ce_only_loss(
    model: OmegaCoreLMFast,
    readout_states: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, torch.Tensor | int]:
    """Full-coefficient CE over projected readout chunks only.

    No [B,T,V] tensor is created.  The only vocabulary-sized tensor is one
    [chunk_size,V] result from ``logits_from_projected`` at a time.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    projected = model.project(readout_states).reshape(-1, model.dimension)
    flat_targets = targets.reshape(-1)
    weights = torch.ones_like(flat_targets, dtype=projected.dtype) if valid_mask is None else valid_mask.reshape(-1).to(projected.dtype)
    if projected.shape[0] != flat_targets.shape[0]:
        raise ValueError("readout and target token counts differ")
    total = projected.new_zeros(())
    chunks = 0
    for start in range(0, projected.shape[0], chunk_size):
        stop = min(start + chunk_size, projected.shape[0])
        logits = model.logits_from_projected(projected[start:stop])
        if logits.ndim != 2 or logits.shape[0] > chunk_size:
            raise RuntimeError("CE path produced non-chunked vocabulary logits")
        total = total + _masked_ce(logits, flat_targets[start:stop], weights[start:stop])
        chunks += 1
    denominator = weights.sum().clamp_min(1.0)
    ce = total / denominator
    return {"ce": ce, "total": ce, "chunks": chunks}


def ce_only_forward_loss(
    model: OmegaCoreLMFast,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    previous_state: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
    """Run student recurrence and CE without invoking ``forward_window``."""
    result = model.recur_states(input_ids, previous_state, valid_mask)
    if len(result) == 4:
        next_state, _, _, readout_states = result
    else:
        next_state, readout_states = result
    losses = ce_only_loss(model, readout_states, targets, valid_mask, chunk_size=chunk_size)
    return next_state, readout_states, losses


def phase0_measurement(
    combination: str,
    *,
    student_forward_seconds: float,
    vocab_loss_seconds: float,
    backward_seconds: float,
    clip_seconds: float,
    adamw_seconds: float,
    total_seconds: float,
    rss_bytes_peak: int,
    teacher_forward_seconds: float = 0.0,
    teacher_loaded: bool = False,
) -> dict[str, Any]:
    if combination not in PHASE0_COMBINATIONS:
        raise ValueError(f"unsupported Phase 0 combination: {combination}")
    distill = combination.startswith("DISTILL-")
    if not distill and (teacher_loaded or teacher_forward_seconds != 0.0):
        raise ValueError("CE-only Phase 0 measurement cannot contain teacher work")
    return {
        "combination": combination,
        "fresh_process": True,
        "updates_total": PHASE0_UPDATES_PER_COMBINATION,
        "warmup_updates": PHASE0_WARMUP_UPDATES,
        "measured_updates": PHASE0_MEASURED_UPDATES,
        "student_forward_seconds": float(student_forward_seconds),
        "vocab_loss_seconds": float(vocab_loss_seconds),
        "backward_seconds": float(backward_seconds),
        "clip_seconds": float(clip_seconds),
        "adamw_seconds": float(adamw_seconds),
        "total_seconds": float(total_seconds),
        "rss_bytes_peak": int(rss_bytes_peak),
        "teacher_forward_seconds": float(teacher_forward_seconds) if distill else 0.0,
        "teacher_loaded": bool(teacher_loaded) if distill else False,
        "teacher_forward_calls": PHASE0_MEASURED_UPDATES if distill else 0,
        "kl_calculations": PHASE0_MEASURED_UPDATES if distill else 0,
    }


def _self_hashed(value: dict[str, Any], field: str = "report_self_hash") -> dict[str, Any]:
    unsigned = dict(value)
    unsigned[field] = "__SELF_HASH__"
    encoded = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    result = dict(value)
    result[field] = digest
    written = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    check = written.replace(f'"{field}": "{digest}"'.encode("utf-8"), f'"{field}": "__SELF_HASH__"'.encode("utf-8"), 1)
    if hashlib.sha256(check).hexdigest() != digest:
        raise RuntimeError("self-hash construction failed")
    return result


def verify_self_hash(value: Mapping[str, Any], field: str = "report_self_hash") -> bool:
    digest = value.get(field)
    if not isinstance(digest, str):
        return False
    unsigned = dict(value)
    unsigned[field] = "__SELF_HASH__"
    return hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest() == digest


def build_phase0_report(measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    measurement_rows = [dict(row) for row in measurements]
    if len(measurement_rows) != len(PHASE0_COMBINATIONS):
        raise ValueError("Phase 0 requires exactly four worker rows")
    rows = {str(row["combination"]): row for row in measurement_rows}
    if set(rows) != set(PHASE0_COMBINATIONS):
        raise ValueError("Phase 0 requires exactly four combination measurements")
    for name, row in rows.items():
        required = {"student_forward_seconds", "vocab_loss_seconds", "backward_seconds", "clip_seconds", "adamw_seconds", "total_seconds", "rss_bytes_peak"}
        if not required.issubset(row) or row.get("fresh_process") is not True or row.get("updates_total") != 6 or row.get("updates_executed") not in {None, 6}:
            raise ValueError(f"Phase 0 protocol mismatch for {name}")
        if name.startswith("DISTILL-"):
            if row.get("teacher_loaded") is not True or row.get("teacher_forward_calls") != PHASE0_MEASURED_UPDATES or row.get("kl_calculations") != PHASE0_MEASURED_UPDATES:
                raise ValueError(f"distillation worker teacher/KL accounting mismatch for {name}")
        elif row.get("teacher_loaded") or row.get("teacher_imported") or row.get("teacher_forward_calls", 0) != 0 or row.get("kl_calculations", 0) != 0:
            raise ValueError(f"CE-only teacher/KL contamination for {name}")
    distill_seconds = rows["DISTILL-K1"]["total_seconds"] + rows["DISTILL-K4"]["total_seconds"]
    ce_seconds = rows["CE-only-K1"]["total_seconds"] + rows["CE-only-K4"]["total_seconds"]
    if ce_seconds <= 0 or distill_seconds <= 0:
        raise ValueError("Phase 0 total seconds must be positive")
    q_cost = distill_seconds / ce_seconds
    u_equal_cost = 2 * math.floor((TOTAL_UPDATES / q_cost) / 2)
    report = {
        "schema": "omega-ce-only-baseline-phase0-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "addendum": ADDENDUM,
        "commit_reference": COMMIT_REFERENCE,
        "quality_gate": None,
        "protocol": {
            "fresh_process_per_combination": True,
            "total_updates": 24,
            "updates_per_combination": 6,
            "warmup_updates": 2,
            "measured_updates": 4,
            "combinations": list(PHASE0_COMBINATIONS),
        },
        "measurements": {name: rows[name] for name in PHASE0_COMBINATIONS},
        "q_cost": q_cost,
        "U_equal_cost": u_equal_cost,
        "formula": "q_cost=(T_DISTILL_K1+T_DISTILL_K4)/(T_CE_K1+T_CE_K4); U_equal_cost=2*floor((2000/q_cost)/2)",
    }
    return _self_hashed(report)


def synthetic_phase0_report() -> dict[str, Any]:
    values = {
        "DISTILL-K1": 8.0,
        "CE-only-K1": 3.0,
        "DISTILL-K4": 10.0,
        "CE-only-K4": 4.0,
    }
    measurements = [
        phase0_measurement(
            name,
            student_forward_seconds=values[name] * 0.40,
            vocab_loss_seconds=values[name] * 0.25,
            backward_seconds=values[name] * 0.20,
            clip_seconds=values[name] * 0.02,
            adamw_seconds=values[name] * 0.03,
            total_seconds=values[name],
            rss_bytes_peak=123456,
            teacher_forward_seconds=1.5 if name.startswith("DISTILL-") else 0.0,
            teacher_loaded=name.startswith("DISTILL-"),
        )
        for name in PHASE0_COMBINATIONS
    ]
    return build_phase0_report(measurements)


def _phase0_synthetic_documents(vocab_size: int = 17, count: int = PHYSICAL_BATCH) -> list[dict[str, Any]]:
    return [{"document_index": index, "tokens": [((index + 1) * 3 + position) % vocab_size for position in range(RETAINED_TOKENS)]} for index in range(count)]


def _phase0_model_from_oracle(seed: int, k: int, *, smoke: bool) -> OmegaCoreLMFast:
    if smoke:
        set_seed(seed)
        return OmegaCoreLMFast(vocab_size=17, dimension=4, slots=1, rounds=k, variant="shared").to(dtype=torch.float32)
    path = oracle_checkpoint_path(seed, k)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = OmegaCoreLMFast(vocab_size=TOKENIZER_VOCAB, dimension=DIMENSION, slots=SLOTS, rounds=k, variant="shared").to(dtype=torch.float32)
    model.load_state_dict(payload["model"])
    return model


def _phase0_sample_rss() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _phase0_step_documents(documents: list[Mapping[str, Any]], update: int) -> list[Mapping[str, Any]]:
    start = (update // 2) * PHYSICAL_BATCH
    return [documents[(start + offset) % len(documents)] for offset in range(PHYSICAL_BATCH)]


def _phase0_worker(combination: str, *, smoke: bool) -> dict[str, Any]:
    """Execute exactly one fresh-process Phase 0 combination.

    Distillation imports R1 lazily inside its branch. CE-only loads F and frozen
    model state only; it never imports the R1 runner or constructs a teacher.
    """
    if combination not in PHASE0_COMBINATIONS:
        raise ValueError(combination)
    distill = combination.startswith("DISTILL-")
    k = int(combination.rsplit("K", 1)[1])
    seed = SEEDS[0] if k == 1 else SEEDS[1]
    configure_cpu_runtime()
    set_seed(seed)
    teacher: torch.nn.Module | None = None
    if distill:
        # Deliberately keep R1 teacher path inside distillation worker branch.
        import run_scientific_scoping_a as r1

        if smoke:
            documents = r1.synthetic_documents(PHYSICAL_BATCH, 17)
            teacher = r1.TinyTeacher(17)
            model = r1.make_f_model(vocab_size=17, dimensions=(4, 1), variant="shared_K1" if k == 1 else "shared_K4")
        else:
            documents, _, _, _, teacher = r1._load_real_documents(pair_count=1000)
            model = r1.make_f_model(vocab_size=TOKENIZER_VOCAB, dimensions=(DIMENSION, SLOTS), variant="shared_K1" if k == 1 else "shared_K4")
    else:
        documents = _phase0_synthetic_documents() if smoke else load_real_frozen_documents()[1]
        model = _phase0_model_from_oracle(seed, k, smoke=smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
    timings = {name: 0.0 for name in ("student_forward_seconds", "vocab_loss_seconds", "backward_seconds", "clip_seconds", "adamw_seconds", "total_seconds", "teacher_forward_seconds")}
    rss_peak = _phase0_sample_rss()
    measured_teacher_calls = 0
    measured_kl = 0
    previous_state: torch.Tensor | None = None
    for update in range(PHASE0_UPDATES_PER_COMBINATION):
        batch = _phase0_step_documents(documents, update)
        source = torch.tensor([document["tokens"] for document in batch], dtype=torch.long)
        window = update % 2
        start = window * WINDOW_TOKENS
        inputs = source[:, start : start + WINDOW_TOKENS]
        targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
        state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if window == 0 else previous_state
        if state is None:
            raise RuntimeError("Phase 0 missing window state")
        optimizer.zero_grad(set_to_none=True)
        total_started = time.perf_counter()
        if distill:
            import run_scientific_scoping_a as r1

            forward_started = time.perf_counter()
            result = model.forward_window(inputs, state)
            forward_seconds = time.perf_counter() - forward_started
            next_state, logits = result[0], result[1]
            trace = result[2] if len(result) >= 3 else None
            teacher_started = time.perf_counter()
            teacher_logits = r1.teacher_window_logits(teacher, source, window)  # type: ignore[arg-type]
            teacher_seconds = time.perf_counter() - teacher_started
            loss_started = time.perf_counter()
            if trace is not None:
                loss = r1._distillation_loss_from_trace(model, trace, teacher_logits[:, :WINDOW_TOKENS], targets, torch.ones_like(targets, dtype=torch.bool))
            else:
                loss = r1.distillation_loss(logits, teacher_logits[:, :WINDOW_TOKENS], targets, torch.ones_like(targets, dtype=torch.bool))["total"]
            loss_seconds = time.perf_counter() - loss_started
            if update >= PHASE0_WARMUP_UPDATES:
                timings["student_forward_seconds"] += forward_seconds
                timings["teacher_forward_seconds"] += teacher_seconds
                timings["vocab_loss_seconds"] += loss_seconds
                measured_teacher_calls += 1
                measured_kl += 1
        else:
            forward_started = time.perf_counter()
            result = model.recur_states(inputs, state)
            forward_seconds = time.perf_counter() - forward_started
            next_state, _, _, readout_states = result
            loss_started = time.perf_counter()
            losses = ce_only_loss(model, readout_states, targets)
            loss_seconds = time.perf_counter() - loss_started
            loss = losses["total"]
            if update >= PHASE0_WARMUP_UPDATES:
                timings["student_forward_seconds"] += forward_seconds
                timings["vocab_loss_seconds"] += loss_seconds
        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_started
        clip_started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        clip_seconds = time.perf_counter() - clip_started
        adamw_started = time.perf_counter()
        optimizer.step()
        adamw_seconds = time.perf_counter() - adamw_started
        total_seconds = time.perf_counter() - total_started
        rss_peak = max(rss_peak, _phase0_sample_rss())
        if update >= PHASE0_WARMUP_UPDATES:
            timings["backward_seconds"] += backward_seconds
            timings["clip_seconds"] += clip_seconds
            timings["adamw_seconds"] += adamw_seconds
            timings["total_seconds"] += total_seconds
        previous_state = next_state.detach() if window == 0 else None
    row = phase0_measurement(
        combination,
        **timings,
        rss_bytes_peak=rss_peak,
        teacher_loaded=distill,
    )
    row.update({"worker_pid": os.getpid(), "teacher_imported": distill, "updates_executed": PHASE0_UPDATES_PER_COMBINATION, "teacher_forward_calls": measured_teacher_calls, "kl_calculations": measured_kl})
    return row


def phase0_worker_main(combination: str, *, smoke: bool, confirm_real_execution: bool) -> int:
    if not smoke:
        require_real_authorization(confirm_real_execution, "Phase 0 worker real execution")
    row = _phase0_worker(combination, smoke=smoke)
    print(json.dumps(row, sort_keys=True))
    return 0


def run_phase0_subprocesses(output_dir: Path, *, confirm_real_execution: bool, smoke: bool = False) -> dict[str, Any]:
    """Run four isolated workers, then build report from their measured rows."""
    if not smoke:
        require_real_authorization(confirm_real_execution, "Phase 0 real execution")
    rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for combination in PHASE0_COMBINATIONS:
        command = [sys.executable, str(Path(__file__).resolve()), PHASE0_WORKER_FLAG, "--combination", combination]
        if smoke:
            command.append("--smoke-worker")
        else:
            command.append("--confirm-real-execution")
        commands.append(command)
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Phase 0 worker failed for {combination}: {completed.stderr}")
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not output_lines:
            raise RuntimeError(f"Phase 0 worker emitted no result for {combination}")
        row = json.loads(output_lines[-1])
        rows.append(row)
    report = build_phase0_report(rows)
    report["worker_commands"] = commands
    report = _self_hashed({key: value for key, value in report.items() if key != "report_self_hash"})
    write_json(output_dir / "phase0_report.json", report)
    return report


def _curve_by_update(curve: Iterable[Mapping[str, Any]]) -> dict[int, float]:
    return {int(point["update"]): float(point["nll"]) for point in curve}


def classify_deltas(final_deltas: Iterable[float], ceiling: float = NONINFERIOR_CEILING) -> str:
    values = list(float(value) for value in final_deltas)
    if not values:
        raise ValueError("cannot classify empty delta set")
    if all(value <= ceiling for value in values):
        return "NONINFERIOR"
    if all(value > ceiling for value in values):
        return "INFERIOR"
    return "MIXED"


def equal_cost_plan(classification: str, u_equal_cost: int) -> dict[str, Any]:
    if classification not in {"NONINFERIOR", "INFERIOR", "MIXED"}:
        raise ValueError(classification)
    required = classification != "NONINFERIOR"
    return {
        "required": required,
        "scheduled": required,
        "executed": False,
        "updates": int(u_equal_cost) if required else None,
        "reason": "automatic continuation when Phase A is not NONINFERIOR" if required else "not required",
    }


def classify_equal_cost_deltas(final_deltas: Iterable[float], ceiling: float = NONINFERIOR_CEILING) -> str:
    values = list(float(value) for value in final_deltas)
    if not values:
        raise ValueError("cannot classify empty equal-cost delta set")
    if all(value <= ceiling for value in values):
        return "CE-ONLY-COST-NONINFERIOR"
    if all(value > ceiling for value in values):
        return "DISTILLATION-COST-ADVANTAGE"
    return "COST-MIXED"


def build_equal_cost_comparison(
    ce_cost_results: Iterable[Mapping[str, Any]],
    baseline_curves: Mapping[tuple[int, int], Iterable[Mapping[str, Any]]],
    *,
    u_equal_cost: int,
    ceiling: float = NONINFERIOR_CEILING,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in ce_cost_results:
        seed = int(result["seed"])
        k = int(result["K"])
        curve = _curve_by_update(result["validation_curve"])
        baseline = _curve_by_update(baseline_curves[(seed, k)])
        if not curve:
            raise ValueError("equal-cost result has empty validation curve")
        final_update = max(curve)
        distill_nll = baseline.get(TOTAL_UPDATES)
        if distill_nll is None:
            raise ValueError("R1 baseline missing update 2000")
        rows.append({
            "seed": seed,
            "K": k,
            "update": final_update,
            "ce_nll": curve[final_update],
            "distill_nll_at_2000": distill_nll,
            "delta_ce_minus_distill_at_equal_cost": curve[final_update] - distill_nll,
        })
    deltas = [row["delta_ce_minus_distill_at_equal_cost"] for row in rows]
    return {
        "target_update": int(u_equal_cost),
        "ce_noninferior_ceiling": ceiling,
        "combinations": rows,
        "classification": classify_equal_cost_deltas(deltas, ceiling),
    }


def run_equal_cost_continuation(
    phase_a_results: Iterable[Mapping[str, Any]],
    *,
    u_equal_cost: int,
    continuation_runner: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Continue all four CE paths with unchanged configuration after Phase A."""
    if u_equal_cost < TOTAL_UPDATES:
        raise ValueError("U_equal_cost is before checkpoint_2000; no continuation is executable")
    results = list(phase_a_results)
    if len(results) != 4:
        raise ValueError("equal-cost continuation requires four Phase A CE results")
    return [continuation_runner(result, u_equal_cost) for result in results]


def build_phase_a_comparison(
    ce_results: Iterable[Mapping[str, Any]],
    baseline_curves: Mapping[tuple[int, int], Iterable[Mapping[str, Any]]],
    *,
    u_equal_cost: int,
    ceiling: float = NONINFERIOR_CEILING,
) -> dict[str, Any]:
    combinations: list[dict[str, Any]] = []
    for result in ce_results:
        seed = int(result["seed"])
        k = int(result["K"])
        ce_curve = _curve_by_update(result["validation_curve"])
        r1_curve = _curve_by_update(baseline_curves[(seed, k)])
        if set(ce_curve) != set(BOUNDARIES) or set(r1_curve) != set(BOUNDARIES):
            raise ValueError("Phase A comparison requires five frozen boundaries")
        deltas = [{"update": update, "delta_ce_minus_distill": ce_curve[update] - r1_curve[update]} for update in BOUNDARIES]
        combinations.append({"seed": seed, "K": k, "deltas": deltas, "delta_at_2000": deltas[-1]["delta_ce_minus_distill"]})
    final_deltas = [row["delta_at_2000"] for row in combinations]
    classification = classify_deltas(final_deltas, ceiling)
    delta_ce_k1_minus_k4: list[dict[str, Any]] = []
    for seed in SEEDS:
        k1 = next(row for row in combinations if row["seed"] == seed and row["K"] == 1)
        k4 = next(row for row in combinations if row["seed"] == seed and row["K"] == 4)
        delta_ce_k1_minus_k4.append({
            "seed": seed,
            "boundaries": [
                {"update": left["update"], "delta_ce_k1_minus_k4": left["delta_ce_minus_distill"] - right["delta_ce_minus_distill"]}
                for left, right in zip(k1["deltas"], k4["deltas"])
            ],
        })
    return {
        "classification_at_2000": classification,
        "ce_noninferior_ceiling": ceiling,
        "combinations": combinations,
        "informational_Delta_CE_K1_minus_K4": delta_ce_k1_minus_k4,
        "equal_cost_continuation": equal_cost_plan(classification, u_equal_cost),
    }


def load_r1_baseline_curves() -> dict[tuple[int, int], list[dict[str, Any]]]:
    curves: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for seed in SEEDS:
        for k in KS:
            path = R1_FULL_DIR / "runs" / f"shared_K{k}_seed_{seed}" / "validation_curve.json"
            curve = json.loads(path.read_text(encoding="utf-8"))
            curves[(seed, k)] = [point for point in curve if int(point["update"]) in BOUNDARIES]
    return curves


def _manifest_key(document: Mapping[str, Any]) -> tuple[str, str]:
    return str(document["full_text_sha256"]), str(document["retained_513_token_sha256"])


def _load_frozen_split_tokens(dataset: Any, tokenizer: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recover token payloads by matching source rows to frozen records.

    The manifest remains authority for order, dedupe, and selection.  This
    helper never creates or rewrites a manifest and rejects any source drift.
    """
    from run_omega_core_lm_0_r1_training_technical_preflight import reconstruct_documents, sha256_text

    expected = list(manifest["documents"])
    expected_keys = [_manifest_key(document) for document in expected]
    expected_by_key = {key: document for key, document in zip(expected_keys, expected)}
    recovered: dict[tuple[str, str], dict[str, Any]] = {}
    for document_index, source in enumerate(reconstruct_documents(dataset)):
        token_ids = list(tokenizer.encode(str(source["text"]), add_special_tokens=False))
        if len(token_ids) < RETAINED_TOKENS:
            continue
        retained = token_ids[:RETAINED_TOKENS]
        token_hash = hashlib.sha256(b"".join(int(token).to_bytes(4, "little") for token in retained)).hexdigest()
        key = (sha256_text(str(source["text"])), token_hash)
        if key in expected_by_key and key not in recovered:
            recovered[key] = {
                **expected_by_key[key],
                "document_index": int(expected_by_key[key]["document_index"]),
                "tokens": retained,
                "source_document_index": document_index,
            }
    if list(recovered) != expected_keys:
        missing = [key for key in expected_keys if key not in recovered]
        raise ValueError(f"frozen manifest source mismatch; missing {len(missing)} documents")
    return [recovered[key] for key in expected_keys]


def _load_secondary_tokens(dataset: Any, tokenizer: Any, excluded: set[tuple[str, str]]) -> list[dict[str, Any]]:
    from run_omega_core_lm_0_r1_training_technical_preflight import reconstruct_documents, sha256_text

    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document_index, source in enumerate(reconstruct_documents(dataset)):
        token_ids = list(tokenizer.encode(str(source["text"]), add_special_tokens=False))
        if len(token_ids) < RETAINED_TOKENS:
            continue
        retained = token_ids[:RETAINED_TOKENS]
        key = (
            sha256_text(str(source["text"])),
            hashlib.sha256(b"".join(int(token).to_bytes(4, "little") for token in retained)).hexdigest(),
        )
        if key in excluded or key in seen:
            continue
        seen.add(key)
        documents.append({"document_index": document_index, "tokens": retained, "full_text_sha256": key[0], "retained_513_token_sha256": key[1]})
        if len(documents) == 60:
            break
    if len(documents) != 60:
        raise ValueError(f"secondary U2 validation requires 60 documents, got {len(documents)}")
    return documents


def load_real_frozen_documents() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only token data needed by CE training, without constructing teacher."""
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer
    from run_omega_core_lm_0_r1_training_technical_preflight import (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_REVISION,
    )

    manifests = load_frozen_manifests()
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilgpt2",
        revision="2290a62682d06624634c1f46a6ad5be0f47f38aa",
        use_fast=True,
        local_files_only=True,
    )
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise ValueError(f"tokenizer vocab mismatch: {len(tokenizer)}")
    config = DownloadConfig(local_files_only=True)
    train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=config)
    validation_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION, download_config=config)
    train_documents = _load_frozen_split_tokens(train_dataset, tokenizer, manifests["train"])
    validation_documents = _load_frozen_split_tokens(validation_dataset, tokenizer, manifests["validation"])
    train_keys = {_manifest_key(document) for document in manifests["train"]["documents"]}
    secondary_documents = _load_secondary_tokens(validation_dataset, tokenizer, train_keys)
    return manifests, train_documents, validation_documents, secondary_documents


def evaluate_ce_only(model: OmegaCoreLMFast, documents: list[Mapping[str, Any]]) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for document in documents:
            source = torch.tensor([document["tokens"]], dtype=torch.long)
            state = model.initial_state(1, device=torch.device("cpu"))
            for window in (0, 1):
                start = window * WINDOW_TOKENS
                inputs = source[:, start : start + WINDOW_TOKENS]
                targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
                state, _, losses = ce_only_forward_loss(model, inputs, targets, state)
                count = int(targets.numel())
                total_nll += float(losses["ce"].item()) * count
                total_tokens += count
    nll = total_nll / max(1, total_tokens)
    return {"nll": nll, "tokens": total_tokens, "finite": math.isfinite(nll)}


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, update: int, config: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"update": update, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": dict(config)}, temporary)
    os.replace(temporary, path)


def run_ce_training(
    *,
    run_dir: Path,
    seed: int,
    k: int,
    train_documents: list[Mapping[str, Any]],
    validation_documents: list[Mapping[str, Any]],
    secondary_documents: list[Mapping[str, Any]] | None = None,
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    total_updates: int = TOTAL_UPDATES,
    model_factory: Callable[[int, int], OmegaCoreLMFast] = fresh_model,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run one authorized CE-only path; caller must perform initialization gate first."""
    if total_updates <= 0 or total_updates % 2:
        raise ValueError("total_updates must be positive and even")
    if len(train_documents) != 602 or len(validation_documents) != 8:
        raise ValueError("real CE training requires frozen 602/8 document sets")
    if resume_checkpoint is not None and total_updates <= TOTAL_UPDATES:
        raise ValueError("continuation target must exceed checkpoint_2000")
    boundaries = tuple(sorted(set(update for update in (*BOUNDARIES, total_updates) if update <= total_updates)))
    model = model_factory(seed, k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
    config = {
        "campaign_id": CAMPAIGN_ID,
        "addendum": ADDENDUM,
        "commit_reference": COMMIT_REFERENCE,
        "seed": seed,
        "K": k,
        "variant": f"shared_K{k}",
        "dimension": DIMENSION,
        "slots": SLOTS,
        "loss": "full_cross_entropy_only",
        "teacher_loaded": False,
        "teacher_forward_calls": 0,
        "kl_calculations": 0,
        "chunk_size": CHUNK_SIZE,
        "policy": validate_policy(),
        "train_manifest_sha256": train_manifest["manifest_sha256"],
        "validation_manifest_sha256": validation_manifest["manifest_sha256"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "train_manifest.json", train_manifest)
    write_json(run_dir / "validation_manifest.json", validation_manifest)
    start_update = 0
    if resume_checkpoint is None:
        curve: list[dict[str, Any]] = [{"update": 0, **evaluate_ce_only(model, validation_documents)}]
    else:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("update", -1)) != TOTAL_UPDATES:
            raise ValueError("equal-cost continuation must resume checkpoint_02000")
        if payload.get("config", {}).get("seed") != seed or payload.get("config", {}).get("K") != k:
            raise ValueError("equal-cost continuation seed/K mismatch")
        for key in ("loss", "chunk_size", "policy", "dimension", "slots"):
            if payload.get("config", {}).get(key) != config.get(key):
                raise ValueError(f"equal-cost continuation hyperparameter drift: {key}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_update = TOTAL_UPDATES
        curve_path = resume_checkpoint.parent / "validation_curve.json"
        if curve_path.is_file():
            previous = json.loads(curve_path.read_text(encoding="utf-8"))
            curve = previous.get("curve", previous) if isinstance(previous, (dict, list)) else []
        else:
            curve = [{"update": TOTAL_UPDATES, **evaluate_ce_only(model, validation_documents)}]
    secondary_validation: dict[str, Any] | None = None
    if resume_checkpoint is None:
        _save_checkpoint(run_dir / "checkpoint_00000.pt", model, optimizer, 0, config)
    pair_state: torch.Tensor | None = None
    model.train()
    ledger: list[dict[str, Any]] = []
    for source_update in range(start_update, total_updates):
        pair_index = source_update // 2
        window = source_update % 2
        pair_rows = train_manifest["cyclic_pairs"]["pairs"]
        pair = pair_rows[pair_index % len(pair_rows)]
        documents = [train_documents[int(index)] for index in pair["document_indices"]]
        source = torch.tensor([document["tokens"] for document in documents], dtype=torch.long)
        start = window * WINDOW_TOKENS
        inputs = source[:, start : start + WINDOW_TOKENS]
        targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
        state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if window == 0 else (pair_state.detach() if pair_state is not None else None)
        if state is None:
            raise RuntimeError("missing detached window-0 state")
        optimizer.zero_grad(set_to_none=True)
        next_state, _, losses = ce_only_forward_loss(model, inputs, targets, state)
        loss = losses["total"]
        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite CE loss")
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
        optimizer.step()
        pair_state = next_state.detach() if window == 0 else None
        completed = source_update + 1
        ledger.append({
            "update": completed,
            "window": window,
            "pair": pair_index,
            "input_shape": [PHYSICAL_BATCH, WINDOW_TOKENS],
            "student_forward_calls": 1,
            "teacher_forward_calls": 0,
            "kl_calculations": 0,
            "teacher_loaded": False,
            "backward_calls": 1,
            "clip_calls": 1,
            "optimizer_steps": 1,
            "loss": float(loss.detach().item()),
            "pre_clip_grad_norm": pre_clip,
        })
        if completed in boundaries:
            model.eval()
            curve.append({"update": completed, **evaluate_ce_only(model, validation_documents)})
            if completed == total_updates and secondary_documents is not None:
                secondary_validation = evaluate_ce_only(model, secondary_documents)
            model.train()
            _save_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", model, optimizer, completed, config)
    write_json(run_dir / "validation_curve.json", {"curve": curve})
    write_json(run_dir / "ledger.json", {"updates": ledger, "teacher_loaded": False, "teacher_forward_calls": 0, "kl_calculations": 0})
    return {"seed": seed, "K": k, "variant": f"shared_K{k}", "validation_curve": curve, "run_dir": run_dir.as_posix(), "start_update": start_update, "end_update": total_updates, "teacher_loaded": False, "teacher_forward_calls": 0, "kl_calculations": 0, "secondary_validation": secondary_validation}


def _ngram_frequencies(tokens: list[int], n: int) -> dict[str, int]:
    counts: dict[tuple[int, ...], int] = {}
    for index in range(max(0, len(tokens) - n + 1)):
        gram = tuple(tokens[index : index + n])
        counts[gram] = counts.get(gram, 0) + 1
    return {" ".join(map(str, gram)): count for gram, count in sorted(counts.items()) if count > 1}


def _cycle_lengths(tokens: list[int]) -> list[int]:
    return [period for period in range(1, min(8, len(tokens)) + 1) if len(tokens) >= 2 * period and tokens[-2 * period : -period] == tokens[-period:]]


def generation_metrics(tokens: list[int], eos_position: int | None) -> dict[str, Any]:
    max_run = 0
    current_run = 0
    previous: int | None = None
    for token in tokens:
        current_run = current_run + 1 if token == previous else 1
        max_run = max(max_run, current_run)
        previous = token
    total = len(tokens)
    return {
        "unique_token_proportion": len(set(tokens)) / total if total else 0.0,
        "distinct_1": len(set(tokens)) / total if total else 0.0,
        "distinct_2": len(set(zip(tokens, tokens[1:]))) / max(1, total - 1),
        "max_same_token_run": max_run,
        "repeated_ngram_frequencies": {str(n): _ngram_frequencies(tokens, n) for n in (2, 3, 4)},
        "exact_cycle_lengths_1_to_8": _cycle_lengths(tokens),
        "eos_position": eos_position,
    }


def _ce_generation_logits(model: Any, tokens: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = model.recur_states(tokens, state)
    if len(result) == 4:
        next_state, _, _, readout_states = result
    else:
        next_state, readout_states = result
    projected = model.project(readout_states[:, -1:])
    logits = model.logits_from_projected(projected)
    return next_state, logits[:, -1, :] if logits.ndim == 3 else logits


def generate_ce_one(
    model: Any,
    prompt_token_ids: Iterable[int],
    tokenizer: Any,
    *,
    metadata: Mapping[str, Any],
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Greedy CE generation using zero state and CE lexical projection path."""
    prompt = [int(token) for token in prompt_token_ids]
    if len(prompt) != GENERATION_PROMPT_LENGTH:
        raise ValueError("generation prompt must contain exactly 32 tokens")
    model.eval()
    generated: list[int] = []
    eos_position: int | None = None
    with torch.inference_mode():
        state = model.initial_state(1, device=torch.device("cpu"))
        state, logits = _ce_generation_logits(model, torch.tensor([prompt], dtype=torch.long), state)
        for index in range(max_new_tokens):
            token = int(torch.argmax(logits, dim=-1).item())
            generated.append(token)
            if token == EOS_TOKEN_ID:
                eos_position = index + 1
                break
            state, logits = _ce_generation_logits(model, torch.tensor([[token]], dtype=torch.long), state)
    decoded = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
    return {
        **dict(metadata),
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "decoded_generation": decoded,
        "generated_length": len(generated),
        "stop_reason": "EOS" if eos_position is not None else "MAX_LENGTH",
        "metrics": generation_metrics(generated, eos_position),
    }


def run_ce_generation_audit(
    model_specs: Iterable[Mapping[str, Any]],
    prompts: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    model_loader: Callable[[Mapping[str, Any]], Any],
    *,
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    spec_rows = list(model_specs)
    prompt_rows = list(prompts)
    for spec in spec_rows:
        model = model_loader(spec)
        for prompt in prompt_rows:
            metadata = {
                "architecture": "CE-only",
                "K": int(spec["K"]),
                "seed": int(spec["seed"]),
                "checkpoint_update": int(spec.get("checkpoint_update", TOTAL_UPDATES)),
                "prompt_document_id": str(prompt["document_id"]),
                "prompt_document_hash": str(prompt["document_hash"]),
            }
            records.append(generate_ce_one(model, prompt["token_ids"], tokenizer, metadata=metadata, max_new_tokens=max_new_tokens))
    if len(records) != EXPECTED_GENERATION_RECORDS:
        raise ValueError(f"CE generation audit requires 32 records, got {len(records)}")
    return {
        "documentary_only": True,
        "training_performed": False,
        "quality_gate": None,
        "protocol": {"prompt_count": len(prompt_rows), "models": len(spec_rows), "generation_count": len(records), "zero_state": True, "greedy": True, "max_new_tokens": max_new_tokens, "eos_token_id": EOS_TOKEN_ID},
        "generations": records,
    }


def compare_generation_records(ce_records: Iterable[Mapping[str, Any]], historical_records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    historical = {(int(row["K"]), int(row["seed"]), str(row["prompt_document_id"])): row for row in historical_records}
    comparisons: list[dict[str, Any]] = []
    for row in ce_records:
        key = (int(row["K"]), int(row["seed"]), str(row["prompt_document_id"]))
        reference = historical.get(key)
        if reference is None:
            raise ValueError(f"historical generation missing {key}")
        comparisons.append({
            "K": key[0],
            "seed": key[1],
            "prompt_document_id": key[2],
            "ce_generated_token_ids": list(row["generated_token_ids"]),
            "historical_r1_generated_token_ids": list(reference["generated_token_ids"]),
            "ce_metrics": row["metrics"],
            "historical_r1_metrics": reference["metrics"],
        })
    if len(comparisons) != EXPECTED_GENERATION_RECORDS:
        raise ValueError("generation comparison requires 32 CE records")
    return {"documentary_only": True, "quality_gate": None, "comparison_count": len(comparisons), "comparisons": comparisons}


def load_historical_generation_records(repo_root: Path = REPO_ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    command = ["git", "show", f"{HISTORICAL_GENERATION_COMMIT}:{GENERATION_AUDIT_RELATIVE}"]
    completed = subprocess.run(command, cwd=repo_root, check=True, capture_output=True)
    historical = json.loads(completed.stdout.decode("utf-8"))
    records = [row for row in historical.get("generations", []) if row.get("architecture") == "R1"]
    if len(records) != 32:
        raise ValueError(f"historical R1 generation audit must contain 32 outputs, got {len(records)}")
    if any(int(row.get("checkpoint_update", -1)) != 2000 for row in records):
        raise ValueError("generation audit includes non-2000 checkpoint")
    return historical, records


def build_generation_audit_report(
    *,
    repo_root: Path = REPO_ROOT,
    ce_generation_report: Mapping[str, Any] | None = None,
    equal_cost_records: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    historical, records = load_historical_generation_records(repo_root)
    report: dict[str, Any] = {
        "schema": "omega-ce-only-baseline-generation-audit-v1",
        "campaign_id": CAMPAIGN_ID,
        "documentary_only": True,
        "training_performed": False,
        "generation_source_commit": HISTORICAL_GENERATION_COMMIT,
        "generation_source_path": GENERATION_AUDIT_RELATIVE,
        "protocol": {
            "prompt_count": 8,
            "models": 4,
            "generation_count": 32,
            "decoding": historical.get("decoding"),
            "max_new_tokens": historical.get("max_new_tokens", 64),
            "zero_state": True,
            "greedy_argmax": True,
            "eos_token_id": EOS_TOKEN_ID,
            "metrics": ["unique_token_proportion", "distinct_1", "distinct_2", "max_same_token_run", "repeated_ngram_frequencies", "exact_cycle_lengths_1_to_8", "eos_position"],
        },
        "historical_generations": records,
        "ce_generations": list(ce_generation_report.get("generations", [])) if ce_generation_report else None,
        "ce_vs_historical_comparison": compare_generation_records(ce_generation_report["generations"], records) if ce_generation_report else None,
        "optional_equal_cost_audit": equal_cost_records,
    }
    return _self_hashed(report)


def run_ce_generation_from_checkpoints(ce_root: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run guarded CE generation from four Phase A checkpoint_02000 files."""
    historical, historical_r1 = load_historical_generation_records(repo_root)
    prompts: list[dict[str, Any]] = []
    seen_prompt_ids: set[str] = set()
    for row in historical_r1:
        prompt_id = str(row["prompt_document_id"])
        if prompt_id in seen_prompt_ids:
            continue
        seen_prompt_ids.add(prompt_id)
        prompts.append({"document_id": prompt_id, "document_hash": row["prompt_document_hash"], "token_ids": row["prompt_token_ids"]})
    if len(prompts) != 8:
        raise ValueError("historical generation audit did not provide eight prompts")
    specs = [{"K": k, "seed": seed, "checkpoint_update": TOTAL_UPDATES} for seed in SEEDS for k in KS]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2", revision="2290a62682d06624634c1f46a6ad5be0f47f38aa", use_fast=True, local_files_only=True)

    def loader(spec: Mapping[str, Any]) -> OmegaCoreLMFast:
        path = ce_root / "runs" / f"CE-K{int(spec['K'])}_seed_{int(spec['seed'])}" / "checkpoint_02000.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = OmegaCoreLMFast(vocab_size=TOKENIZER_VOCAB, dimension=DIMENSION, slots=SLOTS, rounds=int(spec["K"]), variant="shared")
        model.load_state_dict(payload["model"])
        return model

    generation = run_ce_generation_audit(specs, prompts, tokenizer, loader)
    generation["historical_source_commit"] = HISTORICAL_GENERATION_COMMIT
    generation["historical_source_rows"] = len(historical_r1)
    return generation


def require_real_authorization(confirm: bool, action: str) -> None:
    if not confirm:
        raise RealExecutionAuthorizationError(f"{action} requires --confirm-real-execution")


def run_phase_a_real(output_dir: Path, *, confirm_real_execution: bool, phase0_report_path: Path | None = None) -> dict[str, Any]:
    """Real CE-only entrypoint; never reachable without explicit authorization."""
    require_real_authorization(confirm_real_execution, "Phase A real execution")
    if phase0_report_path is None:
        raise ValueError("Phase A requires frozen Phase 0 report for equal-cost planning")
    phase0_report = json.loads(phase0_report_path.read_text(encoding="utf-8"))
    if not verify_self_hash(phase0_report):
        raise ValueError("Phase 0 report self-hash verification failed")
    manifests = load_frozen_manifests()
    gate = initialization_gate()
    loaded_manifests, train_documents, validation_documents, secondary_documents = load_real_frozen_documents()
    if loaded_manifests["train"]["manifest_sha256"] != manifests["train"]["manifest_sha256"]:
        raise ValueError("frozen manifest changed between gate and data load")
    baselines = load_r1_baseline_curves()
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for k in KS:
            results.append(
                run_ce_training(
                    run_dir=output_dir / "runs" / f"CE-K{k}_seed_{seed}",
                    seed=seed,
                    k=k,
                    train_documents=train_documents,
                    validation_documents=validation_documents,
                    secondary_documents=secondary_documents,
                    train_manifest=manifests["train"],
                    validation_manifest=manifests["validation"],
                )
            )
    comparison = build_phase_a_comparison(results, baselines, u_equal_cost=int(phase0_report["U_equal_cost"]))
    comparison["equal_cost_continuation"]["u_equal_cost_source"] = phase0_report_path.as_posix()
    if comparison["classification_at_2000"] != "NONINFERIOR":
        target = int(phase0_report["U_equal_cost"])
        if target >= TOTAL_UPDATES:
            def continue_one(result: Mapping[str, Any], update_target: int) -> Mapping[str, Any]:
                seed = int(result["seed"])
                k = int(result["K"])
                return run_ce_training(
                    run_dir=output_dir / "equal_cost" / f"CE-K{k}_seed_{seed}",
                    seed=seed,
                    k=k,
                    train_documents=train_documents,
                    validation_documents=validation_documents,
                    secondary_documents=None,
                    train_manifest=manifests["train"],
                    validation_manifest=manifests["validation"],
                    total_updates=update_target,
                    resume_checkpoint=Path(result["run_dir"]) / "checkpoint_02000.pt",
                )

            cost_results = run_equal_cost_continuation(results, u_equal_cost=target, continuation_runner=continue_one)
            comparison["equal_cost_continuation"]["executed"] = True
            comparison["equal_cost_result"] = build_equal_cost_comparison(cost_results, baselines, u_equal_cost=target)
        else:
            comparison["equal_cost_continuation"]["executed"] = False
            comparison["equal_cost_continuation"]["reason"] = "U_equal_cost is before checkpoint_2000; existing CE@2000 is already beyond target"
            comparison["equal_cost_result"] = build_equal_cost_comparison(results, baselines, u_equal_cost=target)
    report = {
        "schema": "omega-ce-only-baseline-phase-a-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "addendum": ADDENDUM,
        "commit_reference": COMMIT_REFERENCE,
        "phase0_report": phase0_report_path.as_posix(),
        "training_performed": True,
        "initialization_gate": gate,
        "frozen_manifests": manifests["paths"],
        "baseline_r1_validation_curves": {
            f"seed_{seed}_K{k}": str(R1_FULL_DIR / "runs" / f"shared_K{k}_seed_{seed}" / "validation_curve.json")
            for seed in SEEDS
            for k in KS
        },
        "primary_validation": {"documents": 8, "boundaries": list(BOUNDARIES)},
        "secondary_validation": {
            "documents": len(secondary_documents),
            "update": TOTAL_UPDATES,
            "observational_only": True,
            "reclassification": False,
        },
        "results": results,
        "comparison": comparison,
    }
    report = _self_hashed(report)
    write_json(output_dir / "phase_a_report.json", report)
    return report


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="emit synthetic Phase 0 report only")
    parser.add_argument("--phase0", action="store_true")
    parser.add_argument("--phase-a", action="store_true")
    parser.add_argument("--generation-audit", action="store_true")
    parser.add_argument(PHASE0_WORKER_FLAG, action="store_true")
    parser.add_argument("--combination", choices=PHASE0_COMBINATIONS)
    parser.add_argument("--smoke-worker", action="store_true")
    parser.add_argument("--confirm-real-execution", action="store_true")
    parser.add_argument("--phase0-measurements", type=Path)
    parser.add_argument("--phase0-report", type=Path)
    parser.add_argument("--ce-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.phase0_worker:
        if args.combination is None:
            parser.error("--phase0-worker requires --combination")
        return phase0_worker_main(args.combination, smoke=args.smoke_worker, confirm_real_execution=args.confirm_real_execution)
    selected = sum(bool(value) for value in (args.smoke, args.phase0, args.phase_a, args.generation_audit))
    if selected != 1:
        parser.error("select exactly one of --smoke, --phase0, --phase-a, --generation-audit")
    if args.smoke:
        report = synthetic_phase0_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.generation_audit:
        require_real_authorization(args.confirm_real_execution, "generation audit")
        ce_generation = run_ce_generation_from_checkpoints(args.ce_root) if args.ce_root is not None else None
        report = build_generation_audit_report(ce_generation_report=ce_generation)
        write_json(args.output_dir / "generation_audit_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.phase_a:
        report = run_phase_a_real(args.output_dir, confirm_real_execution=args.confirm_real_execution, phase0_report_path=args.phase0_report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    report = run_phase0_subprocesses(args.output_dir, confirm_real_execution=args.confirm_real_execution, smoke=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
