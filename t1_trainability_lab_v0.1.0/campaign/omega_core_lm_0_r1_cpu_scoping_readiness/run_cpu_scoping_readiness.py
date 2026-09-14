"""Run the authorized CPU-only OMEGA R1 scoping unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT / "campaign" / "omega_core_lm_0_gpu_environment_preparation"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from omega_nominal_microbatch_runner import (  # noqa: E402
    APPROVED_SELECTION_MANIFEST,
    EFFECTIVE_BATCH,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_TOKENS,
    TECHNICAL_VARIANT_SEEDS,
    TOKENS_PER_WINDOW,
    WINDOWS,
    OmegaCoreLM0R1Technical,
    RunLedger,
    _preflight_source,
    backend_options,
    construct_adamw,
    memory_observed,
    microbatch_count_for_physical_batch,
    microbatch_update,
    set_technical_seed,
    state_source_update_for_update,
    update_window_schedule,
)
from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    TOKENIZER_VOCAB,
)


CAMPAIGN_ID = "OMEGA-CORE-LM-0-R1-CPU-SCOPING-READINESS"
DEFAULT_CACHE_ROOT = Path(r"C:\Users\danil\.cache\huggingface")
CONFIGS = (("A", 2), ("B", 4), ("C", 8))
VARIANTS = (("shared_K1", "shared", 1), ("shared_K4", "shared", 4))
STAGE2_UPDATES = 6
STAGE3_UPDATES = 4
SCOPE_SEEDS = 2
SCOPE_UPDATES = 2000
SYNTHETIC_SEED = 9100
SYNTHETIC_VOCAB = 17
SYNTHETIC_DIMENSION = 16
SYNTHETIC_SLOTS = 4
PROFILE_PHASES = (
    "teacher_forward_seconds",
    "student_prelude_seconds",
    "student_recurrent_rounds_seconds",
    "readout_seconds",
    "ce_kl_softmax_seconds",
    "student_backward_seconds",
    "optimizer_step_seconds",
    "ledger_instrumentation_seconds",
)
COMPILE_MODES = ("eager", "default", "max-autotune")
COMPILE_ABS_TOLERANCE = 1e-5
COMPILE_UPDATES = 3
EXECUTION_ORDER = (1, 2, 3, "3-profile", "3-compile", 4)


class AppendOnlyJsonl:
    """Durable append-only stage ledger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value: dict[str, Any]) -> None:
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


class TinyTeacher(torch.nn.Module):
    """Small frozen teacher fixture; stage 2 never interprets language."""

    def __init__(self, vocab_size: int = SYNTHETIC_VOCAB) -> None:
        super().__init__()
        values = torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size)
        self.table = torch.nn.Parameter(values / 100.0, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return type("Output", (), {"logits": self.table[input_ids % self.table.shape[0]]})()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    unsigned = dict(report)
    unsigned["report_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(_json_safe(unsigned), indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    written = dict(report)
    written["report_self_hash"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(written), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_identity() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, float]:
    total = torch.zeros((), dtype=torch.float64)
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        total += parameter.grad.detach().double().square().sum()
    norm = float(total.sqrt().item())
    return finite and math.isfinite(norm), norm


def _synthetic_fixture(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    source = torch.arange(EFFECTIVE_BATCH * SOURCE_TOKENS, dtype=torch.long, device=device).reshape(EFFECTIVE_BATCH, SOURCE_TOKENS) % SYNTHETIC_VOCAB
    mask = torch.ones(EFFECTIVE_BATCH, TOKENS_PER_WINDOW, dtype=torch.bool, device=device)
    return source, mask, mask.clone(), [f"synthetic-document-{index}" for index in range(EFFECTIVE_BATCH)]


def _stage1_variant(*, rounds: int, variant_name: str, variant: str, warmup_tokens: int, measure_tokens: int, warmup_windows: int, measure_windows: int) -> dict[str, Any]:
    set_technical_seed(SYNTHETIC_SEED + rounds)
    student = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, rounds=rounds, variant=variant).eval()
    state = student.initial_state(1, device=torch.device("cpu"))
    token_ids = torch.randint(TOKENIZER_VOCAB, (1, warmup_tokens + measure_tokens), dtype=torch.long)
    with torch.inference_mode():
        for index in range(warmup_tokens):
            state, _ = student.forward_token(token_ids[:, index], state)
        token_started = time.perf_counter()
        for index in range(warmup_tokens, warmup_tokens + measure_tokens):
            state, _ = student.forward_token(token_ids[:, index], state)
        token_seconds = time.perf_counter() - token_started

        window_tokens = 256
        window_warmup = torch.randint(TOKENIZER_VOCAB, (1, window_tokens), dtype=torch.long)
        for _ in range(warmup_windows):
            state, _ = student.forward_window(window_warmup, state)
        window_measure = torch.randint(TOKENIZER_VOCAB, (measure_windows, window_tokens), dtype=torch.long)
        window_state = state
        window_started = time.perf_counter()
        for index in range(measure_windows):
            window_state, _ = student.forward_window(window_measure[index : index + 1], window_state)
        window_seconds = time.perf_counter() - window_started

    return {
        "variant": variant_name,
        "rounds": rounds,
        "weights": "random FP32 weights",
        "forward_token": {
            "warmup_tokens": warmup_tokens,
            "measurement_tokens": measure_tokens,
            "seconds": token_seconds,
            "tokens_per_second": measure_tokens / token_seconds if token_seconds > 0 else None,
            "persistent_state": True,
        },
        "forward_window": {
            "warmup_windows": warmup_windows,
            "measurement_windows": measure_windows,
            "tokens_per_window": window_tokens,
            "seconds": window_seconds,
            "tokens_per_second": measure_windows * window_tokens / window_seconds if window_seconds > 0 else None,
            "persistent_state": True,
        },
        "teacher_used": False,
        "loss_used": False,
        "backward_used": False,
        "language_interpretation": False,
    }


def run_stage1(*, ledger: AppendOnlyJsonl, warmup_tokens: int, measure_tokens: int, warmup_windows: int, measure_windows: int) -> dict[str, Any]:
    results = []
    for _, variant, rounds in VARIANTS:
        result = _stage1_variant(rounds=rounds, variant_name=f"shared_K{rounds}", variant=variant, warmup_tokens=warmup_tokens, measure_tokens=measure_tokens, warmup_windows=warmup_windows, measure_windows=measure_windows)
        results.append(result)
        ledger.append({"stage": 1, "record_type": "inference_benchmark", **result})
    return {
        "status": "completed",
        "contract": "CPU inference cost only; random weights; no teacher/loss/backward",
        "results": results,
    }


def _pair_records(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for start in range(0, len(updates), 2):
        first, second = updates[start : start + 2]
        seconds = float(first["elapsed_seconds"]) + float(second["elapsed_seconds"])
        pairs.append({
            "updates": [first["update"], second["update"]],
            "windows": [first["window"], second["window"]],
            "window0_seconds": float(first["elapsed_seconds"]),
            "window1_seconds": float(second["elapsed_seconds"]),
            "pair_seconds": seconds,
            "pair_seconds_per_update": seconds / 2.0,
            "valid_tokens": int(first["valid_tokens"]) + int(second["valid_tokens"]),
            "tokens_per_second": (int(first["valid_tokens"]) + int(second["valid_tokens"])) / seconds if seconds > 0 else None,
            "phase": "warmup" if start == 0 else "measured",
            "phase_timings": {
                key: float(first.get("phase_timings", {}).get(key, 0.0)) + float(second.get("phase_timings", {}).get(key, 0.0))
                for key in PROFILE_PHASES
            },
        })
    return pairs


def _run_training_updates(*, run_root: Path, run_id: str, stage: int, variant_name: str, model_variant: str, rounds: int, seed: int, physical_batch: int, source: torch.Tensor, valid_mask: torch.Tensor, input_valid_mask: torch.Tensor, document_ids: list[str], teacher: torch.nn.Module, updates: int, source_manifest: dict[str, Any] | None, profile_phases: bool = False) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    set_technical_seed(seed)
    vocab_size = SYNTHETIC_VOCAB if stage == 2 else TOKENIZER_VOCAB
    student = OmegaCoreLM0R1Technical(vocab_size=vocab_size, dimension=SYNTHETIC_DIMENSION if stage == 2 else 128, slots=SYNTHETIC_SLOTS if stage == 2 else 8, rounds=rounds, variant=model_variant).to(dtype=torch.float32)
    optimizer = construct_adamw(student)
    ledger = RunLedger(run_root / "runs" / f"{run_id}-{variant_name}-b{physical_batch}", f"{run_id}-{variant_name}-b{physical_batch}")
    state: torch.Tensor | None = None
    update_results: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    for update, window in enumerate(update_window_schedule(updates)):
        if window == 0:
            state = None
        phase_timings = {key: 0.0 for key in PROFILE_PHASES} if profile_phases else None
        ledger.profile = phase_timings
        state, metrics = microbatch_update(
            ledger=ledger,
            run_id=f"{run_id}-{variant_name}-b{physical_batch}",
            variant=variant_name,
            update=update,
            student=student,
            teacher=teacher,
            source=source,
            valid_mask=valid_mask,
            input_valid_mask=input_valid_mask,
            optimizer=optimizer,
            window=window,
            persistent_state=state,
            physical_batch=physical_batch,
            document_ids=document_ids,
            source_manifest=source_manifest,
            state_source_update=state_source_update_for_update(update),
            profile=phase_timings,
        )
        memory = memory_observed(torch.device("cpu"))
        explicit_grad_norm = float(metrics["pre_clip_grad_norm"])
        finite_grad = math.isfinite(explicit_grad_norm)
        update_results.append({
            "update": update,
            "window": window,
            "elapsed_seconds": float(metrics["elapsed_seconds"]),
            "total_update_seconds": float(metrics["elapsed_seconds"]),
            "valid_tokens": int(metrics["valid_tokens"]),
            "finite_loss": all(math.isfinite(float(value)) for value in metrics["loss_summary"].values()),
            "finite_gradients": finite_grad,
            "pre_clip_grad_norm": float(metrics["pre_clip_grad_norm"]),
            "explicit_grad_norm": explicit_grad_norm,
            "microbatches": int(metrics["microbatches"]),
            "memory": memory,
            "phase_timings": phase_timings or {},
        })
        memories.append(memory)
    return {
        "variant": variant_name,
        "rounds": rounds,
        "physical_batch": physical_batch,
        "microbatches": microbatch_count_for_physical_batch(physical_batch),
        "effective_batch": EFFECTIVE_BATCH,
        "updates": update_results,
        "pairs": _pair_records(update_results),
        "total_update_seconds": sum(float(item["total_update_seconds"]) for item in update_results),
        "phase_totals": {
            key: sum(float(item.get("phase_timings", {}).get(key, 0.0)) for item in update_results)
            for key in PROFILE_PHASES
        },
        "peak_rss_bytes": max(int(item["rss_bytes"]) for item in memories),
        "all_finite": all(item["finite_loss"] and item["finite_gradients"] for item in update_results),
        "parameter_hash_after": parameter_hash(student),
        "_parameters": {name: parameter.detach().cpu().clone() for name, parameter in student.named_parameters()},
    }, {name: parameter.detach().cpu().clone() for name, parameter in student.named_parameters()}


def run_stage2(*, ledger: AppendOnlyJsonl, output_dir: Path) -> dict[str, Any]:
    source, valid_mask, input_valid_mask, document_ids = _synthetic_fixture(torch.device("cpu"))
    results: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in VARIANTS}
    snapshots: dict[str, dict[str, dict[str, torch.Tensor]]] = {name: {} for name, _, _ in VARIANTS}
    for variant_name, model_variant, rounds in VARIANTS:
        for label, physical_batch in CONFIGS:
            run_id = f"stage2-{uuid.uuid4().hex[:10]}"
            result, parameters = _run_training_updates(
                run_root=output_dir,
                run_id=run_id,
                stage=2,
                variant_name=variant_name,
                model_variant=model_variant,
                rounds=rounds,
                seed=SYNTHETIC_SEED + rounds,
                physical_batch=physical_batch,
                source=source,
                valid_mask=valid_mask,
                input_valid_mask=input_valid_mask,
                document_ids=document_ids,
                teacher=TinyTeacher(),
                updates=STAGE2_UPDATES,
                source_manifest=None,
            )
            result["config"] = label
            snapshots[variant_name][label] = parameters
            result["fixture"] = {"vocab": SYNTHETIC_VOCAB, "dimension": SYNTHETIC_DIMENSION, "slots": SYNTHETIC_SLOTS}
            result["timing"] = {
                "warmup_pair": result["pairs"][0],
                "measured_pairs": result["pairs"][1:],
                "mean_pair_seconds_per_update": sum(item["pair_seconds"] for item in result["pairs"][1:]) / 4.0,
                "measured_tokens_per_second": (4 * 2048) / sum(item["pair_seconds"] for item in result["pairs"][1:]),
            }
            result.pop("_parameters", None)
            results[variant_name].append(result)
            ledger.append({"stage": 2, "record_type": "synthetic_efficiency_result", **result})

    for variant_name in results:
        by_config = {item["config"]: item for item in results[variant_name]}
        reference = snapshots[variant_name]["A"]
        for label, item in by_config.items():
            max_delta = max(float((reference[name] - snapshots[variant_name][label][name]).abs().max().item()) for name in reference)
            item["equivalence"] = {
                "reference_config": "A",
                "max_abs_parameter_delta_vs_A": max_delta,
                "allclose_vs_A": max_delta <= 1e-4,
            }
        gates = {
            "configs": list(by_config) == ["A", "B", "C"],
            "six_updates_each": all(len(item["updates"]) == STAGE2_UPDATES for item in by_config.values()),
            "pair_schedule": all(item["pairs"] == _pair_records(item["updates"]) for item in by_config.values()),
            "valid_tokens_per_window": all(all(update["valid_tokens"] == 2048 for update in item["updates"]) for item in by_config.values()),
            "finite_losses_and_gradients": all(item["all_finite"] for item in by_config.values()),
            "physical_batch_equivalence": all(item["equivalence"]["allclose_vs_A"] for item in by_config.values()),
            "fp32_only": True,
        }
        for item in by_config.values():
            item["gates"] = gates
            ledger.append({"stage": 2, "record_type": "synthetic_efficiency_gate", "variant": variant_name, "config": item["config"], "gates": gates, "equivalence": item["equivalence"]})
        # Pick only after all three configs completed and passed operational gates.
        if all(gates.values()):
            chosen = min(by_config.values(), key=lambda item: item["timing"]["mean_pair_seconds_per_update"])
            by_config["A"]["selection_candidate"] = {"physical_batch": chosen["physical_batch"], "reason": "lowest measured synthetic mean pair seconds/update"}

    all_results = [item for items in results.values() for item in items]
    return {
        "status": "completed" if all(all(item.get("gates", {}).values()) for item in all_results) else "failed",
        "contract": {
            "configurations": [{"label": label, "physical_batch": batch, "microbatches": EFFECTIVE_BATCH // batch} for label, batch in CONFIGS],
            "effective_batch": EFFECTIVE_BATCH,
            "valid_targets_per_window": 2048,
            "updates_per_variant_and_config": STAGE2_UPDATES,
            "pair_schedule": {"warmup_updates": [0, 1], "measured_updates": [2, 3, 4, 5]},
            "total_updates": 36,
            "precision": "float32",
            "loss": "exact existing distillation_loss",
            "optimizer": "exact existing AdamW",
            "clipping": "exact existing clip_norm=1.0",
            "calendar": update_window_schedule(STAGE2_UPDATES),
        },
        "variants": results,
    }


def _load_local_real_runtime(cache_root: Path) -> tuple[Any, Any, torch.nn.Module, dict[str, Any]]:
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
        download_config=DownloadConfig(local_files_only=True),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    source, input_masks, target_masks, manifest = _preflight_source(dataset, tokenizer, torch.device("cpu"))
    actual_documents = [{key: value for key, value in item.items() if key != "document_id"} for item in manifest["selected_documents"]]
    if actual_documents != APPROVED_SELECTION_MANIFEST["selected_documents"]:
        raise RuntimeError("real source manifest differs from exact approved eight-document manifest")
    return source, input_masks[0], teacher, {"selection_manifest": manifest, "valid_tokens_per_window": [int(mask.sum()) for mask in target_masks]}


def _estimate_real_batch_requirement(physical_batch: int) -> int:
    teacher_logits = physical_batch * 512 * TOKENIZER_VOCAB * 4
    student_logits_and_backward = physical_batch * 256 * TOKENIZER_VOCAB * 4 * 3
    return int((teacher_logits + student_logits_and_backward) * 1.5)


def choose_stage3_batch(stage2: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for variant_items in stage2["variants"].values() for item in variant_items if item.get("gates") and all(item["gates"].values())]
    if not candidates:
        raise RuntimeError("stage 2 has no safe completed configuration")
    grouped: dict[int, list[float]] = {}
    for item in candidates:
        grouped.setdefault(int(item["physical_batch"]), []).append(float(item["timing"]["mean_pair_seconds_per_update"]))
    ranking = sorted((sum(times) / len(times), batch) for batch, times in grouped.items())
    best_seconds, best_batch = ranking[0]
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except ImportError:
        available = None
    required = _estimate_real_batch_requirement(best_batch)
    safe = available is None or available > required
    return {
        "physical_batch": best_batch,
        "microbatches": microbatch_count_for_physical_batch(best_batch),
        "selection": "lowest synthetic mean pair seconds/update across shared_K1/shared_K4",
        "synthetic_mean_pair_seconds_per_update": best_seconds,
        "memory_requirement_estimate_bytes": required,
        "available_system_bytes_before_real_teacher": available,
        "safe": safe,
    }


def run_stage3(*, ledger: AppendOnlyJsonl, output_dir: Path, cache_root: Path, stage2: dict[str, Any]) -> dict[str, Any]:
    selection = choose_stage3_batch(stage2)
    if not selection["safe"]:
        return {"status": "blocked", "classification": "BLOCKED_MEMORY_SAFETY_GATE", "batch_selection": selection, "error": "insufficient available system memory for selected real-teacher batch"}
    source, valid_mask, teacher, source_meta = _load_local_real_runtime(cache_root)
    input_valid_mask = valid_mask.clone()
    document_ids = list(source_meta["selection_manifest"]["selected_document_ids"])
    results = []
    for variant_name, model_variant, rounds in VARIANTS:
        run_id = f"stage3-{uuid.uuid4().hex[:10]}"
        result, _ = _run_training_updates(
            run_root=output_dir,
            run_id=run_id,
            stage=3,
            variant_name=variant_name,
            model_variant=model_variant,
            rounds=rounds,
            seed=TECHNICAL_VARIANT_SEEDS[variant_name],
            physical_batch=selection["physical_batch"],
            source=source,
            valid_mask=valid_mask,
            input_valid_mask=input_valid_mask,
            document_ids=document_ids,
            teacher=teacher,
            updates=STAGE3_UPDATES,
            source_manifest=source_meta["selection_manifest"],
            profile_phases=True,
        )
        result.pop("_parameters", None)
        result["loss_finite_evidence"] = [item["finite_loss"] for item in result["updates"]]
        result["gradient_finite_evidence"] = [item["finite_gradients"] for item in result["updates"]]
        results.append(result)
        ledger.append({"stage": 3, "record_type": "real_teacher_result", **result})
    compile_comparison = run_compile_comparison(
        ledger=ledger,
        output_dir=output_dir,
        source=source,
        valid_mask=valid_mask,
        input_valid_mask=input_valid_mask,
        document_ids=document_ids,
        teacher=teacher,
        physical_batch=selection["physical_batch"],
        source_manifest=source_meta["selection_manifest"],
    )
    return {
        "status": "completed",
        "classification": "CPU_REAL_TEACHER_SCOPING_ONLY",
        "batch_selection": selection,
        "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "local_files_only": True, "frozen": True, "eval": True},
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "local_files_only": True, "split": "train"},
        "selection_manifest": source_meta["selection_manifest"],
        "valid_tokens_per_window": source_meta["valid_tokens_per_window"],
        "updates_requested_per_variant": STAGE3_UPDATES,
        "pairs_requested_per_variant": 2,
        "variants": results,
        "compile_comparison": compile_comparison,
        "nll_interpretation": False,
    }


def _max_abs_diff(left: Any, right: Any) -> float:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return float((left - right).abs().max().item()) if left.numel() else 0.0
    if left is None or right is None:
        return 0.0 if left is right else math.inf
    return abs(float(left) - float(right))


def _max_capture_diff(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    left_value = left[key]
    right_value = right[key]
    if isinstance(left_value, list):
        if len(left_value) != len(right_value):
            return math.inf
        return max((_max_abs_diff(a, b) for a, b in zip(left_value, right_value, strict=True)), default=0.0)
    return _max_abs_diff(left_value, right_value)


def _compile_failure_record(*, variant_name: str, mode: str, physical_batch: int, error: Exception, compile_seconds: float, updates_completed: int = 0) -> dict[str, Any]:
    return {
        "variant": variant_name,
        "mode": mode,
        "physical_batch": physical_batch,
        "status": "failed",
        "failure_type": type(error).__name__,
        "failure": str(error),
        "compile_time_seconds": compile_seconds,
        "first_execution_seconds": None,
        "steady_state_seconds_per_update": None,
        "updates_completed": updates_completed,
        "updates_requested": COMPILE_UPDATES,
    }


def _compile_equivalence(max_abs_diffs: dict[str, float]) -> str:
    if all(value <= COMPILE_ABS_TOLERANCE for value in max_abs_diffs.values()):
        return "PASS"
    return "PERFORMANCE_INTERESTING/SCIENTIFIC_SCOPING_INELIGIBLE"


def _compile_mode_result(*, mode: str, variant_name: str, model_variant: str, rounds: int, physical_batch: int, source: torch.Tensor, valid_mask: torch.Tensor, input_valid_mask: torch.Tensor, document_ids: list[str], teacher: torch.nn.Module, source_manifest: dict[str, Any], output_dir: Path, initial_state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, dimension=128, slots=8, rounds=rounds, variant=model_variant).to(dtype=torch.float32)
    model.load_state_dict(initial_state_dict)
    model.train()
    compile_started = time.perf_counter()
    forward_window_fn = model.forward_window
    try:
        if mode == "default":
            forward_window_fn = torch.compile(forward_window_fn)
        elif mode == "max-autotune":
            forward_window_fn = torch.compile(forward_window_fn, mode="max-autotune")
        compile_seconds = time.perf_counter() - compile_started
    except Exception as error:
        return _compile_failure_record(variant_name=variant_name, mode=mode, physical_batch=physical_batch, error=error, compile_seconds=time.perf_counter() - compile_started), None

    optimizer = construct_adamw(model)
    run_id = f"compile-{uuid.uuid4().hex[:10]}-{variant_name}-{mode}"
    run_ledger = RunLedger(output_dir / "runs" / run_id, run_id)
    state: torch.Tensor | None = None
    update_seconds: list[float] = []
    first_capture: dict[str, Any] = {}
    try:
        for update, window in enumerate(update_window_schedule(COMPILE_UPDATES)):
            if window == 0:
                state = None
            started = time.perf_counter()
            state, metrics = microbatch_update(
                ledger=run_ledger,
                run_id=run_id,
                variant=variant_name,
                update=update,
                student=model,
                teacher=teacher,
                source=source,
                valid_mask=valid_mask,
                input_valid_mask=input_valid_mask,
                optimizer=optimizer,
                window=window,
                persistent_state=state,
                physical_batch=physical_batch,
                document_ids=document_ids,
                source_manifest=source_manifest,
                state_source_update=state_source_update_for_update(update),
                forward_window_fn=forward_window_fn,
                comparison_capture=first_capture if update == 0 else None,
            )
            update_seconds.append(float(metrics["elapsed_seconds"]))
            update_seconds[-1] = max(update_seconds[-1], time.perf_counter() - started)
    except Exception as error:
        failed = _compile_failure_record(variant_name=variant_name, mode=mode, physical_batch=physical_batch, error=error, compile_seconds=compile_seconds, updates_completed=len(update_seconds))
        failed["first_execution_seconds"] = update_seconds[0] if update_seconds else None
        failed["steady_state_seconds_per_update"] = sum(update_seconds[1:]) / len(update_seconds[1:]) if len(update_seconds) > 1 else None
        return failed, None
    return {
        "variant": variant_name,
        "mode": mode,
        "physical_batch": physical_batch,
        "status": "completed",
        "compile_time_seconds": compile_seconds,
        "first_execution_seconds": update_seconds[0],
        "steady_state_seconds_per_update": sum(update_seconds[1:]) / len(update_seconds[1:]),
        "updates_completed": len(update_seconds),
        "updates_requested": COMPILE_UPDATES,
    }, first_capture


def run_compile_comparison(*, ledger: AppendOnlyJsonl, output_dir: Path, source: torch.Tensor, valid_mask: torch.Tensor, input_valid_mask: torch.Tensor, document_ids: list[str], teacher: torch.nn.Module, physical_batch: int, source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare eager and CPU compile modes after real stage-3 profiling."""
    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for variant_name, model_variant, rounds in VARIANTS:
        set_technical_seed(TECHNICAL_VARIANT_SEEDS[variant_name])
        base_model = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, dimension=128, slots=8, rounds=rounds, variant=model_variant).to(dtype=torch.float32)
        initial_state_dict = {name: value.detach().clone() for name, value in base_model.state_dict().items()}
        captures: dict[str, dict[str, Any]] = {}
        for mode in COMPILE_MODES:
            result, capture = _compile_mode_result(
                mode=mode,
                variant_name=variant_name,
                model_variant=model_variant,
                rounds=rounds,
                physical_batch=physical_batch,
                source=source,
                valid_mask=valid_mask,
                input_valid_mask=input_valid_mask,
                document_ids=document_ids,
                teacher=teacher,
                source_manifest=source_manifest,
                output_dir=output_dir,
                initial_state_dict=initial_state_dict,
            )
            result["initial_parameter_hash"] = hashlib.sha256(b"".join(value.detach().cpu().contiguous().numpy().tobytes() for value in initial_state_dict.values())).hexdigest()
            results.append(result)
            ledger.append({"stage": 3, "record_type": "compile_comparison_mode", **result})
            if capture is not None:
                captures[mode] = capture
        eager_capture = captures.get("eager")
        for mode in ("default", "max-autotune"):
            compiled_capture = captures.get(mode)
            if eager_capture is None or compiled_capture is None:
                comparison = {
                    "variant": variant_name,
                    "mode": mode,
                    "status": "not_comparable",
                    "equivalence": "COMPILE_FAILED_CLOSED",
                    "reason": "eager and compiled one-update captures are required",
                }
            else:
                max_abs_diffs = {
                    "loss": _max_capture_diff(eager_capture, compiled_capture, "loss"),
                    "final_state": _max_capture_diff(eager_capture, compiled_capture, "final_state"),
                    "gradients_before_optimizer_step": _max_capture_diff(eager_capture, compiled_capture, "gradients_before_optimizer_step"),
                    "parameter_update": _max_capture_diff(eager_capture, compiled_capture, "parameter_update"),
                }
                comparison = {
                    "variant": variant_name,
                    "mode": mode,
                    "status": "completed",
                    "same_initial_weights_and_input": True,
                    "max_abs_diffs": max_abs_diffs,
                    "tolerance_absolute": COMPILE_ABS_TOLERANCE,
                    "equivalence": _compile_equivalence(max_abs_diffs),
                }
            comparisons.append(comparison)
            ledger.append({"stage": 3, "record_type": "compile_equivalence", **comparison})
    failures = [item for item in results if item["status"] != "completed"]
    return {
        "status": "completed" if not failures else "completed_with_failures",
        "classification": "CPU_COMPILE_COMPARISON_FAIL_CLOSED" if failures else "CPU_COMPILE_COMPARISON_MEASUREMENT_ONLY",
        "physical_batch": physical_batch,
        "effective_batch": EFFECTIVE_BATCH,
        "microbatches": microbatch_count_for_physical_batch(physical_batch),
        "source_shape": list(source.shape),
        "valid_tokens_per_window": [int(mask.sum()) for mask in valid_mask.reshape(EFFECTIVE_BATCH, TOKENS_PER_WINDOW)],
        "selected_document_ids": document_ids,
        "selection_manifest": source_manifest,
        "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "local_files_only": True, "frozen": True, "eval": True},
        "same_source_manifest": True,
        "same_initial_weights_and_input": True,
        "precision": "float32",
        "modes": list(COMPILE_MODES),
        "updates_per_variant_and_mode": COMPILE_UPDATES,
        "first_execution_definition": "update 0 includes lazy torch.compile graph compilation when applicable",
        "steady_state_definition": "arithmetic mean of updates 1 and 2",
        "tolerance_policy": {"absolute": COMPILE_ABS_TOLERANCE, "relax_after_results": False, "compared_fields": ["loss", "final_state", "gradients_before_optimizer_step", "parameter_update"]},
        "mode_results": results,
        "equivalence_results": comparisons,
    }


def run_stage4(stage3: dict[str, Any]) -> dict[str, Any]:
    if stage3.get("status") != "completed":
        return {"status": "blocked", "classification": "BLOCKED_STAGE3", "formula": None, "assumptions": ["Stage 4 requires measured real-teacher CPU stage-3 times."]}
    times = {item["variant"]: sum(float(update["elapsed_seconds"]) for update in item["updates"]) / len(item["updates"]) for item in stage3["variants"]}
    seconds_per_complete_run = sum(times.values()) * SCOPE_UPDATES
    total_updates = SCOPE_SEEDS * SCOPE_UPDATES * 2
    total_seconds = SCOPE_SEEDS * seconds_per_complete_run
    return {
        "status": "projected_only",
        "classification": "SCOPE_A_PROJECTION_NO_2000_UPDATE_RUN",
        "times": {"real_stage3_mean_seconds_per_update": times},
        "formula": "hours = seeds * updates_per_variant * (t_shared_K1 + t_shared_K4) / 3600",
        "assumptions": [
            "Stage-3 real-teacher CPU mean per-update times represent every projected update.",
            "Two seeds means two complete shared_K1 plus shared_K4 runs.",
            "No validation, checkpoint, I/O, or restart overhead is included.",
            "This is a CPU projection and makes no hardware transfer claim.",
        ],
        "seeds": SCOPE_SEEDS,
        "updates_per_variant": SCOPE_UPDATES,
        "variants_per_seed": ["shared_K1", "shared_K4"],
        "total_updates": total_updates,
        "seconds_per_complete_seed_run": seconds_per_complete_run,
        "total_seconds": total_seconds,
        "hours": total_seconds / 3600.0,
        "days": total_seconds / 86400.0,
    }


def run_all(*, output_dir: Path, cache_root: Path, warmup_tokens: int = 16, measure_tokens: int = 64, warmup_windows: int = 1, measure_windows: int = 2, command: list[str] | None = None) -> dict[str, Any]:
    if torch.device("cpu").type != "cpu":
        raise RuntimeError("CPU device construction failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = AppendOnlyJsonl(output_dir / "cpu_scoping_ledger.jsonl")
    report: dict[str, Any] = {
        "schema": "omega-core-lm-0-r1-cpu-scoping-readiness-v2",
        "campaign": CAMPAIGN_ID,
        "status": "running",
        "command": command or [],
        "execution_order": list(EXECUTION_ORDER),
        "cpu_identity": cpu_identity(),
        "backend": backend_options(),
        "safety_boundary": {
            "device_requested": "cpu",
            "gpu_used": False,
            "external_communication": False,
            "spend": False,
            "dtype": "float32",
            "bf16": False,
            "amp": False,
            "local_files_only": True,
            "cache_root": str(cache_root),
        },
        "limitations": ["CPU measurements are not GPU performance claims.", "Stage 4 is projection only; no 2000-update scientific run was executed."],
    }
    try:
        report["stage1_inference"] = run_stage1(ledger=ledger, warmup_tokens=warmup_tokens, measure_tokens=measure_tokens, warmup_windows=warmup_windows, measure_windows=measure_windows)
        report["stage2_efficiency"] = run_stage2(ledger=ledger, output_dir=output_dir)
        try:
            stage3_result = run_stage3(ledger=ledger, output_dir=output_dir, cache_root=cache_root, stage2=report["stage2_efficiency"])
            report["stage3_compile_comparison"] = stage3_result.pop("compile_comparison", {"status": "blocked", "classification": "BLOCKED_STAGE3_PROFILING"})
            report["stage3_real_teacher"] = stage3_result
        except (MemoryError, OSError, RuntimeError, ValueError) as error:
            report["stage3_real_teacher"] = {"status": "blocked", "classification": "BLOCKED_LOCAL_LOAD_OR_MEMORY", "error_type": type(error).__name__, "error": str(error), "no_fallback_batch": True}
            report["stage3_compile_comparison"] = {"status": "blocked", "classification": "BLOCKED_STAGE3_PROFILING", "error": "compile phase requires completed real-teacher stage-3 profiling"}
        report["stage4_scope_a_projection"] = run_stage4(report["stage3_real_teacher"])
        report["status"] = "completed" if report["stage3_real_teacher"]["status"] == "completed" else "blocked"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    _write_report(output_dir / "cpu_scoping_report.json", report)
    ledger.append({"stage": 4, "record_type": "final_report", "status": report["status"], "report": str(output_dir / "cpu_scoping_report.json")})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CAMPAIGN_ID)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--measure-tokens", type=int, default=64)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--measure-windows", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_all(output_dir=args.output_dir, cache_root=args.cache_root, warmup_tokens=args.warmup_tokens, measure_tokens=args.measure_tokens, warmup_windows=args.warmup_windows, measure_windows=args.measure_windows, command=[sys.executable, *sys.argv])
    print(json.dumps(_json_safe({"status": report["status"], "report": str(args.output_dir / "cpu_scoping_report.json"), "stage3": report.get("stage3_real_teacher", {}).get("status"), "stage4": report.get("stage4_scope_a_projection", {}).get("status")}), sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
