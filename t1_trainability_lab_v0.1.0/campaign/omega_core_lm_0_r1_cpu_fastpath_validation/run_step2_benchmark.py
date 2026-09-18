"""Authorized OMEGA R1 Step 2 CPU fastpath benchmark.

This runner is deliberately isolated from SCOPE-A except for shared CPU
runtime configuration. It measures four complete training-update routes on
the eight pinned train documents, one trainer at a time, and stops on the
first failure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn


UNIT_DIR = Path(__file__).resolve().parent
LAB_ROOT = UNIT_DIR.parents[1]
SCRIPTS_DIR = LAB_ROOT / "scripts"
GPU_PREP_DIR = LAB_ROOT / "campaign" / "omega_core_lm_0_gpu_environment_preparation"
SCOPE_DIR = LAB_ROOT / "campaign" / "omega_core_lm_0_r1_scientific_scoping_a"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(UNIT_DIR))
sys.path.insert(0, str(GPU_PREP_DIR))
sys.path.insert(0, str(SCOPE_DIR))

from run_scientific_scoping_a import configure_cpu_runtime  # noqa: E402

from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    OmegaCoreLM0R1Technical,
    distillation_loss,
    select_documents,
)
from omega_fast_candidate import (  # noqa: E402
    OmegaCoreLMFast,
    distillation_loss_lse,
    teacher_targets,
)
from omega_nominal_microbatch_runner import (  # noqa: E402
    APPROVED_MANIFEST_PATH,
    APPROVED_MANIFEST_SHA256,
    APPROVED_SELECTION_MANIFEST,
    validate_selected_documents,
)


# Configure exactly once, before any benchmark tensor or autograd work.
configure_cpu_runtime()
THREADS = 4
INTEROP_THREADS = 1

DEVICE = torch.device("cpu")
DTYPE = torch.float32
SEED_BY_VARIANT = {"shared_K1": 20260913, "shared_K4": 20260914}
VARIANTS = {
    "shared_K1": {"rounds": 1, "model_variant": "shared"},
    "shared_K4": {"rounds": 4, "model_variant": "shared"},
}
CONFIGS = ("R", "F", "L", "C")
UPDATES = 6
WARMUP_UPDATES = 2
MEASURED_UPDATES = 4
TOKENS_PER_WINDOW = 256
SOURCE_TOKENS = 513
VOCAB_SIZE = 50257
DIMENSION = 128
SLOTS = 8
TEMPERATURE = 2.0
BASE_LR = 3e-4
CLIP_NORM = 1.0
WINDOW_SCHEDULE = [0, 1, 0, 1, 0, 1]
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"


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


def memory_snapshot() -> dict[str, int | None]:
    process = psutil.Process()
    info = process.memory_info()
    peak_wset = getattr(info, "peak_wset", None)
    return {
        "rss_bytes": int(info.rss),
        "available_system_bytes": int(psutil.virtual_memory().available),
        "os_peak_working_set_bytes": int(peak_wset) if peak_wset is not None else None,
    }


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def construct_adamw(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def load_approved_source() -> tuple[Tensor, list[str], dict[str, Any], Any]:
    """Load exact train data and reject any selection drift before training."""
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    if len(tokenizer) != VOCAB_SIZE:
        raise RuntimeError(f"tokenizer vocab mismatch: expected {VOCAB_SIZE}, got {len(tokenizer)}")
    selected, manifest = select_documents(dataset, tokenizer)
    document_ids = validate_selected_documents(selected, manifest, APPROVED_SELECTION_MANIFEST)
    source = torch.tensor([item["tokens"] for item in selected], dtype=torch.long, device=DEVICE)
    if tuple(source.shape) != (8, SOURCE_TOKENS):
        raise RuntimeError(f"approved source shape mismatch: {tuple(source.shape)}")
    manifest["approved_manifest_provenance"] = APPROVED_SELECTION_MANIFEST["provenance"]
    manifest["selected_document_ids"] = document_ids
    manifest["source_shape"] = list(source.shape)
    manifest["valid_target_tokens_per_window"] = [2048, 2048]
    return source, document_ids, manifest, tokenizer


def load_teacher() -> nn.Module:
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    teacher.to(DEVICE)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def teacher_window_logits(teacher: nn.Module, source: Tensor, window: int) -> Tensor:
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    with torch.no_grad():
        output = teacher(input_ids=context)
        result = output.logits[:, start : start + 256].detach().clone().float()
    del output
    if result.dtype != DTYPE:
        raise RuntimeError(f"teacher route produced non-FP32 logits: {result.dtype}")
    return result


def make_initial_references() -> dict[str, nn.Module]:
    references: dict[str, nn.Module] = {}
    for name, config in VARIANTS.items():
        torch.manual_seed(SEED_BY_VARIANT[name])
        references[name] = OmegaCoreLM0R1Technical(
            vocab_size=VOCAB_SIZE,
            dimension=DIMENSION,
            slots=SLOTS,
            rounds=config["rounds"],
            variant=config["model_variant"],
        ).to(DEVICE).float()
    return references


def make_model(config_name: str, reference: nn.Module) -> nn.Module:
    if config_name == "R":
        return copy.deepcopy(reference).to(DEVICE).float()
    return OmegaCoreLMFast.from_reference(reference).to(DEVICE).float()


def update_inputs(source: Tensor, window: int) -> tuple[Tensor, Tensor, Tensor]:
    if window == 0:
        return source[:, :256], source[:, 1:257], torch.zeros(8, SLOTS, DIMENSION, dtype=DTYPE, device=DEVICE)
    return source[:, 256:512], source[:, 257:513], None  # type: ignore[return-value]


def config_description(config_name: str, selected_route: str | None = None) -> dict[str, Any]:
    if config_name == "R":
        return {"model": "OmegaCoreLM0R1Technical", "loss": "original distillation_loss", "teacher": "current route"}
    if config_name == "F":
        return {"model": "local corrected OmegaCoreLMFast", "loss": "original distillation_loss", "teacher": "current route"}
    if config_name == "L":
        return {"model": "local corrected OmegaCoreLMFast", "loss": "distillation_loss_lse", "teacher": "current route; target preparation inside timed update"}
    return {
        "model": "local corrected OmegaCoreLMFast",
        "loss": "original distillation_loss" if selected_route == "F" else "distillation_loss_lse",
        "teacher": "FP32 cache read inside timed update",
        "selected_route": selected_route,
    }


def cache_payload(route: str, teacher: nn.Module, source: Tensor) -> tuple[dict[int, dict[str, Tensor]], float, dict[str, Any]]:
    started = time.perf_counter()
    cache: dict[int, dict[str, Tensor]] = {}
    total_bytes = 0
    for window in (0, 1):
        logits = teacher_window_logits(teacher, source, window)
        if route == "F":
            cache[window] = {"teacher_logits": logits.float().contiguous()}
        else:
            probs, neg_entropy = teacher_targets(logits.float(), temperature=TEMPERATURE)
            cache[window] = {"teacher_probs": probs.float().contiguous(), "teacher_neg_entropy": neg_entropy.float().contiguous()}
        for tensor in cache[window].values():
            if tensor.dtype != DTYPE:
                raise RuntimeError(f"cache tensor is not FP32: {tensor.dtype}")
            total_bytes += tensor.numel() * tensor.element_size()
        del logits
    elapsed = time.perf_counter() - started
    fmt = "FP32 teacher logits" if route == "F" else "FP32 teacher probabilities + FP32 negative entropy"
    return cache, elapsed, {"route": route, "format": fmt, "dtype": "torch.float32", "bytes": total_bytes, "creation_seconds": elapsed}


def store_and_read_cache(cache: dict[int, dict[str, Tensor]], path: Path) -> dict[int, dict[str, Tensor]]:
    torch.save(cache, path)
    loaded = torch.load(path, map_location=DEVICE, weights_only=True)
    for tensors in loaded.values():
        for tensor in tensors.values():
            if tensor.dtype != DTYPE:
                raise RuntimeError(f"cache readback changed dtype: {tensor.dtype}")
    return loaded


def run_config(
    config_name: str,
    references: dict[str, nn.Module],
    teacher: nn.Module,
    source: Tensor,
    document_ids: list[str],
    output_dir: Path,
    selected_route: str | None = None,
    cache: dict[int, dict[str, Tensor]] | None = None,
) -> dict[str, Any]:
    config_started = time.perf_counter()
    config_memory_before = memory_snapshot()
    variant_results: dict[str, Any] = {}
    ledger_path = output_dir / f"{config_name.lower()}_ledger.jsonl"

    with ledger_path.open("w", encoding="utf-8", newline="\n") as ledger:
        for variant_name, variant_config in VARIANTS.items():
            reference = references[variant_name]
            model = make_model(config_name, reference)
            optimizer = construct_adamw(model)
            state: Tensor | None = None
            updates: list[dict[str, Any]] = []
            for update_index, window in enumerate(WINDOW_SCHEDULE):
                optimizer.zero_grad(set_to_none=True)
                if window == 0:
                    state = None
                input_ids, targets, reset_state = update_inputs(source, window)
                previous = reset_state if window == 0 else state
                if previous is None:
                    raise RuntimeError(f"window 1 missing state at update {update_index}")
                started = time.perf_counter()
                phase: dict[str, float] = {}

                phase_started = time.perf_counter()
                if config_name in {"R", "F"}:
                    teacher_logits = teacher_window_logits(teacher, source, window)
                    teacher_targets_for_lse = None
                elif config_name == "L":
                    teacher_logits = teacher_window_logits(teacher, source, window)
                    target_probs, target_neg_entropy = teacher_targets(teacher_logits.float(), temperature=TEMPERATURE)
                    teacher_targets_for_lse = (target_probs.float(), target_neg_entropy.float())
                else:
                    if cache is None:
                        raise RuntimeError("C config requires selected-route cache")
                    cached = cache[window]
                    teacher_logits = cached.get("teacher_logits")
                    teacher_targets_for_lse = (
                        cached["teacher_probs"],
                        cached["teacher_neg_entropy"],
                    ) if "teacher_probs" in cached else None
                    if teacher_logits is not None:
                        teacher_logits = teacher_logits.clone()
                    elif teacher_targets_for_lse is not None:
                        teacher_targets_for_lse = tuple(tensor.clone() for tensor in teacher_targets_for_lse)  # type: ignore[assignment]
                    else:
                        raise RuntimeError("cache entry has no usable teacher payload")
                phase["teacher_or_cache_read_seconds"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                if config_name == "R":
                    next_state, student_logits = model.forward_window(input_ids, previous)
                    trace = None
                elif config_name == "F":
                    next_state, student_logits, trace = model.forward_window(input_ids, previous)
                else:
                    next_state, _, candidate_states, readout_states = model.recur_states(input_ids, previous)
                    trace = {"candidate_states": candidate_states, "readout_states": readout_states}
                    student_logits = None
                phase["student_recurrence_forward_seconds"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                if config_name in {"R", "F"} or (config_name == "C" and selected_route == "F"):
                    if teacher_logits is None or student_logits is None:
                        raise RuntimeError("logit loss route missing logits")
                    losses = distillation_loss(student_logits, teacher_logits, targets, torch.ones_like(targets, dtype=torch.bool))
                else:
                    if teacher_targets_for_lse is None or trace is None:
                        raise RuntimeError("LSE loss route missing FP32 target payload")
                    losses = distillation_loss_lse(
                        model, trace["readout_states"], teacher_targets_for_lse[0], teacher_targets_for_lse[1], targets,
                        torch.ones_like(targets, dtype=torch.bool), temperature=TEMPERATURE,
                    )
                phase["loss_seconds"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                losses["total"].backward()
                phase["backward_seconds"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                pre_clip_norm = grad_norm(model)
                returned_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
                post_clip_norm = grad_norm(model)
                phase["clipping_seconds"] = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                optimizer.step()
                phase["adamw_seconds"] = time.perf_counter() - phase_started
                next_state = next_state.detach()
                state = next_state if window == 1 else next_state

                record: dict[str, Any] = {
                    "config": config_name,
                    "variant": variant_name,
                    "update": update_index,
                    "window": window,
                    "phase": "warmup" if update_index < WARMUP_UPDATES else "measured",
                    "document_ids": document_ids,
                    "input_range": [0, 256] if window == 0 else [256, 512],
                    "target_range": [1, 257] if window == 0 else [257, 513],
                    "teacher_context_range": [0, 256] if window == 0 else [0, 512],
                    "valid_tokens": 2048,
                    "teacher_or_cache_read_seconds": phase["teacher_or_cache_read_seconds"],
                    "student_recurrence_forward_seconds": phase["student_recurrence_forward_seconds"],
                    "loss_seconds": phase["loss_seconds"],
                    "backward_seconds": phase["backward_seconds"],
                    "clipping_seconds": phase["clipping_seconds"],
                    "adamw_seconds": phase["adamw_seconds"],
                    "loss": {key: float(losses[key].detach().item()) for key in ("ce", "kl", "total")},
                    "pre_clip_grad_norm": pre_clip_norm,
                    "clip_returned_norm": returned_clip_norm,
                    "post_clip_grad_norm": post_clip_norm,
                    "finite": all(bool(torch.isfinite(losses[key]).item()) for key in ("ce", "kl", "total"))
                    and all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
                    "parameter_hash_after": parameter_hash(model),
                    "memory_after": memory_snapshot(),
                }
                phase_started = time.perf_counter()
                ledger.write(json.dumps(json_safe(record), sort_keys=True, separators=(",", ":")) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                phase["ledger_seconds"] = time.perf_counter() - phase_started
                record["ledger_seconds"] = phase["ledger_seconds"]
                record["total_seconds"] = time.perf_counter() - started
                updates.append(record)
                if not record["finite"]:
                    raise RuntimeError(f"non-finite values in {config_name}/{variant_name}/update-{update_index}")
            measured = [item for item in updates if item["phase"] == "measured"]
            variant_results[variant_name] = {
                "updates": updates,
                "warmup_total_seconds": sum(item["total_seconds"] for item in updates if item["phase"] == "warmup"),
                "measured_total_seconds": sum(item["total_seconds"] for item in measured),
                "measured_by_window_seconds": {
                    "window_0": sum(item["total_seconds"] for item in measured if item["window"] == 0),
                    "window_1": sum(item["total_seconds"] for item in measured if item["window"] == 1),
                },
                "all_updates_finite": all(item["finite"] for item in updates),
                "parameter_hash_before": parameter_hash(references[variant_name]),
            }
            del model, optimizer, state
    config_memory_after = memory_snapshot()
    all_samples = [config_memory_before, config_memory_after]
    for result in variant_results.values():
        all_samples.extend(item["memory_after"] for item in result["updates"])
    sampled_peak_rss = max(item["rss_bytes"] for item in all_samples)
    sampled_min_available = min(item["available_system_bytes"] for item in all_samples)
    os_peaks = [item["os_peak_working_set_bytes"] for item in all_samples if item["os_peak_working_set_bytes"] is not None]
    return {
        "config": config_name,
        "definition": config_description(config_name, selected_route),
        "variants": variant_results,
        "updates_completed": sum(len(item["updates"]) for item in variant_results.values()),
        "measured": {
            "t_K1_seconds": variant_results["shared_K1"]["measured_total_seconds"],
            "t_K4_seconds": variant_results["shared_K4"]["measured_total_seconds"],
            "joint_K1_K4_seconds": sum(item["measured_total_seconds"] for item in variant_results.values()),
            "by_variant_and_window_seconds": {
                name: result["measured_by_window_seconds"] for name, result in variant_results.items()
            },
        },
        "memory": {
            "before": config_memory_before,
            "after": config_memory_after,
            "peak_process_memory": {
                "bytes": max(os_peaks) if os_peaks else sampled_peak_rss,
                "measurement": "OS peak working set" if os_peaks else "observed sampled RSS",
            },
            "observed_peak_rss_bytes": sampled_peak_rss,
            "minimum_available_system_bytes": sampled_min_available,
            "samples": all_samples,
        },
        "elapsed_seconds": time.perf_counter() - config_started,
        "ledger": ledger_path,
    }


def build_report_base(run_id: str, source_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "omega-core-lm-0-r1-cpu-fastpath-validation-step2-v1",
        "status": "blocked",
        "step": 2,
        "step_name": "four-route CPU FP32 eager benchmark",
        "run_id": run_id,
        "full_campaign_launched": False,
        "scope_a_relaunched": False,
        "test_split_loaded": False,
        "projection_calculated": False,
        "exact_update_contract": {"configs": list(CONFIGS), "variants": list(VARIANTS), "updates_per_config_variant": UPDATES, "total_updates": 48, "warmup_updates": WARMUP_UPDATES, "measured_updates": MEASURED_UPDATES, "window_schedule": WINDOW_SCHEDULE},
        "configuration": {"batch": 8, "window_tokens": 256, "dimension": DIMENSION, "slots": SLOTS, "vocab_size": VOCAB_SIZE, "device": "cpu", "dtype": "float32", "execution": "eager", "temperature": TEMPERATURE, "optimizer": {"type": "AdamW", "lr": BASE_LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "clip_norm": CLIP_NORM}},
        "config_definitions": {name: config_description(name) for name in CONFIGS if name != "C"},
        "selection_decision": None,
        "source_manifest": source_manifest,
        "variants": {},
        "cache": None,
        "errors": [],
        "limitations": ["All updates are discardable technical measurements.", "No validation or test split was loaded.", "No SCOPE-A projection or quality interpretation was calculated.", "Memory peak is exact OS working set only when exposed by psutil; otherwise it is explicitly sampled RSS."],
    }


def canonical_report_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_report(path: Path, report: dict[str, Any]) -> tuple[str, str]:
    # Normalize once so digest and persisted bytes share one immutable value graph.
    snapshot = json_safe(report)
    if not isinstance(snapshot, dict):
        raise TypeError("report snapshot must be a dictionary")
    snapshot["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    unsigned_bytes = canonical_report_bytes(snapshot)
    digest = sha256_bytes(unsigned_bytes)
    snapshot["artifact_self_hash"] = digest
    encoded = canonical_report_bytes(snapshot)
    path.write_bytes(encoded)

    persisted = path.read_bytes()
    try:
        persisted_report = json.loads(persisted.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("report self-hash verification failed: invalid persisted JSON") from exc
    if not isinstance(persisted_report, dict):
        raise RuntimeError("report self-hash verification failed: persisted report is not an object")
    stored_hash = persisted_report.get("artifact_self_hash")
    if not isinstance(stored_hash, str):
        raise RuntimeError("report self-hash verification failed: missing artifact_self_hash")
    persisted_report["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    reconstructed_bytes = canonical_report_bytes(persisted_report)
    if persisted != encoded:
        raise RuntimeError("report self-hash verification failed: persisted bytes changed")
    if reconstructed_bytes != unsigned_bytes:
        raise RuntimeError("report self-hash verification failed: placeholder bytes differ")
    if stored_hash != digest or sha256_bytes(reconstructed_bytes) != stored_hash:
        raise RuntimeError("report self-hash verification failed: digest mismatch")
    return digest, sha256_bytes(persisted)


def main() -> int:
    started = time.perf_counter()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    output_dir = UNIT_DIR / "results" / "step2_benchmark" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "step2_report.json"
    report: dict[str, Any] = {
        "run_id": run_id,
        "status": "blocked",
        "errors": [],
        "full_campaign_launched": False,
        "scope_a_relaunched": False,
        "test_split_loaded": False,
        "projection_calculated": False,
    }
    try:
        step1_report = json.loads((UNIT_DIR / "step1_report.json").read_text(encoding="utf-8"))
        if step1_report.get("status") != "passed" or step1_report.get("step") != 1:
            raise RuntimeError("Step 1 gate is not passed; Step 2 remains blocked")
        source, document_ids, source_manifest, tokenizer = load_approved_source()
        del tokenizer
        teacher = load_teacher()
        references = make_initial_references()
        report = build_report_base(run_id, source_manifest)
        report["environment"] = {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "psutil": psutil.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "fixed_thread_configuration": {"intraop": THREADS, "interop": INTEROP_THREADS, "changed_inside_loops": False},
            "memory_before_process_runs": memory_snapshot(),
        }
        report["provenance"] = {
            "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "split": "train", "corpus_replacement_claim": False},
            "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "route": {"window_0_context": [0, 256], "window_1_context": [0, 512]}},
            "approved_manifest": {"path": APPROVED_MANIFEST_PATH, "artifact_sha256": APPROVED_MANIFEST_SHA256},
            "step1_commit": "bd42b21",
            "reference_origin_commit": "6b3a9ad",
            "variant_seeds": SEED_BY_VARIANT,
        }
        for config_name in ("R", "F", "L"):
            result = run_config(config_name, references, teacher, source, document_ids, output_dir)
            report["variants"][config_name] = result
        f_joint = report["variants"]["F"]["measured"]["joint_K1_K4_seconds"]
        l_joint = report["variants"]["L"]["measured"]["joint_K1_K4_seconds"]
        selected_route = "F" if f_joint < l_joint else "L"
        report["selection_decision"] = {
            "rule": "select faster measured joint K1+K4 update cost; ties select L deterministically",
            "F_joint_K1_K4_seconds": f_joint,
            "L_joint_K1_K4_seconds": l_joint,
            "chosen_route": selected_route,
        }
        cache_started_memory = memory_snapshot()
        cache, cache_creation_seconds, cache_meta = cache_payload(selected_route, teacher, source)
        cache_path = output_dir / "teacher_cache.pt"
        readback = store_and_read_cache(cache, cache_path)
        report["cache"] = {**cache_meta, "path": cache_path, "storage_readback_verified": True, "memory_before_creation": cache_started_memory}
        report["config_definitions"]["C"] = config_description("C", selected_route)
        report["variants"]["C"] = run_config("C", references, teacher, source, document_ids, output_dir, selected_route, readback)
        if sum(result["updates_completed"] for result in report["variants"].values()) != 48:
            raise RuntimeError("exact update contract violated")
        report["status"] = "completed"
        report["exact_updates_completed"] = 48
        report["environment"]["memory_after_process_runs"] = memory_snapshot()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["source_hashes"] = {
            "own_runner_sha256": sha256_file(Path(__file__)),
            "own_test_sha256": sha256_file(UNIT_DIR / "test_step2_benchmark.py"),
            "own_readme_sha256": sha256_file(UNIT_DIR / "README_step2.md"),
            "reference_baseline_sha256": sha256_file(SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py"),
            "local_fast_candidate_sha256": sha256_file(UNIT_DIR / "omega_fast_candidate.py"),
            "step1_report_sha256": sha256_file(UNIT_DIR / "step1_report.json"),
            "fable_read_only_reference_sha256": sha256_file(LAB_ROOT / "campaign" / "omega_core_lm_0_r1_fable_optimization_proposal" / "omega_fast.py"),
        }
    except Exception as exc:
        report.setdefault("errors", []).append({"type": type(exc).__name__, "message": str(exc), "stage": "step2_benchmark"})
        report["status"] = "blocked"
        report["exact_updates_completed"] = sum(result.get("updates_completed", 0) for result in report.get("variants", {}).values())
        report["elapsed_seconds"] = time.perf_counter() - started
    digest, file_hash = write_report(report_path, report)
    print(json.dumps({"status": report["status"], "report": report_path.relative_to(UNIT_DIR).as_posix(), "artifact_self_hash": digest, "file_sha256": file_hash, "exact_updates_completed": report.get("exact_updates_completed", 0), "selection": report.get("selection_decision")}, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
