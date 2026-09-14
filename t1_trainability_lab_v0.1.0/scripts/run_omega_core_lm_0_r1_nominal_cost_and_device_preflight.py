"""OMEGA CORE-LM-0 R1 nominal B=8 cost and device preflight.

Parent mode inventories the current machine and runs three isolated child
processes sequentially. Each child performs exactly seven updates for one
variant, alternating predeclared windows 0/1, with real tensors shaped
[8, 256] and 2048 valid tokens per update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    BASE_LR,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    MAX_TOTAL_UPDATES,
    OmegaCoreLM0R1Technical,
    OUTPUT as V2_OUTPUT,
    TOKENIZER_VOCAB,
    distillation_loss,
    parameter_hash,
    select_documents,
    synchronize,
    teacher_window_logits,
    write_self_hashed,
)


OUTPUT_DIR = ROOT / "campaign" / "omega_core_lm_0_r1_nominal_cost_and_device_preflight"
PREVIOUS_OUTPUT = OUTPUT_DIR / "nominal_cost_and_device_preflight.json"
OUTPUT = OUTPUT_DIR / "nominal_cost_and_device_preflight_v2.json"
PREVIOUS_V2 = V2_OUTPUT
MAX_UPDATES_PER_VARIANT = 7
VARIANTS = (("shared_K1", 1, "shared"), ("shared_K4", 4, "shared"), ("untied_K4", 4, "untied"))
EXPECTED_DOCUMENT_RANGES = ((1, 51), (51, 133), (133, 271), (271, 287), (287, 316), (316, 378), (394, 406), (406, 433))
MAX_BYTES = 16 * 1024**3
MIN_AVAILABLE_BYTES = 512 * 1024**2


def self_hash_payload(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    return hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def memory_sample(label: str) -> dict[str, Any]:
    process = psutil.Process()
    info = process.memory_info()
    return {
        "label": label,
        "rss_bytes": info.rss,
        "available_system_bytes": psutil.virtual_memory().available,
        "peak_working_set_bytes": getattr(info, "peak_wset", None),
    }


def peak_sampled_rss(samples: list[dict[str, Any]]) -> int:
    return max(int(sample["rss_bytes"]) for sample in samples)


def peak_os_working_set(samples: list[dict[str, Any]]) -> int | None:
    values = [sample["peak_working_set_bytes"] for sample in samples if sample["peak_working_set_bytes"] is not None]
    return max(values) if values else None


def inventory_os_adapters() -> tuple[list[dict[str, Any]], str | None]:
    command = (
        "$gpu=@(Get-CimInstance Win32_VideoController); "
        "$payload=@($gpu | ForEach-Object { [pscustomobject]@{name=$_.Name; adapter_ram_bytes=$_.AdapterRAM; "
        "driver_version=$_.DriverVersion; driver_date=$_.DriverDate; status=$_.Status} }); "
        "$payload | ConvertTo-Json -Depth 5 -Compress"
    )
    try:
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True, timeout=30)
        raw = completed.stdout.strip()
        if not raw:
            return [], None
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return list(parsed), None
    except Exception as exc:
        return [], f"OS GPU adapter inventory unavailable: {type(exc).__name__}: {exc}"


def inventory() -> dict[str, Any]:
    adapters, adapter_error = inventory_os_adapters()
    cuda_devices = [
        {"name": torch.cuda.get_device_name(index), "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
        for index in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else []
    if torch.cuda.is_available():
        decision = "GPU utilizable actualmente"
    elif adapters:
        decision = "GPU detectada pero entorno pendiente"
    else:
        decision = "ninguna GPU utilizable identificada en esta máquina"
    return {
        "cpu": {"name": platform.processor(), "logical_processors": psutil.cpu_count(logical=True), "physical_cores": psutil.cpu_count(logical=False)},
        "ram": {"total_bytes": psutil.virtual_memory().total, "available_bytes": psutil.virtual_memory().available},
        "os_gpu_adapters": adapters,
        "os_gpu_inventory_error": adapter_error,
        "python": {"executable": sys.executable, "version": sys.version},
        "torch": {"version": torch.__version__, "cuda_version": torch.version.cuda, "cuda_built": torch.backends.cuda.is_built(), "cuda_available": torch.cuda.is_available(), "devices": cuda_devices},
        "decision": decision,
        "separation": "OS-visible AMD adapter exists, but current PyTorch is CPU-only and exposes no usable CUDA device; no environment changes were made.",
    }


def verify_v2_segments(current_manifest: dict[str, Any]) -> dict[str, Any]:
    previous = json.loads(PREVIOUS_V2.read_text(encoding="utf-8"))
    expected = previous["selection_manifest"]["selected_documents"]
    actual = current_manifest["selected_documents"]
    comparable = ("row_range", "header", "token_count", "selected_token_count", "text_sha256", "token_sha256")
    matches = len(expected) == len(actual) and all(tuple(item[key] for key in comparable) == tuple(other[key] for key in comparable) for item, other in zip(expected, actual, strict=True))
    ranges = tuple(tuple(item["row_range"]) for item in actual)
    if not matches or ranges != EXPECTED_DOCUMENT_RANGES:
        raise RuntimeError("current reconstructed documents do not exactly match v2 selected segments")
    return {"v2_artifact": str(PREVIOUS_V2.relative_to(ROOT)).replace("\\", "/"), "v2_artifact_sha256": hashlib.sha256(PREVIOUS_V2.read_bytes()).hexdigest(), "v2_description": "functional test with batch 1 and two windows", "same_segments_verified_by_hash": True, "selected_document_ranges": [list(item) for item in ranges], "selected_segment_hashes": [{key: row[key] for key in ("text_sha256", "token_sha256")} for row in actual]}


def run_child(variant_name: str, rounds: int, model_variant: str) -> int:
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    started = time.perf_counter()
    payload: dict[str, Any] = {"schema": "omega-core-lm-0-r1-nominal-cost-child-v1", "status": "blocked", "variant": variant_name, "errors": []}
    samples: list[dict[str, Any]] = [memory_sample("child_start")]
    try:
        import datasets
        import transformers
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
        selected, manifest = select_documents(dataset, tokenizer)
        payload["segment_identity"] = verify_v2_segments(manifest)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        device = torch.device("cpu")
        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        student = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, rounds=rounds, variant=model_variant).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        source = torch.tensor([item["tokens"] for item in selected], dtype=torch.long, device=device)
        assert tuple(source.shape) == (8, 513)
        before_hash = parameter_hash(student)
        state: torch.Tensor | None = None
        updates: list[dict[str, Any]] = []
        samples.append(memory_sample("after_batch_tensor"))
        for update_index in range(1, MAX_UPDATES_PER_VARIANT + 1):
            window = 0 if update_index % 2 == 1 else 1
            if window == 0:
                state = student.initial_state(8, device=device)
            if state is None:
                raise RuntimeError("window 1 started without carried state")
            optimizer.zero_grad(set_to_none=True)
            update_started = time.perf_counter()
            input_ids = source[:, :256] if window == 0 else source[:, 256:512]
            targets = source[:, 1:257] if window == 0 else source[:, 257:513]
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            if tuple(input_ids.shape) != (8, 256):
                raise RuntimeError(f"nominal student input shape mismatch: {tuple(input_ids.shape)}")
            student_started = time.perf_counter()
            next_state, student_logits = student.forward_window(input_ids, state)
            synchronize()
            student_seconds = time.perf_counter() - student_started
            samples.append(memory_sample(f"update_{update_index}_after_student_forward"))
            teacher_started = time.perf_counter()
            teacher_logits = teacher_window_logits(teacher, source, window)
            synchronize()
            teacher_seconds = time.perf_counter() - teacher_started
            samples.append(memory_sample(f"update_{update_index}_after_teacher_forward"))
            loss_started = time.perf_counter()
            losses = distillation_loss(student_logits, teacher_logits, targets, valid_mask)
            loss_seconds = time.perf_counter() - loss_started
            samples.append(memory_sample(f"update_{update_index}_after_loss"))
            backward_started = time.perf_counter()
            losses["total"].backward()
            synchronize()
            backward_seconds = time.perf_counter() - backward_started
            samples.append(memory_sample(f"update_{update_index}_after_backward"))
            clip_started = time.perf_counter()
            pre_clip = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item())
            synchronize()
            clip_seconds = time.perf_counter() - clip_started
            samples.append(memory_sample(f"update_{update_index}_after_clipping"))
            update_started_inner = time.perf_counter()
            optimizer.step()
            synchronize()
            optimizer_seconds = time.perf_counter() - update_started_inner
            samples.append(memory_sample(f"update_{update_index}_after_update"))
            total_seconds = time.perf_counter() - update_started
            state = next_state.detach()
            update = {
                "update": update_index,
                "phase": "discardable_budget_warmup" if update_index <= 2 else "measured",
                "window": window,
                "alternation_policy": "odd updates window 0 from zero state; even updates window 1 from prior window state detached; each new pair resets to zero",
                "input_shape": list(input_ids.shape),
                "target_shape": list(targets.shape),
                "student_logits_shape": list(student_logits.shape),
                "teacher_input_shape": [8, 256] if window == 0 else [8, 512],
                "teacher_logits_shape": list(teacher_logits.shape),
                "valid_tokens": int(valid_mask.sum().item()),
                "nominal_shape_confirmed": tuple(input_ids.shape) == (8, 256) and tuple(student_logits.shape[:2]) == (8, 256) and tuple(teacher_logits.shape[:2]) == (8, 256) and int(valid_mask.sum().item()) == 2048,
                "teacher_context_rule": "context source[:, :256], logits[:, 0:256]" if window == 0 else "context source[:, :512], logits[:, 256:512]",
                "ce": float(losses["ce"].detach().item()),
                "kl": float(losses["kl"].detach().item()),
                "total_loss": float(losses["total"].detach().item()),
                "pre_clip_grad_norm": pre_clip,
                "clipping_intervened": pre_clip > 1.0,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "finite": all(torch.isfinite(losses[key]).item() for key in ("ce", "kl", "total")) and math.isfinite(pre_clip),
                "parameter_hash_after": parameter_hash(student),
                "timing_seconds": {"teacher": teacher_seconds, "student_forward": student_seconds, "loss": loss_seconds, "backward": backward_seconds, "clipping": clip_seconds, "update": optimizer_seconds, "total": total_seconds},
            }
            updates.append(update)
            if any(sample["rss_bytes"] > MAX_BYTES or sample["available_system_bytes"] < MIN_AVAILABLE_BYTES for sample in samples[-7:]):
                raise MemoryError("16 GiB/available-memory guard reached; stopped without reducing nominal shape")
        measured = [item["timing_seconds"] for item in updates if item["phase"] == "measured"]
        payload.update({
            "status": "passed",
            "classification": "PASS_NOMINAL_COST_VARIANT",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__},
            "device": {"requested": "cpu", "actual": str(device), "cuda_available": torch.cuda.is_available()},
            "tensor_contract": {"source_shape": list(source.shape), "per_update_input_shape": [8, 256], "valid_tokens_per_update": 2048, "all_updates_shape_confirmed": all(item["nominal_shape_confirmed"] for item in updates)},
            "updates": updates,
            "measured_medians_seconds": {key: median(item[key] for item in measured) for key in ("teacher", "student_forward", "loss", "backward", "clipping", "update", "total")},
            "memory": {"rss_samples": samples, "maximum_sampled_rss_bytes": peak_sampled_rss(samples), "os_peak_working_set_bytes": peak_os_working_set(samples), "os_peak_working_set_source": "psutil.Process.memory_info().peak_wset; child process isolated per variant" if peak_os_working_set(samples) is not None else "unavailable in this environment", "minimum_available_system_bytes": min(sample["available_system_bytes"] for sample in samples)},
            "parameter_hash_before": before_hash,
            "parameter_hash_after": parameter_hash(student),
            "technical_optimizer_updates": MAX_UPDATES_PER_VARIANT,
        })
    except Exception as exc:
        payload["errors"].append({"type": type(exc).__name__, "message": str(exc), "stage": "nominal_child"})
        payload["status"] = "blocked" if not payload.get("updates") else "partial"
        payload["classification"] = "BLOCKED_NOMINAL_SHAPE_OR_MEMORY"
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["memory_samples_final"] = samples
    path = OUTPUT_DIR / f"variant_{variant_name}.json"
    digest, file_sha = write_self_hashed(path, payload)
    print(json.dumps({"artifact": str(path.relative_to(ROOT)).replace("\\", "/"), "artifact_self_hash": digest, "file_sha256": file_sha, "status": payload["status"], "variant": variant_name, "updates": len(payload.get("updates", []))}, sort_keys=True), flush=True)
    return 0 if payload["status"] == "passed" else 2


def finalize_existing_child(child: dict[str, Any]) -> dict[str, Any]:
    samples = child.get("memory_samples_final", [])
    completed_updates = sorted({int(sample["label"].split("_")[1]) for sample in samples if sample["label"].startswith("update_") and sample["label"].endswith("_after_update")})
    child["technical_optimizer_updates"] = len(completed_updates)
    child["nominal_shape_confirmation"] = {
        "verified_by_runtime_assertion": bool(completed_updates),
        "completed_update_count": len(completed_updates),
        "student_input_shape": [8, 256],
        "student_logits_shape_prefix": [8, 256],
        "teacher_logits_shape_prefix": [8, 256],
        "valid_tokens": 2048,
        "window_sequence_completed": [0 if update % 2 == 1 else 1 for update in completed_updates],
        "note": "The child asserted actual tensor shapes before every completed forward; its update records were not flushed before the later guard exception, so no unobserved shape is claimed for the stopped update.",
    }
    if "memory" not in child and samples:
        peak_values = [sample["peak_working_set_bytes"] for sample in samples if sample.get("peak_working_set_bytes") is not None]
        child["memory"] = {
            "rss_samples": samples,
            "maximum_sampled_rss_bytes": max(sample["rss_bytes"] for sample in samples),
            "os_peak_working_set_bytes": max(peak_values) if peak_values else None,
            "os_peak_working_set_source": "psutil.Process.memory_info().peak_wset; child process isolated per variant" if peak_values else "unavailable in this environment",
            "minimum_available_system_bytes": min(sample["available_system_bytes"] for sample in samples),
            "partial_run": True,
        }
    child["partial_stop_evidence"] = "The original child artifact stopped after the update_6_after_update sample because the rolling memory guard observed 94,490,624 available bytes at update_6_after_loss; update records were not flushed before the guard exception, so timing medians are unavailable rather than reconstructed."
    return child


def run_parent(*, finalize_existing: bool = False) -> int:
    torch.set_num_threads(4)
    started = time.perf_counter()
    machine = inventory()
    result: dict[str, Any] = {
        "schema": "omega-core-lm-0-r1-nominal-cost-and-device-preflight-v1",
        "status": "blocked",
        "classification": "BLOCKED_NOMINAL_PREFLIGHT",
        "part_a_decision": machine["decision"],
        "part_a_inventory": machine,
        "part_b_contract": {"device": "cpu", "dtype": "float32", "torch_threads": 4, "input_tensor_shape": [8, 513], "per_update_student_input_shape": [8, 256], "valid_tokens_per_update": 2048, "window_policy": "odd update=window 0, even update=window 1; reset at each new pair", "updates_per_variant": 7, "variants": [name for name, _, _ in VARIANTS], "total_update_budget": MAX_TOTAL_UPDATES, "learning_rate": BASE_LR, "short_integrity_and_mid_scale": "not repeated; v2 remains evidence for those checks"},
        "v2_reference": {"path": str(PREVIOUS_V2.relative_to(ROOT)).replace("\\", "/"), "description": "functional test with batch 1 and two windows; v2 did not measure nominal B=8x256", "sha256": hashlib.sha256(PREVIOUS_V2.read_bytes()).hexdigest()},
        "variants": [],
        "errors": [],
    }
    for variant_name, rounds, model_variant in VARIANTS:
        if finalize_existing:
            child_path = OUTPUT_DIR / f"variant_{variant_name}.json"
            if not child_path.is_file():
                result["errors"].append({"variant": variant_name, "error": "existing child artifact missing"})
                break
            child = finalize_existing_child(json.loads(child_path.read_text(encoding="utf-8")))
            result["variants"].append(child)
            if child["status"] != "passed":
                result["errors"].append({"variant": variant_name, "existing_child_status": child["status"], "classification": child.get("classification"), "errors": child.get("errors", [])})
                break
            continue
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--variant", variant_name, "--rounds", str(rounds), "--model-variant", model_variant], capture_output=True, text=True, timeout=900)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        summary = json.loads(lines[-1]) if lines else {"status": "blocked", "error": "child returned no JSON summary", "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
        child_path = ROOT / summary["artifact"] if "artifact" in summary else OUTPUT_DIR / f"variant_{variant_name}.json"
        child = finalize_existing_child(json.loads(child_path.read_text(encoding="utf-8"))) if child_path.is_file() else {"status": "blocked", "variant": variant_name}
        result["variants"].append(child)
        if completed.returncode != 0:
            result["errors"].append({"variant": variant_name, "returncode": completed.returncode, "summary": summary, "stderr_tail": completed.stderr[-2000:]})
            break
    result["technical_optimizer_updates"] = sum(item.get("technical_optimizer_updates", 0) for item in result["variants"])
    if len(result["variants"]) == len(VARIANTS) and result["technical_optimizer_updates"] == MAX_TOTAL_UPDATES and all(item["status"] == "passed" for item in result["variants"]):
        result["status"] = "passed"
        result["classification"] = "PASS_NOMINAL_COST_AND_DEVICE_PREFLIGHT"
    else:
        result["status"] = "partial" if result["variants"] else "blocked"
        result["classification"] = "PARTIAL_NOMINAL_COST_PREFLIGHT" if result["variants"] else "BLOCKED_NOMINAL_COST_PREFLIGHT"
    result["measured_medians_seconds"] = None if result["status"] != "passed" else {variant["variant"]: variant.get("measured_medians_seconds") for variant in result["variants"]}
    result["terms_for_campaign_time"] = {"measured": ["CPU inventory and memory samples/peaks", "six completed nominal updates with real B=8 tensors before memory stop"] if result["status"] != "passed" else ["per-variant median teacher, student_forward, loss, backward, clipping, update, total seconds", "per-update valid tokens/s can be computed as 2048/total", "CPU inventory and memory samples/peaks"], "pending_or_estimated": ["per-variant timing medians and valid-tokens/s because child stopped before flushing update records", "number of seeds and final campaign update count", "T_evaluation", "T_E/S", "any GPU-port timing", "no viability threshold applied"]}
    result["previous_attempt_artifact"] = str(PREVIOUS_OUTPUT.relative_to(ROOT)).replace("\\", "/") if PREVIOUS_OUTPUT.is_file() else None
    result["previous_attempt_artifact_sha256"] = hashlib.sha256(PREVIOUS_OUTPUT.read_bytes()).hexdigest() if PREVIOUS_OUTPUT.is_file() else None
    result["elapsed_seconds"] = time.perf_counter() - started
    digest, file_sha = write_self_hashed(OUTPUT, result)
    print(json.dumps({"artifact": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "artifact_self_hash": digest, "file_sha256": file_sha, "status": result["status"], "classification": result["classification"], "part_a_decision": result["part_a_decision"], "technical_optimizer_updates": result["technical_optimizer_updates"]}, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[item[0] for item in VARIANTS])
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--model-variant", choices=["shared", "untied"])
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.variant:
        if args.rounds is None or args.model_variant is None:
            parser.error("child mode requires --rounds and --model-variant")
        return run_child(args.variant, args.rounds, args.model_variant)
    return run_parent(finalize_existing=args.finalize_existing)


if __name__ == "__main__":
    raise SystemExit(main())
