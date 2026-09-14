"""OMEGA CORE-LM-0 R1 runtime recovery and GPU readiness fixtures.

This unit performs no real-language updates. It validates recoverable event
logging, tensor lifetime handling, explicit microbatch equivalence, and a
read-only GPU support matrix using small synthetic tensors only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
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
    OmegaCoreLM0R1Technical,
    distillation_loss,
    parameter_hash,
    write_self_hashed,
)


OUTPUT_DIR = ROOT / "campaign" / "omega_core_lm_0_r1_runtime_recovery_and_gpu_readiness"
OUTPUT = OUTPUT_DIR / "runtime_recovery_and_gpu_readiness.json"
PREVIOUS_NOMINAL = ROOT / "campaign" / "omega_core_lm_0_r1_nominal_cost_and_device_preflight" / "nominal_cost_and_device_preflight_v2.json"
EVENTS = OUTPUT_DIR / "events.jsonl"
MICROBATCH_SIZE = 2
EFFECTIVE_BATCH = 8
MICROBATCH_COUNT = 4
MICROBATCH_TOLERANCE = 1e-5
PHASES = ("update_started", "student_forward_completed", "teacher_forward_completed", "loss_completed", "backward_completed", "optimizer_step_started", "optimizer_step_completed", "update_completed")


class TinyTeacher(torch.nn.Module):
    def __init__(self, vocab_size: int = 11) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.arange(512 * vocab_size, dtype=torch.float32).reshape(512, vocab_size) / 100.0, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return type("Output", (), {"logits": self.table[input_ids % self.table.shape[0]]})()


class AppendOnlyLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        encoded = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_memory() -> dict[str, int | None]:
    process = psutil.Process()
    info = process.memory_info()
    return {"rss_bytes": info.rss, "available_system_bytes": psutil.virtual_memory().available, "peak_working_set_bytes": getattr(info, "peak_wset", None)}


def append_phase(ledger: AppendOnlyLedger, *, variant: str, update: int, window: int, phase: str, started: float, input_shape: list[int], student_shape: list[int] | None, teacher_shape: list[int] | None, valid_tokens: int, metrics: dict[str, float] | None = None, status: str = "completed") -> None:
    event = {"variant": variant, "update": update, "window": window, "phase": phase, "status": status, "input_shape": input_shape, "student_logits_shape": student_shape, "teacher_logits_shape": teacher_shape, "valid_tokens": valid_tokens, "elapsed_seconds": time.perf_counter() - started, "memory": event_memory()}
    if metrics:
        event["metrics"] = metrics
    ledger.append(event)


def consolidate(path: Path, *, interrupted: bool) -> dict[str, Any]:
    events = read_events(path)
    update_ids = sorted({event["update"] for event in events})
    applied = sorted({event["update"] for event in events if event["phase"] == "optimizer_step_completed" and event["status"] == "APPLIED"})
    completed = sorted({event["update"] for event in events if event["phase"] == "update_completed" and event["status"] == "completed"})
    uncertain = sorted(set(update_ids) - set(applied))
    complete_run = bool(update_ids) and not interrupted and len(completed) == len(update_ids) and not uncertain
    totals = [event["metrics"]["total_loss"] for event in events if event["phase"] == "loss_completed" and "metrics" in event]
    return {"events_read": len(events), "updates_seen": update_ids, "optimizer_updates_applied": applied, "updates_completed": completed, "updates_uncertain_or_incomplete": uncertain, "medians": {"total_loss": median(totals)} if complete_run and totals else None, "consolidation_complete": complete_run}


def run_recovery_fixture(path: Path, fault_phase: str | None) -> dict[str, Any]:
    if path.exists():
        path.unlink()
    torch.manual_seed(1001)
    student = OmegaCoreLM0R1Technical(vocab_size=11, dimension=8, slots=2, rounds=1).float()
    teacher = TinyTeacher(11).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    source = torch.arange(2 * 5, dtype=torch.long).reshape(2, 5) % 11
    ledger = AppendOnlyLedger(path)
    started = time.perf_counter()
    state = student.initial_state(2, device=torch.device("cpu"))
    input_ids = source[:, :4]
    targets = source[:, 1:5]
    valid = torch.ones_like(targets, dtype=torch.bool)
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="update_started", started=started, input_shape=list(input_ids.shape), student_shape=None, teacher_shape=None, valid_tokens=int(valid.sum()))
    state, student_logits = student.forward_window(input_ids, state)
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="student_forward_completed", started=started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=None, valid_tokens=int(valid.sum()))
    teacher_logits = teacher(input_ids).logits
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="teacher_forward_completed", started=started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=int(valid.sum()))
    losses = distillation_loss(student_logits, teacher_logits, targets, valid)
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="loss_completed", started=started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=int(valid.sum()), metrics={"ce": float(losses["ce"].item()), "kl": float(losses["kl"].item()), "total_loss": float(losses["total"].item())})
    if fault_phase == "after_loss":
        raise RuntimeError("controlled fault after loss")
    losses["total"].backward()
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="backward_completed", started=started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=int(valid.sum()))
    if fault_phase == "after_backward":
        raise RuntimeError("controlled fault after backward")
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="optimizer_step_started", started=started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=int(valid.sum()), status="started")
    optimizer.step()
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="optimizer_step_completed", started=started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=int(valid.sum()), status="APPLIED")
    student.zero_grad(set_to_none=True)
    del student_logits, teacher_logits, losses
    append_phase(ledger, variant="tiny_fixture", update=1, window=0, phase="update_completed", started=started, input_shape=list(input_ids.shape), student_shape=list(input_ids.shape) + [11], teacher_shape=list(input_ids.shape) + [11], valid_tokens=int(valid.sum()))
    if fault_phase == "after_update":
        raise RuntimeError("controlled fault after update before consolidation")
    return consolidate(path, interrupted=False)


def run_fault_injection_tests() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name, phase, expected in (("after_loss", "after_loss", 0), ("after_backward", "after_backward", 0), ("after_update", "after_update", 1)):
        path = OUTPUT_DIR / f"fault_{name}.events.jsonl"
        try:
            run_recovery_fixture(path, phase)
            raised = False
        except RuntimeError as exc:
            raised = True
            message = str(exc)
        summary = consolidate(path, interrupted=True)
        events = read_events(path)
        cases[name] = {"raised_controlled_fault": raised, "message": message if raised else None, "events_survived": len(events), "phases_survived": [event["phase"] for event in events], "applied_updates": summary["optimizer_updates_applied"], "completed_updates": summary["updates_completed"], "expected_applied_count": expected, "medians": summary["medians"], "correct": raised and len(summary["optimizer_updates_applied"]) == expected and summary["medians"] is None}
    return {"cases": cases, "all_passed": all(case["correct"] for case in cases.values())}


def run_normal_recovery_ledger() -> dict[str, Any]:
    return run_recovery_fixture(EVENTS, None)


def run_storage_audit() -> dict[str, Any]:
    torch.manual_seed(2002)
    batch, context, vocab = 2, 512, 7
    full = torch.randn(batch, context, vocab)
    old_slice = full[:, 256:].detach()
    old_storage_bytes = old_slice.untyped_storage().nbytes()
    del full
    retained_after_full_delete = old_slice.untyped_storage().nbytes()
    compact = old_slice.clone()
    compact_storage_bytes = compact.untyped_storage().nbytes()
    student_base = OmegaCoreLM0R1Technical(vocab_size=vocab, dimension=8, slots=2, rounds=1).float()
    student_old = copy.deepcopy(student_base)
    student_new = copy.deepcopy(student_base)
    inputs = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 1]], dtype=torch.long)
    valid = torch.ones_like(targets, dtype=torch.bool)
    _, old_logits = student_old.forward_window(inputs, student_old.initial_state(batch, device=torch.device("cpu")))
    _, new_logits = student_new.forward_window(inputs, student_new.initial_state(batch, device=torch.device("cpu")))
    old_loss = distillation_loss(old_logits, old_slice[:, :4], targets, valid)["total"]
    new_loss = distillation_loss(new_logits, compact[:, :4], targets, valid)["total"]
    old_loss.backward()
    new_loss.backward()
    gradient_error = max(float((a.grad - b.grad).abs().max().item()) for a, b in zip(student_old.parameters(), student_new.parameters(), strict=True) if a.grad is not None and b.grad is not None)
    logits_equal = torch.equal(old_slice, compact)
    return {"full_logits_shape": [batch, context, vocab], "slice_shape": list(old_slice.shape), "full_storage_nbytes": old_storage_bytes, "detached_slice_storage_nbytes": retained_after_full_delete, "compact_copy_storage_nbytes": compact_storage_bytes, "slice_retains_full_storage": retained_after_full_delete == old_storage_bytes, "compact_copy_reduces_storage": compact_storage_bytes < retained_after_full_delete, "logits_exactly_equal": logits_equal, "loss_exactly_equal": bool(torch.equal(old_loss, new_loss)), "loss_abs_error": float((old_loss - new_loss).abs().item()), "student_gradient_max_abs_error": gradient_error, "student_gradient_within_tolerance": gradient_error <= MICROBATCH_TOLERANCE, "teacher_output_reference_deleted_before_compact_lifetime": True}


def run_microbatch_equivalence() -> dict[str, Any]:
    torch.manual_seed(3003)
    vocab = 7
    tokens = torch.arange(EFFECTIVE_BATCH * 5, dtype=torch.long).reshape(EFFECTIVE_BATCH, 5) % vocab
    targets = tokens[:, 1:]
    teacher_logits = torch.randn(EFFECTIVE_BATCH, 4, vocab)
    valid = torch.ones_like(targets, dtype=torch.bool)
    joint = OmegaCoreLM0R1Technical(vocab_size=vocab, dimension=8, slots=2, rounds=1).float()
    micro = copy.deepcopy(joint)
    _, joint_logits = joint.forward_window(tokens[:, :4], joint.initial_state(EFFECTIVE_BATCH, device=torch.device("cpu")))
    joint_loss = distillation_loss(joint_logits, teacher_logits, targets, valid)["total"]
    joint_loss.backward()
    joint_grads = [parameter.grad.detach().clone() for parameter in joint.parameters()]
    micro_loss_value = 0.0
    micro.zero_grad(set_to_none=True)
    for start in range(0, EFFECTIVE_BATCH, MICROBATCH_SIZE):
        stop = start + MICROBATCH_SIZE
        _, logits = micro.forward_window(tokens[start:stop, :4], micro.initial_state(MICROBATCH_SIZE, device=torch.device("cpu")))
        local_loss = distillation_loss(logits, teacher_logits[start:stop], targets[start:stop], valid[start:stop])["total"]
        (local_loss * (MICROBATCH_SIZE / EFFECTIVE_BATCH)).backward()
        micro_loss_value += float(local_loss.detach().item()) * MICROBATCH_SIZE / EFFECTIVE_BATCH
    micro_grads = [parameter.grad.detach().clone() for parameter in micro.parameters()]
    max_gradient_error = max(float((joint_grad - micro_grad).abs().max().item()) for joint_grad, micro_grad in zip(joint_grads, micro_grads, strict=True))
    batch_crossing_modules = [type(module).__name__ for module in joint.modules() if "BatchNorm" in type(module).__name__]
    return {"effective_batch": EFFECTIVE_BATCH, "physical_batch": MICROBATCH_SIZE, "microbatches": MICROBATCH_COUNT, "fixture_tokens_total_per_update": EFFECTIVE_BATCH * 4, "production_contract": {"effective_batch": 8, "physical_batch": 2, "microbatches": 4, "tokens_total_per_update": 2048, "loss_scaling": "each microbatch loss multiplied by valid_tokens_microbatch / 2048; one accumulated backward and one optimizer step", "window_1_teacher_context": "full 512-token prefix for each 2-sequence microbatch"}, "loss_joint": float(joint_loss.detach().item()), "loss_microbatch_weighted": micro_loss_value, "loss_abs_error": abs(float(joint_loss.detach().item()) - micro_loss_value), "max_gradient_abs_error": max_gradient_error, "declared_tolerance": MICROBATCH_TOLERANCE, "within_tolerance": abs(float(joint_loss.detach().item()) - micro_loss_value) <= MICROBATCH_TOLERANCE and max_gradient_error <= MICROBATCH_TOLERANCE, "batch_crossing_normalization_modules": batch_crossing_modules, "equivalence_assumption_verified": not batch_crossing_modules, "optimizer_steps_in_fixture": 0, "no_real_language_updates": True}


def gpu_readiness() -> dict[str, Any]:
    previous = json.loads(PREVIOUS_NOMINAL.read_text(encoding="utf-8"))
    inventory = previous["part_a_inventory"]
    return {
        "source": str(PREVIOUS_NOMINAL.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(PREVIOUS_NOMINAL.read_bytes()).hexdigest(),
        "routes": {
            "current_amd": {
                "model": "AMD Radeon(TM) 820M Graphics",
                "architecture": "not exposed by Win32_VideoController inventory; unresolved without assuming",
                "driver_version": "32.0.13062.3005",
                "driver_status": "Error",
                "documented_windows_rocm_support": False,
                "evidence": "AMD ROCm Radeon/Ryzen Windows matrix for ROCm 7.2.1 lists Windows 11 and specific RX 9070/9070 XT/AI PRO R9700/RX 9060 XT/RX 7900 XTX/PRO W7900/RX 7700 hardware; Radeon 820M is absent. Matrix lists PyTorch 2.9 + ROCm 7.2.1 + Python 3.12.",
                "documentation_url": "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html",
                "authorized_action": "none",
            },
            "runpod_cuda_gpu": {
                "status": "IDENTIFIED_CANDIDATE_REQUIRES_EXPLICIT_SPEND_CONSENT",
                "provider": "RunPod",
                "primary_candidate": {"gpu": "RTX A4000", "memory_gb": 16, "price_per_hour_usd": 0.25, "reason": "current Sol choice; use only if measured workload fits; RTX A5000 remains fallback if memory/performance evidence requires it"},
                "catalog_correction": "Corrected from first artifact: live RunPod catalog reports RTX 3090 stock=low and RTX A6000 stock=low; first artifact incorrectly recorded both as high due missing source stock values.",
                "pods_running_at_inventory": 0,
                "backend": "native CUDA",
                "persistent_volume": {"name": "q4t3-vol", "network_volume_id": "6yrppoqpkz", "size_gb": 50, "data_center": "US-MO-2", "action_when_pod_created": "attach this existing network volume; do not create another volume"},
                "sizing_criterion": "choose GPU memory/price to fit the authorized pilot measurement, not the largest available GPU; this local synthetic unit needs no GPU",
                "candidates": [
                    {"gpu": "RTX A4000", "memory_gb": 16, "price_per_hour_usd": 0.25, "stock": "low"},
                    {"gpu": "RTX A5000", "memory_gb": 24, "price_per_hour_usd": 0.27, "stock": "low"},
                    {"gpu": "RTX 4000 Ada", "memory_gb": 20, "price_per_hour_usd": 0.28, "stock": "low"},
                    {"gpu": "A40", "memory_gb": 48, "price_per_hour_usd": 0.49, "stock": "high"},
                    {"gpu": "RTX 3090", "memory_gb": 24, "price_per_hour_usd": 0.50, "stock": "low"},
                    {"gpu": "RTX A6000", "memory_gb": 48, "price_per_hour_usd": 0.53, "stock": "low"},
                    {"gpu": "RTX 4090", "memory_gb": 24, "price_per_hour_usd": 0.74, "stock": "medium"},
                    {"gpu": "RTX 5090", "memory_gb": 32, "price_per_hour_usd": 0.99, "stock": "medium"},
                ],
                "authorized_action": "do not create pod in this unit; explicit spend authorization still required",
            },
            "authorized_secondary_cpu": {
                "status": "IDENTIFIED_CPU_ONLY",
                "provider": "user VPS",
                "os": "Ubuntu 24.04, kernel 6.8",
                "cpu_cores": 4,
                "ram_total_gb": 7.8,
                "ram_available_gb": 7.1,
                "python": "3.12.3",
                "torch": "not installed",
                "role": "secondary CPU or future microbatching host; smaller memory than laptop, not a solution for full nominal B=8 by itself",
                "authorized_action": "no installation or access performed in this unit",
            },
            "future_resource": {"status": "PENDING", "reason": "No additional resource, budget, machine, or access was assumed."},
        },
        "decision": "IDENTIFIED",
        "decision_reason": "RunPod provides concrete NVIDIA CUDA candidates, but no pod was created because spend approval is not part of this unit; current laptop AMD remains unsupported/unresolved for documented Windows ROCm coverage.",
        "inventory_not_repeated": True,
        "inventory_reference_decision": inventory["decision"],
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema": "omega-core-lm-0-r1-runtime-recovery-and-gpu-readiness-v1", "runtime_recovery": "FAIL", "memory_execution_contract": "BLOCKED", "gpu_target": "UNRESOLVED", "scientific_pilot_started": False, "no_real_language_updates": True, "tests": {}, "errors": []}
    try:
        result["tests"]["normal_recovery_ledger"] = run_normal_recovery_ledger()
        result["tests"]["fault_injection"] = run_fault_injection_tests()
        result["tests"]["storage_audit"] = run_storage_audit()
        result["tests"]["microbatch_equivalence"] = run_microbatch_equivalence()
        result["gpu_readiness"] = gpu_readiness()
        result["gpu_target"] = result["gpu_readiness"]["decision"]
        result["runtime_recovery"] = "PASS" if result["tests"]["fault_injection"]["all_passed"] and result["tests"]["normal_recovery_ledger"]["consolidation_complete"] else "FAIL"
        result["memory_execution_contract"] = "VERIFIED" if result["tests"]["storage_audit"]["compact_copy_reduces_storage"] and result["tests"]["storage_audit"]["student_gradient_within_tolerance"] and result["tests"]["microbatch_equivalence"]["within_tolerance"] else "BLOCKED"
    except Exception as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    result["artifact_notes"] = {"events_ledger": str(EVENTS.relative_to(ROOT)).replace("\\", "/"), "consolidated_from_ledger": True, "no_retain_graph": True, "previous_nominal_attempt_preserved": str(PREVIOUS_NOMINAL.relative_to(ROOT)).replace("\\", "/")}
    digest, file_sha = write_self_hashed(OUTPUT, result)
    print(json.dumps({"artifact": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "artifact_self_hash": digest, "file_sha256": file_sha, "runtime_recovery": result["runtime_recovery"], "memory_execution_contract": result["memory_execution_contract"], "gpu_target": result["gpu_target"], "scientific_pilot_started": False}, sort_keys=True))
    return 0 if not result["errors"] and result["runtime_recovery"] == "PASS" and result["memory_execution_contract"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
