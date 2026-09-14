"""CPU preparation and explicitly authorized OMEGA update-efficiency gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT / "campaign" / "omega_core_lm_0_gpu_environment_preparation"
sys.path.insert(0, str(RUNNER_DIR))

from omega_nominal_microbatch_runner import (  # noqa: E402
    EFFECTIVE_BATCH,
    RunLedger,
    construct_adamw,
    microbatch_count_for_physical_batch,
    microbatch_update,
    set_technical_seed,
    memory_observed,
    OmegaCoreLM0R1Technical,
)


GATE_ID = "OMEGA-CORE-LM-0-R1-PILOT-TRAINING-EFFICIENCY-GATE"
SYNTHETIC_SEED = 9100
SYNTHETIC_VOCAB = 7
CONFIGS = (("A", 2), ("B", 4), ("C", 8))


class TinyTeacher(torch.nn.Module):
    def __init__(self, vocab_size: int = SYNTHETIC_VOCAB) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size) / 100.0,
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> Any:
        return type("Output", (), {"logits": self.table[input_ids % self.table.shape[0]]})()


class AppendOnlyJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value: dict[str, Any]) -> None:
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def validate_gate_options(*, device_name: str, dry_run: bool, torch_compile: bool, authorize_gpu_benchmark: bool) -> torch.device:
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no benchmark was run")
    if device_name == "cpu" and torch_compile:
        raise ValueError("--torch-compile requires --device cuda; CPU mode is dry-run only")
    if device_name == "cpu" and not dry_run:
        raise ValueError("CPU mode requires --dry-run and never records performance")
    if device_name == "cuda" and dry_run:
        raise ValueError("CUDA mode is benchmark mode; omit --dry-run")
    if device_name == "cuda" and not authorize_gpu_benchmark:
        raise PermissionError("CUDA benchmark requires --authorize-gpu-benchmark")
    if torch_compile and not hasattr(torch, "compile"):
        raise RuntimeError("--torch-compile requested but this PyTorch has no torch.compile")
    return torch.device(device_name)


def synthetic_fixture(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    source = torch.arange(EFFECTIVE_BATCH * 513, dtype=torch.long, device=device).reshape(EFFECTIVE_BATCH, 513) % SYNTHETIC_VOCAB
    valid_mask = torch.ones(EFFECTIVE_BATCH, 256, dtype=torch.bool, device=device)
    input_valid_mask = torch.ones_like(valid_mask)
    document_ids = [f"synthetic-document-{index}" for index in range(EFFECTIVE_BATCH)]
    return source, valid_mask, input_valid_mask, document_ids


def run_config(*, label: str, physical_batch: int, device: torch.device, run_root: Path, run_id: str, torch_compile: bool, measure: bool) -> dict[str, Any]:
    microbatch_count = microbatch_count_for_physical_batch(physical_batch)
    torch.manual_seed(SYNTHETIC_SEED)
    student = OmegaCoreLM0R1Technical(vocab_size=SYNTHETIC_VOCAB, dimension=2, slots=1, rounds=1).to(device=device, dtype=torch.float32)
    if torch_compile:
        student = torch.compile(student)
    optimizer = construct_adamw(student)
    teacher = TinyTeacher().to(device=device, dtype=torch.float32)
    source, valid_mask, input_valid_mask, document_ids = synthetic_fixture(device)
    ledger = RunLedger(run_root / "runs" / f"{run_id}-{label}", f"{run_id}-{label}")
    if measure:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    state, metrics = microbatch_update(
        ledger=ledger,
        run_id=f"{run_id}-{label}",
        variant="shared_K1",
        update=0,
        student=student,
        teacher=teacher,
        source=source,
        valid_mask=valid_mask,
        input_valid_mask=input_valid_mask,
        optimizer=optimizer,
        window=0,
        persistent_state=None,
        physical_batch=physical_batch,
        document_ids=document_ids,
    )
    seconds_per_update = None
    cuda_seconds_per_update = None
    memory = memory_observed(device)
    if measure:
        torch.cuda.synchronize(device)
        seconds_per_update = time.perf_counter() - started
        cuda_seconds_per_update = metrics["cuda_elapsed_seconds"]
    return {
        "schema_version": 1,
        "config": label,
        "status": "completed",
        "device": device.type,
        "dtype": "float32",
        "torch_compile": torch_compile,
        "physical_batch": physical_batch,
        "microbatches": microbatch_count,
        "effective_batch": EFFECTIVE_BATCH,
        "valid_tokens": metrics["valid_tokens"],
        "state_shape": list(state.shape),
        "loss_summary": metrics["loss_summary"],
        "pre_clip_grad_norm": metrics["pre_clip_grad_norm"],
        "seconds_per_update": seconds_per_update,
        "cuda_seconds_per_update": cuda_seconds_per_update,
        "gpu_reserved_bytes": memory["gpu_reserved_bytes"] if measure else None,
        "gpu_peak_reserved_bytes": memory["gpu_peak_reserved_bytes"] if measure else None,
        "ledger_path": str(ledger.events_path),
    }


def run_gate(*, device_name: str, output_dir: Path, dry_run: bool, torch_compile: bool, authorize_gpu_benchmark: bool) -> dict[str, Any]:
    device = validate_gate_options(device_name=device_name, dry_run=dry_run, torch_compile=torch_compile, authorize_gpu_benchmark=authorize_gpu_benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_technical_seed(SYNTHETIC_SEED)
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    ledger = AppendOnlyJsonl(output_dir / "efficiency_gate_ledger.jsonl")
    report = AppendOnlyJsonl(output_dir / "efficiency_gate_report.jsonl")
    results: list[dict[str, Any]] = []
    for label, physical_batch in CONFIGS:
        result = run_config(label=label, physical_batch=physical_batch, device=device, run_root=output_dir, run_id=run_id, torch_compile=torch_compile, measure=device.type == "cuda")
        results.append(result)
        ledger.append({"schema_version": 1, "record_type": "config_result", "gate": GATE_ID, "run_id": run_id, **result})
    summary = {
        "schema_version": 1,
        "gate": GATE_ID,
        "run_id": run_id,
        "mode": "cpu_dry_run" if dry_run else "cuda_benchmark",
        "performance_measured": not dry_run,
        "precision": "FP32",
        "results": results,
    }
    report.append(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=GATE_ID)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Run one synthetic update/config without performance measurement")
    parser.add_argument("--torch-compile", action="store_true", help="Use torch.compile in FP32 CUDA benchmark mode")
    parser.add_argument("--authorize-gpu-benchmark", action="store_true", help="Explicit authorization boundary for real CUDA benchmark spend")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)
    print(json.dumps(run_gate(device_name=args.device, output_dir=args.output_dir, dry_run=args.dry_run, torch_compile=args.torch_compile, authorize_gpu_benchmark=args.authorize_gpu_benchmark), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
