"""Run one deterministic T1 training configuration and persist audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability import InputAdapter, OutputReader, RecurrentCore, TokenVocabulary  # noqa: E402
from t1_trainability.data import OUTPUT_CARDINALITIES, encode_batch, load_jsonl  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_split(task: str, split: str, vocabulary: TokenVocabulary) -> TensorDataset:
    examples = load_jsonl(ROOT / "datasets" / task / f"{split}.jsonl")
    input_ids, mask, query_ids, targets = encode_batch(examples, vocabulary)
    return TensorDataset(input_ids, mask, query_ids, targets)


@torch.no_grad()
def evaluate(
    adapter: InputAdapter,
    core: RecurrentCore,
    reader: OutputReader,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    task: str,
) -> tuple[float, float]:
    adapter.eval()
    core.eval()
    reader.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()
    for input_ids, mask, query_ids, targets in loader:
        logits = reader(core(adapter(input_ids, mask)), query_ids, task)  # type: ignore[arg-type]
        loss = criterion(logits, targets)
        if not finite(loss) or not finite(logits):
            raise FloatingPointError("non-finite validation loss or logits")
        batch_size = targets.shape[0]
        total_loss += float(loss) * batch_size
        correct += int((logits.argmax(dim=-1) == targets).sum())
        count += batch_size
    return total_loss / count, correct / count


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": args.task,
        "variant": args.variant,
        "dimension": args.dimension,
        "slots": args.slots,
        "rounds": args.rounds,
        "seed": args.seed,
        "init_gate_probability": args.init_gate_probability,
        "experiment_reason": args.experiment_reason,
        "gate_weight_decay": 0.0,
        "final_norm": "global-RMS after query-conditioned pooling before task head",
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "gradient_clip_norm": 1.0,
        "max_steps": 5000,
        "max_epochs": 100,
        "eval_every_steps": 100,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "git_commit": git_commit(),
    }
    save_json(output_dir / "config.json", config)

    vocabulary = TokenVocabulary()
    train_data = load_split(args.task, "train", vocabulary)
    val_data = load_split(args.task, "val", vocabulary)
    test_data = load_split(args.task, "test", vocabulary)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    adapter = InputAdapter(len(vocabulary), args.dimension, args.slots, max_length=64)
    core = RecurrentCore(
        args.dimension,
        args.slots,
        args.rounds,
        args.variant,
        init_gate_probability=args.init_gate_probability,
    )
    reader = OutputReader(len(vocabulary), args.dimension)
    modules = nn.ModuleList((adapter, core, reader))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in modules.parameters() if parameter is not core.gate_logits], "weight_decay": 1e-4},
            {"params": [core.gate_logits], "weight_decay": 0.0},
        ],
        lr=3e-4,
    )
    criterion = nn.CrossEntropyLoss()

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    best_accuracy = -math.inf
    best_loss = math.inf
    stale_evaluations = 0
    step = 0
    epoch = 0
    train_loss_sum = 0.0
    train_batches = 0
    started = time.perf_counter()

    def record(metric: dict[str, object]) -> None:
        with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(metric, sort_keys=True) + "\n")

    try:
        for epoch in range(1, 101):
            adapter.train()
            core.train()
            reader.train()
            for input_ids, mask, query_ids, targets in train_loader:
                step += 1
                optimizer.zero_grad(set_to_none=True)
                logits = reader(core(adapter(input_ids, mask)), query_ids, args.task)  # type: ignore[arg-type]
                loss = criterion(logits, targets)
                if not finite(loss) or not finite(logits):
                    raise FloatingPointError(f"non-finite training loss/logits at step {step}")
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}")
                if any(not finite(parameter.grad) for parameter in modules.parameters() if parameter.grad is not None):
                    raise FloatingPointError(f"non-finite gradient at step {step}")
                optimizer.step()
                if any(not finite(parameter) for parameter in modules.parameters()):
                    raise FloatingPointError(f"non-finite parameter at step {step}")
                train_loss_sum += float(loss)
                train_batches += 1

                if step % 100 == 0:
                    val_loss, val_accuracy = evaluate(adapter, core, reader, val_loader, args.task)
                    metric = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": train_loss_sum / train_batches,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "gradient_norm": grad_norm,
                        "finite": True,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    record(metric)
                    train_loss_sum = 0.0
                    train_batches = 0
                    improved = val_accuracy >= best_accuracy + 0.001 or (
                        abs(val_accuracy - best_accuracy) < 0.001 and val_loss < best_loss
                    )
                    if improved:
                        best_accuracy = val_accuracy
                        best_loss = val_loss
                        stale_evaluations = 0
                        torch.save(
                            {
                                "config": config,
                                "step": step,
                                "epoch": epoch,
                                "val_loss": val_loss,
                                "val_accuracy": val_accuracy,
                                "adapter": adapter.state_dict(),
                                "core": core.state_dict(),
                                "reader": reader.state_dict(),
                            },
                            output_dir / "best.pt",
                        )
                    else:
                        stale_evaluations += 1
                    if stale_evaluations >= 10:
                        break
                if step >= 5000:
                    break
            if stale_evaluations >= 10 or step >= 5000:
                break

        if step == 0:
            raise RuntimeError("training produced zero optimizer steps")
        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        adapter.load_state_dict(checkpoint["adapter"])
        core.load_state_dict(checkpoint["core"])
        reader.load_state_dict(checkpoint["reader"])
        test_loss, test_accuracy = evaluate(adapter, core, reader, test_loader, args.task)
        final = {
            "status": "completed",
            "task": args.task,
            "variant": args.variant,
            "dimension": args.dimension,
            "slots": args.slots,
            "rounds": args.rounds,
            "seed": args.seed,
            "steps": step,
            "epochs": epoch,
            "best_val_accuracy": best_accuracy,
            "best_val_loss": best_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "elapsed_seconds": time.perf_counter() - started,
            "finite": True,
        }
        save_json(output_dir / "final.json", final)
        return final
    except Exception as error:
        failure = {
            "status": "failed",
            "task": args.task,
            "variant": args.variant,
            "dimension": args.dimension,
            "slots": args.slots,
            "rounds": args.rounds,
            "seed": args.seed,
            "steps": step,
            "epochs": epoch,
            "finite": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
        save_json(output_dir / "final.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", choices=("single", "shared", "untied", "vector-state"), required=True)
    parser.add_argument("--dimension", type=int, choices=(64, 128), default=64)
    parser.add_argument("--slots", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--rounds", type=int, choices=(1, 2, 4, 6, 8), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--init-gate-probability", type=float, default=0.1)
    parser.add_argument("--experiment-reason", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
