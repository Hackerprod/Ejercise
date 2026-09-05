"""Pre-train UnifiedT1U0 ALU on the complete 3072-transition table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    CanonicalExample,
    ExampleDataset,
    IMM_ZERO,
    VALUE_BASE,
    build_optimizer,
    class_ids_for_task,
    collate,
    save_json,
    task_loss,
    run_rounds_with_trace,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_R, UnifiedT1U0  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("ALU_ADD", "ALU_SUB", "ALU_MUL")


def make_table() -> list[CanonicalExample]:
    rows: list[CanonicalExample] = []
    for operation in OPERATIONS:
        for initial in range(32):
            for operand in range(32):
                if operation == "ALU_ADD":
                    target = (initial + operand) % 32
                elif operation == "ALU_SUB":
                    target = (initial - operand) % 32
                else:
                    target = (initial * operand) % 32
                rows.append(
                    CanonicalExample(
                        "sequential_update",
                        (-1, VALUE_BASE + initial, -1, -1),
                        tuple(),
                        tuple(),
                        tuple(),
                        tuple(),
                        tuple(),
                        (OPCODE_IDS[operation],) + (OPCODE_IDS["EMIT"],) * 5,
                        (VALUE_BASE + operand,) + (IMM_ZERO,) * 5,
                        (SLOT_R,) * 6,
                        (SLOT_R,) * 6,
                        (False, True, False, False),
                        VALUE_BASE + target,
                        None,
                        1,
                    )
                )
    return rows


@torch.no_grad()
def evaluate(model: UnifiedT1U0, examples: list[CanonicalExample]) -> dict[str, float | int]:
    class_ids = class_ids_for_task("sequential_update", torch.device("cpu"))
    hits: dict[str, list[bool]] = {operation: [] for operation in OPERATIONS}
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    for batch in loader:
        state = run_rounds_with_trace(model, batch, 1)[0]
        logits = model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids))
        predictions = class_ids[logits.argmax(-1)]
        for operation in OPERATIONS:
            # Table order is operation-major, but select from explicit opcode.
            selected = batch["opcodes"][:, 0] == OPCODE_IDS[operation]
            hits[operation].extend((predictions[selected] == batch["target_ids"][selected]).tolist())
    return {operation: sum(values) / len(values) for operation, values in hits.items()} | {"overall": sum(sum(values) for values in hits.values()) / sum(len(values) for values in hits.values()), "total": sum(len(values) for values in hits.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    examples = make_table()
    model = UnifiedT1U0(64)
    optimizer = build_optimizer(model)
    loader = DataLoader(ExampleDataset(examples), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    iterator = iter(loader)
    best_score = -1.0
    best_step = 0
    started = time.perf_counter()
    metrics_path = args.output_dir / "metrics.jsonl"
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        state, trace = run_rounds_with_trace(model, batch, 1)
        loss = task_loss(model, "sequential_update", state, batch, trace)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ALU loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            model.eval()
            scores = evaluate(model, examples)
            score = min(float(scores[operation]) for operation in OPERATIONS)
            metric = {"step": step, "loss": float(loss.detach()), "gradient_norm": grad_norm, **scores}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if score > best_score:
                best_score = score
                best_step = step
                torch.save({"step": step, "model": model.state_dict()}, args.output_dir / "best.pt")
            model.train()
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    final_scores = evaluate(model, examples)
    final = {"status": "completed", "seed": args.seed, "steps": args.steps, "transitions": len(examples), "best_step": best_step, "best_min_operation_accuracy": best_score, "test": final_scores, "elapsed_seconds": time.perf_counter() - started, "finite": True}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
