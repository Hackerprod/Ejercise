"""Train pointer canary with solver payloads and the production tied decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_pointer_oracle import evaluate  # noqa: E402
from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    ExampleDataset,
    build_canonical_data,
    build_optimizer,
    collate,
    materialize,
    save_json,
    task_loss,
)
from t1_trainability.unified import OPCODE_IDS, CandidateState, ReadResult, SLOT_P, UnifiedT1U0  # noqa: E402


def run_oracle(model: UnifiedT1U0, batch: dict[str, object], rounds: int) -> torch.Tensor:
    data = materialize(model, batch)  # type: ignore[arg-type]
    state = data["state"]
    pointer_ids = [int(value) for value in data["initial_ids"][:, SLOT_P].tolist()]
    for round_index in range(rounds):
        active = data["hops"] > round_index
        payload = torch.zeros_like(state[:, SLOT_P, :])
        for row_index in torch.where(active)[0].tolist():
            current = pointer_ids[row_index]
            mapping = {
                int(data["key_ids"][row_index, memory_index]): int(data["value_ids"][row_index, memory_index])
                for memory_index in torch.where(data["row_mask"][row_index])[0].tolist()
            }
            destination = mapping[current]
            payload[row_index] = model.token_embedding(torch.tensor(destination))
            pointer_ids[row_index] = destination
        opcode = torch.where(active, torch.full_like(data["hops"], OPCODE_IDS["READ_P"]), torch.full_like(data["hops"], OPCODE_IDS["EMIT"]))
        read_result = ReadResult(
            payload=payload,
            attention=torch.zeros(state.shape[0], data["memory_types"].shape[1]),
            margin=torch.zeros(state.shape[0]),
            valid=active,
        )
        candidates = model.core(
            model.normalize_state(state, data["presence"]),
            model.opcode_embedding(opcode),
            model.token_embedding(data["immediates"][:, round_index]),
            payload,
            model.slot_type_embeddings,
            data["presence"],
        )
        state = model.commit(
            state,
            CandidateState(candidates.values),
            read_result,
            opcode,
            torch.full_like(data["hops"], SLOT_P),
            data["presence"],
        )
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0a_pointer_oracle_seed101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets = build_canonical_data(args.output_dir)
    model = UnifiedT1U0(64)
    optimizer = build_optimizer(model)
    loader = DataLoader(ExampleDataset(datasets["pointer_chasing"]["train"]), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    iterator = iter(loader)
    best_validation = -1.0
    best_step = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        state = run_oracle(model, batch, batch["opcodes"].shape[1])
        loss = task_loss(model, "pointer_chasing", state, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite oracle pointer loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            validation = evaluate(model, datasets["pointer_chasing"]["val"], rounds=4, direct_decoder=True)
            metric = {"step": step, "loss": float(loss.detach()), "gradient_norm": grad_norm, "validation_accuracy_hop4": validation["4"]}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if validation["4"] > best_validation:
                best_validation = validation["4"]
                best_step = step
                torch.save({"step": step, "model": model.state_dict()}, args.output_dir / "best.pt")
            model.train()
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test = {str(rounds): evaluate(model, datasets["pointer_chasing"]["test"], rounds, direct_decoder=True) for rounds in (1, 2, 4)}
    final = {"status": "completed", "finite": True, "seed": args.seed, "steps": args.steps, "best_step": best_step, "best_validation_accuracy_hop4": best_validation, "reader": "oracle", "reader_calls": model.memory_reader.call_count, "test_accuracy_by_round_and_hop": test, "elapsed_seconds": time.perf_counter() - started}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
