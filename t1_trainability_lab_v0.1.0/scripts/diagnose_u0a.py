"""Read-only U0-A seed101 diagnostics required by Addendum 4."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    CandidateState,
    ReadResult,
    READ_OPCODE_IDS,
    SLOT_W,
    UnifiedT1U0,
)
from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    TASKS,
    build_optimizer,
    build_canonical_data,
    collate,
    materialize,
    run_rounds,
    task_loss,
)
from torch.utils.data import DataLoader  # noqa: E402
from train_u0a import ExampleDataset  # noqa: E402


def load_model(checkpoint_path: Path) -> UnifiedT1U0:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def empty_read(batch_size: int, memory_width: int, dimension: int) -> ReadResult:
    return ReadResult(
        payload=torch.zeros(batch_size, dimension),
        attention=torch.zeros(batch_size, memory_width),
        margin=torch.zeros(batch_size),
        valid=torch.zeros(batch_size, dtype=torch.bool),
    )


@torch.no_grad()
def run_rounds_without_workspace_correction(model: UnifiedT1U0, batch: dict[str, object], rounds: int) -> torch.Tensor:
    data = materialize(model, batch)  # type: ignore[arg-type]
    state = data["state"]
    for round_index in range(rounds):
        immediate_vectors = model.token_embedding(data["immediates"][:, round_index])
        opcode = data["opcodes"][:, round_index]
        if torch.isin(opcode, torch.tensor(tuple(READ_OPCODE_IDS))).any():
            read_result = model.memory_reader(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                opcode,
                immediate_vectors,
                data["source_slots"][:, round_index],
            )
        else:
            read_result = empty_read(state.shape[0], data["memory_types"].shape[1], model.dimension)
        candidates = model.core(
            model.normalize_state(state, data["presence"]),
            model.opcode_embedding(opcode),
            immediate_vectors,
            read_result.payload,
            model.slot_type_embeddings,
            data["presence"],
        )
        values = candidates.values.clone()
        values[:, SLOT_W, :] = 0.0
        state = model.commit(
            state,
            CandidateState(values),
            read_result,
            opcode,
            data["destination_slots"][:, round_index],
            data["presence"],
        )
    return state


@torch.no_grad()
def workspace_errors(model: UnifiedT1U0, examples: list[object], *, force_zero: bool) -> dict[str, dict[str, float]]:
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    values: dict[str, dict[str, list[float]]] = {str(h): {str(r): [] for r in (1, 2, 4, 6)} for h in (2, 4, 6)}
    for batch in loader:
        for rounds in (1, 2, 4, 6):
            state = run_rounds_without_workspace_correction(model, batch, rounds) if force_zero else run_rounds(model, batch, rounds)
            error = torch.linalg.vector_norm(state[:, SLOT_W] - batch["target_vectors"], dim=-1) / torch.linalg.vector_norm(batch["target_vectors"], dim=-1).clamp_min(1e-8)
            for hop in (2, 4, 6):
                selected = batch["hops"] == hop
                values[str(hop)][str(rounds)].extend(error[selected].tolist())
    return {hop: {rounds: sum(items) / len(items) for rounds, items in rows.items()} for hop, rows in values.items()}


def optimizer_isolation(model: UnifiedT1U0, batch: dict[str, object]) -> dict[str, object]:
    model.train()
    optimizer = build_optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    loss = task_loss(model, "pointer_chasing", run_rounds(model, batch, batch["opcodes"].shape[1]), batch)  # type: ignore[arg-type]
    loss.backward()
    inactive_prefixes = ("core.", "commit.operation_heads.", "evidence_decoder.", "register_decoder.", "workspace_decoder.")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name.startswith(inactive_prefixes)}
    gradients = {name: parameter.grad is None for name, parameter in model.named_parameters() if name.startswith(inactive_prefixes)}
    gradient_norms = {name: None if parameter.grad is None else float(parameter.grad.norm()) for name, parameter in model.named_parameters() if name.startswith(inactive_prefixes)}
    optimizer.step()
    unchanged = {name: torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if name in before}
    return {
        "loss": float(loss.detach()),
        "inactive_grad_is_none": gradients,
        "inactive_grad_norm": gradient_norms,
        "inactive_bit_identical_after_step": unchanged,
        "all_inactive_grads_none": all(gradients.values()),
        "all_inactive_bit_identical": all(unchanged.values()),
    }


def workspace_gradient(model: UnifiedT1U0, batch: dict[str, object]) -> dict[str, object]:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = task_loss(model, "workspace_accumulation", run_rounds(model, batch, batch["opcodes"].shape[1]), batch)  # type: ignore[arg-type]
    loss.backward()
    parameters = {name: parameter for name, parameter in model.named_parameters() if name.startswith("core.mlp.network.2.")}
    return {name: {"grad_is_none": parameter.grad is None, "grad_norm": None if parameter.grad is None else float(parameter.grad.norm())} for name, parameter in parameters.items()}


@torch.no_grad()
def workspace_payload_errors(model: UnifiedT1U0, examples: list[object]) -> dict[str, float]:
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    errors: dict[str, list[float]] = {str(round_index): [] for round_index in range(6)}
    for batch in loader:
        data = materialize(model, batch)  # type: ignore[arg-type]
        state = data["state"]
        for round_index in range(6):
            if round_index >= data["raw_values"].shape[1]:
                break
            immediate_vectors = model.token_embedding(data["immediates"][:, round_index])
            opcode = data["opcodes"][:, round_index]
            result = model.memory_reader(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                opcode,
                immediate_vectors,
                data["source_slots"][:, round_index],
            )
            active = data["hops"] > round_index
            expected = data["raw_values"][:, round_index, :]
            if active.any():
                relative = torch.linalg.vector_norm(result.payload[active] - expected[active], dim=-1) / torch.linalg.vector_norm(expected[active], dim=-1).clamp_min(1e-8)
                errors[str(round_index)].extend(relative.tolist())
            values = model.core(
                model.normalize_state(state, data["presence"]),
                model.opcode_embedding(opcode),
                immediate_vectors,
                result.payload,
                model.slot_type_embeddings,
                data["presence"],
            )
            state = model.commit(state, values, result, opcode, data["destination_slots"][:, round_index], data["presence"])
    return {round_index: sum(values) / len(values) for round_index, values in errors.items() if values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0a_seed101" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0a_seed101" / "diagnostics.json")
    args = parser.parse_args()
    datasets = build_canonical_data(args.checkpoint.parent)
    model = load_model(args.checkpoint)
    normal = workspace_errors(model, datasets["workspace_accumulation"]["test"], force_zero=False)
    forced = workspace_errors(model, datasets["workspace_accumulation"]["test"], force_zero=True)
    pointer_batch = next(iter(DataLoader(ExampleDataset(datasets["pointer_chasing"]["train"][:BATCH_SIZE]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)))
    workspace_batch = next(iter(DataLoader(ExampleDataset(datasets["workspace_accumulation"]["train"][:BATCH_SIZE]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)))
    output = {
        "checkpoint": str(args.checkpoint),
        "workspace_normal": normal,
        "workspace_forced_correction_zero": forced,
        "workspace_correction_gradient": workspace_gradient(model, workspace_batch),
        "workspace_payload_relative_error_by_round": workspace_payload_errors(model, datasets["workspace_accumulation"]["test"]),
        "optimizer_isolation_pointer_batch": optimizer_isolation(model, pointer_batch),
        "scheduler": {"present": False, "optimizer_lr": 3e-4, "configured_total_steps": None, "note": "train_u0a.py uses constant AdamW learning rate; no 5000-step or 30000-step scheduler exists"},
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
