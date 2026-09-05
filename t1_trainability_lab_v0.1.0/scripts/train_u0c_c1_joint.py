"""Train C1 jointly from approved U0-A and C0 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any
import sys

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    READ_MODE_SELECT,
    ROW_VEC,
    SLOT_COUNT,
    SLOT_P,
    SLOT_W,
    UnifiedT1U0,
)
from t1_trainability.workspace import TransformCorrectionMLP  # noqa: E402
from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    DIMENSION,
    TASKS,
    ExampleDataset,
    build_canonical_data,
    build_sequential_h1_table,
    collate,
    immediate_vectors,
    task_loss,
    train_one_step,
)
from train_u0c_c0_oracle import (  # noqa: E402
    H_VALUES,
    MAX_H,
    TRANSFORM_COUNT,
    apply_transform,
)


SEED = 101
TRAIN_TRANSFORM_SEED = 50101
VAL_TRANSFORM_SEED = 50202
TEST_TRANSFORM_SEED = 50303
TRANSFORM_SAMPLES_PER_H = 1024
U0A_CHECKPOINT = ROOT / "campaign" / "u0a_iso_clean_seed101_12000" / "best.pt"
C0_CHECKPOINT = ROOT / "campaign" / "u0c_c0_oracle_seed101_network_trainable" / "final.pt"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class C1JointModel(UnifiedT1U0):
    """Single unified model with C0 correction registered in its state dict."""

    def __init__(self) -> None:
        super().__init__(DIMENSION)
        self.correction_mlp = TransformCorrectionMLP(DIMENSION, transform_count=TRANSFORM_COUNT)


def load_approved_model() -> C1JointModel:
    model = C1JointModel()
    u0a = torch.load(U0A_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(u0a["model"], strict=False)
    c0 = torch.load(C0_CHECKPOINT, map_location="cpu", weights_only=False)
    correction_state = {name.removeprefix("correction_mlp."): value for name, value in c0["model"].items() if name.startswith("correction_mlp.")}
    model.correction_mlp.load_state_dict(correction_state, strict=True)
    return model


def build_optimizer(model: C1JointModel) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if name.startswith("correction_mlp.") or parameter.ndim == 1 or lowered.endswith("bias") or "embedding" in lowered or "decoder" in lowered or "alu_" in lowered or "operation_heads" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(({"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0.0}), lr=3e-4)


def make_transform_split(seed: int, samples_per_h: int = TRANSFORM_SAMPLES_PER_H) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    samples = samples_per_h * len(H_VALUES)
    key_ids = torch.stack([torch.randperm(256, generator=generator)[:6] for _ in range(samples)])
    row_order = torch.stack([torch.randperm(6, generator=generator) for _ in range(samples)])
    values = torch.randn((samples, 6, DIMENSION), generator=generator)
    query_logical = torch.stack([torch.randperm(6, generator=generator)[:MAX_H] for _ in range(samples)])
    transforms = torch.randint(TRANSFORM_COUNT, (samples, MAX_H), generator=generator)
    lengths = torch.tensor([h for h in H_VALUES for _ in range(samples_per_h)], dtype=torch.long)
    active = torch.arange(MAX_H).view(1, -1) < lengths.view(-1, 1)
    targets = torch.zeros((samples, DIMENSION))
    target_deltas = torch.zeros((samples, MAX_H, DIMENSION))
    physical_query = row_order.argsort(dim=1).gather(1, query_logical)
    requested_values = values.gather(1, physical_query.unsqueeze(-1).expand(-1, -1, DIMENSION))
    for round_index in range(MAX_H):
        target_deltas[:, round_index] = apply_transform(requested_values[:, round_index], transforms[:, round_index])
    targets = (target_deltas * active.unsqueeze(-1)).sum(dim=1)
    return {
        "key_ids": key_ids,
        "row_order": row_order,
        "values": values,
        "query_logical": query_logical,
        "query_physical": physical_query,
        "transform_ids": transforms,
        "lengths": lengths,
        "target_deltas": target_deltas,
        "targets": targets,
    }


def transform_manifest(payload: dict[str, Tensor], seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "samples": int(payload["lengths"].shape[0]),
        "h_counts": {str(h): int((payload["lengths"] == h).sum()) for h in H_VALUES},
        "key_ids_sha256": tensor_sha256(payload["key_ids"]),
        "row_order_sha256": tensor_sha256(payload["row_order"]),
        "values_sha256": tensor_sha256(payload["values"]),
        "transform_ids_sha256": tensor_sha256(payload["transform_ids"]),
        "targets_sha256": tensor_sha256(payload["targets"]),
        "target_source": "external values transformed by generator-side apply_transform; never model payload",
    }


def run_transform_batch(model: C1JointModel, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = batch["key_ids"].shape[0]
    memory_keys_logical = model.token_embedding(batch["key_ids"])
    memory_keys = memory_keys_logical.gather(1, batch["row_order"].unsqueeze(-1).expand(-1, -1, DIMENSION))
    memory_types = torch.full(batch["row_order"].shape, ROW_VEC, dtype=torch.long)
    row_mask = torch.ones_like(memory_types, dtype=torch.bool)
    immediates = torch.full((batch_size,), 511, dtype=torch.long)
    source_slots = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    destination_slots = torch.full((batch_size,), SLOT_W, dtype=torch.long)
    read_modes = torch.full((batch_size,), READ_MODE_SELECT, dtype=torch.long)
    presence = torch.zeros((batch_size, SLOT_COUNT), dtype=torch.bool)
    presence[:, SLOT_P] = True
    presence[:, SLOT_W] = True
    state = torch.zeros((batch_size, SLOT_COUNT, DIMENSION))
    predicted_deltas = torch.zeros_like(batch["target_deltas"])
    selected = torch.full((batch_size, MAX_H), -1, dtype=torch.long)
    payload_errors = torch.zeros((batch_size, MAX_H))
    for round_index in range(MAX_H):
        opcodes = torch.where(
            batch["lengths"] > round_index,
            torch.full((batch_size,), OPCODE_IDS["ACCUM_W"], dtype=torch.long),
            torch.full((batch_size,), OPCODE_IDS["EMIT"], dtype=torch.long),
        )
        state[:, SLOT_P, :] = memory_keys_logical[torch.arange(batch_size), batch["query_logical"][:, round_index]]
        before = state[:, SLOT_W, :].clone()
        state, _, result = model.step(
            state,
            memory_keys,
            batch["values"],
            memory_types,
            row_mask,
            opcodes,
            immediates,
            source_slots,
            destination_slots,
            presence,
            read_mode=read_modes,
            transform_id=batch["transform_ids"][:, round_index],
            correction_module=model.correction_mlp,
        )
        predicted_deltas[:, round_index] = state[:, SLOT_W, :] - before
        selected[:, round_index] = result.selected_index
        expected = batch["values"].gather(1, batch["query_physical"][:, round_index].view(batch_size, 1, 1).expand(-1, 1, DIMENSION)).squeeze(1)
        payload_errors[:, round_index] = torch.linalg.vector_norm(result.payload - expected, dim=-1) / torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-8)
    return state[:, SLOT_W, :], predicted_deltas, selected, payload_errors, batch["targets"]


@torch.no_grad()
def evaluate_transformed(model: C1JointModel, payload: dict[str, Tensor], batch_size: int = BATCH_SIZE) -> dict[str, object]:
    dataset = TensorDataset(*(payload[key] for key in ("key_ids", "row_order", "values", "query_logical", "query_physical", "transform_ids", "lengths", "target_deltas", "targets")))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    final_cosine: list[Tensor] = []
    final_error: list[Tensor] = []
    selected_hits: list[Tensor] = []
    payload_errors: list[Tensor] = []
    delta_errors: list[Tensor] = []
    for row in loader:
        batch = dict(zip(("key_ids", "row_order", "values", "query_logical", "query_physical", "transform_ids", "lengths", "target_deltas", "targets"), row))
        output, predicted, selected, read_errors, targets = run_transform_batch(model, batch)
        active = torch.arange(MAX_H).view(1, -1) < batch["lengths"].view(-1, 1)
        expected = batch["query_physical"]
        selected_hits.append((selected == expected) & active)
        payload_errors.append(torch.where(active, read_errors, torch.zeros_like(read_errors)))
        delta_errors.append(torch.where(active, torch.linalg.vector_norm(predicted - batch["target_deltas"], dim=-1) / torch.linalg.vector_norm(batch["target_deltas"], dim=-1).clamp_min(1e-8), torch.zeros_like(read_errors)))
        final_cosine.append(torch.nn.functional.cosine_similarity(output, targets, dim=-1))
        final_error.append(torch.linalg.vector_norm(output - targets, dim=-1) / torch.linalg.vector_norm(targets, dim=-1).clamp_min(1e-8))
    all_hits = torch.cat(selected_hits)
    all_payload = torch.cat(payload_errors)
    all_delta = torch.cat(delta_errors)
    lengths = payload["lengths"]
    cosines = torch.cat(final_cosine)
    errors = torch.cat(final_error)
    by_h: dict[str, object] = {}
    for h in H_VALUES:
        mask = lengths == h
        by_h[str(h)] = {
            "samples": int(mask.sum()),
            "final_cosine": float(cosines[mask].mean()),
            "final_relative_error": float(errors[mask].mean()),
            "top1_accuracy_by_round": {str(r + 1): float(all_hits[mask, r].float().mean()) for r in range(h)},
            "payload_relative_error_by_round": {str(r + 1): float(all_payload[mask, r].mean()) for r in range(h)},
            "epsilon_delta_relative_by_round": {str(r + 1): float(all_delta[mask, r].mean()) for r in range(h)},
        }
    return {
        "by_h": by_h,
        "top1_accuracy_active_rounds": float(all_hits.float().sum() / ((lengths).sum())),
        "max_payload_relative_error": float(all_payload.max()),
        "max_epsilon_delta_relative": float(all_delta.max()),
        "min_final_cosine": float(cosines.min()),
        "max_final_relative_error": float(errors.max()),
        "passes_final_gate": float(cosines.min()) > 0.999 and float(errors.max()) <= 0.01,
        "passes_delta_gate": float(all_delta.max()) <= 0.01,
    }


@torch.no_grad()
def evaluate_oracle_transform(model: C1JointModel, payload: dict[str, Tensor]) -> dict[str, object]:
    workspace = torch.zeros_like(payload["targets"])
    predicted = []
    for round_index in range(MAX_H):
        evidence = payload["values"].gather(1, payload["query_physical"][:, round_index].view(-1, 1, 1).expand(-1, 1, DIMENSION)).squeeze(1)
        correction = model.correction_mlp(evidence, workspace, payload["transform_ids"][:, round_index])
        delta = evidence + correction
        active = (payload["lengths"] > round_index).unsqueeze(-1)
        workspace = torch.where(active, workspace + delta, workspace)
        predicted.append(torch.where(active, delta, torch.zeros_like(delta)))
    predicted_deltas = torch.stack(predicted, dim=1)
    target = payload["targets"]
    errors = torch.linalg.vector_norm(workspace - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-8)
    cosines = torch.nn.functional.cosine_similarity(workspace, target, dim=-1)
    target_delta_error = torch.linalg.vector_norm(predicted_deltas - payload["target_deltas"], dim=-1) / torch.linalg.vector_norm(payload["target_deltas"], dim=-1).clamp_min(1e-8)
    active = torch.arange(MAX_H).view(1, -1) < payload["lengths"].view(-1, 1)
    by_h = {}
    for h in H_VALUES:
        mask = payload["lengths"] == h
        by_h[str(h)] = {
            "samples": int(mask.sum()),
            "final_cosine": float(cosines[mask].mean()),
            "final_relative_error": float(errors[mask].mean()),
            "epsilon_delta_relative_by_round": {str(r + 1): float(target_delta_error[mask, r].mean()) for r in range(h)},
        }
    return {"by_h": by_h, "min_final_cosine": float(cosines.min()), "max_final_relative_error": float(errors.max()), "max_epsilon_delta_relative": float(target_delta_error[active].max()), "passes_final_gate": float(cosines.min()) > 0.999 and float(errors.max()) <= 0.01, "passes_delta_gate": float(target_delta_error[active].max()) <= 0.01, "predicted_deltas": predicted_deltas}


def validation_score(historical: dict[str, object], transformed: dict[str, object]) -> float:
    values: list[float] = []
    for task, matrix in historical.items():
        if task == "workspace_accumulation":
            values.extend(1.0 if value < 1e-3 else 0.0 for rows in matrix.values() for value in rows.values())
        else:
            values.extend(value for rows in matrix.values() for value in rows.values())
    values.append(max(0.0, min(1.0, 1.0 - float(transformed["max_final_relative_error"]))))
    return sum(values) / len(values)


def save_checkpoint(path: Path, config: dict[str, object], model: C1JointModel, optimizer: torch.optim.Optimizer, step: int, best_score: float, best_step: int) -> None:
    torch.save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best_score": best_score, "best_step": best_step, "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state()}, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_joint_seed101")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = build_canonical_data(args.output_dir)
    transform_train = make_transform_split(TRAIN_TRANSFORM_SEED)
    transform_val = make_transform_split(VAL_TRANSFORM_SEED)
    transform_test = make_transform_split(TEST_TRANSFORM_SEED)
    loaders = {task: DataLoader(ExampleDataset(datasets[task]["train"]), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + index), collate_fn=collate) for index, task in enumerate(TASKS)}
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    h1_loader = DataLoader(ExampleDataset(build_sequential_h1_table()), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + len(TASKS)), collate_fn=collate)
    h1_iterator = iter(h1_loader)
    transform_loader = DataLoader(TensorDataset(*(transform_train[key] for key in ("key_ids", "row_order", "values", "query_logical", "query_physical", "transform_ids", "lengths", "target_deltas", "targets"))), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + 10),)
    transform_iterator = iter(transform_loader)
    model = load_approved_model()
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = build_optimizer(model)
    config: dict[str, object] = {"phase": "T1-U0-C1 joint", "seed": SEED, "steps": args.steps, "batch_size": BATCH_SIZE, "dimension": DIMENSION, "loss": "(six historical task losses + transformed loss) / 7", "reader": "one SharedMemoryReader; read_mode per instruction; historical workspace BLEND; transformed task SELECT", "transformed_query": "explicit P direction from sampled codebook key; no old Norm(W)+index route", "optimizer": "AdamW(lr=3e-4); weight_decay=1e-4 base decay, 0 corrector/typed; one step per superstep", "schedule": "sequential H1 replay 5:1 with composition; one batch each of six tasks plus transformed task", "initial_u0a": str(U0A_CHECKPOINT.relative_to(ROOT)), "initial_c0": str(C0_CHECKPOINT.relative_to(ROOT)), "train_manifests": transform_manifest(transform_train, TRAIN_TRANSFORM_SEED), "val_manifests": transform_manifest(transform_val, VAL_TRANSFORM_SEED), "test_manifests": transform_manifest(transform_test, TEST_TRANSFORM_SEED), "validation_steps": [0, 100, 500, 1000, "every 1000 thereafter"], "frozen_structural": ["core.workspace_correction"], "final_checkpoint": "final.pt", "best_checkpoint": "best.pt"}
    save_json(args.output_dir / "config.json", config)
    metrics_path = args.output_dir / "metrics.jsonl"
    latest_path = args.output_dir / "latest.pt"
    start_step = 0
    best_score = -float("inf")
    best_step = 0
    if args.resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_score, best_step = int(state["step"]), float(state["best_score"]), int(state["best_step"])
        random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
    else:
        metrics_path.unlink(missing_ok=True)
        torch.save({"config": config, "model": model.state_dict()}, args.output_dir / "step0.pt")
    started = time.perf_counter()
    def validate(step: int) -> dict[str, object]:
        model.eval()
        historical = __import__("train_u0a").evaluate_all(model, datasets, "val")
        transformed = evaluate_transformed(model, transform_val)
        oracle = evaluate_oracle_transform(model, transform_val)
        score = validation_score(historical, transformed)
        model.train()
        return {"step": step, "validation_score": score, "historical": historical, "transformed_real": transformed, "transformed_oracle": {key: value for key, value in oracle.items() if key != "predicted_deltas"}}
    if start_step == 0:
        initial_validation = validate(0)
        initial_validation["kind"] = "validation"
        metrics_path.write_text(json.dumps(initial_validation, sort_keys=True) + "\n", encoding="utf-8")
        best_score = float(initial_validation["validation_score"])
        save_checkpoint(args.output_dir / "best.pt", config, model, optimizer, 0, best_score, 0)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        task_losses: dict[str, Tensor] = {}
        for task in TASKS:
            if task == "sequential_update" and step % 6 != 0:
                try:
                    batch = next(h1_iterator)
                except StopIteration:
                    h1_iterator = iter(h1_loader)
                    batch = next(h1_iterator)
            else:
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(loaders[task])
                    batch = next(iterators[task])
            value = train_one_step(model, task, batch)
            if not torch.isfinite(value):
                raise FloatingPointError(f"non-finite historical loss at step {step}: {task}")
            task_losses[task] = value
            (value / 7.0).backward()
        try:
            transform_tuple = next(transform_iterator)
        except StopIteration:
            transform_iterator = iter(transform_loader)
            transform_tuple = next(transform_iterator)
        transform_batch = dict(zip(("key_ids", "row_order", "values", "query_logical", "query_physical", "transform_ids", "lengths", "target_deltas", "targets"), transform_tuple))
        output, predicted, _, _, targets = run_transform_batch(model, transform_batch)
        active = (torch.arange(MAX_H).view(1, -1) < transform_batch["lengths"].view(-1, 1)).unsqueeze(-1)
        delta_loss = (((predicted - transform_batch["target_deltas"]) * active) ** 2).sum() / active.sum().clamp_min(1)
        final_loss = (output - targets).square().mean()
        transform_loss = delta_loss + 0.25 * final_loss
        if not torch.isfinite(transform_loss):
            raise FloatingPointError(f"non-finite transformed loss at step {step}")
        (transform_loss / 7.0).backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        metric: dict[str, object] = {"kind": "train", "step": step, "loss": float(sum(value.detach() for value in task_losses.values()) / 7.0 + transform_loss.detach() / 7.0), "task_loss": {task: float(value.detach()) for task, value in task_losses.items()}, "transformed_delta_loss": float(delta_loss.detach()), "transformed_final_loss": float(final_loss.detach()), "gradient_norm": grad_norm}
        if step in {100, 500, 1000} or step % 1000 == 0 or step == args.steps:
            metric["validation"] = validate(step)
            score = float(metric["validation"]["validation_score"])
            if score > best_score:
                best_score, best_step = score, step
                save_checkpoint(args.output_dir / "best.pt", config, model, optimizer, step, best_score, best_step)
        with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(metric, sort_keys=True) + "\n")
        if step % 100 == 0 or step == args.steps:
            save_checkpoint(latest_path, config, model, optimizer, step, best_score, best_step)
    torch.save({"config": config, "model": model.state_dict(), "step": args.steps}, args.output_dir / "final.pt")
    model.eval()
    final_historical = __import__("train_u0a").evaluate_all(model, datasets, "test")
    final_real = evaluate_transformed(model, transform_test)
    final_oracle = evaluate_oracle_transform(model, transform_test)
    update_groups: dict[str, dict[str, float | int]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split(".", 1)[0]
        entry = update_groups.setdefault(group, {"trainable_parameter_count": 0, "changed_parameter_count": 0, "update_l2": 0.0})
        entry["trainable_parameter_count"] += parameter.numel()
        update = parameter.detach() - initial_parameters[name]
        entry["update_l2"] += float(update.square().sum())
        if not torch.equal(parameter.detach(), initial_parameters[name]):
            entry["changed_parameter_count"] += 1
    final = {"status": "completed", "seed": SEED, "steps": args.steps, "best_step": best_step, "best_validation_score": best_score, "elapsed_seconds": time.perf_counter() - started, "final_checkpoint": str((args.output_dir / "final.pt").relative_to(ROOT)), "best_checkpoint": str((args.output_dir / "best.pt").relative_to(ROOT)), "historical_test": final_historical, "transformed_real_reader": final_real, "transformed_oracle_reader": {key: value for key, value in final_oracle.items() if key != "predicted_deltas"}, "trainable_parameter_updates": update_groups}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
