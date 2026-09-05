"""Train/evaluate isolated U0-C0 with exact oracle reader payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import UnifiedT1U0  # noqa: E402
from t1_trainability.workspace import TransformCorrectionMLP  # noqa: E402


SEED = 101
DATASET_SEEDS = {"train": 10101, "val": 20202, "test": 30303}
DIMENSION = 64
TRANSFORM_COUNT = 4
TRANSFORM_NAMES = ("identity", "negation", "circular_shift", "pair_signed_permutation")
H_VALUES = (1, 2, 4, 6)
MAX_H = max(H_VALUES)
SAMPLES_PER_H = 2048
ROUNDS = (1, 2, 4, 6)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def apply_transform(payload: Tensor, transform_id: Tensor) -> Tensor:
    """Apply fixed generator-only orthogonal transform to D-dimensional payload."""
    output = payload.clone()
    identity = transform_id == 0
    output = torch.where((transform_id == 1).unsqueeze(-1), -payload, output)
    output = torch.where((transform_id == 2).unsqueeze(-1), torch.roll(payload, shifts=1, dims=-1), output)
    pair = payload.reshape(*payload.shape[:-1], DIMENSION // 2, 2)
    pair_signed = torch.stack((pair[..., 1], -pair[..., 0]), dim=-1).reshape_as(payload)
    output = torch.where((transform_id == 3).unsqueeze(-1), pair_signed, output)
    if not bool((identity | (transform_id == 1) | (transform_id == 2) | (transform_id == 3)).all()):
        raise ValueError("unknown transform id")
    return output


def make_split(seed: int) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    evidence = torch.randn((SAMPLES_PER_H * len(H_VALUES), MAX_H, DIMENSION), generator=generator)
    transform_ids = torch.randint(0, TRANSFORM_COUNT, (len(evidence), MAX_H), generator=generator)
    for row in range(len(evidence)):
        length = H_VALUES[row // SAMPLES_PER_H]
        if length > 1 and bool((transform_ids[row, :length] == transform_ids[row, 0]).all()):
            transform_ids[row, 1] = (transform_ids[row, 0] + 1) % TRANSFORM_COUNT
    deltas = torch.zeros_like(evidence)
    for round_index in range(MAX_H):
        deltas[:, round_index, :] = apply_transform(evidence[:, round_index, :], transform_ids[:, round_index])
    lengths = torch.tensor([h for h in H_VALUES for _ in range(SAMPLES_PER_H)], dtype=torch.long)
    active = torch.arange(MAX_H).view(1, -1) < lengths.view(-1, 1)
    targets = (deltas * active.unsqueeze(-1)).sum(dim=1)
    return {"evidence": evidence, "transform_ids": transform_ids, "lengths": lengths, "deltas": deltas, "targets": targets}


def dataset_manifest(payload: dict[str, Tensor], seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "samples": int(payload["lengths"].shape[0]),
        "h_counts": {str(h): int((payload["lengths"] == h).sum()) for h in H_VALUES},
        "evidence_sha256": tensor_sha256(payload["evidence"]),
        "transform_ids_sha256": tensor_sha256(payload["transform_ids"]),
        "targets_sha256": tensor_sha256(payload["targets"]),
        "target_source": "generator-side apply_transform(evidence, transform_id); never model payload",
    }


class OracleReader:
    """Diagnostic reader that exposes exactly one requested evidence vector."""

    @staticmethod
    def read(evidence: Tensor, round_index: int) -> Tensor:
        return evidence[:, round_index, :]


class C0OracleModel(nn.Module):
    """Frozen U0-A shell plus trainable transform-conditioned correction only."""

    def __init__(self, dimension: int = DIMENSION, *, train_residual_network: bool = False) -> None:
        super().__init__()
        self.frozen_base = UnifiedT1U0(dimension)
        for parameter in self.frozen_base.parameters():
            parameter.requires_grad_(False)
        self.frozen_base.eval()
        self.correction_mlp = TransformCorrectionMLP(dimension, transform_count=TRANSFORM_COUNT)
        if not train_residual_network:
            for parameter in self.correction_mlp.network.parameters():
                parameter.requires_grad_(False)

    def transition(self, payload: Tensor, workspace: Tensor, transform_id: Tensor) -> tuple[Tensor, Tensor]:
        correction = self.correction_mlp(payload, workspace, transform_id)
        delta = payload + correction
        return workspace + delta, delta


def run_sequence(model: C0OracleModel, evidence: Tensor, transform_ids: Tensor, lengths: Tensor, *, rounds: int = MAX_H) -> tuple[Tensor, Tensor, Tensor]:
    workspace = torch.zeros_like(evidence[:, 0, :])
    predicted_deltas = torch.zeros_like(evidence)
    states = [workspace]
    for round_index in range(rounds):
        payload = OracleReader.read(evidence, round_index)
        next_workspace, delta = model.transition(payload, workspace, transform_ids[:, round_index])
        active = (lengths > round_index).unsqueeze(-1)
        workspace = torch.where(active, next_workspace, workspace)
        predicted_deltas[:, round_index, :] = torch.where(active, delta, torch.zeros_like(delta))
        states.append(workspace)
    return workspace, predicted_deltas, torch.stack(states, dim=1)


def relative_error(predicted: Tensor, target: Tensor) -> Tensor:
    return torch.linalg.vector_norm(predicted - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-8)


@torch.no_grad()
def evaluate(model: C0OracleModel, payload: dict[str, Tensor]) -> dict[str, object]:
    evidence, transform_ids, lengths = payload["evidence"], payload["transform_ids"], payload["lengths"]
    targets, target_deltas = payload["targets"], payload["deltas"]
    output, predicted_deltas, states = run_sequence(model, evidence, transform_ids, lengths)
    final_cosine = torch.nn.functional.cosine_similarity(output, targets, dim=-1)
    final_error = relative_error(output, targets)
    by_h: dict[str, object] = {}
    by_h_final_transform: dict[str, object] = {}
    delta_by_h_round_transform: dict[str, object] = {}
    for h in H_VALUES:
        h_mask = lengths == h
        by_h[str(h)] = {
            "samples": int(h_mask.sum()),
            "cosine": float(final_cosine[h_mask].mean()),
            "normalized_error": float(final_error[h_mask].mean()),
        }
        by_h_final_transform[str(h)] = {}
        for transform_id, transform_name in enumerate(TRANSFORM_NAMES):
            selected = h_mask & (transform_ids[:, h - 1] == transform_id)
            by_h_final_transform[str(h)][transform_name] = {
                "samples": int(selected.sum()),
                "cosine": float(final_cosine[selected].mean()) if selected.any() else None,
                "normalized_error": float(final_error[selected].mean()) if selected.any() else None,
            }
        delta_by_h_round_transform[str(h)] = {}
        for round_index in range(h):
            delta_by_h_round_transform[str(h)][str(round_index + 1)] = {}
            for transform_id, transform_name in enumerate(TRANSFORM_NAMES):
                selected = h_mask & (transform_ids[:, round_index] == transform_id)
                epsilon = relative_error(predicted_deltas[:, round_index, :], target_deltas[:, round_index, :])
                delta_by_h_round_transform[str(h)][str(round_index + 1)][transform_name] = {
                    "samples": int(selected.sum()),
                    "epsilon_delta_relative": float(epsilon[selected].mean()) if selected.any() else None,
                    "epsilon_delta_absolute": float(torch.linalg.vector_norm((predicted_deltas - target_deltas)[:, round_index, :], dim=-1)[selected].mean()) if selected.any() else None,
                }
    truncation: dict[str, object] = {}
    for h in H_VALUES:
        h_mask = lengths == h
        truncation[str(h)] = {}
        for rounds in ROUNDS:
            if rounds >= h:
                continue
            truncated, _, _ = run_sequence(model, evidence[h_mask], transform_ids[h_mask], lengths[h_mask], rounds=rounds)
            truncation[str(h)][str(rounds)] = {
                "cosine_to_full_target": float(torch.nn.functional.cosine_similarity(truncated, targets[h_mask], dim=-1).mean()),
                "normalized_error_to_full_target": float(relative_error(truncated, targets[h_mask]).mean()),
            }
    zero_model = C0OracleModel(DIMENSION, train_residual_network=False)
    zero_model.correction_mlp.load_state_dict(model.correction_mlp.state_dict())
    with torch.no_grad():
        zero_model.correction_mlp.payload_basis.weight.zero_()
        zero_model.correction_mlp.payload_basis.bias.zero_()
        zero_model.correction_mlp.network[-1].weight.zero_()
        zero_model.correction_mlp.network[-1].bias.zero_()
    zero_output, _, _ = run_sequence(zero_model, evidence, transform_ids, lengths)
    zero_cosine = torch.nn.functional.cosine_similarity(zero_output, targets, dim=-1)
    zero_error = relative_error(zero_output, targets)
    zero_by_transform = {}
    for transform_id, transform_name in enumerate(TRANSFORM_NAMES):
        selected = transform_ids[torch.arange(len(lengths)), lengths - 1] == transform_id
        zero_by_transform[transform_name] = {
            "cosine": float(zero_cosine[selected].mean()),
            "normalized_error": float(zero_error[selected].mean()),
        }
    identity_ids = torch.zeros_like(transform_ids)
    identity_active = (torch.arange(MAX_H).view(1, -1) < lengths.view(-1, 1)).unsqueeze(-1)
    identity_targets = (evidence * identity_active).sum(dim=1)
    identity_output, _, _ = run_sequence(model, evidence, identity_ids, lengths)
    identity_zero_output, _, _ = run_sequence(zero_model, evidence, identity_ids, lengths)
    identity_cosine = torch.nn.functional.cosine_similarity(identity_output, identity_targets, dim=-1)
    identity_zero_cosine = torch.nn.functional.cosine_similarity(identity_zero_output, identity_targets, dim=-1)
    identity_error = relative_error(identity_output, identity_targets)
    identity_zero_error = relative_error(identity_zero_output, identity_targets)
    return {
        "final_by_h": by_h,
        "final_by_h_final_transform": by_h_final_transform,
        "epsilon_delta_by_h_round_transform": delta_by_h_round_transform,
        "truncation_by_target_h": truncation,
        "forced_correction_zero_by_final_transform": zero_by_transform,
        "identity_control": {
            "note": "Identity is structural control, not learning evidence.",
            "max_abs_correction": float(model.correction_mlp(torch.zeros(1, DIMENSION), torch.zeros(1, DIMENSION), torch.zeros(1, dtype=torch.long)).abs().max()),
            "all_identity_sequence": {
                "cosine": float(identity_cosine.mean()),
                "normalized_error": float(identity_error.mean()),
                "forced_correction_zero_cosine": float(identity_zero_cosine.mean()),
                "forced_correction_zero_normalized_error": float(identity_zero_error.mean()),
            },
        },
        "state_round_count": int(states.shape[1] - 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-network", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c0_oracle_seed101")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_payload = make_split(DATASET_SEEDS["train"])
    val_payload = make_split(DATASET_SEEDS["val"])
    test_payload = make_split(DATASET_SEEDS["test"])
    manifests = {name: dataset_manifest(payload, DATASET_SEEDS[name]) for name, payload in (("train", train_payload), ("val", val_payload), ("test", test_payload))}
    model = C0OracleModel(DIMENSION, train_residual_network=args.train_network)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected_prefixes = ("correction_mlp.",)
    if not trainable or not all(name.startswith(expected_prefixes) for name, _ in trainable):
        raise AssertionError(f"unexpected trainable parameters: {[name for name, _ in trainable]}")
    if any(parameter.requires_grad for parameter in model.frozen_base.parameters()):
        raise AssertionError("frozen U0-A base has trainable parameters")
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.lr, weight_decay=0.0)
    dataset = TensorDataset(train_payload["evidence"], train_payload["transform_ids"], train_payload["lengths"], train_payload["deltas"], train_payload["targets"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    iterator = iter(loader)
    metrics: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        try:
            evidence, transform_ids, lengths, target_deltas, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            evidence, transform_ids, lengths, target_deltas, targets = next(iterator)
        workspace = torch.zeros((evidence.shape[0], DIMENSION))
        predicted_deltas = []
        for round_index in range(MAX_H):
            payload = OracleReader.read(evidence, round_index)
            next_workspace, delta = model.transition(payload, workspace, transform_ids[:, round_index])
            active = (lengths > round_index).unsqueeze(-1)
            workspace = torch.where(active, next_workspace, workspace)
            predicted_deltas.append(torch.where(active, delta, torch.zeros_like(delta)))
        predicted_delta_tensor = torch.stack(predicted_deltas, dim=1)
        active = (torch.arange(MAX_H).view(1, -1) < lengths.view(-1, 1)).unsqueeze(-1)
        delta_loss = ((predicted_delta_tensor - target_deltas).square() * active).sum() / active.sum().clamp_min(1)
        final_loss = (workspace - targets).square().mean()
        loss = delta_loss + 0.25 * final_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0))
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            metrics.append({"step": step, "loss": float(loss.detach()), "delta_loss": float(delta_loss.detach()), "final_loss": float(final_loss.detach()), "grad_norm": grad_norm})
    config = {
        "phase": "T1-U0-C0 oracle",
        "seed": SEED,
        "dimension": DIMENSION,
        "h_values": H_VALUES,
        "rounds": ROUNDS,
        "transform_names": TRANSFORM_NAMES,
        "reader": "diagnostic oracle; each round exposes exactly evidence[:, round_index, :] and no other memory",
        "target": "generator-side transformed evidence sum; independent of model payload",
        "transition": "W_next = W + Y + correction_mlp(Y, RMSNorm(W), transform_embedding)",
        "head": "Identity(W)",
        "frozen_base": "UnifiedT1U0 reader/core/ALU adapters/codebooks/decoders/existing embeddings requires_grad=False; base not in correction forward",
        "trainable": "correction_mlp payload basis + new transform embedding only" if args.train_network else "correction_mlp payload basis + new transform embedding only; residual MLP branch remains zero/frozen in C0",
        "residual_network_trainable": args.train_network,
        "optimizer": f"AdamW(lr={args.lr:g}, weight_decay=0), one step per batch",
        "steps": args.steps,
        "sample_count_per_h": SAMPLES_PER_H,
        "dataset_manifests": manifests,
    }
    save_json(args.output_dir / "config.json", config)
    save_json(args.output_dir / "training_metrics.json", metrics)
    torch.save({"config": config, "model": model.state_dict()}, args.output_dir / "final.pt")
    results = evaluate(model, test_payload)
    final = {"status": "completed", "finite": True, "seed": SEED, "config": config, "training_metrics": metrics, "results": results, "elapsed_seconds": time.perf_counter() - started}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
