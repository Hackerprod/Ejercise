"""Read-only C1 workspace drift audit for existing checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import SLOT_W  # noqa: E402
from train_u0a import BATCH_SIZE, ExampleDataset, build_canonical_data, collate, immediate_vectors, materialize, run_rounds  # noqa: E402
import train_u0c_c1_joint as c1  # noqa: E402


CHECKPOINTS = {
    "step0": ROOT / "campaign" / "u0c_c1_joint_seed101" / "step0.pt",
    "best": ROOT / "campaign" / "u0c_c1_joint_seed101" / "best.pt",
    "final": ROOT / "campaign" / "u0c_c1_joint_seed101" / "final.pt",
}
AUDIT_DIR = ROOT / "campaign" / "u0c_c1_workspace_audit_seed101"


def load_model(path: Path) -> c1.C1JointModel:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = c1.C1JointModel()
    model.load_state_dict(state["model"], strict=True)
    return model


@torch.no_grad()
def summarize_trace(model: c1.C1JointModel, examples: list[object]) -> dict[str, object]:
    model.eval()
    loader = DataLoader(ExampleDataset(examples), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    metrics: dict[str, dict[str, list[float]]] = {str(h): {key: [] for key in ("observed_error_norm", "reconstructed_error_norm", "reconstruction_delta_norm", "c_mass_deviation_norm")} for h in (2, 4, 6)}
    samples: dict[str, dict[str, object]] = {}
    for batch in loader:
        data = materialize(model, batch)
        state = data["state"]
        attention = torch.zeros((state.shape[0], data["opcodes"].shape[1], data["memory_keys"].shape[1]))
        for round_index in range(data["opcodes"].shape[1]):
            state, _, read_result = model.step(
                state,
                data["memory_keys"],
                data["memory_values"],
                data["memory_types"],
                data["row_mask"],
                data["opcodes"][:, round_index],
                immediate_vectors(model, data["immediates"][:, round_index]),
                data["source_slots"][:, round_index],
                data["destination_slots"][:, round_index],
                data["presence"],
                read_mode=data["read_modes"][:, round_index],
            )
            attention[:, round_index] = read_result.attention_soft
        active = (torch.arange(attention.shape[1]).view(1, -1) < data["hops"].view(-1, 1)).to(dtype=attention.dtype)
        contributions = (attention * active.unsqueeze(-1)).sum(dim=1)
        reconstructed = torch.einsum("bm,bmd->bd", contributions, data["memory_values"])
        observed = state[:, SLOT_W, :]
        target = data["target_vectors"]
        legal_rows = data["row_mask"].to(dtype=contributions.dtype)
        formula_error = torch.einsum("bm,bmd->bd", contributions - legal_rows, data["memory_values"])
        for h in (2, 4, 6):
            selected = data["hops"] == h
            if not selected.any():
                continue
            observed_error = observed[selected] - target[selected]
            reconstructed_error = reconstructed[selected] - target[selected]
            reconstruction_delta = reconstructed[selected] - observed[selected]
            c_deviation = contributions[selected] - legal_rows[selected]
            formula_error_selected = formula_error[selected]
            metrics[str(h)]["observed_error_norm"].extend(torch.linalg.vector_norm(observed_error, dim=-1).tolist())
            metrics[str(h)]["reconstructed_error_norm"].extend(torch.linalg.vector_norm(reconstructed_error, dim=-1).tolist())
            metrics[str(h)]["reconstruction_delta_norm"].extend(torch.linalg.vector_norm(reconstruction_delta, dim=-1).tolist())
            metrics[str(h)].setdefault("formula_delta_norm", []).extend(torch.linalg.vector_norm(formula_error_selected - observed_error, dim=-1).tolist())
            metrics[str(h)]["c_mass_deviation_norm"].extend(torch.linalg.vector_norm(c_deviation, dim=-1).tolist())
            if str(h) not in samples:
                first = selected.nonzero(as_tuple=False)[0, 0]
                samples[str(h)] = {
                    "attention_pi_by_round": attention[first, :h].tolist(),
                    "contribution_c": contributions[first].tolist(),
                    "observed_error_norm": float(torch.linalg.vector_norm(observed[first] - target[first])),
                    "reconstructed_error_norm": float(torch.linalg.vector_norm(reconstructed[first] - target[first])),
                }
    summary = {}
    for h, values in metrics.items():
        summary[h] = {
            "samples": len(values["observed_error_norm"]),
            "observed_error_norm_mean": sum(values["observed_error_norm"]) / len(values["observed_error_norm"]),
            "reconstructed_error_norm_mean": sum(values["reconstructed_error_norm"]) / len(values["reconstructed_error_norm"]),
            "reconstruction_delta_norm_max": max(values["reconstruction_delta_norm"]),
            "formula_vs_observed_error_norm_max": max(values["formula_delta_norm"]),
            "c_mass_deviation_norm_mean": sum(values["c_mass_deviation_norm"]) / len(values["c_mass_deviation_norm"]),
        }
    return {"by_h": summary, "sample_traces": samples}


def zero_candidate_and_corrector_calls(model: c1.C1JointModel, examples: list[object]) -> dict[str, object]:
    candidate_zero = all(torch.count_nonzero(parameter.detach()) == 0 for parameter in model.core.workspace_correction.parameters())
    calls = {"count": 0}

    def count_call(*_: object) -> None:
        calls["count"] += 1

    hook = model.correction_mlp.register_forward_hook(count_call)
    summarize_trace(model, examples)
    hook.remove()
    return {"old_workspace_candidate_zero": candidate_zero, "old_workspace_candidate_requires_grad": [parameter.requires_grad for parameter in model.core.workspace_correction.parameters()], "new_corrector_call_count_raw": calls["count"]}


def restore_reader_and_codebook(final_model: c1.C1JointModel, step0_path: Path) -> c1.C1JointModel:
    restored = copy.deepcopy(final_model)
    step0 = torch.load(step0_path, map_location="cpu", weights_only=False)["model"]
    current = restored.state_dict()
    for name in current:
        if name.startswith("memory_reader.") or name == "token_embedding.weight":
            current[name].copy_(step0[name])
    restored.load_state_dict(current, strict=True)
    return restored


@torch.no_grad()
def workspace_errors(model: c1.C1JointModel, examples: list[object]) -> dict[str, float]:
    loader = DataLoader(ExampleDataset(examples), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    values: dict[str, list[float]] = {str(h): [] for h in (2, 4, 6)}
    model.eval()
    for batch in loader:
        data = materialize(model, batch)
        state = run_rounds(model, batch, batch["opcodes"].shape[1])
        error = torch.linalg.vector_norm(state[:, SLOT_W] - data["target_vectors"], dim=-1) / torch.linalg.vector_norm(data["target_vectors"], dim=-1).clamp_min(1e-8)
        for h in (2, 4, 6):
            selected = data["hops"] == h
            values[str(h)].extend(error[selected].tolist())
    return {h: sum(items) / len(items) for h, items in values.items()}


def gradient_vector(model: c1.C1JointModel, prefixes: tuple[str, ...]) -> Tensor:
    chunks = [parameter.grad.detach().reshape(-1) if parameter.grad is not None else torch.zeros_like(parameter).reshape(-1) for name, parameter in model.named_parameters() if name.startswith(prefixes)]
    return torch.cat(chunks) if chunks else torch.zeros(1)


def fixed_gradient_audit(model: c1.C1JointModel, raw_batch: dict[str, Tensor], transformed_batch: dict[str, Tensor]) -> dict[str, object]:
    model.train()
    raw_state = run_rounds(model, raw_batch, raw_batch["opcodes"].shape[1])
    raw_loss = (raw_state[:, SLOT_W, :] - raw_batch["target_vectors"]).square().mean()
    model.zero_grad(set_to_none=True)
    raw_loss.backward()
    raw_reader = gradient_vector(model, ("memory_reader.",))
    raw_token = gradient_vector(model, ("token_embedding",))

    def transform_loss(delta_scale: float) -> Tensor:
        output, predicted, _, _, targets = c1.run_transform_batch(model, transformed_batch)
        active = (torch.arange(c1.MAX_H).view(1, -1) < transformed_batch["lengths"].view(-1, 1)).unsqueeze(-1)
        delta_loss = (((predicted - transformed_batch["target_deltas"]) * active) ** 2).sum() / active.sum().clamp_min(1)
        final_loss = (output - targets).square().mean()
        return delta_loss * delta_scale + 0.25 * final_loss

    model.zero_grad(set_to_none=True)
    transform_actual_loss = transform_loss(1.0)
    transform_actual_loss.backward()
    transform_reader = gradient_vector(model, ("memory_reader.",))
    transform_token = gradient_vector(model, ("token_embedding",))
    model.zero_grad(set_to_none=True)
    transform_no64_loss = transform_loss(1.0 / c1.DIMENSION)
    transform_no64_loss.backward()
    transform_no64_reader = gradient_vector(model, ("memory_reader.",))
    transform_no64_token = gradient_vector(model, ("token_embedding",))

    def norm(value: Tensor) -> float:
        return float(value.norm())

    def cosine(left: Tensor, right: Tensor) -> float | None:
        if left.norm() == 0 or right.norm() == 0:
            return None
        return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())

    return {
        "raw_loss": float(raw_loss.detach()),
        "transformed_loss_actual": float(transform_actual_loss.detach()),
        "transformed_loss_delta_divided_by_64": float(transform_no64_loss.detach()),
        "reader": {"raw_norm": norm(raw_reader), "transformed_norm": norm(transform_reader), "transformed_norm_no64": norm(transform_no64_reader), "cosine_raw_vs_transformed": cosine(raw_reader, transform_reader), "cosine_raw_vs_transformed_no64": cosine(raw_reader, transform_no64_reader)},
        "token_embedding": {"raw_norm": norm(raw_token), "transformed_norm": norm(transform_token), "transformed_norm_no64": norm(transform_no64_token), "cosine_raw_vs_transformed": cosine(raw_token, transform_token), "cosine_raw_vs_transformed_no64": cosine(raw_token, transform_no64_token)},
    }


def main() -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = build_canonical_data(AUDIT_DIR)
    raw_batch = collate(datasets["workspace_accumulation"]["train"][:BATCH_SIZE])
    transformed = c1.make_transform_split(c1.TRAIN_TRANSFORM_SEED)
    transformed_batch = {key: value[:BATCH_SIZE] for key, value in transformed.items() if key != "key_ids_sha256"}
    output: dict[str, object] = {"status": "completed", "checkpoints": {}}
    for label, path in CHECKPOINTS.items():
        model = load_model(path)
        raw_trace = summarize_trace(model, datasets["workspace_accumulation"]["test"])
        output["checkpoints"][label] = {"raw_attention_reconstruction": raw_trace, "raw_route_invariants": zero_candidate_and_corrector_calls(model, datasets["workspace_accumulation"]["test"]), "workspace_blend_errors": workspace_errors(model, datasets["workspace_accumulation"]["test"]), "fixed_gradient_audit": fixed_gradient_audit(model, raw_batch, transformed_batch)}
    final_model = load_model(CHECKPOINTS["final"])
    restored = restore_reader_and_codebook(final_model, CHECKPOINTS["step0"])
    output["final_reader_codebook_restoration"] = {"restored_components": ["memory_reader", "token_embedding.weight"], "workspace_blend_errors": workspace_errors(restored, datasets["workspace_accumulation"]["test"])}
    (AUDIT_DIR / "final.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
