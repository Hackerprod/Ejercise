"""Evaluate C0-reader oracle/real-reader combinations without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    ROW_VEC,
    SLOT_COUNT,
    SLOT_P,
    UnifiedT1U0,
)
from train_u0c_c0_oracle import (  # noqa: E402
    C0OracleModel,
    DIMENSION,
    H_VALUES,
    MAX_H,
    SEED,
    TRANSFORM_NAMES,
    apply_transform,
    make_split,
    relative_error,
    run_sequence,
)


READER_CHECKPOINT = ROOT / "campaign" / "u0a_iso_clean_seed101_12000" / "best.pt"
C0_CHECKPOINT = ROOT / "campaign" / "u0c_c0_oracle_seed101_network_trainable" / "final.pt"
DATASET_SEED = 40404
MEMORY_ROW_IDS = tuple(range(10, 16))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def external_targets(payload: Tensor, transform_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    deltas = torch.zeros_like(payload)
    for round_index in range(MAX_H):
        deltas[:, round_index, :] = apply_transform(payload[:, round_index, :], transform_ids[:, round_index])
    active = (torch.arange(MAX_H).view(1, -1) < lengths.view(-1, 1)).unsqueeze(-1)
    return deltas, (deltas * active).sum(dim=1)


def transform_exact(payload: Tensor, transform_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    deltas, _ = external_targets(payload, transform_ids, lengths)
    workspace = torch.zeros_like(payload[:, 0, :])
    states = [workspace]
    for round_index in range(MAX_H):
        active = (lengths > round_index).unsqueeze(-1)
        workspace = torch.where(active, workspace + deltas[:, round_index, :], workspace)
        states.append(workspace)
    return workspace, torch.stack(states, dim=1)


def summarize(name: str, output: Tensor, predicted_deltas: Tensor, target: Tensor, target_deltas: Tensor, transform_ids: Tensor, lengths: Tensor) -> dict[str, object]:
    cosine = torch.nn.functional.cosine_similarity(output, target, dim=-1)
    error = relative_error(output, target)
    delta_error = relative_error(predicted_deltas, target_deltas)
    final_by_h: dict[str, object] = {}
    final_by_h_transform: dict[str, object] = {}
    epsilon_by_h_round_transform: dict[str, object] = {}
    for h in H_VALUES:
        h_mask = lengths == h
        final_by_h[str(h)] = {
            "samples": int(h_mask.sum()),
            "cosine": float(cosine[h_mask].mean()),
            "normalized_error": float(error[h_mask].mean()),
        }
        final_by_h_transform[str(h)] = {}
        epsilon_by_h_round_transform[str(h)] = {}
        for transform_id, transform_name in enumerate(TRANSFORM_NAMES):
            selected_final = h_mask & (transform_ids[:, h - 1] == transform_id)
            final_by_h_transform[str(h)][transform_name] = {
                "samples": int(selected_final.sum()),
                "cosine": float(cosine[selected_final].mean()) if selected_final.any() else None,
                "normalized_error": float(error[selected_final].mean()) if selected_final.any() else None,
            }
        for round_index in range(h):
            epsilon_by_h_round_transform[str(h)][str(round_index + 1)] = {}
            for transform_id, transform_name in enumerate(TRANSFORM_NAMES):
                selected = h_mask & (transform_ids[:, round_index] == transform_id)
                epsilon_by_h_round_transform[str(h)][str(round_index + 1)][transform_name] = {
                    "samples": int(selected.sum()),
                    "epsilon_delta_relative": float(delta_error[selected, round_index].mean()) if selected.any() else None,
                }
    final_metrics = [entry[transform]["normalized_error"] for entry in final_by_h_transform.values() for transform in entry if entry[transform]["normalized_error"] is not None]
    final_cosines = [entry[transform]["cosine"] for entry in final_by_h_transform.values() for transform in entry if entry[transform]["cosine"] is not None]
    epsilon_metrics = [entry[transform]["epsilon_delta_relative"] for rounds in epsilon_by_h_round_transform.values() for entry in rounds.values() for transform in entry if entry[transform]["epsilon_delta_relative"] is not None]
    return {
        "name": name,
        "final_by_h": final_by_h,
        "final_by_h_final_transform": final_by_h_transform,
        "epsilon_delta_by_h_round_transform": epsilon_by_h_round_transform,
        "gate": {
            "min_final_cosine": min(final_cosines),
            "max_final_normalized_error": max(final_metrics),
            "max_epsilon_delta_relative": max(epsilon_metrics),
            "passes_final_gate": min(final_cosines) > 0.999 and max(final_metrics) <= 0.01,
            "passes_delta_gate": max(epsilon_metrics) <= 0.01,
        },
    }


def load_reader() -> UnifiedT1U0:
    model = UnifiedT1U0(DIMENSION)
    checkpoint = torch.load(READER_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def audit_reader_provenance(reader: UnifiedT1U0) -> dict[str, object]:
    checkpoint = torch.load(READER_CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["model"]
    loaded_state = reader.state_dict()
    names = [
        name for name in checkpoint_state
        if name.startswith("memory_reader.")
        or "embedding" in name
        or name == "slot_type_embeddings"
        or name.startswith("commit.register_codebook")
    ]
    mismatches: list[str] = []
    max_abs_diff = 0.0
    for name in names:
        if name not in loaded_state or not torch.equal(loaded_state[name], checkpoint_state[name]):
            mismatches.append(name)
            if name in loaded_state:
                max_abs_diff = max(max_abs_diff, float((loaded_state[name] - checkpoint_state[name]).abs().max()))
    return {
        "checkpoint": str(READER_CHECKPOINT.relative_to(ROOT)),
        "strict_load": True,
        "compared_tensor_count": len(names),
        "compared_tensor_names": names,
        "all_equal": not mismatches,
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches,
        "compared_groups": ["memory_reader.query", "memory_reader.input_norm", "memory_reader.condition_projection", "all existing embeddings", "token_embedding/codebook"],
    }


def load_c0() -> C0OracleModel:
    model = C0OracleModel(DIMENSION, train_residual_network=True)
    checkpoint = torch.load(C0_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def make_real_reader_payload(reader: UnifiedT1U0, evidence: Tensor, transform_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
    samples = evidence.shape[0]
    row_count = len(MEMORY_ROW_IDS)
    row_ids = torch.tensor(MEMORY_ROW_IDS, dtype=torch.long)
    memory_keys = reader.token_embedding(row_ids).unsqueeze(0).expand(samples, -1, -1)
    memory_values = torch.randn((samples, row_count, DIMENSION), generator=torch.Generator().manual_seed(DATASET_SEED + 17))
    # Cycle through distinct rows within each trajectory so externally computed
    # transformed targets cannot cancel trivially from repeated evidence.
    query_indices = (torch.arange(MAX_H).view(1, -1) + torch.arange(samples).view(-1, 1)) % row_count
    oracle_payload = memory_values.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, DIMENSION))
    # Keep target evidence independent from reader output, while using same
    # frozen-reader inputs for every condition.
    evidence.copy_(oracle_payload)
    memory_types = torch.full((samples, row_count), ROW_VEC, dtype=torch.long)
    row_mask = torch.ones((samples, row_count), dtype=torch.bool)
    real_payload = torch.zeros_like(evidence)
    hard_payload = torch.zeros_like(evidence)
    attention_max = torch.zeros((samples, MAX_H))
    attention_target = torch.zeros((samples, MAX_H))
    top1_accuracy = torch.zeros((samples, MAX_H), dtype=torch.bool)
    for round_index in range(MAX_H):
        state = torch.zeros(samples, SLOT_COUNT, DIMENSION)
        state[:, SLOT_P, :] = memory_keys[torch.arange(samples), query_indices[:, round_index]]
        result = reader.memory_reader(
            state,
            memory_keys,
            memory_values,
            memory_types,
            row_mask,
            torch.full((samples,), OPCODE_IDS["ACCUM_W"], dtype=torch.long),
            torch.full((samples,), 511, dtype=torch.long),
            torch.full((samples,), SLOT_P, dtype=torch.long),
        )
        real_payload[:, round_index, :] = result.payload
        selected = result.attention.argmax(dim=-1)
        hard_payload[:, round_index, :] = memory_values[torch.arange(samples), selected]
        attention_max[:, round_index] = result.attention.max(dim=-1).values
        attention_target[:, round_index] = result.attention.gather(1, query_indices[:, round_index].unsqueeze(-1)).squeeze(-1)
        top1_accuracy[:, round_index] = selected == query_indices[:, round_index]
    active = torch.arange(MAX_H).view(1, -1) < lengths.view(-1, 1)
    reader_cosine = torch.nn.functional.cosine_similarity(real_payload, evidence, dim=-1)
    diagnostics = {
        "memory_row_ids": MEMORY_ROW_IDS,
        "reader_payload_cosine_by_h": {
            str(h): float(reader_cosine[lengths == h, :h].mean()) for h in H_VALUES
        },
        "attention_max_by_h": {str(h): float(attention_max[lengths == h, :h].mean()) for h in H_VALUES},
        "attention_target_by_h": {str(h): float(attention_target[lengths == h, :h].mean()) for h in H_VALUES},
        "top1_accuracy_by_h_round": {
            str(h): {str(round_index + 1): float(top1_accuracy[lengths == h, round_index].float().mean()) for round_index in range(h)} for h in H_VALUES
        },
        "top1_accuracy_over_active_rounds": float(top1_accuracy[active].float().mean()),
        "payload_cosine_min": float(reader_cosine[active].min()),
        "payload_cosine_max": float(reader_cosine[active].max()),
    }
    return real_payload, hard_payload, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c0_reader_seed101")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = make_split(DATASET_SEED)
    evidence = split["evidence"].clone()
    transform_ids, lengths = split["transform_ids"], split["lengths"]
    reader = load_reader()
    c0_model = load_c0()
    provenance = audit_reader_provenance(reader)
    real_payload, hard_payload, reader_diagnostics = make_real_reader_payload(reader, evidence, transform_ids, lengths)
    oracle_deltas, oracle_target = external_targets(evidence, transform_ids, lengths)
    real_deltas, real_target = external_targets(real_payload, transform_ids, lengths)

    oracle_exact_output, _ = transform_exact(evidence, transform_ids, lengths)
    oracle_learned_output, oracle_learned_predicted, _ = run_sequence(c0_model, evidence, transform_ids, lengths)
    real_exact_output, _ = transform_exact(real_payload, transform_ids, lengths)
    real_learned_output, real_learned_predicted, _ = run_sequence(c0_model, real_payload, transform_ids, lengths)
    hard_exact_output, _ = transform_exact(hard_payload, transform_ids, lengths)
    hard_learned_output, hard_learned_predicted, _ = run_sequence(c0_model, hard_payload, transform_ids, lengths)

    results = {
        "oracle_exact_evaluator": summarize("oracle + exact transform evaluator", oracle_exact_output, oracle_deltas, oracle_target, oracle_deltas, transform_ids, lengths),
        "oracle_learned_c0": summarize("oracle + learned correction (official C0-oracle)", oracle_learned_output, oracle_learned_predicted, oracle_target, oracle_deltas, transform_ids, lengths),
        "real_reader_exact_evaluator": summarize("real reader + exact transform evaluator", real_exact_output, real_deltas, oracle_target, oracle_deltas, transform_ids, lengths),
        "real_reader_learned_c0": summarize("real reader + learned correction", real_learned_output, real_learned_predicted, oracle_target, oracle_deltas, transform_ids, lengths),
        "real_reader_top1_exact_evaluator": summarize("real reader top1 + exact transform evaluator", hard_exact_output, external_targets(hard_payload, transform_ids, lengths)[0], oracle_target, oracle_deltas, transform_ids, lengths),
        "real_reader_top1_learned_c0": summarize("real reader top1 + learned correction", hard_learned_output, hard_learned_predicted, oracle_target, oracle_deltas, transform_ids, lengths),
    }
    real_exact_gate = results["real_reader_exact_evaluator"]["gate"]
    result = {
        "status": "completed",
        "seed": SEED,
        "dataset_seed": DATASET_SEED,
        "target": "external generator target from raw memory row values and transform sequence; never from model payload",
        "reader_checkpoint": str(READER_CHECKPOINT.relative_to(ROOT)),
        "c0_checkpoint": str(C0_CHECKPOINT.relative_to(ROOT)),
        "reader": "frozen U0-A SharedMemoryReader; one query/read per round; no correction training in C0-reader",
        "transform_names": TRANSFORM_NAMES,
        "h_values": H_VALUES,
        "reader_diagnostics": reader_diagnostics,
        "reader_provenance": provenance,
        "conditions": results,
        "training_stop_decision": {
            "real_reader_exact_condition_fails_gate": not bool(real_exact_gate["passes_final_gate"]),
            "do_not_train_more_correction": not bool(real_exact_gate["passes_final_gate"]),
            "reason": "Condition 3 isolates reader error; additional correction steps must not be used to reconstruct mixed/missed evidence." if not real_exact_gate["passes_final_gate"] else "Condition 3 passes; no extra correction training was performed.",
        },
    }
    save_json(args.output_dir / "final.json", result)
    save_json(args.output_dir / "config.json", {
        "phase": "T1-U0-C0-reader",
        "seed": SEED,
        "dataset_seed": DATASET_SEED,
        "reader_checkpoint": str(READER_CHECKPOINT.relative_to(ROOT)),
        "c0_checkpoint": str(C0_CHECKPOINT.relative_to(ROOT)),
        "conditions": list(results),
        "reader": "real frozen U0-A reader; no retraining",
        "target": "external raw memory value transformed by generator-only A_tau",
    })
    save_json(args.output_dir / "reader_audit.json", {
        "reader_provenance": provenance,
        "top1_accuracy_by_h_round": reader_diagnostics["top1_accuracy_by_h_round"],
        "top1_accuracy_over_active_rounds": reader_diagnostics["top1_accuracy_over_active_rounds"],
        "soft_vs_top1_gates": {
            key: results[key]["gate"] for key in (
                "real_reader_exact_evaluator",
                "real_reader_learned_c0",
                "real_reader_top1_exact_evaluator",
                "real_reader_top1_learned_c0",
            )
        },
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
