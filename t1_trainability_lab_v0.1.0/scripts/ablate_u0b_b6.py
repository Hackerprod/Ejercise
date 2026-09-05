"""Run U0-B6 workspace replacement ablation without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablate_u0b_b1 import summary  # noqa: E402
from train_u0a import (  # noqa: E402
    build_canonical_data,
    build_sequential_h1_table,
    collate,
    evaluate_accuracy,
    evaluate_all,
    immediate_vectors,
    materialize,
    run_rounds,
)
from t1_trainability.unified import OPCODE_IDS, SLOT_W, TypedCommit, UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)
HOPS = (2, 4, 6)
ROUNDS = (1, 2, 4, 6)


class B6WorkspaceReplaceCommit(TypedCommit):
    """Runtime-only B6 variant; replace W with current reader payload Y."""

    def forward(self, state, candidates, read_result, opcode, destination_slot, presence_mask, register_codebook=None, alu_logits=None):  # type: ignore[no-untyped-def]
        next_state = super().forward(
            state,
            candidates,
            read_result,
            opcode,
            destination_slot,
            presence_mask,
            register_codebook=register_codebook,
            alu_logits=alu_logits,
        )
        active = (opcode == OPCODE_IDS["ACCUM_W"]) & (destination_slot == SLOT_W)
        workspace = torch.where(active.unsqueeze(-1), read_result.payload, next_state[:, SLOT_W, :])
        return torch.cat((next_state[:, :SLOT_W, :], workspace.unsqueeze(1), next_state[:, SLOT_W + 1 :, :]), dim=1)


def load_model(checkpoint: Path, ablated: bool) -> UnifiedT1U0:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    if ablated:
        model.commit = B6WorkspaceReplaceCommit(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_workspace_cosine(model: UnifiedT1U0, examples: list[object]) -> dict[str, dict[str, float]]:
    hits = {str(hop): {str(rounds): [] for rounds in ROUNDS} for hop in HOPS}
    for offset in range(0, len(examples), 256):
        batch = collate(examples[offset : offset + 256])
        for rounds in ROUNDS:
            state = run_rounds(model, batch, rounds)
            cosine = F.cosine_similarity(state[:, SLOT_W, :], batch["target_vectors"], dim=-1)
            for hop in HOPS:
                selected = batch["hops"] == hop
                hits[str(hop)][str(rounds)].extend(cosine[selected].tolist())
    return {hop: {rounds: sum(values) / len(values) for rounds, values in rounds_map.items()} for hop, rounds_map in hits.items()}


@torch.no_grad()
def evaluate_workspace_payload_diagnostics(model: UnifiedT1U0, examples: list[object]) -> dict[str, dict[str, float]]:
    payload_cosines = {str(hop): [] for hop in HOPS}
    raw_cosines = {str(hop): [] for hop in HOPS}
    effective_components = {str(hop): [] for hop in HOPS}
    state_payload_deltas = {str(hop): [] for hop in HOPS}
    for offset in range(0, len(examples), 256):
        batch = collate(examples[offset : offset + 256])
        data = materialize(model, batch)
        state = data["state"]
        for round_index in range(6):
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
            )
            active = batch["hops"] == round_index + 1
            for index in active.nonzero(as_tuple=False).flatten().tolist():
                hop = str(int(batch["hops"][index]))
                payload_cosines[hop].append(float(F.cosine_similarity(read_result.payload[index], batch["target_vectors"][index], dim=0)))
                raw_cosines[hop].append(float(F.cosine_similarity(data["raw_values"][index, round_index], batch["target_vectors"][index], dim=0)))
                state_payload_deltas[hop].append(float((state[index, SLOT_W] - read_result.payload[index]).abs().max()))
                weights = read_result.attention[index][batch["row_mask"][index]]
                effective_components[hop].append(float(1.0 / weights.square().sum().clamp_min(1e-12)))
    return {
        "payload_cosine_to_full_sum": {hop: sum(values) / len(values) for hop, values in payload_cosines.items()},
        "raw_current_vector_cosine_to_full_sum": {hop: sum(values) / len(values) for hop, values in raw_cosines.items()},
        "attention_effective_component_count": {hop: sum(values) / len(values) for hop, values in effective_components.items()},
        "replacement_state_payload_max_abs_delta": {hop: max(values) for hop, values in state_payload_deltas.items()},
    }


def b6_summary(metrics: dict[str, object], cosine: dict[str, dict[str, float]], payload_diagnostics: dict[str, dict[str, float]]) -> dict[str, object]:
    output = summary(metrics)
    output["workspace_cosine"] = cosine
    output["workspace_final_cosine"] = {hop: cosine[hop]["6"] for hop in cosine}
    output["workspace_payload_diagnostics"] = payload_diagnostics
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "u0b_b6_workspace_replace.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    runs: dict[str, object] = {}
    for seed in args.seeds:
        run_dir = ROOT / "campaign" / f"u0a_iso_clean_seed{seed}_12000"
        checkpoint = run_dir / "best.pt"
        datasets = build_canonical_data(run_dir)
        baseline_model = load_model(checkpoint, ablated=False)
        ablated_model = load_model(checkpoint, ablated=True)
        baseline_metrics = evaluate_all(baseline_model, datasets, "test")
        ablated_metrics = evaluate_all(ablated_model, datasets, "test")
        baseline_metrics["sequential_update_h1_table"] = evaluate_accuracy(
            baseline_model,
            "sequential_update",
            build_sequential_h1_table(),
            rounds=1,
        )
        ablated_metrics["sequential_update_h1_table"] = evaluate_accuracy(
            ablated_model,
            "sequential_update",
            build_sequential_h1_table(),
            rounds=1,
        )
        workspace_test = datasets["workspace_accumulation"]["test"]
        baseline_cosine = evaluate_workspace_cosine(baseline_model, workspace_test)
        ablated_cosine = evaluate_workspace_cosine(ablated_model, workspace_test)
        baseline_payload_diagnostics = evaluate_workspace_payload_diagnostics(baseline_model, workspace_test)
        ablated_payload_diagnostics = evaluate_workspace_payload_diagnostics(ablated_model, workspace_test)
        baseline = b6_summary(baseline_metrics, baseline_cosine, baseline_payload_diagnostics)
        ablation = b6_summary(ablated_metrics, ablated_cosine, ablated_payload_diagnostics)
        runs[str(seed)] = {
            "training_performed": False,
            "optimizer_steps": 0,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "baseline": baseline,
            "ablation": ablation,
            "delta_ablation_minus_baseline": {
                "pointer_final_h4": ablation["pointer_final_h4"] - baseline["pointer_final_h4"],
                "multi_hop_final_h3": ablation["multi_hop_final_h3"] - baseline["multi_hop_final_h3"],
                "associative_final": ablation["associative_final"] - baseline["associative_final"],
                "workspace_h6_error": ablation["workspace_h6_error"] - baseline["workspace_h6_error"],
                "workspace_cosine_h4": ablation["workspace_final_cosine"]["4"] - baseline["workspace_final_cosine"]["4"],
                "workspace_cosine_h6": ablation["workspace_final_cosine"]["6"] - baseline["workspace_final_cosine"]["6"],
            },
        }

    result = {
        "phase": "T1-U0-B6",
        "ablation": "workspace residual-to-replacement: W_next=Y instead of W+Y",
        "implementation_note": "Only TypedCommit ACCUM_W writes to W are replaced with current reader payload; all other opcodes and paths remain unchanged.",
        "training_performed": False,
        "seeds": list(args.seeds),
        "theoretical_final_cosine": {"2": 1 / (2**0.5), "4": 1 / (4**0.5), "6": 1 / (6**0.5)},
        "theoretical_check_note": "The 1/sqrt(H) reference applies to one independent raw Gaussian payload; report reader payload mixing separately.",
        "theoretical_check_status": "REVIEW_REQUIRED: actual reader payload cosine is not consistent with one independent raw Gaussian vector across seeds.",
        "observed_final_cosine_range": {
            "4": [
                min(runs[seed]["ablation"]["workspace_final_cosine"]["4"] for seed in runs),
                max(runs[seed]["ablation"]["workspace_final_cosine"]["4"] for seed in runs),
            ],
            "6": [
                min(runs[seed]["ablation"]["workspace_final_cosine"]["6"] for seed in runs),
                max(runs[seed]["ablation"]["workspace_final_cosine"]["6"] for seed in runs),
            ],
        },
        "theoretical_observed_final_cosine": {
            seed: {
                "4": runs[seed]["ablation"]["workspace_final_cosine"]["4"],
                "6": runs[seed]["ablation"]["workspace_final_cosine"]["6"],
            }
            for seed in runs
        },
        "runs": runs,
        "expected_signature": {
            "workspace_cosine": "approaches 1/sqrt(H) for independent Gaussian payloads: H4~0.50, H6~0.408",
            "workspace_error": "increases under replacement",
            "unrelated": "pointer, associative, ALU, and retrieval tasks unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
