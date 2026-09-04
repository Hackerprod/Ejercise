"""Aggregate completed T1 runs and compute post-hoc gate diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability import InputAdapter, OutputReader, RecurrentCore, TokenVocabulary  # noqa: E402
from t1_trainability.data import OUTPUT_CARDINALITIES, encode_batch, load_jsonl  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
TASKS = ("associative_recall", "multi_hop", "variable_binding", "sequential_update", "length_generalization")


def run_path(root: Path, task: str, variant: str, slots: int, rounds: int, seed: int) -> Path:
    return root / task / variant / f"D64_S{slots}_R{rounds}" / f"seed_{seed}"


def load_bundle(run_dir: Path, task: str, variant: str, slots: int, rounds: int):
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    vocabulary = TokenVocabulary()
    adapter = InputAdapter(len(vocabulary), 64, slots, max_length=64)
    core = RecurrentCore(64, slots, rounds, variant)
    reader = OutputReader(len(vocabulary), 64)
    adapter.load_state_dict(checkpoint["adapter"])
    core.load_state_dict(checkpoint["core"])
    reader.load_state_dict(checkpoint["reader"])
    return vocabulary, adapter, core, reader


def test_examples(task: str):
    return load_jsonl(ROOT / "datasets" / task / "test.jsonl")


def forward_mode(adapter, core, reader, input_ids, mask, query_ids, task, mode="normal", sample_states=False):
    initial = adapter(input_ids, mask)
    state = initial
    states = [state]
    order = tuple(reversed(range(core.rounds))) if mode == "reverse_rounds" else tuple(range(core.rounds))
    for round_index in order:
        mixed = core.slot_mix(state)
        depth = torch.zeros_like(core.depth_embedding.weight[round_index]) if mode == "zero_depth" else core.depth_embedding.weight[round_index]
        update = core.cores[round_index if core.variant == "untied" else 0](mixed + depth.view(1, 1, core.dimension))
        gate = torch.sigmoid(core.gate_logits[round_index]).view(1, 1, core.dimension)
        base = initial if mode == "freeze_initial" else state
        state = core.rms_norm(base + gate * update)
        states.append(state)
    logits = reader(state, query_ids, task)
    return logits, tuple(states) if sample_states else None


@torch.no_grad()
def accuracy_for_mode(adapter, core, reader, examples, vocabulary, task, mode="normal", limit=None):
    rows = examples[:limit] if limit else examples
    input_ids, mask, query_ids, targets = encode_batch(rows, vocabulary)
    correct = 0
    for start in range(0, len(rows), 128):
        logits, _ = forward_mode(adapter, core, reader, input_ids[start:start + 128], mask[start:start + 128], query_ids[start:start + 128], task, mode)
        correct += int((logits.argmax(dim=-1) == targets[start:start + 128]).sum())
    return correct / len(rows)


def gate4_diagnostics(run_dir: Path, task: str, slots: int, rounds: int) -> dict[str, object]:
    vocabulary, adapter, core, reader = load_bundle(run_dir, task, "shared", slots, rounds)
    examples = test_examples(task)[:512]
    input_ids, mask, query_ids, targets = encode_batch(examples, vocabulary)
    final_states: list[Tensor] = []
    normal_logits: list[Tensor] = []
    attention_entropies: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            ids = input_ids[start:start + 128]
            batch_mask = mask[start:start + 128]
            state0 = adapter(ids, batch_mask)
            token_values = adapter.token_embedding(ids)
            positions = torch.arange(ids.shape[1]).view(1, -1)
            token_values = token_values + adapter.position_embedding(positions)
            queries = adapter.query(adapter.slot_queries).unsqueeze(0).expand(ids.shape[0], -1, -1)
            scores = torch.matmul(queries, adapter.key(token_values).transpose(-2, -1)) * adapter.scale
            scores = scores.masked_fill(~batch_mask[:, None, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            attention_entropies.append(-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1))
            logits, states = forward_mode(adapter, core, reader, ids, batch_mask, query_ids[start:start + 128], task, sample_states=True)
            final_states.append(states[-1])
            normal_logits.append(logits)
    state = torch.cat(final_states)
    logits = torch.cat(normal_logits)
    metrics: dict[str, object] = {"samples": len(examples), "test_accuracy": float((logits.argmax(-1) == targets).float().mean())}
    metrics["attention_entropy_mean"] = float(torch.cat(attention_entropies).mean())
    if slots > 1:
        normalized = F.normalize(state, dim=-1)
        cosine_values = []
        ranks = []
        for row in state:
            cosine = torch.matmul(F.normalize(row, dim=-1), F.normalize(row, dim=-1).transpose(0, 1))
            cosine_values.append(cosine[~torch.eye(slots, dtype=torch.bool)].abs().mean())
            singular = torch.linalg.svdvals(row)
            probabilities = singular / singular.sum().clamp_min(1e-12)
            ranks.append(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
        metrics["mean_absolute_slot_cosine"] = float(torch.stack(cosine_values).mean())
        metrics["effective_rank_mean"] = float(torch.stack(ranks).mean())
    else:
        metrics["mean_absolute_slot_cosine"] = None
        metrics["effective_rank_mean"] = 1.0
    ablation_deltas = []
    for slot in range(slots):
        ablated = state.clone()
        ablated[:, slot, :] = 0
        ablated_logits = reader(ablated, query_ids[: len(examples)], task)
        ablated_accuracy = float((ablated_logits.argmax(-1) == targets).float().mean())
        ablation_deltas.append(float(metrics["test_accuracy"] - ablated_accuracy))
    metrics["slot_ablation_accuracy_deltas"] = ablation_deltas

    detached_state = state.detach().requires_grad_(True)
    loss = F.cross_entropy(reader(detached_state, query_ids[: len(examples)], task), targets)
    loss.backward()
    slot_norms = detached_state.grad.norm(dim=-1).mean(dim=0)
    threshold = float(slot_norms.max()) * 0.01
    metrics["slot_gradient_norms"] = [float(value) for value in slot_norms]
    metrics["slots_with_significant_gradient_fraction"] = float((slot_norms >= threshold).float().mean()) if slots else 0.0
    return metrics


def gate6_diagnostics(run_dir: Path, task: str) -> dict[str, object]:
    vocabulary, adapter, core, reader = load_bundle(run_dir, task, "shared", 4, 4)
    examples = test_examples(task)[:512]
    input_ids, mask, query_ids, targets = encode_batch(examples, vocabulary)
    state_changes = []
    modes = {}
    with torch.no_grad():
        for mode in ("normal", "reverse_rounds", "zero_depth", "freeze_initial"):
            correct = 0
            for start in range(0, len(examples), 128):
                logits, states = forward_mode(adapter, core, reader, input_ids[start:start + 128], mask[start:start + 128], query_ids[start:start + 128], task, mode, sample_states=True)
                correct += int((logits.argmax(-1) == targets[start:start + 128]).sum())
                if mode == "normal":
                    state_changes.extend(float((states[index] - states[index - 1]).norm(dim=-1).mean()) for index in range(1, len(states)))
            modes[mode] = correct / len(examples)
    return {"samples": len(examples), "accuracy_by_mode": modes, "mean_state_change_norm": statistics.mean(state_changes)}


def aggregate(finals: list[dict[str, object]], task: str, variant: str, slots: int, rounds: int) -> list[float]:
    return [float(row["test_accuracy"]) for row in finals if row["task"] == task and row["variant"] == variant and row["slots"] == slots and row["rounds"] == rounds]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "campaign")
    args = parser.parse_args()
    runs_root = args.campaign_dir / "runs"
    finals = [json.loads(path.read_text()) for path in runs_root.rglob("final.json")]
    summary: dict[str, object] = {"run_count": len(finals), "gate1": {}, "gate2": {}, "gate3": {}, "gate4": {}, "gate5": {}, "gate6": {}}

    for task in TASKS:
        values = aggregate(finals, task, "shared", 4, 4)
        summary["gate1"][task] = {"test_accuracy_per_seed": values, "mean": statistics.mean(values), "min": min(values), "max": max(values)}

    r1 = aggregate(finals, "multi_hop", "shared", 4, 1)
    r4 = aggregate(finals, "multi_hop", "shared", 4, 4)
    summary["gate2"] = {"r1_per_seed": r1, "r4_per_seed": r4, "gain_pp_per_seed": [(b - a) * 100 for a, b in zip(r1, r4)], "mean_gain_pp": (statistics.mean(r4) - statistics.mean(r1)) * 100}

    for task in TASKS[:4]:
        shared = aggregate(finals, task, "shared", 4, 4)
        untied = aggregate(finals, task, "untied", 4, 4)
        summary["gate3"][task] = {"shared_per_seed": shared, "untied_per_seed": untied, "retention_per_seed": [s / u if u else None for s, u in zip(shared, untied)], "mean_shared": statistics.mean(shared), "mean_untied": statistics.mean(untied), "mean_delta_pp": (statistics.mean(shared) - statistics.mean(untied)) * 100}

    for task in ("associative_recall", "multi_hop"):
        summary["gate4"][task] = {}
        for slots in (1, 4, 8):
            run = run_path(runs_root, task, "shared", slots, 4, 101)
            summary["gate4"][task][f"S{slots}"] = {"accuracy_per_seed": aggregate(finals, task, "shared", slots, 4), "diagnostics_seed101": gate4_diagnostics(run, task, slots, 4)}

    for seed in SEEDS:
        run = run_path(runs_root, "length_generalization", "shared", 4, 8, seed)
        vocabulary, adapter, core, reader = load_bundle(run, "length_generalization", "shared", 4, 8)
        examples = test_examples("length_generalization")
        by_hop: dict[str, float] = {}
        for hop in (4, 5, 6):
            selected = [row for row in examples if int(row.metadata["hop_count"]) == hop]
            by_hop[str(hop)] = accuracy_for_mode(adapter, core, reader, selected, vocabulary, "length_generalization")
        summary["gate5"][str(seed)] = {"accuracy_by_ood_hop": by_hop, "aggregate": aggregate(finals, "length_generalization", "shared", 4, 8)[SEEDS.index(seed)]}

    for seed in SEEDS:
        run = run_path(runs_root, "multi_hop", "shared", 4, 4, seed)
        summary["gate6"][str(seed)] = gate6_diagnostics(run, "multi_hop")

    output = args.campaign_dir / "analysis_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
