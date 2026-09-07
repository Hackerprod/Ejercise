"""T1-CTRL-5 goal-conditioned supervisor pilot.

``--prepare-only`` generates paired frozen-reference data and runs the
zero-conditioned initialization preflight. Training is intentionally skipped
in that mode.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, load_base_manifests, load_ctrl1, load_executor, reference_values
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_COUNT
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, FETCH, FETCH_AND_ADJUST, GOAL_NAMES, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, oracle_action, pair_value
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from train_u0c_ctrl4 import ACTION_NAMES, ACTION_COUNT, SupervisorMLP, expected_read_row, supervisor_features


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SUPERVISOR_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401" / "final.pt"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl5_pilot_seed4501"
GOAL_IDS = {FETCH: 0, FETCH_AND_ADJUST: 1}


class GoalConditionedSupervisor(nn.Module):
    """7 observations + 2 goal projection -> 32 SiLU -> 6 actions (518 params)."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_projection = nn.Linear(7, 32)
        self.goal_projection = nn.Linear(2, 32, bias=False)
        self.output_projection = nn.Linear(32, ACTION_COUNT)

    def forward(self, features: Tensor, goal_descriptor: Tensor) -> Tensor:
        hidden = F.silu(self.observation_projection(features) + self.goal_projection(goal_descriptor))
        return self.output_projection(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def copy_approved_supervisor(target: GoalConditionedSupervisor, checkpoint: Path) -> dict[str, Any]:
    base = SupervisorMLP()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(payload["supervisor"], strict=True)
    with torch.no_grad():
        target.observation_projection.weight.copy_(base.network[0].weight)
        target.observation_projection.bias.copy_(base.network[0].bias)
        target.output_projection.weight.copy_(base.network[2].weight)
        target.output_projection.bias.copy_(base.network[2].bias)
        target.goal_projection.weight.zero_()
    return {"path": str(checkpoint), "sha256": sha256(checkpoint), "seed": payload.get("seed")}


@torch.no_grad()
def initialization_preflight(conditioned: GoalConditionedSupervisor, approved: SupervisorMLP, data_root: Path) -> dict[str, Any]:
    conditioned.eval(); approved.eval()
    if parameter_count(conditioned) != 518:
        raise RuntimeError(f"conditioned supervisor must have 518 parameters, got {parameter_count(conditioned)}")
    if not torch.equal(conditioned.goal_projection.weight, torch.zeros_like(conditioned.goal_projection.weight)):
        raise RuntimeError("goal projection is not zero at initialization")
    comparisons = 0
    max_logit_delta = 0.0
    mismatches = 0
    for split in ("train", "val", "test"):
        observations = torch.load(data_root / split / "observations.pt", map_location="cpu", weights_only=False)["features"]
        for start in range(0, len(observations), 65536):
            features = observations[start:start + 65536]
            base_logits = approved(features)
            for goal in GOAL_NAMES:
                descriptor = torch.zeros((len(features), 2))
                descriptor[:, GOAL_IDS[goal]] = 1.0
                conditioned_logits = conditioned(features, descriptor)
                max_logit_delta = max(max_logit_delta, float((conditioned_logits - base_logits).abs().max().item()))
                mismatches += int((conditioned_logits.argmax(-1) != base_logits.argmax(-1)).sum().item())
                comparisons += len(features)
    return {"comparisons_per_goal": comparisons, "max_logit_delta": max_logit_delta, "argmax_mismatches": mismatches, "pass": max_logit_delta == 0.0 and mismatches == 0}


def generate_split(source_root: Path, split: str) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    """Build paired goals from frozen CTRL-4 traces without rerunning the core.

    ADJUST rows are retained verbatim. FETCH retains the shared prefix through
    COPY and relabels the first post-COPY observation as EMIT. That observation
    is already the real state produced by COPY, not a target-repaired state.
    """
    source = source_root / split
    source_observations = torch.load(source / "observations.pt", map_location="cpu", weights_only=False)
    source_labels = torch.load(source / "labels.pt", map_location="cpu", weights_only=False)
    action_tensor = source_labels["action"]
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (episode, reference) in enumerate(zip(source_labels["episode"].tolist(), source_labels["reference"].tolist())):
        groups.setdefault((episode, reference), []).append(index)
    all_features: list[Tensor] = []
    all_goals: list[Tensor] = []
    all_actions: list[Tensor] = []
    all_metadata: dict[str, list[int]] = {"episode": [], "graph": [], "hops": [], "x0": [], "reference": [], "goal_id": [], "pair_id": [], "decision": [], "v_E": [], "v_R": []}
    paired_rows = 0
    pair_prefix_mismatches = 0
    for (episode, reference), indices in groups.items():
        actions = action_tensor[indices].tolist()
        copy_position = actions.index(COPY_E_R)
        fetch_indices = indices[:copy_position + 2]
        if len(fetch_indices) != copy_position + 2:
            raise RuntimeError(f"missing post-COPY observation for episode={episode} reference={reference}")
        adjust_indices = indices
        fetch_features = source_observations["features"][fetch_indices].clone()
        adjust_features = source_observations["features"][adjust_indices]
        shared_count = copy_position + 1
        if not torch.equal(fetch_features[:shared_count], adjust_features[:shared_count]):
            pair_prefix_mismatches += 1
        paired_rows += 1
        for goal_name, selected_indices, selected_features in ((FETCH, fetch_indices, fetch_features), (FETCH_AND_ADJUST, adjust_indices, adjust_features)):
            selected_actions = action_tensor[selected_indices].clone()
            if goal_name == FETCH:
                selected_actions[-1] = EMIT
            all_features.append(selected_features)
            all_goals.append(torch.tensor([[float(goal_name == FETCH), float(goal_name == FETCH_AND_ADJUST)]]).expand(len(selected_indices), -1).clone())
            all_actions.append(selected_actions)
            x0 = int(source_labels["x"][indices[0]].item())
            pair_id = episode * VALUE_COUNT + reference
            for index in selected_indices:
                all_metadata["episode"].append(episode); all_metadata["graph"].append(int(source_labels["graph"][index].item())); all_metadata["hops"].append(int(source_labels["hops"][index].item())); all_metadata["x0"].append(x0); all_metadata["reference"].append(reference); all_metadata["goal_id"].append(GOAL_IDS[goal_name]); all_metadata["pair_id"].append(pair_id); all_metadata["decision"].append(int(source_labels["decision"][index].item())); all_metadata["v_E"].append(int(source_labels["v_E"][index].item())); all_metadata["v_R"].append(int(source_labels["v_R"][index].item()))
    observations = {"features": torch.cat(all_features)}
    metadata_dtypes = {"episode": torch.int32, "graph": torch.int16, "hops": torch.uint8, "x0": torch.uint8, "reference": torch.uint8, "goal_id": torch.uint8, "pair_id": torch.int32, "decision": torch.uint8, "v_E": torch.bool, "v_R": torch.bool}
    label_tensor = {"action": torch.cat(all_actions), "goal": torch.cat(all_goals), **{key: torch.tensor(value, dtype=metadata_dtypes[key]) for key, value in all_metadata.items()}}
    counts = {goal: {name: int(((label_tensor["goal"][:, GOAL_IDS[goal]] == 1) & (label_tensor["action"] == action)).sum()) for action, name in enumerate(ACTION_NAMES)} for goal in GOAL_NAMES}
    return observations, label_tensor, {"episodes": len({episode for episode, _ in groups}), "paired_rows": paired_rows, "pair_prefix_mismatches": pair_prefix_mismatches, "observations": len(label_tensor["action"]), "class_counts_by_goal": counts, "feature_count": 7, "goal_descriptor_count": 2, "paired_goals": True, "source": str(source_root), "reference_curriculum": "reference_values: anchors + current x + adjacent valid values (7-9 values per x), inherited frozen CTRL-4 traces", "features": ["ctrl1_p_0", "ctrl1_p_1", "ctrl2_order", "ctrl2_keep", "ctrl2_reverse", "v_E", "v_R"]}


def save_split(root: Path, split: str, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, str]:
    output = root / split; output.mkdir(parents=True, exist_ok=True)
    torch.save(observations, output / "observations.pt"); torch.save(labels, output / "labels.pt")
    return {name: sha256(output / f"{name}.pt") for name in ("observations", "labels")}


def prepare_data(root: Path, source_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"splits": {}, "frozen": True, "labels_are_features": False, "paired_goal_conditioning": True}
    for split in ("train", "val", "test"):
        observations, labels, split_info = generate_split(source_root, split)
        metadata["splits"][split] = {**split_info, "hashes": save_split(root, split, observations, labels)}
    (root / "data_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def train_supervisor(model: GoalConditionedSupervisor, train: tuple[dict[str, Tensor], dict[str, Tensor]], val: tuple[dict[str, Tensor], dict[str, Tensor]], root: Path, seed: int) -> dict[str, Any]:
    observations, labels = train; val_observations, val_labels = val
    goal_buckets = {goal: {action: torch.where((labels["goal"][:, GOAL_IDS[goal]] == 1) & (labels["action"] == action))[0] for action in range(ACTION_COUNT)} for goal in GOAL_NAMES}
    batch_counts = {FETCH: {READ_P: 16, READ_E: 16, COPY_E_R: 16, EMIT: 16}, FETCH_AND_ADJUST: {READ_P: 11, READ_E: 11, COPY_E_R: 11, INCREASE: 11, DECREASE: 10, EMIT: 10}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); criterion = nn.CrossEntropyLoss(); best_loss = float("inf"); best_state: dict[str, Tensor] | None = None; best_step = 0; metrics: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed + 1)
    for step in range(1, 5001):
        model.train(); selected: list[Tensor] = []
        for goal in GOAL_NAMES:
            for action, count in batch_counts[goal].items():
                bucket = goal_buckets[goal][action]
                selected.append(bucket[torch.randint(len(bucket), (count,), generator=generator)])
        indices = torch.cat(selected); progress = (step - 1) / 4999; lr = 1e-3 + (1e-5 - 1e-3) * progress; optimizer.param_groups[0]["lr"] = lr; optimizer.zero_grad(set_to_none=True)
        logits = model(observations["features"][indices], labels["goal"][indices]); loss = criterion(logits, labels["action"][indices]); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == 5000:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_observations["features"], val_labels["goal"]); val_loss = float(criterion(val_logits, val_labels["action"]).item()); val_accuracy = float((val_logits.argmax(-1) == val_labels["action"]).float().mean().item())
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": lr})
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss; best_step = step; best_state = copy.deepcopy(model.state_dict())
    if best_state is None: raise RuntimeError("missing conditioned supervisor validation checkpoint")
    torch.save({"supervisor": model.state_dict(), "seed": seed, "updates": 5000, "trainable_parameters": parameter_count(model)}, root / "final.pt")
    torch.save({"supervisor": best_state, "seed": seed, "step": best_step, "validation_loss": best_loss, "trainable_parameters": parameter_count(model)}, root / "selected.pt")
    (root / "training_metrics.jsonl").write_text("".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics), encoding="utf-8")
    return {"updates": 5000, "batch_size": 128, "batch_counts": batch_counts, "seed": seed, "best_step": best_step, "best_validation_loss": best_loss, "trainable_parameters": parameter_count(model)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--supervisor-checkpoint", type=Path, default=SUPERVISOR_CHECKPOINT); parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--seed", type=int, default=4501); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests(); model = load_executor(); ctrl1 = load_ctrl1(); scorer_payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
    source_root = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401"
    data = prepare_data(args.output_root, source_root)
    torch.manual_seed(args.seed); random.seed(args.seed); conditioned = GoalConditionedSupervisor(); approved = SupervisorMLP(); approved_payload = torch.load(args.supervisor_checkpoint, map_location="cpu", weights_only=False); approved.load_state_dict(approved_payload["supervisor"], strict=True); approved.eval(); source = copy_approved_supervisor(conditioned, args.supervisor_checkpoint)
    torch.save({"supervisor": copy.deepcopy(conditioned.state_dict()), "seed": args.seed, "initialization": "approved CTRL-4 weights copied; goal projection B zero; seed set before construction", "trainable_parameters": parameter_count(conditioned), "source_supervisor": source}, args.output_root / "initial.pt")
    init = initialization_preflight(conditioned, approved, args.output_root)
    result: dict[str, Any] = {"status": "prepared" if init["pass"] else "failed", "task": "T1-CTRL-5", "seed": args.seed, "supervisor_parameters": parameter_count(conditioned), "source_supervisor": source, "data": data, "initialization_preflight": init, "training": False}
    if not init["pass"]: raise RuntimeError(f"conditioned initialization preflight failed: {init}")
    if not args.prepare_only:
        train = (torch.load(args.output_root / "train" / "observations.pt", weights_only=False), torch.load(args.output_root / "train" / "labels.pt", weights_only=False)); val = (torch.load(args.output_root / "val" / "observations.pt", weights_only=False), torch.load(args.output_root / "val" / "labels.pt", weights_only=False)); result["training"] = train_supervisor(conditioned, train, val, args.output_root, args.seed)
    output = args.output_root / ("prepare_results.json" if args.prepare_only else "results.json"); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
