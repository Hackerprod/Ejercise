"""T1-CTRL-6 three-objective supervisor pilot.

CLAMP (1, 1) is deliberately absent from every generated split and every
training-time metric. The held-out composition test is a separate evaluator.
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

from ctrl2_common import reference_values
from evaluate_u0c_c1_e_r_alu import VALUE_COUNT
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, INCREASE, READ_E, READ_P
from evaluate_u0c_ctrl5 import goal_descriptor
from evaluate_u0c_ctrl6_preflight import GOALS, supervisor_features_ctrl6, target_value
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl6_pilot_seed4601"

TRAIN_GOALS = {(0, 0): "NONE", (1, 0): "LOWER", (0, 1): "UPPER"}
GOAL_IDS = {name: index for index, name in enumerate(TRAIN_GOALS.values())}
ACTION_NAMES = ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")
ACTION_COUNT = len(ACTION_NAMES)
BOUND_PAIRS = ((0, 0), (0, 8), (8, 16), (8, 24), (16, 31), (31, 31))


def goal_mask(constraints: Tensor, goal: str) -> Tensor:
    if goal == "NONE":
        return (constraints[:, 0] == 0) & (constraints[:, 1] == 0)
    if goal == "LOWER":
        return (constraints[:, 0] == 1) & (constraints[:, 1] == 0)
    if goal == "UPPER":
        return (constraints[:, 0] == 0) & (constraints[:, 1] == 1)
    raise ValueError(f"unsupported training goal: {goal}")


class GoalConditionedSupervisor614(nn.Module):
    """10 observations + 2 condition indicators -> 32 SiLU -> 6 actions."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_projection = nn.Linear(10, 32)
        self.goal_projection = nn.Linear(2, 32, bias=False)
        self.output_projection = nn.Linear(32, ACTION_COUNT)

    def forward(self, features: Tensor, constraints: Tensor) -> Tensor:
        hidden = F.silu(self.observation_projection(features) + self.goal_projection(constraints))
        return self.output_projection(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def source_groups(source_root: Path, split: str) -> tuple[dict[str, Tensor], dict[tuple[int, int], list[int]]]:
    observations = torch.load(source_root / split / "observations.pt", map_location="cpu", weights_only=False)
    labels = torch.load(source_root / split / "labels.pt", map_location="cpu", weights_only=False)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (episode, reference) in enumerate(zip(labels["episode"].tolist(), labels["reference"].tolist())):
        groups.setdefault((episode, reference), []).append(index)
    return {"features": observations["features"], **labels}, groups


def comparison_table(scorer: OrdinalSharedScorer, model: Any) -> Tensor:
    values = model.token_embedding(torch.arange(288, 320))
    q_values, _ = __import__("evaluate_u0c_ctrl2_o_canon").canonical_value_view(model, values)
    table = torch.zeros((VALUE_COUNT, VALUE_COUNT, 3))
    for value in range(VALUE_COUNT):
        for reference in range(VALUE_COUNT):
            table[value, reference] = F.softmax(scorer.logits(q_values[value:value + 1], q_values[reference:reference + 1]), dim=-1)[0]
    return table


def scenario_rows(labels: dict[str, Tensor], groups: dict[tuple[int, int], list[int]], episode: int, x0: int, lower: int, upper: int, target: int) -> tuple[list[int], list[int]]:
    if target < x0:
        source_reference = 0
    elif target > x0:
        source_reference = 31
    else:
        source_reference = x0
    indices = groups.get((episode, source_reference))
    if not indices:
        raise RuntimeError(f"missing source trace episode={episode} reference={source_reference}")
    source_actions = labels["action"][indices].tolist()
    copy_position = source_actions.index(COPY_E_R)
    required = copy_position + 1 + abs(target - x0) + 1
    if len(indices) < required:
        raise RuntimeError(f"source trace too short episode={episode} x0={x0} target={target}")
    selected = indices[:required]
    actions = labels["action"][indices[:copy_position + 1]].tolist()
    actions.extend([INCREASE if target > x0 else DECREASE] * abs(target - x0))
    actions.append(EMIT)
    return selected, actions


def generate_split(source_root: Path, split: str, cmp_table: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    labels, groups = source_groups(source_root, split)
    features: list[Tensor] = []; constraints: list[Tensor] = []; actions: list[int] = []
    metadata: dict[str, list[int]] = {"episode": [], "graph": [], "hops": [], "x0": [], "lower": [], "upper": [], "goal_id": [], "pair_id": [], "decision": [], "v_E": [], "v_R": []}
    prefix_mismatches = 0; pair_count = 0
    episodes = sorted({episode for episode, _ in groups})
    for episode in episodes:
        episode_indices = next(indices for (episode_id, _), indices in groups.items() if episode_id == episode)
        x0 = int(labels["x"][episode_indices[0]].item())
        for bound_index, (lower, upper) in enumerate(BOUND_PAIRS):
            pair_rows: dict[str, tuple[list[int], list[int]]] = {}
            for goal_name, descriptor in TRAIN_GOALS.items():
                target = target_value(x0, lower, upper, goal_name)
                pair_rows[descriptor] = scenario_rows(labels, groups, episode, x0, lower, upper, target)
            shared = pair_rows["NONE"][0][:len([a for a in pair_rows["NONE"][1] if a in (READ_P, READ_E, COPY_E_R)])]
            if not shared:
                raise RuntimeError("empty shared prefix")
            pair_count += 1
            for goal_name, descriptor in TRAIN_GOALS.items():
                selected, expected_actions = pair_rows[descriptor]
                for decision, (index, action) in enumerate(zip(selected, expected_actions)):
                    source_feature = labels["features"][index] if "features" in labels else None
                    if source_feature is None:
                        source_feature = torch.zeros(7)
                    x = int(labels["x"][index].item()) if decision < len(selected) else x0
                    nav = source_feature[:2]
                    availability = source_feature[5:7]
                    comparison = torch.cat((cmp_table[x, lower], cmp_table[x, upper])) if bool(availability[1]) else torch.zeros(6)
                    features.append(torch.cat((nav, comparison, availability)))
                    constraints.append(torch.tensor([float(goal_name[0]), float(goal_name[1])]))
                    actions.append(action)
                    metadata["episode"].append(int(labels["episode"][index].item())); metadata["graph"].append(int(labels["graph"][index].item())); metadata["hops"].append(int(labels["hops"][index].item())); metadata["x0"].append(x0); metadata["lower"].append(lower); metadata["upper"].append(upper); metadata["goal_id"].append(GOAL_IDS[descriptor]); metadata["pair_id"].append(episode * len(BOUND_PAIRS) + bound_index); metadata["decision"].append(decision); metadata["v_E"].append(int(availability[0].item())); metadata["v_R"].append(int(availability[1].item()))
    observations = {"features": torch.stack(features)}
    dtypes = {"episode": torch.int32, "graph": torch.int16, "hops": torch.uint8, "x0": torch.uint8, "lower": torch.uint8, "upper": torch.uint8, "goal_id": torch.uint8, "pair_id": torch.int32, "decision": torch.uint8, "v_E": torch.bool, "v_R": torch.bool}
    label_tensor = {"action": torch.tensor(actions, dtype=torch.long), "constraints": torch.stack(constraints), **{key: torch.tensor(value, dtype=dtypes[key]) for key, value in metadata.items()}}
    counts = {goal: {name: int((goal_mask(label_tensor["constraints"], goal) & (label_tensor["action"] == action)).sum()) for action, name in enumerate(ACTION_NAMES)} for goal in TRAIN_GOALS.values()}
    return observations, label_tensor, {"episodes": len(episodes), "paired_rows": pair_count, "observations": len(actions), "pair_prefix_mismatches": prefix_mismatches, "class_counts_by_goal": counts, "feature_count": 10, "goal_descriptor_count": 2, "clamp_excluded": True, "bounds": list(BOUND_PAIRS), "features": ["ctrl1_p_0", "ctrl1_p_1", "cmp_lower_order", "cmp_lower_keep", "cmp_lower_reverse", "cmp_upper_order", "cmp_upper_keep", "cmp_upper_reverse", "v_E", "v_R"]}


def save_split(root: Path, split: str, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, str]:
    output = root / split; output.mkdir(parents=True, exist_ok=True); torch.save(observations, output / "observations.pt"); torch.save(labels, output / "labels.pt")
    return {name: sha256(output / f"{name}.pt") for name in ("observations", "labels")}


def train_supervisor(model: GoalConditionedSupervisor614, train: tuple[dict[str, Tensor], dict[str, Tensor]], val: tuple[dict[str, Tensor], dict[str, Tensor]], root: Path, seed: int) -> dict[str, Any]:
    observations, labels = train; val_observations, val_labels = val
    goal_buckets = {goal: {action: torch.where(goal_mask(labels["constraints"], goal) & (labels["action"] == action))[0] for action in range(ACTION_COUNT)} for goal in TRAIN_GOALS.values()}
    batch_counts = {"NONE": {READ_P: 11, READ_E: 11, COPY_E_R: 10, EMIT: 10}, "LOWER": {READ_P: 9, READ_E: 9, COPY_E_R: 8, INCREASE: 9, EMIT: 8}, "UPPER": {READ_P: 9, READ_E: 9, COPY_E_R: 8, DECREASE: 9, EMIT: 8}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); criterion = nn.CrossEntropyLoss(); generator = torch.Generator().manual_seed(seed + 1); metrics: list[dict[str, Any]] = []; best_loss = float("inf"); best_state: dict[str, Tensor] | None = None; best_step = 0
    for step in range(1, 5001):
        model.train(); selected: list[Tensor] = []
        for goal in TRAIN_GOALS.values():
            for action, count in batch_counts[goal].items():
                bucket = goal_buckets[goal][action]
                if len(bucket) == 0: raise RuntimeError(f"missing training bucket goal={goal} action={ACTION_NAMES[action]}")
                selected.append(bucket[torch.randint(len(bucket), (count,), generator=generator)])
        indices = torch.cat(selected); progress = (step - 1) / 4999; lr = 1e-3 + (1e-5 - 1e-3) * progress; optimizer.param_groups[0]["lr"] = lr; optimizer.zero_grad(set_to_none=True); logits = model(observations["features"][indices], labels["constraints"][indices]); loss = criterion(logits, labels["action"][indices]); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == 5000:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_observations["features"], val_labels["constraints"]); val_loss = float(criterion(val_logits, val_labels["action"]).item()); val_accuracy = float((val_logits.argmax(-1) == val_labels["action"]).float().mean().item())
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": lr})
            if val_loss < best_loss - 1e-12: best_loss = val_loss; best_step = step; best_state = copy.deepcopy(model.state_dict())
    if best_state is None: raise RuntimeError("missing validation checkpoint")
    torch.save({"supervisor": model.state_dict(), "seed": seed, "updates": 5000, "trainable_parameters": parameter_count(model), "clamp_trained": False}, root / "final.pt"); torch.save({"supervisor": best_state, "seed": seed, "step": best_step, "validation_loss": best_loss, "trainable_parameters": parameter_count(model), "clamp_trained": False}, root / "selected.pt"); (root / "training_metrics.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in metrics), encoding="utf-8")
    return {"updates": 5000, "batch_size": 128, "batch_counts": batch_counts, "seed": seed, "best_step": best_step, "best_validation_loss": best_loss, "trainable_parameters": parameter_count(model), "clamp_trained": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--seed", type=int, default=4601); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    model = __import__("ctrl2_common").load_executor(); ctrl1 = __import__("ctrl2_common").load_ctrl1(); payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); cmp_table = comparison_table(scorer, model)
    data: dict[str, Any] = {"splits": {}, "clamp_excluded": True}
    for split in ("train", "val", "test"):
        observations, labels, info = generate_split(SOURCE_ROOT, split, cmp_table); data["splits"][split] = {**info, "hashes": save_split(args.output_root, split, observations, labels)}
    (args.output_root / "data_metadata.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.manual_seed(args.seed); random.seed(args.seed); supervisor = GoalConditionedSupervisor614(); torch.save({"supervisor": copy.deepcopy(supervisor.state_dict()), "seed": args.seed, "initialization": "random from seed before construction; no CTRL-5 weights", "trainable_parameters": parameter_count(supervisor), "clamp_trained": False}, args.output_root / "initial.pt")
    result: dict[str, Any] = {"status": "prepared", "task": "T1-CTRL-6", "seed": args.seed, "supervisor_parameters": parameter_count(supervisor), "data": data, "training": False, "clamp_excluded": True}
    if not args.prepare_only: result["training"] = train_supervisor(supervisor, (torch.load(args.output_root / "train" / "observations.pt", weights_only=False), torch.load(args.output_root / "train" / "labels.pt", weights_only=False)), (torch.load(args.output_root / "val" / "observations.pt", weights_only=False), torch.load(args.output_root / "val" / "labels.pt", weights_only=False)), args.output_root, args.seed)
    output = args.output_root / ("prepare_results.json" if args.prepare_only else "results.json"); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
