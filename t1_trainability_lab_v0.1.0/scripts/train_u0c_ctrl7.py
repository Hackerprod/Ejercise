"""T1-CTRL-7 three-objective supervisor pilot.

FLOOR_AND_AVOID is excluded from every generated split and training metric.
The final checkpoint is the model after exactly 5000 updates; validation is
reported diagnostically and never used to replace that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, READ_E, READ_P
from evaluate_u0c_ctrl7_preflight import target_value
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"
TRAIN_GOALS = {(0, 0): "NONE", (1, 0): "FLOOR", (0, 1): "AVOID"}
GOAL_IDS = {name: index for index, name in enumerate(TRAIN_GOALS.values())}
ACTION_NAMES = ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")
BASE_BOUND_PAIRS = ((0, 0), (0, 8), (8, 16), (8, 24), (16, 30), (30, 30))


def bound_pairs_for_x(x0: int) -> tuple[tuple[int, int], ...]:
    pairs = list(BASE_BOUND_PAIRS)
    if x0 < 30:
        pairs.append((x0 + 1, x0 + 1))
    pairs.append((x0 + 1, 0)) if x0 < 31 else None
    pairs.append((0, x0))
    pairs.append((0, (x0 + 1) % 31))
    return tuple(dict.fromkeys(pairs))


class GoalConditionedSupervisor614(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation_projection = nn.Linear(10, 32)
        self.goal_projection = nn.Linear(2, 32, bias=False)
        self.output_projection = nn.Linear(32, 6)

    def forward(self, features: Tensor, constraints: Tensor) -> Tensor:
        return self.output_projection(F.silu(self.observation_projection(features) + self.goal_projection(constraints)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def goal_mask(constraints: Tensor, descriptor: str) -> Tensor:
    key = next(key for key, name in TRAIN_GOALS.items() if name == descriptor)
    return (constraints[:, 0] == key[0]) & (constraints[:, 1] == key[1])


def load_split(root: Path, split: str) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[tuple[int, int], list[int]]]:
    observations = torch.load(root / split / "observations.pt", weights_only=False)
    labels = torch.load(root / split / "labels.pt", weights_only=False)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (episode, reference) in enumerate(zip(labels["episode"].tolist(), labels["reference"].tolist())):
        groups.setdefault((episode, reference), []).append(index)
    return observations, labels, groups


def source_rows(labels: dict[str, Tensor], groups: dict[tuple[int, int], list[int]], episode: int, x0: int, target: int) -> tuple[list[int], list[int]]:
    reference = 0 if target < x0 else 31 if target > x0 else x0
    indices = groups[(episode, reference)]
    source_actions = labels["action"][indices].tolist()
    copy_position = source_actions.index(COPY_E_R)
    distance = abs(target - x0)
    selected = indices[: copy_position + 1 + distance + 1]
    actions = source_actions[: copy_position + 1]
    actions.extend([INCREASE] * distance)
    actions.append(EMIT)
    return selected, actions


def generate_split(source_root: Path, split: str, comparison: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    source_observations, labels, groups = load_split(source_root, split)
    features: list[Tensor] = []
    constraints: list[Tensor] = []
    actions: list[int] = []
    metadata: dict[str, list[int]] = {key: [] for key in ("episode", "graph", "hops", "x0", "lower", "forbidden", "goal_id", "pair_id", "decision", "v_E", "v_R")}
    episodes = sorted({episode for episode, _ in groups})
    for episode in episodes:
        episode_indices = groups[(episode, 0)]
        x0 = int(labels["x"][episode_indices[0]].item())
        bound_pairs = bound_pairs_for_x(x0)
        for pair_id, (lower, forbidden) in enumerate(bound_pairs):
            for descriptor_key, descriptor in TRAIN_GOALS.items():
                target = target_value(x0, lower, forbidden, descriptor_key)
                selected, expected_actions = source_rows(labels, groups, episode, x0, target)
                value = x0
                for decision, (index, action) in enumerate(zip(selected, expected_actions)):
                    source_feature = source_observations["features"][index]
                    v_e = bool(labels["v_E"][index].item())
                    v_r = bool(labels["v_R"][index].item())
                    nav = source_feature[:2]
                    cmp = torch.cat((comparison[value, lower], comparison[value, forbidden])) if v_r else torch.zeros(6)
                    features.append(torch.cat((nav, cmp, torch.tensor([float(v_e), float(v_r)]))))
                    constraints.append(torch.tensor([float(descriptor_key[0]), float(descriptor_key[1])]))
                    actions.append(action)
                    for key, val in (("episode", episode), ("graph", int(labels["graph"][index])), ("hops", int(labels["hops"][index])), ("x0", x0), ("lower", lower), ("forbidden", forbidden), ("goal_id", GOAL_IDS[descriptor]), ("pair_id", episode * len(bound_pairs) + pair_id), ("decision", decision), ("v_E", int(v_e)), ("v_R", int(v_r))):
                        metadata[key].append(val)
                    if action == INCREASE:
                        value += 1
    observation_tensor = {"features": torch.stack(features)}
    dtypes = {"episode": torch.int32, "graph": torch.int16, "hops": torch.uint8, "x0": torch.uint8, "lower": torch.uint8, "forbidden": torch.uint8, "goal_id": torch.uint8, "pair_id": torch.int32, "decision": torch.uint8, "v_E": torch.bool, "v_R": torch.bool}
    label_tensor = {"action": torch.tensor(actions, dtype=torch.long), "constraints": torch.stack(constraints), **{key: torch.tensor(values, dtype=dtypes[key]) for key, values in metadata.items()}}
    counts = {goal: {name: int((goal_mask(label_tensor["constraints"], goal) & (label_tensor["action"] == action)).sum()) for action, name in enumerate(ACTION_NAMES)} for goal in TRAIN_GOALS.values()}
    info = {"episodes": len(episodes), "observations": len(actions), "feature_count": 10, "goal_descriptor_count": 2, "clamp_excluded": True, "bounds": "base pairs plus x-specific causal pairs", "class_counts_by_goal": counts}
    return observation_tensor, label_tensor, info


def save_split(root: Path, split: str, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, str]:
    output = root / split
    output.mkdir(parents=True, exist_ok=True)
    torch.save(observations, output / "observations.pt")
    torch.save(labels, output / "labels.pt")
    return {name: sha256(output / f"{name}.pt") for name in ("observations", "labels")}


def train(model: GoalConditionedSupervisor614, train_data: tuple[dict[str, Tensor], dict[str, Tensor]], val_data: tuple[dict[str, Tensor], dict[str, Tensor]], root: Path, seed: int) -> dict[str, Any]:
    observations, labels = train_data
    val_observations, val_labels = val_data
    buckets = {goal: {action: torch.where(goal_mask(labels["constraints"], goal) & (labels["action"] == action))[0] for action in (READ_P, READ_E, COPY_E_R, INCREASE, EMIT)} for goal in TRAIN_GOALS.values()}
    batch_counts = {"NONE": {READ_P: 8, READ_E: 8, COPY_E_R: 8, EMIT: 8}, "FLOOR": {READ_P: 8, READ_E: 8, COPY_E_R: 8, INCREASE: 8, EMIT: 16}, "AVOID": {READ_P: 8, READ_E: 8, COPY_E_R: 8, INCREASE: 8, EMIT: 16}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 1)
    metrics: list[dict[str, Any]] = []
    for step in range(1, 5001):
        selected = []
        for goal, counts in batch_counts.items():
            for action, count in counts.items():
                bucket = buckets[goal][action]
                if not len(bucket):
                    raise RuntimeError(f"missing bucket goal={goal} action={ACTION_NAMES[action]}")
                selected.append(bucket[torch.randint(len(bucket), (count,), generator=generator)])
        indices = torch.cat(selected)
        progress = (step - 1) / 4999
        lr = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(observations["features"][indices], labels["constraints"][indices]), labels["action"][indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == 5000:
            with torch.no_grad():
                logits = model(val_observations["features"], val_labels["constraints"])
                val_loss = float(criterion(logits, val_labels["action"]).item())
                val_accuracy = float((logits.argmax(-1) == val_labels["action"]).float().mean().item())
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": lr})
    torch.save({"supervisor": model.state_dict(), "seed": seed, "updates": 5000, "trainable_parameters": parameter_count(model), "clamp_trained": False}, root / "final.pt")
    (root / "training_metrics.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in metrics), encoding="utf-8")
    return {"updates": 5000, "batch_size": 128, "batch_counts": batch_counts, "seed": seed, "final_validation": metrics[-1], "trainable_parameters": parameter_count(model), "clamp_trained": False}


def comparison_table(scorer: OrdinalSharedScorer, model: Any) -> Tensor:
    from evaluate_u0c_ctrl2_o_canon import canonical_value_view
    values = model.token_embedding(torch.arange(288, 320))
    q_values, _ = canonical_value_view(model, values)
    table = torch.zeros((32, 32, 3))
    for left in range(32):
        for right in range(32):
            table[left, right] = F.softmax(scorer.logits(q_values[left:left + 1], q_values[right:right + 1]), dim=-1)[0]
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    from ctrl2_common import load_executor
    model = load_executor()
    payload = torch.load(SCORER_CHECKPOINT, weights_only=False)
    scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval()
    comparison = comparison_table(scorer, model)
    metadata: dict[str, Any] = {"splits": {}, "clamp_excluded": True, "train_goals": list(TRAIN_GOALS.values()), "bound_pairs": "base pairs plus x-specific causal pairs"}
    for split in ("train", "val", "test"):
        observations, labels, info = generate_split(SOURCE_ROOT, split, comparison)
        metadata["splits"][split] = {**info, "hashes": save_split(args.output_root, split, observations, labels)}
    (args.output_root / "data_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.manual_seed(args.seed); random.seed(args.seed)
    supervisor = GoalConditionedSupervisor614()
    torch.save({"supervisor": supervisor.state_dict(), "seed": args.seed, "initialization": "random from explicit seed", "trainable_parameters": parameter_count(supervisor), "clamp_trained": False}, args.output_root / "initial.pt")
    training = train(supervisor, (torch.load(args.output_root / "train" / "observations.pt", weights_only=False), torch.load(args.output_root / "train" / "labels.pt", weights_only=False)), (torch.load(args.output_root / "val" / "observations.pt", weights_only=False), torch.load(args.output_root / "val" / "labels.pt", weights_only=False)), args.output_root, args.seed)
    result = {"status": "trained", "task": "T1-CTRL-7", "seed": args.seed, "supervisor_parameters": parameter_count(supervisor), "data": metadata, "training": training, "clamp_excluded": True}
    (args.output_root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output_root / "results.json"), "checkpoint": str(args.output_root / "final.pt"), "checkpoint_sha256": sha256(args.output_root / "final.pt"), "status": result["status"], "training": training}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
