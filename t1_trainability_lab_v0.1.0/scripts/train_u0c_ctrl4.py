"""T1-CTRL-4 supervisor data preparation and pilot training.

Use ``--prepare-only`` to build and inspect frozen-reference data without
starting the 5,000-update supervisor pilot.
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
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from evaluate_u0c_ctrl3 import dispatch_adjustment_iterative
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, INCREASE, READ_E, READ_P, dispatch_unified_action, oracle_action, pair_value
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, materialize_graph_batch
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t1_trainability.unified import ROW_PAIR, ROW_REL


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SCORER_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401"
ACTION_NAMES = ("READ_P", "READ_E", "COPY_E_R", "INCREASE", "DECREASE", "EMIT")
ACTION_COUNT = len(ACTION_NAMES)


class SupervisorMLP(nn.Module):
    """7 -> 32 -> SiLU -> 6 next-primitive supervisor."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, ACTION_COUNT))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def supervisor_features(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, state: Tensor, goal: Tensor, reference: int, *, v_e: bool, v_r: bool) -> Tensor:
    """Build seven real recognizer outputs plus execution availability bits."""
    navigation_probabilities = F.softmax(ctrl1(state[:, SLOT_P], goal), dim=-1)
    if v_r:
        q_r, _ = canonical_value_view(model, state[:, SLOT_R])
        q_b, _ = canonical_value_view(model, model.token_embedding(torch.tensor([VALUE_BASE + reference])))
        comparison_probabilities = F.softmax(scorer.logits(q_r, q_b), dim=-1)
    else:
        comparison_probabilities = torch.zeros((state.shape[0], 3))
    availability = torch.tensor([[float(v_e), float(v_r)]])
    return torch.cat((navigation_probabilities, comparison_probabilities, availability), dim=-1)


def expected_read_row(graph: dict[str, Any], symbolic_pointer: int, action: int, goal_key: int) -> int:
    if action == READ_P:
        return next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == symbolic_pointer)
    if action == READ_E:
        return next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == goal_key)
    return -1


@torch.no_grad()
def generate_split(model: Any, ctrl1: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    features: list[Tensor] = []
    labels: list[int] = []
    metadata: dict[str, list[int]] = {"episode": [], "graph": [], "hops": [], "x": [], "reference": [], "decision": [], "v_E": [], "v_R": []}
    for episode in manifest["episodes"]:
        graph = manifest["graphs"][episode["graph"]]
        memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
        for reference in reference_values(pair_value(manifest, episode)):
            state = torch.zeros((1, SLOT_COUNT, DIMENSION))
            state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] + KEY_BASE]))
            presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
            goal = model.token_embedding(torch.tensor([episode["goal_key"] + KEY_BASE]))
            symbolic_pointer = episode["start_key"]
            symbolic_x = pair_value(manifest, episode)
            read_e_done = False
            copied = False
            v_e = False
            v_r = False
            for decision in range(38):
                action = oracle_action(symbolic_pointer, episode["goal_key"], read_e_done=read_e_done, copied=copied, symbolic_x=symbolic_x, reference=reference)
                features.append(supervisor_features(model, ctrl1, scorer, state, goal, reference, v_e=v_e, v_r=v_r).squeeze(0))
                labels.append(action)
                for key in metadata:
                    metadata[key].append({"episode": episode["episode"], "graph": episode["graph"], "hops": episode["distance"], "x": symbolic_x, "reference": reference, "decision": decision, "v_E": int(v_e), "v_R": int(v_r)}[key])
                state, operation = dispatch_unified_action(model, memory_keys, memory_values, memory_types, row_mask, state, presence, action, v_e=v_e, v_r=v_r)
                if action == READ_P:
                    expected_row = expected_read_row(graph, symbolic_pointer, action, episode["goal_key"])
                    symbolic_pointer = graph["rows"][expected_row]["value"]
                    v_e = False
                elif action == READ_E:
                    read_e_done = True
                    v_e = True
                elif action == COPY_E_R:
                    copied = True
                    v_r = True
                elif action == INCREASE:
                    symbolic_x += 1
                    v_r = True
                elif action == DECREASE:
                    symbolic_x -= 1
                    v_r = True
                elif action == EMIT:
                    break
            else:
                raise RuntimeError(f"oracle data generation timed out at episode {episode['episode']}")
    observations = {"features": torch.stack(features)}
    label_tensor = {"action": torch.tensor(labels, dtype=torch.long), **{key: torch.tensor(value, dtype=torch.long) for key, value in metadata.items()}}
    counts = {name: int((label_tensor["action"] == action).sum()) for action, name in enumerate(ACTION_NAMES)}
    return observations, label_tensor, {"episodes": len(manifest["episodes"]), "observations": len(labels), "class_counts": counts, "feature_count": 7, "features": ["ctrl1_p_0", "ctrl1_p_1", "ctrl2_order", "ctrl2_keep", "ctrl2_reverse", "v_E", "v_R"]}


def save_split(root: Path, split: str, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, str]:
    output = root / split
    output.mkdir(parents=True, exist_ok=True)
    torch.save(observations, output / "observations.pt")
    torch.save(labels, output / "labels.pt")
    return {name: sha256(output / f"{name}.pt") for name in ("observations", "labels")}


def prepare_data(root: Path, manifests: dict[str, dict[str, Any]], model: Any, ctrl1: Any, scorer: OrdinalSharedScorer) -> dict[str, Any]:
    metadata: dict[str, Any] = {"splits": {}, "frozen": True, "labels_are_features": False}
    for split, manifest in manifests.items():
        observations, labels, split_info = generate_split(model, ctrl1, scorer, manifest)
        metadata["splits"][split] = {**split_info, "hashes": save_split(root, split, observations, labels)}
    (root / "data_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def train_supervisor(model: SupervisorMLP, train: tuple[dict[str, Tensor], dict[str, Tensor]], val: tuple[dict[str, Tensor], dict[str, Tensor]], root: Path, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    observations, labels = train
    val_observations, val_labels = val
    buckets = [torch.where(labels["action"] == action)[0] for action in range(ACTION_COUNT)]
    batch_counts = (22, 22, 21, 21, 21, 21)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_step = 0
    metrics: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed + 1)
    for step in range(1, 5001):
        model.train()
        indices = torch.cat([bucket[torch.randint(len(bucket), (count,), generator=generator)] for bucket, count in zip(buckets, batch_counts)])
        progress = (step - 1) / 4999
        lr = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(observations["features"][indices]), labels["action"][indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == 5000:
            model.eval()
            with torch.no_grad():
                logits = model(val_observations["features"])
                val_loss = float(criterion(logits, val_labels["action"]).item())
                val_accuracy = float((logits.argmax(-1) == val_labels["action"]).float().mean().item())
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": lr})
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss; best_step = step; best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("missing supervisor validation checkpoint")
    torch.save({"supervisor": model.state_dict(), "seed": seed, "updates": 5000, "trainable_parameters": parameter_count(model)}, root / "final.pt")
    torch.save({"supervisor": best_state, "seed": seed, "step": best_step, "validation_loss": best_loss, "trainable_parameters": parameter_count(model)}, root / "selected.pt")
    (root / "training_metrics.jsonl").write_text("".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics), encoding="utf-8")
    return {"updates": 5000, "batch_size": 128, "batch_counts": list(batch_counts), "seed": seed, "best_step": best_step, "best_validation_loss": best_loss, "trainable_parameters": parameter_count(model)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scorer-checkpoint", type=Path, default=SCORER_CHECKPOINT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=4401)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    model = load_executor(); ctrl1 = load_ctrl1()
    payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval()
    data = prepare_data(args.output_root, manifests, model, ctrl1, scorer)
    result: dict[str, Any] = {"status": "prepared", "task": "T1-CTRL-4", "seed": args.seed, "supervisor_parameters": parameter_count(SupervisorMLP()), "data": data, "checkpoint_ctrl2": {"path": str(args.scorer_checkpoint), "sha256": sha256(args.scorer_checkpoint), "training_seed": payload.get("controller_seed")}, "training": False}
    if not args.prepare_only:
        train = (torch.load(args.output_root / "train" / "observations.pt", weights_only=False), torch.load(args.output_root / "train" / "labels.pt", weights_only=False))
        val = (torch.load(args.output_root / "val" / "observations.pt", weights_only=False), torch.load(args.output_root / "val" / "labels.pt", weights_only=False))
        result["training"] = train_supervisor(SupervisorMLP(), train, val, args.output_root, args.seed)
    output = args.output_root / ("prepare_results.json" if args.prepare_only else "results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
