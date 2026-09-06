"""T1-CTRL-1 supervised controller pilot on a frozen C1 executor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE, VALUE_COUNT, C1JointModel, load_approved_model  # noqa: E402
from train_u0a import immediate_vectors  # noqa: E402
from t1_trainability.unified import OPCODE_IDS, ROW_PAIR, ROW_REL, SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R, SLOT_W  # noqa: E402


DISTANCES = (0, 1, 2, 3, 4)
MEMORY_ROWS = 32
NODES = 16
DECISION_LIMIT = 6
ADVANCE = 0
COLLECT = 1
ACTION_NAMES = ("ADVANCE", "COLLECT")
CHECKPOINT = ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt"
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl1_pilot_seed101_frozen"


def graph_manifest(graph_count: int, seed: int, split: str) -> dict[str, Any]:
    rng = random.Random(seed)
    graphs: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    episode_id = 0
    for graph_id in range(graph_count):
        cycle = rng.sample(range(256), NODES)
        rng.shuffle(cycle)
        pair_values = rng.sample(range(VALUE_COUNT), NODES)
        rows = [{"kind": ROW_REL, "key": cycle[index], "value": cycle[(index + 1) % NODES]} for index in range(NODES)]
        rows.extend({"kind": ROW_PAIR, "key": cycle[index], "value": pair_values[index]} for index in range(NODES))
        rng.shuffle(rows)
        start_index = rng.randrange(NODES)
        start_key = cycle[start_index]
        graphs.append({"graph": graph_id, "keys": cycle, "start_key": start_key, "rows": rows})
        for distance in DISTANCES:
            goal_key = cycle[(start_index + distance) % NODES]
            goal_value = next(row["value"] for row in rows if row["kind"] == ROW_PAIR and row["key"] == goal_key)
            episodes.append({"episode": episode_id, "graph": graph_id, "distance": distance, "start_key": start_key, "goal_key": goal_key, "target_id": VALUE_BASE + ((goal_value + 1) % VALUE_COUNT)})
            episode_id += 1
    manifest = {"split": split, "seed": seed, "graph_count": graph_count, "episode_count": len(episodes), "memory_rows": MEMORY_ROWS, "rows_per_type": {"REL": NODES, "PAIR": NODES}, "distances": list(DISTANCES), "decision_limit": DECISION_LIMIT, "graphs": graphs, "episodes": episodes}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if len(manifest["graphs"]) != manifest["graph_count"] or len(manifest["episodes"]) != manifest["graph_count"] * len(DISTANCES):
        raise ValueError("CTRL-1 manifest cardinality mismatch")
    for graph in manifest["graphs"]:
        rows = graph["rows"]
        if len(rows) != MEMORY_ROWS or sum(row["kind"] == ROW_REL for row in rows) != NODES or sum(row["kind"] == ROW_PAIR for row in rows) != NODES:
            raise ValueError(f"invalid graph rows {graph['graph']}")
        if len(set(graph["keys"])) != NODES or len({row["value"] for row in rows if row["kind"] == ROW_PAIR}) != NODES:
            raise ValueError(f"graph {graph['graph']} uniqueness failure")
        relation = {row["key"]: row["value"] for row in rows if row["kind"] == ROW_REL}
        if set(relation) != set(graph["keys"]) or set(relation.values()) != set(graph["keys"]):
            raise ValueError(f"graph {graph['graph']} is not a cycle")
        current = graph["start_key"]
        visited: set[int] = set()
        for _ in range(NODES):
            visited.add(current)
            current = relation[current]
        if len(visited) != NODES or current != graph["start_key"]:
            raise ValueError(f"graph {graph['graph']} cycle traversal failed")
    for episode in manifest["episodes"]:
        graph = manifest["graphs"][episode["graph"]]
        start_index = graph["keys"].index(episode["start_key"])
        goal_index = graph["keys"].index(episode["goal_key"])
        if (goal_index - start_index) % NODES != episode["distance"]:
            raise ValueError(f"episode {episode['episode']} distance mismatch")


def manifest_pairing_check(manifest: dict[str, Any]) -> dict[str, Any]:
    paired = 0
    for graph in manifest["graphs"]:
        episodes = [episode for episode in manifest["episodes"] if episode["graph"] == graph["graph"]]
        if len({episode["start_key"] for episode in episodes}) == 1 and {episode["distance"] for episode in episodes} == set(DISTANCES) and any(episode["distance"] == 0 for episode in episodes) and any(episode["distance"] > 0 for episode in episodes):
            paired += 1
    return {"graphs_with_same_start_and_multiple_goal_actions": paired, "expected_graphs": manifest["graph_count"]}


def materialize_graph_batch(model: C1JointModel, graphs: list[dict[str, Any]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    memory_keys = torch.stack([model.token_embedding(torch.tensor([row["key"] + KEY_BASE for row in graph["rows"]])) for graph in graphs])
    memory_values = torch.stack([model.token_embedding(torch.tensor([row["value"] + KEY_BASE if row["kind"] == ROW_REL else VALUE_BASE + row["value"] for row in graph["rows"]])) for graph in graphs])
    memory_types = torch.tensor([[row["kind"] for row in graph["rows"]] for graph in graphs], dtype=torch.long)
    return memory_keys, memory_values, memory_types, torch.ones_like(memory_types, dtype=torch.bool)


def append_observation(store: dict[str, list[Any]], pointers: Tensor, goals: Tensor, episodes: list[dict[str, Any]], decision: int) -> None:
    store["pointer"].extend(pointers.detach().clone())
    store["goal"].extend(goals.detach().clone())
    store["episode"].extend(int(episode["episode"]) for episode in episodes)
    store["graph"].extend(int(episode["graph"]) for episode in episodes)
    store["distance"].extend(int(episode["distance"]) for episode in episodes)
    store["decision"].extend([decision] * len(episodes))


@torch.no_grad()
def generate_oracle_observations(model: C1JointModel, manifest: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    observations: dict[str, list[Any]] = {"pointer": [], "goal": [], "episode": [], "graph": [], "distance": [], "decision": []}
    labels: dict[str, list[int]] = {"action": [], "episode": [], "graph": [], "distance": [], "decision": []}
    for distance in DISTANCES:
        episodes = [episode for episode in manifest["episodes"] if episode["distance"] == distance]
        graphs = [manifest["graphs"][episode["graph"]] for episode in episodes]
        memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, graphs)
        batch_size = len(episodes)
        state = torch.zeros((batch_size, SLOT_COUNT, DIMENSION))
        state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] for episode in episodes]))
        presence = torch.ones((batch_size, SLOT_COUNT), dtype=torch.bool)
        zero = immediate_vectors(model, torch.full((batch_size,), 511, dtype=torch.long))
        for decision in range(distance + 1):
            append_observation(observations, state[:, SLOT_P], model.token_embedding(torch.tensor([episode["goal_key"] for episode in episodes])), episodes, decision)
            action = COLLECT if decision == distance else ADVANCE
            labels["action"].extend([action] * batch_size)
            labels["episode"].extend(int(episode["episode"]) for episode in episodes)
            labels["graph"].extend(int(episode["graph"]) for episode in episodes)
            labels["distance"].extend([distance] * batch_size)
            labels["decision"].extend([decision] * batch_size)
            if decision < distance:
                state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_P"], dtype=torch.long), zero, torch.full((batch_size,), SLOT_P, dtype=torch.long), torch.full((batch_size,), SLOT_P, dtype=torch.long), presence, read_mode="BLEND", read_set="explicit")
    observation_tensors = {"pointer": torch.stack(observations["pointer"]), "goal": torch.stack(observations["goal"]), "episode": torch.tensor(observations["episode"], dtype=torch.long), "graph": torch.tensor(observations["graph"], dtype=torch.long), "distance": torch.tensor(observations["distance"], dtype=torch.long), "decision": torch.tensor(observations["decision"], dtype=torch.long)}
    label_tensors = {"action": torch.tensor(labels["action"], dtype=torch.long), "episode": torch.tensor(labels["episode"], dtype=torch.long), "graph": torch.tensor(labels["graph"], dtype=torch.long), "distance": torch.tensor(labels["distance"], dtype=torch.long), "decision": torch.tensor(labels["decision"], dtype=torch.long)}
    return observation_tensors, label_tensors


class Ctrl1MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4 * DIMENSION, DIMENSION), nn.SiLU(), nn.Linear(DIMENSION, 2))

    def forward(self, pointer: Tensor, goal: Tensor) -> Tensor:
        p = F.normalize(pointer, dim=-1)
        q = F.normalize(goal, dim=-1)
        features = torch.cat((p, q, p - q, p * q), dim=-1)
        return self.network(features)


def freeze_executor(model: nn.Module) -> tuple[int, int]:
    before = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    after = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return before, after


def save_tensor_pair(output: Path, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> None:
    torch.save({key: value for key, value in observations.items()}, output / "observations.pt")
    torch.save({key: value for key, value in labels.items()}, output / "labels.pt")


def train_controller(controller: Ctrl1MLP, train_obs: dict[str, Tensor], train_labels: dict[str, Tensor], val_obs: dict[str, Tensor], val_labels: dict[str, Tensor], output: Path) -> dict[str, Any]:
    batch_size = 128
    max_updates = 5000
    lr_initial = 1e-3
    lr_final = 1e-5
    optimizer = torch.optim.AdamW(controller.parameters(), lr=lr_initial, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = sum(parameter.numel() for parameter in controller.parameters() if parameter.requires_grad)
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    rng = torch.Generator().manual_seed(101)
    metrics: list[dict[str, Any]] = []
    def evaluate_validation() -> tuple[float, float]:
        controller.eval()
        with torch.no_grad():
            logits = controller(val_obs["pointer"], val_obs["goal"])
            loss = float(criterion(logits, val_labels["action"]).item())
            accuracy = float((logits.argmax(-1) == val_labels["action"]).float().mean().item())
        controller.train()
        return loss, accuracy
    for step in range(1, max_updates + 1):
        indices = torch.randint(train_obs["pointer"].shape[0], (batch_size,), generator=rng)
        progress = (step - 1) / (max_updates - 1)
        lr = lr_initial + (lr_final - lr_initial) * progress
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = controller(train_obs["pointer"][indices], train_obs["goal"][indices])
        loss = criterion(logits, train_labels["action"][indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == max_updates:
            val_loss, val_accuracy = evaluate_validation()
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": lr})
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(controller.state_dict())
    final_state = copy.deepcopy(controller.state_dict())
    if best_state is None:
        raise RuntimeError("validation checkpoint was not produced")
    torch.save({"controller": final_state, "step": max_updates, "executor_frozen": True, "trainable_controller_parameters": trainable_parameters}, output / "final.pt")
    torch.save({"controller": best_state, "step": best_step, "validation_loss": best_loss, "selection_rule": "minimum validation CE, ties keep earliest checkpoint", "executor_frozen": True, "trainable_controller_parameters": trainable_parameters}, output / "selected.pt")
    (output / "training_metrics.jsonl").write_text("".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics), encoding="utf-8")
    return {"updates": max_updates, "batch_size": batch_size, "lr_initial": lr_initial, "lr_final": lr_final, "weight_decay": 0.0, "trainable_controller_parameters": trainable_parameters, "best_step": best_step, "best_validation_loss": best_loss, "selection_rule": "minimum validation CE, ties keep earliest checkpoint"}


def pointer_decoder_id(model: C1JointModel, state: Tensor) -> Tensor:
    class_ids = torch.arange(256)
    return class_ids[model.pointer_decoder(state[:, SLOT_P], model.token_embedding(class_ids)).argmax(-1)]


@torch.no_grad()
def evaluate_oracle_test(model: C1JointModel, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for distance in DISTANCES:
        episodes = [episode for episode in manifest["episodes"] if episode["distance"] == distance]
        graphs = [manifest["graphs"][episode["graph"]] for episode in episodes]
        memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, graphs)
        batch_size = len(episodes)
        state = torch.zeros((batch_size, SLOT_COUNT, DIMENSION))
        state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] for episode in episodes]))
        presence = torch.ones((batch_size, SLOT_COUNT), dtype=torch.bool)
        zero = immediate_vectors(model, torch.full((batch_size,), 511, dtype=torch.long))
        trace: list[list[dict[str, Any]]] = [[] for _ in episodes]
        current = [episode["start_key"] for episode in episodes]
        reads_ok = [True] * batch_size
        for decision in range(distance):
            expected = [next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == current[item]) for item, graph in enumerate(graphs)]
            state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_P"], dtype=torch.long), zero, torch.full((batch_size,), SLOT_P, dtype=torch.long), torch.full((batch_size,), SLOT_P, dtype=torch.long), presence, read_mode="BLEND", read_set="explicit")
            selected = read.selected_index.tolist()
            for item in range(batch_size):
                match = int(selected[item]) == expected[item]
                reads_ok[item] &= match
                trace[item].append({"decision": decision, "action": "ADVANCE", "instruction": "READ_P", "selected_row": int(selected[item]), "expected_row": expected[item], "row_match": match, "attention_mass_correct": float(read.attention_soft[item, expected[item]]), "selection_margin": float(read.selection_margin[item])})
                current[item] = graphs[item]["rows"][expected[item]]["value"]
        expected_pair = [next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"]) for graph, episode in zip(graphs, episodes)]
        state, _, read_e = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_E"], dtype=torch.long), zero, torch.full((batch_size,), SLOT_P, dtype=torch.long), torch.full((batch_size,), SLOT_E, dtype=torch.long), presence, read_mode="BLEND", read_set="explicit", diagnostic_read_e_select=False)
        selected_e = read_e.selected_index.tolist()
        for item in range(batch_size):
            match = int(selected_e[item]) == expected_pair[item]
            reads_ok[item] &= match
            trace[item].append({"decision": distance, "action": "COLLECT", "instruction": "READ_E", "selected_row": int(selected_e[item]), "expected_row": expected_pair[item], "row_match": match, "attention_mass_correct": float(read_e.attention_soft[item, expected_pair[item]]), "selection_margin": float(read_e.selection_margin[item])})
        state[:, SLOT_R] = state[:, SLOT_E].clone()
        copy_ok = torch.equal(state[:, SLOT_R], state[:, SLOT_E])
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["ALU_ADD"], dtype=torch.long), immediate_vectors(model, torch.full((batch_size,), VALUE_BASE + 1, dtype=torch.long)), torch.full((batch_size,), SLOT_R, dtype=torch.long), torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=torch.zeros(batch_size, dtype=torch.long), read_set="explicit")
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["EMIT"], dtype=torch.long), zero, torch.full((batch_size,), SLOT_R, dtype=torch.long), torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=torch.zeros(batch_size, dtype=torch.long), read_set="explicit")
        decoded = (VALUE_BASE + 0) + (model.register_decoder(state[:, SLOT_R], model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))).argmax(-1))
        for item, episode in enumerate(episodes):
            target = int(episode["target_id"])
            exact = int(decoded[item]) == target
            trace[item].append({"instruction": "COPY", "copy_equal": copy_ok})
            trace[item].append({"instruction": "ALU_ADD", "decoded_r_id": int(decoded[item]), "expected_r_id": target, "exact": exact})
            results.append({"episode": int(episode["episode"]), "graph": int(episode["graph"]), "distance": distance, "target_id": target, "predicted_id": int(decoded[item]), "final_exact": exact, "episode_success": bool(exact and reads_ok[item] and copy_ok), "timeout": False, "first_control_error": None, "first_execution_error": None, "read_rows_ok": reads_ok[item], "copy_equal": copy_ok, "trace": trace[item]})
    return results


def action_accuracy(controller: Ctrl1MLP, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Any]:
    controller.eval()
    with torch.no_grad():
        predictions = controller(observations["pointer"], observations["goal"]).argmax(-1)
    by_distance = {str(distance): {"samples": int((labels["distance"] == distance).sum()), "correct": int(((predictions == labels["action"]) & (labels["distance"] == distance)).sum())} for distance in DISTANCES}
    return {"samples": len(predictions), "correct": int((predictions == labels["action"]).sum()), "accuracy": float((predictions == labels["action"]).float().mean()), "by_distance": by_distance}


@torch.no_grad()
def free_execution(model: C1JointModel, controller: Ctrl1MLP, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        graph = manifest["graphs"][episode["graph"]]
        memory_keys, memory_values, memory_types, row_mask = materialize_graph_batch(model, [graph])
        state = torch.zeros((1, SLOT_COUNT, DIMENSION))
        state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"]]))
        presence = torch.ones((1, SLOT_COUNT), dtype=torch.bool)
        zero = immediate_vectors(model, torch.tensor([511]))
        goal = model.token_embedding(torch.tensor([episode["goal_key"]]))
        symbolic_pointer = episode["start_key"]
        first_control_error: dict[str, Any] | None = None
        first_execution_error: dict[str, Any] | None = None
        trace: list[dict[str, Any]] = []
        collected = False
        for decision in range(DECISION_LIMIT):
            logits = controller(state[:, SLOT_P], goal)
            action = int(logits.argmax(-1).item())
            expected_action = COLLECT if symbolic_pointer == episode["goal_key"] else ADVANCE
            action_correct = action == expected_action
            if not action_correct and first_control_error is None:
                first_control_error = {"decision": decision, "predicted": ACTION_NAMES[action], "expected": ACTION_NAMES[expected_action]}
            entry: dict[str, Any] = {"decision": decision, "predicted_action": ACTION_NAMES[action], "expected_action": ACTION_NAMES[expected_action], "action_correct": action_correct, "pointer_decoder_id": int(pointer_decoder_id(model, state).item()), "goal_key": int(episode["goal_key"])}
            if action == ADVANCE:
                expected_row = next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_REL and row["key"] == symbolic_pointer)
                state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode="BLEND", read_set="explicit")
                selected = int(read.selected_index.item())
                row_ok = selected == expected_row
                pointer_ok = int(pointer_decoder_id(model, state).item()) == graph["rows"][expected_row]["value"]
                execution_error = action_correct and not (row_ok and pointer_ok)
                if execution_error and first_execution_error is None:
                    first_execution_error = {"decision": decision, "instruction": "READ_P", "selected_row": selected, "expected_row": expected_row, "pointer_decoder_id": int(pointer_decoder_id(model, state).item()), "expected_pointer": graph["rows"][expected_row]["value"]}
                entry.update({"instruction": "READ_P", "selected_row": selected, "expected_row": expected_row, "row_match": row_ok, "pointer_decoder_id_after": int(pointer_decoder_id(model, state).item()), "expected_pointer": graph["rows"][expected_row]["value"], "execution_error": execution_error})
                symbolic_pointer = graph["rows"][expected_row]["value"]
            else:
                expected_row = next(index for index, row in enumerate(graph["rows"]) if row["kind"] == ROW_PAIR and row["key"] == episode["goal_key"])
                state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode="BLEND", read_set="explicit", diagnostic_read_e_select=False)
                selected = int(read.selected_index.item())
                state[:, SLOT_R] = state[:, SLOT_E].clone()
                state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["ALU_ADD"]]), immediate_vectors(model, torch.tensor([VALUE_BASE + 1])), torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
                state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["EMIT"]]), zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
                predicted = int((VALUE_BASE + model.register_decoder(state[:, SLOT_R], model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))).argmax(-1)).item())
                final_exact = predicted == episode["target_id"]
                row_ok = selected == expected_row
                execution_error = action_correct and not (row_ok and final_exact)
                if execution_error and first_execution_error is None:
                    first_execution_error = {"decision": decision, "instruction": "READ_E_OR_SUFFIX", "selected_row": selected, "expected_row": expected_row, "predicted_id": predicted, "target_id": episode["target_id"]}
                entry.update({"instruction": "READ_E→COPY→ALU_ADD→EMIT", "selected_row": selected, "expected_row": expected_row, "row_match": row_ok, "predicted_id": predicted, "target_id": episode["target_id"], "execution_error": execution_error})
                trace.append(entry)
                collected = True
                break
            trace.append(entry)
        timeout = not collected
        final_predicted = trace[-1].get("predicted_id") if collected else None
        results.append({"episode": int(episode["episode"]), "graph": int(episode["graph"]), "distance": int(episode["distance"]), "target_id": int(episode["target_id"]), "predicted_id": final_predicted, "final_exact": final_predicted == episode["target_id"], "episode_success": bool(collected and final_predicted == episode["target_id"]), "timeout": timeout, "first_control_error": first_control_error, "first_execution_error": first_execution_error, "trace": trace})
    return results


def summarize_episodes(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [result for result in results if result["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "episode_success_count": sum(result["episode_success"] for result in selected), "final_exact_count": sum(result["final_exact"] for result in selected), "timeouts": sum(result["timeout"] for result in selected), "first_control_error_count": sum(result["first_control_error"] is not None for result in selected), "first_execution_error_count": sum(result["first_execution_error"] is not None for result in selected)}
    return {"samples": len(results), "episode_success_count": sum(result["episode_success"] for result in results), "final_exact_count": sum(result["final_exact"] for result in results), "timeouts": sum(result["timeout"] for result in results), "by_distance": by_distance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-seed", type=int, default=1101)
    parser.add_argument("--val-seed", type=int, default=1201)
    parser.add_argument("--test-seed", type=int, default=1301)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = {"train": graph_manifest(2000, args.train_seed, "train"), "val": graph_manifest(200, args.val_seed, "validation"), "test": graph_manifest(400, args.test_seed, "test")}
    manifest_hashes: dict[str, str] = {}
    for split, manifest in manifests.items():
        path = args.output_root / f"{split}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    executor = load_approved_model()
    executor.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    executor.eval()
    executor_trainable_before, executor_trainable_after = freeze_executor(executor)
    datasets: dict[str, tuple[dict[str, Tensor], dict[str, Tensor]]] = {}
    observation_hashes: dict[str, dict[str, str]] = {}
    for split, manifest in manifests.items():
        split_output = args.output_root / split
        split_output.mkdir(parents=True, exist_ok=True)
        observations, labels = generate_oracle_observations(executor, manifest)
        save_tensor_pair(split_output, observations, labels)
        observation_hashes[split] = {name: hashlib.sha256((split_output / f"{name}.pt").read_bytes()).hexdigest() for name in ("observations", "labels")}
        datasets[split] = (observations, labels)
    controller = Ctrl1MLP()
    training = train_controller(controller, datasets["train"][0], datasets["train"][1], datasets["val"][0], datasets["val"][1], args.output_root)
    selected_payload = torch.load(args.output_root / "selected.pt", map_location="cpu", weights_only=False)
    final_payload = torch.load(args.output_root / "final.pt", map_location="cpu", weights_only=False)
    test_oracle = evaluate_oracle_test(executor, manifests["test"])
    selected_controller = Ctrl1MLP(); selected_controller.load_state_dict(selected_payload["controller"])
    final_controller = Ctrl1MLP(); final_controller.load_state_dict(final_payload["controller"])
    test_action_selected = action_accuracy(selected_controller, datasets["test"][0], datasets["test"][1])
    test_action_final = action_accuracy(final_controller, datasets["test"][0], datasets["test"][1])
    free_selected = free_execution(executor, selected_controller, manifests["test"])
    free_final = free_execution(executor, final_controller, manifests["test"])
    (args.output_root / "test_oracle_traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in test_oracle), encoding="utf-8")
    (args.output_root / "test_free_selected_traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in free_selected), encoding="utf-8")
    (args.output_root / "test_free_final_traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in free_final), encoding="utf-8")
    summary = {"status": "completed", "task": "T1-CTRL-1", "checkpoint_final": {"path": str(args.checkpoint), "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}, "checkpoint_selected": {"path": str(args.output_root / "selected.pt"), "step": selected_payload["step"], "validation_loss": selected_payload.get("validation_loss")}, "manifest_hashes": manifest_hashes, "observation_hashes": observation_hashes, "split_sizes": {split: {"graphs": manifest["graph_count"], "episodes": manifest["episode_count"], "observations": int(datasets[split][0]["pointer"].shape[0]), "pairing": manifest_pairing_check(manifest)} for split, manifest in manifests.items()}, "executor_freeze": {"trainable_parameters_before": executor_trainable_before, "trainable_parameters_after": executor_trainable_after, "controller_trainable_parameters": training["trainable_controller_parameters"]}, "training": training, "oracle_executor_test": summarize_episodes(test_oracle), "controller_on_oracle_test_selected": test_action_selected, "controller_on_oracle_test_final": test_action_final, "free_selected": summarize_episodes(free_selected), "free_final": summarize_episodes(free_final), "policy_inputs": ["real_pointer_state", "frozen_goal_embedding"], "policy_excluded": ["distance", "decision_index", "opcode", "target", "trace", "memory", "expected_row", "already_at_goal_flag"], "diagnostic_read_e_select": False, "target_source": "oracle graph interpreter for labels/metrics; controller never receives targets or symbolic state"}
    (args.output_root / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
