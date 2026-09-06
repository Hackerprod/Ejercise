"""CTRL-1 oracle-policy preflight on frozen C1-anneal seed101."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_u0c_c1_e_r_alu import (  # noqa: E402
    DIMENSION,
    KEY_BASE,
    VALUE_BASE,
    VALUE_COUNT,
    C1JointModel,
    load_approved_model,
)
from train_u0a import immediate_vectors  # noqa: E402
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    ROW_PAIR,
    ROW_REL,
    SLOT_COUNT,
    SLOT_E,
    SLOT_P,
    SLOT_R,
    SLOT_W,
)


DISTANCES = (0, 1, 2, 3, 4)
GRAPH_COUNT = 256
MEMORY_ROWS = 32
NODES = 16
DECISION_LIMIT = 6


@dataclass(frozen=True)
class Episode:
    episode: int
    graph: int
    distance: int
    start_key: int
    goal_key: int
    rows: tuple[dict[str, int], ...]
    target_id: int


def make_manifest(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    graphs: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    episode_id = 0
    for graph_id in range(GRAPH_COUNT):
        keys = rng.sample(range(256), NODES)
        values = rng.sample(range(VALUE_COUNT), NODES)
        cycle = list(keys)
        rng.shuffle(cycle)
        rows = [{"kind": ROW_REL, "key": cycle[index], "value": cycle[(index + 1) % NODES]} for index in range(NODES)]
        rows.extend({"kind": ROW_PAIR, "key": cycle[index], "value": values[index]} for index in range(NODES))
        rng.shuffle(rows)
        start_index = rng.randrange(NODES)
        start_key = cycle[start_index]
        graphs.append({"graph": graph_id, "keys": cycle, "rows": rows, "start_key": start_key})
        for distance in DISTANCES:
            goal_key = cycle[(start_index + distance) % NODES]
            goal_value = next(row["value"] for row in rows if row["kind"] == ROW_PAIR and row["key"] == goal_key)
            episodes.append({"episode": episode_id, "graph": graph_id, "distance": distance, "start_key": start_key, "goal_key": goal_key, "target_id": VALUE_BASE + ((goal_value + 1) % VALUE_COUNT)})
            episode_id += 1
    manifest = {"seed": seed, "graph_count": GRAPH_COUNT, "nodes_per_graph": NODES, "memory_rows": MEMORY_ROWS, "rows_per_type": {"REL": NODES, "PAIR": NODES}, "distances": list(DISTANCES), "decision_limit": DECISION_LIMIT, "graphs": graphs, "episodes": episodes}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if len(manifest["graphs"]) != GRAPH_COUNT or len(manifest["episodes"]) != GRAPH_COUNT * len(DISTANCES):
        raise ValueError("invalid CTRL-1 manifest cardinality")
    for graph in manifest["graphs"]:
        rows = graph["rows"]
        if len(rows) != MEMORY_ROWS or sum(row["kind"] == ROW_REL for row in rows) != NODES or sum(row["kind"] == ROW_PAIR for row in rows) != NODES:
            raise ValueError(f"invalid graph row geometry {graph['graph']}")
        keys = graph["keys"]
        if len(set(keys)) != NODES or len(set(row["value"] for row in rows if row["kind"] == ROW_PAIR)) != NODES:
            raise ValueError(f"graph {graph['graph']} violates key/value uniqueness")
        rel = {row["key"]: row["value"] for row in rows if row["kind"] == ROW_REL}
        if set(rel) != set(keys) or set(rel.values()) != set(keys):
            raise ValueError(f"graph {graph['graph']} is not a complete directed cycle")
        current = graph["start_key"]
        visited: set[int] = set()
        for _ in range(NODES):
            if current in visited:
                break
            visited.add(current)
            current = rel[current]
        if len(visited) != NODES or current != graph["start_key"]:
            raise ValueError(f"graph {graph['graph']} does not form one full cycle")
    for episode in manifest["episodes"]:
        graph = manifest["graphs"][episode["graph"]]
        start_index = graph["keys"].index(episode["start_key"])
        goal_index = graph["keys"].index(episode["goal_key"])
        if (goal_index - start_index) % NODES != episode["distance"]:
            raise ValueError(f"distance mismatch in episode {episode['episode']}")


def manifest_checks(manifest: dict[str, Any]) -> dict[str, Any]:
    complete_cycles = 0
    unique_pair_values = 0
    goals_with_outgoing = 0
    rows_permuted = 0
    for graph in manifest["graphs"]:
        rows = graph["rows"]
        relation = {row["key"]: row["value"] for row in rows if row["kind"] == ROW_REL}
        current = graph["start_key"]
        visited: set[int] = set()
        for _ in range(NODES):
            visited.add(current)
            current = relation[current]
        complete_cycles += int(len(visited) == NODES and current == graph["start_key"])
        unique_pair_values += int(len({row["value"] for row in rows if row["kind"] == ROW_PAIR}) == NODES)
        goals_with_outgoing += sum(episode["goal_key"] in relation for episode in manifest["episodes"] if episode["graph"] == graph["graph"])
        logical_rows = [{"kind": ROW_REL, "key": graph["keys"][index], "value": graph["keys"][(index + 1) % NODES]} for index in range(NODES)] + [{"kind": ROW_PAIR, "key": graph["keys"][index], "value": next(row["value"] for row in rows if row["kind"] == ROW_PAIR and row["key"] == graph["keys"][index])} for index in range(NODES)]
        rows_permuted += int(rows != logical_rows)
    return {"complete_cycle_graphs": complete_cycles, "pair_value_unique_graphs": unique_pair_values, "episode_goals_with_outgoing_rel": goals_with_outgoing, "rows_physically_permuted_graphs": rows_permuted, "all_row_types_only_rel_pair": all(row["kind"] in {ROW_REL, ROW_PAIR} for graph in manifest["graphs"] for row in graph["rows"]), "all_rows_unmasked": True, "no_terminal_or_special_rows": True}


def materialize_batch(model: C1JointModel, episodes: list[dict[str, Any]], graphs: list[dict[str, Any]]) -> tuple[Tensor, Tensor, Tensor, Tensor, list[int], list[int]]:
    rows_by_episode = [graphs[episode["graph"]]["rows"] for episode in episodes]
    memory_keys = torch.stack([model.token_embedding(torch.tensor([row["key"] + KEY_BASE for row in rows])) for rows in rows_by_episode])
    memory_values = torch.stack([model.token_embedding(torch.tensor([row["value"] + KEY_BASE if row["kind"] == ROW_REL else VALUE_BASE + row["value"] for row in rows])) for rows in rows_by_episode])
    memory_types = torch.tensor([[row["kind"] for row in rows] for rows in rows_by_episode], dtype=torch.long)
    row_mask = torch.ones_like(memory_types, dtype=torch.bool)
    rel_expected: list[int] = []
    pair_expected: list[int] = []
    for episode, rows in zip(episodes, rows_by_episode):
        rel_expected.append(next(index for index, row in enumerate(rows) if row["kind"] == ROW_REL and row["key"] == episode["start_key"]))
        pointer = rows[rel_expected[-1]]["value"]
        pair_expected.append(next(index for index, row in enumerate(rows) if row["kind"] == ROW_PAIR and row["key"] == pointer))
    return memory_keys, memory_values, memory_types, row_mask, rel_expected, pair_expected


def decode_register(model: C1JointModel, state: Tensor) -> Tensor:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    return class_ids[model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids)).argmax(-1)]


def relation_index(rows: list[dict[str, int]], key: int) -> int:
    return next(index for index, row in enumerate(rows) if row["kind"] == ROW_REL and row["key"] == key)


def pair_index(rows: list[dict[str, int]], key: int) -> int:
    return next(index for index, row in enumerate(rows) if row["kind"] == ROW_PAIR and row["key"] == key)


@torch.no_grad()
def execute_distance(model: C1JointModel, episodes: list[dict[str, Any]], graphs: list[dict[str, Any]], distance: int) -> list[dict[str, Any]]:
    memory_keys, memory_values, memory_types, row_mask, _, _ = materialize_batch(model, episodes, graphs)
    batch_size = len(episodes)
    state = torch.zeros((batch_size, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([episode["start_key"] for episode in episodes]))
    presence = torch.ones((batch_size, SLOT_COUNT), dtype=torch.bool)
    zero = immediate_vectors(model, torch.full((batch_size,), 511, dtype=torch.long))
    read_mode = torch.zeros(batch_size, dtype=torch.long)
    source_p = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    destination_p = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    destination_e = torch.full((batch_size,), SLOT_E, dtype=torch.long)
    current_symbolic = [episode["start_key"] for episode in episodes]
    traces: list[list[dict[str, Any]]] = [[] for _ in episodes]
    read_mismatch = [False for _ in episodes]
    for decision in range(DECISION_LIMIT):
        active = decision < distance
        if not active:
            break
        expected_rows = [relation_index(graphs[episode["graph"]]["rows"], current_symbolic[item]) for item, episode in enumerate(episodes)]
        before = state.clone()
        state, _, read = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_P"], dtype=torch.long), zero, source_p, destination_p, presence, read_mode="BLEND", read_set="explicit")
        selected = read.selected_index.tolist()
        for item, episode in enumerate(episodes):
            match = int(selected[item]) == expected_rows[item]
            read_mismatch[item] |= not match
            traces[item].append({"decision": decision, "action": "ADVANCE", "instruction": "READ_P", "symbolic_pointer": current_symbolic[item], "goal_key": episode["goal_key"], "selected_row": int(selected[item]), "expected_row": expected_rows[item], "row_match": match, "attention_mass_correct": float(read.attention_soft[item, expected_rows[item]]), "selection_margin": float(read.selection_margin[item]), "payload_l2": float((read.payload[item] - memory_values[item, expected_rows[item]]).norm()), "p_changed": not torch.equal(before[item, SLOT_P], state[item, SLOT_P])})
            current_symbolic[item] = graphs[episode["graph"]]["rows"][expected_rows[item]]["value"]
    expected_collect_rows = [pair_index(graphs[episode["graph"]]["rows"], episode["goal_key"]) for episode in episodes]
    before_collect = state.clone()
    state, _, read_e = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_E"], dtype=torch.long), zero, source_p, destination_e, presence, read_mode="BLEND", read_set="explicit", diagnostic_read_e_select=False)
    selected_e = read_e.selected_index.tolist()
    for item, episode in enumerate(episodes):
        match = int(selected_e[item]) == expected_collect_rows[item]
        read_mismatch[item] |= not match
        traces[item].append({"decision": distance, "action": "COLLECT", "instruction": "READ_E", "symbolic_pointer": current_symbolic[item], "goal_key": episode["goal_key"], "selected_row": int(selected_e[item]), "expected_row": expected_collect_rows[item], "row_match": match, "attention_mass_correct": float(read_e.attention_soft[item, expected_collect_rows[item]]), "selection_margin": float(read_e.selection_margin[item]), "payload_l2": float((read_e.payload[item] - memory_values[item, expected_collect_rows[item]]).norm()), "p_unchanged": torch.equal(before_collect[item, SLOT_P], state[item, SLOT_P])})
    before_copy = state.clone()
    state[:, SLOT_R] = state[:, SLOT_E].clone()
    copy_state = state.clone()
    copy_equal = torch.equal(copy_state[:, SLOT_R], copy_state[:, SLOT_E])
    before_alu = state.clone()
    add_one = immediate_vectors(model, torch.full((batch_size,), VALUE_BASE + 1, dtype=torch.long))
    state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["ALU_ADD"], dtype=torch.long), add_one, torch.full((batch_size,), SLOT_R, dtype=torch.long), torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=read_mode, read_set="explicit")
    decoded = decode_register(model, state).tolist()
    before_emit = state.clone()
    state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["EMIT"], dtype=torch.long), zero, source_p, torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=read_mode, read_set="explicit")
    final_decoded = decode_register(model, state).tolist()
    results: list[dict[str, Any]] = []
    for item, episode in enumerate(episodes):
        target = int(episode["target_id"])
        alu_exact = int(decoded[item]) == target
        final_exact = int(final_decoded[item]) == target
        alu_slots_conserved = torch.equal(before_alu[item, SLOT_P], state[item, SLOT_P]) and torch.equal(before_alu[item, SLOT_E], state[item, SLOT_E]) and torch.equal(before_alu[item, SLOT_W], state[item, SLOT_W])
        traces[item].append({"instruction": "COPY", "copy_equal": bool(torch.equal(copy_state[item, SLOT_R], copy_state[item, SLOT_E])), "r_equals_e": bool(torch.equal(copy_state[item, SLOT_E], copy_state[item, SLOT_R]))})
        traces[item].append({"instruction": "ALU_ADD", "decoded_r_id": int(decoded[item]), "expected_r_id": target, "exact": alu_exact, "p_e_w_conserved": alu_slots_conserved})
        traces[item].append({"instruction": "EMIT", "decoded_r_id": int(final_decoded[item]), "expected_r_id": target, "exact": final_exact, "state_unchanged": torch.equal(before_emit[item], state[item])})
        results.append({"episode": int(episode["episode"]), "graph": int(episode["graph"]), "distance": distance, "start_key": int(episode["start_key"]), "goal_key": int(episode["goal_key"]), "target_id": target, "predicted_id": int(final_decoded[item]), "final_exact": final_exact, "alu_exact": alu_exact, "episode_success": final_exact and alu_exact and not read_mismatch[item], "decision_count": distance + 1, "decision_limit": DECISION_LIMIT, "read_mismatch": read_mismatch[item], "copy_equal": copy_equal, "trace": traces[item]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "campaign" / "u0c_ctrl1_preflight_seed101_frozen" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_ctrl1_preflight_seed101_frozen" / "seed101")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = make_manifest(args.seed)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_manifest(manifest)
    model = load_approved_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    results: list[dict[str, Any]] = []
    for distance in DISTANCES:
        distance_episodes = [episode for episode in manifest["episodes"] if episode["distance"] == distance]
        results.extend(execute_distance(model, distance_episodes, manifest["graphs"], distance))
    by_distance: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [result for result in results if result["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "episode_success_count": sum(result["episode_success"] for result in selected), "final_exact_count": sum(result["final_exact"] for result in selected), "alu_exact_count": sum(result["alu_exact"] for result in selected), "read_mismatch_count": sum(result["read_mismatch"] for result in selected), "copy_equal_count": sum(result["copy_equal"] for result in selected)}
    failing = [distance for distance in DISTANCES if by_distance[str(distance)]["episode_success_count"] < GRAPH_COUNT]
    summary = {"status": "completed", "policy": "oracle: ADVANCE iff symbolic_pointer != goal; COLLECT iff equal", "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "diagnostic_read_e_select": False, "graphs": GRAPH_COUNT, "episodes": len(results), "memory_rows": MEMORY_ROWS, "rows_per_type": {"REL": NODES, "PAIR": NODES}, "distances": list(DISTANCES), "decision_limit": DECISION_LIMIT, "generator_checks": manifest_checks(manifest), "by_distance": by_distance, "first_failing_distance": min(failing) if failing else None, "target_source": "independent cycle/pair interpreter; no target/state reinjection"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
