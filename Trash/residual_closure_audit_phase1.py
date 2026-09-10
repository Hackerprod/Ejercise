"""Phase 1 residual-closure audit sidecar.

This file is intentionally standalone and read-only with respect to MRDL
production source and the frozen checkpoint. It rehydrates the primary VSA
exactly once over train transitions, then captures the production
``_candidate_signal`` before and after each bind. Test transitions use isolated
copies of that train state and never mutate the primary replay.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path("/root/mrdl")
CHECKPOINT = ROOT / "results/phase_e_stage1_random_frozen.checkpoint.json"
CORPUS = ROOT / "data/TinyStories-valid.txt"
TRAIN_DUMP = ROOT / "results/residual_closure_audit_phase1_train.jsonl.gz"
SUMMARY = ROOT / "results/residual_closure_audit_phase1_summary.json"
CHECKPOINT_SHA = "47e3f089e1717cbbede3097fe4aafbd98c6c4d2155e06a1122c40d8420ad3940"
EXPECTED_TRAIN_TRANSITIONS = 56049
EXPECTED_TEST_TRANSITIONS = 13951
EXPECTED_MODELED_EDGES = 3501
SPLIT_SEED = 1729
STORY_COUNT = 10000
STORY_TOKEN_LIMIT = 8
PROHIBITED_PHASES = (
    "level_3_affine_local_vs_tfa",
    "two_hop_rollout",
    "rescue_oracle",
)
RUN_PROHIBITED_PHASES = False
SIGNAL_DIMENSION = 10
SIGNAL_COMPONENTS = {
    "depth": 0,
    "continuation": 1,
    "cleanup": 2,
    "role": 3,
    "bindings": 4,
    "route_signature": 5,
    "coverage": 6,
    "operator": 7,
    "score": 8,
    "bias": 9,
}
REPRESENTATIONS = {
    "full": tuple(range(SIGNAL_DIMENSION)),
    "no_route_signature": tuple(index for index in range(SIGNAL_DIMENSION) if index != SIGNAL_COMPONENTS["route_signature"]),
    "no_continuation": tuple(index for index in range(SIGNAL_DIMENSION) if index != SIGNAL_COMPONENTS["continuation"]),
    "no_bindings": tuple(index for index in range(SIGNAL_DIMENSION) if index != SIGNAL_COMPONENTS["bindings"]),
}

sys.path.insert(0, str(ROOT))

from benchmarks.phase_e_stage import load_pipeline_checkpoint, load_stage_stories, split_stage_stories
from mrdl.evidence import MemoryLevel
from mrdl.lanes import PropagationCandidate


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def float64_tuple(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != SIGNAL_DIMENSION:
        raise AssertionError(f"_candidate_signal dimension changed: {len(result)}")
    return result


def float64_hash(values: Sequence[float]) -> str:
    return sha256_bytes(b"".join(struct.pack("<d", float(value)) for value in values))


def binding_hash(vsa: Any, role: str) -> str:
    bundle = vsa._bundles.get(role)
    payload = {
        "role": role,
        "dimension": int(vsa.dimension),
        "seed": int(vsa.seed),
        "values": None if bundle is None else tuple(int(value) for value in bundle.values),
    }
    return canonical_hash(payload)


def open_expectation_payload(pipeline: Any, source: str) -> dict[str, Any]:
    targets = sorted(target for (edge_source, target) in pipeline.edge_memory.edges if edge_source == source)
    return {"source": source, "targets": targets, "count": len(targets)}


def open_expectation_hash(pipeline: Any, source: str) -> str:
    return canonical_hash(open_expectation_payload(pipeline, source))


def route_signature(candidate: Any) -> str:
    return canonical_hash(
        {
            "route": tuple(candidate.route),
            "member_routes": tuple(tuple(route) for route in candidate.member_routes),
            "route_record_ids": tuple(candidate.route_record_ids),
        }
    )


def provenance_hash(
    story_id: str,
    timestep: int,
    edge_id: str,
    source: str,
    destination: str,
    candidate: Any,
) -> str:
    return canonical_hash(
        {
            "story_id": story_id,
            "timestep": timestep,
            "edge_record_id": edge_id,
            "source_token_id": source,
            "destination_token_id": destination,
            "provenance_roots": tuple(candidate.provenance_roots),
            "route": tuple(candidate.route),
        }
    )


def edge_records(pipeline: Any) -> dict[tuple[str, str], dict[str, Any]]:
    return dict(pipeline.edge_memory.edges)


def lane_candidates(pipeline: Any) -> dict[str, Any]:
    full, _ = pipeline._ensure_engines()
    records = {}
    for record in full.records:
        records[str(record.record_id)] = record
    return records


def make_candidate(record: Any) -> PropagationCandidate:
    return PropagationCandidate(
        record.node_id,
        record.base_score,
        record.operator,
        None,
        (record.node_id, record.record_id),
    )


def context_seeds(story: Sequence[str], timestep: int, context_window: int) -> tuple[str, ...]:
    return tuple(story[:timestep][-context_window:])


def capture_signal(
    pipeline: Any,
    candidate: Any,
    story: Sequence[str],
    timestep: int,
) -> tuple[float, ...]:
    seeds = context_seeds(story, timestep, pipeline.context_window)
    signal = pipeline._candidate_signal(1, candidate, seeds)
    return float64_tuple(signal)


def make_row(
    pipeline: Any,
    story_id: str,
    story: Sequence[str],
    timestep: int,
    edge: dict[str, Any],
    candidate: Any,
    residual_in: Sequence[float],
    residual_out: Sequence[float],
    binding_in: str,
    binding_out: str,
    expectation_in: str,
    expectation_out: str,
) -> dict[str, Any]:
    source = story[timestep - 1]
    destination = story[timestep]
    route_sig = route_signature(candidate)
    provenance = provenance_hash(story_id, timestep, str(edge["record_id"]), source, destination, candidate)
    mode_id = edge.get("mode_id")
    return {
        "story_id": story_id,
        "timestep": timestep,
        "edge_record_id": str(edge["record_id"]),
        "mode_id": mode_id,
        "source_token_id": source,
        "destination_token_id": destination,
        "destination_state": destination,
        "memory_level": MemoryLevel(edge["level"]).value,
        "residual_in_full": list(residual_in),
        "residual_out_full": list(residual_out),
        "components": {
            "in": {
                "bindings": float(residual_in[SIGNAL_COMPONENTS["bindings"]]),
                "continuation": float(residual_in[SIGNAL_COMPONENTS["continuation"]]),
                "route_signature": route_sig,
                "route_scalar": float(residual_in[SIGNAL_COMPONENTS["route_signature"]]),
            },
            "out": {
                "bindings": float(residual_out[SIGNAL_COMPONENTS["bindings"]]),
                "continuation": float(residual_out[SIGNAL_COMPONENTS["continuation"]]),
                "route_signature": route_sig,
                "route_scalar": float(residual_out[SIGNAL_COMPONENTS["route_signature"]]),
            },
        },
        "bindings_in": float(residual_in[SIGNAL_COMPONENTS["bindings"]]),
        "bindings_out": float(residual_out[SIGNAL_COMPONENTS["bindings"]]),
        "continuation_in": float(residual_in[SIGNAL_COMPONENTS["continuation"]]),
        "continuation_out": float(residual_out[SIGNAL_COMPONENTS["continuation"]]),
        "route_signature_in": route_sig,
        "route_signature_out": route_sig,
        "route_scalar_in": float(residual_in[SIGNAL_COMPONENTS["route_signature"]]),
        "route_scalar_out": float(residual_out[SIGNAL_COMPONENTS["route_signature"]]),
        "binding_hash_in": binding_in,
        "binding_hash_out": binding_out,
        "open_expectation_hash_in": expectation_in,
        "open_expectation_hash_out": expectation_out,
        "open_expectation_count": open_expectation_payload(pipeline, source)["count"],
        "provenance_hash": provenance,
        "quantized_residual_in_hash": float64_hash(residual_in),
        "quantized_residual_out_hash": float64_hash(residual_out),
    }


def replay_train(
    pipeline: Any,
    train_records: Sequence[dict[str, Any]],
    edge_map: dict[tuple[str, str], dict[str, Any]],
    candidates: dict[str, Any],
    output_path: Path,
) -> tuple[list[tuple[Any, ...]], int]:
    examples: list[tuple[Any, ...]] = []
    bind_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as output:
        for record in train_records:
            story_id = str(record["story_id"])
            story = tuple(record["tokens"])
            for timestep in range(1, len(story)):
                source, destination = story[timestep - 1], story[timestep]
                edge = edge_map.get((source, destination))
                if edge is None:
                    raise AssertionError(f"train transition absent from checkpoint: {source}->{destination}")
                edge_id = str(edge["record_id"])
                candidate = candidates[edge_id]
                binding_in = binding_hash(pipeline.vsa_memory, source)
                expectation_in = open_expectation_hash(pipeline, source)
                residual_in = capture_signal(pipeline, candidate, story, timestep)
                pipeline.vsa_memory.bind(source, destination)
                bind_count += 1
                binding_out = binding_hash(pipeline.vsa_memory, source)
                expectation_out = open_expectation_hash(pipeline, source)
                residual_out = capture_signal(pipeline, candidate, story, timestep)
                row = make_row(
                    pipeline, story_id, story, timestep, edge, candidate,
                    residual_in, residual_out, binding_in, binding_out,
                    expectation_in, expectation_out,
                )
                output.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
                examples.append(
                    (
                        story_id,
                        edge_id,
                        row["mode_id"],
                        destination,
                        residual_in,
                        residual_out,
                        row["binding_hash_out"],
                        row["open_expectation_hash_out"],
                    )
                )
    return examples, bind_count


def load_train_dump(path: Path) -> list[tuple[Any, ...]]:
    examples: list[tuple[Any, ...]] = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            residual_in = float64_tuple(row["residual_in_full"])
            residual_out = float64_tuple(row["residual_out_full"])
            examples.append(
                (
                    str(row["story_id"]),
                    str(row["edge_record_id"]),
                    row.get("mode_id"),
                    str(row["destination_token_id"]),
                    residual_in,
                    residual_out,
                    str(row["binding_hash_out"]),
                    str(row["open_expectation_hash_out"]),
                )
            )
    return examples


def replay_train_state(
    pipeline: Any,
    train_records: Sequence[dict[str, Any]],
) -> int:
    bind_count = 0
    for record in train_records:
        story = tuple(record["tokens"])
        for timestep in range(1, len(story)):
            pipeline.vsa_memory.bind(story[timestep - 1], story[timestep])
            bind_count += 1
    return bind_count


def level1_metrics(examples: Sequence[tuple[Any, ...]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], Counter[tuple[Any, ...]]] = defaultdict(Counter)
    for story_id, edge_id, mode_id, destination, residual_in, residual_out, binding_out, expectation_out in examples:
        group_key = (edge_id, mode_id, residual_in)
        output_key = (destination, binding_out, expectation_out, residual_out)
        groups[group_key][output_key] += 1
    singleton = [counter for counter in groups.values() if sum(counter.values()) == 1]
    repeated = [counter for counter in groups.values() if sum(counter.values()) >= 2]
    identical = [counter for counter in repeated if len(counter) == 1]
    collisions = [counter for counter in repeated if len(counter) >= 2]
    total = len(examples)
    correct_mass = sum(max(counter.values()) for counter in groups.values())
    collision_samples = []
    for group_key, counter in sorted(groups.items(), key=lambda item: (-(sum(item[1].values())), repr(item[0])))[:10]:
        if sum(counter.values()) >= 2 and len(counter) >= 2:
            edge_id, mode_id, residual_in = group_key
            collision_samples.append(
                {
                    "edge_record_id": edge_id,
                    "mode_id": mode_id,
                    "group_count": sum(counter.values()),
                    "distinct_outputs": len(counter),
                    "quantized_residual_in_hash": float64_hash(residual_in),
                }
            )
    return {
        "grouping": ["edge_record_id", "mode_id", "exact_binary64_residual_in_full"],
        "output_identity": ["destination_state", "binding_hash_out", "open_expectation_hash_out", "exact_binary64_residual_out_full"],
        "observations": total,
        "groups": len(groups),
        "a_star_det": correct_mass / max(1, total),
        "singleton_groups": len(singleton),
        "singleton_observations": sum(sum(counter.values()) for counter in singleton),
        "repeated_groups": len(repeated),
        "repeated_observations": sum(sum(counter.values()) for counter in repeated),
        "repeated_identical_output_groups": len(identical),
        "repeated_identical_output_observations": sum(sum(counter.values()) for counter in identical),
        "collision_groups_ge_2_distinct_outputs": len(collisions),
        "collision_observations": sum(sum(counter.values()) for counter in collisions),
        "max_group_size": max((sum(counter.values()) for counter in groups.values()), default=0),
        "collision_samples": collision_samples,
    }


def collect_test_examples(
    pipeline: Any,
    train_vsa: Any,
    test_records: Sequence[dict[str, Any]],
    edge_map: dict[tuple[str, str], dict[str, Any]],
    candidates: dict[str, Any],
) -> tuple[list[tuple[Any, ...]], int, int, int]:
    test_pipeline = copy.copy(pipeline)
    examples: list[tuple[Any, ...]] = []
    bind_count = 0
    transition_count = 0
    absent_edge_count = 0
    for record in test_records:
        test_pipeline.vsa_memory = copy.deepcopy(train_vsa)
        story_id = str(record["story_id"])
        story = tuple(record["tokens"])
        for timestep in range(1, len(story)):
            transition_count += 1
            source, destination = story[timestep - 1], story[timestep]
            edge = edge_map.get((source, destination))
            if edge is None:
                test_pipeline.vsa_memory.bind(source, destination)
                bind_count += 1
                absent_edge_count += 1
                continue
            edge_id = str(edge["record_id"])
            candidate = candidates[edge_id]
            residual_in = capture_signal(test_pipeline, candidate, story, timestep)
            test_pipeline.vsa_memory.bind(source, destination)
            bind_count += 1
            residual_out = capture_signal(test_pipeline, candidate, story, timestep)
            examples.append(
                (
                    story_id,
                    edge_id,
                    edge.get("mode_id"),
                    destination,
                    residual_in,
                    residual_out,
                    binding_hash(test_pipeline.vsa_memory, source),
                    open_expectation_hash(test_pipeline, source),
                )
            )
    return examples, bind_count, transition_count, absent_edge_count


def nearest_neighbor_metrics(
    train_examples: Sequence[tuple[Any, ...]],
    test_examples: Sequence[tuple[Any, ...]],
) -> dict[str, Any]:
    train_by_group: dict[tuple[Any, ...], list[tuple[int, tuple[Any, ...]]]] = defaultdict(list)
    for index, example in enumerate(train_examples):
        train_by_group[(example[1], example[2])].append((index, example))
    unique_train_by_representation: dict[str, dict[tuple[Any, ...], list[tuple[int, tuple[Any, ...]]]]] = {}
    for representation, dimensions in REPRESENTATIONS.items():
        unique_groups: dict[tuple[Any, ...], list[tuple[int, tuple[Any, ...]]]] = defaultdict(list)
        seen_inputs: set[tuple[Any, ...]] = set()
        for index, example in enumerate(train_examples):
            group = (example[1], example[2])
            input_key = (group, tuple(example[4][dimension] for dimension in dimensions))
            if input_key not in seen_inputs:
                seen_inputs.add(input_key)
                unique_groups[group].append((index, example))
        unique_train_by_representation[representation] = unique_groups
    results: dict[str, Any] = {}
    for name, dimensions in REPRESENTATIONS.items():
        sse_nn = 0.0
        sse_mean = 0.0
        eligible = 0
        missing = 0
        zero_distance = 0
        zero_denominator_rows = 0
        means: dict[tuple[Any, ...], tuple[float, ...]] = {}
        for example in test_examples:
            group = (example[1], example[2])
            all_candidates = train_by_group.get(group)
            candidates = unique_train_by_representation[name].get(group)
            if not candidates or not all_candidates:
                missing += 1
                continue
            if group not in means:
                outputs = [candidate[1][5] for candidate in all_candidates]
                means[group] = tuple(sum(output[index] for output in outputs) / len(outputs) for index in range(SIGNAL_DIMENSION))
            test_input = example[4]
            test_output = example[5]
            best_full = (math.inf, -1, None)
            best_no_route = (math.inf, -1, None)
            best_no_continuation = (math.inf, -1, None)
            best_no_bindings = (math.inf, -1, None)
            for train_index, candidate in candidates:
                train_input = candidate[4]
                full_distance = 0.0
                dropped_route = 0.0
                dropped_continuation = 0.0
                dropped_bindings = 0.0
                for index in range(SIGNAL_DIMENSION):
                    difference = test_input[index] - train_input[index]
                    squared = difference * difference
                    full_distance += squared
                    if index == SIGNAL_COMPONENTS["route_signature"]:
                        dropped_route = squared
                    elif index == SIGNAL_COMPONENTS["continuation"]:
                        dropped_continuation = squared
                    elif index == SIGNAL_COMPONENTS["bindings"]:
                        dropped_bindings = squared
                no_route_distance = full_distance - dropped_route
                no_continuation_distance = full_distance - dropped_continuation
                no_bindings_distance = full_distance - dropped_bindings
                if full_distance < best_full[0] or (full_distance == best_full[0] and train_index < best_full[1]):
                    best_full = (full_distance, train_index, candidate)
                if no_route_distance < best_no_route[0] or (no_route_distance == best_no_route[0] and train_index < best_no_route[1]):
                    best_no_route = (no_route_distance, train_index, candidate)
                if no_continuation_distance < best_no_continuation[0] or (no_continuation_distance == best_no_continuation[0] and train_index < best_no_continuation[1]):
                    best_no_continuation = (no_continuation_distance, train_index, candidate)
                if no_bindings_distance < best_no_bindings[0] or (no_bindings_distance == best_no_bindings[0] and train_index < best_no_bindings[1]):
                    best_no_bindings = (no_bindings_distance, train_index, candidate)
            best = {
                "full": best_full,
                "no_route_signature": best_no_route,
                "no_continuation": best_no_continuation,
                "no_bindings": best_no_bindings,
            }
            best_distance, _, best_example = best[name]
            if best_example is None:
                raise AssertionError("nearest-neighbor candidate unexpectedly absent")
            if best_distance == 0.0:
                zero_distance += 1
            predicted_output = best_example[5]
            mean_output = means[group]
            nn_error = sum((test_output[index] - predicted_output[index]) ** 2 for index in dimensions)
            mean_error = sum((test_output[index] - mean_output[index]) ** 2 for index in dimensions)
            sse_nn += nn_error
            sse_mean += mean_error
            eligible += 1
            if mean_error <= 1e-15:
                zero_denominator_rows += 1
        gain = None if sse_mean <= 1e-15 else 1.0 - sse_nn / sse_mean
        results[name] = {
            "dimensions": list(dimensions),
            "distance": "squared Euclidean over listed binary64 _candidate_signal components",
            "eligible_test_observations": eligible,
            "missing_train_group_observations": missing,
            "zero_distance_matches": zero_distance,
            "sse_nn": sse_nn,
            "sse_to_edge_mean": sse_mean,
            "zero_denominator_rows": zero_denominator_rows,
            "g_local": gain,
            "zero_denominator_behavior": "aggregate G_local is null when aggregate SSE_to_edge_mean <= 1e-15; row-level zero denominators counted and retained",
            "tie_behavior": "exact distance ties choose lowest train observation index",
            "same_story_exclusion": "train/test story IDs are disjoint; no same-story candidate is eligible",
        }
    return {
        "protocol": {
            "candidate_source": "train only",
            "grouping": ["edge_record_id", "mode_id"],
            "prediction": "nearest train residual_in_full; paired train residual_out_full",
            "representations": "no_route_signature drops signal index 5 route scalar; no_continuation drops index 1; no_bindings drops index 4",
        },
        "train_observations": len(train_examples),
        "test_observations": len(test_examples),
        "representations": results,
    }


def filter_coverage(train_examples: Sequence[tuple[Any, ...]], modeled_edges: int) -> dict[str, Any]:
    counts = Counter(example[1] for example in train_examples)
    qualifying = {edge_id for edge_id, count in counts.items() if count >= 16}
    observations_on_qualifying = sum(count for edge_id, count in counts.items() if edge_id in qualifying)
    return {
        "modeled_edges": modeled_edges,
        "edges_with_train_observations": len(counts),
        "threshold": 16,
        "qualifying_edges": len(qualifying),
        "fraction_modeled_edges_ge_16": len(qualifying) / max(1, modeled_edges),
        "train_observations": len(train_examples),
        "train_observations_on_qualifying_edges": observations_on_qualifying,
        "fraction_train_observations_on_edges_ge_16": observations_on_qualifying / max(1, len(train_examples)),
        "filter_definition": "direct counts over every observed train transition; no dump or metric is filtered by this threshold",
    }


def metadata(pipeline: Any, checkpoint_payload: dict[str, Any], train_count: int, test_count: int) -> dict[str, Any]:
    config = checkpoint_payload["pipeline_config"]
    checkpoint_metadata = checkpoint_payload.get("metadata", {})
    return {
        "encoder_version": checkpoint_metadata.get("encoder_version", "not recorded in checkpoint metadata"),
        "checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA},
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_pipeline_config": config,
        "quantization_precision": {
            "vsa_state": "production VSAVector bipolar integer values (-1/+1)",
            "residual": "production Python float, captured as IEEE-754 binary64 without rounding",
            "grouping": "exact binary64 tuple; float64 hash is little-endian struct.pack('<d')",
            "explicit_checkpoint_quantization": "no quantization or scale metadata field recorded",
            "embedding_values": "checkpoint serialized float values, no additional transformation",
        },
        "signal_source": {
            "implementation": "mrdl.language.MRDLLanguagePipeline._candidate_signal",
            "signal_dimension": SIGNAL_DIMENSION,
            "component_indices": SIGNAL_COMPONENTS,
            "candidate": "actual checkpoint lane record for observed direct edge, depth=1, route=(source_token_id, edge_record_id)",
            "context_seeds": "same trailing context_window tokens used by production _propagation_context",
            "transition": "capture before, one real vsa_memory.bind(source,destination), capture after",
        },
        "protocol": {
            "corpus": str(CORPUS),
            "story_count": STORY_COUNT,
            "story_token_limit": STORY_TOKEN_LIMIT,
            "split_seed": SPLIT_SEED,
            "train_stories": 8007,
            "test_stories": 1993,
            "train_transitions": train_count,
            "test_transitions": test_count,
            "mode_id": "not present in checkpoint edge records; rows carry null",
            "test_vsa": "isolated deepcopy of post-train state per test story; never assigned back to primary train pipeline",
        },
        "prohibited_phases": {"declared": list(PROHIBITED_PHASES), "executed": list(PROHIBITED_PHASES) if RUN_PROHIBITED_PHASES else []},
    }


def main() -> None:
    started = time.time()
    if RUN_PROHIBITED_PHASES or PROHIBITED_PHASES:
        assert not RUN_PROHIBITED_PHASES, "prohibited Phase 1 extension enabled"
    checkpoint_bytes = CHECKPOINT.read_bytes()
    checkpoint_sha = sha256_bytes(checkpoint_bytes)
    assert checkpoint_sha == CHECKPOINT_SHA, (checkpoint_sha, CHECKPOINT_SHA)
    checkpoint_payload = json.loads(checkpoint_bytes)
    records = load_stage_stories(CORPUS, STORY_COUNT, STORY_TOKEN_LIMIT)
    train, test, manifest = split_stage_stories(records, SPLIT_SEED)
    train_records = [record for record in records if record["story_id"] in set(manifest["train_story_ids"])]
    test_records = [record for record in records if record["story_id"] in set(manifest["test_story_ids"])]
    assert len(train) == len(train_records) == 8007
    assert len(test) == len(test_records) == 1993
    assert not set(manifest["train_story_ids"]) & set(manifest["test_story_ids"])
    assert not set(record["story_id"] for record in train_records) & set(record["story_id"] for record in test_records)
    expected_test_transitions = sum(len(story) - 1 for story in test)
    assert expected_test_transitions == EXPECTED_TEST_TRANSITIONS

    pipeline = load_pipeline_checkpoint(CHECKPOINT)
    assert len(pipeline.edge_memory.edges) == EXPECTED_MODELED_EDGES
    edge_map = edge_records(pipeline)
    candidates = {edge_id: make_candidate(record) for edge_id, record in lane_candidates(pipeline).items()}
    assert set(edge["record_id"] for edge in edge_map.values()) <= set(candidates)
    controller_updates_before = pipeline.controller.updates
    edge_count_before = len(pipeline.edge_memory.edges)
    if "--reuse-train-dump" in sys.argv:
        train_examples = load_train_dump(TRAIN_DUMP)
        train_bind_count = replay_train_state(pipeline, train_records)
        replay_mode = "validated_existing_dump_plus_single_ordered_state_replay"
    else:
        train_examples, train_bind_count = replay_train(pipeline, train_records, edge_map, candidates, TRAIN_DUMP)
        replay_mode = "fresh_capture_and_single_ordered_state_replay"
    assert len(train_examples) == EXPECTED_TRAIN_TRANSITIONS
    assert train_bind_count == EXPECTED_TRAIN_TRANSITIONS
    assert pipeline.vsa_memory.binding_count == EXPECTED_TRAIN_TRANSITIONS
    assert pipeline.controller.updates == controller_updates_before
    assert len(pipeline.edge_memory.edges) == edge_count_before

    train_vsa = copy.deepcopy(pipeline.vsa_memory)
    test_rows, test_copy_bind_count, test_transition_count, test_absent_edge_count = collect_test_examples(
        pipeline, train_vsa, test_records, edge_map, candidates,
    )
    assert test_transition_count == EXPECTED_TEST_TRANSITIONS
    assert test_copy_bind_count == EXPECTED_TEST_TRANSITIONS
    assert pipeline.vsa_memory.binding_count == EXPECTED_TRAIN_TRANSITIONS
    assert pipeline.controller.updates == controller_updates_before
    assert len(pipeline.edge_memory.edges) == edge_count_before

    dump_bytes = TRAIN_DUMP.read_bytes()
    dump_sha = sha256_bytes(dump_bytes)
    with gzip.open(TRAIN_DUMP, "rt", encoding="utf-8") as dump:
        dump_lines = sum(1 for _ in dump)
    assert dump_lines == EXPECTED_TRAIN_TRANSITIONS
    level1 = level1_metrics(train_examples)
    nearest = nearest_neighbor_metrics(train_examples, test_rows)
    nearest["test_transition_count_total"] = test_transition_count
    nearest["test_transitions_absent_from_checkpoint"] = test_absent_edge_count
    coverage = filter_coverage(train_examples, len(edge_map))
    summary = {
        "diagnostic": "residual_closure_audit_phase1",
        "scope": "read-only/offline; new sidecar and artifacts only",
        "artifacts": {
            "sidecar": str(ROOT / "residual_closure_audit_phase1.py"),
            "train_dump": {"path": str(TRAIN_DUMP), "bytes": len(dump_bytes), "sha256": dump_sha, "gzip_jsonl_lines": dump_lines},
            "summary": str(SUMMARY),
        },
        "metadata": metadata(pipeline, checkpoint_payload, len(train_examples), len(test_rows)),
        "split_manifest_hash": canonical_hash(manifest),
        "level1": level1,
        "level2": nearest,
        "filter_coverage": coverage,
        "assertions": {
            "checkpoint_sha256": checkpoint_sha == CHECKPOINT_SHA,
            "exact_train_transition_count": len(train_examples) == EXPECTED_TRAIN_TRANSITIONS,
            "single_primary_replay_bind_count": train_bind_count == EXPECTED_TRAIN_TRANSITIONS,
            "test_story_exclusion": not set(manifest["train_story_ids"]) & set(manifest["test_story_ids"]),
            "primary_train_state_unchanged_by_test_copies": pipeline.vsa_memory.binding_count == EXPECTED_TRAIN_TRANSITIONS,
            "no_prohibited_phases": not RUN_PROHIBITED_PHASES,
            "no_ge16_dump_filter": dump_lines == EXPECTED_TRAIN_TRANSITIONS,
        },
        "replay": {
            "train_mode": replay_mode,
            "primary_train_bind_count": train_bind_count,
            "isolated_test_copy_bind_count": test_copy_bind_count,
            "test_transition_count": test_transition_count,
            "test_transitions_absent_from_checkpoint": test_absent_edge_count,
            "test_rows_with_modeled_edge": len(test_rows),
            "controller_updates_before_after": [controller_updates_before, pipeline.controller.updates],
            "edge_count_before_after": [edge_count_before, len(pipeline.edge_memory.edges)],
        },
        "elapsed_seconds": time.time() - started,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    summary_bytes = SUMMARY.read_bytes()
    print(json.dumps({
        "status": "phase1_complete",
        "summary": str(SUMMARY),
        "summary_sha256": sha256_bytes(summary_bytes),
        "train_dump": str(TRAIN_DUMP),
        "train_dump_sha256": dump_sha,
        "metrics": {"level1": level1, "level2": nearest, "filter_coverage": coverage},
        "replay": summary["replay"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
