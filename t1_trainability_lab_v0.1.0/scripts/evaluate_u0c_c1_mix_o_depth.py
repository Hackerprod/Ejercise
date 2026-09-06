"""Frozen T1-MIX-O depth/operation-variety generalization evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    TRANSFORM_OPS,
    VALUE_BASE,
    VALUE_COUNT,
    C1JointModel,
    apply_operation,
    immediate_vectors,
    load_approved_model,
)
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


DEPTHS = (2, 3, 4, 6, 8, 16)
PROGRAMS_PER_DEPTH = 256
DISTRACTOR_KEYS = 8
IMM_ZERO = 511
OLD_PATTERNS = {
    ("ALU_SUB", "ALU_MUL"),
    ("ALU_MUL", "ALU_SUB"),
    ("ALU_ADD", "ALU_MUL"),
    ("ALU_MUL", "ALU_ADD"),
    ("ALU_SUB", "ALU_MUL", "ALU_ADD"),
    ("ALU_MUL", "ALU_SUB", "ALU_ADD"),
    ("ALU_ADD", "ALU_MUL", "ALU_SUB"),
    ("ALU_MUL", "ALU_ADD", "ALU_SUB"),
}


@dataclass(frozen=True)
class DepthRow:
    kind: int
    key: int
    value: int


@dataclass(frozen=True)
class DepthProgram:
    index: int
    depth: int
    operations: tuple[str, ...]
    operands: tuple[int, ...]
    start_key: int
    pair_value: int
    rows: tuple[DepthRow, ...]
    dependency_witness: bool


def apply_chain(operations: tuple[str, ...], operands: tuple[int, ...], initial: int) -> tuple[list[int], int]:
    value = initial
    states = [value]
    for operation, operand in zip(operations, operands):
        value = apply_operation(operation, value, operand)
        states.append(value)
    return states, value


def make_programs(seed: int) -> list[DepthProgram]:
    rng = random.Random(seed)
    programs: list[DepthProgram] = []
    index = 0
    for depth in DEPTHS:
        used: set[tuple[tuple[str, ...], tuple[int, ...], int]] = set()
        for item in range(PROGRAMS_PER_DEPTH):
            witness_requested = item < PROGRAMS_PER_DEPTH // 2
            for _ in range(64):
                operations = tuple(rng.choice(TRANSFORM_OPS) for _ in range(depth))
                if depth in (2, 3) and operations in OLD_PATTERNS:
                    continue
                operands = tuple(rng.randrange(1, VALUE_COUNT) for _ in range(depth))
                if witness_requested:
                    operands = tuple(operand if operation != "ALU_MUL" else (operand | 1) for operation, operand in zip(operations, operands))
                elif "ALU_MUL" in operations:
                    mul_index = operations.index("ALU_MUL")
                    if operands[mul_index] % 2:
                        operands = operands[:mul_index] + (operands[mul_index] - 1 or 2,) + operands[mul_index + 1 :]
                start_key, value_key, *distractors = rng.sample(range(256), 2 + DISTRACTOR_KEYS)
                pair_value = rng.randrange(VALUE_COUNT)
                signature = (operations, operands, pair_value)
                if signature in used:
                    continue
                used.add(signature)
                rows = [DepthRow(ROW_REL, start_key, value_key), DepthRow(ROW_PAIR, value_key, pair_value)]
                for key in distractors:
                    rows.extend((DepthRow(ROW_REL, key, start_key), DepthRow(ROW_PAIR, key, rng.randrange(VALUE_COUNT))))
                rng.shuffle(rows)
                programs.append(DepthProgram(index, depth, operations, operands, start_key, pair_value, tuple(rows), witness_requested))
                index += 1
                break
            else:
                raise RuntimeError(f"could not generate depth-{depth} program {item} within bounded attempts")
    return programs


def materialize_batch(model: C1JointModel, programs: list[DepthProgram]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    memory_keys = torch.stack([model.token_embedding(torch.tensor([row.key + KEY_BASE for row in program.rows])) for program in programs])
    memory_values = torch.stack([model.token_embedding(torch.tensor([row.value + KEY_BASE if row.kind == ROW_REL else VALUE_BASE + row.value for row in program.rows])) for program in programs])
    memory_types = torch.tensor([[row.kind for row in program.rows] for program in programs], dtype=torch.long)
    row_mask = torch.ones_like(memory_types, dtype=torch.bool)
    return memory_keys, memory_values, memory_types, row_mask


def decode_register(model: C1JointModel, state: Tensor) -> Tensor:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    return class_ids[model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids)).argmax(-1)]


def choose_intervention(program: DepthProgram, intervention_index: int) -> tuple[int, int]:
    states, baseline_target = apply_chain(program.operations, program.operands, program.pair_value)
    original = states[intervention_index + 1]
    suffix_operations = program.operations[intervention_index + 1 :]
    suffix_operands = program.operands[intervention_index + 1 :]
    for offset in range(1, VALUE_COUNT):
        candidate = (original + offset) % VALUE_COUNT
        _, candidate_target = apply_chain(suffix_operations, suffix_operands, candidate)
        if candidate_target != baseline_target:
            return candidate, candidate_target
    return (original + 1) % VALUE_COUNT, baseline_target


@torch.no_grad()
def execute_batch(model: C1JointModel, programs: list[DepthProgram], *, intervention: bool) -> list[dict[str, Any]]:
    memory_keys, memory_values, memory_types, row_mask = materialize_batch(model, programs)
    batch_size = len(programs)
    state = torch.zeros((batch_size, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([program.start_key for program in programs]))
    presence = torch.ones((batch_size, SLOT_COUNT), dtype=torch.bool)
    zero_immediate = immediate_vectors(model, torch.full((batch_size,), IMM_ZERO, dtype=torch.long))
    read_mode = torch.zeros(batch_size, dtype=torch.long)
    read_p_source = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    read_p_destination = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    read_e_destination = torch.full((batch_size,), SLOT_E, dtype=torch.long)
    state, _, read_p_result = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_P"], dtype=torch.long), zero_immediate, read_p_source, read_p_destination, presence, read_mode=read_mode, read_set="explicit")
    state, _, read_e_result = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_E"], dtype=torch.long), zero_immediate, read_p_source, read_e_destination, presence, read_mode=read_mode, read_set="explicit")
    state[:, SLOT_R] = state[:, SLOT_E].clone()
    intervention_index = programs[0].depth // 2 - 1
    intervention_values = [choose_intervention(program, intervention_index)[0] for program in programs]
    expected_values: list[list[int]] = []
    for program, intervention_value in zip(programs, intervention_values):
        states, _ = apply_chain(program.operations, program.operands, program.pair_value)
        if intervention:
            states = states[: intervention_index + 1] + [intervention_value]
            value = intervention_value
            for operation, operand in zip(program.operations[intervention_index + 1 :], program.operands[intervention_index + 1 :]):
                value = apply_operation(operation, value, operand)
                states.append(value)
        expected_values.append(states)
    decoded_by_alu: list[list[int]] = [[] for _ in programs]
    conservation_by_alu: list[list[bool]] = [[] for _ in programs]
    for round_index in range(programs[0].depth):
        before = state.clone()
        opcodes = torch.tensor([OPCODE_IDS[program.operations[round_index]] for program in programs], dtype=torch.long)
        operand_ids = torch.tensor([VALUE_BASE + program.operands[round_index] for program in programs], dtype=torch.long)
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcodes, immediate_vectors(model, operand_ids), read_p_source, torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=read_mode, read_set="explicit")
        if intervention and round_index == intervention_index:
            state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + value for value in intervention_values]))
        decoded = decode_register(model, state).tolist()
        for item, value in enumerate(decoded):
            decoded_by_alu[item].append(int(value))
            conservation_by_alu[item].append(bool(torch.equal(before[item, SLOT_P], state[item, SLOT_P]) and torch.equal(before[item, SLOT_E], state[item, SLOT_E]) and torch.equal(before[item, SLOT_W], state[item, SLOT_W])))
    state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["EMIT"], dtype=torch.long), zero_immediate, read_p_source, torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=read_mode, read_set="explicit")
    final_decoded = decode_register(model, state).tolist()
    results: list[dict[str, Any]] = []
    for item, program in enumerate(programs):
        expected_ids = [VALUE_BASE + value for value in expected_values[item][1:]]
        expected_final = expected_ids[-1]
        actual = decoded_by_alu[item]
        mismatches = [index for index, (predicted, expected) in enumerate(zip(actual, expected_ids)) if predicted != expected]
        results.append({"program": program.index, "depth": program.depth, "dependency_witness": program.dependency_witness, "operations": list(program.operations), "operands": list(program.operands), "target_id": expected_final, "predicted_id": int(final_decoded[item]), "exact_hit": int(final_decoded[item]) == expected_final, "intermediate_exact": not mismatches, "decoded_r_ids_after_each_alu": actual, "expected_r_ids_after_each_alu": expected_ids, "first_bad_alu": mismatches[0] if mismatches else None, "all_alu_slots_conserved": all(conservation_by_alu[item]), "read_p_selected_row": int(read_p_result.selected_index[item]), "read_e_selected_row": int(read_e_result.selected_index[item]), "intervention_index": intervention_index, "intervention_value": intervention_values[item] if intervention else None})
    return results


def summarize(results: list[dict[str, Any]], programs: list[DepthProgram]) -> dict[str, Any]:
    by_depth: dict[str, Any] = {}
    for depth in DEPTHS:
        selected = [result for result in results if result["depth"] == depth]
        witness = [result for result in selected if result["dependency_witness"]]
        general = [result for result in selected if not result["dependency_witness"]]
        by_depth[str(depth)] = {"samples": len(selected), "exact_count": sum(result["exact_hit"] for result in selected), "intermediate_exact_count": sum(result["intermediate_exact"] for result in selected), "all_alu_slot_conservation_count": sum(result["all_alu_slots_conserved"] for result in selected), "dependency_witness": {"samples": len(witness), "exact_count": sum(result["exact_hit"] for result in witness), "intermediate_exact_count": sum(result["intermediate_exact"] for result in witness)}, "general_arithmetic": {"samples": len(general), "exact_count": sum(result["exact_hit"] for result in general), "intermediate_exact_count": sum(result["intermediate_exact"] for result in general)}}
    failures = [depth for depth in DEPTHS if by_depth[str(depth)]["intermediate_exact_count"] < by_depth[str(depth)]["samples"]]
    return {"by_depth": by_depth, "first_failing_length": min(failures) if failures else None, "programs": len(programs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_mix_o_depth_seed101_frozen")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    programs = make_programs(args.seed)
    manifest = {"seed": args.seed, "programs_per_depth": PROGRAMS_PER_DEPTH, "depths": DEPTHS, "memory_rows": 2 + DISTRACTOR_KEYS * 2, "programs": [asdict(program) for program in programs]}
    manifest["programs"] = [{**item, "operations": list(item["operations"]), "operands": list(item["operands"]), "rows": [dict(row) for row in item["rows"]]} for item in manifest["programs"]]
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_bytes(manifest_bytes + b"\n")
    model = load_approved_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    results: list[dict[str, Any]] = []
    intervention_results: list[dict[str, Any]] = []
    for depth in DEPTHS:
        batch = [program for program in programs if program.depth == depth]
        results.extend(execute_batch(model, batch, intervention=False))
        intervention_results.extend(execute_batch(model, batch, intervention=True))
    baseline_summary = summarize(results, programs)
    intervention_summary = summarize(intervention_results, programs)
    by_program = {result["program"]: result for result in results}
    intervention_by_program = {result["program"]: result for result in intervention_results}
    summary = {"status": "completed", "seed": args.seed, "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "manifest_sha256": manifest_sha256, "read_set": "explicit", "memory_rows": 2 + DISTRACTOR_KEYS * 2, "baseline": baseline_summary, "mid_intervention": {"summary": intervention_summary, "exact_against_alternative_count": sum(result["exact_hit"] for result in intervention_results), "target_changed_count": sum(by_program[index]["target_id"] != intervention_by_program[index]["target_id"] for index in by_program), "target_changed_witness_count": sum(by_program[index]["target_id"] != intervention_by_program[index]["target_id"] for index in by_program if by_program[index]["dependency_witness"])}, "target_source": "independent symbolic interpreter; no target/state reinjection"}
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for baseline, intervention in zip(sorted(results, key=lambda result: result["program"]), sorted(intervention_results, key=lambda result: result["program"])):
            stream.write(json.dumps({"baseline": baseline, "mid_intervention": intervention}, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
