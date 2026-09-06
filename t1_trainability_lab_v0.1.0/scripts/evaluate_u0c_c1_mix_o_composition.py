"""Frozen T1-MIX-O chained ALU and interleaved P->W composition gates."""

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
import torch.nn.functional as F

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
    READ_MODE_SELECT,
    ROW_PAIR,
    ROW_REL,
    ROW_VEC,
    SLOT_COUNT,
    SLOT_E,
    SLOT_P,
    SLOT_R,
    SLOT_W,
)


@dataclass(frozen=True)
class Row:
    kind: int
    key: int
    value: int | Tensor


@dataclass
class ChainProgram:
    index: int
    operations: tuple[str, ...]
    operands: tuple[int, ...]
    start_key: int
    pair_value: int
    alternate_value: int
    rows: list[Row]


@dataclass
class InterleavedProgram:
    index: int
    operations: tuple[str, str]
    operands: tuple[int, int]
    start_key: int
    pair_value: int
    evidence: tuple[Tensor, Tensor]
    transforms: tuple[int, int]
    rows: list[Row]


def make_chain_programs(per_length: int, seed: int) -> list[ChainProgram]:
    rng = random.Random(seed)
    patterns = {
        2: (("ALU_SUB", "ALU_MUL"), ("ALU_ADD", "ALU_SUB"), ("ALU_MUL", "ALU_SUB"), ("ALU_SUB", "ALU_ADD")),
        3: (("ALU_SUB", "ALU_MUL", "ALU_ADD"), ("ALU_ADD", "ALU_SUB", "ALU_MUL"), ("ALU_MUL", "ALU_SUB", "ALU_ADD"), ("ALU_SUB", "ALU_ADD", "ALU_MUL")),
    }
    programs: list[ChainProgram] = []
    index = 0
    for length in (2, 3):
        for item in range(per_length):
            operations = patterns[length][item % len(patterns[length])]
            start_key, value_key, *distractors = rng.sample(range(256), 2 + 8)
            pair_value = rng.randrange(VALUE_COUNT)
            operands = tuple(rng.randrange(1, VALUE_COUNT) for _ in operations)
            swapped = (operations[1], operations[0], *operations[2:])
            while symbolic_chain(ChainProgram(-1, operations, operands, start_key, pair_value, pair_value, []))["target_value"] == symbolic_chain(ChainProgram(-1, swapped, operands, start_key, pair_value, pair_value, []))["target_value"]:
                operands = tuple(rng.randrange(1, VALUE_COUNT) for _ in operations)
            alternate_value = (pair_value + 1) % VALUE_COUNT
            rows = [Row(ROW_REL, start_key, value_key), Row(ROW_PAIR, value_key, pair_value)]
            for key in distractors:
                rows.extend((Row(ROW_REL, key, start_key), Row(ROW_PAIR, key, rng.randrange(VALUE_COUNT))))
            rng.shuffle(rows)
            programs.append(ChainProgram(index, operations, operands, start_key, pair_value, alternate_value, rows))
            index += 1
    return programs


def make_interleaved_programs(count: int, seed: int) -> list[InterleavedProgram]:
    rng = random.Random(seed + 707)
    programs: list[InterleavedProgram] = []
    for index in range(count):
        start_key, pointer_key, second_key, *distractors = rng.sample(range(256), 3 + 8)
        pair_value = rng.randrange(VALUE_COUNT)
        operations = ("ALU_SUB", "ALU_MUL") if index % 2 == 0 else ("ALU_ADD", "ALU_SUB")
        operands = (rng.randrange(1, VALUE_COUNT), rng.randrange(1, VALUE_COUNT))
        evidence = (torch.randn(DIMENSION, generator=torch.Generator().manual_seed(seed + index * 2)), torch.randn(DIMENSION, generator=torch.Generator().manual_seed(seed + index * 2 + 1)))
        transforms = (index % 4, (index + 1) % 4)
        rows: list[Row] = [
            Row(ROW_REL, start_key, pointer_key),
            Row(ROW_PAIR, pointer_key, pair_value),
            Row(ROW_REL, pointer_key, second_key),
            Row(ROW_VEC, pointer_key, evidence[0]),
            Row(ROW_VEC, second_key, evidence[1]),
        ]
        for key in distractors:
            rows.extend((Row(ROW_REL, key, start_key), Row(ROW_VEC, key, torch.randn(DIMENSION, generator=torch.Generator().manual_seed(seed + 1000 + index * 8 + key)))))
        rng.shuffle(rows)
        programs.append(InterleavedProgram(index, operations, operands, start_key, pair_value, evidence, transforms, rows))
    return programs


def materialize(model: C1JointModel, rows: list[Row]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    keys = model.token_embedding(torch.tensor([row.key + KEY_BASE for row in rows]))
    values = torch.stack([model.token_embedding(torch.tensor(KEY_BASE + int(row.value))) if row.kind == ROW_REL else model.token_embedding(torch.tensor(VALUE_BASE + int(row.value))) if row.kind == ROW_PAIR else row.value for row in rows])
    types = torch.tensor([row.kind for row in rows], dtype=torch.long).unsqueeze(0)
    return keys, values, types, torch.ones_like(types, dtype=torch.bool)


def symbolic_chain(program: ChainProgram, *, initial_value: int | None = None, operations: tuple[str, ...] | None = None) -> dict[str, Any]:
    value = program.pair_value if initial_value is None else initial_value
    ops = operations or program.operations
    states = [value]
    for operation, operand in zip(ops, program.operands):
        value = apply_operation(operation, value, operand)
        states.append(value)
    return {"target_value": value, "states": states}


def symbolic_interleaved(program: InterleavedProgram, *, interleaved_order: bool = False) -> dict[str, Any]:
    value = program.pair_value
    workspace = torch.zeros(DIMENSION)
    state = {"P": program.start_key, "E": program.pair_value, "R": value, "W": workspace}
    trace: list[dict[str, Any]] = []
    instructions = [("ALU", 0), ("ACCUM", 0), ("READ_P", 0), ("ACCUM", 1), ("ALU", 1)] if not interleaved_order else [("ALU", 0), ("ALU", 1), ("ACCUM", 0), ("READ_P", 0), ("ACCUM", 1)]
    for kind, index in instructions:
        before = snapshot_symbolic_state(state)
        if kind == "ALU":
            state["R"] = apply_operation(program.operations[index], int(state["R"]), program.operands[index])
        elif kind == "ACCUM":
            state["W"] = state["W"] + apply_transform(program.evidence[index], program.transforms[index])
        elif kind == "READ_P":
            state["P"] = program.rows[next(row_i for row_i, row in enumerate(program.rows) if row.kind == ROW_REL and row.key == state["P"])].value
        trace.append({"instruction": kind, "index": index, "before": before, "after": snapshot_symbolic_state(state)})
    return {"target_value": int(state["R"]), "target_workspace": state["W"].tolist(), "trace": trace}


def snapshot_symbolic_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, Tensor) else value for key, value in state.items()}


def apply_transform(value: Tensor, transform: int) -> Tensor:
    if transform == 0:
        return value
    if transform == 1:
        return -value
    if transform == 2:
        return torch.roll(value, shifts=1, dims=-1)
    pairs = value.reshape(DIMENSION // 2, 2)
    return torch.stack((pairs[:, 1], -pairs[:, 0]), dim=-1).reshape(DIMENSION)


def decode_r(model: C1JointModel, state: Tensor) -> tuple[int, Tensor]:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    logits = model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids))
    return int(class_ids[logits.argmax(-1)].item()), logits.squeeze(0)


@torch.no_grad()
def run_chain(model: C1JointModel, program: ChainProgram, *, mode: str = "baseline") -> dict[str, Any]:
    memory_keys, memory_values, memory_types, row_mask = materialize(model, program.rows)
    presence = torch.tensor([[True, True, True, True]])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([program.start_key]))
    immediate_zero = torch.tensor([511])
    trace: list[dict[str, Any]] = []
    intervention_value = (program.pair_value + 3) % VALUE_COUNT
    operations = program.operations if mode != "swap_ops" else (program.operations[1], program.operations[0], *program.operations[2:])
    symbolic = symbolic_chain(program, initial_value=intervention_value if mode == "intervene_r" else None, operations=operations)
    instructions: list[tuple[str, int | None]] = [("READ_P", None), ("READ_E", None), ("COPY", None)]
    instructions.extend((operation, index) for index, operation in enumerate(operations))
    instructions.append(("EMIT", None))
    for instruction, op_index in instructions:
        before = state.clone()
        selected = -1
        payload = None
        head_logits = None
        if instruction == "COPY":
            state[:, SLOT_R] = state[:, SLOT_E].clone()
            if mode == "intervene_r":
                state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + intervention_value]))
            elif mode == "e_after_copy":
                state[:, SLOT_E] = model.token_embedding(torch.tensor([VALUE_BASE + ((program.pair_value + 5) % VALUE_COUNT)]))
        elif instruction in {"READ_P", "READ_E"}:
            opcode = torch.tensor([OPCODE_IDS[instruction]])
            destination = torch.tensor([SLOT_P if instruction == "READ_P" else SLOT_E])
            state, _, result = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate_zero, torch.tensor([SLOT_P]), destination, presence, read_mode=torch.tensor([0]), read_set="explicit")
            selected = int(result.selected_index.item())
            payload = result.payload
        elif instruction in TRANSFORM_OPS:
            opcode = torch.tensor([OPCODE_IDS[instruction]])
            operand = immediate_vectors(model, torch.tensor([VALUE_BASE + program.operands[int(op_index)]], dtype=torch.long))
            state, candidates, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, operand, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
            head_logits = candidates.alu_logits.squeeze(0)
        else:
            opcode = torch.tensor([OPCODE_IDS["EMIT"]])
            state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate_zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
        record: dict[str, Any] = {"instruction": instruction, "p": state[:, SLOT_P].squeeze(0).tolist(), "e": state[:, SLOT_E].squeeze(0).tolist(), "r": state[:, SLOT_R].squeeze(0).tolist(), "w": state[:, SLOT_W].squeeze(0).tolist(), "selected_row": selected, "p_conserved": bool(instruction == "READ_P" or torch.equal(before[:, SLOT_P], state[:, SLOT_P])), "e_conserved": bool(instruction == "READ_E" or torch.equal(before[:, SLOT_E], state[:, SLOT_E])), "r_conserved": bool(instruction in TRANSFORM_OPS + ("COPY",) or torch.equal(before[:, SLOT_R], state[:, SLOT_R])), "w_conserved": bool(torch.equal(before[:, SLOT_W], state[:, SLOT_W]) or instruction == "EMIT")}
        if payload is not None:
            record["payload"] = payload.squeeze(0).tolist()
        if head_logits is not None:
            decoded, decoder_logits = decode_r(model, state)
            record.update({"alu_head_logits": head_logits.tolist(), "decoded_r_id": decoded, "register_decoder_logits": decoder_logits.tolist()})
        trace.append(record)
    predicted, decoder_logits = decode_r(model, state)
    target = VALUE_BASE + symbolic["target_value"]
    return {"target_id": target, "predicted_id": predicted, "exact_hit": predicted == target, "trace": trace, "symbolic": symbolic, "final_decoder_logits": decoder_logits.tolist()}


@torch.no_grad()
def run_interleaved(model: C1JointModel, program: InterleavedProgram, *, moved: bool = False) -> dict[str, Any]:
    memory_keys, memory_values, memory_types, row_mask = materialize(model, program.rows)
    presence = torch.tensor([[True, True, True, True]])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([program.start_key]))
    immediate_zero = torch.tensor([511])
    trace: list[dict[str, Any]] = []
    instructions = [("READ_P", None), ("READ_E", None), ("COPY", None), (program.operations[0], 0), ("ACCUM_W", 0), ("READ_P", None), ("ACCUM_W", 1), (program.operations[1], 1), ("EMIT", None)]
    if moved:
        instructions = [("READ_P", None), ("READ_E", None), ("COPY", None), (program.operations[0], 0), (program.operations[1], 1), ("ACCUM_W", 0), ("READ_P", None), ("ACCUM_W", 1), ("EMIT", None)]
    for instruction, index in instructions:
        before = state.clone()
        selected = -1
        payload = None
        head_logits = None
        if instruction == "COPY":
            state[:, SLOT_R] = state[:, SLOT_E].clone()
        elif instruction in {"READ_P", "READ_E"}:
            opcode = torch.tensor([OPCODE_IDS[instruction]])
            dest = torch.tensor([SLOT_P if instruction == "READ_P" else SLOT_E])
            state, _, result = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate_zero, torch.tensor([SLOT_P]), dest, presence, read_mode=torch.tensor([0]), read_set="explicit")
            selected = int(result.selected_index.item())
            payload = result.payload
        elif instruction == "ACCUM_W":
            opcode = torch.tensor([OPCODE_IDS[instruction]])
            state, _, result = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate_vectors(model, immediate_zero), torch.tensor([SLOT_P]), torch.tensor([SLOT_W]), presence, read_mode=torch.tensor([READ_MODE_SELECT]), transform_id=torch.tensor([program.transforms[index]]), correction_module=model.correction_mlp, read_set="explicit")
            selected = int(result.selected_index.item())
            payload = result.payload
        elif instruction in TRANSFORM_OPS:
            opcode = torch.tensor([OPCODE_IDS[instruction]])
            operand = immediate_vectors(model, torch.tensor([VALUE_BASE + program.operands[index]], dtype=torch.long))
            state, candidates, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, operand, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
            head_logits = candidates.alu_logits
        else:
            state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["EMIT"]]), immediate_zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set="explicit")
        record: dict[str, Any] = {"instruction": instruction, "p": state[:, SLOT_P].squeeze(0).tolist(), "e": state[:, SLOT_E].squeeze(0).tolist(), "r": state[:, SLOT_R].squeeze(0).tolist(), "w": state[:, SLOT_W].squeeze(0).tolist(), "selected_row": selected, "p_conserved": bool(instruction == "READ_P" or torch.equal(before[:, SLOT_P], state[:, SLOT_P])), "e_conserved": bool(instruction == "READ_E" or torch.equal(before[:, SLOT_E], state[:, SLOT_E])), "r_conserved": bool(instruction in TRANSFORM_OPS + ("COPY",) or torch.equal(before[:, SLOT_R], state[:, SLOT_R])), "w_conserved": bool(instruction == "ACCUM_W" or torch.equal(before[:, SLOT_W], state[:, SLOT_W]))}
        if payload is not None:
            record["payload"] = payload.squeeze(0).tolist()
        if head_logits is not None:
            decoded, _ = decode_r(model, state)
            record.update({"alu_head_logits": head_logits.squeeze(0).tolist(), "decoded_r_id": decoded})
        trace.append(record)
    predicted, decoder_logits = decode_r(model, state)
    symbolic = symbolic_interleaved(program, interleaved_order=moved)
    workspace = state[:, SLOT_W].squeeze(0)
    target_value = VALUE_BASE + symbolic["target_value"]
    target_workspace = torch.tensor(symbolic["target_workspace"])
    return {"target_id": target_value, "predicted_id": predicted, "exact_hit": predicted == target_value, "workspace_cosine": float(F.cosine_similarity(workspace.view(1, -1), target_workspace.view(1, -1)).item()), "workspace_relative_error": float((workspace - target_workspace).norm().div(target_workspace.norm().clamp_min(1e-8)).item()), "trace": trace, "symbolic": symbolic, "decoder_logits": decoder_logits.tolist()}


def aggregate_chain(results: list[dict[str, Any]], programs: list[ChainProgram]) -> dict[str, Any]:
    return {"samples": len(results), "exact_count": sum(result["exact_hit"] for result in results), "by_length": {str(length): {"samples": sum(len(p.operations) == length for p in programs), "exact_count": sum(len(p.operations) == length and r["exact_hit"] for p, r in zip(programs, results))} for length in (2, 3)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_mix_o_composition_seed101_frozen")
    parser.add_argument("--per-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    model = load_approved_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    chains = make_chain_programs(args.per_length, args.seed)
    interleaved = make_interleaved_programs(args.per_length * 2, args.seed)
    chain_baseline = [run_chain(model, program) for program in chains]
    chain_intervention = [run_chain(model, program, mode="intervene_r") for program in chains]
    chain_e_after_copy = [run_chain(model, program, mode="e_after_copy") for program in chains]
    chain_swapped = [run_chain(model, program, mode="swap_ops") for program in chains]
    interleaved_baseline = [run_interleaved(model, program) for program in interleaved]
    interleaved_moved = [run_interleaved(model, program, moved=True) for program in interleaved]
    summary = {"status": "completed", "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "read_set": "explicit", "chain": {"baseline": aggregate_chain(chain_baseline, chains), "intervene_r": aggregate_chain(chain_intervention, chains), "e_after_copy": aggregate_chain(chain_e_after_copy, chains), "swap_ops": aggregate_chain(chain_swapped, chains), "r_intervention_target_changed": sum(a["target_id"] != b["target_id"] for a, b in zip(chain_baseline, chain_intervention)), "e_after_copy_output_unchanged": sum(a["predicted_id"] == b["predicted_id"] for a, b in zip(chain_baseline, chain_e_after_copy)), "swap_target_changed": sum(a["target_id"] != b["target_id"] for a, b in zip(chain_baseline, chain_swapped))}, "interleaved": {"baseline_exact": sum(item["exact_hit"] for item in interleaved_baseline), "moved_exact": sum(item["exact_hit"] for item in interleaved_moved), "baseline_workspace_gate": sum(item["workspace_cosine"] > 0.999 and item["workspace_relative_error"] <= 0.01 for item in interleaved_baseline), "moved_workspace_gate": sum(item["workspace_cosine"] > 0.999 and item["workspace_relative_error"] <= 0.01 for item in interleaved_moved), "samples": len(interleaved), "r_order_invariant": sum(a["predicted_id"] == b["predicted_id"] for a, b in zip(interleaved_baseline, interleaved_moved))}, "target_source": "independent symbolic interpreter; no target/state reinjection"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for name, programs, results in (("chain_baseline", chains, chain_baseline), ("chain_intervene_r", chains, chain_intervention), ("chain_e_after_copy", chains, chain_e_after_copy), ("chain_swapped", chains, chain_swapped), ("interleaved_baseline", interleaved, interleaved_baseline), ("interleaved_moved", interleaved, interleaved_moved)):
            for program, result in zip(programs, results):
                stream.write(json.dumps({"run": name, "program": program.index, "operations": list(program.operations), "operands": list(program.operands), **result}, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
