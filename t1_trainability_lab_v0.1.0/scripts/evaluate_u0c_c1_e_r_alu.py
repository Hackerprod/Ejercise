"""Frozen T1 E->R->ALU evaluator with symbolic execution and causal controls."""

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
sys.path.insert(0, str(ROOT))

from train_u0c_c1_joint import C1JointModel, load_approved_model  # noqa: E402
from train_u0a import immediate_vectors  # noqa: E402
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    ROW_PAIR,
    ROW_REL,
    SLOT_COUNT,
    SLOT_E,
    SLOT_P,
    SLOT_R,
)


DIMENSION = 64
KEY_BASE = 0
VALUE_BASE = 288
VALUE_COUNT = 32
IMM_ZERO = 511
TRANSFORM_OPS = ("ALU_ADD", "ALU_SUB", "ALU_MUL")


@dataclass(frozen=True)
class MemoryRow:
    kind: int
    key: int
    value: int


@dataclass
class Program:
    index: int
    operation: str
    start_key: int
    pair_value: int
    operand: int
    alternate_pair_value: int
    alternate_e_value: int
    rows: list[MemoryRow]

    @property
    def target_value(self) -> int:
        return apply_operation(self.operation, self.pair_value, self.operand)


def apply_operation(operation: str, left: int, right: int) -> int:
    if operation == "ALU_ADD":
        return (left + right) % VALUE_COUNT
    if operation == "ALU_SUB":
        return (left - right) % VALUE_COUNT
    if operation == "ALU_MUL":
        return (left * right) % VALUE_COUNT
    raise ValueError(f"unknown operation: {operation}")


def make_programs(per_operation: int, seed: int) -> list[Program]:
    randomizer = random.Random(seed)
    programs: list[Program] = []
    index = 0
    for operation in TRANSFORM_OPS:
        for item in range(per_operation):
            start_key, value_key, *distractor_keys = randomizer.sample(range(256), 2 + 8)
            if operation == "ALU_SUB" and item == 0:
                pair_value, operand = 23, 7
            else:
                pair_value = randomizer.randrange(1, VALUE_COUNT) if operation == "ALU_MUL" else randomizer.randrange(VALUE_COUNT)
                operand = randomizer.randrange(1, VALUE_COUNT)
            swapped_operation = "ALU_ADD" if operation != "ALU_ADD" else "ALU_SUB"
            while apply_operation(swapped_operation, pair_value, operand) == apply_operation(operation, pair_value, operand):
                operand = randomizer.randrange(1, VALUE_COUNT)
            alternate_pair_value = randomizer.randrange(VALUE_COUNT)
            while apply_operation(operation, alternate_pair_value, operand) == apply_operation(operation, pair_value, operand):
                alternate_pair_value = randomizer.randrange(VALUE_COUNT)
            alternate_e_value = randomizer.randrange(VALUE_COUNT)
            while apply_operation(operation, alternate_e_value, operand) == apply_operation(operation, pair_value, operand):
                alternate_e_value = randomizer.randrange(VALUE_COUNT)
            rows = [MemoryRow(ROW_REL, start_key, value_key), MemoryRow(ROW_PAIR, value_key, pair_value)]
            for distractor in distractor_keys:
                rows.append(MemoryRow(ROW_REL, distractor, start_key))
                rows.append(MemoryRow(ROW_PAIR, distractor, randomizer.randrange(VALUE_COUNT)))
            randomizer.shuffle(rows)
            programs.append(Program(index, operation, start_key, pair_value, operand, alternate_pair_value, alternate_e_value, rows))
            index += 1
    return programs


def symbolic_execute(program: Program, *, pair_value: int | None = None, e_value: int | None = None, copy: bool = True, operation: str | None = None) -> dict[str, Any]:
    """Execute declared instructions against logical rows, never model state."""
    rows = list(program.rows)
    if pair_value is not None:
        pair_row = next(index for index, row in enumerate(rows) if row.kind == ROW_PAIR and row.key == program.rows[next(i for i, item in enumerate(program.rows) if item.kind == ROW_REL and item.key == program.start_key)].value)
        rows[pair_row] = MemoryRow(ROW_PAIR, rows[pair_row].key, pair_value)
    state: dict[str, int | None] = {"P": program.start_key, "E": None, "R": 0}
    trace: list[dict[str, Any]] = []
    active_operation = operation or program.operation
    instructions = ("READ_P", "READ_E", "COPY_E_TO_R", active_operation, "EMIT")
    for instruction in instructions:
        before = dict(state)
        expected_row = -1
        if instruction == "READ_P":
            expected_row = next(index for index, row in enumerate(rows) if row.kind == ROW_REL and row.key == state["P"])
            state["P"] = rows[expected_row].value
        elif instruction == "READ_E":
            expected_row = next(index for index, row in enumerate(rows) if row.kind == ROW_PAIR and row.key == state["P"])
            state["E"] = rows[expected_row].value
            if e_value is not None:
                state["E"] = e_value
        elif instruction == "COPY_E_TO_R":
            if copy:
                state["R"] = state["E"]
        elif instruction in TRANSFORM_OPS:
            state["R"] = apply_operation(instruction, int(state["R"]), program.operand)
        trace.append({"instruction": instruction, "before": before, "after": dict(state), "expected_row": expected_row})
    return {"target_value": int(state["R"]), "state": state, "trace": trace}


def materialize_memory(model: C1JointModel, rows: list[MemoryRow]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key_ids = torch.tensor([row.key + KEY_BASE for row in rows], dtype=torch.long)
    value_ids = torch.tensor(
        [row.value + KEY_BASE if row.kind == ROW_REL else VALUE_BASE + row.value for row in rows], dtype=torch.long
    )
    memory_keys = model.token_embedding(key_ids)
    memory_values = model.token_embedding(value_ids)
    memory_types = torch.tensor([row.kind for row in rows], dtype=torch.long).unsqueeze(0)
    row_mask = torch.ones_like(memory_types, dtype=torch.bool)
    return memory_keys, memory_values, memory_types, row_mask


def cosine(left: Tensor, right: Tensor) -> float:
    return float(F.cosine_similarity(left.view(1, -1), right.view(1, -1)).item())


def value_logits(model: C1JointModel, state: Tensor) -> tuple[Tensor, int, float]:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)
    logits = model.register_decoder(state[:, SLOT_R, :], model.token_embedding(class_ids))
    position = int(logits.argmax(-1).item())
    return logits.squeeze(0), int(class_ids[position]), float(logits[0, position].item())


@torch.no_grad()
def run_program(model: C1JointModel, program: Program, *, mode: str = "baseline", operand_mode: str = "historical_vector", read_set: str = "legacy") -> dict[str, Any]:
    rows = list(program.rows)
    if mode == "pair_intervention":
        pair_row = next(index for index, row in enumerate(rows) if row.kind == ROW_PAIR and row.key == next(item.value for item in rows if item.kind == ROW_REL and item.key == program.start_key))
        rows[pair_row] = MemoryRow(ROW_PAIR, rows[pair_row].key, program.alternate_pair_value)
    memory_keys, memory_values, memory_types, row_mask = materialize_memory(model, rows)
    presence = torch.tensor([[True, True, True, False]])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([program.start_key + KEY_BASE]))
    immediate_zero = torch.tensor([IMM_ZERO])
    symbolic = symbolic_execute(
        program,
        pair_value=program.alternate_pair_value if mode == "pair_intervention" else None,
        e_value=program.alternate_e_value if mode == "e_intervention" else None,
        copy=mode != "disable_copy",
        operation=("ALU_ADD" if program.operation != "ALU_ADD" else "ALU_SUB") if mode == "swap_operation" else None,
    )
    target_value = symbolic["target_value"]
    trace: list[dict[str, Any]] = []

    def record(instruction: str, before: Tensor, after: Tensor, *, selected: int = -1, payload: Tensor | None = None, logits: Tensor | None = None) -> None:
        entry: dict[str, Any] = {
            "instruction": instruction,
            "p": after[:, SLOT_P].squeeze(0).tolist(),
            "e": after[:, SLOT_E].squeeze(0).tolist(),
            "r": after[:, SLOT_R].squeeze(0).tolist(),
            "selected_row": selected,
            "w_unchanged": bool(torch.equal(after[:, 3], before[:, 3])),
            "p_unchanged": bool(torch.equal(after[:, SLOT_P], before[:, SLOT_P])),
            "e_unchanged": bool(torch.equal(after[:, SLOT_E], before[:, SLOT_E])),
            "r_unchanged": bool(torch.equal(after[:, SLOT_R], before[:, SLOT_R])),
        }
        if payload is not None:
            entry["payload"] = payload.squeeze(0).tolist()
        if logits is not None:
            entry["r_logits"] = logits.tolist()
            entry["decoded_r_id"] = int(VALUE_BASE + int(logits.argmax().item()))
        trace.append(entry)

    before = state.clone()
    state, _, result = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), immediate_zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode=torch.tensor([0]), read_set=read_set)
    expected_read_p = symbolic["trace"][0]["expected_row"]
    record("READ_P", before, state, selected=int(result.selected_index.item()), payload=result.payload)
    trace[-1]["expected_row"] = expected_read_p
    trace[-1]["row_hit"] = int(result.selected_index.item()) == expected_read_p

    before = state.clone()
    state, _, result = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), immediate_zero, torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode=torch.tensor([0]), read_set=read_set)
    expected_read_e = symbolic["trace"][1]["expected_row"]
    record("READ_E", before, state, selected=int(result.selected_index.item()), payload=result.payload)
    trace[-1]["expected_row"] = expected_read_e
    trace[-1]["row_hit"] = int(result.selected_index.item()) == expected_read_e
    trace[-1]["payload_value_target"] = VALUE_BASE + program.pair_value
    trace[-1]["payload_target_rel_error"] = float((result.payload - model.token_embedding(torch.tensor([VALUE_BASE + (program.alternate_pair_value if mode == "pair_intervention" else program.pair_value)]))).norm().div(result.payload.norm().clamp_min(1e-8)).item())

    before = state.clone()
    if mode == "e_intervention":
        state[:, SLOT_E] = model.token_embedding(torch.tensor([VALUE_BASE + program.alternate_e_value]))
    if mode == "diagnostic_oracle_r":
        state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + program.pair_value]))
    elif mode != "disable_copy":
        state[:, SLOT_R] = state[:, SLOT_E]
    copy_equal = bool(torch.equal(state[:, SLOT_R], state[:, SLOT_E]))
    record("COPY_E_TO_R", before, state)
    trace[-1]["copy_equal_direct"] = copy_equal
    trace[-1]["source_e"] = before[:, SLOT_E].squeeze(0).tolist()
    trace[-1]["r_after_copy"] = state[:, SLOT_R].squeeze(0).tolist()

    before = state.clone()
    operation = ("ALU_ADD" if program.operation != "ALU_ADD" else "ALU_SUB") if mode == "swap_operation" else program.operation
    operand_ids = torch.tensor([VALUE_BASE + program.operand], dtype=torch.long, device=state.device)
    operand = operand_ids if operand_mode == "integer_id" else immediate_vectors(model, operand_ids)
    state, candidates, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS[operation]]), operand, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set=read_set)
    decoder_logits, decoded_id, decoded_score = value_logits(model, state)
    head_logits = candidates.alu_logits.squeeze(0)
    record(operation, before, state, logits=head_logits)
    trace[-1]["decoded_r_id"] = decoded_id
    trace[-1]["decoded_score"] = decoded_score
    trace[-1]["alu_head_logits"] = head_logits.tolist()
    trace[-1]["register_decoder_logits"] = decoder_logits.tolist()
    trace[-1]["expected_value_id"] = VALUE_BASE + target_value

    before = state.clone()
    state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["EMIT"]]), immediate_zero, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]), read_set=read_set)
    record("EMIT", before, state)
    trace[-1]["emit_preserves_r"] = bool(torch.equal(before[:, SLOT_R], state[:, SLOT_R]))
    final_logits, final_id, final_score = value_logits(model, state)
    return {
        "target_value": target_value,
        "target_id": VALUE_BASE + target_value,
        "predicted_id": final_id,
        "exact_hit": final_id == VALUE_BASE + target_value,
        "final_score": final_score,
        "trace": trace,
        "symbolic_trace": symbolic["trace"],
        "symbolic_state": symbolic["state"],
        "final_r": state[:, SLOT_R].squeeze(0).tolist(),
        "final_logits": final_logits.tolist(),
        "pair_value": program.pair_value,
        "operand": program.operand,
        "operation": program.operation,
    }


def aggregate(results: list[dict[str, Any]], programs: list[Program]) -> dict[str, Any]:
    by_operation: dict[str, dict[str, Any]] = {}
    for operation in TRANSFORM_OPS:
        selected = [(program, result) for program, result in zip(programs, results) if program.operation == operation]
        by_operation[operation] = {"samples": len(selected), "exact_count": sum(result["exact_hit"] for _, result in selected), "exact_rate": sum(result["exact_hit"] for _, result in selected) / max(1, len(selected))}
    row_decisions = [trace for result in results for trace in result["trace"] if "row_hit" in trace]
    copy_decisions = [trace for result in results for trace in result["trace"] if trace["instruction"] == "COPY_E_TO_R"]
    return {"samples": len(results), "exact_count": sum(result["exact_hit"] for result in results), "exact_rate": sum(result["exact_hit"] for result in results) / max(1, len(results)), "row_hit_rate": sum(trace["row_hit"] for trace in row_decisions) / max(1, len(row_decisions)), "copy_equal_direct_rate": sum(trace.get("copy_equal_direct", False) for trace in copy_decisions) / max(1, len(copy_decisions)), "by_operation": by_operation}


@torch.no_grad()
def run_matrix_case(model: C1JointModel, program: Program, *, operand_mode: str, presence_mode: str) -> dict[str, Any]:
    """Run diagnostic ALU with canonical R, varying only operand route/presence."""
    memory_keys, memory_values, memory_types, row_mask = materialize_memory(model, program.rows)
    presence = torch.tensor([[False, True, False, False]]) if presence_mode == "only_r" else torch.tensor([[True, True, True, False]])
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + program.pair_value]))
    pre_alu: dict[str, Any] = {"p": state[:, SLOT_P].squeeze(0).tolist(), "e": state[:, SLOT_E].squeeze(0).tolist()}
    if presence_mode == "p_e_r":
        state[:, SLOT_P] = model.token_embedding(torch.tensor([program.start_key + KEY_BASE]))
        state, _, read_p = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), torch.tensor([IMM_ZERO]), torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode=torch.tensor([0]))
        state, _, read_e = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), torch.tensor([IMM_ZERO]), torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode=torch.tensor([0]))
        pre_alu = {"p": state[:, SLOT_P].squeeze(0).tolist(), "e": state[:, SLOT_E].squeeze(0).tolist(), "read_p_row": int(read_p.selected_index.item()), "read_e_row": int(read_e.selected_index.item())}
        state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + program.pair_value]))
    operand_ids = torch.tensor([VALUE_BASE + program.operand], dtype=torch.long, device=state.device)
    operand = operand_ids if operand_mode == "integer_id" else immediate_vectors(model, operand_ids)
    before = state.clone()
    state, candidates, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS[program.operation]]), operand, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), presence, read_mode=torch.tensor([0]))
    decoder_logits, decoded_id, decoded_score = value_logits(model, state)
    head_logits = candidates.alu_logits.squeeze(0)
    target_id = VALUE_BASE + program.target_value
    return {
        "program": program.index,
        "operation": program.operation,
        "pair_value": program.pair_value,
        "operand": program.operand,
        "target_id": target_id,
        "predicted_id": decoded_id,
        "exact_hit": decoded_id == target_id,
        "operand_mode": operand_mode,
        "presence_mode": presence_mode,
        "presence": presence.squeeze(0).tolist(),
        "pre_alu": pre_alu,
        "r_before": before[:, SLOT_R].squeeze(0).tolist(),
        "r_after": state[:, SLOT_R].squeeze(0).tolist(),
        "alu_head_logits": head_logits.tolist(),
        "register_decoder_logits": decoder_logits.tolist(),
        "decoded_score": decoded_score,
    }


def matrix_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(f"{case['operand_mode']}__{case['presence_mode']}", []).append(case)
    return {name: {"samples": len(items), "exact_count": sum(item["exact_hit"] for item in items), "exact_rate": sum(item["exact_hit"] for item in items) / max(1, len(items)), "by_operation": {operation: {"samples": sum(item["operation"] == operation for item in items), "exact_count": sum(item["operation"] == operation and item["exact_hit"] for item in items)} for operation in TRANSFORM_OPS}} for name, items in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_mix_o_e_r_alu_seed101_frozen")
    parser.add_argument("--per-operation", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = load_approved_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    programs = make_programs(args.per_operation, args.seed)
    modes = ("baseline", "pair_intervention", "e_intervention", "disable_copy", "swap_operation")
    runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    traces: list[dict[str, Any]] = []
    for program in programs:
        for mode in modes:
            result = run_program(model, program, mode=mode)
            runs[mode].append(result)
            traces.append({"program": program.index, "mode": mode, "start_key": program.start_key, "rows": [row.__dict__ for row in program.rows], **result})

    matrix_cases = [
        run_matrix_case(model, program, operand_mode=operand_mode, presence_mode=presence_mode)
        for operand_mode in ("integer_id", "historical_vector")
        for presence_mode in ("only_r", "p_e_r")
        for program in programs
    ]

    # Diagnostics are authorized only when baseline fails: preserve same P/E/R presence.
    diagnostics: dict[str, Any] = {}
    baseline = runs["baseline"]
    if not all(result["exact_hit"] for result in baseline):
        diagnostic_modes = ("diagnostic_oracle_r", "diagnostic_actual_e")
        for mode in diagnostic_modes:
            diagnostic_results = [run_program(model, program, mode=mode) for program in programs]
            diagnostics[mode] = aggregate(diagnostic_results, programs)
            for program, result in zip(programs, diagnostic_results):
                traces.append({"program": program.index, "mode": mode, "start_key": program.start_key, "rows": [row.__dict__ for row in program.rows], **result})

    summary: dict[str, Any] = {
        "status": "completed",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "programs": len(programs),
        "coverage": {operation: args.per_operation for operation in TRANSFORM_OPS},
        "program": "READ_P(BLEND) -> READ_E(BLEND) -> COPY_E_TO_R -> one ALU -> EMIT(R)",
        "key_namespace": "KEY_BASE shared by REL and PAIR addressing",
        "numeric_codebook": "VALUE_BASE=288..319 for PAIR values, immediates, and R decode",
        "presence": [True, True, True, False],
        "target_source": "independent symbolic interpreter over serialized rows; executor receives no target/value oracle",
        "runs": {mode: aggregate(results, programs) for mode, results in runs.items()},
        "matrix_2x2": matrix_summary(matrix_cases),
        "diagnostics": diagnostics,
        "causal_comparisons": {
            mode: {
                "symbolic_target_changed": sum(a["target_value"] != b["target_value"] for a, b in zip(runs["baseline"], runs[mode])),
                "model_predicted_value_changed": sum(a["predicted_id"] != b["predicted_id"] for a, b in zip(runs["baseline"], runs[mode])),
            }
            for mode in ("pair_intervention", "e_intervention", "disable_copy", "swap_operation")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, sort_keys=True) + "\n")
    with (args.output_dir / "matrix_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for case in matrix_cases:
            stream.write(json.dumps(case, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
