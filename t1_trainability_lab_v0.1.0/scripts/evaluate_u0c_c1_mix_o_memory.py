"""Frozen T1-MIX-O memory-size generalization evaluator."""

from __future__ import annotations

import argparse
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
    apply_operation,
    immediate_vectors,
    load_approved_model,
)
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    READ_MODE_SELECT,
    ROW_PAIR,
    ROW_REL,
    SLOT_COUNT,
    SLOT_E,
    SLOT_P,
    SLOT_R,
    SLOT_W,
)


MEMORY_SIZES = (18, 32, 64, 128)
E1_MANIFEST = ROOT / "campaign" / "u0c_c1_mix_o_depth_seed101_frozen" / "manifest.json"
MEMORY_OUTPUT_ROOT = ROOT / "campaign" / "u0c_c1_mix_o_memory_frozen"


def apply_chain(operations: list[str], operands: list[int], initial: int) -> list[int]:
    values = [initial]
    for operation, operand in zip(operations, operands):
        values.append(apply_operation(operation, values[-1], operand))
    return values


def canonical_row(row: dict[str, int]) -> tuple[int, int, int]:
    return int(row["kind"]), int(row["key"]), int(row["value"])


def load_e1_programs() -> tuple[list[dict[str, Any]], str]:
    source_bytes = E1_MANIFEST.read_bytes()
    source = json.loads(source_bytes)
    programs = [program for program in source["programs"] if int(program["depth"]) == 2]
    if len(programs) != 256:
        raise ValueError(f"expected 256 ALU=2 Eje 1 programs, found {len(programs)}")
    return programs, hashlib.sha256(source_bytes).hexdigest()


def build_manifest(seed: int = 101) -> dict[str, Any]:
    programs, source_sha256 = load_e1_programs()
    rng = random.Random(seed + 1702)
    serialized_programs: list[dict[str, Any]] = []
    for program in programs:
        original_rows = [{"kind": int(row["kind"]), "key": int(row["key"]), "value": int(row["value"])} for row in program["rows"]]
        relevant_rel = next(row for row in original_rows if row["kind"] == ROW_REL and row["key"] == int(program["start_key"]))
        value_key = relevant_rel["value"]
        used_rel = {row["key"] for row in original_rows if row["kind"] == ROW_REL}
        used_pair = {row["key"] for row in original_rows if row["kind"] == ROW_PAIR}
        memories: dict[str, list[dict[str, int]]] = {"18": list(original_rows)}
        cumulative = list(original_rows)
        for size in (32, 64, 128):
            target_each_type = size // 2
            while sum(row["kind"] == ROW_REL for row in cumulative) < target_each_type:
                key = rng.randrange(256)
                if key in used_rel or key == int(program["start_key"]):
                    continue
                used_rel.add(key)
                cumulative.append({"kind": ROW_REL, "key": key, "value": int(program["start_key"])})
            while sum(row["kind"] == ROW_PAIR for row in cumulative) < target_each_type:
                key = rng.randrange(256)
                if key in used_pair or key == value_key:
                    continue
                used_pair.add(key)
                cumulative.append({"kind": ROW_PAIR, "key": key, "value": rng.randrange(VALUE_COUNT)})
            physical = list(cumulative)
            rng.shuffle(physical)
            memories[str(size)] = physical
        target = VALUE_BASE + apply_chain(list(program["operations"]), list(program["operands"]), int(program["pair_value"]))[-1]
        serialized_programs.append({"program": int(program["index"]), "operations": list(program["operations"]), "operands": [int(value) for value in program["operands"]], "start_key": int(program["start_key"]), "pair_value": int(program["pair_value"]), "target_id": target, "dependency_witness": bool(program["dependency_witness"]), "memories": memories})
    manifest = {"seed": seed, "source_e1_manifest": str(E1_MANIFEST), "source_e1_manifest_sha256": source_sha256, "memory_sizes": list(MEMORY_SIZES), "rows_per_type": {str(size): size // 2 for size in MEMORY_SIZES}, "program_count": len(serialized_programs), "programs": serialized_programs}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest["memory_sizes"] != list(MEMORY_SIZES) or manifest["program_count"] != 256:
        raise ValueError("invalid memory manifest dimensions")
    for program in manifest["programs"]:
        row_sets: dict[str, set[tuple[int, int, int]]] = {}
        for size in MEMORY_SIZES:
            rows = program["memories"][str(size)]
            if len(rows) != size or sum(row["kind"] == ROW_REL for row in rows) != size // 2 or sum(row["kind"] == ROW_PAIR for row in rows) != size // 2:
                raise ValueError(f"invalid row geometry for program {program['program']} size {size}")
            row_sets[str(size)] = {canonical_row(row) for row in rows}
        for smaller, larger in zip(MEMORY_SIZES, MEMORY_SIZES[1:]):
            if not row_sets[str(smaller)].issubset(row_sets[str(larger)]):
                raise ValueError(f"memory inclusion violated for program {program['program']}: {smaller}->{larger}")
        for size in MEMORY_SIZES:
            rows = program["memories"][str(size)]
            rel_matches = [row for row in rows if row["kind"] == ROW_REL and row["key"] == program["start_key"]]
            if len(rel_matches) != 1:
                raise ValueError(f"ambiguous REL query for program {program['program']} size {size}")
            pair_matches = [row for row in rows if row["kind"] == ROW_PAIR and row["key"] == rel_matches[0]["value"]]
            if len(pair_matches) != 1:
                raise ValueError(f"ambiguous PAIR query for program {program['program']} size {size}")
            if int(pair_matches[0]["value"]) != int(program["pair_value"]):
                raise ValueError(f"PAIR content changed for program {program['program']} size {size}")


def materialize_batch(model: C1JointModel, programs: list[dict[str, Any]], size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, list[int], list[int]]:
    memory_rows = [program["memories"][str(size)] for program in programs]
    memory_keys = torch.stack([model.token_embedding(torch.tensor([row["key"] + KEY_BASE for row in rows])) for rows in memory_rows])
    memory_values = torch.stack([model.token_embedding(torch.tensor([row["value"] + KEY_BASE if row["kind"] == ROW_REL else VALUE_BASE + row["value"] for row in rows])) for rows in memory_rows])
    memory_types = torch.tensor([[row["kind"] for row in rows] for rows in memory_rows], dtype=torch.long)
    row_mask = torch.ones_like(memory_types, dtype=torch.bool)
    rel_expected: list[int] = []
    pair_expected: list[int] = []
    for program, rows in zip(programs, memory_rows):
        rel_expected.append(next(index for index, row in enumerate(rows) if row["kind"] == ROW_REL and row["key"] == program["start_key"]))
        value_key = rows[rel_expected[-1]]["value"]
        pair_expected.append(next(index for index, row in enumerate(rows) if row["kind"] == ROW_PAIR and row["key"] == value_key))
    return memory_keys, memory_values, memory_types, row_mask, rel_expected, pair_expected


def symbolic_pair_value(program: dict[str, Any], size: int) -> int:
    rows = program["memories"][str(size)]
    pointer = next(row["value"] for row in rows if row["kind"] == ROW_REL and row["key"] == program["start_key"])
    return next(row["value"] for row in rows if row["kind"] == ROW_PAIR and row["key"] == pointer)


def decode_register(model: C1JointModel, state: Tensor) -> Tensor:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    return class_ids[model.register_decoder(state[:, SLOT_R], model.token_embedding(class_ids)).argmax(-1)]


def relative_error(actual: Tensor, expected: Tensor) -> Tensor:
    return (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-8)


@torch.no_grad()
def execute_size(model: C1JointModel, programs: list[dict[str, Any]], size: int, *, read_e_select: bool = False) -> list[dict[str, Any]]:
    memory_keys, memory_values, memory_types, row_mask, rel_expected, pair_expected = materialize_batch(model, programs, size)
    batch_size = len(programs)
    state = torch.zeros((batch_size, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([program["start_key"] for program in programs]))
    initial_state = state.clone()
    presence = torch.ones((batch_size, SLOT_COUNT), dtype=torch.bool)
    zero = immediate_vectors(model, torch.full((batch_size,), 511, dtype=torch.long))
    read_mode = torch.zeros(batch_size, dtype=torch.long)
    source_p = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    destination_p = torch.full((batch_size,), SLOT_P, dtype=torch.long)
    destination_e = torch.full((batch_size,), SLOT_E, dtype=torch.long)
    state, _, read_p = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_P"], dtype=torch.long), zero, source_p, destination_p, presence, read_mode=read_mode, read_set="explicit")
    after_read_p = state.clone()
    read_e_mode = torch.full((batch_size,), READ_MODE_SELECT if read_e_select else 0, dtype=torch.long)
    state, _, read_e = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["READ_E"], dtype=torch.long), zero, source_p, destination_e, presence, read_mode=read_e_mode, read_set="explicit", diagnostic_read_e_select=read_e_select)
    after_read_e = state.clone()
    state[:, SLOT_R] = state[:, SLOT_E].clone()
    copy_equal = torch.equal(state[:, SLOT_R], state[:, SLOT_E])
    expected_by_program = [apply_chain(list(program["operations"]), list(program["operands"]), symbolic_pair_value(program, size)) for program in programs]
    decoded_by_alu: list[list[int]] = [[] for _ in programs]
    conservation_by_alu: list[list[bool]] = [[] for _ in programs]
    for round_index in range(2):
        before = state.clone()
        opcode = torch.tensor([OPCODE_IDS[program["operations"][round_index]] for program in programs], dtype=torch.long)
        operand_ids = torch.tensor([VALUE_BASE + program["operands"][round_index] for program in programs], dtype=torch.long)
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, immediate_vectors(model, operand_ids), torch.full((batch_size,), SLOT_R, dtype=torch.long), torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=read_mode, read_set="explicit")
        decoded = decode_register(model, state).tolist()
        for item, value in enumerate(decoded):
            decoded_by_alu[item].append(int(value))
            conservation_by_alu[item].append(bool(torch.equal(before[item, SLOT_P], state[item, SLOT_P]) and torch.equal(before[item, SLOT_E], state[item, SLOT_E]) and torch.equal(before[item, SLOT_W], state[item, SLOT_W])))
    before_emit = state.clone()
    state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.full((batch_size,), OPCODE_IDS["EMIT"], dtype=torch.long), zero, source_p, torch.full((batch_size,), SLOT_R, dtype=torch.long), presence, read_mode=read_mode, read_set="explicit")
    final_decoded = decode_register(model, state).tolist()
    read_p_mass = read_p.attention_soft[torch.arange(batch_size), torch.tensor(rel_expected)]
    read_e_mass = read_e.attention_soft[torch.arange(batch_size), torch.tensor(pair_expected)]
    read_p_margin = read_p.selection_margin
    read_e_margin = read_e.selection_margin
    read_p_payload_error = relative_error(read_p.payload, memory_values[torch.arange(batch_size), torch.tensor(rel_expected)])
    read_e_payload_error = relative_error(read_e.payload, memory_values[torch.arange(batch_size), torch.tensor(pair_expected)])
    results: list[dict[str, Any]] = []
    for item, program in enumerate(programs):
        expected_ids = [VALUE_BASE + value for value in expected_by_program[item][1:]]
        actual_ids = decoded_by_alu[item]
        divergence = [index for index, (actual, expected) in enumerate(zip(actual_ids, expected_ids)) if actual != expected]
        results.append({"program": int(program["program"]), "dependency_witness": bool(program["dependency_witness"]), "size": size, "target_id": int(expected_ids[-1]), "predicted_id": int(final_decoded[item]), "exact_hit": int(final_decoded[item]) == int(expected_ids[-1]), "decoded_r_ids_after_each_alu": actual_ids, "expected_r_ids_after_each_alu": expected_ids, "intermediate_exact": not divergence, "first_divergence": divergence[0] if divergence else None, "read_p": {"selected_row": int(read_p.selected_index[item]), "expected_row": rel_expected[item], "match": int(read_p.selected_index[item]) == rel_expected[item], "mass_correct": float(read_p_mass[item]), "margin": float(read_p_margin[item]), "payload_relative_error": float(read_p_payload_error[item])}, "read_e": {"selected_row": int(read_e.selected_index[item]), "expected_row": pair_expected[item], "match": int(read_e.selected_index[item]) == pair_expected[item], "mass_correct": float(read_e_mass[item]), "margin": float(read_e_margin[item]), "payload_relative_error": float(read_e_payload_error[item])}, "copy_equal": copy_equal, "state": {"read_p_non_target_slots_conserved": bool(torch.equal(initial_state[item, SLOT_E], after_read_p[item, SLOT_E]) and torch.equal(initial_state[item, SLOT_R], after_read_p[item, SLOT_R]) and torch.equal(initial_state[item, SLOT_W], after_read_p[item, SLOT_W])), "read_e_non_target_slots_conserved": bool(torch.equal(after_read_p[item, SLOT_P], after_read_e[item, SLOT_P]) and torch.equal(after_read_p[item, SLOT_R], after_read_e[item, SLOT_R]) and torch.equal(after_read_p[item, SLOT_W], after_read_e[item, SLOT_W])), "copy_direct_equal": copy_equal, "alu_non_target_slots_conserved": all(conservation_by_alu[item]), "emit_no_state_change": bool(torch.equal(before_emit[item], state[item]))}})
    return results


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    sizes: dict[str, Any] = {}
    for size in MEMORY_SIZES:
        selected = [result for result in results if result["size"] == size]
        witness = [result for result in selected if result["dependency_witness"]]
        general = [result for result in selected if not result["dependency_witness"]]
        sizes[str(size)] = {"samples": len(selected), "exact_count": sum(result["exact_hit"] for result in selected), "intermediate_exact_count": sum(result["intermediate_exact"] for result in selected), "read_p_match_count": sum(result["read_p"]["match"] for result in selected), "read_e_match_count": sum(result["read_e"]["match"] for result in selected), "copy_equal_count": sum(result["copy_equal"] for result in selected), "state_contract_count": sum(all(result["state"].values()) for result in selected), "read_p_mass_correct_mean": sum(result["read_p"]["mass_correct"] for result in selected) / len(selected), "read_p_mass_correct_min": min(result["read_p"]["mass_correct"] for result in selected), "read_e_mass_correct_mean": sum(result["read_e"]["mass_correct"] for result in selected) / len(selected), "read_e_mass_correct_min": min(result["read_e"]["mass_correct"] for result in selected), "read_p_margin_min": min(result["read_p"]["margin"] for result in selected), "read_e_margin_min": min(result["read_e"]["margin"] for result in selected), "read_p_payload_relative_error_max": max(result["read_p"]["payload_relative_error"] for result in selected), "read_e_payload_relative_error_max": max(result["read_e"]["payload_relative_error"] for result in selected), "dependency_witness": {"samples": len(witness), "exact_count": sum(result["exact_hit"] for result in witness), "intermediate_exact_count": sum(result["intermediate_exact"] for result in witness)}, "general_arithmetic": {"samples": len(general), "exact_count": sum(result["exact_hit"] for result in general), "intermediate_exact_count": sum(result["intermediate_exact"] for result in general)}}
    failures = [size for size in MEMORY_SIZES if sizes[str(size)]["exact_count"] < 256 or sizes[str(size)]["intermediate_exact_count"] < 256]
    return {"by_size": sizes, "first_failing_size": min(failures) if failures else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--manifest", type=Path, default=MEMORY_OUTPUT_ROOT / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=MEMORY_OUTPUT_ROOT / "seed101")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(args.seed)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_manifest(manifest)
    programs = manifest["programs"]
    model = load_approved_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    results: list[dict[str, Any]] = []
    for size in MEMORY_SIZES:
        results.extend(execute_size(model, programs, size))
    target_consistency = all(len({VALUE_BASE + apply_chain(list(program["operations"]), list(program["operands"]), symbolic_pair_value(program, size))[-1] for size in MEMORY_SIZES}) == 1 for program in programs)
    summary = {"status": "completed", "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "programs": len(programs), "memory_sizes": list(MEMORY_SIZES), "rows_per_type": {str(size): size // 2 for size in MEMORY_SIZES}, "read_policy": {"READ_P": "BLEND", "READ_E": "BLEND", "ALU": "explicit", "COPY": "effective E to R"}, "summary": aggregate(results), "target_consistency_across_memories": target_consistency, "target_source": "independent symbolic interpreter over serialized memory; no target/state reinjection"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
