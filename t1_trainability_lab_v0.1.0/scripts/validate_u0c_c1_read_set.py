"""Validate explicit ALU read-set contract without changing checkpoint weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_u0c_c1_e_r_alu as alu  # noqa: E402
from train_u0a import immediate_vectors  # noqa: E402
from train_u0c_c1_joint import C1JointModel, load_approved_model  # noqa: E402
from t1_trainability.unified import OPCODE_IDS, ROW_REL, SLOT_COUNT, SLOT_E, SLOT_P, SLOT_R  # noqa: E402


VALUE_BASE = 288
VALUE_COUNT = 32
IMM_ZERO = 511
PRESENCE_ONLY_R = torch.tensor([False, True, False, False])
PRESENCE_P_E_R_W = torch.tensor([True, True, True, True])


def codebook(model: C1JointModel) -> Tensor:
    return model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))


def decode(model: C1JointModel, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    classes = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    logits = model.register_decoder(state[:, SLOT_R, :], model.token_embedding(classes))
    return logits, classes[logits.argmax(-1)], logits


def alu_batch(model: C1JointModel, state: Tensor, operations: Tensor, operands: Tensor, presence: Tensor, *, read_set: str) -> dict[str, Tensor]:
    batch = state.shape[0]
    memory_keys = torch.zeros((batch, 1, model.dimension))
    memory_values = torch.zeros_like(memory_keys)
    memory_types = torch.zeros((batch, 1), dtype=torch.long)
    row_mask = torch.zeros((batch, 1), dtype=torch.bool)
    operand_ids = (VALUE_BASE + operands).to(dtype=torch.long)
    operand_vectors = immediate_vectors(model, operand_ids)
    next_state, candidates, _ = model.step(
        state,
        memory_keys,
        memory_values,
        memory_types,
        row_mask,
        operations,
        operand_vectors,
        torch.full((batch,), SLOT_R, dtype=torch.long),
        torch.full((batch,), SLOT_R, dtype=torch.long),
        presence,
        read_mode=torch.zeros(batch, dtype=torch.long),
        read_set=read_set,
    )
    decoder_logits, predicted, _ = decode(model, next_state)
    return {"state": next_state, "candidate_r": candidates.values[:, SLOT_R, :], "head_logits": candidates.alu_logits, "decoder_logits": decoder_logits, "predicted": predicted}


def states_for_programs(model: C1JointModel, programs: list[alu.Program]) -> Tensor:
    states = []
    for program in programs:
        state = torch.zeros((SLOT_COUNT, model.dimension))
        state[SLOT_P] = model.token_embedding(torch.tensor(program.start_key + alu.KEY_BASE))
        memory_keys, memory_values, memory_types, row_mask = alu.materialize_memory(model, program.rows)
        state = state.unsqueeze(0)
        presence = torch.tensor([[True, True, True, False]])
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), torch.tensor([IMM_ZERO]), torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), presence, read_mode=torch.tensor([0]))
        state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), torch.tensor([IMM_ZERO]), torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), presence, read_mode=torch.tensor([0]))
        state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + program.pair_value]))
        states.append(state.squeeze(0))
    return torch.stack(states)


def max_abs(left: Tensor, right: Tensor) -> float:
    return float((left - right).abs().max().item())


def signed_margin(logits: Tensor, target: Tensor) -> Tensor:
    result = []
    for row, target_id in zip(logits, target):
        index = int(target_id - VALUE_BASE)
        rival = torch.cat((row[:index], row[index + 1 :])).max()
        result.append(row[index] - rival)
    return torch.stack(result)


def canonical_contexts(model: C1JointModel, programs: list[alu.Program]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    context = states_for_programs(model, programs)
    context[:, SLOT_R, :] = torch.stack([model.token_embedding(torch.tensor(VALUE_BASE + p.pair_value)) for p in programs])
    only_r = torch.zeros_like(context)
    only_r[:, SLOT_R, :] = context[:, SLOT_R, :]
    operations = torch.tensor([OPCODE_IDS[p.operation] for p in programs], dtype=torch.long)
    operands = torch.tensor([p.operand for p in programs], dtype=torch.long)
    return only_r, context, operations, operands


def activation_bridge(model: C1JointModel, programs: list[alu.Program]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    only_r, context, operations, operands = canonical_contexts(model, programs)
    reference = alu_batch(model, only_r, operations, operands, PRESENCE_ONLY_R.expand(len(programs), -1), read_set="explicit")
    explicit = alu_batch(model, context, operations, operands, PRESENCE_P_E_R_W.expand(len(programs), -1), read_set="explicit")
    legacy = alu_batch(model, context, operations, operands, torch.tensor([True, True, True, False]).expand(len(programs), -1), read_set="legacy")
    target = torch.tensor([VALUE_BASE + p.target_value for p in programs])
    records: list[dict[str, Any]] = []
    for index, program in enumerate(programs):
        records.append({
            "program": program.index,
            "operation": program.operation,
            "target_id": int(target[index]),
            "reference_prediction_id": int(reference["predicted"][index]),
            "explicit_prediction_id": int(explicit["predicted"][index]),
            "legacy_prediction_id": int(legacy["predicted"][index]),
            "reference_head_margin": float(signed_margin(reference["head_logits"][index:index + 1], target[index:index + 1])[0]),
            "explicit_head_margin": float(signed_margin(explicit["head_logits"][index:index + 1], target[index:index + 1])[0]),
            "reference_decoder_margin": float(signed_margin(reference["decoder_logits"][index:index + 1], target[index:index + 1])[0]),
            "explicit_decoder_margin": float(signed_margin(explicit["decoder_logits"][index:index + 1], target[index:index + 1])[0]),
            "candidate_r_max_abs_vs_reference": max_abs(explicit["candidate_r"][index], reference["candidate_r"][index]),
            "head_logits_max_abs_vs_reference": max_abs(explicit["head_logits"][index], reference["head_logits"][index]),
            "register_r_max_abs_vs_reference": max_abs(explicit["state"][index, SLOT_R], reference["state"][index, SLOT_R]),
            "decoder_logits_max_abs_vs_reference": max_abs(explicit["decoder_logits"][index], reference["decoder_logits"][index]),
        })
    summary = {
        "reference_exact": int((reference["predicted"] == target).sum()),
        "explicit_exact": int((explicit["predicted"] == target).sum()),
        "legacy_exact": int((legacy["predicted"] == target).sum()),
        "samples": len(programs),
        "max_candidate_r_abs": max(item["candidate_r_max_abs_vs_reference"] for item in records),
        "max_head_logits_abs": max(item["head_logits_max_abs_vs_reference"] for item in records),
        "max_register_r_abs": max(item["register_r_max_abs_vs_reference"] for item in records),
        "max_decoder_logits_abs": max(item["decoder_logits_max_abs_vs_reference"] for item in records),
    }
    return summary, records


def invariance_check(model: C1JointModel, programs: list[alu.Program]) -> dict[str, Any]:
    only_r, context, operations, operands = canonical_contexts(model, programs)
    context[:, SLOT_R, :] = only_r[:, SLOT_R, :]
    context[:, SLOT_P, :] = torch.stack([model.token_embedding(torch.tensor((p.start_key + 17) % 256)) for p in programs])
    context[:, SLOT_E, :] = torch.stack([model.token_embedding(torch.tensor(VALUE_BASE + ((p.pair_value + 11) % VALUE_COUNT))) for p in programs])
    context[:, 3, :] = torch.stack([model.token_embedding(torch.tensor(VALUE_BASE + ((p.pair_value + 19) % VALUE_COUNT))) for p in programs])
    result = alu_batch(model, context, operations, operands, PRESENCE_P_E_R_W.expand(len(programs), -1), read_set="explicit")
    reference = alu_batch(model, only_r, operations, operands, PRESENCE_ONLY_R.expand(len(programs), -1), read_set="explicit")
    after_e = context.clone()
    after_e[:, SLOT_E, :] = torch.stack([model.token_embedding(torch.tensor(VALUE_BASE + ((p.pair_value + 7) % VALUE_COUNT))) for p in programs])
    after_e_result = alu_batch(model, after_e, operations, operands, PRESENCE_P_E_R_W.expand(len(programs), -1), read_set="explicit")
    before_copy_a = context.clone()
    before_copy_b = context.clone()
    before_copy_a[:, SLOT_R, :] = before_copy_a[:, SLOT_E, :]
    before_copy_b[:, SLOT_R, :] = after_e[:, SLOT_E, :]
    before_a = alu_batch(model, before_copy_a, operations, operands, PRESENCE_P_E_R_W.expand(len(programs), -1), read_set="explicit")
    before_b = alu_batch(model, before_copy_b, operations, operands, PRESENCE_P_E_R_W.expand(len(programs), -1), read_set="explicit")
    return {
        "context_vs_reference_r_max_abs": max_abs(result["state"][:, SLOT_R], reference["state"][:, SLOT_R]),
        "p_preserved": bool(torch.equal(result["state"][:, SLOT_P], context[:, SLOT_P])),
        "e_preserved": bool(torch.equal(result["state"][:, SLOT_E], context[:, SLOT_E])),
        "w_preserved": bool(torch.equal(result["state"][:, 3], context[:, 3])),
        "after_copy_e_change_r_max_abs": max_abs(result["state"][:, SLOT_R], after_e_result["state"][:, SLOT_R]),
        "before_copy_changed_predictions": int((before_a["predicted"] != before_b["predicted"]).sum()),
        "before_copy_changed_r_max_abs": max_abs(before_a["state"][:, SLOT_R], before_b["state"][:, SLOT_R]),
    }


def exhaustive_table(model: C1JointModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    operations = torch.tensor([OPCODE_IDS[name] for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL") for _ in range(VALUE_COUNT * VALUE_COUNT)], dtype=torch.long)
    initial = torch.tensor([value for _ in range(3) for value in range(VALUE_COUNT) for _ in range(VALUE_COUNT)], dtype=torch.long)
    operands = torch.tensor([operand for _ in range(3) for _ in range(VALUE_COUNT) for operand in range(VALUE_COUNT)], dtype=torch.long)
    block = VALUE_COUNT * VALUE_COUNT
    add_targets = (initial[:block].view(VALUE_COUNT, VALUE_COUNT) + operands[:block].view(VALUE_COUNT, VALUE_COUNT)) % VALUE_COUNT
    sub_targets = (initial[block : 2 * block].view(VALUE_COUNT, VALUE_COUNT) - operands[block : 2 * block].view(VALUE_COUNT, VALUE_COUNT)) % VALUE_COUNT
    mul_targets = (initial[2 * block :].view(VALUE_COUNT, VALUE_COUNT) * operands[2 * block :].view(VALUE_COUNT, VALUE_COUNT)) % VALUE_COUNT
    target_values = torch.cat((add_targets.reshape(-1), sub_targets.reshape(-1), mul_targets.reshape(-1))).to(dtype=torch.long) + VALUE_BASE
    base_state = torch.zeros((len(initial), SLOT_COUNT, model.dimension))
    base_state[:, SLOT_R, :] = model.token_embedding(VALUE_BASE + initial)
    reference = alu_batch(model, base_state, operations, operands, PRESENCE_ONLY_R.expand(len(initial), -1), read_set="explicit")
    records: list[dict[str, Any]] = []
    context_specs = ("codebook", "random", "e_equals_r")
    for context_name in context_specs:
        context = base_state.clone()
        if context_name == "codebook":
            context[:, SLOT_P, :] = model.token_embedding((initial + 1) % 256)
            context[:, SLOT_E, :] = model.token_embedding(VALUE_BASE + ((initial + 5) % VALUE_COUNT))
            context[:, 3, :] = model.token_embedding(VALUE_BASE + ((initial + 9) % VALUE_COUNT))
        elif context_name == "random":
            generator = torch.Generator().manual_seed(1101)
            context[:, SLOT_P, :] = torch.randn((len(initial), model.dimension), generator=generator)
            context[:, SLOT_E, :] = torch.randn((len(initial), model.dimension), generator=generator)
            context[:, 3, :] = torch.randn((len(initial), model.dimension), generator=generator)
        else:
            context[:, SLOT_P, :] = model.token_embedding((initial + 13) % 256)
            context[:, SLOT_E, :] = context[:, SLOT_R, :]
            context[:, 3, :] = model.token_embedding(VALUE_BASE + ((initial + 21) % VALUE_COUNT))
        result = alu_batch(model, context, operations, operands, PRESENCE_P_E_R_W.expand(len(initial), -1), read_set="explicit")
        records.extend({"context": context_name, "index": index, "operation": int(operations[index]), "initial": int(initial[index]), "operand": int(operands[index]), "target_id": int(target_values[index]), "reference_prediction_id": int(reference["predicted"][index]), "explicit_prediction_id": int(result["predicted"][index]), "same_as_reference": bool(result["predicted"][index] == reference["predicted"][index])} for index in range(len(initial)))
    summary = {"transitions": len(initial), "contexts": len(context_specs), "reference_exact": int((reference["predicted"] == target_values).sum()), "context_cases": len(records), "context_matches_reference": sum(record["same_as_reference"] for record in records), "expected_reference": "3072/3072", "expected_context_matches": "9216/9216"}
    return summary, records


def gradient_check(model: C1JointModel) -> dict[str, Any]:
    state = torch.randn((1, SLOT_COUNT, model.dimension), requires_grad=True)
    operations = torch.tensor([OPCODE_IDS["ALU_SUB"]])
    operands = torch.tensor([7])
    result = alu_batch(model, state, operations, operands, PRESENCE_P_E_R_W.unsqueeze(0), read_set="explicit")
    result["state"][0, SLOT_R].sum().backward()
    return {"grad_p_max_abs": float(state.grad[0, SLOT_P].abs().max()), "grad_r_max_abs": float(state.grad[0, SLOT_R].abs().max()), "grad_e_max_abs": float(state.grad[0, SLOT_E].abs().max()), "grad_w_max_abs": float(state.grad[0, 3].abs().max()), "excluded_grads_zero": bool(state.grad[0, SLOT_P].abs().max() == 0 and state.grad[0, SLOT_E].abs().max() == 0 and state.grad[0, 3].abs().max() == 0)}


@torch.no_grad()
def mixed_batch_check(model: C1JointModel) -> dict[str, Any]:
    """Ensure explicit policy is per-sample when opcodes share one batch."""
    operations = torch.tensor([OPCODE_IDS["ALU_ADD"], OPCODE_IDS["READ_P"], OPCODE_IDS["ALU_SUB"], OPCODE_IDS["EMIT"]])
    state = torch.zeros((4, SLOT_COUNT, model.dimension))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([1, 1, 1, 1]))
    state[:, SLOT_E] = model.token_embedding(torch.tensor([VALUE_BASE + 2] * 4))
    state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + 3, VALUE_BASE + 4, VALUE_BASE + 5, VALUE_BASE + 6]))
    state[:, 3] = model.token_embedding(torch.tensor([VALUE_BASE + 7] * 4))
    memory_keys = model.token_embedding(torch.tensor([1])).expand(4, 1, -1)
    memory_values = model.token_embedding(torch.tensor([2])).expand(4, 1, -1)
    memory_types = torch.full((4, 1), ROW_REL, dtype=torch.long)
    row_mask = torch.ones((4, 1), dtype=torch.bool)
    immediate_ids = torch.tensor([3, IMM_ZERO, 5, IMM_ZERO])
    immediate = immediate_vectors(model, immediate_ids)
    sources = torch.tensor([SLOT_R, SLOT_P, SLOT_R, SLOT_R])
    destinations = torch.tensor([SLOT_R, SLOT_P, SLOT_R, SLOT_R])
    presence = PRESENCE_P_E_R_W.expand(4, -1)
    mixed_state, _, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, operations, immediate, sources, destinations, presence, read_mode=torch.zeros(4, dtype=torch.long), read_set="explicit")
    separate_states = []
    for index in range(4):
        single, _, _ = model.step(state[index:index + 1], memory_keys[index:index + 1], memory_values[index:index + 1], memory_types[index:index + 1], row_mask[index:index + 1], operations[index:index + 1], immediate[index:index + 1], sources[index:index + 1], destinations[index:index + 1], presence[index:index + 1], read_mode=torch.zeros(1, dtype=torch.long), read_set="explicit")
        separate_states.append(single.squeeze(0))
    separate = torch.stack(separate_states)
    mixed_predictions = decode(model, mixed_state)[1]
    separate_predictions = decode(model, separate)[1]
    return {"opcodes": ["ALU_ADD", "READ_P", "ALU_SUB", "EMIT"], "max_state_abs_batch_vs_separate": max_abs(mixed_state, separate), "batch_allclose_separate": bool(torch.allclose(mixed_state, separate, rtol=1e-5, atol=1e-5)), "batch_matches_separate_bitwise": bool(torch.equal(mixed_state, separate)), "predictions_match": bool(torch.equal(mixed_predictions, separate_predictions))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_mix_o_e_r_alu_read_set_seed101_frozen")
    parser.add_argument("--per-operation", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = load_approved_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    programs = alu.make_programs(args.per_operation, args.seed)
    bridge, bridge_records = activation_bridge(model, programs)
    invariance = invariance_check(model, programs)
    exhaustive, exhaustive_records = exhaustive_table(model)
    gradients = gradient_check(model)
    mixed_batch = mixed_batch_check(model)
    full_legacy = [alu.run_program(model, program, mode="baseline", read_set="legacy") for program in programs]
    full_explicit = [alu.run_program(model, program, mode="baseline", read_set="explicit") for program in programs]
    full_program = {"legacy_exact": sum(result["exact_hit"] for result in full_legacy), "explicit_exact": sum(result["exact_hit"] for result in full_explicit), "samples": len(programs), "explicit_read_set": "ALU sources R only; P/E remain present and COPY uses effective E"}
    summary = {"status": "completed", "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "read_set": "explicit", "legacy_contract": "legacy", "programs": len(programs), "bridge": bridge, "full_program": full_program, "invariance": invariance, "exhaustive": exhaustive, "gradient": gradients, "mixed_batch": mixed_batch}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "bridge_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in bridge_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (args.output_dir / "exhaustive_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in exhaustive_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (args.output_dir / "full_program_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for program, legacy, explicit in zip(programs, full_legacy, full_explicit):
            stream.write(json.dumps({"program": program.index, "legacy": {"target_id": legacy["target_id"], "predicted_id": legacy["predicted_id"], "exact_hit": legacy["exact_hit"]}, "explicit": {"target_id": explicit["target_id"], "predicted_id": explicit["predicted_id"], "exact_hit": explicit["exact_hit"]}}, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
