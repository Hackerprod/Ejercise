"""Activation and causal-intervention audit for the frozen E->R->ALU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_u0c_c1_e_r_alu as base  # noqa: E402
from train_u0c_c1_joint import C1JointModel, load_approved_model  # noqa: E402
from train_u0a import immediate_vectors  # noqa: E402
from t1_trainability.unified import OPCODE_IDS, SLOT_E, SLOT_P, SLOT_R, SLOT_COUNT  # noqa: E402


VALUE_BASE = 288
VALUE_COUNT = 32
IMM_ZERO = 511
PRESENCE_ONLY_R = torch.tensor([[False, True, False, False]])
PRESENCE_P_E_R = torch.tensor([[True, True, True, False]])


def signed_margin(logits: Tensor, target_position: int) -> float:
    target = logits[target_position]
    other = torch.cat((logits[:target_position], logits[target_position + 1 :]))
    return float((target - other.max()).item())


def tensor_metrics(left: Tensor, right: Tensor) -> dict[str, float]:
    difference = left - right
    return {"max_abs": float(difference.abs().max().item()), "l2": float(difference.norm().item()), "left_norm": float(left.norm().item()), "right_norm": float(right.norm().item())}


def core_diagnostics(
    model: C1JointModel,
    state: Tensor,
    opcode: Tensor,
    operand_vector: Tensor,
    presence: Tensor,
    *,
    mixed_override_r: Tensor | None = None,
    r_only_attention: bool = False,
) -> dict[str, Tensor]:
    """Mirror SharedRecurrentCore.forward while exposing internal tensors."""
    core = model.core
    normalized_state = model.normalize_state(state, presence)
    opcode_embedding = model.opcode_embedding(opcode.to(dtype=torch.long))
    read_payload = torch.zeros_like(state[:, SLOT_R, :])
    condition = core.condition(torch.cat((opcode_embedding, operand_vector, read_payload), dim=-1)).unsqueeze(1)
    present = presence.to(dtype=state.dtype).unsqueeze(-1)
    conditioned = (normalized_state + model.slot_type_embeddings + condition) * present

    left = core.alu_left_projection(normalized_state[:, SLOT_R, :])
    right = core.alu_right_projection(operand_vector)
    op_state = core.alu_opcode_projection(opcode_embedding)
    alu_features = torch.cat((left, right, op_state, left - right), dim=-1)
    alu_delta = torch.zeros_like(conditioned[:, SLOT_R, :])
    for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL"):
        if int(opcode.item()) == OPCODE_IDS[name]:
            adapter = core.alu_adapters[name]
            alu_delta = adapter["up"](adapter["down"](alu_features))
    conditioned = conditioned.clone()
    conditioned[:, SLOT_R, :] = conditioned[:, SLOT_R, :] + alu_delta

    query = core.query(conditioned)
    key = core.key(conditioned)
    value = core.value(conditioned)
    scores = torch.matmul(query, key.transpose(-2, -1)) * core.scale
    scores = scores.masked_fill(~presence.unsqueeze(1), torch.finfo(scores.dtype).min)
    if r_only_attention:
        read_mask = torch.ones_like(presence.unsqueeze(1), dtype=torch.bool).expand(-1, SLOT_COUNT, -1).clone()
        read_mask[:, SLOT_R, :] = False
        read_mask[:, SLOT_R, SLOT_R] = True
        scores = scores.masked_fill(~read_mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=-1)
    attention = torch.nan_to_num(attention)
    mixed = core.output(torch.matmul(attention, value)) * present
    if mixed_override_r is not None:
        mixed = mixed.clone()
        mixed[:, SLOT_R, :] = mixed_override_r
    candidate = core.mlp(core.norm(conditioned + mixed)) * present
    return {
        "normalized_state": normalized_state,
        "alu_features": alu_features,
        "alu_delta": alu_delta,
        "conditioned": conditioned,
        "query": query,
        "key": key,
        "value": value,
        "attention": attention,
        "mixed": mixed,
        "candidate": candidate,
    }


def alu_outputs(model: C1JointModel, diagnostics: dict[str, Tensor], opcode: Tensor, target_id: int) -> dict[str, Any]:
    codebook = model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))
    candidate_r = diagnostics["candidate"][:, SLOT_R, :]
    head_logits = model.commit.select_alu_logits(candidate_r, opcode).squeeze(0)
    register_r = torch.softmax(head_logits, dim=-1) @ codebook
    decoder_logits = model.register_decoder(register_r.unsqueeze(0), codebook).squeeze(0)
    target_position = target_id - VALUE_BASE
    return {
        "head_logits": head_logits,
        "register_r": register_r,
        "decoder_logits": decoder_logits,
        "head_prediction_id": VALUE_BASE + int(head_logits.argmax().item()),
        "decoder_prediction_id": VALUE_BASE + int(decoder_logits.argmax().item()),
        "head_margin": signed_margin(head_logits, target_position),
        "decoder_margin": signed_margin(decoder_logits, target_position),
    }


def read_context(model: C1JointModel, program: base.Program) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    memory_keys, memory_values, memory_types, row_mask = base.materialize_memory(model, program.rows)
    state = torch.zeros((1, SLOT_COUNT, base.DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor([program.start_key + base.KEY_BASE]))
    state, _, read_p = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_P"]]), torch.tensor([IMM_ZERO]), torch.tensor([SLOT_P]), torch.tensor([SLOT_P]), PRESENCE_P_E_R, read_mode=torch.tensor([0]))
    state, _, read_e = model.step(state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([OPCODE_IDS["READ_E"]]), torch.tensor([IMM_ZERO]), torch.tensor([SLOT_P]), torch.tensor([SLOT_E]), PRESENCE_P_E_R, read_mode=torch.tensor([0]))
    state[:, SLOT_R] = model.token_embedding(torch.tensor([VALUE_BASE + program.pair_value]))
    return state, memory_keys, memory_values, memory_types, row_mask, torch.tensor([int(read_p.selected_index.item()), int(read_e.selected_index.item())])


def audit_pair(model: C1JointModel, program: base.Program) -> dict[str, Any]:
    context_state, memory_keys, memory_values, memory_types, row_mask, selected_rows = read_context(model, program)
    opcode = torch.tensor([OPCODE_IDS[program.operation]])
    operand_ids = torch.tensor([VALUE_BASE + program.operand], dtype=torch.long)
    operand_vector = immediate_vectors(model, operand_ids)
    reference_state = torch.zeros_like(context_state)
    reference_state[:, SLOT_R] = context_state[:, SLOT_R]
    reference = core_diagnostics(model, reference_state, opcode, operand_vector, PRESENCE_ONLY_R)
    context = core_diagnostics(model, context_state, opcode, operand_vector, PRESENCE_P_E_R)
    mixed_delta = context["mixed"][:, SLOT_R, :] - reference["mixed"][:, SLOT_R, :]
    attention = context["attention"][:, SLOT_R, :]
    values = context["value"]
    reconstructed = F.linear(
        attention[:, SLOT_P].unsqueeze(-1) * (values[:, SLOT_P, :] - values[:, SLOT_R, :])
        + attention[:, SLOT_E].unsqueeze(-1) * (values[:, SLOT_E, :] - values[:, SLOT_R, :]),
        model.core.output.weight,
        None,
    )
    target_id = VALUE_BASE + program.target_value
    reference_output = alu_outputs(model, reference, opcode, target_id)
    context_output = alu_outputs(model, context, opcode, target_id)
    override = core_diagnostics(model, context_state, opcode, operand_vector, PRESENCE_P_E_R, mixed_override_r=reference["mixed"][:, SLOT_R, :])
    override_output = alu_outputs(model, override, opcode, target_id)
    masked = core_diagnostics(model, context_state, opcode, operand_vector, PRESENCE_P_E_R, r_only_attention=True)
    masked_output = alu_outputs(model, masked, opcode, target_id)
    pre_mix_fields = ("normalized_state", "alu_features", "alu_delta", "conditioned", "query", "key", "value")
    def r_channel(tensor: Tensor) -> Tensor:
        return tensor if tensor.ndim == 2 else tensor[:, SLOT_R, :]
    pre_mix = {field: tensor_metrics(r_channel(context[field]), r_channel(reference[field])) for field in pre_mix_fields}
    return {
        "program": program.index,
        "operation": program.operation,
        "pair_value": program.pair_value,
        "operand": program.operand,
        "target_id": target_id,
        "selected_read_rows": selected_rows.tolist(),
        "pre_mix": pre_mix,
        "attention_r_to_p": float(attention[0, SLOT_P].item()),
        "attention_r_to_r": float(attention[0, SLOT_R].item()),
        "attention_r_to_e": float(attention[0, SLOT_E].item()),
        "delta_m_r_norm": float(mixed_delta.norm().item()),
        "delta_m_r_formula_norm": float(reconstructed.norm().item()),
        "delta_m_r_residual_norm": float((mixed_delta - reconstructed).norm().item()),
        "reference": {"head_prediction_id": reference_output["head_prediction_id"], "decoder_prediction_id": reference_output["decoder_prediction_id"], "head_margin": reference_output["head_margin"], "decoder_margin": reference_output["decoder_margin"], "head_logits": reference_output["head_logits"].tolist(), "decoder_logits": reference_output["decoder_logits"].tolist()},
        "context": {"head_prediction_id": context_output["head_prediction_id"], "decoder_prediction_id": context_output["decoder_prediction_id"], "head_margin": context_output["head_margin"], "decoder_margin": context_output["decoder_margin"], "head_logits": context_output["head_logits"].tolist(), "decoder_logits": context_output["decoder_logits"].tolist()},
        "mixed_override": {"head_prediction_id": override_output["head_prediction_id"], "decoder_prediction_id": override_output["decoder_prediction_id"], "head_margin": override_output["head_margin"], "decoder_margin": override_output["decoder_margin"], "head_logits": override_output["head_logits"].tolist(), "decoder_logits": override_output["decoder_logits"].tolist(), "head_logit_max_abs_vs_reference": float((override_output["head_logits"] - reference_output["head_logits"]).abs().max().item()), "decoder_logit_max_abs_vs_reference": float((override_output["decoder_logits"] - reference_output["decoder_logits"]).abs().max().item())},
        "r_only_mask": {"head_prediction_id": masked_output["head_prediction_id"], "decoder_prediction_id": masked_output["decoder_prediction_id"], "head_margin": masked_output["head_margin"], "decoder_margin": masked_output["decoder_margin"], "head_logits": masked_output["head_logits"].tolist(), "decoder_logits": masked_output["decoder_logits"].tolist(), "head_logit_max_abs_vs_reference": float((masked_output["head_logits"] - reference_output["head_logits"]).abs().max().item()), "decoder_logit_max_abs_vs_reference": float((masked_output["decoder_logits"] - reference_output["decoder_logits"]).abs().max().item())},
    }


def run_full_program_variant(model: C1JointModel, program: base.Program, *, mode: str) -> dict[str, Any]:
    state, memory_keys, memory_values, memory_types, row_mask, selected_rows = read_context(model, program)
    operand_vector = immediate_vectors(model, torch.tensor([VALUE_BASE + program.operand], dtype=torch.long))
    opcode = torch.tensor([OPCODE_IDS[program.operation]])
    if mode == "baseline":
        state, candidates, _ = model.step(state, memory_keys, memory_values, memory_types, row_mask, opcode, operand_vector, torch.tensor([SLOT_R]), torch.tensor([SLOT_R]), PRESENCE_P_E_R, read_mode=torch.tensor([0]))
        head_logits = candidates.alu_logits.squeeze(0)
    else:
        reference_state = torch.zeros_like(state)
        reference_state[:, SLOT_R] = state[:, SLOT_R]
        reference = core_diagnostics(model, reference_state, opcode, operand_vector, PRESENCE_ONLY_R)
        diagnostics = core_diagnostics(model, state, opcode, operand_vector, PRESENCE_P_E_R, mixed_override_r=reference["mixed"][:, SLOT_R, :] if mode == "mixed_override" else None, r_only_attention=mode == "r_only_mask")
        head_logits = model.commit.select_alu_logits(diagnostics["candidate"][:, SLOT_R, :], opcode).squeeze(0)
        codebook = model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))
        state[:, SLOT_R] = torch.softmax(head_logits, dim=-1) @ codebook
    codebook = model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))
    decoder_logits = model.register_decoder(state[:, SLOT_R, :], codebook).squeeze(0)
    target_id = VALUE_BASE + program.target_value
    return {"program": program.index, "mode": mode, "operation": program.operation, "target_id": target_id, "predicted_id": VALUE_BASE + int(decoder_logits.argmax().item()), "exact_hit": bool(int(decoder_logits.argmax().item()) == program.target_value), "selected_read_rows": selected_rows.tolist(), "head_logits": head_logits.tolist(), "decoder_logits": decoder_logits.tolist(), "r": state[:, SLOT_R].squeeze(0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_mix_o_e_r_alu_seed101_activation_audit")
    parser.add_argument("--per-operation", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = load_approved_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    programs = base.make_programs(args.per_operation, args.seed)
    pairs = [audit_pair(model, program) for program in programs]
    full_programs = [{"program": program.index, "operation": program.operation, "pair_value": program.pair_value, "operand": program.operand, "modes": [run_full_program_variant(model, program, mode=mode) for mode in ("baseline", "mixed_override", "r_only_mask")]} for program in programs]
    pre_mix_max = {field: max(item["pre_mix"][field]["max_abs"] for item in pairs) for field in ("normalized_state", "alu_features", "alu_delta", "conditioned", "query", "key", "value")}
    matrix = {
        "reference_only_r": {"exact_count": sum(item["reference"]["decoder_prediction_id"] == item["target_id"] for item in pairs), "samples": len(pairs)},
        "context_p_e_r": {"exact_count": sum(item["context"]["decoder_prediction_id"] == item["target_id"] for item in pairs), "samples": len(pairs)},
        "mixed_r_override": {"exact_count": sum(item["mixed_override"]["decoder_prediction_id"] == item["target_id"] for item in pairs), "samples": len(pairs)},
        "r_only_attention_mask": {"exact_count": sum(item["r_only_mask"]["decoder_prediction_id"] == item["target_id"] for item in pairs), "samples": len(pairs)},
    }
    full_summary = {mode: {"exact_count": sum(item["modes"][index]["exact_hit"] for item in full_programs), "samples": len(full_programs)} for index, mode in enumerate(("baseline", "mixed_override", "r_only_mask"))}
    activation_outputs = {
        mode: {
            "head_exact_count": sum(item[mode]["head_prediction_id"] == item["target_id"] for item in pairs),
            "decoder_exact_count": sum(item[mode]["decoder_prediction_id"] == item["target_id"] for item in pairs),
            "min_head_margin": min(item[mode]["head_margin"] for item in pairs),
            "min_decoder_margin": min(item[mode]["decoder_margin"] for item in pairs),
        }
        for mode in ("reference", "context", "mixed_override", "r_only_mask")
    }
    intervention_equivalence = {
        mode: {
            "max_head_logit_abs_vs_reference": max(item[mode]["head_logit_max_abs_vs_reference"] for item in pairs),
            "max_decoder_logit_abs_vs_reference": max(item[mode]["decoder_logit_max_abs_vs_reference"] for item in pairs),
        }
        for mode in ("mixed_override", "r_only_mask")
    }
    summary = {"status": "completed", "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "programs": len(programs), "coverage": {operation: args.per_operation for operation in base.TRANSFORM_OPS}, "presence_reference": [False, True, False, False], "presence_context": [True, True, True, False], "operand_route": "train_u0a.immediate_vectors -> token_embedding", "pre_mix_max_abs": pre_mix_max, "max_delta_m_r_residual_norm": max(item["delta_m_r_residual_norm"] for item in pairs), "matrix": matrix, "activation_outputs": activation_outputs, "intervention_equivalence": intervention_equivalence, "full_program": full_summary, "interpretation": "Only-R reference is exact; mixed_R substitution and R-only attention are causal diagnostics; full program uses actual READ_P, READ_E, and COPY."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "activation_pairs.jsonl").open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, sort_keys=True) + "\n")
    with (args.output_dir / "full_program.jsonl").open("w", encoding="utf-8") as stream:
        for item in full_programs:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
