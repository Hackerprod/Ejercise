"""Frozen T1-MIX-O evaluator for model-produced pointer composition.

This runner is deliberately inference-only.  Symbolic execution builds targets
and intervention references, while the model receives only the initial P and
the oracle instruction fields for each declared round.
"""

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
sys.path.insert(0, str(ROOT))

from train_u0c_c0_oracle import apply_transform  # noqa: E402
from train_u0c_c1_joint import C1JointModel, load_approved_model  # noqa: E402
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    READ_MODE_BLEND,
    READ_MODE_SELECT,
    ROW_REL,
    ROW_VEC,
    SLOT_COUNT,
    SLOT_P,
    SLOT_W,
)


DIMENSION = 64
IMM_ZERO = 511
TRANSFORM_COUNT = 4
TRANSFORM_NAMES = ("identity", "negation", "circular_shift", "pair_signed_permutation")


@dataclass
class Program:
    length: int
    keys: tuple[int, ...]
    evidence: tuple[Tensor, ...]
    transforms: tuple[int, ...]
    memory_keys: Tensor
    memory_values: Tensor
    memory_value_keys: Tensor
    memory_types: Tensor
    row_mask: Tensor
    rel_rows: tuple[int, ...]
    vec_rows: tuple[int, ...]
    alternate_keys: tuple[int, ...]
    alternate_evidence: tuple[Tensor, ...]

    @property
    def target(self) -> Tensor:
        return torch.stack(
            [apply_transform(value, torch.tensor(transform)) for value, transform in zip(self.evidence, self.transforms)]
        ).sum(dim=0)


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_programs(count: int, seed: int) -> list[Program]:
    generator = torch.Generator().manual_seed(seed)
    randomizer = random.Random(seed)
    programs: list[Program] = []
    for index in range(count):
        length = 2 if index < count // 2 else 3
        key_pool = randomizer.sample(range(256), 2 * (length + 1) + 8)
        keys = tuple(key_pool[: length + 1])
        alternate_keys = tuple(key_pool[length + 1 : 2 * (length + 1)])
        evidence = tuple(torch.randn(DIMENSION, generator=generator) for _ in range(length))
        alternate_evidence = tuple(torch.randn(DIMENSION, generator=generator) for _ in range(length))
        transforms = tuple(randomizer.randrange(TRANSFORM_COUNT) for _ in range(length))

        # Base branch plus an alternate branch.  Extra rows are legal distractors;
        # no row mask is narrowed to the answer.
        rows: list[tuple[int, int, Tensor, int]] = []
        for step in range(length):
            rows.append((keys[step], keys[step + 1], torch.zeros(DIMENSION), ROW_REL))
            rows.append((keys[step + 1], keys[step + 1], evidence[step], ROW_VEC))
            rows.append((alternate_keys[step], alternate_keys[step + 1], torch.zeros(DIMENSION), ROW_REL))
            rows.append((alternate_keys[step + 1], alternate_keys[step + 1], alternate_evidence[step], ROW_VEC))
        for distractor in key_pool[2 * (length + 1) :]:
            rows.append((distractor, key_pool[0], torch.zeros(DIMENSION), ROW_REL))
            rows.append((distractor, distractor, torch.randn(DIMENSION, generator=generator), ROW_VEC))
        randomizer.shuffle(rows)

        memory_keys = torch.stack([torch.nn.functional.one_hot(torch.tensor(key), 256).float() for key, _, _, _ in rows])
        # Replace one-hot keys with logical IDs later through the model codebook.
        # Keeping IDs here avoids giving the executor any learned representation.
        memory_values = torch.stack(
            [torch.zeros(DIMENSION) if row_type == ROW_REL else value for _, _, value, row_type in rows]
        )
        memory_value_keys = torch.tensor([value_key if row_type == ROW_REL else -1 for _, value_key, _, row_type in rows], dtype=torch.long)
        memory_types = torch.tensor([row_type for _, _, _, row_type in rows], dtype=torch.long)
        rel_rows = tuple(row for row, (_, _, _, row_type) in enumerate(rows) if row_type == ROW_REL)
        vec_rows = tuple(row for row, (_, _, _, row_type) in enumerate(rows) if row_type == ROW_VEC)
        programs.append(
            Program(
                length,
                keys,
                evidence,
                transforms,
                memory_keys,
                memory_values,
                memory_value_keys,
                memory_types,
                torch.ones(len(rows), dtype=torch.bool),
                rel_rows,
                vec_rows,
                alternate_keys,
                alternate_evidence,
            )
        )
    return programs


def build_model_memory(model: C1JointModel, program: Program) -> tuple[Tensor, Tensor]:
    """Materialize canonical memory using frozen model token codebook."""
    # memory_keys stores one-hot IDs only to keep generation independent of model.
    key_ids = program.memory_keys.argmax(dim=-1)
    key_vectors = model.token_embedding(key_ids)
    values = program.memory_values.clone()
    rel_mask = program.memory_types == ROW_REL
    values[rel_mask] = model.token_embedding(program.memory_value_keys[rel_mask])
    return key_vectors, values


def symbolic_trace(program: Program, *, alternate: bool = False) -> dict[str, Any]:
    keys = program.alternate_keys if alternate else program.keys
    evidence = program.alternate_evidence if alternate else program.evidence
    pointers = [keys[0]]
    for step in range(program.length):
        pointers.append(keys[step + 1])
    deltas = [
        apply_transform(value, torch.tensor(transform))
        for value, transform in zip(evidence, program.transforms)
    ]
    return {
        "pointers": pointers,
        "evidence_norms": [float(value.norm()) for value in evidence],
        "deltas": [delta.tolist() for delta in deltas],
        "target": torch.stack(deltas).sum(dim=0).tolist(),
    }


def cosine(left: Tensor, right: Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.view(1, -1), right.view(1, -1)).item())


def rel_error(left: Tensor, right: Tensor) -> float:
    return float((left - right).norm().div(right.norm().clamp_min(1e-8)).item())


@torch.no_grad()
def run_program(
    model: C1JointModel,
    program: Program,
    *,
    mode: str = "baseline",
    extra_emit: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    memory_keys, memory_values = build_model_memory(model, program)
    memory_types = program.memory_types.clone().unsqueeze(0)
    row_mask = program.row_mask.clone().unsqueeze(0)
    if mode == "memory_intervention":
        # Change REL(a,b) to REL(a,b_alt), leaving all legal rows and distractors.
        key_ids = program.memory_keys.argmax(dim=-1)
        relation_row = next(row for row in program.rel_rows if int(key_ids[row]) == program.keys[0])
        memory_values[relation_row] = model.token_embedding(torch.tensor(program.alternate_keys[1]))

    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = model.token_embedding(torch.tensor(program.keys[0]))
    presence = torch.tensor([[True, False, False, True]])
    immediate = torch.tensor([IMM_ZERO])
    trace: list[dict[str, Any]] = []
    if mode == "pointer_intervention":
        expected = symbolic_trace(program)
        pointer_target_values = (program.evidence[0], *program.alternate_evidence[1:])
        pointer_deltas = [
            apply_transform(value, torch.tensor(transform))
            for value, transform in zip(pointer_target_values, program.transforms)
        ]
        expected["pointers"] = [program.keys[0], program.keys[1], *program.alternate_keys[2:]]
        expected["target"] = torch.stack(pointer_deltas).sum(dim=0).tolist()
    else:
        expected = symbolic_trace(program, alternate=mode == "memory_intervention")
        if mode == "memory_intervention":
            expected["pointers"] = [program.keys[0], *program.alternate_keys[1:]]
    expected_pointers = expected["pointers"]
    target = torch.tensor(expected["target"])
    pointer_intervention_key = program.alternate_keys[1]
    memory_key_ids = program.memory_keys.argmax(dim=-1)

    instruction_count = program.length * 2
    for instruction_index in range(instruction_count + (1 if extra_emit else 0)):
        is_read_p = instruction_index % 2 == 0 and instruction_index < instruction_count
        is_accum = instruction_index % 2 == 1 and instruction_index < instruction_count
        opcode_name = "READ_P" if is_read_p else ("ACCUM_W" if is_accum else "EMIT")
        opcode = torch.tensor([OPCODE_IDS[opcode_name]])
        source_slot = torch.tensor([SLOT_P])
        destination_slot = torch.tensor([SLOT_P if is_read_p else SLOT_W])
        read_mode = torch.tensor([READ_MODE_BLEND if is_read_p else READ_MODE_SELECT])
        before = state.clone()
        if mode == "pointer_intervention" and instruction_index == 2:
            state[:, SLOT_P] = model.token_embedding(torch.tensor(pointer_intervention_key))
        previous_pointer = state[:, SLOT_P].clone()
        state, _, result = model.step(
            state,
            memory_keys,
            memory_values,
            memory_types,
            row_mask,
            opcode,
            immediate,
            source_slot,
            destination_slot,
            presence,
            read_mode=read_mode,
            transform_id=(torch.tensor([program.transforms[instruction_index // 2]]) if is_accum else None),
            correction_module=(model.correction_mlp if is_accum else None),
        )
        if mode == "freeze_p" and is_read_p:
            state[:, SLOT_P] = previous_pointer
        delta = state[:, SLOT_W] - before[:, SLOT_W]
        if mode == "replace_w" and is_accum:
            state[:, SLOT_W] = delta
            delta = state[:, SLOT_W] - before[:, SLOT_W]
        step_record: dict[str, Any] = {
            "instruction": instruction_index + 1,
            "opcode": opcode_name,
            "read_mode": "BLEND" if is_read_p else "SELECT",
            "transform": TRANSFORM_NAMES[program.transforms[instruction_index // 2]] if is_accum else None,
            "p_before": before[:, SLOT_P].squeeze(0).tolist(),
            "p_after": state[:, SLOT_P].squeeze(0).tolist(),
            "w_before": before[:, SLOT_W].squeeze(0).tolist(),
            "w_after": state[:, SLOT_W].squeeze(0).tolist(),
            "selected_row": int(result.selected_index.item()),
            "selected_margin": float(result.selection_margin.item()),
            "payload": result.payload.squeeze(0).tolist(),
            "payload_valid": bool(result.valid.item()),
            "delta": delta.squeeze(0).tolist(),
            "p_intact": bool(torch.equal(state[:, SLOT_P], before[:, SLOT_P])) if not is_read_p else True,
            "w_intact": bool(torch.equal(state[:, SLOT_W], before[:, SLOT_W])) if not is_accum else True,
        }
        if is_read_p:
            pointer_index = instruction_index // 2 + 1
            step_record["expected_pointer_key"] = expected_pointers[pointer_index]
            if mode == "pointer_intervention" and pointer_index == 2:
                source_key = program.alternate_keys[1]
            elif mode == "pointer_intervention" and pointer_index > 2:
                source_key = program.alternate_keys[pointer_index - 1]
            else:
                source_key = expected_pointers[pointer_index - 1]
            expected_rows = [
                row for row in program.rel_rows if int(memory_key_ids[row]) == source_key
            ]
            step_record["expected_row"] = expected_rows[0] if expected_rows else -1
            step_record["row_hit"] = bool(step_record["selected_row"] in expected_rows)
            step_record["pointer_cosine_to_expected"] = cosine(
                state[:, SLOT_P], model.token_embedding(torch.tensor(expected_pointers[pointer_index]))
            )
        if is_accum:
            evidence_index = instruction_index // 2
            if mode == "pointer_intervention" and evidence_index > 0:
                expected_evidence = program.alternate_evidence[evidence_index]
            elif mode == "memory_intervention":
                expected_evidence = program.alternate_evidence[evidence_index]
            else:
                expected_evidence = program.evidence[evidence_index]
            expected_delta = apply_transform(expected_evidence, torch.tensor(program.transforms[evidence_index]))
            step_record["expected_payload_rel_error"] = rel_error(result.payload.squeeze(0), expected_evidence)
            step_record["expected_delta_rel_error"] = rel_error(delta.squeeze(0), expected_delta)
            expected_key = expected_pointers[evidence_index + 1]
            expected_rows = [
                row for row in program.vec_rows if int(memory_key_ids[row]) == expected_key
            ]
            step_record["expected_row"] = expected_rows[0] if expected_rows else -1
            step_record["row_hit"] = bool(step_record["selected_row"] in expected_rows)
        trace.append(step_record)

    output = state[:, SLOT_W].squeeze(0)
    row_decisions = [item["row_hit"] for item in trace if "row_hit" in item]
    return output, {
        "target": target.tolist(),
        "trace": trace,
        "cosine": cosine(output, target),
        "relative_error": rel_error(output, target),
        "row_hit_rate": sum(row_decisions) / max(1, len(row_decisions)),
        "row_hits": sum(row_decisions),
        "row_decisions": len(row_decisions),
    }


def summarize_runs(model: C1JointModel, programs: list[Program], *, controls: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mode_names = ("baseline", "pointer_intervention", "memory_intervention", "freeze_p", "replace_w") if controls else ("baseline",)
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in mode_names}
    traces: list[dict[str, Any]] = []
    for program_index, program in enumerate(programs):
        for mode in mode_names:
            output, result = run_program(model, program, mode=mode)
            runs[mode].append(result)
            traces.append({"program": program_index, "length": program.length, "keys": list(program.keys), "transforms": list(program.transforms), "mode": mode, **result})
        if controls:
            baseline_output, _ = run_program(model, program)
            emit_output, emit_result = run_program(model, program, extra_emit=True)
            runs.setdefault("extra_emit", []).append(emit_result)
            traces.append({"program": program_index, "length": program.length, "keys": list(program.keys), "transforms": list(program.transforms), "mode": "extra_emit", **emit_result})
            if not torch.equal(baseline_output, emit_output):
                raise AssertionError("EMIT changed workspace state")

    def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [item["relative_error"] for item in results]
        cosines = [item["cosine"] for item in results]
        by_length = {
            str(length): {
                "samples": sum(program.length == length for program in programs),
                "pass_count": sum(program.length == length and item["cosine"] > 0.999 and item["relative_error"] <= 0.01 for program, item in zip(programs, results)),
                "worst_relative_error": max(item["relative_error"] for program, item in zip(programs, results) if program.length == length),
                "worst_cosine": min(item["cosine"] for program, item in zip(programs, results) if program.length == length),
                "row_hit_rate": sum(item["row_hits"] for program, item in zip(programs, results) if program.length == length) / max(1, sum(item["row_decisions"] for program, item in zip(programs, results) if program.length == length)),
            }
            for length in (2, 3)
        }
        return {
            "samples": len(results),
            "pass_count": sum(c > 0.999 and e <= 0.01 for c, e in zip(cosines, errors)),
            "worst_relative_error": max(errors),
            "worst_cosine": min(cosines),
            "row_hit_rate": sum(item["row_hits"] for item in results) / max(1, sum(item["row_decisions"] for item in results)),
            "by_length": by_length,
        }

    summary = {"gate": {"cosine_gt": 0.999, "relative_error_lte": 0.01}, "runs": {name: aggregate(values) for name, values in runs.items()}, "state_conservation": {"freeze_p": "must fail chain advancement", "replace_w": "must lose prior contributions", "extra_emit": "must preserve output"} if controls else {}}
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0c_c1_mix_o_seed101_frozen")
    parser.add_argument("--samples", type=int, default=32, help="Total programs; half length 4-update, half length 6-update")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only, for larger regression sampling")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise ValueError("--samples must be an even number >= 2")
    torch.manual_seed(args.seed)
    model = load_approved_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    programs = make_programs(args.samples, args.seed)
    summary, traces = summarize_runs(model, programs, controls=not args.baseline_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.update({"status": "completed", "seed": args.seed, "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "samples": args.samples, "program_updates": {"4": args.samples // 2, "6": args.samples // 2}, "target_source": "independent symbolic interpreter; model never receives target, expected pointer, physical row, or transformed vector", "memory_manifest": [{"program": index, "length": program.length, "memory_rows": int(program.memory_types.shape[0]), "keys": list(program.keys), "transforms": list(program.transforms), "memory_types_sha256": tensor_sha256(program.memory_types)} for index, program in enumerate(programs)]})
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
