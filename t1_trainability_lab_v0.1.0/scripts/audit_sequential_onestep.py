"""Audit all 3072 elementary ALU transitions with teacher-forced H=1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_u0a import VALUE_BASE, VALUE_CLASS_IDS, immediate_vectors, save_json  # noqa: E402
from t1_trainability.unified import OPCODE_IDS, SLOT_R, UnifiedT1U0  # noqa: E402


OPERATIONS = ("ALU_ADD", "ALU_SUB", "ALU_MUL")
VALUE_IDS = torch.tensor(VALUE_CLASS_IDS, dtype=torch.long)


def load_model(path: Path) -> UnifiedT1U0:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def build_transitions() -> list[tuple[str, int, int, int]]:
    rows: list[tuple[str, int, int, int]] = []
    for operation in OPERATIONS:
        for initial in range(32):
            for operand in range(32):
                if operation == "ALU_ADD":
                    target = (initial + operand) % 32
                elif operation == "ALU_SUB":
                    target = (initial - operand) % 32
                else:
                    target = (initial * operand) % 32
                rows.append((operation, initial, operand, target))
    return rows


@torch.no_grad()
def audit(model: UnifiedT1U0, rows: list[tuple[str, int, int, int]]) -> dict[str, object]:
    codebook = model.token_embedding(VALUE_IDS)
    buckets: dict[str, list[dict[str, float]]] = {operation: [] for operation in OPERATIONS}
    for start in range(0, len(rows), 256):
        chunk = rows[start : start + 256]
        operation_ids = torch.tensor([OPCODE_IDS[row[0]] for row in chunk], dtype=torch.long)
        initial_ids = torch.tensor([VALUE_BASE + row[1] for row in chunk], dtype=torch.long)
        operand_ids = torch.tensor([VALUE_BASE + row[2] for row in chunk], dtype=torch.long)
        target_ids = torch.tensor([VALUE_BASE + row[3] for row in chunk], dtype=torch.long)
        state = torch.zeros(len(chunk), 4, model.dimension)
        state[:, SLOT_R] = model.token_embedding(initial_ids)
        presence = torch.tensor([[False, True, False, False]] * len(chunk), dtype=torch.bool)
        state, _, _ = model.step(
            state,
            torch.zeros(len(chunk), 1, model.dimension),
            torch.zeros(len(chunk), 1, model.dimension),
            torch.zeros(len(chunk), 1, dtype=torch.long),
            torch.zeros(len(chunk), 1, dtype=torch.bool),
            operation_ids,
            immediate_vectors(model, operand_ids),
            torch.full((len(chunk),), SLOT_R, dtype=torch.long),
            torch.full((len(chunk),), SLOT_R, dtype=torch.long),
            presence,
        )
        output = state[:, SLOT_R]
        logits = model.register_decoder(output, codebook)
        prediction_index = logits.argmax(dim=-1)
        predicted = VALUE_IDS[prediction_index]
        target_vectors = model.token_embedding(target_ids)
        cosine_matrix = F.normalize(output, dim=-1) @ F.normalize(codebook, dim=-1).transpose(0, 1)
        nearest_cosine = cosine_matrix.max(dim=-1).values
        nearest_distance = torch.cdist(output, codebook).min(dim=-1).values
        top_logits = logits.topk(2, dim=-1).values
        margins = top_logits[:, 0] - top_logits[:, 1]
        target_distance = torch.linalg.vector_norm(output - target_vectors, dim=-1)
        for index, row in enumerate(chunk):
            buckets[row[0]].append(
                {
                    "correct": float(predicted[index] == target_ids[index]),
                    "nearest_codebook_cosine": float(nearest_cosine[index]),
                    "decoder_margin": float(margins[index]),
                    "target_embedding_distance": float(target_distance[index]),
                    "nearest_codebook_distance": float(nearest_distance[index]),
                }
            )

    def summarize(values: list[dict[str, float]]) -> dict[str, float | int]:
        return {
            "accuracy": sum(row["correct"] for row in values) / len(values),
            "correct": int(sum(row["correct"] for row in values)),
            "total": len(values),
            "mean_nearest_codebook_cosine": sum(row["nearest_codebook_cosine"] for row in values) / len(values),
            "mean_decoder_margin": sum(row["decoder_margin"] for row in values) / len(values),
            "mean_target_embedding_distance": sum(row["target_embedding_distance"] for row in values) / len(values),
            "mean_nearest_codebook_distance": sum(row["nearest_codebook_distance"] for row in values) / len(values),
        }

    summaries = {operation: summarize(values) for operation, values in buckets.items()}
    all_values = [value for values in buckets.values() for value in values]
    return {"transitions": len(rows), "teacher_forced_horizon": 1, "by_operation": summaries, "overall": summarize(all_values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    result = {"checkpoint": str(args.checkpoint), "retrained": False, **audit(model, build_transitions())}
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
