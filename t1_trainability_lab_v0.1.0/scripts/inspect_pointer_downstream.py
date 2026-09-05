"""Inspect pointer decoder/codebook alignment on oracle states, no training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_u0a import KEY_BASE, POINTER_CLASS_IDS, ExampleDataset, build_canonical_data, collate, materialize  # noqa: E402
from t1_trainability.unified import SLOT_P, UnifiedT1U0  # noqa: E402


@torch.no_grad()
def inspect(model: UnifiedT1U0, examples: list[object]) -> dict[str, float]:
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    class_ids = torch.tensor(POINTER_CLASS_IDS, dtype=torch.long)
    basis = F.normalize(model.token_embedding(class_ids), dim=-1)
    raw_hits = 0
    decoder_hits = 0
    count = 0
    for batch in loader:
        data = materialize(model, batch)
        target_ids = data["target_ids"]
        raw_state = model.token_embedding(target_ids)
        raw_logits = F.normalize(raw_state, dim=-1) @ basis.transpose(0, 1)
        decoder_logits = model.pointer_decoder(raw_state, model.token_embedding(class_ids))
        raw_hits += int((class_ids[raw_logits.argmax(-1)] == target_ids).sum())
        decoder_hits += int((class_ids[decoder_logits.argmax(-1)] == target_ids).sum())
        count += len(target_ids)
    return {"raw_codebook_nearest_accuracy": raw_hits / count, "decoder_accuracy_on_exact_payload": decoder_hits / count, "samples": count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0a_canaries_seed101_sameinit" / "pointer_chasing" / "best.pt")
    args = parser.parse_args()
    model = UnifiedT1U0(64)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=False)
    datasets = build_canonical_data(args.checkpoint.parents[1])
    print(inspect(model, datasets["pointer_chasing"]["test"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
