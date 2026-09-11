"""Shared deterministic T2-I3 manifest and frozen component helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, tensorize
from t2_i3_think0 import Think0
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN = ROOT / "campaign"; WRITER_CHECKPOINT = CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"; CALIBRATION_SALT = "T2-I3-THINK0-CAL-v1"; MANIFEST_PATH = CAMPAIGN / "t2_i3_calibration_manifest.json"


def pair_digest(lower: int, forbidden: int) -> str: return hashlib.sha256(f"{CALIBRATION_SALT}|AT_LEAST|{lower}|AVOID|{forbidden}|NORMAL".encode("utf-8")).hexdigest()


def build_calibration_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    pairs = [{"lower": lower, "forbidden": forbidden, "digest": pair_digest(lower, forbidden), "bucket": int(pair_digest(lower, forbidden)[:16], 16) % 8} for lower in range(32) for forbidden in range(32) if lower != forbidden]; calibration = [pair for pair in pairs if pair["bucket"] == 0]; heldout = [pair for pair in pairs if pair["bucket"] != 0]; manifest = {"task": "T2-I3", "version": 1, "salt": CALIBRATION_SALT, "hash": "sha256", "bucket_rule": "int(digest[:16],16)%8", "calibration_bucket": 0, "candidate_count": len(pairs), "calibration_count": len(calibration), "heldout_count": len(heldout), "calibration": calibration, "heldout": heldout, "excluded_interaction": "lower == forbidden", "calibration_text": "AT_LEAST VALUE_L AND AVOID VALUE_F", "calibration_order": "NORMAL"}; path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"); manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest(); return manifest


def load_writer() -> CompetitiveSemanticWriter:
    writer = CompetitiveSemanticWriter(); writer.load_state_dict(torch.load(WRITER_CHECKPOINT, weights_only=False)["writer"], strict=True); writer.eval()
    for parameter in writer.parameters(): parameter.requires_grad = False
    return writer


def encode_writer(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    with torch.no_grad(): return writer(*tensorize([text]))


def load_think(checkpoint: Path | None = None) -> Think0:
    think = Think0()
    if checkpoint is not None: think.load_state_dict(torch.load(checkpoint, weights_only=False)["think"], strict=True)
    return think
