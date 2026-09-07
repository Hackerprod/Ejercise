from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ctrl2_common import INCREASE, KEEP
from evaluate_u0c_ctrl3 import validate_transition


def test_rejects_wrong_alu_output_that_still_reduces_distance() -> None:
    result = validate_transition(
        x_before=10,
        x_after=13,
        reference=12,
        selected_action=INCREASE,
        expected_action=INCREASE,
        emitted=False,
        raw_before_sha256="before",
        raw_after_sha256="after",
    )
    assert result["action_correct"]
    assert result["distance_decreased"]
    assert not result["exact_operation"]
    assert not result["valid"]


def test_keep_requires_emit_and_unchanged_register() -> None:
    valid = validate_transition(x_before=10, x_after=10, reference=10, selected_action=KEEP, expected_action=KEEP, emitted=True, raw_before_sha256="same", raw_after_sha256="same")
    changed = validate_transition(x_before=10, x_after=10, reference=10, selected_action=KEEP, expected_action=KEEP, emitted=True, raw_before_sha256="before", raw_after_sha256="after")
    assert valid["valid"]
    assert not changed["valid"]
