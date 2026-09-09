import pytest

from t2_i0_instruction_r1 import parse_instruction_r1


@pytest.mark.parametrize(
    ("instruction", "constraints", "lower", "forbidden"),
    (
        ("NOOP VALUE_0", (0, 0), 0, 0),
        ("NOOP VALUE_0 AND NOOP VALUE_0", (0, 0), 0, 0),
        ("AT_LEAST VALUE_14 AND NOOP VALUE_5", (1, 0), 14, 0),
        ("NOOP VALUE_5 AND AT_LEAST VALUE_14", (1, 0), 14, 0),
        ("AVOID VALUE_12 AND NOOP VALUE_5", (0, 1), 0, 12),
        ("NOOP VALUE_5 AND AVOID VALUE_12", (0, 1), 0, 12),
        ("AT_LEAST VALUE_14 AND AVOID VALUE_12", (1, 1), 14, 12),
        ("AVOID VALUE_12 AND AT_LEAST VALUE_14", (1, 1), 14, 12),
    ),
)
def test_r1_parser_semantics(instruction, constraints, lower, forbidden) -> None:
    parsed = parse_instruction_r1(instruction)
    assert parsed.constraints == constraints
    assert parsed.lower == lower
    assert parsed.forbidden == forbidden


def test_r1_vocab_has_no_ambiguous_unknown_tokens() -> None:
    assert len(parse_instruction_r1("NOOP VALUE_0").token_ids) == 2
    with pytest.raises(ValueError):
        parse_instruction_r1("AT_LEAST VALUE_1 OR NOOP VALUE_0")
