import pytest

from t2_i0_instruction import parse_instruction, tokenize_instruction


def test_vocab_has_36_symbols() -> None:
    parsed = parse_instruction("AT_LEAST VALUE_14 AND AVOID VALUE_12")
    assert len(parsed.token_ids) == 5
    assert len(set(parsed.token_ids)) == 5


@pytest.mark.parametrize(
    ("instruction", "constraints", "lower", "forbidden"),
    (
        ("KEEP", (0, 0), 0, 0),
        ("AT_LEAST VALUE_31", (1, 0), 31, 0),
        ("AVOID VALUE_30", (0, 1), 0, 30),
        ("AT_LEAST VALUE_14 AND AVOID VALUE_12", (1, 1), 14, 12),
    ),
)
def test_parse_instruction_returns_ctrl_bits_and_arguments(instruction, constraints, lower, forbidden) -> None:
    parsed = parse_instruction(instruction)
    assert parsed.constraints == constraints
    assert parsed.lower == lower
    assert parsed.forbidden == forbidden


def test_tokenizer_rejects_unknown_and_noncanonical_grammar() -> None:
    with pytest.raises(ValueError):
        tokenize_instruction("AT_MOST VALUE_3")
    with pytest.raises(ValueError):
        parse_instruction("AVOID VALUE_3 AND AT_LEAST VALUE_2")
