"""T2-I0 oracle instruction tokenizer/parser for Baseline A plumbing."""

from __future__ import annotations

from dataclasses import dataclass
import re


OPERATORS = ("KEEP", "AT_LEAST", "AVOID", "AND")
VALUE_TOKENS = tuple(f"VALUE_{value}" for value in range(32))
VOCABULARY = OPERATORS + VALUE_TOKENS
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
_VALUE_PATTERN = re.compile(r"VALUE_([0-9]|[12][0-9]|3[01])\Z")


@dataclass(frozen=True)
class ParsedInstruction:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    constraints: tuple[int, int]
    lower: int
    forbidden: int


def tokenize_instruction(instruction: str) -> tuple[str, ...]:
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    tokens = tuple(instruction.strip().split())
    if not tokens or any(token not in TOKEN_IDS for token in tokens):
        raise ValueError(f"unknown instruction token(s): {instruction!r}")
    return tokens


def _value(token: str) -> int:
    match = _VALUE_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"expected VALUE_0..VALUE_31, got {token!r}")
    return int(match.group(1))


def parse_instruction(instruction: str) -> ParsedInstruction:
    tokens = tokenize_instruction(instruction)
    constraints = (0, 0)
    lower = 0
    forbidden = 0
    if tokens == ("KEEP",):
        pass
    elif len(tokens) == 2 and tokens[0] == "AT_LEAST":
        lower = _value(tokens[1]); constraints = (1, 0)
    elif len(tokens) == 2 and tokens[0] == "AVOID":
        forbidden = _value(tokens[1]); constraints = (0, 1)
    elif len(tokens) == 5 and tokens[0] == "AT_LEAST" and tokens[2:] == ("AND", "AVOID", tokens[4]):
        lower = _value(tokens[1]); forbidden = _value(tokens[4]); constraints = (1, 1)
    else:
        raise ValueError(f"invalid instruction grammar: {instruction!r}")
    return ParsedInstruction(tokens=tokens, token_ids=tuple(TOKEN_IDS[token] for token in tokens), constraints=constraints, lower=lower, forbidden=forbidden)
