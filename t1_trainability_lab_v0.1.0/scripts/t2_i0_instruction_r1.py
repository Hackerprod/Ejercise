"""T2-I0-R1 structural-deconfounding instruction grammar."""

from __future__ import annotations

from dataclasses import dataclass
import re


OPERATORS = ("KEEP", "AT_LEAST", "AVOID", "AND", "NOOP")
VALUE_TOKENS = tuple(f"VALUE_{value}" for value in range(32))
VOCABULARY = OPERATORS + VALUE_TOKENS
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
VALUE_PATTERN = re.compile(r"VALUE_([0-9]|[12][0-9]|3[01])\Z")


@dataclass(frozen=True)
class ParsedInstructionR1:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    constraints: tuple[int, int]
    lower: int
    forbidden: int


def tokenize_instruction_r1(instruction: str) -> tuple[str, ...]:
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    tokens = tuple(instruction.strip().split())
    if not tokens or any(token not in TOKEN_IDS for token in tokens):
        raise ValueError(f"unknown instruction token(s): {instruction!r}")
    return tokens


def _value(token: str) -> int:
    match = VALUE_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"expected VALUE_0..VALUE_31, got {token!r}")
    return int(match.group(1))


def parse_instruction_r1(instruction: str) -> ParsedInstructionR1:
    tokens = tokenize_instruction_r1(instruction)
    if tokens == ("KEEP",):
        return ParsedInstructionR1(tokens, tuple(TOKEN_IDS[token] for token in tokens), (0, 0), 0, 0)
    if len(tokens) not in (2, 5) or tokens[0] not in ("AT_LEAST", "AVOID", "NOOP") or not tokens[1].startswith("VALUE_"):
        raise ValueError(f"invalid instruction grammar: {instruction!r}")
    clauses = [tokens[:2]]
    if len(tokens) == 5:
        if tokens[2] != "AND" or tokens[3] not in ("AT_LEAST", "AVOID", "NOOP") or not tokens[4].startswith("VALUE_"):
            raise ValueError(f"invalid instruction grammar: {instruction!r}")
        clauses.append(tokens[3:])
    lower = 0
    forbidden = 0
    constraints = [0, 0]
    for operator, value_token in clauses:
        value = _value(value_token)
        if operator == "AT_LEAST":
            constraints[0] = 1; lower = value
        elif operator == "AVOID":
            constraints[1] = 1; forbidden = value
    return ParsedInstructionR1(tokens, tuple(TOKEN_IDS[token] for token in tokens), tuple(constraints), lower, forbidden)
