"""T2-I1 lexical abstraction grammar with two aliases per operator."""

from __future__ import annotations

from dataclasses import dataclass
import re


OPERATORS = ("KEEP", "AT_LEAST", "AVOID", "AND", "NOOP", "MINIMUM", "EXCLUDE")
VALUE_TOKENS = tuple(f"VALUE_{value}" for value in range(32))
VOCABULARY = OPERATORS + VALUE_TOKENS
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
VALUE_PATTERN = re.compile(r"VALUE_([0-9]|[12][0-9]|3[01])\Z")


@dataclass(frozen=True)
class ParsedInstructionI1:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    constraints: tuple[int, int]
    lower: int
    forbidden: int


def tokenize_instruction_i1(instruction: str) -> tuple[str, ...]:
    if not isinstance(instruction, str): raise TypeError("instruction must be a string")
    tokens = tuple(instruction.strip().split())
    if not tokens or any(token not in TOKEN_IDS for token in tokens): raise ValueError(f"unknown instruction token(s): {instruction!r}")
    return tokens


def parse_instruction_i1(instruction: str) -> ParsedInstructionI1:
    tokens = tokenize_instruction_i1(instruction)
    if tokens == ("KEEP",): return ParsedInstructionI1(tokens, tuple(TOKEN_IDS[token] for token in tokens), (0, 0), 0, 0)
    if len(tokens) not in (2, 5) or tokens[0] not in ("AT_LEAST", "MINIMUM", "AVOID", "EXCLUDE", "NOOP") or not VALUE_PATTERN.fullmatch(tokens[1]): raise ValueError(f"invalid instruction grammar: {instruction!r}")
    clauses = [tokens[:2]]
    if len(tokens) == 5:
        if tokens[2] != "AND" or tokens[3] not in ("AT_LEAST", "MINIMUM", "AVOID", "EXCLUDE", "NOOP") or not VALUE_PATTERN.fullmatch(tokens[4]): raise ValueError(f"invalid instruction grammar: {instruction!r}")
        clauses.append(tokens[3:])
    constraints = [0, 0]; lower = forbidden = 0
    for operator, value_token in clauses:
        value = int(VALUE_PATTERN.fullmatch(value_token).group(1))
        if operator in ("AT_LEAST", "MINIMUM"): constraints[0] = 1; lower = value
        elif operator in ("AVOID", "EXCLUDE"): constraints[1] = 1; forbidden = value
    return ParsedInstructionI1(tokens, tuple(TOKEN_IDS[token] for token in tokens), tuple(constraints), lower, forbidden)
