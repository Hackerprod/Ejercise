"""T2-I0-R1.1 grammar helpers with deterministic NOOP dummy variation."""

from __future__ import annotations

from t2_i0_instruction_r1 import ParsedInstructionR1, parse_instruction_r1, tokenize_instruction_r1, TOKEN_IDS, VOCABULARY


DUMMY_OFFSETS = (0, 7, 13, 23)


def dummy_values(value: int) -> tuple[int, ...]:
    return tuple((value + offset) % 32 for offset in DUMMY_OFFSETS)


def instruction_for_r1_1(constraints: tuple[int, int], value: int, *, variant: int = 0, dummy: int = 0, dummy2: int | None = None) -> str:
    noop = f"NOOP VALUE_{dummy}"
    if constraints == (0, 0):
        return noop if variant == 0 else f"{noop} AND NOOP VALUE_{dummy if dummy2 is None else dummy2}"
    operator = f"AT_LEAST VALUE_{value}" if constraints == (1, 0) else f"AVOID VALUE_{value}"
    if variant == 0:
        return operator
    return f"{operator} AND {noop}" if variant == 1 else f"{noop} AND {operator}"


def r1_1_corpus() -> list[tuple[str, tuple[int, int]]]:
    rows = [(instruction_for_r1_1((0, 0), 0, dummy=dummy), (0, 0)) for dummy in range(32)]
    rows.extend((instruction_for_r1_1((0, 0), 0, variant=1, dummy=dummy, dummy2=(dummy + 11) % 32), (0, 0)) for dummy in range(32))
    for constraints in ((1, 0), (0, 1)):
        for value in range(32):
            for dummy in dummy_values(value):
                rows.extend((instruction_for_r1_1(constraints, value, variant=variant, dummy=dummy), constraints) for variant in (0, 1, 2))
    return rows
