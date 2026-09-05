"""T1 recurrent-core architecture."""

from .adapters import InputAdapter, OutputReader
from .data import (
    DEFAULT_COUNTS,
    OUTPUT_CARDINALITIES,
    TASK_NAMES,
    TokenVocabulary,
    generate_all_datasets,
)
from .model import RecurrentCore
from .unified import (
    OPCODE_IDS,
    OPCODES,
    CandidateState,
    ReadResult,
    SharedMemoryReader,
    SharedRecurrentCore,
    TypedCommit,
    UnifiedT1U0,
)

__all__ = [
    "DEFAULT_COUNTS",
    "InputAdapter",
    "OUTPUT_CARDINALITIES",
    "OutputReader",
    "RecurrentCore",
    "CandidateState",
    "OPCODE_IDS",
    "OPCODES",
    "ReadResult",
    "SharedMemoryReader",
    "SharedRecurrentCore",
    "TASK_NAMES",
    "TokenVocabulary",
    "TypedCommit",
    "UnifiedT1U0",
    "generate_all_datasets",
]
