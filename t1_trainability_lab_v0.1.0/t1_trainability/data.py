"""Deterministic synthetic task formats and dataset generators for T1-B."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch


TaskName = Literal[
    "associative_recall",
    "multi_hop",
    "variable_binding",
    "sequential_update",
    "length_generalization",
    "pointer_chasing",
]
SplitName = Literal["train", "val", "test"]

TASK_NAMES: tuple[TaskName, ...] = (
    "associative_recall",
    "multi_hop",
    "variable_binding",
    "sequential_update",
    "length_generalization",
)
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
OUTPUT_CARDINALITIES: dict[TaskName, int] = {
    "associative_recall": 32,
    "multi_hop": 32,
    "variable_binding": 8,
    "sequential_update": 32,
    "length_generalization": 32,
    "pointer_chasing": 256,
}
DEFAULT_COUNTS: dict[SplitName, int] = {"train": 10_000, "val": 2_000, "test": 2_000}


def symbol_token(index: int) -> str:
    return f"SYM:{index:02d}"


def value_token(index: int) -> str:
    return f"VALUE:{index:02d}"


def color_token(index: int) -> str:
    return f"COLOR:{index:02d}"


def object_token(index: int) -> str:
    return f"OBJECT:{index:02d}"


def hops_token(count: int) -> str:
    return f"HOPS:{count}"


def pointer_key_token(index: int) -> str:
    return f"KEY:{index:03d}"


class TokenVocabulary:
    """Stable vocabulary for typed, non-semantic synthetic tokens."""

    def __init__(self) -> None:
        tokens = [
            "PAD",
            "TASK:ASSOCIATIVE_RECALL",
            "TASK:MULTI_HOP",
            "TASK:VARIABLE_BINDING",
            "TASK:SEQUENTIAL_UPDATE",
            "TASK:LENGTH_GENERALIZATION",
            "TASK:POINTER_CHASING",
            "FACT",
            "PAIR",
            "QUERY",
            "REL",
            "BIND",
            "ASSIGN",
            "ATTR",
            "ATTRIBUTE:COLOR",
            "INITIAL",
            "STEP",
            "RESULT",
            "MEM",
            "SRC",
            "DST",
            "SEP",
            "START",
            "VAR:X",
            "OP:ADD",
            "OP:SUB",
            "OP:MUL",
        ]
        tokens.extend(symbol_token(index) for index in range(32))
        tokens.extend(value_token(index) for index in range(32))
        tokens.extend(color_token(index) for index in range(8))
        tokens.extend(object_token(index) for index in range(32))
        tokens.extend(hops_token(count) for count in range(1, 7))
        tokens.extend(pointer_key_token(index) for index in range(256))
        self.tokens = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as error:
            raise ValueError(f"Unknown typed token: {error.args[0]}") from error

    def id_for(self, token: str) -> int:
        return self.token_to_id[token]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"tokens": self.tokens}, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TaskExample:
    """One serialized synthetic example before token IDs are padded."""

    task: TaskName
    split: SplitName
    tokens: tuple[str, ...]
    query_token: str
    target: int
    output_cardinality: int
    metadata: dict[str, int | str]

    def to_record(self) -> dict[str, object]:
        return {
            "task": self.task,
            "split": self.split,
            "tokens": list(self.tokens),
            "query_token": self.query_token,
            "target": self.target,
            "output_cardinality": self.output_cardinality,
            "metadata": self.metadata,
        }


def _sample_distinct(rng: random.Random, count: int, upper_bound: int) -> list[int]:
    return rng.sample(range(upper_bound), count)


def _task_token(task: TaskName) -> str:
    return f"TASK:{task.upper()}"


def _associative_recall(rng: random.Random, split: SplitName) -> TaskExample:
    pair_count = rng.randint(3, 6)
    keys = _sample_distinct(rng, pair_count, 32)
    values = [rng.randrange(32) for _ in range(pair_count)]
    query_index = rng.randrange(pair_count)
    tokens: list[str] = [_task_token("associative_recall")]
    for key, value in zip(keys, values):
        tokens.extend(("FACT", "PAIR", symbol_token(key), value_token(value)))
    tokens.extend(("QUERY", symbol_token(keys[query_index])))
    return TaskExample(
        task="associative_recall",
        split=split,
        tokens=tuple(tokens),
        query_token=symbol_token(keys[query_index]),
        target=values[query_index],
        output_cardinality=32,
        metadata={"pair_count": pair_count},
    )


def _multi_hop(rng: random.Random, split: SplitName, *, generalization: bool) -> TaskExample:
    if generalization:
        hop_count = rng.randint(1, 3) if split != "test" else rng.randint(4, 6)
        task: TaskName = "length_generalization"
    else:
        # Multi-hop directly measures R=1 versus R=4; length OOD is separate.
        hop_count = rng.randint(1, 4)
        task = "multi_hop"
    chain = _sample_distinct(rng, hop_count + 1, 32)
    relations = list(zip(chain, chain[1:]))
    rng.shuffle(relations)
    tokens: list[str] = [_task_token(task)]
    for source, destination in relations:
        tokens.extend(("REL", symbol_token(source), symbol_token(destination)))
    tokens.extend(("QUERY", symbol_token(chain[0]), hops_token(hop_count)))
    return TaskExample(
        task=task,
        split=split,
        tokens=tuple(tokens),
        query_token=symbol_token(chain[0]),
        target=chain[-1],
        output_cardinality=32,
        metadata={"hop_count": hop_count},
    )


def _pointer_chasing(rng: random.Random, split: SplitName) -> TaskExample:
    """Generate one-hop-per-round pointer chasing with opaque per-example keys."""

    hop_count = rng.randint(1, 4)
    keys = _sample_distinct(rng, 9, 256)
    path = keys[:5]
    path_edges = list(zip(path, path[1:]))
    distractor_edges = [
        (source, rng.choice([key for key in keys if key != source]))
        for source in keys[5:]
    ]
    edges = path_edges + distractor_edges
    rng.shuffle(edges)
    tokens: list[str] = [_task_token("pointer_chasing")]
    for source, destination in edges:
        tokens.extend(("MEM", "SRC", pointer_key_token(source), "DST", pointer_key_token(destination), "SEP"))
    tokens.extend(("START", pointer_key_token(path[0]), "QUERY"))
    metadata = {
        "hop_count": hop_count,
        "memory_size": len(edges),
        "start_key": path[0],
        "target_key": path[hop_count],
        "memory_sources": ",".join(str(source) for source, _ in edges),
        "memory_destinations": ",".join(str(destination) for _, destination in edges),
    }
    return TaskExample(
        task="pointer_chasing",
        split=split,
        tokens=tuple(tokens),
        query_token=pointer_key_token(path[0]),
        target=path[hop_count],
        output_cardinality=256,
        metadata=metadata,
    )


def generate_pointer_examples(split: SplitName, count: int, seed: int) -> list[TaskExample]:
    """Generate deterministic pointer-chasing examples without changing T1-B datasets."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(seed + 500_003 + SPLIT_NAMES.index(split) * 10_007)
    return [_pointer_chasing(rng, split) for _ in range(count)]


def _variable_binding(rng: random.Random, split: SplitName) -> TaskExample:
    distractor_count = rng.randint(1, 4)
    target_object, *distractors = _sample_distinct(rng, distractor_count + 1, 32)
    target_color = rng.randrange(8)
    distractor_colors = [rng.randrange(8) for _ in distractors]
    attributes = [(target_object, target_color), *zip(distractors, distractor_colors)]
    rng.shuffle(attributes)
    tokens: list[str] = [
        _task_token("variable_binding"),
        "BIND",
        "ASSIGN",
        "VAR:X",
        object_token(target_object),
    ]
    for object_index, color in attributes:
        tokens.extend(("ATTR", object_token(object_index), "ATTRIBUTE:COLOR", color_token(color)))
    tokens.extend(("QUERY", "VAR:X", "ATTRIBUTE:COLOR"))
    return TaskExample(
        task="variable_binding",
        split=split,
        tokens=tuple(tokens),
        query_token="VAR:X",
        target=target_color,
        output_cardinality=8,
        metadata={"distractor_count": distractor_count},
    )


def _apply_operation(value: int, operation: str, operand: int) -> int:
    if operation == "OP:ADD":
        return value + operand
    if operation == "OP:SUB":
        return value - operand
    if operation == "OP:MUL":
        return value * operand
    raise ValueError(f"Unknown operation: {operation}")


def _sequential_update(rng: random.Random, split: SplitName) -> TaskExample:
    step_count = rng.randint(3, 6)
    target = rng.randrange(32)

    # Construct trajectory backwards from a uniform target. This keeps all
    # intermediate values in [0, 31] while preventing multiplication from
    # collapsing the target distribution toward zero.
    for _attempt in range(1000):
        multiply_position = rng.randrange(step_count)
        reverse_values = [target]
        reverse_operations: list[str] = []
        reverse_operands: list[int] = []
        possible = True
        for reverse_step in range(step_count):
            forward_step = step_count - reverse_step - 1
            operation = "OP:MUL" if forward_step == multiply_position else rng.choice(("OP:ADD", "OP:SUB"))
            current = reverse_values[-1]
            if operation == "OP:ADD":
                operands = list(range(1, current + 1)) or [0]
                operand = rng.choice(operands)
                previous = current - operand
            elif operation == "OP:SUB":
                operands = list(range(1, 32 - current)) or [0]
                operand = rng.choice(operands)
                previous = current + operand
            else:
                factors = [factor for factor in range(2, 32) if current % factor == 0]
                if not factors:
                    possible = False
                    break
                operand = rng.choice(factors)
                previous = current // operand
            reverse_values.append(previous)
            reverse_operations.append(operation)
            reverse_operands.append(operand)
        if possible:
            operations = list(reversed(reverse_operations))
            operands = list(reversed(reverse_operands))
            initial = reverse_values[-1]
            current = initial
            for operation, operand in zip(operations, operands):
                current = _apply_operation(current, operation, operand)
            if current == target:
                break
    else:
        raise RuntimeError("Could not generate bounded sequential update")

    tokens: list[str] = [_task_token("sequential_update"), "INITIAL", value_token(initial)]
    for operation, operand in zip(operations, operands):
        tokens.extend(("STEP", operation, value_token(operand)))
    tokens.append("RESULT")
    return TaskExample(
        task="sequential_update",
        split=split,
        tokens=tuple(tokens),
        query_token="RESULT",
        target=target,
        output_cardinality=32,
        metadata={"step_count": step_count, "result_in_range": int(0 <= target < 32)},
    )


def generate_examples(task: TaskName, split: SplitName, count: int, seed: int) -> list[TaskExample]:
    """Generate deterministic examples for one task/split."""

    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task: {task}")
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    if count < 0:
        raise ValueError("count must be non-negative")

    task_index = TASK_NAMES.index(task)
    split_index = SPLIT_NAMES.index(split)
    rng = random.Random(seed + task_index * 100_003 + split_index * 10_007)
    examples: list[TaskExample] = []
    for _ in range(count):
        if task == "associative_recall":
            example = _associative_recall(rng, split)
        elif task == "multi_hop":
            example = _multi_hop(rng, split, generalization=False)
        elif task == "variable_binding":
            example = _variable_binding(rng, split)
        elif task == "sequential_update":
            example = _sequential_update(rng, split)
        else:
            example = _multi_hop(rng, split, generalization=True)
        examples.append(example)
    return examples


def generate_all_datasets(
    seed: int = 20260904,
    counts: dict[SplitName, int] | None = None,
) -> dict[TaskName, dict[SplitName, list[TaskExample]]]:
    selected_counts = counts or DEFAULT_COUNTS
    return {
        task: {
            split: generate_examples(task, split, selected_counts[split], seed)
            for split in SPLIT_NAMES
        }
        for task in TASK_NAMES
    }


def write_jsonl(path: Path, examples: Iterable[TaskExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_record(), separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[TaskExample]:
    examples: list[TaskExample] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            examples.append(
                TaskExample(
                    task=record["task"],
                    split=record["split"],
                    tokens=tuple(record["tokens"]),
                    query_token=record["query_token"],
                    target=int(record["target"]),
                    output_cardinality=int(record["output_cardinality"]),
                    metadata=record["metadata"],
                )
            )
    return examples


def dataset_summary(examples: Iterable[TaskExample]) -> dict[str, object]:
    rows = list(examples)
    if not rows:
        raise ValueError("Cannot summarize an empty dataset")
    task = rows[0].task
    cardinality = rows[0].output_cardinality
    targets = Counter(row.target for row in rows)
    if any(row.task != task for row in rows):
        raise ValueError("Mixed tasks in one summary")
    if any(not 0 <= row.target < row.output_cardinality for row in rows):
        raise ValueError(f"Target outside output vocabulary for {task}")
    return {
        "task": task,
        "count": len(rows),
        "output_cardinality": cardinality,
        "random_baseline_accuracy": 1.0 / cardinality,
        "target_counts": {str(index): targets.get(index, 0) for index in range(cardinality)},
    }


def encode_batch(
    examples: list[TaskExample],
    vocabulary: TokenVocabulary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad examples into input IDs, mask, query IDs, and exact targets."""

    if not examples:
        raise ValueError("Cannot encode an empty batch")
    encoded = [vocabulary.encode(example.tokens) for example in examples]
    max_length = max(len(row) for row in encoded)
    input_ids = torch.full((len(encoded), max_length), vocabulary.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_length), dtype=torch.bool)
    query_ids = torch.empty(len(encoded), dtype=torch.long)
    targets = torch.empty(len(encoded), dtype=torch.long)
    for index, (example, token_ids) in enumerate(zip(examples, encoded)):
        length = len(token_ids)
        input_ids[index, :length] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[index, :length] = True
        query_ids[index] = vocabulary.id_for(example.query_token)
        targets[index] = example.target
    return input_ids, attention_mask, query_ids, targets
