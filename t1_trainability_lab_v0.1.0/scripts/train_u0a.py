"""Train T1-U0-A on one shared checkpoint with oracle round routing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import load_jsonl  # noqa: E402
from t1_trainability.unified import (  # noqa: E402
    OPCODE_IDS,
    ROW_ASSIGN,
    ROW_ATTR,
    ROW_PAIR,
    ROW_REL,
    ROW_VEC,
    SLOT_COUNT,
    SLOT_E,
    SLOT_P,
    SLOT_R,
    SLOT_W,
    UnifiedT1U0,
)


TASKS = (
    "pointer_chasing",
    "multi_hop",
    "associative_recall",
    "variable_binding",
    "sequential_update",
    "workspace_accumulation",
)
SEED = 101
DIMENSION = 64
BATCH_SIZE = 128
TOTAL_STEPS = 30_000
VALIDATION_INTERVAL = 3_000
IMM_ZERO = 511
KEY_BASE = 0
SYM_BASE = 256
VALUE_BASE = 288
OBJECT_BASE = 320
COLOR_BASE = 352
INDEX_BASE = 384
ATTRIBUTE_COLOR_ID = 400
POINTER_CLASS_IDS = tuple(range(KEY_BASE, KEY_BASE + 256))
MULTI_HOP_CLASS_IDS = tuple(range(SYM_BASE, SYM_BASE + 32))
VALUE_CLASS_IDS = tuple(range(VALUE_BASE, VALUE_BASE + 32))
COLOR_CLASS_IDS = tuple(range(COLOR_BASE, COLOR_BASE + 8))


@dataclass
class CanonicalExample:
    task: str
    initial_ids: tuple[int, int, int, int]
    memory_keys: tuple[int, ...]
    memory_values: tuple[int, ...]
    memory_types: tuple[int, ...]
    memory_attribute_ids: tuple[int, ...]
    memory_raw_values: tuple[Tensor | None, ...]
    opcodes: tuple[int, ...]
    immediates: tuple[int, ...]
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]
    presence: tuple[bool, bool, bool, bool]
    target_id: int | None
    target_vector: Tensor | None
    hop_count: int
    read_modes: tuple[int, ...] = ()


class ExampleDataset(Dataset[CanonicalExample]):
    def __init__(self, examples: list[CanonicalExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CanonicalExample:
        return self.examples[index]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_training_checkpoint(
    path: Path,
    *,
    config: dict[str, object],
    step: int,
    model: UnifiedT1U0,
    optimizer: torch.optim.Optimizer,
    best_score: float,
    best_step: int,
) -> None:
    """Persist enough state to continue training after an external timeout."""

    torch.save(
        {
            "config": config,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "best_step": best_step,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def build_optimizer(model: UnifiedT1U0) -> torch.optim.Optimizer:
    """Exclude typed/small parameters from AdamW decay; freeze U0-A W correction."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim == 1
            or lowered.endswith("bias")
            or "embedding" in lowered
            or "decoder" in lowered
            or "alu_" in lowered
            or "operation_heads" in lowered
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=3e-4,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_int(token: str) -> int:
    return int(token.split(":", 1)[1])


def parse_pointer(split: str) -> list[CanonicalExample]:
    rows = load_jsonl(ROOT / "datasets" / "pointer_chasing" / f"{split}.jsonl")
    result = []
    for row in rows:
        sources = [int(value) for value in str(row.metadata["memory_sources"]).split(",")]
        destinations = [int(value) for value in str(row.metadata["memory_destinations"]).split(",")]
        hop = int(row.metadata["hop_count"])
        result.append(
            CanonicalExample(
                "pointer_chasing",
                (KEY_BASE + int(row.metadata["start_key"]), -1, -1, -1),
                tuple(KEY_BASE + value for value in sources),
                tuple(KEY_BASE + value for value in destinations),
                tuple(ROW_REL for _ in sources),
                tuple(0 for _ in sources),
                tuple(None for _ in sources),
                tuple([OPCODE_IDS["READ_P"]] * hop + [OPCODE_IDS["EMIT"]] * (4 - hop)),
                tuple([IMM_ZERO] * 4),
                tuple([SLOT_P] * 4),
                tuple([SLOT_P] * 4),
                (True, False, False, False),
                KEY_BASE + row.target,
                None,
                hop,
            )
        )
    return result


def parse_multi_hop(split: str) -> list[CanonicalExample]:
    rows = load_jsonl(ROOT / "datasets" / "multi_hop" / f"{split}.jsonl")
    result = []
    for row in rows:
        sources: list[int] = []
        destinations: list[int] = []
        for index, token in enumerate(row.tokens):
            if token == "REL":
                sources.append(parse_int(row.tokens[index + 1]))
                destinations.append(parse_int(row.tokens[index + 2]))
        hop = int(row.metadata["hop_count"])
        result.append(
            CanonicalExample(
                "multi_hop",
                (SYM_BASE + parse_int(row.query_token), -1, -1, -1),
                tuple(SYM_BASE + value for value in sources),
                tuple(SYM_BASE + value for value in destinations),
                tuple(ROW_REL for _ in sources),
                tuple(0 for _ in sources),
                tuple(None for _ in sources),
                tuple([OPCODE_IDS["READ_P"]] * hop + [OPCODE_IDS["EMIT"]] * (4 - hop)),
                tuple([IMM_ZERO] * 4),
                tuple([SLOT_P] * 4),
                tuple([SLOT_P] * 4),
                (True, False, False, False),
                SYM_BASE + row.target,
                None,
                hop,
            )
        )
    return result


def parse_associative(split: str) -> list[CanonicalExample]:
    rows = load_jsonl(ROOT / "datasets" / "associative_recall" / f"{split}.jsonl")
    result = []
    for row in rows:
        keys: list[int] = []
        values: list[int] = []
        for index, token in enumerate(row.tokens):
            if token == "PAIR":
                keys.append(parse_int(row.tokens[index + 1]))
                values.append(parse_int(row.tokens[index + 2]))
        result.append(
            CanonicalExample(
                "associative_recall",
                (SYM_BASE + parse_int(row.query_token), -1, -1, -1),
                tuple(SYM_BASE + value for value in keys),
                tuple(VALUE_BASE + value for value in values),
                tuple(ROW_PAIR for _ in keys),
                tuple(0 for _ in keys),
                tuple(None for _ in keys),
                (OPCODE_IDS["READ_E"], OPCODE_IDS["EMIT"], OPCODE_IDS["EMIT"], OPCODE_IDS["EMIT"]),
                (IMM_ZERO,) * 4,
                (SLOT_P,) * 4,
                (SLOT_E,) * 4,
                (True, False, True, False),
                VALUE_BASE + row.target,
                None,
                1,
            )
        )
    return result


def parse_variable(split: str) -> list[CanonicalExample]:
    rows = load_jsonl(ROOT / "datasets" / "variable_binding" / f"{split}.jsonl")
    result = []
    for row in rows:
        keys: list[int] = []
        values: list[int] = []
        types: list[int] = []
        attributes: list[int] = []
        for index, token in enumerate(row.tokens):
            if token == "ASSIGN":
                keys.append(OBJECT_BASE + 32)
                values.append(OBJECT_BASE + parse_int(row.tokens[index + 2]))
                types.append(ROW_ASSIGN)
                attributes.append(0)
            elif token == "ATTR":
                reference = parse_int(row.tokens[index + 1])
                keys.append(OBJECT_BASE + reference)
                values.append(COLOR_BASE + parse_int(row.tokens[index + 3]))
                types.append(ROW_ATTR)
                attributes.append(ATTRIBUTE_COLOR_ID)
        result.append(
            CanonicalExample(
                "variable_binding",
                (OBJECT_BASE + 32, -1, -1, -1),
                tuple(keys),
                tuple(values),
                tuple(types),
                tuple(attributes),
                tuple(None for _ in keys),
                (OPCODE_IDS["READ_P"], OPCODE_IDS["READ_E"], OPCODE_IDS["EMIT"], OPCODE_IDS["EMIT"]),
                (IMM_ZERO, ATTRIBUTE_COLOR_ID, IMM_ZERO, IMM_ZERO),
                (SLOT_P, SLOT_P, SLOT_E, SLOT_E),
                (SLOT_P, SLOT_E, SLOT_E, SLOT_E),
                (True, False, True, False),
                COLOR_BASE + row.target,
                None,
                2,
            )
        )
    return result


def parse_sequential(split: str) -> list[CanonicalExample]:
    rows = load_jsonl(ROOT / "datasets" / "sequential_update" / f"{split}.jsonl")
    result = []
    operation_ids = {"OP:ADD": "ALU_ADD", "OP:SUB": "ALU_SUB", "OP:MUL": "ALU_MUL"}
    for row in rows:
        initial = 0
        opcodes: list[int] = []
        immediate: list[int] = []
        for index, token in enumerate(row.tokens):
            if token == "INITIAL":
                initial = parse_int(row.tokens[index + 1])
            elif token == "STEP":
                opcodes.append(OPCODE_IDS[operation_ids[row.tokens[index + 1]]])
                immediate.append(VALUE_BASE + parse_int(row.tokens[index + 2]))
        hop = len(opcodes)
        opcodes.extend([OPCODE_IDS["EMIT"]] * (6 - hop))
        immediate.extend([IMM_ZERO] * (6 - hop))
        result.append(
            CanonicalExample(
                "sequential_update",
                (-1, VALUE_BASE + initial, -1, -1),
                tuple(),
                tuple(),
                tuple(),
                tuple(),
                tuple(),
                tuple(opcodes),
                tuple(immediate),
                (SLOT_R,) * 6,
                (SLOT_R,) * 6,
                (False, True, False, False),
                VALUE_BASE + row.target,
                None,
                hop,
            )
        )
    return result


def build_sequential_h1_table() -> list[CanonicalExample]:
    """Build complete elementary ALU table for ISO replay batches."""

    operations = ("ALU_ADD", "ALU_SUB", "ALU_MUL")
    result: list[CanonicalExample] = []
    for operation in operations:
        for initial in range(32):
            for operand in range(32):
                if operation == "ALU_ADD":
                    target = (initial + operand) % 32
                elif operation == "ALU_SUB":
                    target = (initial - operand) % 32
                else:
                    target = (initial * operand) % 32
                result.append(
                    CanonicalExample(
                        "sequential_update",
                        (-1, VALUE_BASE + initial, -1, -1),
                        tuple(), tuple(), tuple(), tuple(), tuple(),
                        (OPCODE_IDS[operation],) + (OPCODE_IDS["EMIT"],) * 5,
                        (VALUE_BASE + operand,) + (IMM_ZERO,) * 5,
                        (SLOT_R,) * 6, (SLOT_R,) * 6,
                        (False, True, False, False), VALUE_BASE + target, None, 1,
                    )
                )
    return result


def parse_workspace(split: str) -> list[CanonicalExample]:
    payload = torch.load(ROOT / "datasets" / "t1w_workspace" / f"{split}.pt", map_location="cpu", weights_only=False)
    vectors = payload["vectors"]
    lengths = payload["lengths"]
    result = []
    for index in range(len(lengths)):
        hop = int(lengths[index])
        rows = []
        for round_index in range(hop):
            rows.append((INDEX_BASE + round_index, -1, ROW_VEC, 0, vectors[index, round_index].clone()))
        opcodes = [OPCODE_IDS["ACCUM_W"]] * hop + [OPCODE_IDS["EMIT"]] * (6 - hop)
        result.append(
            CanonicalExample(
                "workspace_accumulation",
                (-1, -1, -1, -1),
                tuple(row[0] for row in rows),
                tuple(-1 for _ in rows),
                tuple(row[2] for row in rows),
                tuple(row[3] for row in rows),
                tuple(row[4] for row in rows),
                tuple(opcodes),
                tuple([INDEX_BASE + round_index for round_index in range(hop)] + [IMM_ZERO] * (6 - hop)),
                (SLOT_W,) * 6,
                (SLOT_W,) * 6,
                (False, False, False, True),
                None,
                payload["targets"][index].clone(),
                hop,
            )
        )
    return result


PARSERS = {
    "pointer_chasing": parse_pointer,
    "multi_hop": parse_multi_hop,
    "associative_recall": parse_associative,
    "variable_binding": parse_variable,
    "sequential_update": parse_sequential,
    "workspace_accumulation": parse_workspace,
}


def build_canonical_data(output_dir: Path) -> dict[str, dict[str, list[CanonicalExample]]]:
    data_dir = output_dir / "canonical_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_manifest: dict[str, str] = {}
    for task in TASKS:
        source_task = "t1w_workspace" if task == "workspace_accumulation" else task
        for split in ("train", "val", "test"):
            source = ROOT / "datasets" / source_task / f"{split}.{'pt' if task == 'workspace_accumulation' else 'jsonl'}"
            source_manifest[str(source.relative_to(ROOT))] = file_sha256(source)
    save_json(output_dir / "source_manifest.json", source_manifest)
    datasets: dict[str, dict[str, list[CanonicalExample]]] = {}
    for task in TASKS:
        datasets[task] = {}
        for split in ("train", "val", "test"):
            examples = PARSERS[task](split)
            torch.save(examples, data_dir / f"{task}_{split}.pt")
            datasets[task][split] = examples
    save_json(output_dir / "canonical_manifest.json", {task: {split: len(rows) for split, rows in values.items()} for task, values in datasets.items()})
    return datasets


def collate(rows: list[CanonicalExample]) -> dict[str, Any]:
    batch = len(rows)
    max_memory = max((len(row.memory_keys) for row in rows), default=1)
    max_rounds = max(len(row.opcodes) for row in rows)
    key_ids = torch.zeros(batch, max_memory, dtype=torch.long)
    value_ids = torch.full((batch, max_memory), -1, dtype=torch.long)
    memory_types = torch.zeros(batch, max_memory, dtype=torch.long)
    attribute_ids = torch.full((batch, max_memory), IMM_ZERO, dtype=torch.long)
    row_mask = torch.zeros(batch, max_memory, dtype=torch.bool)
    raw_mask = torch.zeros(batch, max_memory, dtype=torch.bool)
    raw_values = torch.zeros(batch, max_memory, DIMENSION)
    initial_ids = torch.full((batch, SLOT_COUNT), -1, dtype=torch.long)
    opcodes = torch.full((batch, max_rounds), OPCODE_IDS["EMIT"], dtype=torch.long)
    immediates = torch.full((batch, max_rounds), IMM_ZERO, dtype=torch.long)
    source_slots = torch.zeros((batch, max_rounds), dtype=torch.long)
    destination_slots = torch.zeros((batch, max_rounds), dtype=torch.long)
    read_modes = torch.zeros((batch, max_rounds), dtype=torch.long)
    presence = torch.zeros((batch, SLOT_COUNT), dtype=torch.bool)
    hops = torch.zeros(batch, dtype=torch.long)
    target_ids = torch.full((batch,), -1, dtype=torch.long)
    target_vectors = torch.zeros(batch, DIMENSION)
    target_is_vector = torch.zeros(batch, dtype=torch.bool)
    intermediate_target_ids = torch.full((batch, max_rounds), -1, dtype=torch.long)
    for index, row in enumerate(rows):
        initial_ids[index] = torch.tensor(row.initial_ids)
        length = len(row.memory_keys)
        if length:
            key_ids[index, :length] = torch.tensor(row.memory_keys)
            value_ids[index, :length] = torch.tensor(row.memory_values)
            memory_types[index, :length] = torch.tensor(row.memory_types)
            attribute_ids[index, :length] = torch.tensor(row.memory_attribute_ids)
            row_mask[index, :length] = True
            for memory_index, raw_value in enumerate(row.memory_raw_values):
                if raw_value is not None:
                    raw_mask[index, memory_index] = True
                    raw_values[index, memory_index] = raw_value
        rounds = len(row.opcodes)
        opcodes[index, :rounds] = torch.tensor(row.opcodes)
        immediates[index, :rounds] = torch.tensor(row.immediates)
        source_slots[index, :rounds] = torch.tensor(row.source_slots)
        destination_slots[index, :rounds] = torch.tensor(row.destination_slots)
        if row.read_modes:
            read_modes[index, :rounds] = torch.tensor(row.read_modes)
        presence[index] = torch.tensor(row.presence)
        hops[index] = row.hop_count
        if row.target_id is not None:
            target_ids[index] = row.target_id
        if row.target_vector is not None:
            target_vectors[index] = row.target_vector
            target_is_vector[index] = True
        if row.task == "sequential_update":
            current = int(row.initial_ids[SLOT_R])
            for round_index, opcode in enumerate(row.opcodes):
                if opcode == OPCODE_IDS["ALU_ADD"]:
                    current = VALUE_BASE + ((current - VALUE_BASE) + (row.immediates[round_index] - VALUE_BASE)) % 32
                elif opcode == OPCODE_IDS["ALU_SUB"]:
                    current = VALUE_BASE + ((current - VALUE_BASE) - (row.immediates[round_index] - VALUE_BASE)) % 32
                elif opcode == OPCODE_IDS["ALU_MUL"]:
                    current = VALUE_BASE + ((current - VALUE_BASE) * (row.immediates[round_index] - VALUE_BASE)) % 32
                if opcode in {OPCODE_IDS["ALU_ADD"], OPCODE_IDS["ALU_SUB"], OPCODE_IDS["ALU_MUL"]}:
                    intermediate_target_ids[index, round_index] = current
    return {
        "key_ids": key_ids,
        "value_ids": value_ids,
        "memory_types": memory_types,
        "attribute_ids": attribute_ids,
        "row_mask": row_mask,
        "raw_mask": raw_mask,
        "raw_values": raw_values,
        "initial_ids": initial_ids,
        "opcodes": opcodes,
        "immediates": immediates,
        "source_slots": source_slots,
        "destination_slots": destination_slots,
        "read_modes": read_modes,
        "presence": presence,
        "hops": hops,
        "target_ids": target_ids,
        "target_vectors": target_vectors,
        "target_is_vector": target_is_vector,
        "intermediate_target_ids": intermediate_target_ids,
    }


def materialize(model: UnifiedT1U0, batch: dict[str, Any]) -> dict[str, Any]:
    token_embedding = model.token_embedding
    key_vectors = token_embedding(batch["key_ids"])
    attribute_vectors = token_embedding(batch["attribute_ids"])
    key_vectors = key_vectors + attribute_vectors * (batch["memory_types"] == ROW_ATTR).unsqueeze(-1)
    discrete_values = token_embedding(batch["value_ids"].clamp_min(0))
    values = torch.where(batch["raw_mask"].unsqueeze(-1), batch["raw_values"], discrete_values)
    state = torch.zeros(batch["initial_ids"].shape[0], SLOT_COUNT, model.dimension)
    for slot in range(SLOT_COUNT):
        active = batch["initial_ids"][:, slot] >= 0
        if active.any():
            state[active, slot] = token_embedding(batch["initial_ids"][active, slot])
    return {**batch, "state": state, "memory_keys": key_vectors, "memory_values": values}


def immediate_vectors(model: UnifiedT1U0, immediate_ids: Tensor) -> Tensor:
    vectors = model.token_embedding(immediate_ids)
    return torch.where((immediate_ids == IMM_ZERO).unsqueeze(-1), torch.zeros_like(vectors), vectors)


def run_rounds(model: UnifiedT1U0, batch: dict[str, Any], rounds: int) -> Tensor:
    data = materialize(model, batch)
    state = data["state"]
    for round_index in range(rounds):
        round_immediate = immediate_vectors(model, data["immediates"][:, round_index])
        opcode = data["opcodes"][:, round_index]
        state, _, _ = model.step(
            state,
            data["memory_keys"],
            data["memory_values"],
            data["memory_types"],
            data["row_mask"],
            opcode,
            round_immediate,
            data["source_slots"][:, round_index],
            data["destination_slots"][:, round_index],
            data["presence"],
            read_mode=data["read_modes"][:, round_index],
            read_set="legacy",
        )
    return state


def run_rounds_with_trace(model: UnifiedT1U0, batch: dict[str, Any], rounds: int) -> tuple[Tensor, list[Tensor]]:
    data = materialize(model, batch)
    state = data["state"]
    logits_trace: list[Tensor] = []
    for round_index in range(rounds):
        round_immediate = immediate_vectors(model, data["immediates"][:, round_index])
        state, candidates, _ = model.step(
            state,
            data["memory_keys"],
            data["memory_values"],
            data["memory_types"],
            data["row_mask"],
            data["opcodes"][:, round_index],
            round_immediate,
            data["source_slots"][:, round_index],
            data["destination_slots"][:, round_index],
            data["presence"],
            read_set="legacy",
        )
        logits_trace.append(candidates.alu_logits if candidates.alu_logits is not None else torch.zeros(state.shape[0], 32, device=state.device, dtype=state.dtype))
    return state, logits_trace


def class_ids_for_task(task: str, device: torch.device) -> Tensor:
    if task in {"pointer_chasing"}:
        values = POINTER_CLASS_IDS
    elif task == "multi_hop":
        values = MULTI_HOP_CLASS_IDS
    elif task in {"associative_recall", "sequential_update"}:
        values = VALUE_CLASS_IDS
    elif task == "variable_binding":
        values = COLOR_CLASS_IDS
    else:
        raise ValueError(f"task has no discrete decoder: {task}")
    return torch.tensor(values, dtype=torch.long, device=device)


def decode(model: UnifiedT1U0, slot_state: Tensor, class_ids: Tensor, decoder: nn.Module) -> Tensor:
    return decoder(slot_state, model.token_embedding(class_ids)), class_ids


def task_loss(model: UnifiedT1U0, task: str, state: Tensor, batch: dict[str, Any], logits_trace: list[Tensor] | None = None) -> Tensor:
    if task == "workspace_accumulation":
        return (state[:, SLOT_W, :] - batch["target_vectors"]).square().mean()
    slot = SLOT_P if task in {"pointer_chasing", "multi_hop"} else SLOT_E if task in {"associative_recall", "variable_binding"} else SLOT_R
    decoder = model.pointer_decoder if slot == SLOT_P else model.evidence_decoder if slot == SLOT_E else model.register_decoder
    class_ids = class_ids_for_task(task, state.device)
    logits, class_ids = decode(model, state[:, slot, :], class_ids, decoder)
    target_index = torch.searchsorted(class_ids, batch["target_ids"])
    loss = nn.functional.cross_entropy(logits, target_index)
    if task == "sequential_update" and logits_trace is not None:
        intermediate_losses: list[Tensor] = []
        for round_index, round_logits in enumerate(logits_trace):
            targets = batch["intermediate_target_ids"][:, round_index]
            active = targets >= 0
            if active.any():
                intermediate_losses.append(nn.functional.cross_entropy(round_logits[active], (targets[active] - VALUE_BASE).to(dtype=torch.long)))
        if intermediate_losses:
            loss = loss + torch.stack(intermediate_losses).mean()
    return loss


def train_one_step(model: UnifiedT1U0, task: str, batch: dict[str, Any]) -> Tensor:
    if task == "sequential_update":
        state, logits_trace = run_rounds_with_trace(model, batch, batch["opcodes"].shape[1])
        return task_loss(model, task, state, batch, logits_trace)
    state = run_rounds(model, batch, batch["opcodes"].shape[1])
    return task_loss(model, task, state, batch)


@torch.no_grad()
def evaluate_accuracy(model: UnifiedT1U0, task: str, examples: list[CanonicalExample], rounds: int) -> float:
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    correct = 0
    count = 0
    for batch in loader:
        state = run_rounds(model, batch, min(rounds, batch["opcodes"].shape[1]))
        if task == "workspace_accumulation":
            error = torch.linalg.vector_norm(state[:, SLOT_W] - batch["target_vectors"], dim=-1)
            target_norm = torch.linalg.vector_norm(batch["target_vectors"], dim=-1).clamp_min(1e-8)
            correct += int(((error / target_norm) < 1e-3).sum())
        else:
            slot = SLOT_P if task in {"pointer_chasing", "multi_hop"} else SLOT_E if task in {"associative_recall", "variable_binding"} else SLOT_R
            decoder = model.pointer_decoder if slot == SLOT_P else model.evidence_decoder if slot == SLOT_E else model.register_decoder
            class_ids = class_ids_for_task(task, state.device)
            logits, class_ids = decode(model, state[:, slot], class_ids, decoder)
            predicted = class_ids[logits.argmax(dim=-1)]
            correct += int((predicted == batch["target_ids"]).sum())
        count += len(batch["target_ids"])
    return correct / count


@torch.no_grad()
def evaluate_matrix(model: UnifiedT1U0, task: str, examples: list[CanonicalExample], rounds_values: tuple[int, ...], hop_values: tuple[int, ...]) -> dict[str, dict[str, float]]:
    matrix = {str(hop): {str(rounds): 0.0 for rounds in rounds_values} for hop in hop_values}
    counts = {str(hop): {str(rounds): 0 for rounds in rounds_values} for hop in hop_values}
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    for batch in loader:
        for rounds in rounds_values:
            state = run_rounds(model, batch, min(rounds, batch["opcodes"].shape[1]))
            if task == "workspace_accumulation":
                values = torch.linalg.vector_norm(state[:, SLOT_W] - batch["target_vectors"], dim=-1) / torch.linalg.vector_norm(batch["target_vectors"], dim=-1).clamp_min(1e-8)
                hits = values < 1e-3
            else:
                slot = SLOT_P if task in {"pointer_chasing", "multi_hop"} else SLOT_E if task in {"associative_recall", "variable_binding"} else SLOT_R
                decoder = model.pointer_decoder if slot == SLOT_P else model.evidence_decoder if slot == SLOT_E else model.register_decoder
                class_ids = class_ids_for_task(task, state.device)
                logits, class_ids = decode(model, state[:, slot], class_ids, decoder)
                hits = class_ids[logits.argmax(dim=-1)] == batch["target_ids"]
            for hop in hop_values:
                selected = batch["hops"] == hop
                matrix[str(hop)][str(rounds)] += float(hits[selected].sum())
                counts[str(hop)][str(rounds)] += int(selected.sum())
    return {hop: {rounds: matrix[hop][rounds] / counts[hop][rounds] for rounds in matrix[hop]} for hop in matrix}


@torch.no_grad()
def evaluate_workspace_error(model: UnifiedT1U0, examples: list[CanonicalExample]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {str(hop): {} for hop in (2, 4, 6)}
    loader = DataLoader(ExampleDataset(examples), batch_size=256, shuffle=False, collate_fn=collate)
    for rounds in (1, 2, 4, 6):
        for batch in loader:
            state = run_rounds(model, batch, min(rounds, batch["opcodes"].shape[1]))
            error = torch.linalg.vector_norm(state[:, SLOT_W] - batch["target_vectors"], dim=-1) / torch.linalg.vector_norm(batch["target_vectors"], dim=-1).clamp_min(1e-8)
            for hop in (2, 4, 6):
                selected = batch["hops"] == hop
                result[str(hop)].setdefault(str(rounds), []).extend(error[selected].tolist())
    return {hop: {rounds: float(sum(values) / len(values)) for rounds, values in rounds_map.items()} for hop, rounds_map in result.items()}


def evaluate_all(model: UnifiedT1U0, datasets: dict[str, dict[str, list[CanonicalExample]]], split: str) -> dict[str, object]:
    outputs: dict[str, object] = {}
    outputs["pointer_chasing"] = evaluate_matrix(model, "pointer_chasing", datasets["pointer_chasing"][split], (1, 2, 4), (1, 2, 3, 4))
    outputs["multi_hop"] = evaluate_matrix(model, "multi_hop", datasets["multi_hop"][split], (1, 2, 3, 4), (1, 2, 3, 4))
    outputs["associative_recall"] = evaluate_matrix(model, "associative_recall", datasets["associative_recall"][split], (1, 2, 4), (1,))
    outputs["variable_binding"] = evaluate_matrix(model, "variable_binding", datasets["variable_binding"][split], (1, 2, 4), (2,))
    outputs["sequential_update"] = evaluate_matrix(model, "sequential_update", datasets["sequential_update"][split], (1, 2, 4, 6), (3, 4, 5, 6))
    outputs["workspace_accumulation"] = evaluate_workspace_error(model, datasets["workspace_accumulation"][split])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "u0a_seed101")
    parser.add_argument("--resume", action="store_true", help="resume from output-dir/latest.pt")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets = build_canonical_data(args.output_dir)
    sequential_h1 = build_sequential_h1_table()
    model = UnifiedT1U0(DIMENSION)
    optimizer = build_optimizer(model)
    loaders = {task: DataLoader(ExampleDataset(datasets[task]["train"]), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed + index), collate_fn=collate) for index, task in enumerate(TASKS)}
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    sequential_h1_loader = DataLoader(ExampleDataset(sequential_h1), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed + len(TASKS)), collate_fn=collate)
    sequential_h1_iterator = iter(sequential_h1_loader)
    config = {"phase": "T1-U0-A ISO-UPDATE unified oracle routing", "seed": args.seed, "tasks": list(TASKS), "dimension": DIMENSION, "state": "X=[P,R,E,W]", "batching": "one homogeneous batch per task per superstep; mean combined loss", "steps": args.steps, "supersteps": args.steps, "batches_per_task": args.steps, "batch_size": BATCH_SIZE, "sequential_schedule": "five supersteps H1 complete 3072 transition table, then one superstep existing H3-H6 composition source; at 12000: 10000 H1 + 2000 composition", "reader": "one SharedMemoryReader; one read per READ_P/READ_E/ACCUM_W round", "core": "one SharedRecurrentCore shared across tasks and rounds", "commit": "TypedCommit direct Y paths; P/E replacement, R canonicalized ALU overwrite, W residual", "workspace_correction": "hard-frozen in U0-A; reserved for U0-C", "routing": "oracle opcode/immediate/source/destination/row mask", "optimizer": "AdamW lr=3e-4; weight_decay=1e-4 trunk only; typed embeddings/heads/decoders/norms/biases weight_decay=0; one step per superstep", "loss_combination": "sum six task losses divided by six", "datasets": "existing sources adapted to canonical REL/PAIR/ASSIGN/ATTR/VEC; sequential H1 replay table generated in-memory; no source regeneration", "resumable": "latest.pt stores model, optimizer, step, best metrics, and RNG state; --resume continues at next absolute superstep"}
    save_json(args.output_dir / "config.json", config)
    metrics_path = args.output_dir / "metrics.jsonl"
    resume_path = args.output_dir / "latest.pt"
    start_step = 0
    best_score = -float("inf")
    best_step = 0
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_score = float(checkpoint["best_score"])
        best_step = int(checkpoint["best_step"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
    else:
        metrics_path.unlink(missing_ok=True)
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        task_losses: dict[str, Tensor] = {}
        for task in TASKS:
            if task == "sequential_update" and step % 6 != 0:
                try:
                    batch = next(sequential_h1_iterator)
                except StopIteration:
                    sequential_h1_iterator = iter(sequential_h1_loader)
                    batch = next(sequential_h1_iterator)
            else:
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(loaders[task])
                    batch = next(iterators[task])
            task_loss_value = train_one_step(model, task, batch)
            if not torch.isfinite(task_loss_value):
                raise FloatingPointError(f"non-finite loss for {task} at superstep {step}")
            task_losses[task] = task_loss_value
        loss = torch.stack(tuple(task_losses.values())).mean()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at superstep {step}")
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            validation = {task_name: evaluate_accuracy(model, task_name, datasets[task_name]["val"], rounds=6) if task_name != "workspace_accumulation" else evaluate_accuracy(model, task_name, datasets[task_name]["val"], rounds=6) for task_name in TASKS}
            score = sum(validation.values()) / len(validation)
            metric = {"superstep": step, "loss": float(loss.detach()), "task_loss": {task: float(value.detach()) for task, value in task_losses.items()}, "gradient_norm": grad_norm, "validation_score": score, "validation_accuracy": validation}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if score > best_score:
                best_score = score
                best_step = step
                save_training_checkpoint(args.output_dir / "best.pt", config=config, step=step, model=model, optimizer=optimizer, best_score=best_score, best_step=best_step)
            save_training_checkpoint(resume_path, config=config, step=step, model=model, optimizer=optimizer, best_score=best_score, best_step=best_step)
            model.train()
    if start_step == args.steps:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_outputs = evaluate_all(model, datasets, "test")
    test_outputs["sequential_update_h1_table"] = evaluate_accuracy(model, "sequential_update", sequential_h1, rounds=1)
    final = {"status": "completed", "finite": True, "seed": args.seed, "steps": args.steps, "supersteps": args.steps, "batches_per_task": args.steps, "optimizer_steps": args.steps, "sequential_h1_batches": args.steps - args.steps // 6, "sequential_composition_batches": args.steps // 6, "best_step": best_step, "best_validation_score": best_score, "test": test_outputs, "elapsed_seconds": time.perf_counter() - started, "resumed_from_step": start_step}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
