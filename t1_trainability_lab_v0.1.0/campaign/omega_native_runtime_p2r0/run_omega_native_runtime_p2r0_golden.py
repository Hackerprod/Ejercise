"""Dump P2-R0 recurrent-forward golden cases.

This module is Python-only.  It uses the verified P0 forward oracle to create
portable float32 inputs, parameters, outputs, upstreams, and autograd
gradients for the later standalone C-ABI kernel.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
P0_DIR = CAMPAIGN_ROOT / "omega_native_runtime_p0"
CE_DIR = CAMPAIGN_ROOT / "omega_ce_only_baseline"
PROBE_DIR = CAMPAIGN_ROOT / "omega_teacher_hidden_cache_probe"
for path in (P0_DIR, CE_DIR, PROBE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_omega_native_runtime_p0 as p0  # noqa: E402
import run_omega_ce_only_baseline as ce  # noqa: E402


CAMPAIGN_ID = "OMEGA-NATIVE-RUNTIME-P2R0"
DEFAULT_OUTPUT_ROOT = HERE / "golden_cases"
WINDOW_TOKENS = 256
BATCH = 8
SLOTS = 8
DIMENSION = 128
EPS = 1e-6
MAGIC = "OMEGA-P2R0-GOLDEN"
FORMAT_VERSION = 1
HEADER_LENGTH_STRUCT = "<I"

TENSOR_ORDER = (
    "token_part",
    "previous_state",
    "state_part_weight",
    "prelude_norm_weight",
    "block_qkv_weight",
    "block_qkv_bias",
    "block_out_weight",
    "block_out_bias",
    "block_fc1_weight",
    "block_fc1_bias",
    "block_fc2_weight",
    "block_fc2_bias",
    "block_norm_weight",
    "depth_embedding_weight",
    "gate_logits",
    "next_state",
    "readout_states",
    "d_readout_states",
    "d_next_state",
    "d_token_part",
    "d_previous_state",
    "d_state_part_weight",
    "d_prelude_norm_weight",
    "d_block_qkv_weight",
    "d_block_qkv_bias",
    "d_block_out_weight",
    "d_block_out_bias",
    "d_block_fc1_weight",
    "d_block_fc1_bias",
    "d_block_fc2_weight",
    "d_block_fc2_bias",
    "d_block_norm_weight",
    "d_depth_embedding_weight",
    "d_gate_logits",
)


def _recurrent_forward_from_token_part(model: nn.Module, token_part: Tensor, previous_state: Tensor) -> tuple[Tensor, Tensor]:
    """Production recurrent loop with embedding/prelude already materialized."""
    batch, sequence, state_dimension = token_part.shape
    slots = int(model.slots)
    dimension = int(model.dimension)
    if state_dimension != slots * dimension:
        raise ValueError("token_part last dimension does not match model state dimension")
    state_part_weight = model.prelude.weight[:, dimension:]
    depth_biases, gates = model._round_constants()
    blocks = [model.blocks[0] if model.variant == "shared" else model.blocks[index] for index in range(model.rounds)]
    state = previous_state
    readout_states: list[Tensor] = []
    for position in range(sequence):
        write = token_part[:, position] + torch.nn.functional.linear(state.mean(dim=1), state_part_weight)
        anchor = torch.nn.functional.rms_norm(
            state.flatten(start_dim=1) + write,
            (state_dimension,),
            model.prelude_norm_weight,
            EPS,
        ).view(batch, slots, dimension)
        candidate = anchor
        for round_index, block in enumerate(blocks):
            candidate = block(candidate, anchor, depth_biases[round_index], gates[round_index])
        readout_states.append(candidate)
        state = candidate
    return state, torch.stack(readout_states, dim=1)


def _tensor(value: Tensor) -> Tensor:
    return value.detach().cpu().contiguous().to(dtype=torch.float32)


def _require_grad(value: Tensor, name: str) -> Tensor:
    if value.grad is None:
        raise RuntimeError(f"missing autograd gradient: {name}")
    return _tensor(value.grad)


def make_golden_case(model: nn.Module, token_ids: Tensor, *, k: int, window: int, seed: int, source_pair: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one case using P0's verified forward and real PyTorch backward."""
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be rank 2, got {tuple(token_ids.shape)}")
    batch = int(token_ids.shape[0])
    slots = int(model.slots)
    dimension = int(model.dimension)
    previous_state = torch.randn((batch, slots, dimension), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    with torch.no_grad():
        _, _, boundaries, _ = p0._student_forward_timed(model, token_ids, previous_state)
        token_part = boundaries["embedding_prelude_boundary"].detach().clone()
        expected_next_state, expected_readout = _recurrent_forward_from_token_part(model, token_part, previous_state)
        oracle_next_state = expected_next_state.detach().clone()
        oracle_readout = expected_readout.detach().clone()
    # P0 forward and the explicit recurrent-only oracle must agree exactly.
    if not torch.equal(oracle_next_state, boundaries["recurrent_readout_boundary"][:, -1].detach()):
        raise AssertionError("recurrent next_state differs from P0 oracle")
    if not torch.equal(oracle_readout, boundaries["recurrent_readout_boundary"].detach()):
        raise AssertionError("recurrent readout differs from P0 oracle")

    torch.manual_seed(seed + 1)
    d_readout_states = torch.randn_like(oracle_readout)
    d_next_state = torch.randn_like(oracle_next_state)

    model.zero_grad(set_to_none=True)
    token_part_for_backward = token_part.detach().clone().requires_grad_(True)
    token_part_for_backward.retain_grad()
    previous_state_for_backward = previous_state.detach().clone().requires_grad_(True)
    previous_state_for_backward.retain_grad()
    next_state, readout_states = _recurrent_forward_from_token_part(model, token_part_for_backward, previous_state_for_backward)
    objective = (readout_states * d_readout_states).sum() + (next_state * d_next_state).sum()
    objective.backward()

    block = model.blocks[0]
    dimension = int(model.dimension)
    gradients = {
        "d_token_part": _require_grad(token_part_for_backward, "token_part"),
        "d_previous_state": _require_grad(previous_state_for_backward, "previous_state"),
        "d_state_part_weight": _tensor(model.prelude.weight.grad[:, dimension:]) if model.prelude.weight.grad is not None else None,
        "d_prelude_norm_weight": _require_grad(model.prelude_norm_weight, "prelude_norm_weight"),
        "d_block_qkv_weight": _require_grad(block.qkv.weight, "block.qkv.weight"),
        "d_block_qkv_bias": _require_grad(block.qkv.bias, "block.qkv.bias"),
        "d_block_out_weight": _require_grad(block.out.weight, "block.out.weight"),
        "d_block_out_bias": _require_grad(block.out.bias, "block.out.bias"),
        "d_block_fc1_weight": _require_grad(block.fc1.weight, "block.fc1.weight"),
        "d_block_fc1_bias": _require_grad(block.fc1.bias, "block.fc1.bias"),
        "d_block_fc2_weight": _require_grad(block.fc2.weight, "block.fc2.weight"),
        "d_block_fc2_bias": _require_grad(block.fc2.bias, "block.fc2.bias"),
        "d_block_norm_weight": _require_grad(block.norm_weight, "block.norm_weight"),
        "d_depth_embedding_weight": _require_grad(model.depth_embedding.weight, "depth_embedding.weight"),
        "d_gate_logits": _require_grad(model.gate_logits, "gate_logits"),
    }
    if gradients["d_state_part_weight"] is None:
        raise RuntimeError("missing autograd gradient: state_part_weight")

    tensors: dict[str, Tensor] = {
        "token_part": _tensor(token_part),
        "previous_state": _tensor(previous_state),
        "state_part_weight": _tensor(model.prelude.weight[:, dimension:]),
        "prelude_norm_weight": _tensor(model.prelude_norm_weight),
        "block_qkv_weight": _tensor(block.qkv.weight),
        "block_qkv_bias": _tensor(block.qkv.bias),
        "block_out_weight": _tensor(block.out.weight),
        "block_out_bias": _tensor(block.out.bias),
        "block_fc1_weight": _tensor(block.fc1.weight),
        "block_fc1_bias": _tensor(block.fc1.bias),
        "block_fc2_weight": _tensor(block.fc2.weight),
        "block_fc2_bias": _tensor(block.fc2.bias),
        "block_norm_weight": _tensor(block.norm_weight),
        "depth_embedding_weight": _tensor(model.depth_embedding.weight),
        "gate_logits": _tensor(model.gate_logits),
        "next_state": _tensor(oracle_next_state),
        "readout_states": _tensor(oracle_readout),
        "d_readout_states": _tensor(d_readout_states),
        "d_next_state": _tensor(d_next_state),
        **{name: value for name, value in gradients.items() if value is not None},
    }
    if tuple(tensors) != TENSOR_ORDER:
        raise AssertionError("golden tensor order drift")
    metadata = {
        "campaign_id": CAMPAIGN_ID,
        "case": f"K{k}_window{window}",
        "K": k,
        "window": window,
        "source_pair": source_pair,
        "seed": seed,
        "config": {"batch": batch, "slots": slots, "dimension": dimension, "rounds": k, "sequence_length": int(token_ids.shape[1]), "dtype": "float32", "eps": EPS},
        "gradient_objective": "sum(readout_states * d_readout_states) + sum(next_state * d_next_state)",
    }
    return tensors, metadata


def write_golden_case(path: Path, tensors: Mapping[str, Tensor], metadata: Mapping[str, Any]) -> None:
    if tuple(tensors) != TENSOR_ORDER:
        raise ValueError("tensors must follow TENSOR_ORDER")
    payload = bytearray()
    descriptors: dict[str, Any] = {}
    for name in TENSOR_ORDER:
        value = _tensor(tensors[name])
        raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
        descriptors[name] = {"shape": list(value.shape), "dtype": "float32", "byte_offset": len(payload), "byte_length": len(raw)}
        payload.extend(raw)
    header = {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "header_encoding": "utf-8 JSON",
        "endianness": "little",
        "payload_dtype": "float32",
        "tensor_order": list(TENSOR_ORDER),
        "tensors": descriptors,
        "metadata": dict(metadata),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(HEADER_LENGTH_STRUCT, len(header_bytes)) + header_bytes + payload)


def read_golden_case(path: Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    raw = path.read_bytes()
    (header_length,) = struct.unpack_from(HEADER_LENGTH_STRUCT, raw, 0)
    header_start = struct.calcsize(HEADER_LENGTH_STRUCT)
    header = json.loads(raw[header_start : header_start + header_length].decode("utf-8"))
    if header.get("magic") != MAGIC or header.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported golden format")
    payload_start = header_start + header_length
    tensors: dict[str, Tensor] = {}
    for name in header["tensor_order"]:
        descriptor = header["tensors"][name]
        begin = payload_start + int(descriptor["byte_offset"])
        end = begin + int(descriptor["byte_length"])
        array = torch.frombuffer(bytearray(raw[begin:end]), dtype=torch.float32).clone().reshape(tuple(descriptor["shape"]))
        tensors[name] = array
    return header, tensors


def _load_tokens() -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    frozen, documents, _ = p0.hidden.base.load_frozen_train_documents()
    return documents, frozen["cyclic_pairs"]["pairs"]


def dump_golden_cases(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[Path]:
    documents, pairs = _load_tokens()
    paths: list[Path] = []
    for k in (1, 4):
        model = ce.fresh_model(20260913, k)
        for window in (0, 1):
            source_pair = 0
            positions = [int(index) for index in pairs[source_pair]["document_indices"]]
            source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
            offset = window * WINDOW_TOKENS
            token_ids = source[:, offset : offset + WINDOW_TOKENS]
            tensors, metadata = make_golden_case(model, token_ids, k=k, window=window, seed=20261000 + k * 10 + window, source_pair=source_pair)
            path = output_root / f"golden_K{k}_window{window}.bin"
            write_golden_case(path, tensors, metadata)
            paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    paths = dump_golden_cases(args.output_root)
    print(json.dumps({"campaign_id": CAMPAIGN_ID, "files": [path.as_posix() for path in paths]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
