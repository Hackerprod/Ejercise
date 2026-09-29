"""Capture and certify real K4 recurrent backward replay fixtures.

Modes run in separate processes:
  capture     runs canonical CE+KL updates 0..3 and writes fixtures;
  torch-check replays isolated PyTorch forward/backward from fixtures;
  native-check loads only NumPy/ctypes and invokes the native 4-worker ABI.

This utility performs correctness checks only. It contains no benchmark loop.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
P0_DIR = CAMPAIGN_ROOT / "omega_native_runtime_p0"
CE_DIR = CAMPAIGN_ROOT / "omega_ce_only_baseline"
FAST_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
DEFAULT_OUTPUT_ROOT = HERE / "results" / "recurrent_backward_replay_final"

MAGIC = "OMEGA-RECURRENT-BACKWARD-REPLAY"
FORMAT_VERSION = 1
HEADER_LENGTH = struct.Struct("<I")
MODEL_SEED = 20260913
ROUNDS = 4
BATCH = 8
WINDOW_TOKENS = 256
SLOTS = 8
DIMENSION = 128
EPS = 1e-6
ACCEPTED_DLL_SHA256 = "6f38e3b1dcada32c98aef50e870e201510bab9c75b875438823915ba53b854d5"

PARAMETER_NAMES = (
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
)
GRADIENT_NAMES = (
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
TENSOR_ORDER = (
    "token_ids",
    "targets",
    "token_part",
    "previous_state",
    "state_part_weight_storage",
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
    "G_R",
    "G_S",
    *GRADIENT_NAMES,
)
FLOAT_TENSORS = frozenset(name for name in TENSOR_ORDER if name not in {"token_ids", "targets"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _tensor_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value)
    dtype = "<i8" if name in {"token_ids", "targets"} else "<f4"
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def write_fixture(
    path: Path,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_layouts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(tensors) != TENSOR_ORDER:
        raise ValueError("fixture tensors must follow TENSOR_ORDER exactly")
    payload = bytearray()
    descriptors: dict[str, Any] = {}
    for name in TENSOR_ORDER:
        array = _tensor_numpy(tensors[name], name)
        raw = array.tobytes(order="C")
        descriptor: dict[str, Any] = {
            "shape": [int(item) for item in array.shape],
            "dtype": "int64" if name in {"token_ids", "targets"} else "float32",
            "byte_offset": len(payload),
            "byte_length": len(raw),
            "sha256": _sha256_bytes(raw),
        }
        if name in source_layouts:
            descriptor["source_layout"] = dict(source_layouts[name])
        descriptors[name] = descriptor
        payload.extend(raw)
    header = {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "header_encoding": "UTF-8 JSON",
        "endianness": "little",
        "tensor_order": list(TENSOR_ORDER),
        "tensors": descriptors,
        "payload_sha256": _sha256_bytes(bytes(payload)),
        "metadata": dict(metadata),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    raw_file = HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {path}")
    path.write_bytes(raw_file)
    return {
        "path": path.as_posix(),
        "bytes": len(raw_file),
        "sha256": _sha256_bytes(raw_file),
        "payload_sha256": header["payload_sha256"],
        "metadata": dict(metadata),
    }


def read_fixture(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    raw = path.read_bytes()
    if len(raw) < HEADER_LENGTH.size:
        raise ValueError(f"fixture is shorter than header prefix: {path}")
    (header_length,) = HEADER_LENGTH.unpack_from(raw, 0)
    header_start = HEADER_LENGTH.size
    payload_start = header_start + header_length
    if payload_start > len(raw):
        raise ValueError(f"fixture header exceeds file length: {path}")
    header = json.loads(raw[header_start:payload_start].decode("utf-8"))
    if header.get("magic") != MAGIC or header.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported replay fixture format: {path}")
    if tuple(header.get("tensor_order", ())) != TENSOR_ORDER:
        raise ValueError(f"fixture tensor order mismatch: {path}")
    payload = raw[payload_start:]
    if _sha256_bytes(payload) != header.get("payload_sha256"):
        raise ValueError(f"fixture payload SHA-256 mismatch: {path}")
    arrays: dict[str, np.ndarray] = {}
    for name in TENSOR_ORDER:
        descriptor = header["tensors"][name]
        dtype = np.dtype("<i8" if descriptor["dtype"] == "int64" else "<f4")
        shape = tuple(int(item) for item in descriptor["shape"])
        count = math.prod(shape)
        begin = int(descriptor["byte_offset"])
        length = int(descriptor["byte_length"])
        if length != count * dtype.itemsize or begin < 0 or begin + length > len(payload):
            raise ValueError(f"invalid tensor descriptor for {name}: {path}")
        value_bytes = payload[begin : begin + length]
        if _sha256_bytes(value_bytes) != descriptor["sha256"]:
            raise ValueError(f"tensor SHA-256 mismatch for {name}: {path}")
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=count, offset=begin).reshape(shape)
    if sum(int(header["tensors"][name]["byte_length"]) for name in TENSOR_ORDER) != len(payload):
        raise ValueError(f"fixture has unclaimed payload bytes: {path}")
    state_layout = header["tensors"]["state_part_weight"].get("source_layout", {})
    if state_layout.get("strides_elements") != [2 * DIMENSION, 1] or state_layout.get("storage_offset_elements") != DIMENSION:
        raise ValueError("fixture did not preserve fused prelude state-view layout")
    if not np.array_equal(arrays["state_part_weight_storage"][:, DIMENSION:], arrays["state_part_weight"]):
        raise ValueError("state_part_weight logical values do not match saved fused storage view")
    if not np.isfinite(arrays["G_R"]).all() or not np.isfinite(arrays["G_S"]).all():
        raise ValueError("upstream gradients contain non-finite values")
    if np.any(arrays["G_S"] != 0):
        raise ValueError("G_S must be exactly zero for detached state transport")
    return header, arrays, _sha256_bytes(raw)


def _training_modules() -> tuple[Any, Any, Any]:
    for directory in (HERE, P0_DIR, CE_DIR, FAST_DIR):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    r2 = importlib.import_module("run_omega_native_runtime_r2_benchmark")
    p0, ce, _bridge = r2._load_r1_modules()
    return r2, p0, ce


def _model_state_hash(ce: Any, model: Any) -> str:
    return str(ce.state_dict_hash(model.state_dict()))


def _capture_oracle_gradients(model: Any, token_part: Any, previous_state: Any, upstream_readout: Any, golden: Any) -> tuple[dict[str, Any], Any, Any]:
    import torch

    token_leaf = token_part.detach().clone().requires_grad_(True)
    state_leaf = previous_state.detach().clone().requires_grad_(True)
    next_state, readout_states = golden._recurrent_forward_from_token_part(model, token_leaf, state_leaf)
    upstream_next = torch.zeros_like(next_state)
    block = model.blocks[0]
    requested = (
        token_leaf,
        state_leaf,
        model.prelude.weight,
        model.prelude_norm_weight,
        block.qkv.weight,
        block.qkv.bias,
        block.out.weight,
        block.out.bias,
        block.fc1.weight,
        block.fc1.bias,
        block.fc2.weight,
        block.fc2.bias,
        block.norm_weight,
        model.depth_embedding.weight,
        model.gate_logits,
    )
    gradients = torch.autograd.grad(
        (next_state, readout_states),
        requested,
        grad_outputs=(upstream_next, upstream_readout),
    )
    named = {
        "d_token_part": gradients[0].detach().clone(),
        "d_previous_state": gradients[1].detach().clone(),
        "d_state_part_weight": gradients[2][:, DIMENSION:].detach().contiguous().clone(),
        "d_prelude_norm_weight": gradients[3].detach().clone(),
        "d_block_qkv_weight": gradients[4].detach().clone(),
        "d_block_qkv_bias": gradients[5].detach().clone(),
        "d_block_out_weight": gradients[6].detach().clone(),
        "d_block_out_bias": gradients[7].detach().clone(),
        "d_block_fc1_weight": gradients[8].detach().clone(),
        "d_block_fc1_bias": gradients[9].detach().clone(),
        "d_block_fc2_weight": gradients[10].detach().clone(),
        "d_block_fc2_bias": gradients[11].detach().clone(),
        "d_block_norm_weight": gradients[12].detach().clone(),
        "d_depth_embedding_weight": gradients[13].detach().clone(),
        "d_gate_logits": gradients[14].detach().clone(),
    }
    return named, next_state.detach(), readout_states.detach()


def capture_fixtures(manifest_path: Path | None, cache_file: Path | None, output_root: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    expected_paths = (
        output_root / "fixtures" / "replay_K4_update2_window0.bin",
        output_root / "fixtures" / "replay_K4_update3_window1.bin",
        output_root / "capture_report.json",
    )
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite replay artifacts: {existing}")
    r2, p0, ce = _training_modules()
    policy = ce.validate_policy()
    if int(torch.get_num_threads()) != 4 or int(torch.get_num_interop_threads()) != 1:
        raise RuntimeError("PyTorch replay capture requires intraop=4 and interop=1")
    memory_gate = p0.integration.memory_safety_gate()
    manifest_path = manifest_path or Path(p0.SEALED_MANIFEST)
    cache_file = cache_file or Path(p0.DEFAULT_CACHE_FILE)
    if not manifest_path.is_file() or not cache_file.is_file():
        raise FileNotFoundError("sealed teacher manifest or hidden cache is missing")

    _provenance = p0.integration.verify_provenance(manifest_path, cache_file=cache_file)
    manifest, resolved_cache = p0.hidden.hidden_cache_load_manifest(manifest_path)
    teacher_payload, _, _ = p0.hidden.preload_hidden_cache(resolved_cache, manifest)
    teacher_weight, teacher_bias = p0.hidden.load_lm_head(manifest_path.parent, manifest["lm_head"])
    frozen, documents, _ = p0.hidden.base.load_frozen_train_documents()

    model = ce.fresh_model(MODEL_SEED, ROUNDS)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ce.BASE_LR,
        betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS,
        weight_decay=ce.WEIGHT_DECAY,
    )
    state = model.initial_state(int(p0.PHYSICAL_BATCH), device=torch.device("cpu"))
    golden = importlib.import_module("run_omega_native_runtime_p2r0_golden")
    fixtures: list[dict[str, Any]] = []
    target_updates = {2, 3}
    source_pair = 1

    for update in range(4):
        window = update % 2
        pair_index = update // 2
        positions = [int(index) for index in frozen["cyclic_pairs"]["pairs"][pair_index]["document_indices"]]
        source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
        offset = window * WINDOW_TOKENS
        token_ids = source[:, offset : offset + WINDOW_TOKENS]
        targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
        if window == 0:
            state = model.initial_state(int(p0.PHYSICAL_BATCH), device=torch.device("cpu"))
        previous_state = state.detach().clone()
        teacher_logits = r2._teacher_logits(p0, teacher_payload, teacher_weight, teacher_bias, positions, window)
        optimizer.zero_grad(set_to_none=True)
        next_state, student_logits, boundaries = model.forward_window(token_ids, previous_state)
        token_part = F.linear(
            model.embedding(token_ids), model.prelude.weight[:, :DIMENSION], model.prelude.bias
        )
        readout_states = boundaries["readout_states"]
        losses = p0.hidden.base.distillation_loss(student_logits, teacher_logits, targets)

        if update in target_updates:
            if pair_index != source_pair:
                raise AssertionError("requested replay updates must use source_pair=1")
            # Production forward is untimed; verify its recurrent boundary.
            token_part_direct = F.linear(
                model.embedding(token_ids), model.prelude.weight[:, :DIMENSION], model.prelude.bias
            )
            if not torch.equal(token_part_direct, token_part):
                raise AssertionError(f"production token_part mismatch at update={update}")
            if not torch.equal(next_state, readout_states[:, -1]):
                raise AssertionError(f"production next_state/readout boundary mismatch at update={update}")

            upstream_readout = torch.autograd.grad(losses["total"], readout_states, retain_graph=True)[0].detach().clone()
            if not torch.isfinite(upstream_readout).all() or float(torch.linalg.vector_norm(upstream_readout).item()) == 0.0:
                raise AssertionError(f"real CE+KL G_R is invalid or zero at update={update}")
            recurrent_gradients, isolated_next, isolated_readout = _capture_oracle_gradients(
                model, token_part, previous_state, upstream_readout, golden
            )
            if not torch.equal(isolated_next, next_state) or not torch.equal(isolated_readout, readout_states):
                raise AssertionError(f"isolated recurrent forward differs from P0 at update={update}")
            if tuple(model.prelude.weight[:, DIMENSION:].stride()) != (2 * DIMENSION, 1):
                raise AssertionError("training state_part_weight lost its fused prelude row stride")

            block = model.blocks[0]
            tensors: dict[str, Any] = {
                "token_ids": token_ids.detach().clone(),
                "targets": targets.detach().clone(),
                "token_part": token_part.detach().clone(),
                "previous_state": previous_state.detach().clone(),
                "state_part_weight_storage": model.prelude.weight.detach().clone(),
                "state_part_weight": model.prelude.weight[:, DIMENSION:].detach().clone(),
                "prelude_norm_weight": model.prelude_norm_weight.detach().clone(),
                "block_qkv_weight": block.qkv.weight.detach().clone(),
                "block_qkv_bias": block.qkv.bias.detach().clone(),
                "block_out_weight": block.out.weight.detach().clone(),
                "block_out_bias": block.out.bias.detach().clone(),
                "block_fc1_weight": block.fc1.weight.detach().clone(),
                "block_fc1_bias": block.fc1.bias.detach().clone(),
                "block_fc2_weight": block.fc2.weight.detach().clone(),
                "block_fc2_bias": block.fc2.bias.detach().clone(),
                "block_norm_weight": block.norm_weight.detach().clone(),
                "depth_embedding_weight": model.depth_embedding.weight.detach().clone(),
                "gate_logits": model.gate_logits.detach().clone(),
                "next_state": next_state.detach().clone(),
                "readout_states": readout_states.detach().clone(),
                "G_R": upstream_readout,
                "G_S": torch.zeros_like(next_state),
                **recurrent_gradients,
            }
            if tuple(tensors) != TENSOR_ORDER:
                raise AssertionError("replay tensor order drift")

            state_source = (
                "fresh zero initial_state at window=0"
                if update == 2
                else "detached update=2/window=0 next_state transported after update=2 AdamW step"
            )
            teacher_raw = teacher_logits.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
            input_raw = token_ids.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()
            target_raw = targets.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()
            metadata = {
                "schema": "omega-recurrent-backward-replay-v1",
                "fixture_id": f"K4_update{update}_window{window}",
                "model_seed": MODEL_SEED,
                "K": ROUNDS,
                "update_index": update,
                "window_index": window,
                "pair_index": pair_index,
                "source_pair_index": source_pair,
                "source_document_positions": positions,
                "state_source": state_source,
                "prior_adamw_updates_completed": update,
                "warmup_updates": 0,
                "model_state_sha256_before_update": _model_state_hash(ce, model),
                "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
                "teacher_logits_sha256": _sha256_bytes(teacher_raw),
                "token_ids_sha256": _sha256_bytes(input_raw),
                "targets_sha256": _sha256_bytes(target_raw),
                "loss": {
                    "ce": float(losses["ce"].detach().item()),
                    "kl": float(losses["kl"].detach().item()),
                    "total": float(losses["total"].detach().item()),
                    "formula": "0.5 * CE + 0.5 * KL; CE uses mean reduction; KL uses batchmean * temperature^2",
                    "temperature": float(getattr(p0.hidden.base, "TEMPERATURE", 2.0)),
                    "G_R_source": "torch.autograd.grad(total_loss, OmegaCoreLMFast.forward_window readout_states, retain_graph=True)",
                    "G_R_l2_norm": float(torch.linalg.vector_norm(upstream_readout).item()),
                    "G_S_source": "exact zero; inter-window state is detached and no external state loss exists",
                },
                "model": {
                    "implementation": "shared K4 OmegaCoreLMFast production forward",
                    "dtype": "float32",
                    "device": "cpu",
                    "physical_batch": BATCH,
                    "sequence_length": WINDOW_TOKENS,
                    "slots": SLOTS,
                    "dimension": DIMENSION,
                    "eps": EPS,
                    "state_part_weight_layout": {
                        "shape": [SLOTS * DIMENSION, DIMENSION],
                        "strides_elements": [2 * DIMENSION, 1],
                        "storage_offset_elements": DIMENSION,
                        "storage_tensor": "state_part_weight_storage",
                    },
                },
                "runtime": {
                    "torch_version": str(torch.__version__),
                    "numpy_version": str(np.__version__),
                    "intraop_threads": int(torch.get_num_threads()),
                    "interop_threads": int(torch.get_num_interop_threads()),
                    "training_policy": policy,
                    "reference": "R2 _teacher_logits + OmegaCoreLMFast.forward_window + hidden.base.distillation_loss (untimed)",
                    "memory_safety_gate": memory_gate,
                },
                "provenance_verification": "p0.integration.verify_provenance passed before capture",
            }
            source_layouts = {
                "state_part_weight": {
                    "strides_elements": [int(item) for item in model.prelude.weight[:, DIMENSION:].stride()],
                    "storage_offset_elements": int(model.prelude.weight[:, DIMENSION:].storage_offset()),
                    "view_of": "state_part_weight_storage",
                },
                "state_part_weight_storage": {
                    "strides_elements": [int(item) for item in model.prelude.weight.stride()],
                    "storage_offset_elements": int(model.prelude.weight.storage_offset()),
                },
            }
            fixture_name = f"replay_K4_update{update}_window{window}.bin"
            fixtures.append(write_fixture(output_root / "fixtures" / fixture_name, tensors, metadata, source_layouts))

        # Same CE+KL backward, global-norm clipping, and AdamW update as R2.
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM)
        optimizer.step()
        state = next_state.detach().clone()

    report = {
        "schema": "omega-recurrent-backward-replay-capture-report-v1",
        "status": "CAPTURE_PASS",
        "updates_executed": 4,
        "timing_performed": False,
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
        "provenance_verification": "passed",
        "memory_safety_gate": memory_gate,
        "data_reference": {
            "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
            "verified_cache_manifest": True,
        },
    }
    _write_json(output_root / "capture_report.json", report)
    return report


class _ReplayModel:
    """Small differentiable recurrent parameter view for fixture-only replay."""

    def __init__(self, arrays: Mapping[str, np.ndarray], torch: Any, fast_block_type: Any) -> None:
        import torch.nn as nn
        import torch.nn.functional as F

        self.slots = SLOTS
        self.dimension = DIMENSION
        self.rounds = ROUNDS
        self.variant = "shared"
        prelude_storage = torch.from_numpy(np.array(arrays["state_part_weight_storage"], copy=True))
        self.prelude = SimpleNamespace(weight=nn.Parameter(prelude_storage))
        self.prelude_norm_weight = nn.Parameter(torch.from_numpy(np.array(arrays["prelude_norm_weight"], copy=True)))
        self.depth_embedding = SimpleNamespace(
            weight=nn.Parameter(torch.from_numpy(np.array(arrays["depth_embedding_weight"], copy=True)))
        )
        self.gate_logits = nn.Parameter(torch.from_numpy(np.array(arrays["gate_logits"], copy=True)))
        block = fast_block_type(DIMENSION, SLOTS)
        block_targets = {
            "qkv.weight": block.qkv.weight,
            "qkv.bias": block.qkv.bias,
            "out.weight": block.out.weight,
            "out.bias": block.out.bias,
            "fc1.weight": block.fc1.weight,
            "fc1.bias": block.fc1.bias,
            "fc2.weight": block.fc2.weight,
            "fc2.bias": block.fc2.bias,
            "norm_weight": block.norm_weight,
        }
        with torch.no_grad():
            for name, value in (
                ("qkv.weight", arrays["block_qkv_weight"]),
                ("qkv.bias", arrays["block_qkv_bias"]),
                ("out.weight", arrays["block_out_weight"]),
                ("out.bias", arrays["block_out_bias"]),
                ("fc1.weight", arrays["block_fc1_weight"]),
                ("fc1.bias", arrays["block_fc1_bias"]),
                ("fc2.weight", arrays["block_fc2_weight"]),
                ("fc2.bias", arrays["block_fc2_bias"]),
                ("norm_weight", arrays["block_norm_weight"]),
            ):
                block_targets[name].copy_(torch.from_numpy(np.array(value, copy=True)))
        self.blocks = [block]
        self._functional = F

    def _round_constants(self) -> tuple[list[Any], Any]:
        F = self._functional
        gates = self._functional.sigmoid(self.gate_logits)
        block = self.blocks[0]
        biases = [
            block.fc1.bias + F.linear(self.depth_embedding.weight[index], block.fc1.weight)
            for index in range(self.rounds)
        ]
        return biases, gates


def _torch_setup() -> tuple[Any, Any]:
    import torch

    _r2, p0, ce = _training_modules()
    policy = ce.validate_policy()
    if int(torch.get_num_threads()) != 4 or int(torch.get_num_interop_threads()) != 1:
        raise RuntimeError("PyTorch replay checks require intraop=4 and interop=1")
    golden = importlib.import_module("run_omega_native_runtime_p2r0_golden")
    fast_module = importlib.import_module("omega_fast_candidate")
    return torch, (p0, ce, golden, fast_module.FastWorkspaceUpdateBlock, policy)


def torch_check(fixtures: Sequence[Path]) -> dict[str, Any]:
    import torch

    torch, (_p0, _ce, golden, fast_block_type, policy) = _torch_setup()
    results: list[dict[str, Any]] = []
    for fixture_path in fixtures:
        header, arrays, file_sha = read_fixture(fixture_path)
        model = _ReplayModel(arrays, torch, fast_block_type)
        state_view = model.prelude.weight[:, DIMENSION:]
        if tuple(state_view.stride()) != (2 * DIMENSION, 1):
            raise AssertionError("torch fixture replay did not reconstruct original state view stride")
        if not torch.equal(state_view, torch.from_numpy(np.array(arrays["state_part_weight"], copy=True))):
            raise AssertionError("torch fixture replay changed state_part_weight values")

        token_part = torch.from_numpy(np.array(arrays["token_part"], copy=True)).requires_grad_(True)
        previous_state = torch.from_numpy(np.array(arrays["previous_state"], copy=True)).requires_grad_(True)
        next_state, readout_states = golden._recurrent_forward_from_token_part(model, token_part, previous_state)
        expected_next = torch.from_numpy(np.array(arrays["next_state"], copy=True))
        expected_readout = torch.from_numpy(np.array(arrays["readout_states"], copy=True))
        if not torch.equal(next_state, expected_next) or not torch.equal(readout_states, expected_readout):
            raise AssertionError(f"PyTorch replay forward mismatch: {fixture_path.name}")

        upstream_readout = torch.from_numpy(np.array(arrays["G_R"], copy=True))
        upstream_next = torch.from_numpy(np.array(arrays["G_S"], copy=True))
        block = model.blocks[0]
        requested = (
            token_part,
            previous_state,
            model.prelude.weight,
            model.prelude_norm_weight,
            block.qkv.weight,
            block.qkv.bias,
            block.out.weight,
            block.out.bias,
            block.fc1.weight,
            block.fc1.bias,
            block.fc2.weight,
            block.fc2.bias,
            block.norm_weight,
            model.depth_embedding.weight,
            model.gate_logits,
        )
        gradients = torch.autograd.grad(
            (next_state, readout_states), requested, grad_outputs=(upstream_next, upstream_readout)
        )
        actual = (
            gradients[0],
            gradients[1],
            gradients[2][:, DIMENSION:],
            *gradients[3:],
        )
        checks: list[dict[str, Any]] = []
        for name, value in zip(GRADIENT_NAMES, actual):
            expected = torch.from_numpy(np.array(arrays[name], copy=True))
            exact = bool(torch.equal(value, expected))
            max_abs = float((value.detach() - expected).abs().max().item()) if value.numel() else 0.0
            checks.append({"name": name, "exact": exact, "max_abs_error": max_abs})
            if not exact:
                raise AssertionError(f"PyTorch replay gradient mismatch for {name}: {max_abs}")
        results.append({
            "fixture": fixture_path.name,
            "fixture_sha256": file_sha,
            "update": int(header["metadata"]["update_index"]),
            "window": int(header["metadata"]["window_index"]),
            "forward_exact": True,
            "gradient_count": len(checks),
            "gradients": checks,
        })
    return {
        "schema": "omega-recurrent-backward-replay-torch-report-v1",
        "status": "TORCH_CORRECTNESS_PASS",
        "timing_performed": False,
        "torch_version": str(torch.__version__),
        "thread_policy": policy,
        "fixtures": results,
    }


_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_SIZE_T_PTR = ctypes.POINTER(ctypes.c_size_t)
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


class _Config(ctypes.Structure):
    _fields_ = [
        ("sequence_length", ctypes.c_size_t),
        ("batch", ctypes.c_size_t),
        ("slots", ctypes.c_size_t),
        ("dimension", ctypes.c_size_t),
        ("rounds", ctypes.c_size_t),
        ("training", ctypes.c_int),
        ("instrumentation", ctypes.c_int),
    ]


class _MatrixView(ctypes.Structure):
    _fields_ = [
        ("data", _FLOAT_PTR),
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
        ("row_stride", ctypes.c_ssize_t),
    ]


class _Params(ctypes.Structure):
    _fields_ = [("state_part_weight", _MatrixView)] + [
        (name, _FLOAT_PTR)
        for name in (
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
            "depth_embedding",
            "gate_logits",
        )
    ]


class _Grads(ctypes.Structure):
    _fields_ = (
        [(name, _FLOAT_PTR) for name in GRADIENT_NAMES]
        + [("sum_abs_" + name, _FLOAT_PTR) for name in GRADIENT_NAMES]
        + [("count_" + name, _SIZE_T_PTR) for name in GRADIENT_NAMES]
        + [("fp64_d_depth_embedding_weight", _DOUBLE_PTR), ("depth_max_level", _SIZE_T_PTR)]
    )


def _float_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(_FLOAT_PTR)


def _native_gradient_result(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    sum_abs: np.ndarray,
    counts: np.ndarray,
    fp64_depth: np.ndarray,
    depth_max: np.ndarray,
) -> dict[str, Any]:
    atol = 1.0e-5
    rtol = 1.0e-4
    unit_roundoff = 5.9604644775390625e-8
    actual64 = actual.astype(np.float64, copy=False).reshape(-1)
    expected64 = expected.astype(np.float64, copy=False).reshape(-1)
    sum64 = sum_abs.astype(np.float64, copy=False).reshape(-1)
    count64 = counts.astype(np.uint64, copy=False).reshape(-1)
    error = np.abs(actual64 - expected64)
    normal = np.isfinite(actual64) & (error <= atol + rtol * np.abs(expected64))
    normal_fail_indices = np.flatnonzero(~normal)
    fallback_pass_count = 0
    fail_count = 0
    for index in normal_fail_indices:
        contribution_sum = float(sum64[index])
        contribution_count = int(count64[index])
        difference = float(error[index])
        if name == "d_depth_embedding_weight":
            reference_oracle_error = abs(float(expected64[index]) - float(fp64_depth.reshape(-1)[index]))
            native_oracle_error = abs(float(actual64[index]) - float(fp64_depth.reshape(-1)[index]))
            depth = int(depth_max.reshape(-1)[index])
            depth_times_unit = depth * unit_roundoff
            gamma_h = depth_times_unit / (1.0 - depth_times_unit) if depth_times_unit < 1.0 else math.inf
            fallback_pass = (
                native_oracle_error <= gamma_h * contribution_sum
                and native_oracle_error < reference_oracle_error
            )
        else:
            count_times_epsilon = contribution_count * unit_roundoff
            gamma_n = count_times_epsilon / (1.0 - count_times_epsilon) if count_times_epsilon < 1.0 else math.inf
            eta = difference / max(contribution_sum, 1.0e-30)
            fallback_pass = contribution_count > 0 and count_times_epsilon < 1.0 and eta <= 2.0 * gamma_n
        if fallback_pass:
            fallback_pass_count += 1
        else:
            fail_count += 1
    finite = bool(np.isfinite(actual64).all())
    return {
        "name": name,
        "status": "PASS" if finite and fail_count == 0 else "FAIL",
        "max_abs_error": float(error.max()) if error.size else 0.0,
        "max_relative_error": float((error / np.maximum(np.abs(expected64), 1.0e-30)).max()) if error.size else 0.0,
        "normal_gate_fail_count": int(normal_fail_indices.size),
        "cancellation_fallback_pass_count": fallback_pass_count,
        "fail_count": fail_count,
        "finite": finite,
        "gate": "abs_error <= 1e-5 + 1e-4*abs(reference), else existing accumulation/oracle adjudication",
    }


def native_check(
    fixtures: Sequence[Path],
    dll_path: Path,
    expected_dll_sha256: str = ACCEPTED_DLL_SHA256,
) -> dict[str, Any]:
    resolved_dll = dll_path.resolve()
    dll_sha = _sha256_bytes(resolved_dll.read_bytes())
    if dll_sha != expected_dll_sha256:
        raise ValueError(f"DLL SHA-256 does not match expected native candidate: {dll_sha}")
    library = ctypes.CDLL(str(resolved_dll))
    required_exports = (
        "omega_runtime_create",
        "omega_runtime_destroy",
        "omega_runtime_workspace_bytes",
        "omega_runtime_forward",
        "omega_runtime_backward",
    )
    missing = [name for name in required_exports if not hasattr(library, name)]
    if missing:
        raise RuntimeError(f"accepted DLL lacks persistent runtime ABI exports: {missing}")
    library.omega_runtime_create.argtypes = [ctypes.c_size_t]
    library.omega_runtime_create.restype = ctypes.c_void_p
    library.omega_runtime_destroy.argtypes = [ctypes.c_void_p]
    library.omega_runtime_destroy.restype = None
    library.omega_runtime_workspace_bytes.argtypes = [ctypes.c_void_p, _Config]
    library.omega_runtime_workspace_bytes.restype = ctypes.c_size_t
    library.omega_runtime_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Config),
        ctypes.POINTER(_Params),
        _FLOAT_PTR,
        _FLOAT_PTR,
        _FLOAT_PTR,
        _FLOAT_PTR,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.omega_runtime_forward.restype = ctypes.c_int
    library.omega_runtime_backward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Config),
        ctypes.POINTER(_Params),
        _FLOAT_PTR,
        _FLOAT_PTR,
        _FLOAT_PTR,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(_Grads),
    ]
    library.omega_runtime_backward.restype = ctypes.c_int

    runtime = library.omega_runtime_create(4)
    if not runtime:
        raise RuntimeError("omega_runtime_create(4) failed")
    fixture_results: list[dict[str, Any]] = []
    try:
        for fixture_path in fixtures:
            header, arrays, file_sha = read_fixture(fixture_path)
            metadata = header["metadata"]
            if int(metadata["K"]) != ROUNDS or tuple(arrays["token_part"].shape) != (BATCH, WINDOW_TOKENS, SLOTS * DIMENSION):
                raise ValueError(f"unexpected native fixture shape/config: {fixture_path}")
            state_storage = arrays["state_part_weight_storage"]
            state_matrix = state_storage[:, DIMENSION:]
            if state_matrix.strides != (2 * DIMENSION * 4, 4):
                raise ValueError(f"native state matrix view lost original row stride: {state_matrix.strides}")
            if not np.array_equal(state_matrix, arrays["state_part_weight"]):
                raise ValueError("native state matrix storage view differs from saved logical parameter")

            config = _Config(WINDOW_TOKENS, BATCH, SLOTS, DIMENSION, ROUNDS, 1, 1)
            matrix_view = _MatrixView(
                _float_ptr(state_matrix),
                SLOTS * DIMENSION,
                DIMENSION,
                2 * DIMENSION,
            )
            params = _Params(
                matrix_view,
                *(
                    _float_ptr(arrays[name])
                    for name in (
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
                    )
                ),
            )
            workspace_bytes = int(library.omega_runtime_workspace_bytes(runtime, config))
            if workspace_bytes <= 0:
                raise RuntimeError("native runtime returned invalid workspace size")
            workspace = ctypes.create_string_buffer(workspace_bytes)
            next_state = np.empty_like(arrays["previous_state"])
            readout_states = np.empty_like(arrays["readout_states"])
            status = int(
                library.omega_runtime_forward(
                    runtime,
                    ctypes.byref(config),
                    ctypes.byref(params),
                    _float_ptr(arrays["token_part"]),
                    _float_ptr(arrays["previous_state"]),
                    _float_ptr(next_state),
                    _float_ptr(readout_states),
                    ctypes.cast(workspace, ctypes.c_void_p),
                    workspace_bytes,
                )
            )
            if status != 0:
                raise RuntimeError(f"native runtime forward failed with status {status}")

            gradients = {name: np.empty_like(arrays[name]) for name in GRADIENT_NAMES}
            sum_abs = {name: np.empty_like(arrays[name]) for name in GRADIENT_NAMES}
            contribution_counts = {
                name: np.empty(arrays[name].shape, dtype=np.uintp) for name in GRADIENT_NAMES
            }
            fp64_depth = np.empty_like(arrays["d_depth_embedding_weight"], dtype=np.float64)
            depth_max = np.empty_like(arrays["d_depth_embedding_weight"], dtype=np.uintp)
            grads = _Grads(
                *(_float_ptr(gradients[name]) for name in GRADIENT_NAMES),
                *(_float_ptr(sum_abs[name]) for name in GRADIENT_NAMES),
                *(value.ctypes.data_as(_SIZE_T_PTR) for value in (contribution_counts[name] for name in GRADIENT_NAMES)),
                fp64_depth.ctypes.data_as(_DOUBLE_PTR),
                depth_max.ctypes.data_as(_SIZE_T_PTR),
            )
            backward_status = int(
                library.omega_runtime_backward(
                    runtime,
                    ctypes.byref(config),
                    ctypes.byref(params),
                    _float_ptr(arrays["token_part"]),
                    _float_ptr(arrays["G_R"]),
                    _float_ptr(arrays["G_S"]),
                    ctypes.cast(workspace, ctypes.c_void_p),
                    workspace_bytes,
                    ctypes.byref(grads),
                )
            )
            if backward_status != 0:
                raise RuntimeError(f"native runtime backward failed with status {backward_status}")

            forward_checks = []
            for name, actual in (("next_state", next_state), ("readout_states", readout_states)):
                expected = arrays[name]
                difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                max_abs = float(difference.max()) if difference.size else 0.0
                failed = int(np.count_nonzero(~np.isfinite(actual) | (difference > 1.0e-5)))
                forward_checks.append({
                    "name": name,
                    "status": "PASS" if failed == 0 else "FAIL",
                    "max_abs_error": max_abs,
                    "atol": 1.0e-5,
                    "failed_elements": failed,
                })
            gradient_checks = [
                _native_gradient_result(
                    name,
                    gradients[name],
                    arrays[name],
                    sum_abs[name],
                    contribution_counts[name],
                    fp64_depth,
                    depth_max,
                )
                for name in GRADIENT_NAMES
            ]
            passed = all(item["status"] == "PASS" for item in (*forward_checks, *gradient_checks))
            fixture_results.append({
                "fixture": fixture_path.name,
                "fixture_sha256": file_sha,
                "update": int(metadata["update_index"]),
                "window": int(metadata["window_index"]),
                "native_runtime_threads": 4,
                "forward_calls": 1,
                "backward_calls": 1,
                "status": "PASS" if passed else "FAIL",
                "forward": forward_checks,
                "gradients": gradient_checks,
            })
            if not passed:
                raise AssertionError(f"native correctness failed for {fixture_path.name}")
    finally:
        library.omega_runtime_destroy(runtime)
    return {
        "schema": "omega-recurrent-backward-replay-native-report-v1",
        "status": "NATIVE_CORRECTNESS_PASS",
        "timing_performed": False,
        "native_library_sha256": dll_sha,
        "native_library_bytes": int(resolved_dll.stat().st_size),
        "native_runtime_threads": 4,
        "torch_loaded": False,
        "fixtures": fixture_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "torch-check", "native-check"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixture", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-file", type=Path)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--expected-dll-sha256", default=ACCEPTED_DLL_SHA256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.mode == "capture":
        report = capture_fixtures(args.manifest, args.cache_file, args.output_root)
    elif not args.fixture:
        parser.error("--fixture is required for check modes")
    elif args.mode == "torch-check":
        report = torch_check(args.fixture)
    else:
        if args.dll is None:
            parser.error("--dll is required for native-check")
        report = native_check(args.fixture, args.dll, args.expected_dll_sha256)

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        _write_json(args.output, report)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
