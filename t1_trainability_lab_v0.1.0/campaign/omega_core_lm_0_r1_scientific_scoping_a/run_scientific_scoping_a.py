"""OMEGA R1 scientific scoping A runner.

Default execution is deliberately blocked. ``--smoke`` is synthetic and CPU-only;
the full campaign requires both ``--full`` and ``--confirm-smoke``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
F_CANDIDATE_DIR = ROOT / "campaign" / "omega_core_lm_0_r1_cpu_fastpath_validation"
F_CANDIDATE_PATH = F_CANDIDATE_DIR / "omega_fast_candidate.py"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(F_CANDIDATE_DIR))

from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_VOCAB,
    OmegaCoreLM0R1Technical,
    distillation_loss,
    is_level_one_header,
    parameter_hash,
    reconstruct_documents,
    sha256_bytes,
    sha256_text,
    teacher_window_logits,
)
from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402


CAMPAIGN_ID = "OMEGA-CORE-LM-0-R1-SCIENTIFIC-SCOPING-A"
VARIANTS = ("shared_K1", "shared_K4")
SEEDS = (20260913, 20260914)
SCOPE_C_SEED = 20260915
FULL_UPDATES = 2000
FULL_PAIRS = 1000
FULL_BOUNDARIES = (0, 500, 1000, 1500, 2000)
SCOPE_B_UPDATES = 5000
SCOPE_B_PAIRS = 2500
SCOPE_B_CONTINUE_MIN_UPDATE = 2500
SCOPE_B_BOUNDARIES = (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
SMOKE_UPDATES = 4
SMOKE_PAIRS = 2
SMOKE_BOUNDARIES = (0, 2, 4)
PHYSICAL_BATCH = 8
EFFECTIVE_BATCH = 8
MICROBATCHES = 1
WINDOW_TOKENS = 256
RETAINED_TOKENS = 513
TEMPERATURE = 2.0
BASE_LR = 3e-4
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
VALIDATION_DOCUMENTS = 8
SMOKE_VALIDATION_DOCUMENTS = 2
MIN_AVAILABLE_BYTES = 1 << 30
CPU_INTRAOP_THREADS = 4
CPU_INTEROP_THREADS = 1


class MemorySafetyError(RuntimeError):
    """Hard stop raised before work can exceed the memory safety floor."""


class NumericalSafetyError(RuntimeError):
    """Hard stop raised before backward or optimizer step on non-finite values."""


_CPU_RUNTIME_CONFIGURED = False


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def token_hash(tokens: Iterable[int]) -> str:
    return sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in tokens))


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def implementation_identity() -> dict[str, Any]:
    return {
        "implementation": "F",
        "class": "omega_fast_candidate.OmegaCoreLMFast",
        "candidate_source_path": F_CANDIDATE_PATH.relative_to(ROOT).as_posix(),
        "candidate_source_sha256": file_hash(F_CANDIDATE_PATH),
        "dependency": {
            "class": "run_omega_core_lm_0_r1_training_technical_preflight.OmegaCoreLM0R1Technical",
            "source_path": (SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py").relative_to(ROOT).as_posix(),
        },
    }


def source_hashes() -> dict[str, str]:
    current = Path(__file__).resolve()
    source = SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    return {
        "runner": file_hash(current),
        "current_r1_source": file_hash(source),
        "f_candidate_source": file_hash(F_CANDIDATE_PATH),
    }


def configure_cpu_runtime() -> None:
    global _CPU_RUNTIME_CONFIGURED
    if _CPU_RUNTIME_CONFIGURED:
        return
    try:
        torch.set_num_threads(CPU_INTRAOP_THREADS)
        torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    except RuntimeError as exc:
        raise RuntimeError("cannot set CPU runtime to 4 intraop/1 interop before model work") from exc
    _CPU_RUNTIME_CONFIGURED = True


def memory_guard(stage: str, samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    sample = {"stage": stage, "available_bytes": available, "required_bytes": MIN_AVAILABLE_BYTES}
    if samples is not None:
        samples.append(sample)
    if available < MIN_AVAILABLE_BYTES:
        raise MemorySafetyError(
            f"hard stop before {stage}: available memory {available} bytes below {MIN_AVAILABLE_BYTES} bytes"
        )
    return sample


def _minimum_available_memory(samples: list[dict[str, Any]]) -> int | None:
    values = [int(sample["available_bytes"]) for sample in samples]
    return min(values) if values else None


def _finite_loss(loss: torch.Tensor, stage: str) -> None:
    if not bool(torch.isfinite(loss).all().item()):
        raise NumericalSafetyError(f"non-finite loss at {stage}; optimizer step aborted")


def _finite_gradients(model: torch.nn.Module, stage: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise NumericalSafetyError(f"non-finite gradient for {name} at {stage}; optimizer step aborted")


def validate_policy() -> dict[str, Any]:
    configure_cpu_runtime()
    policy = {
        "device": "cpu",
        "dtype": "float32",
        "execution": "eager",
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "microbatches": MICROBATCHES,
        "optimizer": "AdamW",
        "teacher_frozen": True,
        "intraop_threads": CPU_INTRAOP_THREADS,
        "interop_threads": CPU_INTEROP_THREADS,
    }
    if policy["device"] != "cpu" or policy["dtype"] != "float32" or policy["execution"] != "eager":
        raise AssertionError("scoping A policy drift")
    if (PHYSICAL_BATCH, EFFECTIVE_BATCH, MICROBATCHES) != (8, 8, 1):
        raise AssertionError("batch policy drift")
    if torch.get_num_threads() != CPU_INTRAOP_THREADS:
        raise AssertionError("intraop thread policy drift")
    return policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def is_eligible_document(document: dict[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    text = str(document["text"])
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    if len(tokens) < RETAINED_TOKENS:
        return None
    retained = tokens[:RETAINED_TOKENS]
    return {
        "document_index": int(document["document_index"]),
        "row_range": list(document["row_range"]),
        "header": str(document["header"]),
        "token_count": len(tokens),
        "selected_token_count": len(retained),
        "full_text_sha256": sha256_text(text),
        "retained_513_token_sha256": token_hash(retained),
        "tokens": retained,
    }


def collect_all_eligible_documents(dataset: Any, tokenizer: Any) -> list[dict[str, Any]]:
    """Collect every eligible reconstructed document in stable source order."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document_index, document in enumerate(reconstruct_documents(dataset)):
        document = {**document, "document_index": document_index}
        eligible = is_eligible_document(document, tokenizer)
        if eligible is None:
            continue
        key = (eligible["full_text_sha256"], eligible["retained_513_token_sha256"])
        if key in seen:
            continue
        seen.add(key)
        result.append(eligible)
    return result


def synthetic_documents(count: int, vocab_size: int) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for index in range(count):
        tokens = [((index + 1) * 3 + position) % vocab_size for position in range(RETAINED_TOKENS)]
        text = f"synthetic-{index}-" + ",".join(str(token) for token in tokens)
        documents.append(
            {
                "document_index": index,
                "row_range": [index, index + 1],
                "header": f"= Synthetic {index} =",
                "token_count": RETAINED_TOKENS,
                "selected_token_count": RETAINED_TOKENS,
                "full_text_sha256": sha256_text(text),
                "retained_513_token_sha256": token_hash(tokens),
                "tokens": tokens,
            }
        )
    return documents


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "tokens"}


def build_pair_manifest(documents: list[dict[str, Any]], pair_count: int) -> dict[str, Any]:
    if not documents:
        raise ValueError("cannot build pair manifest without eligible documents")
    pairs: list[dict[str, Any]] = []
    total_wraps = 0
    for pair in range(pair_count):
        raw_start = pair * PHYSICAL_BATCH
        indices = [(raw_start + offset) % len(documents) for offset in range(PHYSICAL_BATCH)]
        wraps = ((raw_start + PHYSICAL_BATCH - 1) // len(documents)) - (raw_start // len(documents))
        total_wraps += wraps
        pairs.append(
            {
                "pair": pair,
                "start_offset": raw_start % len(documents),
                "document_indices": indices,
                "document_keys": [
                    [documents[index]["full_text_sha256"], documents[index]["retained_513_token_sha256"]]
                    for index in indices
                ],
                "window_0_and_window_1_share_documents": True,
                "wraps": wraps,
            }
        )
    return {
        "pair_count": pair_count,
        "documents_available": len(documents),
        "documents_consumed": pair_count * PHYSICAL_BATCH,
        "cycle_count": (pair_count * PHYSICAL_BATCH) // len(documents),
        "wraps": total_wraps,
        "pairs": pairs,
    }


def build_manifest(
    documents: list[dict[str, Any]],
    *,
    split_name: str,
    pair_count: int | None = None,
    excluded_keys: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = excluded_keys or set()
    chosen = [
        document
        for document in documents
        if (document["full_text_sha256"], document["retained_513_token_sha256"]) not in excluded
    ]
    if not chosen:
        raise RuntimeError(f"no eligible {split_name} documents remain")
    manifest: dict[str, Any] = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-document-manifest-v1",
        "split": split_name,
        "reconstruction": {
            "header_rule": "level-one header is '= Title ='; deeper headings do not start documents",
            "implementation": "current R1 reconstruct_documents",
        },
        "eligibility": "all reconstructed documents with >=513 GPT-2 tokens; retain first 513",
        "dedupe_key": ["full_text_sha256", "retained_513_token_sha256"],
        "stable_order": "dataset row order after reconstruction and first-occurrence dedupe",
        "documents": [public_document(document) for document in chosen],
        "document_count": len(chosen),
    }
    if pair_count is not None:
        manifest["cyclic_pairs"] = build_pair_manifest(chosen, pair_count)
    manifest["manifest_sha256"] = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return chosen, manifest


def _manifest_corpus_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in {"cyclic_pairs", "manifest_sha256"}}


def _validated_pair_list(manifest: dict[str, Any], label: str, expected_count: int) -> list[dict[str, Any]]:
    pair_manifest = manifest.get("cyclic_pairs")
    if not isinstance(pair_manifest, dict):
        raise ValueError(f"manifest extension verification failed: {label} cyclic_pairs is missing or malformed")
    if pair_manifest.get("pair_count") != expected_count:
        raise ValueError(
            f"manifest extension verification failed: {label} pair count is {pair_manifest.get('pair_count')!r}, expected {expected_count}"
        )
    pairs = pair_manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != expected_count:
        raise ValueError(f"manifest extension verification failed: {label} pairs list is malformed")
    required = {"pair", "start_offset", "document_indices", "document_keys", "window_0_and_window_1_share_documents", "wraps"}
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or not required.issubset(pair):
            raise ValueError(f"manifest extension verification failed: malformed {label} pair prefix at index {index}")
        if pair["pair"] != index:
            raise ValueError(f"manifest extension verification failed: malformed {label} pair prefix at index {index}")
        if not isinstance(pair["document_indices"], list) or len(pair["document_indices"]) != PHYSICAL_BATCH:
            raise ValueError(f"manifest extension verification failed: malformed {label} pair prefix at index {index}")
        if not isinstance(pair["document_keys"], list) or len(pair["document_keys"]) != PHYSICAL_BATCH:
            raise ValueError(f"manifest extension verification failed: malformed {label} pair prefix at index {index}")
    return pairs


def verify_manifest_extension(
    checkpoint_payload: dict[str, Any],
    old_train_manifest: dict[str, Any],
    new_train_manifest: dict[str, Any],
    *,
    expected_old_pair_count: int = FULL_PAIRS,
    expected_new_pair_count: int = SCOPE_B_PAIRS,
) -> dict[str, Any]:
    """Prove that new train manifest is exact cyclic extension of checkpoint input."""
    old_hash = old_train_manifest.get("manifest_sha256")
    new_hash = new_train_manifest.get("manifest_sha256")
    if not isinstance(old_hash, str) or not isinstance(new_hash, str):
        raise ValueError("manifest extension verification failed: manifest hash is missing")
    old_recomputed_hash = canonical_hash(_manifest_corpus_metadata(old_train_manifest) | {"cyclic_pairs": old_train_manifest.get("cyclic_pairs")})
    new_recomputed_hash = canonical_hash(_manifest_corpus_metadata(new_train_manifest) | {"cyclic_pairs": new_train_manifest.get("cyclic_pairs")})
    if old_hash != old_recomputed_hash:
        raise ValueError("manifest extension verification failed: old train manifest hash is invalid")
    if new_hash != new_recomputed_hash:
        raise ValueError("manifest extension verification failed: new train manifest hash is invalid")
    checkpoint_data_hashes = checkpoint_payload.get("data_hashes")
    if not isinstance(checkpoint_data_hashes, dict):
        raise ValueError("manifest extension verification failed: checkpoint data_hashes is missing or malformed")
    checkpoint_hash = checkpoint_data_hashes.get("train_manifest")
    if checkpoint_hash != old_hash:
        raise ValueError("manifest extension verification failed: checkpoint old train_manifest hash mismatch")
    old_metadata = _manifest_corpus_metadata(old_train_manifest)
    new_metadata = _manifest_corpus_metadata(new_train_manifest)
    if old_metadata != new_metadata:
        raise ValueError("manifest extension verification failed: corpus/document metadata changed")
    old_pairs = _validated_pair_list(old_train_manifest, "old", expected_old_pair_count)
    new_pairs = _validated_pair_list(new_train_manifest, "new", expected_new_pair_count)
    old_pair_bytes = canonical_bytes(old_pairs)
    new_prefix_bytes = canonical_bytes(new_pairs[:expected_old_pair_count])
    old_pair_hash = sha256_bytes(old_pair_bytes)
    new_prefix_hash = sha256_bytes(new_prefix_bytes)
    if old_pair_bytes != new_prefix_bytes:
        raise ValueError("manifest extension verification failed: new cyclic pair prefix is not byte-identical")
    return {
        "verified": True,
        "checkpoint_old_train_manifest_sha256": checkpoint_hash,
        "old_manifest_sha256": old_hash,
        "new_manifest_sha256": new_hash,
        "old_pair_count": expected_old_pair_count,
        "new_pair_count": expected_new_pair_count,
        "corpus_metadata_unchanged": True,
        "corpus_metadata_sha256": canonical_hash(old_metadata),
        "prefix": {
            "old_pair_bytes_sha256": old_pair_hash,
            "new_prefix_bytes_sha256": new_prefix_hash,
            "old_byte_length": len(old_pair_bytes),
            "new_prefix_byte_length": len(new_prefix_bytes),
            "byte_identical": True,
        },
    }


def schedule(updates: int) -> list[dict[str, int]]:
    if updates < 0:
        raise ValueError("updates must be non-negative")
    return [{"update": update, "window": update % 2, "pair": update // 2} for update in range(updates)]


def checkpoint_boundaries(total_updates: int, interval: int) -> tuple[int, ...]:
    if total_updates < 0 or interval <= 0 or total_updates % interval:
        raise ValueError("total updates must be a non-negative multiple of checkpoint interval")
    return tuple(range(0, total_updates + 1, interval))


def validate_resume_boundary(update: int, interval: int) -> None:
    if update < 0 or update % interval:
        raise ValueError(f"resume checkpoint must be on a {interval}-update boundary")


def classify_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, float]] = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), {})[str(run["variant"])] = float(run["validation_curve"][-1]["nll"])
    deltas = [
        {"seed": seed, "delta_k1_minus_k4": values["shared_K1"] - values["shared_K4"]}
        for seed, values in sorted(by_seed.items())
    ]
    delta_values = [item["delta_k1_minus_k4"] for item in deltas]
    mean_delta = sum(delta_values) / len(delta_values) if delta_values else 0.0
    if delta_values and all(value > 0 for value in delta_values) and mean_delta > 0:
        classification = "PROMISING"
    elif delta_values and all(value <= 0 for value in delta_values) and mean_delta <= 0:
        classification = "NO_SIGNAL"
    else:
        classification = "MIXED"
    return {"per_seed_delta": deltas, "mean_delta": mean_delta, "classification": classification}


class TinyTeacher(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        table = torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(vocab_size, vocab_size) / 100.0
        self.table = torch.nn.Parameter(table, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return type("TinyTeacherOutput", (), {"logits": self.table[input_ids % self.table.shape[0]]})()


class AppendOnlyLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        segment = int(event.get("execution_segment", 0))
        if segment > 0 and not event.get("parent_checkpoint_hash"):
            raise ValueError("resumed execution segment requires parent_checkpoint_hash")
        encoded = canonical_bytes(event)
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def _state_hashes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, str]:
    return {
        "model_parameter_hash": parameter_hash(model),
        "optimizer_state_hash": sha256_bytes(pickle.dumps(optimizer.state_dict(), protocol=4)),
    }


def _rng_state_hashes(rng_states: dict[str, Any]) -> dict[str, str]:
    return {
        "torch_rng_hash": sha256_bytes(pickle.dumps(rng_states["torch"], protocol=4)),
        "python_rng_hash": sha256_bytes(pickle.dumps(rng_states["python"], protocol=4)),
    }


def _state_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, type(right)) and len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _config_payload(*, variant: str, seed: int, smoke: bool, dimensions: tuple[int, int], integration_smoke: bool = False) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "variant": variant,
        "seed": seed,
        "device": "cpu",
        "dtype": "float32",
        "execution": "eager",
        "implementation_identity": implementation_identity(),
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "microbatches": MICROBATCHES,
        "rounds": 1 if variant == "shared_K1" else 4,
        "model_variant": "shared",
        "dimension": dimensions[0],
        "slots": dimensions[1],
        "loss": "current R1 distillation_loss: 0.5*CE + 0.5*T^2*KL, T=2",
        "optimizer": {"type": "AdamW", "lr": BASE_LR, "betas": list(ADAMW_BETAS), "eps": ADAMW_EPS, "weight_decay": WEIGHT_DECAY, "clip_norm": CLIP_NORM},
        "teacher": {"id": MODEL_ID if not smoke else "synthetic-tiny-teacher", "revision": MODEL_REVISION if not smoke else "synthetic-v1"},
        "data": {"dataset": DATASET_ID if not smoke else "synthetic-fixture", "config": DATASET_CONFIG if not smoke else None, "revision": DATASET_REVISION if not smoke else None},
        "smoke_only": smoke,
        "integration_smoke": integration_smoke,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_hashed_json(path: Path, value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned[field] = "__SELF_HASH__"
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    written = dict(value)
    written[field] = digest
    _write_json(path, written)
    return digest


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    data_position: dict[str, int],
    run_id: str,
    config: dict[str, Any],
    identity_hash: str,
    data_hashes: dict[str, str],
    teacher_hash: str,
    execution_segment: int,
) -> dict[str, Any]:
    return {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "data_position": data_position,
        "rng_states": {"torch": torch.get_rng_state(), "python": random.getstate()},
        "run_id": run_id,
        "seed": config["seed"],
        "execution_segment": execution_segment,
        "identity_hash": identity_hash,
        "config_hash": canonical_hash(config),
        "data_hashes": data_hashes,
        "teacher_hash": teacher_hash,
        "implementation_identity": implementation_identity(),
        "config": config,
        "provenance": {
            "source_hashes": source_hashes(),
            "implementation_identity": implementation_identity(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_hash(path)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be an object")
    return payload, file_hash(path)


def _validate_f_checkpoint(payload: dict[str, Any]) -> None:
    expected = implementation_identity()
    if payload.get("implementation_identity") != expected:
        raise ValueError("resume checkpoint implementation identity is not F/omega_fast_candidate")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("implementation_identity") != expected:
        raise ValueError("resume checkpoint provenance does not identify F/omega_fast_candidate")


def forward_window_adapter(model: torch.nn.Module, input_ids: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = model.forward_window(input_ids, state)
    if not isinstance(result, tuple) or len(result) < 2:
        raise TypeError("forward_window must return state and logits")
    return result[0], result[1]


def _distillation_loss_from_trace(
    model: OmegaCoreLMFast,
    trace: dict[str, torch.Tensor],
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    projected = model.project(trace["readout_states"]).reshape(-1, model.dimension)
    flat_teacher = teacher_logits.reshape(-1, model.vocab_size)
    flat_targets = targets.reshape(-1)
    flat_weights = valid_mask.reshape(-1).to(projected.dtype)
    ce_total = projected.new_zeros(())
    kl_total = projected.new_zeros(())
    for start in range(0, projected.shape[0], 512):
        stop = start + 512
        student_logits = model.logits_from_projected(projected[start:stop])
        ce_total = ce_total + (F.cross_entropy(student_logits, flat_targets[start:stop], reduction="none") * flat_weights[start:stop]).sum()
        student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(flat_teacher[start:stop] / TEMPERATURE, dim=-1)
        kl_total = kl_total + (F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * TEMPERATURE**2 * flat_weights[start:stop]).sum()
    denominator = flat_weights.sum().clamp_min(1.0)
    return 0.5 * (ce_total / denominator) + 0.5 * (kl_total / denominator)


def make_f_model(*, vocab_size: int, dimensions: tuple[int, int], variant: str, memory_samples: list[dict[str, Any]] | None = None) -> OmegaCoreLMFast:
    memory_guard("reference_model_initialization", memory_samples)
    reference = OmegaCoreLM0R1Technical(
        vocab_size=vocab_size,
        dimension=dimensions[0],
        slots=dimensions[1],
        rounds=1 if variant == "shared_K1" else 4,
        variant="shared",
    ).to(dtype=torch.float32)
    try:
        memory_guard("f_candidate_conversion", memory_samples)
        model = OmegaCoreLMFast.from_reference(reference).to(dtype=torch.float32)
    finally:
        del reference
        gc.collect()
    memory_guard("f_candidate_ready", memory_samples)
    return model


def _batch_loss(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    documents: list[dict[str, Any]],
    window: int,
    previous_states: list[torch.Tensor] | None,
    memory_samples: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    configure_cpu_runtime()
    memory_guard("source_assignment", memory_samples)
    source = torch.tensor([document["tokens"] for document in documents], dtype=torch.long)
    input_start = window * WINDOW_TOKENS
    input_ids = source[:, input_start : input_start + WINDOW_TOKENS]
    targets = source[:, input_start + 1 : input_start + WINDOW_TOKENS + 1]
    if window == 0:
        state = model.initial_state(len(documents), device=torch.device("cpu"))
    else:
        if previous_states is None:
            raise ValueError("window 1 requires previous detached window 0 states")
        state = torch.cat(previous_states, dim=0)
    memory_guard("student_forward", memory_samples)
    forward_result = model.forward_window(input_ids, state)
    if not isinstance(forward_result, tuple) or len(forward_result) < 2:
        raise TypeError("forward_window must return state and logits")
    next_state, logits = forward_result[0], forward_result[1]
    trace = forward_result[2] if len(forward_result) >= 3 else None
    if trace is not None and not isinstance(trace, dict):
        raise TypeError("forward_window trace must be a mapping")
    if trace is not None:
        del logits
    memory_guard("teacher_forward", memory_samples)
    teacher_logits = teacher_window_logits(teacher, source, window)
    mask = torch.ones_like(targets, dtype=torch.bool)
    if trace is not None:
        loss = _distillation_loss_from_trace(model, trace, teacher_logits[:, :WINDOW_TOKENS], targets, mask)
    else:
        loss = distillation_loss(logits, teacher_logits[:, :WINDOW_TOKENS], targets, mask)["total"]
    next_states = [next_state[index : index + 1].detach() for index in range(len(documents))]
    return loss, next_states, int(mask.sum().item())


def evaluate_validation(model: torch.nn.Module, documents: list[dict[str, Any]], memory_samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for document in documents:
            source = torch.tensor(document["tokens"], dtype=torch.long).unsqueeze(0)
            state = model.initial_state(1, device=torch.device("cpu"))
            for window in (0, 1):
                memory_guard("validation_forward", memory_samples)
                start = window * WINDOW_TOKENS
                inputs = source[:, start : start + WINDOW_TOKENS]
                targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
                state, logits = forward_window_adapter(model, inputs, state.detach() if window else state)
                values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
                total_nll += float(values.sum().item())
                total_tokens += int(values.numel())
    nll = total_nll / total_tokens
    return {"nll": nll, "tokens": total_tokens, "finite": math.isfinite(nll)}


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_single(
    *,
    run_dir: Path,
    run_id: str,
    seed: int,
    variant: str,
    train_documents: list[dict[str, Any]],
    validation_documents: list[dict[str, Any]],
    train_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
    teacher: torch.nn.Module,
    total_updates: int,
    checkpoint_interval: int,
    smoke: bool,
    dimensions: tuple[int, int],
    resume_checkpoint: Path | None = None,
    old_train_manifest: dict[str, Any] | None = None,
    integration_smoke: bool = False,
    max_updates: int | None = None,
) -> dict[str, Any]:
    validate_policy()
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if max_updates is not None and (max_updates < 0 or max_updates > total_updates):
        raise ValueError("max_updates must be between zero and total_updates")
    if old_train_manifest is not None and resume_checkpoint is None:
        raise ValueError("old_train_manifest is only valid for an explicit resume")
    boundaries = checkpoint_boundaries(total_updates, checkpoint_interval)
    config = _config_payload(variant=variant, seed=seed, smoke=smoke, dimensions=dimensions, integration_smoke=integration_smoke)
    identity_hash = canonical_hash({"campaign_id": CAMPAIGN_ID, "run_id": run_id, "variant": variant, "seed": seed, "implementation_identity": implementation_identity()})
    data_hashes = {"train_manifest": train_manifest["manifest_sha256"], "validation_manifest": validation_manifest["manifest_sha256"]}
    teacher_hash = canonical_hash(config["teacher"])
    ledger_path = run_dir / "ledger.jsonl"
    curve_path = run_dir / "validation_curve.json"
    memory_samples: list[dict[str, Any]] = []
    finite_checks: list[dict[str, Any]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "train_manifest.json", train_manifest)
    _write_json(run_dir / "validation_manifest.json", validation_manifest)
    _write_json(run_dir / "config.json", config)
    ledger = AppendOnlyLedger(ledger_path)
    segment = 0
    parent_checkpoint_hash: str | None = None
    previous_curve: list[dict[str, Any]] = []
    manifest_extension_evidence: dict[str, Any] | None = None
    restoration_evidence: dict[str, Any] | None = None
    if resume_checkpoint is None:
        set_seed(seed)
        model = make_f_model(vocab_size=TOKENIZER_VOCAB if not smoke else 17, dimensions=dimensions, variant=variant, memory_samples=memory_samples)
        memory_guard("optimizer_initialization", memory_samples)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
        initial_payload = _checkpoint_payload(model, optimizer, update=0, data_position={"next_update": 0, "pair": 0}, run_id=run_id, config=config, identity_hash=identity_hash, data_hashes=data_hashes, teacher_hash=teacher_hash, execution_segment=0)
        checkpoint_hash = save_checkpoint(run_dir / "checkpoint_00000.pt", initial_payload)
        ledger.append({"record_type": "checkpoint", "run_id": run_id, "execution_segment": 0, "update": 0, "checkpoint_hash": checkpoint_hash, "parent_checkpoint_hash": None})
        curve: list[dict[str, Any]] = []
        curve.append({"update": 0, **evaluate_validation(model, validation_documents, memory_samples)})
        _write_json(curve_path, curve)
    else:
        payload, parent_checkpoint_hash = load_checkpoint(resume_checkpoint)
        _validate_f_checkpoint(payload)
        checkpoint_update = int(payload.get("update", -1))
        validate_resume_boundary(checkpoint_update, checkpoint_interval)
        expected_data_position = {"next_update": checkpoint_update, "pair": checkpoint_update // 2}
        if payload.get("data_position") != expected_data_position:
            raise ValueError("resume identity mismatch for data_position")
        if old_train_manifest is not None:
            manifest_extension_evidence = verify_manifest_extension(payload, old_train_manifest, train_manifest)
        expected = {"run_id": run_id, "seed": seed, "identity_hash": identity_hash, "config_hash": canonical_hash(config), "data_hashes": data_hashes, "teacher_hash": teacher_hash}
        if old_train_manifest is not None:
            expected["data_hashes"] = {"train_manifest": train_manifest["manifest_sha256"], "validation_manifest": validation_manifest["manifest_sha256"]}
        for key, value in expected.items():
            if payload.get(key) != value:
                if key == "data_hashes" and old_train_manifest is not None:
                    continue
                raise ValueError(f"resume identity mismatch for {key}")
        if old_train_manifest is not None and payload.get("data_hashes", {}).get("validation_manifest") != validation_manifest["manifest_sha256"]:
            raise ValueError("resume identity mismatch for validation_manifest")
        if old_train_manifest is None and payload.get("data_hashes") != data_hashes:
            raise ValueError("resume identity mismatch for data_hashes")
        if checkpoint_update >= total_updates:
            raise ValueError("resume checkpoint is not before requested end update")
        set_seed(seed)
        model = make_f_model(vocab_size=17 if smoke else TOKENIZER_VOCAB, dimensions=dimensions, variant=variant, memory_samples=memory_samples)
        memory_guard("optimizer_initialization", memory_samples)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng_states"]["torch"])
        random.setstate(payload["rng_states"]["python"])
        restored_state_hashes = _state_hashes(model, optimizer)
        restored_rng_hashes = _rng_state_hashes({"torch": torch.get_rng_state(), "python": random.getstate()})
        restoration_evidence = {
            "checkpoint_parent_hash": parent_checkpoint_hash,
            "data_position": payload.get("data_position"),
            "data_position_exact": payload.get("data_position") == expected_data_position,
            "state_hashes": {**restored_state_hashes, **restored_rng_hashes},
            "exact_equality": {
                "model": _state_equal(model.state_dict(), payload["model"]),
                "optimizer": _state_equal(optimizer.state_dict(), payload["optimizer"]),
                "torch_rng": torch.equal(torch.get_rng_state(), payload["rng_states"]["torch"]),
                "python_rng": random.getstate() == payload["rng_states"]["python"],
            },
            "first_resumed_schedule": schedule(total_updates)[checkpoint_update],
        }
        segment = max((int(event.get("execution_segment", 0)) for event in _read_ledger(ledger_path)), default=0) + 1
        ledger.append({"record_type": "resume", "run_id": run_id, "execution_segment": segment, "update": checkpoint_update, "checkpoint_hash": parent_checkpoint_hash, "parent_checkpoint_hash": parent_checkpoint_hash, "preserved_prior_segments": True})
        curve = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else []
        curve = sorted({int(item["update"]): item for item in curve if int(item["update"]) <= checkpoint_update}.values(), key=lambda item: int(item["update"]))
        previous_curve = list(curve)
        start_update = checkpoint_update
        if not curve or int(curve[-1]["update"]) != checkpoint_update:
            curve.append({"update": checkpoint_update, **evaluate_validation(model, validation_documents, memory_samples)})
            _write_json(curve_path, curve)
    start_update = 0 if resume_checkpoint is None else int(payload["update"])
    pair_states: list[torch.Tensor] | None = None
    model.train()
    execution_end = total_updates if max_updates is None else max_updates
    for item in schedule(total_updates)[start_update:execution_end]:
        update = item["update"]
        pair_start = item["pair"] * PHYSICAL_BATCH
        documents = [train_documents[(pair_start + offset) % len(train_documents)] for offset in range(PHYSICAL_BATCH)]
        memory_guard(f"update_{update}:before_zero_grad", memory_samples)
        optimizer.zero_grad(set_to_none=True)
        loss, next_states, valid_tokens = _batch_loss(model, teacher, documents, item["window"], pair_states, memory_samples)
        _finite_loss(loss, f"update_{update}:before_backward")
        loss_value = float(loss.detach().item())
        finite_checks.append({"update": update + 1, "loss": True})
        memory_guard(f"update_{update}:before_backward", memory_samples)
        loss.backward()
        memory_guard(f"update_{update}:before_clip", memory_samples)
        _finite_gradients(model, f"update_{update}:before_clip")
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
        _finite_gradients(model, f"update_{update}:after_clip")
        memory_guard(f"update_{update}:before_optimizer_step", memory_samples)
        _finite_loss(loss, f"update_{update}:before_optimizer_step")
        finite_checks[-1]["gradients_before_clip"] = True
        finite_checks[-1]["gradients_after_clip"] = True
        optimizer.step()
        if item["window"] == 0:
            pair_states = next_states
        else:
            pair_states = None
        completed = update + 1
        ledger.append({
            "record_type": "update",
            "run_id": run_id,
            "execution_segment": segment,
            "update": completed,
            "source_update": update,
            "window": item["window"],
            "pair": item["pair"],
            "document_indices": [document["document_index"] for document in documents],
            "valid_tokens": valid_tokens,
            "input_shape": [PHYSICAL_BATCH, WINDOW_TOKENS],
            "student_output_shape": [PHYSICAL_BATCH, WINDOW_TOKENS, model.vocab_size],
            "teacher_output_shape": [PHYSICAL_BATCH, WINDOW_TOKENS, model.vocab_size],
            "teacher_route": "direct_distilgpt2" if not smoke else "synthetic_tiny_teacher",
            "student_forward_calls": 1,
            "teacher_forward_calls": 1,
            "backward_calls": 1,
            "clip_calls": 1,
            "optimizer_steps": 1,
            "state_mode": "reset" if item["window"] == 0 else "detached_window_0",
            "state_source_update": None if item["window"] == 0 else completed - 1,
            "finite_loss": True,
            "finite_gradients_before_clip": True,
            "finite_gradients_after_clip": True,
            "optimizer_step_applied": True,
            "loss": loss_value,
            "pre_clip_grad_norm": pre_clip,
            "parent_checkpoint_hash": parent_checkpoint_hash,
        })
        if completed in boundaries:
            curve.append({"update": completed, **evaluate_validation(model, validation_documents, memory_samples)})
            curve = sorted({int(item["update"]): item for item in curve}.values(), key=lambda item: int(item["update"]))
            _write_json(curve_path, curve)
            checkpoint_payload = _checkpoint_payload(model, optimizer, update=completed, data_position={"next_update": completed, "pair": completed // 2}, run_id=run_id, config=config, identity_hash=identity_hash, data_hashes=data_hashes, teacher_hash=teacher_hash, execution_segment=segment)
            checkpoint_hash = save_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", checkpoint_payload)
            ledger.append({"record_type": "checkpoint", "run_id": run_id, "execution_segment": segment, "update": completed, "checkpoint_hash": checkpoint_hash, "parent_checkpoint_hash": parent_checkpoint_hash})
            model.train()
        del loss, next_states
    curve = sorted({int(item["update"]): item for item in curve}.values(), key=lambda item: int(item["update"]))
    _write_json(curve_path, curve)
    final_nll = float(curve[-1]["nll"])
    result = {
        "run_id": run_id,
        "seed": seed,
        "variant": variant,
        "mode": "integration_smoke" if integration_smoke else "smoke" if smoke else "campaign",
        "updates": total_updates,
        "checkpoint_boundaries": list(boundaries),
        "validation_curve": curve,
        "final_nll": final_nll,
        "execution_segments": sorted({int(event.get("execution_segment", 0)) for event in _read_ledger(ledger_path)}),
        "run_dir": artifact_path(run_dir),
        "implementation_identity": implementation_identity(),
        "curve_path": artifact_path(curve_path),
        "checkpoint_paths": [
            artifact_path(run_dir / f"checkpoint_{boundary:05d}.pt")
            for boundary in boundaries
            if (run_dir / f"checkpoint_{boundary:05d}.pt").exists()
        ],
        "memory_safety": {
            "minimum_available_bytes": _minimum_available_memory(memory_samples),
            "required_available_bytes": MIN_AVAILABLE_BYTES,
            "samples": memory_samples,
            "hard_stop_triggered": False,
        },
        "finite_safety": {"checks": finite_checks, "all_passed": all(all(item.values()) for item in finite_checks)},
    }
    if resume_checkpoint is not None:
        result["parent_checkpoint_hash"] = parent_checkpoint_hash
        result["previous_validation_curve"] = previous_curve
        result["manifest_extension_evidence"] = manifest_extension_evidence
        result["restoration_evidence"] = restoration_evidence
        result["executed_through_update"] = start_update if not finite_checks else max(int(item["update"]) for item in finite_checks)
    _write_json(run_dir / "run_result.json", result)
    return result


def _smoke_run(output_dir: Path, variant: str) -> dict[str, Any]:
    train_documents = synthetic_documents(8, 17)
    validation_documents = synthetic_documents(2, 17)
    train_documents, train_manifest = build_manifest(train_documents, split_name="synthetic_train", pair_count=SMOKE_PAIRS)
    validation_documents, validation_manifest = build_manifest(validation_documents, split_name="synthetic_validation")
    return run_single(run_dir=output_dir / "runs" / f"{variant}_seed_20260913", run_id=f"smoke_{variant}_20260913", seed=20260913, variant=variant, train_documents=train_documents, validation_documents=validation_documents, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=TinyTeacher(17), total_updates=2, checkpoint_interval=2, smoke=True, dimensions=(4, 1))


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_segments = [_smoke_run(output_dir, variant) for variant in VARIANTS]
    resume_command_base = [sys.executable, str(Path(__file__).resolve()), "--smoke-resume", "--output-dir", str(output_dir)]
    completed = subprocess.run(resume_command_base, cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    report["initial_segment_results"] = first_segments
    _write_hashed_json(output_dir / "smoke_report.json", report, "report_self_hash")
    return report


def run_smoke_resume(output_dir: Path) -> dict[str, Any]:
    train_documents = synthetic_documents(8, 17)
    validation_documents = synthetic_documents(2, 17)
    train_documents, train_manifest = build_manifest(train_documents, split_name="synthetic_train", pair_count=SMOKE_PAIRS)
    validation_documents, validation_manifest = build_manifest(validation_documents, split_name="synthetic_validation")
    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        run_dir = output_dir / "runs" / f"{variant}_seed_20260913"
        results.append(run_single(run_dir=run_dir, run_id=f"smoke_{variant}_20260913", seed=20260913, variant=variant, train_documents=train_documents, validation_documents=validation_documents, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=TinyTeacher(17), total_updates=4, checkpoint_interval=2, smoke=True, dimensions=(4, 1), resume_checkpoint=run_dir / "checkpoint_00002.pt"))
    report = {"schema": "omega-core-lm-0-r1-scientific-scoping-a-smoke-report-v1", "campaign_id": CAMPAIGN_ID, "mode": "smoke_only", "campaign_started": False, "real_data_loaded": False, "full_update_budget": 0, "smoke_schedule": {"updates": 4, "validation_updates": list(SMOKE_BOUNDARIES), "checkpoint_updates": list(SMOKE_BOUNDARIES)}, "runs": results, "classification": classify_results(results), "fresh_process_resume": True, "append_only_segments": True, "source_hashes": source_hashes(), "policy": validate_policy()}
    return report


def _load_real_documents(*, pair_count: int | None = None, integration_smoke: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], torch.nn.Module]:
    validate_policy()
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    download_config = DownloadConfig(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise RuntimeError(f"tokenizer vocab mismatch: expected {TOKENIZER_VOCAB}, got {len(tokenizer)}")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).to(dtype=torch.float32)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=download_config)
    validation_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION, download_config=download_config)
    all_train = collect_all_eligible_documents(train_dataset, tokenizer)
    all_validation = collect_all_eligible_documents(validation_dataset, tokenizer)
    effective_pair_count = SMOKE_PAIRS if integration_smoke else FULL_PAIRS if pair_count is None else pair_count
    train_documents, train_manifest = build_manifest(all_train, split_name="train", pair_count=effective_pair_count)
    train_keys = {(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in train_documents}
    validation_candidates, validation_manifest = build_manifest(all_validation, split_name="validation", excluded_keys=train_keys)
    if len(validation_candidates) < VALIDATION_DOCUMENTS:
        raise RuntimeError(f"only found {len(validation_candidates)} eligible validation documents")
    validation_documents = validation_candidates[:VALIDATION_DOCUMENTS]
    validation_manifest["documents"] = [public_document(document) for document in validation_documents]
    validation_manifest["document_count"] = len(validation_documents)
    validation_manifest["manifest_sha256"] = canonical_hash({key: value for key, value in validation_manifest.items() if key != "manifest_sha256"})
    return train_documents, validation_documents, train_manifest, validation_manifest, teacher


def run_integration_smoke_child(
    *,
    output_dir: Path,
    run_id: str,
    variant: str,
    seed: int,
    phase: str,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if phase not in {"initial", "resume"}:
        raise ValueError(f"unsupported integration smoke phase: {phase}")
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents(integration_smoke=True)
    run_dir = output_dir / "integration_smoke" / run_id / "runs" / f"{variant}_seed_{seed}"
    result = run_single(
        run_dir=run_dir,
        run_id=f"{run_id}_{variant}",
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=2 if phase == "initial" else 4,
        checkpoint_interval=2,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=resume_checkpoint,
        integration_smoke=True,
    )
    child = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-integration-smoke-child-v1",
        "phase": phase,
        "process_id": os.getpid(),
        "variant": variant,
        "run": result,
        "data": {"dataset": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "test_split_loaded": False},
        "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "route": "direct_distilgpt2"},
        "implementation_identity": implementation_identity(),
        "source_hashes": source_hashes(),
    }
    _write_json(run_dir / f"child_{phase}.json", child)
    return child


def _run_integration_child(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"integration smoke child failed ({completed.returncode}): {completed.stderr[-2000:]}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"integration smoke child did not emit JSON: {completed.stdout[-500:]}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("integration smoke child result must be an object")
    return result


def _verify_integration_smoke(
    *,
    run_dir: Path,
    initial_children: list[dict[str, Any]],
    resumed_children: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_updates = list(range(1, SMOKE_UPDATES + 1))
    expected_shape = [PHYSICAL_BATCH, WINDOW_TOKENS, TOKENIZER_VOCAB]
    per_variant: list[dict[str, bool]] = []
    all_update_records: list[dict[str, Any]] = []
    memory_minima: dict[str, int | None] = {}
    run_summaries: list[dict[str, Any]] = []
    for initial, resumed in zip(initial_children, resumed_children):
        variant = str(resumed["variant"])
        initial_run = initial["run"]
        resumed_run = resumed["run"]
        variant_dir = run_dir / "runs" / f"{variant}_seed_{resumed_run['seed']}"
        final_ledger = _read_ledger(variant_dir / "ledger.jsonl")
        updates = [event for event in final_ledger if event.get("record_type") == "update"]
        all_update_records.extend(updates)
        phase_memory = [initial_run["memory_safety"]["minimum_available_bytes"], resumed_run["memory_safety"]["minimum_available_bytes"]]
        memory_minima[variant] = min(value for value in phase_memory if value is not None)
        initial_curve = initial_run["validation_curve"]
        resumed_curve = resumed_run["validation_curve"]
        checkpoints = [variant_dir / f"checkpoint_{boundary:05d}.pt" for boundary in SMOKE_BOUNDARIES]
        pair_documents = {pair: {int(index) for index in event["document_indices"]} for pair in {0, 1} for event in updates if int(event["pair"]) == pair}
        variant_checks = {
            "curve_points": [int(point["update"]) for point in initial_curve] == [0, 2] and [int(point["update"]) for point in resumed_curve] == list(SMOKE_BOUNDARIES),
            "checkpoint_boundaries": all(path.is_file() for path in checkpoints),
            "fresh_process_resume": initial["process_id"] != resumed["process_id"],
            "no_duplicate_updates": [int(event["update"]) for event in updates] == expected_updates,
            "f_identity": initial["implementation_identity"] == implementation_identity() and resumed["implementation_identity"] == implementation_identity() and resumed_run["implementation_identity"] == implementation_identity(),
            "shape_batch_teacher_route": all(event.get("input_shape") == [PHYSICAL_BATCH, WINDOW_TOKENS] and event.get("student_output_shape") == expected_shape and event.get("teacher_output_shape") == expected_shape and event.get("teacher_route") == "direct_distilgpt2" and event.get("student_forward_calls") == 1 and event.get("teacher_forward_calls") == 1 for event in updates),
            "distinct_pair_document_ids": len(pair_documents) == 2 and pair_documents[0].isdisjoint(pair_documents[1]),
            "reset_and_detached_continuity": all((event["window"] == 0 and event["state_mode"] == "reset" and event["state_source_update"] is None) or (event["window"] == 1 and event["state_mode"] == "detached_window_0" and event["state_source_update"] == event["update"] - 1) for event in updates),
            "finite_checks": resumed_run["finite_safety"]["all_passed"] and all(event["finite_loss"] and event["finite_gradients_before_clip"] and event["finite_gradients_after_clip"] for event in updates),
            "real_pinned_data": initial["data"]["revision"] == DATASET_REVISION and resumed["data"]["revision"] == DATASET_REVISION and initial["teacher"]["revision"] == MODEL_REVISION and resumed["teacher"]["revision"] == MODEL_REVISION,
            "test_split_false": not initial["data"]["test_split_loaded"] and not resumed["data"]["test_split_loaded"],
        }
        per_variant.append(variant_checks)
        run_summaries.append({"variant": variant, "process_ids": {"initial": initial["process_id"], "resume": resumed["process_id"]}, "initial": initial_run, "resumed": resumed_run})
    checks = {key: all(item[key] for item in per_variant) for key in per_variant[0]} if per_variant else {}
    checks["both_variants"] = {str(item["variant"]) for item in resumed_children} == set(VARIANTS)
    checks["exact_updates_total"] = len(all_update_records) == 8 and all(event.get("optimizer_steps") == 1 for event in all_update_records)
    checks["all_updates_expected"] = len(all_update_records) == 8 and sorted(int(event["update"]) for event in all_update_records) == [1, 1, 2, 2, 3, 3, 4, 4]
    checks["all_updates_valid_tokens"] = all(int(event["valid_tokens"]) == PHYSICAL_BATCH * WINDOW_TOKENS for event in all_update_records)
    checks["curve_paths_persisted"] = all(Path(ROOT / summary["resumed"]["curve_path"]).is_file() for summary in run_summaries)
    checks["checkpoint_paths_persisted"] = all(all(Path(ROOT / path).is_file() for path in summary["resumed"]["checkpoint_paths"]) for summary in run_summaries)
    checks["status_eligibility"] = all(checks.values())
    return {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-integration-smoke-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": "integration_smoke",
        "status": "completed" if checks["status_eligibility"] else "failed",
        "campaign_started": False,
        "full_update_budget": 0,
        "exact_updates_total": len(all_update_records),
        "variants": list(VARIANTS),
        "updates_per_variant": SMOKE_UPDATES,
        "runs": run_summaries,
        "checks": checks,
        "implementation_identity": implementation_identity(),
        "source_hashes": source_hashes(),
        "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "route": "direct_distilgpt2"},
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "train_split": True, "validation_split": True, "test_split_loaded": False},
        "memory_safety": {"required_available_bytes": MIN_AVAILABLE_BYTES, "minimum_available_bytes_by_variant": memory_minima, "hard_stop_triggered": False},
        "curve_points": list(SMOKE_BOUNDARIES),
        "checkpoint_boundaries": list(SMOKE_BOUNDARIES),
        "fresh_process_resume": checks["fresh_process_resume"],
        "no_scientific_campaign_or_full_launch": True,
        "run_dir": artifact_path(run_dir),
    }


def run_integration_smoke(output_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or f"integration_smoke_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    run_dir = output_dir / "integration_smoke" / run_id
    if run_dir.exists():
        raise FileExistsError(f"integration smoke output already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).resolve())
    initial_children: list[dict[str, Any]] = []
    resumed_children: list[dict[str, Any]] = []
    for variant in VARIANTS:
        base = [sys.executable, script, "--integration-smoke-child", "--integration-smoke-phase", "initial", "--output-dir", str(output_dir), "--integration-smoke-run-id", run_id, "--variant", variant, "--seed", str(SEEDS[0])]
        initial = _run_integration_child(base)
        initial_children.append(initial)
        checkpoint = run_dir / "runs" / f"{variant}_seed_{SEEDS[0]}" / "checkpoint_00002.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"initial integration smoke child did not persist {checkpoint}")
        resume_command = [sys.executable, script, "--integration-smoke-child", "--integration-smoke-phase", "resume", "--output-dir", str(output_dir), "--integration-smoke-run-id", run_id, "--variant", variant, "--seed", str(SEEDS[0]), "--resume-checkpoint", str(checkpoint)]
        resumed_children.append(_run_integration_child(resume_command))
    report = _verify_integration_smoke(run_dir=run_dir, initial_children=initial_children, resumed_children=resumed_children)
    _write_hashed_json(run_dir / "integration_smoke_report.json", report, "report_self_hash")
    return report


def run_full(output_dir: Path) -> dict[str, Any]:
    validate_policy()
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents()
    _write_json(output_dir / "train_manifest.json", train_manifest)
    _write_json(output_dir / "validation_manifest.json", validation_manifest)
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            run_id = f"{variant}_seed_{seed}"
            results.append(run_single(run_dir=output_dir / "runs" / run_id, run_id=run_id, seed=seed, variant=variant, train_documents=train_documents, validation_documents=validation_documents, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=teacher, total_updates=FULL_UPDATES, checkpoint_interval=500, smoke=False, dimensions=(128, 8)))
    report = {"schema": "omega-core-lm-0-r1-scientific-scoping-a-full-report-v1", "campaign_id": CAMPAIGN_ID, "mode": "full_campaign", "campaign_started": True, "configuration": {"variants": list(VARIANTS), "seeds": list(SEEDS), "updates_per_run": FULL_UPDATES, "total_runs": 4, "total_updates": 8000}, "runs": results, "classification": classify_results(results), "source_hashes": source_hashes(), "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION}, "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION}, "manifests": {"train": file_hash(output_dir / "train_manifest.json"), "validation": file_hash(output_dir / "validation_manifest.json")}, "freeze_hash_claim": False}
    _write_hashed_json(output_dir / "full_report.json", report, "report_self_hash")
    return report


def run_resume(output_dir: Path, checkpoint: Path, run_id: str, seed: int, variant: str) -> dict[str, Any]:
    """Resume one authorized campaign run from its last valid 500 boundary."""
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents()
    result = run_single(
        run_dir=checkpoint.resolve().parent,
        run_id=run_id,
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=FULL_UPDATES,
        checkpoint_interval=500,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=checkpoint.resolve(),
    )
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-resume-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": "authorized_resume",
        "campaign_started": True,
        "checkpoint": checkpoint.resolve().as_posix(),
        "run": result,
        "source_hashes": source_hashes(),
        "freeze_hash_claim": False,
    }
    _write_hashed_json(output_dir / "resume_report.json", report, "report_self_hash")
    return report


def _load_scope_b_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], torch.nn.Module, dict[str, Any]]:
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents(pair_count=SCOPE_B_PAIRS)
    _, old_train_manifest = build_manifest(train_documents, split_name="train", pair_count=FULL_PAIRS)
    return train_documents, validation_documents, train_manifest, validation_manifest, teacher, old_train_manifest


def run_scope_b(output_dir: Path, checkpoint: Path, run_id: str, seed: int, variant: str) -> dict[str, Any]:
    """Continue one A checkpoint with explicitly verified 1000-to-2500 pair extension."""
    train_documents, validation_documents, train_manifest, validation_manifest, teacher, old_train_manifest = _load_scope_b_inputs()
    checkpoint = checkpoint.resolve()
    checkpoint_payload, _ = load_checkpoint(checkpoint)
    extension_evidence = verify_manifest_extension(checkpoint_payload, old_train_manifest, train_manifest)
    result = run_single(
        run_dir=checkpoint.parent,
        run_id=run_id,
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=SCOPE_B_UPDATES,
        checkpoint_interval=500,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=checkpoint,
        old_train_manifest=old_train_manifest,
    )
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-b-resume-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "scope": "B",
        "mode": "authorized_scope_b_resume",
        "campaign_started": True,
        "checkpoint": checkpoint.as_posix(),
        "parent_checkpoint_hash": result["parent_checkpoint_hash"],
        "manifest_extension_evidence": extension_evidence,
        "restoration_evidence": result["restoration_evidence"],
        "prior_validation_curve": result["previous_validation_curve"],
        "run": result,
        "source_hashes": source_hashes(),
        "freeze_hash_claim": False,
    }
    _write_hashed_json(output_dir / "scope_b_resume_report.json", report, "report_self_hash")
    return report


def _validate_scope_b_continue_checkpoint(checkpoint_payload: dict[str, Any]) -> int:
    checkpoint_update = int(checkpoint_payload.get("update", -1))
    if checkpoint_update < SCOPE_B_CONTINUE_MIN_UPDATE:
        raise ValueError("SCOPE-B continuation requires checkpoint update >= 2500")
    validate_resume_boundary(checkpoint_update, 500)
    expected_data_position = {"next_update": checkpoint_update, "pair": checkpoint_update // 2}
    if checkpoint_payload.get("data_position") != expected_data_position:
        raise ValueError("SCOPE-B continuation checkpoint has invalid data_position")
    return checkpoint_update


def run_scope_b_continue(output_dir: Path, checkpoint: Path, run_id: str, seed: int, variant: str) -> dict[str, Any]:
    """Continue an already-B checkpoint without rerunning A-to-B extension proof."""
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents(pair_count=SCOPE_B_PAIRS)
    checkpoint = checkpoint.resolve()
    checkpoint_payload, checkpoint_hash = load_checkpoint(checkpoint)
    checkpoint_update = _validate_scope_b_continue_checkpoint(checkpoint_payload)
    checkpoint_data_hashes = checkpoint_payload.get("data_hashes")
    if not isinstance(checkpoint_data_hashes, dict):
        raise ValueError("SCOPE-B continuation checkpoint data_hashes is missing or malformed")
    expected_train_hash = train_manifest["manifest_sha256"]
    expected_validation_hash = validation_manifest["manifest_sha256"]
    if checkpoint_data_hashes.get("train_manifest") != expected_train_hash:
        raise ValueError("SCOPE-B continuation checkpoint train_manifest hash does not match current B manifest")
    if checkpoint_data_hashes.get("validation_manifest") != expected_validation_hash:
        raise ValueError("SCOPE-B continuation checkpoint validation_manifest hash does not match current validation manifest")
    identity_evidence = {
        "checkpoint_update": checkpoint_update,
        "checkpoint_train_manifest_sha256": checkpoint_data_hashes["train_manifest"],
        "current_b_train_manifest_sha256": expected_train_hash,
        "train_manifest_exact_match": True,
        "checkpoint_validation_manifest_sha256": checkpoint_data_hashes["validation_manifest"],
        "current_validation_manifest_sha256": expected_validation_hash,
        "validation_manifest_exact_match": True,
    }
    result = run_single(
        run_dir=checkpoint.parent,
        run_id=run_id,
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=SCOPE_B_UPDATES,
        checkpoint_interval=500,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=checkpoint,
        old_train_manifest=None,
    )
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-b-continuation-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "scope": "B",
        "mode": "authorized_scope_b_continue",
        "campaign_started": True,
        "checkpoint": checkpoint.as_posix(),
        "parent_checkpoint_hash": checkpoint_hash,
        "checkpoint_update": checkpoint_update,
        "manifest_identity_evidence": identity_evidence,
        "extension_verification": {
            "status": "reused_from_checkpoint_already_b_identity",
            "old_to_new_extension_verification_rerun": False,
            "checkpoint_already_identifies_b_manifest": True,
            "normal_exact_hash_identity_validation_active": True,
            "statement": "Old-to-new extension verification is reused from the checkpoint's already-B identity; it is not rerun or bypassed. Normal exact hash identity validation remains active.",
        },
        "restoration_evidence": result["restoration_evidence"],
        "prior_validation_curve": result["previous_validation_curve"],
        "run": result,
        "source_hashes": source_hashes(),
        "freeze_hash_claim": False,
    }
    _write_hashed_json(output_dir / "scope_b_continue_report.json", report, "report_self_hash")
    return report


def run_scope_b_smoke(output_dir: Path, checkpoint: Path, run_id: str, seed: int, variant: str) -> dict[str, Any]:
    """Run two uncheckpointed B updates in isolated output using a real A checkpoint."""
    train_documents, validation_documents, train_manifest, validation_manifest, teacher, old_train_manifest = _load_scope_b_inputs()
    checkpoint = checkpoint.resolve()
    checkpoint_payload, _ = load_checkpoint(checkpoint)
    extension_evidence = verify_manifest_extension(checkpoint_payload, old_train_manifest, train_manifest)
    run_dir = output_dir / "scope_b_smoke" / run_id / "runs" / f"{variant}_seed_{seed}"
    prior_curve_path = checkpoint.parent / "validation_curve.json"
    if not prior_curve_path.is_file():
        raise FileNotFoundError(f"scope B smoke requires prior validation curve: {prior_curve_path}")
    prior_curve = json.loads(prior_curve_path.read_text(encoding="utf-8"))
    if [int(point["update"]) for point in prior_curve] != list(FULL_BOUNDARIES):
        raise ValueError("scope B smoke requires validation curve points 0 through 2000")
    _write_json(run_dir / "validation_curve.json", prior_curve)
    result = run_single(
        run_dir=run_dir,
        run_id=run_id,
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=SCOPE_B_UPDATES,
        checkpoint_interval=500,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=checkpoint,
        old_train_manifest=old_train_manifest,
        max_updates=FULL_UPDATES + 2,
    )
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-b-bounded-smoke-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "scope": "B",
        "mode": "bounded_scope_b_smoke",
        "campaign_started": False,
        "real_data_loaded": True,
        "total_updates": SCOPE_B_UPDATES,
        "uncheckpointed_tail_updates": 2,
        "checkpoint": checkpoint.as_posix(),
        "parent_checkpoint_hash": result["parent_checkpoint_hash"],
        "manifest_extension_evidence": extension_evidence,
        "restoration_evidence": result["restoration_evidence"],
        "prior_validation_curve": result["previous_validation_curve"],
        "run": result,
        "source_hashes": source_hashes(),
        "no_canonical_a_artifacts_modified": True,
    }
    _write_hashed_json(output_dir / "scope_b_smoke_report.json", report, "report_self_hash")
    return report


def run_fresh(output_dir: Path, run_id: str, seed: int, variant: str) -> dict[str, Any]:
    """Run one authorized campaign run from its initial state."""
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents()
    result = run_single(
        run_dir=output_dir / "runs" / run_id,
        run_id=run_id,
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=FULL_UPDATES,
        checkpoint_interval=500,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=None,
    )
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-fresh-run-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": "authorized_fresh_run",
        "campaign_started": True,
        "run": result,
        "source_hashes": source_hashes(),
        "freeze_hash_claim": False,
    }
    _write_hashed_json(output_dir / f"fresh_report_{run_id}.json", report, "report_self_hash")
    return report


def run_scope_c(output_dir: Path, run_id: str, seed: int, variant: str, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    """Run one authorized Scope-C run from its initial state."""
    if seed != SCOPE_C_SEED:
        raise ValueError("Scope-C requires exact seed 20260915")
    train_documents, validation_documents, train_manifest, validation_manifest, teacher = _load_real_documents()
    result = run_single(
        run_dir=output_dir / "runs" / run_id,
        run_id=run_id,
        seed=seed,
        variant=variant,
        train_documents=train_documents,
        validation_documents=validation_documents,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        total_updates=SCOPE_B_UPDATES,
        checkpoint_interval=500,
        smoke=False,
        dimensions=(128, 8),
        resume_checkpoint=resume_checkpoint,
    )
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-scope-c-run-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "scope": "C",
        "mode": "authorized_scope_c_fresh_run",
        "campaign_started": True,
        "run": result,
        "source_hashes": source_hashes(),
        "freeze_hash_claim": False,
    }
    _write_hashed_json(output_dir / f"scope_c_report_{run_id}.json", report, "report_self_hash")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CAMPAIGN_ID)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--integration-smoke", action="store_true")
    parser.add_argument("--integration-smoke-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--integration-smoke-phase", choices=("initial", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--integration-smoke-run-id", help=argparse.SUPPRESS)
    parser.add_argument("--scope-b", action="store_true", help="Resume an A checkpoint with verified 1000-to-2500 pair extension")
    parser.add_argument("--scope-b-continue", action="store_true", help="Continue an already-B checkpoint with exact manifest identity")
    parser.add_argument("--scope-b-smoke", action="store_true", help="Run bounded isolated B continuation smoke")
    parser.add_argument("--scope-c", action="store_true", help="Run a fresh Scope-C seed at the 5000-update schedule")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-smoke", action="store_true", help="Required authorization after smoke review")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.smoke, args.smoke_resume, args.integration_smoke, args.integration_smoke_child, args.scope_b, args.scope_b_continue, args.scope_b_smoke, args.scope_c)) > 1:
        raise SystemExit("choose one execution mode")
    args.output_dir = args.output_dir.resolve()
    if args.smoke:
        print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.smoke_resume:
        print(json.dumps(run_smoke_resume(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.integration_smoke_child:
        if not args.integration_smoke_phase or not args.integration_smoke_run_id or not args.variant or args.seed is None:
            parser.error("integration smoke child requires phase, run id, seed, and variant")
        checkpoint = args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None
        print(json.dumps(run_integration_smoke_child(output_dir=args.output_dir, run_id=args.integration_smoke_run_id, variant=args.variant, seed=args.seed, phase=args.integration_smoke_phase, resume_checkpoint=checkpoint), indent=2, sort_keys=True))
        return 0
    if args.integration_smoke:
        print(json.dumps(run_integration_smoke(args.output_dir, args.integration_smoke_run_id), indent=2, sort_keys=True))
        return 0
    if args.scope_b_smoke:
        if not (args.resume_checkpoint and args.run_id and args.seed is not None and args.variant):
            parser.error("scope B smoke requires --resume-checkpoint --run-id --seed and --variant")
        print(json.dumps(run_scope_b_smoke(args.output_dir, args.resume_checkpoint, args.run_id, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.scope_b_continue:
        if not (args.full and args.confirm_smoke and args.resume_checkpoint and args.run_id and args.seed in SEEDS and args.variant):
            parser.error("scope B continuation requires --full --confirm-smoke --resume-checkpoint --run-id --seed and --variant")
        print(json.dumps(run_scope_b_continue(args.output_dir, args.resume_checkpoint, args.run_id, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.scope_b:
        if not (args.full and args.confirm_smoke and args.resume_checkpoint and args.run_id and args.seed in SEEDS and args.variant):
            parser.error("scope B requires --full --confirm-smoke --resume-checkpoint --run-id --seed and --variant")
        print(json.dumps(run_scope_b(args.output_dir, args.resume_checkpoint, args.run_id, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.scope_c:
        if not (args.full and args.confirm_smoke and args.run_id and args.seed is not None and args.variant):
            parser.error("scope C requires --full --confirm-smoke --run-id --seed and --variant")
        if args.seed != SCOPE_C_SEED:
            parser.error("scope C requires exact seed 20260915")
        checkpoint = args.resume_checkpoint if args.resume_checkpoint is not None else None
        print(json.dumps(run_scope_c(args.output_dir, args.run_id, args.seed, args.variant, resume_checkpoint=checkpoint), indent=2, sort_keys=True))
        return 0
    if args.resume_checkpoint is not None:
        if not (args.full and args.confirm_smoke and args.run_id and args.seed in SEEDS and args.variant):
            parser.error("resume requires --full --confirm-smoke --run-id --seed and --variant")
        print(json.dumps(run_resume(args.output_dir, args.resume_checkpoint, args.run_id, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.full and args.confirm_smoke and any(value is not None for value in (args.run_id, args.seed, args.variant)):
        if not (args.run_id and args.seed in SEEDS and args.variant):
            parser.error("fresh run requires --full --confirm-smoke --run-id --seed and --variant")
        print(json.dumps(run_fresh(args.output_dir, args.run_id, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.full and args.confirm_smoke:
        print(json.dumps(run_full(args.output_dir), indent=2, sort_keys=True))
        return 0
    parser.error("full campaign blocked: run --smoke first, then explicitly provide --full --confirm-smoke")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
