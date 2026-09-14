"""OMEGA R1 scientific scoping A runner.

Default execution is deliberately blocked. ``--smoke`` is synthetic and CPU-only;
the full campaign requires both ``--full`` and ``--confirm-smoke``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

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


CAMPAIGN_ID = "OMEGA-CORE-LM-0-R1-SCIENTIFIC-SCOPING-A"
VARIANTS = ("shared_K1", "shared_K4")
SEEDS = (20260913, 20260914)
FULL_UPDATES = 2000
FULL_PAIRS = 1000
FULL_BOUNDARIES = (0, 500, 1000, 1500, 2000)
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def token_hash(tokens: Iterable[int]) -> str:
    return sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in tokens))


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_hashes() -> dict[str, str]:
    current = Path(__file__).resolve()
    source = SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    return {"runner": file_hash(current), "current_r1_source": file_hash(source)}


def validate_policy() -> dict[str, Any]:
    policy = {
        "device": "cpu",
        "dtype": "float32",
        "execution": "eager",
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "microbatches": MICROBATCHES,
        "optimizer": "AdamW",
        "teacher_frozen": True,
    }
    if policy["device"] != "cpu" or policy["dtype"] != "float32" or policy["execution"] != "eager":
        raise AssertionError("scoping A policy drift")
    if (PHYSICAL_BATCH, EFFECTIVE_BATCH, MICROBATCHES) != (8, 8, 1):
        raise AssertionError("batch policy drift")
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
        "optimizer_state_hash": canonical_hash(optimizer.state_dict()),
    }


def _config_payload(*, variant: str, seed: int, smoke: bool, dimensions: tuple[int, int]) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "variant": variant,
        "seed": seed,
        "device": "cpu",
        "dtype": "float32",
        "execution": "eager",
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
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


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
        "provenance": {"source_hashes": source_hashes(), "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__},
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


def _batch_loss(
    model: OmegaCoreLM0R1Technical,
    teacher: torch.nn.Module,
    documents: list[dict[str, Any]],
    window: int,
    previous_states: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    losses: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    valid_tokens = 0
    for index, document in enumerate(documents):
        source = torch.tensor(document["tokens"], dtype=torch.long).unsqueeze(0)
        input_start = window * WINDOW_TOKENS
        input_ids = source[:, input_start : input_start + WINDOW_TOKENS]
        targets = source[:, input_start + 1 : input_start + WINDOW_TOKENS + 1]
        if window == 0:
            state = model.initial_state(1, device=torch.device("cpu"))
        else:
            if previous_states is None:
                raise ValueError("window 1 requires previous detached window 0 states")
            state = previous_states[index]
        next_state, logits = model.forward_window(input_ids, state)
        teacher_logits = teacher_window_logits(teacher, source, window)
        mask = torch.ones_like(targets, dtype=torch.bool)
        losses.append(distillation_loss(logits, teacher_logits[:, :WINDOW_TOKENS], targets, mask)["total"])
        next_states.append(next_state.detach())
        valid_tokens += int(mask.sum().item())
    return torch.stack(losses).mean(), next_states, valid_tokens


def evaluate_validation(model: OmegaCoreLM0R1Technical, documents: list[dict[str, Any]]) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for document in documents:
            source = torch.tensor(document["tokens"], dtype=torch.long).unsqueeze(0)
            state = model.initial_state(1, device=torch.device("cpu"))
            for window in (0, 1):
                start = window * WINDOW_TOKENS
                inputs = source[:, start : start + WINDOW_TOKENS]
                targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
                state, logits = model.forward_window(inputs, state.detach() if window else state)
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
) -> dict[str, Any]:
    validate_policy()
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    boundaries = checkpoint_boundaries(total_updates, checkpoint_interval)
    config = _config_payload(variant=variant, seed=seed, smoke=smoke, dimensions=dimensions)
    identity_hash = canonical_hash({"campaign_id": CAMPAIGN_ID, "run_id": run_id, "variant": variant, "seed": seed})
    data_hashes = {"train_manifest": train_manifest["manifest_sha256"], "validation_manifest": validation_manifest["manifest_sha256"]}
    teacher_hash = canonical_hash(config["teacher"])
    ledger_path = run_dir / "ledger.jsonl"
    curve_path = run_dir / "validation_curve.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "train_manifest.json", train_manifest)
    _write_json(run_dir / "validation_manifest.json", validation_manifest)
    ledger = AppendOnlyLedger(ledger_path)
    segment = 0
    parent_checkpoint_hash: str | None = None
    previous_curve: list[dict[str, Any]] = []
    if resume_checkpoint is None:
        set_seed(seed)
        model = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB if not smoke else 17, dimension=dimensions[0], slots=dimensions[1], rounds=1 if variant == "shared_K1" else 4, variant="shared").to(dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
        initial_payload = _checkpoint_payload(model, optimizer, update=0, data_position={"next_update": 0, "pair": 0}, run_id=run_id, config=config, identity_hash=identity_hash, data_hashes=data_hashes, teacher_hash=teacher_hash, execution_segment=0)
        checkpoint_hash = save_checkpoint(run_dir / "checkpoint_00000.pt", initial_payload)
        ledger.append({"record_type": "checkpoint", "run_id": run_id, "execution_segment": 0, "update": 0, "checkpoint_hash": checkpoint_hash, "parent_checkpoint_hash": None})
        curve: list[dict[str, Any]] = [{"update": 0, **evaluate_validation(model, validation_documents)}]
    else:
        payload, parent_checkpoint_hash = load_checkpoint(resume_checkpoint)
        checkpoint_update = int(payload.get("update", -1))
        validate_resume_boundary(checkpoint_update, checkpoint_interval)
        expected = {"run_id": run_id, "seed": seed, "identity_hash": identity_hash, "config_hash": canonical_hash(config), "data_hashes": data_hashes, "teacher_hash": teacher_hash}
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"resume identity mismatch for {key}")
        if checkpoint_update >= total_updates:
            raise ValueError("resume checkpoint is not before requested end update")
        model = OmegaCoreLM0R1Technical(vocab_size=17 if smoke else TOKENIZER_VOCAB, dimension=dimensions[0], slots=dimensions[1], rounds=1 if variant == "shared_K1" else 4, variant="shared").to(dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng_states"]["torch"])
        random.setstate(payload["rng_states"]["python"])
        segment = max((int(event.get("execution_segment", 0)) for event in _read_ledger(ledger_path)), default=0) + 1
        ledger.append({"record_type": "resume", "run_id": run_id, "execution_segment": segment, "update": checkpoint_update, "checkpoint_hash": parent_checkpoint_hash, "parent_checkpoint_hash": parent_checkpoint_hash, "preserved_prior_segments": True})
        curve = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else []
        previous_curve = list(curve)
        start_update = checkpoint_update
        if not curve or int(curve[-1]["update"]) != checkpoint_update:
            curve.append({"update": checkpoint_update, **evaluate_validation(model, validation_documents)})
    start_update = 0 if resume_checkpoint is None else int(payload["update"])
    pair_states: list[torch.Tensor] | None = None
    model.train()
    for item in schedule(total_updates)[start_update:]:
        update = item["update"]
        pair_start = item["pair"] * PHYSICAL_BATCH
        documents = [train_documents[(pair_start + offset) % len(train_documents)] for offset in range(PHYSICAL_BATCH)]
        optimizer.zero_grad(set_to_none=True)
        loss, next_states, valid_tokens = _batch_loss(model, teacher, documents, item["window"], pair_states)
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
        optimizer.step()
        if item["window"] == 0:
            pair_states = next_states
        else:
            pair_states = None
        completed = update + 1
        ledger.append({"record_type": "update", "run_id": run_id, "execution_segment": segment, "update": completed, "source_update": update, "window": item["window"], "pair": item["pair"], "document_indices": [document["document_index"] for document in documents], "valid_tokens": valid_tokens, "loss": float(loss.detach().item()), "pre_clip_grad_norm": pre_clip, "parent_checkpoint_hash": parent_checkpoint_hash})
        if completed in boundaries:
            curve.append({"update": completed, **evaluate_validation(model, validation_documents)})
            checkpoint_payload = _checkpoint_payload(model, optimizer, update=completed, data_position={"next_update": completed, "pair": completed // 2}, run_id=run_id, config=config, identity_hash=identity_hash, data_hashes=data_hashes, teacher_hash=teacher_hash, execution_segment=segment)
            checkpoint_hash = save_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", checkpoint_payload)
            ledger.append({"record_type": "checkpoint", "run_id": run_id, "execution_segment": segment, "update": completed, "checkpoint_hash": checkpoint_hash, "parent_checkpoint_hash": parent_checkpoint_hash})
            model.train()
    curve = sorted({int(item["update"]): item for item in curve}.values(), key=lambda item: int(item["update"]))
    _write_json(curve_path, curve)
    final_nll = float(curve[-1]["nll"])
    result = {"run_id": run_id, "seed": seed, "variant": variant, "mode": "smoke" if smoke else "campaign", "updates": total_updates, "checkpoint_boundaries": list(boundaries), "validation_curve": curve, "final_nll": final_nll, "execution_segments": sorted({int(event.get("execution_segment", 0)) for event in _read_ledger(ledger_path)}), "run_dir": run_dir.relative_to(ROOT).as_posix()}
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


def _load_real_documents() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], torch.nn.Module]:
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
    train_documents, train_manifest = build_manifest(all_train, split_name="train", pair_count=FULL_PAIRS)
    train_keys = {(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in train_documents}
    validation_candidates, validation_manifest = build_manifest(all_validation, split_name="validation", excluded_keys=train_keys)
    if len(validation_candidates) < VALIDATION_DOCUMENTS:
        raise RuntimeError(f"only found {len(validation_candidates)} eligible validation documents")
    validation_documents = validation_candidates[:VALIDATION_DOCUMENTS]
    validation_manifest["documents"] = [public_document(document) for document in validation_documents]
    validation_manifest["document_count"] = len(validation_documents)
    validation_manifest["manifest_sha256"] = canonical_hash({key: value for key, value in validation_manifest.items() if key != "manifest_sha256"})
    return train_documents, validation_documents, train_manifest, validation_manifest, teacher


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CAMPAIGN_ID)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-smoke", action="store_true", help="Required authorization after smoke review")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    if args.smoke and args.smoke_resume:
        raise SystemExit("choose one smoke mode")
    args.output_dir = args.output_dir.resolve()
    if args.smoke:
        print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.smoke_resume:
        print(json.dumps(run_smoke_resume(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.resume_checkpoint is not None:
        if not (args.full and args.confirm_smoke and args.run_id and args.seed in SEEDS and args.variant):
            parser.error("resume requires --full --confirm-smoke --run-id --seed and --variant")
        print(json.dumps(run_resume(args.output_dir, args.resume_checkpoint, args.run_id, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.full and args.confirm_smoke:
        print(json.dumps(run_full(args.output_dir), indent=2, sort_keys=True))
        return 0
    parser.error("full campaign blocked: run --smoke first, then explicitly provide --full --confirm-smoke")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
