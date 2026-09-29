"""Isolated OMEGA backend-quality runner; real training is authorization-gated."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from r1_masked_token_mean_loss import (  # noqa: E402
    LOSS_CONTRACT,
    r1_masked_token_mean_loss,
)

CAMPAIGN_ROOT = REPO_ROOT / "t1_trainability_lab_v0.1.0" / "campaign"
P2R0 = CAMPAIGN_ROOT / "omega_native_runtime_p2r0"
P0_DIR = CAMPAIGN_ROOT / "omega_native_runtime_p0"
CE_DIR = CAMPAIGN_ROOT / "omega_ce_only_baseline"
R1_SCOPE = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
EXPANDED_VALIDATION = CAMPAIGN_ROOT / "omega_expanded_frozen_validation"
INPUTS_DIR = HERE / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
QUALIFICATION_MANIFEST = INPUTS_DIR / "qualification_manifest.json"
TRAIN_MANIFEST = INPUTS_DIR / "train_manifest_1000_pairs.json"
VALIDATION_MANIFEST = INPUTS_DIR / "validation_manifest_60.json"
RESULTS_DIR = HERE / "results"
BE376_DLL = Path(r"C:\Users\danil\bpf2a\t1_trainability_lab_v0.1.0\campaign\omega_native_runtime_p2r0\native\build-diagnostics-out-state-candidate-verified\python\omega_recurrent.dll")
BE376_SHA256 = "be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TEACHER_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
K_VALUES = (1, 4)
BLOCK_SEEDS = {"A": (20260913, 20260914), "B": (20260915, 20260916, 20260917)}
UPDATES_PER_RUN = 2000
CHECKPOINT_INTERVAL = 500
EFFECTIVE_BATCH = 8
PHYSICAL_BATCH = 8
ACCUMULATIONS = 1
NATIVE_WORKERS = 4
TORCH_THREADS = 4
TORCH_INTEROP_THREADS = 1
WINDOW_TOKENS = 256
GRAD_CLIP_NORM = 1.0


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode("ascii"))
    h.update(repr(tuple(tensor.shape)).encode("ascii"))
    h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _value_hash(value: Any) -> str:
    if torch.is_tensor(value):
        return _tensor_hash(value)
    if isinstance(value, dict):
        return _canonical_hash({str(key): _value_hash(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return _canonical_hash([_value_hash(item) for item in value])
    return _canonical_hash(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _load_qualification_manifest() -> dict[str, Any]:
    if not QUALIFICATION_MANIFEST.is_file() or not TRAIN_MANIFEST.is_file() or not VALIDATION_MANIFEST.is_file():
        raise FileNotFoundError("prepared quality inputs missing; run prepare_backend_quality_inputs.py --prepare-inputs")
    manifest = json.loads(QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    signature = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError("qualification manifest self-hash mismatch")
    if manifest.get("status") != "PREPARED_NO_SCIENTIFIC_TRAINING":
        raise ValueError("qualification manifest has unexpected status")
    if manifest.get("real_training_authorized") is not False:
        raise ValueError("preparation manifest unexpectedly enables real training")
    block_a = manifest.get("block_a", {})
    if (
        block_a.get("seeds") != [20260913, 20260914]
        or block_a.get("rounds") != [1, 4]
        or block_a.get("backends") != ["pytorch", "native"]
        or block_a.get("independent_run_count") != 8
        or block_a.get("updates_per_run") != 2000
        or block_a.get("total_updates") != 16000
        or block_a.get("one_common_init_per_seed_K_shared_by_backends") is not True
        or block_a.get("fresh_empty_adamw_per_arm") is not True
        or block_a.get("optimizer_warmup_updates") != 0
        or block_a.get("no_benchmark_moments_reused") is not True
    ):
        raise ValueError("sealed Block-A run count/common-init/optimizer contract mismatch")
    resume_contract = manifest.get("technical_resume_test_contract", {})
    expected_resume_contract = {
        "test_id": "REAL_RUNNER_CHECKPOINT_RESUME_V1",
        "seed": 20260913,
        "K_values": [1, 4],
        "backends": ["pytorch", "native"],
        "route_count": 4,
        "continuous_updates_per_route": 4,
        "segmented_prefix_updates_per_route": 3,
        "checkpoint_after_completed_update": 3,
        "fresh_process_resume_updates_per_route": 1,
        "checkpoint_interval": 3,
        "total_technical_optimizer_updates": 32,
        "actual_quality_runner_and_actual_selected_dll": True,
        "canonical_loss_contract": LOSS_CONTRACT,
        "loss_callable": "r1_masked_token_mean_loss",
        "continuous_and_resumed_compared_within_same_backend_K": True,
        "exact_compare": ["model_parameters_and_buffers", "adamw_moments_and_step", "rng", "data_cursor", "recurrent_state", "per_update_loss_clip_and_data_ledger"],
        "exclude_from_exact_compare": ["timestamps", "elapsed_runtime", "memory_samples", "execution_segment_number"],
        "validation_or_test_loaded": False,
        "warmup_optimizer_updates": 0,
        "checkpoint_must_restore_detached_causal_state": True,
    }
    if any(resume_contract.get(key) != value for key, value in expected_resume_contract.items()):
        raise ValueError("sealed real-runner resume-test contract drift")
    loss_contract = manifest.get("loss_contract", {})
    if loss_contract.get("id") != LOSS_CONTRACT:
        raise ValueError("qualification manifest does not pin the canonical R1 masked-token-mean loss")
    expected_loss_contract = {
        "teacher_distribution": "softmax(teacher_logits/tau)",
        "student_distribution": "log_softmax(student_logits/tau)",
        "kl_direction": "teacher||student",
        "kl_reduction": "vocab_sum_then_valid_token_mean",
        "temperature": 2.0,
        "temperature_squared_applied_once": True,
        "ce_weight": 0.5,
        "kl_weight": 0.5,
        "valid_tokens_per_update": EFFECTIVE_BATCH * WINDOW_TOKENS,
        "denominator": "valid_mask.sum(); require > 0",
        "interface": {
            "student_logits": "[B,L,V]",
            "teacher_logits": "[B,L,V]",
            "targets": "[B,L]",
            "valid_mask": "[B,L] bool",
        },
    }
    if any(loss_contract.get(key) != value for key, value in expected_loss_contract.items()):
        raise ValueError("qualification manifest loss contract fields drifted")
    loss_implementation = loss_contract.get("implementation", {})
    loaded_loss_path = Path(r1_masked_token_mean_loss.__code__.co_filename).resolve()
    if (
        loss_implementation.get("callable") != r1_masked_token_mean_loss.__name__
        or Path(loss_implementation.get("path", "")).resolve() != loaded_loss_path
        or _sha256_file(loaded_loss_path) != loss_implementation.get("sha256")
    ):
        raise ValueError("loaded canonical loss callable/path/hash differs from qualification manifest")
    historical_helper = loss_contract.get("historical_nonconforming_helper", {})
    if (
        historical_helper.get("sha256") != manifest["source_identities"]["teacher_logit_cache_base"]["sha256"]
        or historical_helper.get("classification")
        != "historical_nonconforming_for_R1_MASKED_TOKEN_MEAN_V1; preserved unchanged"
    ):
        raise ValueError("historical R2 helper identity does not match pinned source identity")
    r1_reference = loss_contract.get("r1_reference", {})
    if any(
        r1_reference.get(key) != manifest["source_identities"]["r1_technical_preflight_current"].get(key)
        for key in ("path", "sha256")
    ) or r1_reference.get("callable") != "distillation_loss" or loss_contract.get("contract_tests") != manifest.get("loss_contract_test_identity"):
        raise ValueError("R1 oracle or canonical loss-test source identity drifted")
    objective = manifest["qualified_backend_config"].get("objective", {})
    expected_objective = {
        "contract_id": LOSS_CONTRACT,
        "formula": "0.5*CE_valid_token_mean + 0.5*KL_teacher||student_valid_token_mean",
        "temperature": 2.0,
        "teacher_distribution": "softmax(teacher_logits/tau)",
        "student_distribution": "log_softmax(student_logits/tau)",
        "kl_direction": "teacher||student",
        "kl_reduction": "vocab_sum_then_valid_token_mean",
        "ce_weight": 0.5,
        "kl_weight": 0.5,
        "valid_tokens_per_update": EFFECTIVE_BATCH * WINDOW_TOKENS,
        "valid_token_denominator": "valid_mask.sum(); require > 0",
        "interface": expected_loss_contract["interface"],
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise ValueError("qualified config objective does not match canonical R1 loss contract")
    for identity in manifest["source_identities"].values():
        path = Path(identity["path"])
        if _sha256_file(path) != identity["sha256"]:
            raise ValueError(f"source identity drift: {path}")
    for identity_key in ("input_preparation_runner", "runner_identity", "synthetic_sensitivity_runner", "loss_contract_test_identity", "resume_test_harness_identity"):
        identity = manifest[identity_key]
        if _sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise ValueError(f"qualification tooling identity drift: {identity['path']}")
    for identity in manifest["preflight_tool_identities"].values():
        if _sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise ValueError(f"preflight harness source identity drift: {identity['path']}")
    for identity in manifest["prior_preflight_evidence"]["reports"].values():
        if _sha256_file(Path(identity["path"])) != identity["file_sha256"]:
            raise ValueError(f"prior preflight report identity drift: {identity['path']}")
    basis_identity = manifest["prior_preflight_evidence"]["basis_qualification_manifest"]
    if _sha256_file(Path(basis_identity["path"])) != basis_identity["file_sha256"]:
        raise ValueError("prior basis qualification manifest file drift")
    protocol_identity = manifest["protocol_identity"]
    if _sha256_file(Path(protocol_identity["path"])) != protocol_identity["sha256"]:
        raise ValueError("qualification protocol document identity drift")
    for identity_key in ("impact_audit_identity", "kernel_scope_note_identity"):
        identity = manifest[identity_key]
        if _sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise ValueError(f"qualification scope evidence identity drift: {identity['path']}")
    candidate = manifest["qualified_backend_config"]["dll"]
    if _sha256_file(Path(candidate["path"])) != candidate["sha256"] or candidate["sha256"] != BE376_SHA256:
        raise ValueError("selected be376 DLL identity mismatch")
    for key, path in (("train_manifest_file_sha256", TRAIN_MANIFEST), ("validation_manifest_file_sha256", VALIDATION_MANIFEST)):
        expected = manifest["prepared_artifacts"][key]
        if _sha256_file(path) != expected:
            raise ValueError(f"prepared input drift: {path}")
    train_input = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    train_signature = train_input.get("manifest_sha256")
    train_unsigned = dict(train_input)
    train_unsigned.pop("manifest_sha256", None)
    if train_signature != manifest["training_manifest"]["manifest_sha256"] or train_signature != _canonical_hash(train_unsigned):
        raise ValueError("derived training manifest self-hash mismatch")
    validation_input = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    validation_signature = validation_input.get("manifest_sha256")
    validation_unsigned = dict(validation_input)
    validation_unsigned.pop("manifest_sha256", None)
    if validation_signature != manifest["validation_manifest"]["verification"]["manifest_sha256"] or validation_signature != _canonical_hash(validation_unsigned):
        raise ValueError("derived validation manifest self-hash mismatch")
    validation_verification = manifest["validation_manifest"]["verification"]
    report_path = Path(validation_verification["report_path"])
    if _sha256_file(report_path) != validation_verification["report_file_sha256"]:
        raise ValueError("source 60-document validation report changed")
    for key in ("expanded_validation_runner", "r1_evaluator_runner"):
        source_path = Path(validation_verification[f"{key}_path"])
        if _sha256_file(source_path) != validation_verification[f"{key}_sha256"]:
            raise ValueError(f"60-document validation source drift: {source_path}")
    source_train = manifest["training_manifest"]
    if _sha256_file(Path(source_train["source_path"])) != source_train["source_sha256"]:
        raise ValueError("training source manifest file drift")
    return manifest


def _append_ledger_event(path: Path, event: dict[str, Any]) -> str:
    encoded = _canonical_bytes(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _ledger_rows_and_hashes(path: Path) -> list[tuple[dict[str, Any], str]]:
    if not path.exists():
        return []
    rows: list[tuple[dict[str, Any], str]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.strip():
            continue
        rows.append((json.loads(line), hashlib.sha256(line).hexdigest()))
    return rows


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    return [event for event, _digest in _ledger_rows_and_hashes(path)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    if path.exists():
        raise FileExistsError(f"immutable quality checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    file_hash = _sha256_file(path)
    sidecar = {"checkpoint_path": str(path), "checkpoint_sha256": file_hash, "identity_sha256": _canonical_hash(payload["identity"])}
    _write_json(path.with_suffix(path.suffix + ".identity.json"), sidecar)
    return sidecar


def _load_checkpoint(path: Path, expected_identity: dict[str, Any], ledger_path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(path.suffix + ".identity.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if _sha256_file(path) != sidecar.get("checkpoint_sha256"):
        raise ValueError("checkpoint file hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("identity") != expected_identity or sidecar.get("identity_sha256") != _canonical_hash(expected_identity):
        raise ValueError("checkpoint backend/source/config identity mismatch")
    if not payload.get("last_canonical_ledger_event_sha256"):
        raise ValueError("checkpoint missing canonical ledger tail hash")
    rows = _ledger_rows_and_hashes(ledger_path)
    matching = [index for index, (_event, digest) in enumerate(rows) if digest == payload["last_canonical_ledger_event_sha256"]]
    if not matching:
        raise ValueError("checkpoint ledger tail is absent from append-only ledger")
    payload["ledger_rows_after_checkpoint"] = [event for event, _digest in rows[matching[-1] + 1 :]]
    return payload


def _set_deterministic_cpu_threads() -> None:
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False


def _make_synthetic_model(rounds: int) -> torch.nn.Module:
    class SyntheticR1(torch.nn.Module):
        def __init__(self, k: int) -> None:
            super().__init__()
            self.rounds = k
            self.weight = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.tensor(-0.0625, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(0.75, dtype=torch.float32))

        def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            result = state
            for _ in range(self.rounds):
                result = torch.tanh(inputs + self.weight * result + self.bias * self.scale)
            return result

    return SyntheticR1(rounds)


def _synthetic_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: torch.Tensor, update: int, backend: str) -> tuple[torch.Tensor, dict[str, Any]]:
    window = update % 2
    if window == 0:
        state = torch.zeros_like(state)
    inputs = torch.arange(8, dtype=torch.float32) / 16.0 + torch.rand(8) * 0.01 + float(update) / 100.0
    targets = torch.sin(inputs + 0.25)
    optimizer.zero_grad(set_to_none=True)
    # native is a synthetic dispatch stub; no real C DLL is loaded or called here.
    prediction = model(inputs, state)
    loss = torch.square(prediction - targets).mean()
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("synthetic loss non-finite")
    loss.backward()
    pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item())
    clip_coefficient = min(1.0, GRAD_CLIP_NORM / (pre_clip_norm + 1.0e-6))
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("synthetic parameter non-finite")
    next_state = prediction.detach()
    report = {
        "update": update,
        "backend": backend,
        "synthetic_backend_stub": backend == "native",
        "window": window,
        "loss": float(loss.detach().item()),
        "pre_clip_grad_norm": pre_clip_norm,
        "clip_coefficient": clip_coefficient,
        "clip_intervened": clip_coefficient < 1.0,
        "state_sha256": _tensor_hash(next_state),
        "parameter_sha256": _value_hash(model.state_dict()),
        "optimizer_sha256": _value_hash(optimizer.state_dict()),
    }
    return next_state, report


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _synthetic_identity(backend: str, rounds: int, seed: int) -> dict[str, Any]:
    return {"mode": "synthetic-resume-test", "backend": backend, "backend_impl": "native_stub" if backend == "native" else "pytorch", "K": rounds, "seed": seed, "config": {"physical_batch": 8, "effective_batch": 8, "accumulations": 1, "clip_norm": GRAD_CLIP_NORM}}


def _synthetic_init(identity: dict[str, Any]) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Tensor, dict[str, Any]]:
    seed = int(identity["seed"]) + int(identity["K"])
    random.seed(seed)
    torch.manual_seed(seed)
    model = _make_synthetic_model(int(identity["K"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    state = torch.zeros(8, dtype=torch.float32)
    return model, optimizer, state, {"torch": torch.get_rng_state(), "python": random.getstate()}


def _synthetic_checkpoint(path: Path, identity: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: torch.Tensor, next_update: int, last_event_hash: str) -> dict[str, Any]:
    payload = {
        "schema": "omega-backend-quality-checkpoint-synthetic-v1",
        "identity": identity,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "completed_updates": next_update,
        "data_cursor": {"next_update": next_update, "pair": next_update // 2, "window": next_update % 2},
        "recurrent_state": state.detach().cpu().clone(),
        "last_canonical_ledger_event_sha256": last_event_hash,
        "last_canonical_ledger_event": {"phase": "update_completed", "next_update": next_update},
    }
    sidecar = _save_checkpoint(path, payload)
    return {"payload": payload, "sidecar": sidecar}


def _run_synthetic_child(mode: str, backend: str, rounds: int, seed: int, run_dir: Path, checkpoint: Path | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = run_dir / "events.jsonl"
    identity = _synthetic_identity(backend, rounds, seed)
    model, optimizer, state, initial_rng = _synthetic_init(identity)
    per_update: list[dict[str, Any]] = []
    start_update = 0
    canonical_hash_at_boundary = ""
    if mode in {"tail", "resume"}:
        if checkpoint is None:
            raise ValueError("synthetic tail/resume needs a checkpoint")
        loaded = _load_checkpoint(checkpoint, identity, ledger)
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        state = loaded["recurrent_state"].clone()
        torch.set_rng_state(loaded["torch_rng_state"])
        random.setstate(loaded["python_rng_state"])
        start_update = int(loaded["completed_updates"])
        canonical_hash_at_boundary = loaded["last_canonical_ledger_event_sha256"]
        if mode == "resume" and loaded["ledger_rows_after_checkpoint"]:
            tail_updates = sorted({int(event["update"]) for event in loaded["ledger_rows_after_checkpoint"] if "update" in event})
            _append_ledger_event(ledger, {"run_id": run_dir.name, "phase": "discard_uncheckpointed_tail", "status": "NONCANONICAL", "updates": tail_updates, "parent_checkpoint_event_sha256": canonical_hash_at_boundary})

    end_update = 4 if mode in {"continuous", "resume"} else (3 if mode == "tail" else 2)
    for update in range(start_update, end_update):
        state, report = _synthetic_update(model, optimizer, state, update, backend)
        report["canonical"] = mode != "tail"
        per_update.append(report)
        event = {"run_id": run_dir.name, "phase": "update_completed", "status": "APPLIED", **report}
        event_hash = _append_ledger_event(ledger, event)
        if mode == "segment" and update + 1 == end_update:
            _synthetic_checkpoint(run_dir / "checkpoint_00002.pt", identity, model, optimizer, state, update + 1, event_hash)
        if mode == "tail" and update + 1 == end_update:
            _write_json(run_dir / "interrupted_tail.json", {"simulated_interruption_after_update": update, "checkpoint_resume_origin": str(checkpoint), "canonical": False})

    if mode in {"continuous", "resume"}:
        completed = end_update
        _synthetic_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", identity, model, optimizer, state, completed, event_hash if per_update else canonical_hash_at_boundary)
    canonical_per_update = [
        event for event, _digest in _ledger_rows_and_hashes(ledger)
        if event.get("phase") == "update_completed" and event.get("canonical", True)
    ]
    final = {
        "mode": mode,
        "backend": backend,
        "backend_impl": "native_stub" if backend == "native" else "pytorch",
        "K": rounds,
        "seed": seed,
        "completed_updates": end_update,
        "data_cursor": {"next_update": end_update, "pair": end_update // 2, "window": end_update % 2},
        "model_sha256": _value_hash(model.state_dict()),
        "optimizer_sha256": _value_hash(optimizer.state_dict()),
        "torch_rng_sha256": _tensor_hash(torch.get_rng_state()),
        "python_rng_sha256": _canonical_hash(repr(random.getstate())),
        "recurrent_state_sha256": _tensor_hash(state),
        "per_update": canonical_per_update,
        "fresh_process_pid": os.getpid(),
    }
    _write_json(run_dir / f"{mode}_report.json", final)


def _synthetic_resume_check() -> dict[str, Any]:
    run_stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_root = RESULTS_DIR / "preparation" / "synthetic_resume" / run_stamp
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for backend in ("pytorch", "native"):
        for rounds in K_VALUES:
            seed = 88000 + rounds
            case = output_root / backend / f"K{rounds}"
            continuous_dir = case / "continuous"
            resumed_dir = case / "segmented"
            for path in (continuous_dir, resumed_dir):
                if path.exists():
                    raise FileExistsError(f"synthetic resume evidence is immutable; output exists: {path}")
                path.mkdir(parents=True, exist_ok=True)
            command_base = [sys.executable, "-B", str(Path(__file__).resolve()), "--_synthetic-child", "--backend", backend, "--K", str(rounds), "--seed", str(seed)]
            subprocess.run(command_base + ["--mode", "continuous", "--run-dir", str(continuous_dir)], check=True, capture_output=True, text=True)
            subprocess.run(command_base + ["--mode", "segment", "--run-dir", str(resumed_dir)], check=True, capture_output=True, text=True)
            checkpoint = resumed_dir / "checkpoint_00002.pt"
            wrong_backend = "native" if backend == "pytorch" else "pytorch"
            wrong_identity_rejected = False
            try:
                _load_checkpoint(checkpoint, _synthetic_identity(wrong_backend, rounds, seed), resumed_dir / "events.jsonl")
            except ValueError:
                wrong_identity_rejected = True
            if not wrong_identity_rejected:
                raise AssertionError(f"resume checkpoint accepted wrong backend identity for {backend} K{rounds}")
            subprocess.run(command_base + ["--mode", "tail", "--run-dir", str(resumed_dir), "--checkpoint", str(checkpoint)], check=True, capture_output=True, text=True)
            subprocess.run(command_base + ["--mode", "resume", "--run-dir", str(resumed_dir), "--checkpoint", str(checkpoint)], check=True, capture_output=True, text=True)
            continuous = json.loads((continuous_dir / "continuous_report.json").read_text(encoding="utf-8"))
            resumed = json.loads((resumed_dir / "resume_report.json").read_text(encoding="utf-8"))
            if (continuous["model_sha256"], continuous["optimizer_sha256"], continuous["torch_rng_sha256"], continuous["python_rng_sha256"], continuous["recurrent_state_sha256"], continuous["data_cursor"]) != (
                resumed["model_sha256"], resumed["optimizer_sha256"], resumed["torch_rng_sha256"], resumed["python_rng_sha256"], resumed["recurrent_state_sha256"], resumed["data_cursor"]
            ):
                raise AssertionError(f"continuous/resume state mismatch for backend={backend} K={rounds}")
            continuous_hashes = [row["parameter_sha256"] for row in continuous["per_update"]]
            resumed_hashes = [row["parameter_sha256"] for row in resumed["per_update"] if row.get("canonical")]
            if continuous_hashes != resumed_hashes:
                raise AssertionError(f"continuous/resume per-update trajectory mismatch for backend={backend} K={rounds}")
            segmented_events = _read_ledger(resumed_dir / "events.jsonl")
            if not any(event.get("phase") == "discard_uncheckpointed_tail" and event.get("status") == "NONCANONICAL" for event in segmented_events):
                raise AssertionError("synthetic interrupted tail was not marked noncanonical")
            rows.append({"backend": backend, "backend_impl": "native_stub" if backend == "native" else "pytorch", "K": rounds, "seed": seed, "continuous_vs_resume_exact": True, "replayed_tail_marked_noncanonical": True, "wrong_backend_checkpoint_rejected": wrong_identity_rejected, "continuous_pid": continuous["fresh_process_pid"], "resumed_pid": resumed["fresh_process_pid"]})
    report = {
        "schema": "omega-backend-quality-synthetic-resume-v1",
        "synthetic_only": True,
        "real_training": False,
        "real_dll_called": False,
        "cases": rows,
        "status": "PASS" if len(rows) == 4 else "FAIL",
    }
    output = output_root / "synthetic_resume_check.json"
    _write_json(output, report)
    return {"report": report, "path": str(output)}


def _load_real_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    sys.path.insert(0, str(P2R0))
    import run_omega_native_runtime_r2_benchmark as r2  # noqa: PLC0415

    p0, ce, bridge = r2._load_r1_modules()
    sys.path.insert(0, str(R1_SCOPE))
    import run_scientific_scoping_a as r1  # noqa: PLC0415
    sys.path.insert(0, str(EXPANDED_VALIDATION))
    import run_omega_expanded_frozen_validation as expanded  # noqa: PLC0415
    return r2, p0, ce, bridge, (r1, expanded)


def _state_dict_hash(state: dict[str, Any]) -> str:
    return _value_hash(state)


def _prepare_common_initialization(ce: Any, seed: int, rounds: int, output_dir: Path) -> tuple[Path, str]:
    model = ce.fresh_model(seed, rounds)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    state_hash = _state_dict_hash(state)
    recurrent_initial = model.initial_state(EFFECTIVE_BATCH, device=torch.device("cpu")).detach().cpu().clone()
    init_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    empty_optimizer_state = copy.deepcopy(init_optimizer.state_dict())
    historical = R1_SCOPE / "results" / "full_campaign" / "runs" / f"shared_K{rounds}_seed_{seed}" / "checkpoint_00000.pt"
    oracle_result: dict[str, Any] = {"available": historical.is_file(), "path": str(historical)}
    if historical.is_file():
        payload = torch.load(historical, map_location="cpu", weights_only=False)
        if int(payload.get("update", -1)) != 0 or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"historical update-0 checkpoint identity mismatch: {historical}")
        old_state = payload.get("model")
        oracle_result["state_dict_exact"] = isinstance(old_state, dict) and all(
            name in old_state and torch.equal(state[name], old_state[name]) for name in state
        ) and list(state) == list(old_state)
        oracle_result["historical_implementation"] = payload.get("implementation_identity")
        if not oracle_result["state_dict_exact"]:
            raise ValueError(f"fresh paired initialization differs from historical R1@0 oracle: {historical}")
    bundle = {
        "schema": "omega-backend-quality-common-init-v1",
        "seed": seed,
        "K": rounds,
        "architecture": "R1 shared",
        "model_state": state,
        "model_state_sha256": state_hash,
        "model_state_structure": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in state.items()},
        "initial_recurrent_state": recurrent_initial,
        "initial_recurrent_state_sha256": _tensor_hash(recurrent_initial),
        "fresh_adamw_state_sha256": _value_hash(empty_optimizer_state),
        "fresh_adamw_state_empty": not bool(empty_optimizer_state["state"]),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "optimizer": "fresh AdamW state for each backend arm; empty at update0",
        "historical_update0_oracle": oracle_result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"common_init_K{rounds}_seed_{seed}.pt"
    if path.exists():
        raise FileExistsError(f"immutable common initialization already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return path, _sha256_file(path)


def _quality_identity(manifest: dict[str, Any], block: str, backend: str, rounds: int, seed: int, run_id: str, init_bundle_path: Path, init_bundle_sha256: str, initial_state_sha256: str, *, target_updates: int, checkpoint_interval: int, technical_resume_test: bool) -> dict[str, Any]:
    return {
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "block": block,
        "backend": backend,
        "K": rounds,
        "seed": seed,
        "run_id": run_id,
        "common_init_bundle_path": str(init_bundle_path),
        "common_init_bundle_sha256": init_bundle_sha256,
        "common_initial_state_sha256": initial_state_sha256,
        "selected_native_dll_sha256": BE376_SHA256 if backend == "native" else None,
        "r1_architecture_commit": "269a4d79b5a1e6df8c230962e2b8c9e237095f18",
        "training_config": manifest["qualified_backend_config"],
        "training_manifest_sha256": manifest["training_manifest"]["manifest_sha256"],
        "validation_manifest_sha256": manifest["validation_manifest"]["verification"]["manifest_sha256"],
        "execution_contract": {
            "target_updates": target_updates,
            "checkpoint_interval": checkpoint_interval,
            "technical_resume_test": technical_resume_test,
        },
    }


def _append_quality_event(path: Path, event: dict[str, Any]) -> str:
    encoded = _canonical_bytes(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _quality_checkpoint(path: Path, *, identity: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, recurrent_state: torch.Tensor, completed_updates: int, next_update: int, pair_cursor: int, next_window: int, last_event: dict[str, Any], last_event_hash: str, validation_evaluation: dict[str, Any], segment: int) -> str:
    if path.exists():
        raise FileExistsError(f"quality checkpoints are immutable: {path}")
    payload = {
        "schema": "omega-backend-quality-checkpoint-v1",
        "identity": identity,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng_states": {"torch": torch.get_rng_state(), "python": random.getstate()},
        "completed_updates": completed_updates,
        "next_update": next_update,
        "data_cursor": {"next_update": next_update, "pair": pair_cursor, "window": next_window},
        "recurrent_state": recurrent_state.detach().cpu().clone(),
        "last_canonical_ledger_event": last_event,
        "last_canonical_ledger_event_sha256": last_event_hash,
        "validation_evaluation": validation_evaluation,
        "execution_segment": segment,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    digest = _sha256_file(path)
    _write_json(path.with_suffix(path.suffix + ".identity.json"), {"path": str(path), "sha256": digest, "identity_sha256": _canonical_hash(identity)})
    return digest


def _quality_run_child(
    block: str,
    backend: str,
    rounds: int,
    seed: int,
    init_path: Path,
    init_sha: str,
    run_dir: Path,
    manifest: dict[str, Any],
    resume_path: Path | None = None,
    *,
    run_until_update: int = UPDATES_PER_RUN,
    target_updates: int = UPDATES_PER_RUN,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    technical_resume_test: bool = False,
    technical_resume_phase: str = "scientific",
) -> dict[str, Any]:
    if not 0 < run_until_update <= target_updates <= UPDATES_PER_RUN:
        raise ValueError("run/target update bounds violate qualification budget")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if technical_resume_test:
        if block != "RESUME_CHECK" or target_updates != 4 or checkpoint_interval != 3:
            raise ValueError("technical resume test identity must be RESUME_CHECK/4 updates/interval 3")
        if run_until_update not in (3, 4) or technical_resume_phase not in ("continuous", "prefix", "resume"):
            raise ValueError("technical resume test segment bounds/phase invalid")
    elif (run_until_update, target_updates, checkpoint_interval) != (UPDATES_PER_RUN, UPDATES_PER_RUN, CHECKPOINT_INTERVAL):
        raise ValueError("scientific runs must use the frozen 2,000-update/500-checkpoint plan")
    qualified_config = manifest["qualified_backend_config"]
    if qualified_config.get("instrumentation") != 0 or qualified_config.get("profile_compile_and_runtime") is not False:
        raise ValueError("quality run must use diagnostics-instrumentation=0 and profile disabled")
    r2, p0, ce, bridge, modules = _load_real_dependencies()
    r1, expanded = modules
    policy = ce.validate_policy()
    if int(policy["intraop_threads"]) != TORCH_THREADS or int(policy["interop_threads"]) != TORCH_INTEROP_THREADS:
        raise RuntimeError("PyTorch thread policy mismatch")
    docs, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    train_manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    expected_keys = [
        (item["full_text_sha256"], item["retained_513_token_sha256"])
        for item in train_manifest["documents"]
    ]
    actual_keys = [
        (item["full_text_sha256"], item["retained_513_token_sha256"])
        for item in docs
    ]
    if actual_keys != expected_keys:
        raise ValueError("training document identity/order differs from derived manifest")

    init_bundle = torch.load(init_path, map_location="cpu", weights_only=False)
    if _sha256_file(init_path) != init_sha or int(init_bundle["seed"]) != seed or int(init_bundle["K"]) != rounds:
        raise ValueError("common initialization bundle identity mismatch")
    model = ce.fresh_model(seed, rounds)
    model.load_state_dict(init_bundle["model_state"], strict=True)
    if _state_dict_hash(model.state_dict()) != init_bundle["model_state_sha256"]:
        raise ValueError("loaded model differs from common initialization snapshot")
    torch.set_rng_state(init_bundle["torch_rng_state"])
    random.setstate(init_bundle["python_rng_state"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS, eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY)
    if optimizer.state or _value_hash(optimizer.state_dict()) != init_bundle["fresh_adamw_state_sha256"]:
        raise AssertionError("new backend arm must start with empty AdamW state")

    if backend == "native":
        bridge.configure_library(BE376_DLL)
        if not bridge.runtime_abi_available():
            raise RuntimeError("be376 is missing production runtime ABI")
        bridge.configure_runtime(NATIVE_WORKERS)
    state = model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
    if _tensor_hash(state) != init_bundle["initial_recurrent_state_sha256"]:
        raise ValueError("backend arm recurrent initial state differs from common initialization bundle")
    state_part_weight = r2.prepare_native_model(model) if backend == "native" else None
    if technical_resume_test:
        validation_docs = None
    else:
        validation_docs, validation_manifest, _tokenizer = expanded._load_real_validation()
        if validation_manifest["manifest_sha256"] != manifest["validation_manifest"]["verification"]["manifest_sha256"]:
            raise ValueError("common 60-document evaluation manifest drift")

    run_id = f"block{block}_{backend}_K{rounds}_seed_{seed}"
    identity = _quality_identity(
        manifest, block, backend, rounds, seed, run_id, init_path, init_sha,
        init_bundle["model_state_sha256"], target_updates=target_updates,
        checkpoint_interval=checkpoint_interval, technical_resume_test=technical_resume_test,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    validation_curve_path = run_dir / "validation_curve.jsonl"
    if resume_path is None and (ledger_path.exists() or any(run_dir.iterdir())):
        raise FileExistsError(f"quality run directory is not empty: {run_dir}")
    segment = 0
    start_update = 0
    if resume_path is not None:
        checkpoint_meta = json.loads(resume_path.with_suffix(resume_path.suffix + ".identity.json").read_text(encoding="utf-8"))
        if _sha256_file(resume_path) != checkpoint_meta["sha256"] or checkpoint_meta["identity_sha256"] != _canonical_hash(identity):
            raise ValueError("resume checkpoint hash/identity mismatch")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("resume checkpoint belongs to another backend/K/seed/config")
        if int(saved["completed_updates"]) != int(saved["next_update"]) or int(saved["next_update"]) % checkpoint_interval != 0:
            raise ValueError("resume checkpoint is not on the configured verified checkpoint boundary")
        expected_cursor = {"next_update": int(saved["next_update"]), "pair": int(saved["next_update"]) // 2, "window": int(saved["next_update"]) % 2}
        if saved.get("data_cursor") != expected_cursor:
            raise ValueError("resume data cursor/causal boundary mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_states"]["torch"])
        random.setstate(saved["rng_states"]["python"])
        state = saved["recurrent_state"].clone()
        start_update = int(saved["next_update"])
        segment = int(saved["execution_segment"]) + 1
        rows = _read_ledger(ledger_path)
        hashes = [hashlib.sha256(_canonical_bytes(row)).hexdigest() for row in rows]
        if saved["last_canonical_ledger_event_sha256"] not in hashes:
            raise ValueError("resume checkpoint canonical ledger event is missing")
        parent_index = hashes.index(saved["last_canonical_ledger_event_sha256"])
        if rows[parent_index] != saved.get("last_canonical_ledger_event"):
            raise ValueError("resume checkpoint canonical ledger event payload mismatch")
        noncanonical_rows = rows[parent_index + 1 :]
        if noncanonical_rows:
            _append_quality_event(ledger_path, {
                "run_id": run_id, "backend": backend, "K": rounds, "seed": seed,
                "update": start_update, "phase": "uncheckpointed_tail_abandoned",
                "status": "NONCANONICAL", "tail_event_count": len(noncanonical_rows),
                "parent_checkpoint_event_sha256": saved["last_canonical_ledger_event_sha256"],
                "execution_segment": segment,
            })
        current_event = {
            "run_id": run_id, "variant": f"shared_K{rounds}", "backend": backend, "K": rounds, "seed": seed,
            "update": start_update, "phase": "resume_segment_started", "status": "APPLIED",
            "execution_segment": segment, "parent_checkpoint_sha256": _sha256_file(resume_path),
            "data_cursor": saved["data_cursor"], "elapsed_seconds": 0.0,
            "memory": {"rss_bytes": r2._rss_bytes()}, "microbatch": "full",
            "document_id": [], "input_range": [0, 0], "target_range": [0, 0],
            "teacher_context_range": [0, 0], "window": int(saved["data_cursor"]["window"]),
            "state_reset": int(saved["data_cursor"]["window"]) == 0,
            "state_source_update": None if int(saved["data_cursor"]["window"]) == 0 else start_update - 1,
            "valid_tokens": 0,
            "restored_model_state_sha256": _value_hash(model.state_dict()),
            "restored_optimizer_state_sha256": _value_hash(optimizer.state_dict()),
            "restored_recurrent_state_sha256": _tensor_hash(state),
            "restored_torch_rng_sha256": _tensor_hash(torch.get_rng_state()),
        }
        current_event_hash = _append_quality_event(ledger_path, current_event)

    pair_schedule = train_manifest["pairs"]
    if not (len(pair_schedule) == 1000 and start_update <= target_updates and run_until_update <= len(pair_schedule) * 2):
        raise ValueError("quality schedule bounds mismatch")
    eval_records = _read_jsonl(validation_curve_path)
    current_event: dict[str, Any]
    current_event_hash: str

    def evaluate_boundary(completed: int) -> None:
        if technical_resume_test:
            record = {
                "update": completed,
                "evaluated": False,
                "reason": "technical_resume_contract_test; no validation/test evaluation",
            }
            prior = next((item for item in eval_records if int(item["update"]) == completed), None)
            if prior is not None and prior != record:
                raise ValueError(f"technical resume checkpoint metadata differs at {completed}")
            if prior is None:
                eval_records.append(record)
            return
        model.eval()
        metric = r1.evaluate_validation(model, validation_docs)
        if not bool(metric.get("finite")) or int(metric.get("tokens", 0)) <= 0:
            raise FloatingPointError("common reference evaluator produced invalid NLL")
        record = {"update": completed, "nll": float(metric["nll"]), "tokens": int(metric["tokens"]), "evaluator": "pure-PyTorch-R1-F", "canonical": True}
        prior = next((item for item in eval_records if int(item["update"]) == completed), None)
        if prior is not None:
            if any(prior.get(key) != value for key, value in record.items()):
                raise ValueError(f"validation curve boundary {completed} differs from checkpoint recomputation")
        else:
            eval_records.append(record)
        model.train()

    if resume_path is None:
        current_event = {
            "run_id": run_id, "variant": f"shared_K{rounds}", "backend": backend, "K": rounds, "seed": seed,
            "update": 0, "phase": "initialization_verified", "status": "APPLIED", "execution_segment": segment,
            "microbatch": "full", "elapsed_seconds": 0.0,
            "memory": {"rss_bytes": r2._rss_bytes()},
            "document_id": [], "input_range": [0, 0], "target_range": [0, 0],
            "teacher_context_range": [0, 0], "window": 0,
            "state_reset": True, "state_source_update": None, "valid_tokens": 0,
            "model_state_sha256": init_bundle["model_state_sha256"], "optimizer_state_empty": True,
            "data_cursor": {"next_update": 0, "pair": 0, "window": 0},
        }
        current_event_hash = _append_quality_event(ledger_path, current_event)
        evaluate_boundary(0)
        checkpoint_hash = _quality_checkpoint(run_dir / "checkpoint_00000.pt", identity=identity, model=model, optimizer=optimizer,
            recurrent_state=state, completed_updates=0, next_update=0, pair_cursor=0, next_window=0,
            last_event=current_event, last_event_hash=current_event_hash, validation_evaluation=eval_records[-1], segment=segment)
        _append_quality_event(validation_curve_path, {**eval_records[-1], "checkpoint_sha256": checkpoint_hash})
    else:
        saved_update = int(start_update)
        saved_eval = next((item for item in eval_records if int(item["update"]) == saved_update), None)
        if saved_eval is None:
            saved_eval = saved.get("validation_evaluation")
            if not isinstance(saved_eval, dict) or int(saved_eval.get("update", -1)) != saved_update:
                raise ValueError("resume checkpoint missing common validation result at its boundary")
            eval_records.append(saved_eval)
            checkpoint_meta = json.loads(resume_path.with_suffix(resume_path.suffix + ".identity.json").read_text(encoding="utf-8"))
            _append_quality_event(validation_curve_path, {**saved_eval, "checkpoint_sha256": checkpoint_meta["sha256"], "recovered_from_checkpoint": True})

    for update in range(start_update, run_until_update):
        pair_index = update // 2
        window = update % 2
        if window == 0:
            state = model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
        elif update == 0 or update % 2 != 1:
            raise AssertionError("causal state cursor invalid")
        pair = pair_schedule[pair_index]
        positions = [int(index) for index in pair["document_indices"]]
        source = torch.tensor([docs[position]["tokens"] for position in positions], dtype=torch.long)
        offset = window * WINDOW_TOKENS
        inputs = source[:, offset : offset + WINDOW_TOKENS]
        targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
        teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, window)
        previous_state = state.detach() if window == 1 else state
        update_started = time.perf_counter()
        if backend == "native":
            next_state, student_logits, _readouts = r2._native_forward(model, inputs, previous_state, state_part_weight, bridge)
        else:
            next_state, student_logits, _readouts = r2._reference_forward(model, inputs, previous_state, p0)
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        loss_terms = r1_masked_token_mean_loss(student_logits, teacher_logits, targets, valid_mask)
        expected_valid_tokens = int(qualified_config["objective"]["valid_tokens_per_update"])
        if loss_terms["valid_tokens"] != expected_valid_tokens:
            raise ValueError(
                f"valid target count {loss_terms['valid_tokens']} differs from contract {expected_valid_tokens}"
            )
        total_loss = loss_terms["total"]
        if not bool(torch.isfinite(total_loss).all()):
            raise FloatingPointError(f"non-finite loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite gradient before clip at update {update}")
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
        clip_coefficient = min(1.0, ce.CLIP_NORM / (pre_clip_norm + 1.0e-6))
        if not math.isfinite(pre_clip_norm):
            raise FloatingPointError(f"non-finite pre-clip norm at update {update}")
        common_event = {
            "run_id": run_id, "variant": f"shared_K{rounds}", "backend": backend, "K": rounds, "seed": seed,
            "update": update, "microbatch": "full", "execution_segment": segment,
            "window": window, "pair": pair_index, "document_positions": positions,
            "document_id": [
                [docs[position]["full_text_sha256"], docs[position]["retained_513_token_sha256"]]
                for position in positions
            ],
            "input_range": [offset, offset + WINDOW_TOKENS],
            "target_range": [offset + 1, offset + WINDOW_TOKENS + 1],
            "teacher_context_range": [0, 256] if window == 0 else [0, 512],
            "valid_tokens": loss_terms["valid_tokens"],
            "loss_contract_id": LOSS_CONTRACT,
            "loss_callable": r1_masked_token_mean_loss.__name__,
            "loss_implementation_sha256": _sha256_file(Path(r1_masked_token_mean_loss.__code__.co_filename)),
            "memory": {"rss_bytes": r2._rss_bytes()},
            "state_reset": window == 0, "state_source_update": None if window == 0 else update - 1,
            "effective_batch": EFFECTIVE_BATCH, "physical_batch": PHYSICAL_BATCH, "accumulations": ACCUMULATIONS,
            "total_loss": float(total_loss.detach().item()), "ce": float(loss_terms["ce"].detach().item()),
            "kl": float(loss_terms["kl"].detach().item()), "pre_clip_grad_norm": pre_clip_norm,
            "clip_coefficient": clip_coefficient, "clip_intervened": clip_coefficient < 1.0,
            "elapsed_seconds": time.perf_counter() - update_started,
        }
        _append_quality_event(ledger_path, {**common_event, "phase": "optimizer_step_started", "status": "STARTED", "adamw_steps": 0})
        try:
            optimizer.step()
        except Exception:
            _append_quality_event(ledger_path, {**common_event, "phase": "optimizer_step_started", "status": "UNCERTAIN", "adamw_steps": 0})
            raise
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            _append_quality_event(ledger_path, {**common_event, "phase": "optimizer_step_completed", "status": "APPLIED_NONFINITE", "adamw_steps": 1})
            raise FloatingPointError(f"non-finite parameter after optimizer at update {update}")
        state = next_state.detach()
        common_event["elapsed_seconds"] = time.perf_counter() - update_started
        current_event = {**common_event, "phase": "update_completed", "status": "APPLIED", "adamw_steps": 1}
        current_event_hash = _append_quality_event(ledger_path, current_event)
        completed = update + 1
        if completed % checkpoint_interval == 0 or completed == run_until_update:
            evaluate_boundary(completed)
            checkpoint_hash = _quality_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", identity=identity, model=model, optimizer=optimizer,
                recurrent_state=state, completed_updates=completed, next_update=completed, pair_cursor=completed // 2,
                next_window=completed % 2, last_event=current_event, last_event_hash=current_event_hash,
                validation_evaluation=eval_records[-1], segment=segment)
            _append_quality_event(validation_curve_path, {**eval_records[-1], "checkpoint_sha256": checkpoint_hash})

    technical_complete = technical_resume_test and run_until_update == target_updates
    report = {
        "schema": "omega-backend-quality-run-report-v1",
        "status": ("TECHNICAL_COMPLETE" if technical_complete else "TECHNICAL_SEGMENT_COMPLETE") if technical_resume_test else "COMPLETE",
        "block": block,
        "run_id": run_id, "backend": backend, "K": rounds, "seed": seed,
        "updates": run_until_update, "target_updates": target_updates,
        "checkpoint_interval": checkpoint_interval, "technical_resume_test": technical_resume_test,
        "technical_resume_phase": technical_resume_phase if technical_resume_test else None,
        "scientific_quality_training": not technical_resume_test,
        "physical_batch": PHYSICAL_BATCH, "effective_batch": EFFECTIVE_BATCH,
        "accumulations": ACCUMULATIONS, "native_workers": NATIVE_WORKERS if backend == "native" else None,
        "torch_intraop": TORCH_THREADS, "torch_interop": TORCH_INTEROP_THREADS,
        "instrumentation": 0, "diagnostic_statistics": "null/disabled", "profile": False,
        "endpoint": (f"technical_update_{run_until_update}" if technical_resume_test else "update_2000"),
        "endpoint_nll_validation": eval_records[-1],
        "loss_contract": manifest["loss_contract"],
        "validation_curve": eval_records, "test_split_loaded": False,
        "evaluator": None if technical_resume_test else "common pure-PyTorch R1/F reference; native DLL never called for scoring",
        "dll_sha256": BE376_SHA256 if backend == "native" else None,
        "identity": identity, "run_dir": str(run_dir),
    }
    report_name = f"technical_resume_{technical_resume_phase}_report.json" if technical_resume_test else "run_report.json"
    _write_json(run_dir / report_name, report)
    return report


def _load_authorization(path: Path, block: str, manifest: dict[str, Any]) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    expected_state = {"status": "APPROVED", "block": block, "candidate_sha256": BE376_SHA256, "qualification_manifest_sha256": manifest["manifest_sha256"]}
    if any(authorization.get(key) != value for key, value in expected_state.items()):
        raise PermissionError("authorization file does not approve this exact block/candidate/manifest")
    return authorization


def _validate_block_a_preflight_report(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError("Block A requires sealed protocol/identity and real-runner resume preflight report")
    report = json.loads(path.read_text(encoding="utf-8"))
    signature = report.get("report_self_sha256")
    unsigned = dict(report)
    unsigned.pop("report_self_sha256", None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError("Block A preflight report self-hash mismatch")
    resume_path = Path(report.get("resume_test_report_path", ""))
    if not resume_path.is_file() or _sha256_file(resume_path) != report.get("resume_test_report_sha256"):
        raise ValueError("Block A real-runner resume report file identity mismatch")
    resume_report = json.loads(resume_path.read_text(encoding="utf-8"))
    resume_unsigned = dict(resume_report)
    resume_signature = resume_unsigned.pop("report_self_sha256", None)
    if (
        not resume_signature
        or resume_signature != report.get("resume_test_report_self_sha256")
        or resume_signature != _canonical_hash(resume_unsigned)
    ):
        raise ValueError("Block A real-runner resume report self-hash mismatch")
    routes = resume_report.get("routes", [])
    expected_routes = {(backend, rounds) for backend in ("pytorch", "native") for rounds in K_VALUES}
    observed_routes = {(row.get("backend"), int(row.get("K", -1))) for row in routes}
    if (
        report.get("status") != "PASS"
        or report.get("block") != "A"
        or report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]
        or report.get("scientific_block_started") is not False
        or report.get("technical_resume_updates") != 32
        or report.get("block_a_run_count") != 8
        or report.get("block_a_total_updates") != 16000
        or resume_report.get("status") != "PASS"
        or resume_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]
        or resume_report.get("real_optimizer_updates") != 32
        or resume_report.get("test_or_validation_loaded") is not False
        or observed_routes != expected_routes
        or len(routes) != 4
        or any(row.get("route_pass") is not True for row in routes)
    ):
        raise PermissionError("Block A preflight conditions are incomplete or not PASS")
    if report.get("real_dll_sha256") != BE376_SHA256 or report.get("loss_source_sha256") != manifest["loss_contract"]["implementation"]["sha256"]:
        raise PermissionError("Block A preflight DLL/loss identity differs from sealed manifest")
    return report


def _finalize_block(block: str, authorization: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    run_root = RESULTS_DIR / f"block_{block}"
    output = run_root / f"block_{block}_report.json"
    if output.exists():
        raise FileExistsError(f"block reports are immutable: {output}")
    seeds = BLOCK_SEEDS[block]
    runs: list[dict[str, Any]] = []
    init_hashes: dict[tuple[int, int], set[str]] = {}
    for seed in seeds:
        for rounds in K_VALUES:
            for backend in ("pytorch", "native"):
                run_dir = run_root / f"{backend}_K{rounds}_seed_{seed}"
                report_path = run_dir / "run_report.json"
                if not report_path.is_file():
                    raise FileNotFoundError(f"required quality arm missing: {report_path}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("status") != "COMPLETE" or report.get("block") != block:
                    raise ValueError(f"quality arm incomplete or wrong block: {report_path}")
                if (report.get("backend"), int(report.get("K", -1)), int(report.get("seed", -1)), int(report.get("updates", -1))) != (backend, rounds, seed, UPDATES_PER_RUN):
                    raise ValueError(f"quality arm identity/update mismatch: {report_path}")
                if report.get("identity", {}).get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
                    raise ValueError(f"quality arm manifest identity mismatch: {report_path}")
                if report.get("test_split_loaded") is not False or int(report.get("endpoint_nll_validation", {}).get("update", -1)) != UPDATES_PER_RUN:
                    raise ValueError(f"quality arm endpoint/test-split policy mismatch: {report_path}")
                init_hashes.setdefault((seed, rounds), set()).add(str(report["identity"].get("common_initial_state_sha256", "")))
                runs.append({"backend": backend, "K": rounds, "seed": seed, "run_report_path": str(report_path), "run_report_sha256": _sha256_file(report_path), "run_report": report})
    if any(len(hashes) != 1 or "" in hashes for hashes in init_hashes.values()):
        raise ValueError("PyTorch/native arms did not share identical common initialization per seed/K")
    report = {
        "schema": "omega-backend-quality-block-report-v1",
        "block": block,
        "status": "COMPLETE",
        "authorization": authorization,
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(seeds),
        "K": list(K_VALUES),
        "updates_per_run": UPDATES_PER_RUN,
        "run_count": len(runs),
        "total_updates": len(runs) * UPDATES_PER_RUN,
        "common_initialization_hashes": {f"seed_{seed}_K{rounds}": next(iter(hashes)) for (seed, rounds), hashes in init_hashes.items()},
        "runs": runs,
        "formal_tost": "NOT_RUN_UNTIL_ALL_FIVE_PAIRED_SEEDS_COMPLETE",
        "test_split_loaded": False,
    }
    _write_json(output, report)
    report["report_path"] = str(output)
    return report


def _run_real_block(block: str, authorization_path: Path, block_a_report: Path | None, block_a_preflight_report: Path | None) -> dict[str, Any]:
    manifest = _load_qualification_manifest()
    preflight = _validate_block_a_preflight_report(block_a_preflight_report, manifest) if block == "A" and block_a_preflight_report is not None else None
    if block == "A" and preflight is None:
        raise PermissionError("Block A is held until the sealed preflight report passes")
    authorization = _load_authorization(authorization_path, block, manifest)
    if block == "A" and authorization.get("preflight_report_sha256") != _sha256_file(block_a_preflight_report):
        raise PermissionError("Block A authorization does not bind the verified preflight report")
    if block == "B":
        if block_a_report is None or not block_a_report.is_file():
            raise PermissionError("Block B requires an independently reviewed Block A report and separate authorization")
        a_report = json.loads(block_a_report.read_text(encoding="utf-8"))
        if (
            a_report.get("block") != "A"
            or a_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]
            or a_report.get("status") != "COMPLETE"
            or a_report.get("seeds") != list(BLOCK_SEEDS["A"])
            or a_report.get("run_count") != 8
            or a_report.get("total_updates") != 16000
        ):
            raise PermissionError("Block A report identity/status mismatch; Block B cannot start")
    seeds = BLOCK_SEEDS[block]
    r2, _p0, ce, _bridge, _modules = _load_real_dependencies()
    init_root = RESULTS_DIR / f"block_{block}" / "common_initializations"
    init_rows: dict[tuple[int, int], tuple[Path, str]] = {}
    for seed in seeds:
        for rounds in K_VALUES:
            init_rows[(seed, rounds)] = _prepare_common_initialization(ce, seed, rounds, init_root)
    runs: list[dict[str, Any]] = []
    run_root = RESULTS_DIR / f"block_{block}"
    run_root.mkdir(parents=True, exist_ok=True)
    process_environment = os.environ.copy()
    for name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
        process_environment.pop(name, None)
    for seed_index, seed in enumerate(seeds):
        for rounds in K_VALUES:
            pytorch_first = (seed_index + (rounds == 4)) % 2 == 0
            backends = ("pytorch", "native") if pytorch_first else ("native", "pytorch")
            init_path, init_sha = init_rows[(seed, rounds)]
            for backend in backends:
                run_dir = run_root / f"{backend}_K{rounds}_seed_{seed}"
                command = [
                    sys.executable, "-B", str(Path(__file__).resolve()),
                    "--_real-child", "--block", block, "--backend", backend,
                    "--K", str(rounds), "--seed", str(seed),
                    "--init-bundle", str(init_path), "--init-sha256", init_sha,
                    "--run-dir", str(run_dir), "--authorization-file", str(authorization_path),
                    "--confirm-real-training",
                ]
                if block == "A":
                    command.extend(["--block-a-preflight-report", str(block_a_preflight_report)])
                if block == "B":
                    command.append("--confirm-block-b")
                    command.extend(["--block-a-report", str(block_a_report)])
                try:
                    subprocess.run(command, cwd=REPO_ROOT, env=process_environment, check=True)
                except subprocess.CalledProcessError as error:
                    _write_json(run_root / f"block_{block}_INCOMPLETE.json", {
                        "schema": "omega-backend-quality-block-report-v1",
                        "block": block,
                        "status": "INCOMPLETE",
                        "qualification_manifest_sha256": manifest["manifest_sha256"],
                        "completed_runs": runs,
                        "failed_run": {"backend": backend, "K": rounds, "seed": seed, "exit_code": error.returncode, "run_dir": str(run_dir)},
                        "automatic_retry": False,
                    })
                    raise
                report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
                runs.append({"backend": backend, "K": rounds, "seed": seed, "run_report": report})
    return _finalize_block(block, authorization, manifest)


def _paired_t_summary(values: list[float], margin: float) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError("formal paired TOST requires all five paired seeds")
    mean = sum(values) / len(values)
    standard_deviation = float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item())
    half_width = 2.131846786326383 * standard_deviation / math.sqrt(5.0)
    lower = mean - half_width
    upper = mean + half_width
    equivalent = lower > -margin and upper < margin
    outside = lower >= margin or upper <= -margin
    status = "EQUIVALENT" if equivalent else ("NOT_EQUIVALENT" if outside else "INCONCLUSIVE")
    return {
        "seed_values": values,
        "n": 5,
        "mean": mean,
        "sample_sd": standard_deviation,
        "t_critical_90_two_sided_df4": 2.131846786326383,
        "ci90": [lower, upper],
        "equivalence_margin": [-margin, margin],
        "status": status,
    }


def _formal_five_seed_analysis(block_a_path: Path, block_b_path: Path) -> dict[str, Any]:
    output_path = RESULTS_DIR / "formal_quality_analysis.json"
    if output_path.exists():
        raise FileExistsError(f"formal five-seed analysis is immutable and already exists: {output_path}")
    manifest = _load_qualification_manifest()
    block_reports = []
    for path, expected_block in ((block_a_path, "A"), (block_b_path, "B")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE" or report.get("block") != expected_block:
            raise ValueError(f"Block {expected_block} report incomplete or identity mismatch")
        if report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"Block {expected_block} report uses a different qualification manifest")
        block_reports.append(report)
    by_key: dict[tuple[int, int, str], float] = {}
    init_hashes: dict[tuple[int, int, str], str] = {}
    for report in block_reports:
        for item in report["runs"]:
            run = item["run_report"]
            if run.get("status") != "COMPLETE" or int(run.get("updates", -1)) != UPDATES_PER_RUN:
                raise ValueError("run report is incomplete or wrong update budget")
            if run.get("test_split_loaded") is not False or run.get("endpoint") != "update_2000":
                raise ValueError("run report violates fixed validation-only update-2000 endpoint")
            endpoint = run.get("endpoint_nll_validation", {})
            if int(endpoint.get("update", -1)) != UPDATES_PER_RUN or not math.isfinite(float(endpoint.get("nll", float("nan")))):
                raise ValueError("missing/non-finite fixed update-2000 NLL")
            key = (int(item["seed"]), int(item["K"]), str(item["backend"]))
            if key in by_key:
                raise ValueError(f"duplicate backend-quality run key: {key}")
            by_key[key] = float(endpoint["nll"])
            run_identity = run.get("identity", {})
            if (run_identity.get("backend"), run_identity.get("K"), run_identity.get("seed")) != (key[2], key[1], key[0]):
                raise ValueError(f"run identity metadata mismatch for {key}")
            init_hashes[key] = str(run_identity.get("common_initial_state_sha256", ""))
    expected = {(seed, rounds, backend) for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"] for rounds in K_VALUES for backend in ("pytorch", "native")}
    if set(by_key) != expected:
        raise ValueError("all five seeds × K1/K4 × both backend arms are required exactly once")
    for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]:
        for rounds in K_VALUES:
            if not init_hashes[(seed, rounds, "pytorch")] or init_hashes[(seed, rounds, "pytorch")] != init_hashes[(seed, rounds, "native")]:
                raise ValueError(f"backend arms did not share identical initialization snapshot for seed={seed}, K={rounds}")
    deltas = {
        f"K{rounds}": {
            seed: by_key[(seed, rounds, "native")] - by_key[(seed, rounds, "pytorch")]
            for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]
        }
        for rounds in K_VALUES
    }
    eta = {
        seed: deltas["K1"][seed] - deltas["K4"][seed]
        for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]
    }
    margins = manifest["metrics"]
    endpoint_summaries = {
        "delta_K1": _paired_t_summary([deltas["K1"][seed] for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]], float(margins["delta_margin_nats_per_token"])),
        "delta_K4": _paired_t_summary([deltas["K4"][seed] for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]], float(margins["delta_margin_nats_per_token"])),
        "eta_depth": _paired_t_summary([eta[seed] for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]], float(margins["eta_margin_nats_per_token"])),
    }
    statuses = [value["status"] for value in endpoint_summaries.values()]
    qualification = "QUALIFIED" if all(status == "EQUIVALENT" for status in statuses) else (
        "NOT_EQUIVALENT" if any(status == "NOT_EQUIVALENT" for status in statuses) else "INCONCLUSIVE"
    )
    result = {
        "schema": "omega-backend-quality-formal-analysis-v1",
        "status": qualification,
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "blocks": [str(block_a_path), str(block_b_path)],
        "seeds": list(BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]),
        "nll_by_seed_K_backend": {
            f"seed_{seed}": {
                f"K{rounds}": {backend: by_key[(seed, rounds, backend)] for backend in ("pytorch", "native")}
                for rounds in K_VALUES
            }
            for seed in BLOCK_SEEDS["A"] + BLOCK_SEEDS["B"]
        },
        "delta_by_seed": deltas,
        "eta_depth_by_seed": eta,
        "endpoints": endpoint_summaries,
        "does_not_reclassify_strict_trajectory_proximity": "FAIL_RECORDED",
        "does_not_establish_training_quality_equivalence": "NOT_ESTABLISHED",
    }
    _write_json(output_path, result)
    result["report_path"] = str(output_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-resume-check", action="store_true", help="disposable toy check; no real model/data/DLL")
    parser.add_argument("--analyze-five-seeds", action="store_true", help="formal analysis only after complete A+B reports")
    parser.add_argument("--finalize-block", choices=("A", "B"), help="validate complete run set and seal block report; no training")
    parser.add_argument("--resume-run", action="store_true", help="resume one existing backend run from its own checkpoint")
    parser.add_argument("--block-a-report", type=Path)
    parser.add_argument("--block-a-preflight-report", type=Path)
    parser.add_argument("--block-b-report", type=Path)
    parser.add_argument("--run-block", choices=("A", "B"), help="real training; requires explicit authorization file")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--confirm-real-training", action="store_true")
    parser.add_argument("--confirm-block-b", action="store_true")
    parser.add_argument("--_synthetic-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_real-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_technical-resume-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--technical-resume-phase", choices=("continuous", "prefix", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--run-until-update", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--target-updates", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--technical-checkpoint-interval", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--confirm-technical-resume-test", action="store_true")
    parser.add_argument("--mode", choices=("continuous", "segment", "tail", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--block", choices=("A", "B"), help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("pytorch", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--K", type=int, choices=K_VALUES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-bundle", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._synthetic_child:
        if args.mode is None or args.backend is None or args.K is None or args.seed is None or args.run_dir is None:
            parser.error("incomplete synthetic child arguments")
        _run_synthetic_child(args.mode, args.backend, args.K, args.seed, args.run_dir, args.checkpoint)
        return 0
    if args._technical_resume_child:
        required = (args.backend, args.K, args.seed, args.run_dir, args.init_bundle, args.init_sha256, args.technical_resume_phase, args.run_until_update, args.target_updates, args.technical_checkpoint_interval)
        if any(value is None for value in required) or not args.confirm_technical_resume_test:
            parser.error("technical resume child requires complete route/checkpoint arguments and explicit preflight authorization")
        expected_until = {"continuous": 4, "prefix": 3, "resume": 4}[args.technical_resume_phase]
        if args.seed != 20260913 or args.run_until_update != expected_until or args.target_updates != 4 or args.technical_checkpoint_interval != 3:
            raise PermissionError("technical resume child requested updates outside sealed 4 / 3+1 test budget")
        if (args.technical_resume_phase == "resume") != (args.resume_checkpoint is not None):
            raise PermissionError("only fresh-process resume phase may load the update-3 checkpoint")
        manifest = _load_qualification_manifest()
        report = _quality_run_child(
            "RESUME_CHECK", args.backend, args.K, args.seed, args.init_bundle, args.init_sha256,
            args.run_dir, manifest, args.resume_checkpoint,
            run_until_update=args.run_until_update,
            target_updates=args.target_updates,
            checkpoint_interval=args.technical_checkpoint_interval,
            technical_resume_test=True,
            technical_resume_phase=args.technical_resume_phase,
        )
        print(json.dumps({"run_id": report["run_id"], "status": report["status"], "updates": report["updates"], "target_updates": report["target_updates"], "technical_resume_phase": report["technical_resume_phase"]}, sort_keys=True))
        return 0
    if args._real_child:
        if args.block is None or args.backend is None or args.K is None or args.seed is None or args.run_dir is None or args.init_bundle is None or args.init_sha256 is None or args.authorization_file is None or not args.confirm_real_training:
            parser.error("incomplete real child arguments")
        manifest = _load_qualification_manifest()
        _load_authorization(args.authorization_file, args.block, manifest)
        if args.block == "A":
            if args.block_a_preflight_report is None:
                raise PermissionError("Block A child requires its sealed preflight report")
            preflight = _validate_block_a_preflight_report(args.block_a_preflight_report, manifest)
            if _sha256_file(args.block_a_preflight_report) != json.loads(args.authorization_file.read_text(encoding="utf-8")).get("preflight_report_sha256"):
                raise PermissionError("Block A child authorization/preflight report hash mismatch")
        if args.block == "B" and not args.confirm_block_b:
            raise PermissionError("Block B child requires its own explicit confirmation")
        if args.block == "B":
            if args.block_a_report is None or not args.block_a_report.is_file():
                raise PermissionError("Block B child requires completed Block A report")
            a_report = json.loads(args.block_a_report.read_text(encoding="utf-8"))
            if a_report.get("status") != "COMPLETE" or a_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
                raise PermissionError("Block B child rejected incompatible Block A report")
        report = _quality_run_child(args.block, args.backend, args.K, args.seed, args.init_bundle, args.init_sha256, args.run_dir, manifest, args.resume_checkpoint)
        print(json.dumps({"run_id": report["run_id"], "backend": report["backend"], "K": report["K"], "seed": report["seed"], "status": "COMPLETE", "endpoint_nll": report["endpoint_nll_validation"]}, sort_keys=True))
        return 0
    if args.resume_run:
        required = (args.block, args.backend, args.K, args.seed, args.run_dir, args.init_bundle, args.init_sha256, args.resume_checkpoint, args.authorization_file)
        if any(value is None for value in required) or not args.confirm_real_training:
            parser.error("resume-run requires block/backend/K/seed/run-dir/init bundle/checkpoint/authorization and --confirm-real-training")
        manifest = _load_qualification_manifest()
        _load_authorization(args.authorization_file, args.block, manifest)
        if args.block == "A":
            if args.block_a_preflight_report is None:
                raise PermissionError("Block A resume requires its sealed preflight report")
            _validate_block_a_preflight_report(args.block_a_preflight_report, manifest)
            if _sha256_file(args.block_a_preflight_report) != json.loads(args.authorization_file.read_text(encoding="utf-8")).get("preflight_report_sha256"):
                raise PermissionError("Block A resume authorization/preflight report hash mismatch")
        if args.block == "B" and not args.confirm_block_b:
            raise PermissionError("Block B resume requires separate explicit confirmation")
        if args.block == "B":
            if args.block_a_report is None or not args.block_a_report.is_file():
                raise PermissionError("Block B resume requires completed Block A report")
            a_report = json.loads(args.block_a_report.read_text(encoding="utf-8"))
            if a_report.get("status") != "COMPLETE" or a_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
                raise PermissionError("Block B resume rejected incompatible Block A report")
        report = _quality_run_child(args.block, args.backend, args.K, args.seed, args.init_bundle, args.init_sha256, args.run_dir, manifest, args.resume_checkpoint)
        print(json.dumps({"run_id": report["run_id"], "status": report["status"], "completed_updates": report["updates"]}, sort_keys=True))
        return 0
    if args.finalize_block is not None:
        if args.authorization_file is None or not args.confirm_real_training:
            parser.error("finalize-block requires a matching authorization file and --confirm-real-training")
        manifest = _load_qualification_manifest()
        authorization = _load_authorization(args.authorization_file, args.finalize_block, manifest)
        if args.finalize_block == "A":
            if args.block_a_preflight_report is None:
                raise PermissionError("Block A finalization requires its sealed preflight report")
            _validate_block_a_preflight_report(args.block_a_preflight_report, manifest)
            if _sha256_file(args.block_a_preflight_report) != authorization.get("preflight_report_sha256"):
                raise PermissionError("Block A finalization authorization/preflight hash mismatch")
        if args.finalize_block == "B":
            if not args.confirm_block_b or args.block_a_report is None:
                raise PermissionError("Block B finalization requires A report and separate confirmation")
            a_report = json.loads(args.block_a_report.read_text(encoding="utf-8"))
            if a_report.get("status") != "COMPLETE" or a_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
                raise PermissionError("Block A is not complete under this manifest")
        report = _finalize_block(args.finalize_block, authorization, manifest)
        print(json.dumps({"block": report["block"], "status": report["status"], "report_path": report["report_path"]}, sort_keys=True))
        return 0
    if args.analyze_five_seeds and args.run_block is None:
        if args.block_a_report is None or args.block_b_report is None:
            parser.error("five-seed analysis requires both --block-a-report and --block-b-report")
        result = _formal_five_seed_analysis(args.block_a_report, args.block_b_report)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.synthetic_resume_check and args.run_block is None:
        result = _synthetic_resume_check()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.run_block is None:
        parser.error("select --synthetic-resume-check or --run-block")
    if not args.confirm_real_training or args.authorization_file is None:
        raise PermissionError("real training is HOLD: require explicit --confirm-real-training and authorization file")
    if args.run_block == "B" and not args.confirm_block_b:
        raise PermissionError("Block B is not automatic; separate --confirm-block-b authorization required")
    if args.run_block == "A" and args.block_a_preflight_report is None:
        raise PermissionError("Block A requires protocol/source manifest plus real-runner resume preflight report")
    report = _run_real_block(args.run_block, args.authorization_file, args.block_a_report, args.block_a_preflight_report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
