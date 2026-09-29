"""OMEGA-I7-EXECUTION-TOPOLOGY-PROBE.

Technical-only aggregate-throughput benchmark. Each worker keeps PyTorch 4/1,
FP32, batch 8, the canonical R1 loss, and the certified native DLL. No scientific
checkpoints or validation/test scores are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import numpy as np
import psutil
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
P2R0 = HERE.parent / "omega_native_runtime_p2r0"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

import run_backend_quality_qualification as quality  # noqa: E402
import run_loss_contract_causal_smoke as smoke  # noqa: E402
import run_k2_k6_preflight as k_preflight  # noqa: E402


SEED = 20260913
PRIMARY_K = 4
WARMUP_UPDATES = 8
MEASURED_UPDATES = 32
TOTAL_UPDATES = WARMUP_UPDATES + MEASURED_UPDATES
REPETITIONS = 3
NATIVE_WORKER_FAMILIES = {4: (1, 2, 3, 4, 5, 6), 8: (1, 2)}
MIN_AVAILABLE_BYTES = 1 * 1024**3
WORKER_WAIT_TIMEOUT_SECONDS = 1200
TOPOLOGY_CLOSE_RULE = "top two medians differ by no more than the larger of their observed median absolute deviations"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_model(ce: Any, technical_model: Any, seed: int, rounds: int) -> torch.nn.Module:
    if rounds in getattr(ce, "KS", ()):
        return ce.fresh_model(seed, rounds)
    return k_preflight._fresh_model(ce, technical_model, seed, rounds)


def _worker_main(args: argparse.Namespace) -> int:
    output = Path(args.worker_output).resolve()
    barrier = Path(args.barrier_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = quality._load_qualification_manifest()
    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    r1, _expanded = modules
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("topology probe workers require fixed torch intraop/inter-op 4/1")
    if quality._sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("topology probe native DLL hash mismatch")
    bridge.configure_library(quality.BE376_DLL)
    if not bridge.runtime_abi_available():
        raise RuntimeError("topology probe requires the certified native runtime ABI")
    bridge.configure_runtime(int(args.native_workers))

    docs, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    train_manifest = json.loads(quality.TRAIN_MANIFEST.read_text(encoding="utf-8"))
    expected_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in train_manifest["documents"]]
    actual_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in docs]
    if len(docs) != 602 or actual_keys != expected_keys:
        raise ValueError("topology worker documents differ from the frozen 602-document training manifest")
    init_path = Path(args.init_path).resolve()
    if _sha256_file(init_path) != args.init_sha256:
        raise ValueError("topology worker common-init hash mismatch")
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    if int(init.get("seed", -1)) != int(args.seed) or int(init.get("K", -1)) != int(args.K):
        raise ValueError("topology worker common-init seed/K mismatch")

    model = _fresh_model(ce, technical_model, int(args.seed), int(args.K))
    model.load_state_dict(init["model_state"], strict=True)
    if quality._value_hash(model.state_dict()) != init["model_state_sha256"]:
        raise ValueError("topology worker model differs from the common initialization")
    torch.set_rng_state(init["torch_rng_state"].clone())
    random.setstate(init["python_rng_state"])
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ce.BASE_LR,
        betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS,
        weight_decay=ce.WEIGHT_DECAY,
    )
    if optimizer.state or quality._value_hash(optimizer.state_dict()) != init["fresh_adamw_state_sha256"]:
        raise AssertionError("topology worker AdamW must start fresh")
    state = model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
    if quality._tensor_hash(state) != init["initial_recurrent_state_sha256"]:
        raise ValueError("topology worker initial recurrent state mismatch")
    state_part_weight = r2.prepare_native_model(model)
    pair_schedule = train_manifest["pairs"]
    logical_ledger: list[dict[str, Any]] = []
    measured_records: list[dict[str, Any]] = []
    state_trajectory: list[torch.Tensor] = []
    measured_start_ns: int | None = None
    measured_end_ns: int | None = None
    available_min = int(psutil.virtual_memory().available)

    def one_update(update: int, measured: bool) -> None:
        nonlocal state
        if (barrier / "cancel.flag").exists():
            raise RuntimeError("topology group cancelled by its coordinator")
        window = update % 2
        if window == 0:
            state = model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
        pair_index = update // 2
        pair = pair_schedule[pair_index]
        positions = [int(value) for value in pair["document_indices"]]
        source = torch.tensor([docs[position]["tokens"] for position in positions], dtype=torch.long)
        offset = window * 256
        inputs = source[:, offset : offset + 256]
        targets = source[:, offset + 1 : offset + 257]
        teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, window)
        previous_state = state.detach() if window == 1 else state
        next_state, student_logits, _readout = r2._native_forward(
            model, inputs, previous_state, state_part_weight, bridge
        )
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        loss_terms = quality.r1_masked_token_mean_loss(student_logits, teacher_logits, targets, valid_mask)
        if loss_terms["valid_tokens"] != 2048 or not bool(torch.isfinite(loss_terms["total"]).all()):
            raise FloatingPointError(f"invalid canonical loss/target count at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss_terms["total"].backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"nonfinite gradient at update {update}")
        preclip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
        if not math.isfinite(preclip_norm):
            raise FloatingPointError(f"nonfinite clip norm at update {update}")
        clip_coefficient = min(1.0, float(ce.CLIP_NORM) / (preclip_norm + 1.0e-6))
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"nonfinite parameter after update {update}")
        state = next_state.detach()
        if args.capture_state:
            state_trajectory.append(state.detach().cpu().clone())
        event = {
            "update": update,
            "pair": pair_index,
            "window": window,
            "document_positions": positions,
            "valid_tokens": int(loss_terms["valid_tokens"]),
            "loss": float(loss_terms["total"].detach().item()),
            "ce": float(loss_terms["ce"].detach().item()),
            "kl": float(loss_terms["kl"].detach().item()),
            "preclip_norm": preclip_norm,
            "clip_coefficient": clip_coefficient,
            "state_sha256": quality._tensor_hash(state),
        }
        logical_ledger.append(event)
        if measured:
            measured_records.append({key: event[key] for key in ("update", "pair", "window", "valid_tokens", "loss", "ce", "kl", "preclip_norm", "clip_coefficient", "state_sha256")})

    for update in range(WARMUP_UPDATES):
        one_update(update, measured=False)
        available_min = min(available_min, int(psutil.virtual_memory().available))

    ready_path = barrier / f"worker_{args.worker_id}.ready.json"
    ready_path.write_text(
        json.dumps({"worker_id": args.worker_id, "native_workers": args.native_workers, "warmup_updates": WARMUP_UPDATES}),
        encoding="utf-8",
    )
    wait_started = time.monotonic()
    while not (barrier / "release.flag").exists():
        if (barrier / "cancel.flag").exists():
            raise RuntimeError("topology group cancelled before measurement release")
        if time.monotonic() - wait_started > WORKER_WAIT_TIMEOUT_SECONDS:
            raise TimeoutError("timed out waiting for topology group release")
        time.sleep(0.02)

    measured_start_ns = time.perf_counter_ns()
    for update in range(WARMUP_UPDATES, TOTAL_UPDATES):
        one_update(update, measured=True)
        available_min = min(available_min, int(psutil.virtual_memory().available))
    measured_end_ns = time.perf_counter_ns()

    state_hash = quality._tensor_hash(state)
    model_hash = quality._value_hash(model.state_dict())
    optimizer_hash = quality._value_hash(optimizer.state_dict())
    logical_ledger_hash = quality._canonical_hash(logical_ledger)
    result = {
        "status": "COMPLETE",
        "worker_id": int(args.worker_id),
        "K": int(args.K),
        "seed": int(args.seed),
        "native_workers": int(args.native_workers),
        "torch_intraop": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
        "physical_batch": p0.PHYSICAL_BATCH,
        "effective_batch": ce.PHYSICAL_BATCH,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": len(measured_records),
        "measured_start_ns": measured_start_ns,
        "measured_end_ns": measured_end_ns,
        "measured_wall_seconds": (measured_end_ns - measured_start_ns) / 1e9,
        "available_memory_min_bytes": available_min,
        "model_state_sha256": model_hash,
        "optimizer_state_sha256": optimizer_hash,
        "recurrent_state_sha256": state_hash,
        "logical_ledger_sha256": logical_ledger_hash,
        "measured_records": measured_records,
        "ledger_records": logical_ledger,
        "cursor": {"next_update": TOTAL_UPDATES, "pair": TOTAL_UPDATES // 2, "window": TOTAL_UPDATES % 2},
        "validation_or_test_loaded": False,
        "scientific_checkpoint_written": False,
    }
    if args.capture_state:
        torch.save(
            {
                "schema": "omega-i7-topology-safety-final-state-v1",
                "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "recurrent_state": state.detach().cpu().clone(),
                "cursor": result["cursor"],
                "logical_ledger": logical_ledger,
                "state_trajectory": state_trajectory,
                "measured_records": measured_records,
                "model_state_sha256": model_hash,
                "optimizer_state_sha256": optimizer_hash,
                "recurrent_state_sha256": state_hash,
                "logical_ledger_sha256": logical_ledger_hash,
            },
            output.with_name("final_state.pt"),
        )
        result["final_state_path"] = str(output.with_name("final_state.pt"))
        result["final_state_sha256"] = _sha256_file(output.with_name("final_state.pt"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _wait_one_ready(
    process: subprocess.Popen[Any],
    ready_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    deadline: float,
) -> bool:
    while time.monotonic() < deadline:
        if ready_path.is_file():
            return True
        return_code = process.poll()
        if return_code is not None:
            return False
        if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
            return False
        time.sleep(0.1)
    return False


def _run_group(
    *,
    output_dir: Path,
    native_workers: int,
    concurrency: int,
    repetition: int,
    rounds: int,
    seed: int,
    init_path: Path,
    init_sha: str,
    capture_state_worker: int | None = None,
    group_label: str = "topology",
) -> dict[str, Any]:
    group_parent = output_dir / group_label / f"K{rounds}" / f"nw{native_workers}_p{concurrency}"
    group_dir = group_parent / f"rep{repetition}"
    attempt = 1
    while group_dir.exists():
        group_dir = group_parent / f"rep{repetition}_retry{attempt}"
        attempt += 1
    group_dir.mkdir(parents=True, exist_ok=False)
    barrier = group_dir / "barrier"
    barrier.mkdir(parents=True, exist_ok=False)
    processes: list[subprocess.Popen[Any]] = []
    workers: list[dict[str, Any]] = []
    minimum_available = int(psutil.virtual_memory().available)
    if minimum_available < MIN_AVAILABLE_BYTES:
        return {
            "status": "ABORTED_MEMORY_GUARD",
            "K": rounds,
            "native_workers": native_workers,
            "concurrency": concurrency,
            "repetition": repetition,
            "available_bytes_before_spawn": minimum_available,
            "group_dir": str(group_dir),
        }

    try:
        # Start and warm each worker, then hold it at the barrier before spawning
        # the next one. This caps transient allocation spikes on the 32-GiB host.
        for worker_id in range(concurrency):
            worker_out = group_dir / f"worker_{worker_id}.json"
            stdout_path = group_dir / f"worker_{worker_id}.stdout.log"
            stderr_path = group_dir / f"worker_{worker_id}.stderr.log"
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-output",
                str(worker_out),
                "--barrier-dir",
                str(barrier),
                "--worker-id",
                str(worker_id),
                "--native-workers",
                str(native_workers),
                "--K",
                str(rounds),
                "--seed",
                str(seed),
                "--init-path",
                str(init_path),
                "--init-sha256",
                init_sha,
            ]
            if worker_id == capture_state_worker:
                command.append("--capture-state")
            stderr_stream = stderr_path.open("wb")
            stdout_stream = stdout_path.open("wb")
            process = subprocess.Popen(command, cwd=str(LAB_ROOT.parent), stdout=stdout_stream, stderr=stderr_stream)
            processes.append(process)
            workers.append({"process": process, "stderr_stream": stderr_stream, "stdout_stream": stdout_stream, "output": worker_out, "ready": barrier / f"worker_{worker_id}.ready.json"})
            deadline = time.monotonic() + WORKER_WAIT_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                minimum_available = min(minimum_available, int(psutil.virtual_memory().available))
                if workers[-1]["ready"].is_file():
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"worker {worker_id} exited before warmup barrier; rc={process.returncode}")
                if minimum_available < MIN_AVAILABLE_BYTES:
                    raise MemoryError(f"available memory fell below 1 GiB while warming worker {worker_id}")
                time.sleep(0.1)
            else:
                raise TimeoutError(f"worker {worker_id} failed to reach the ready barrier")

        minimum_available = min(minimum_available, int(psutil.virtual_memory().available))
        (barrier / "release.flag").write_text("release\n", encoding="ascii")
        released_ns = time.perf_counter_ns()
        while any(row["process"].poll() is None for row in workers):
            minimum_available = min(minimum_available, int(psutil.virtual_memory().available))
            if minimum_available < MIN_AVAILABLE_BYTES:
                raise MemoryError("available memory fell below 1 GiB during a measured topology group")
            failed = [row for row in workers if row["process"].poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"topology worker exited with error: {failed[0]['process'].returncode}")
            time.sleep(0.2)
        for row in workers:
            row["process"].wait()
            row["stderr_stream"].close()
            row["stdout_stream"].close()
        results = [json.loads(row["output"].read_text(encoding="utf-8")) for row in workers]
        if any(row.get("status") != "COMPLETE" or row.get("measured_updates") != MEASURED_UPDATES for row in results):
            raise RuntimeError("one or more workers did not complete exactly 32 measured updates")
        starts = [int(row["measured_start_ns"]) for row in results]
        ends = [int(row["measured_end_ns"]) for row in results]
        group_wall = (max(ends) - min(starts)) / 1e9
        completed = sum(int(row["measured_updates"]) for row in results)
        return {
            "status": "COMPLETE",
            "K": rounds,
            "native_workers": native_workers,
            "concurrency": concurrency,
            "repetition": repetition,
            "attempt": attempt - 1 if group_dir.name != f"rep{repetition}" else 0,
            "warmup_updates_per_process": WARMUP_UPDATES,
            "measured_updates_per_process": MEASURED_UPDATES,
            "measured_updates_aggregate": completed,
            "group_wall_seconds": group_wall,
            "throughput_aggregate_updates_per_second": completed / group_wall,
            "release_to_last_completion_seconds": (max(ends) - released_ns) / 1e9,
            "available_memory_min_bytes": minimum_available,
            "workers": results,
            "group_dir": str(group_dir),
        }
    except Exception as error:
        (barrier / "cancel.flag").write_text(type(error).__name__ + "\n", encoding="ascii")
        for row in workers:
            if row["process"].poll() is None:
                row["process"].terminate()
        for row in workers:
            try:
                row["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                row["process"].kill()
                row["process"].wait(timeout=10)
            row["stderr_stream"].close()
            row["stdout_stream"].close()
        return {
            "status": "ABORTED_MEMORY_GUARD" if isinstance(error, MemoryError) else "FAILED_RUNTIME",
            "error_type": type(error).__name__,
            "error": str(error),
            "available_memory_min_bytes": minimum_available,
            "group_dir": str(group_dir),
        }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: list[float]) -> float:
    median = _median(values)
    return float(np.median(np.abs(np.asarray(values, dtype=np.float64) - median)))


def _nested_tensor_pairs(left: Any, right: Any, prefix: str = "") -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    rows: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    if torch.is_tensor(left) and torch.is_tensor(right):
        rows.append((prefix, left, right))
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"safety state key mismatch at {prefix}")
        for key in left:
            rows.extend(_nested_tensor_pairs(left[key], right[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"safety state sequence length mismatch at {prefix}")
        for index, (a, b) in enumerate(zip(left, right)):
            rows.extend(_nested_tensor_pairs(a, b, f"{prefix}[{index}]"))
    return rows


def _verify_safety_pair(
    solo_path: Path,
    concurrent_path: Path,
    local_gates: Any,
) -> dict[str, Any]:
    solo = torch.load(solo_path, map_location="cpu", weights_only=False)
    concurrent = torch.load(concurrent_path, map_location="cpu", weights_only=False)
    sections = ("model", "optimizer", "recurrent_state")
    exact: dict[str, bool] = {}
    gate_rows: list[dict[str, Any]] = []
    for section in sections:
        exact[section] = quality._value_hash(solo[section]) == quality._value_hash(concurrent[section])
        for name, left, right in _nested_tensor_pairs(solo[section], concurrent[section], section):
            gate_rows.append(local_gates._tensor_gate(left, right, name=f"safety.{name}"))
    exact["cursor"] = solo["cursor"] == concurrent["cursor"]
    exact["losses_and_ledger"] = solo["logical_ledger"] == concurrent["logical_ledger"]
    exact["logical_ledger_sha256"] = solo["logical_ledger_sha256"] == concurrent["logical_ledger_sha256"]
    exact["measured_records"] = solo["measured_records"] == concurrent["measured_records"]
    logical_fields = ("update", "pair", "window", "document_positions", "valid_tokens")
    numerical_fields = ("loss", "ce", "kl", "preclip_norm", "clip_coefficient")
    logical_identity_exact = len(solo["logical_ledger"]) == len(concurrent["logical_ledger"])
    for left_event, right_event in zip(solo["logical_ledger"], concurrent["logical_ledger"]):
        logical_identity_exact = logical_identity_exact and all(left_event[key] == right_event[key] for key in logical_fields)
        for field in numerical_fields:
            gate_rows.append(local_gates._tensor_gate(
                torch.tensor(left_event[field], dtype=torch.float32),
                torch.tensor(right_event[field], dtype=torch.float32),
                name=f"safety.ledger.{field}.update{left_event['update']}",
            ))
    exact["ledger_structure_exact"] = logical_identity_exact
    exact["ledger_state_hashes_exact"] = all(
        left.get("state_sha256") == right.get("state_sha256")
        for left, right in zip(solo["logical_ledger"], concurrent["logical_ledger"])
    )
    state_trajectory_gate_rows = [
        local_gates._tensor_gate(left_state, right_state, name=f"safety.ledger_state.update{index}")
        for index, (left_state, right_state) in enumerate(zip(solo["state_trajectory"], concurrent["state_trajectory"]))
    ]
    gate_rows.extend(state_trajectory_gate_rows)
    exact["state_trajectory_exact"] = all(
        torch.equal(left_state, right_state)
        for left_state, right_state in zip(solo["state_trajectory"], concurrent["state_trajectory"])
    ) and len(solo["state_trajectory"]) == len(concurrent["state_trajectory"])
    exact_all = all(exact.values())
    numeric_pass = (
        all(bool(row["pass"]) for row in gate_rows)
        and exact["cursor"]
        and exact["ledger_structure_exact"]
        and len(solo["state_trajectory"]) == len(concurrent["state_trajectory"])
    )
    return {
        "status": "BITWISE_EXACT_PASS" if exact_all else ("NUMERIC_GATE_PASS" if numeric_pass else "HOLD_NUMERIC_MISMATCH"),
        "bitwise_exact_by_section": exact,
        "numeric_tensor_gates": gate_rows,
        "all_numeric_tensor_gates_pass": all(bool(row["pass"]) for row in gate_rows),
        "solo_final_state_sha256": _sha256_file(solo_path),
        "concurrent_final_state_sha256": _sha256_file(concurrent_path),
        "solo_path": str(solo_path),
        "concurrent_path": str(concurrent_path),
    }


def run_probe(
    output_dir: Path,
    *,
    resume: bool = False,
    workers_only: int | None = None,
    concurrency_only: int | None = None,
    partial: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"topology probe output exists; resume must be explicit: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    manifest = quality._load_qualification_manifest()
    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("coordinator PyTorch policy is not fixed at 4/1")
    if _sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("certified DLL SHA mismatch before topology probe")
    preflight_path = HERE / "results" / "k2_k6_preflight_20260927T121900Z" / "k2_k6_preflight_report.json"
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight_unsigned = dict(preflight_report)
    preflight_signature = preflight_unsigned.pop("report_self_sha256", None)
    if not preflight_signature or preflight_signature != quality._canonical_hash(preflight_unsigned):
        raise ValueError("K2/K6 preflight report self-hash mismatch")
    k6_init_row = preflight_report["common_initializations"]["6"]
    k6_init_path = Path(k6_init_row["path"])
    if _sha256_file(k6_init_path) != k6_init_row["sha256"]:
        raise ValueError("K6 topology tie-break initialization hash mismatch")

    source_ids = {
        "canonical_loss": quality.r1_masked_token_mean_loss.__code__.co_filename,
        "backend_quality_runner": str(Path(quality.__file__).resolve()),
        "causal_smoke_gates": str(CAMPAIGN / "run_loss_contract_causal_smoke.py"),
        "native_r2_runner": str(Path(r2.__file__).resolve()),
        "native_bridge": str(Path(bridge.__file__).resolve()),
        "topology_runner": str(Path(__file__).resolve()),
    }
    source_hashes = {key: _sha256_file(Path(path).resolve()) for key, path in source_ids.items()}
    local_gates = smoke._load_gate_module(quality)
    progress_path = output_dir / "topology_progress.json"
    if resume:
        if not progress_path.is_file():
            raise FileNotFoundError(f"cannot resume: topology progress report missing: {progress_path}")
        report = json.loads(progress_path.read_text(encoding="utf-8"))
        if report.get("qualification_manifest_sha256") != manifest["manifest_sha256"] or report.get("source_ids") != source_ids:
            raise ValueError("resume report qualification/source identities differ from this probe")
        original_hashes = report.get("source_hashes_before", {})
        for name, digest in source_hashes.items():
            if name != "topology_runner" and original_hashes.get(name) != digest:
                raise ValueError(f"pinned source identity changed during topology resume: {name}")
        history = list(report.get("topology_runner_source_hash_history", [original_hashes.get("topology_runner")]))
        if source_hashes["topology_runner"] not in history:
            history.append(source_hashes["topology_runner"])
        report["topology_runner_source_hash_history"] = history
        report["source_hashes_current_invocation"] = source_hashes
        init_row = report["K4_common_init"]
        init_path = Path(init_row["path"])
        init_sha = str(init_row["sha256"])
        if _sha256_file(init_path) != init_sha:
            raise ValueError("K4 common init hash mismatch while resuming")
    else:
        init_dir = output_dir / "common_initializations"
        init_dir.mkdir(parents=True, exist_ok=False)
        init_path, init_sha = quality._prepare_common_initialization(ce, SEED, PRIMARY_K, init_dir)
        report = {
            "schema": "omega-i7-execution-topology-probe-v1",
            "unit": "OMEGA-I7-EXECUTION-TOPOLOGY-PROBE",
            "status": "IN_PROGRESS",
            "qualification_manifest_sha256": manifest["manifest_sha256"],
            "K4_common_init": {"path": str(init_path), "sha256": init_sha},
            "K6_tiebreak_init": {"path": str(k6_init_path), "sha256": k6_init_row["sha256"]},
            "torch_policy_fixed": {"intraop": 4, "interop": 1, "observed": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}},
            "native_dll": {"path": str(quality.BE376_DLL), "sha256": quality.BE376_SHA256},
            "cpu": {"processor": platform.processor(), "platform": platform.platform()},
            "runtime_policy": policy,
            "native_workers_families": {"4": [1, 2, 3, 4, 5, 6], "8": [1, 2]},
            "repetitions_per_topology": REPETITIONS,
            "warmup_updates_per_process": WARMUP_UPDATES,
            "measured_updates_per_process": MEASURED_UPDATES,
            "K4_is_primary_load": True,
            "tie_break_rule": TOPOLOGY_CLOSE_RULE,
            "minimum_available_memory_guard_bytes": MIN_AVAILABLE_BYTES,
            "affinity_or_priority_changed": False,
            "scientific_checkpoints_written": False,
            "topologies": {},
            "source_ids": source_ids,
            "source_hashes_before": source_hashes,
            "output_dir": str(output_dir),
        }
        quality._write_json(progress_path, report)

    def collect_family(rounds: int, init: Path, init_hash: str, configs: list[tuple[int, int]]) -> None:
        for native_workers, concurrency in configs:
            key = f"native_workers_{native_workers}_concurrency_{concurrency}_K{rounds}"
            entry = report["topologies"].setdefault(
                key,
                {"native_workers": native_workers, "concurrency": concurrency, "K": rounds, "repetitions": []},
            )
            if entry.get("status") in ("ABORTED_MEMORY_GUARD", "FAILED_RUNTIME"):
                continue
            complete_repetitions = {
                int(item["repetition"])
                for item in entry["repetitions"]
                if item.get("status") == "COMPLETE" and item.get("repetition") is not None
            }
            terminal_repetitions = {
                int(item.get("repetition") if item.get("repetition") is not None else index + 1)
                for index, item in enumerate(entry["repetitions"])
                if item.get("status") in ("ABORTED_MEMORY_GUARD", "FAILED_RUNTIME")
            }
            for repetition in range(1, REPETITIONS + 1):
                if repetition in complete_repetitions or repetition in terminal_repetitions:
                    continue
                result = _run_group(
                    output_dir=output_dir,
                    native_workers=native_workers,
                    concurrency=concurrency,
                    repetition=repetition,
                    rounds=rounds,
                    seed=SEED,
                    init_path=init,
                    init_sha=init_hash,
                )
                entry["repetitions"].append(result)
                quality._write_json(progress_path, report)
                if result["status"] != "COMPLETE":
                    entry["status"] = result["status"]
                    break
            completed = [item for item in entry["repetitions"] if item.get("status") == "COMPLETE"]
            if len(completed) == REPETITIONS:
                throughputs = [float(item["throughput_aggregate_updates_per_second"]) for item in completed]
                entry["status"] = "COMPLETE"
                entry["median_aggregate_updates_per_second"] = _median(throughputs)
                entry["median_absolute_deviation"] = _mad(throughputs)

    family4_configs = [(4, concurrency) for concurrency in (1, 2, 3, 4, 5, 6)]
    family8_configs = [(8, concurrency) for concurrency in (1, 2)]
    selected_configs: dict[int, list[tuple[int, int]]] = {4: family4_configs, 8: family8_configs}
    if workers_only is not None:
        selected_configs = {workers_only: selected_configs[workers_only]}
    if concurrency_only is not None:
        selected_configs = {
            workers: [config for config in configs if config[1] == concurrency_only]
            for workers, configs in selected_configs.items()
        }
    for native_workers, configs in selected_configs.items():
        if configs:
            collect_family(PRIMARY_K, init_path, init_sha, configs)

    if partial:
        report["status"] = "IN_PROGRESS"
        report["last_partial_filter"] = {"workers_only": workers_only, "concurrency_only": concurrency_only}
        report["source_hashes_current_invocation"] = source_hashes
        quality._write_json(progress_path, report)
        return report

    valid4 = [
        (key, value)
        for key, value in report["topologies"].items()
        if value["K"] == PRIMARY_K and value["native_workers"] == 4 and value.get("status") == "COMPLETE"
    ]
    valid8 = [
        (key, value)
        for key, value in report["topologies"].items()
        if value["K"] == PRIMARY_K and value["native_workers"] == 8 and value.get("status") == "COMPLETE"
    ]
    if not valid4:
        report["status"] = "HOLD_NO_COMPLETE_WORKERS4_TOPOLOGY"
    else:
        best4_key, best4 = max(valid4, key=lambda row: float(row[1]["median_aggregate_updates_per_second"]))
        best8_key, best8 = max(valid8, key=lambda row: float(row[1]["median_aggregate_updates_per_second"])) if valid8 else (None, None)
        adopt8 = bool(
            best8 is not None
            and float(best8["median_aggregate_updates_per_second"])
            >= 1.10 * float(best4["median_aggregate_updates_per_second"])
        )
        report["primary_decision"] = {
            "best_workers4_topology": best4_key,
            "best_workers4_median_updates_per_second": best4["median_aggregate_updates_per_second"],
            "best_workers8_topology": best8_key,
            "best_workers8_median_updates_per_second": None if best8 is None else best8["median_aggregate_updates_per_second"],
            "workers8_adoption_threshold": 1.10,
            "workers8_adopted": (adopt8 if best8 is not None else None),
            "workers8_evaluability": "MEASURED" if best8 is not None else "UNAVAILABLE_CERTIFIED_DLL_RUNTIME_REJECTS_WORKER_COUNT_8",
            "selected_topology": best8_key if adopt8 else best4_key,
            "selected_native_workers": 8 if adopt8 else 4,
            "selected_concurrency": int((best8 if adopt8 else best4)["concurrency"]),
            "parallelism_requalification_required": adopt8,
        }

        all_valid = sorted(
            [*valid4, *valid8],
            key=lambda row: float(row[1]["median_aggregate_updates_per_second"]),
            reverse=True,
        )
        tie_break_topologies: list[tuple[str, int, int]] = []
        if len(all_valid) >= 2:
            first_key, first = all_valid[0]
            second_key, second = all_valid[1]
            med_gap = abs(float(first["median_aggregate_updates_per_second"]) - float(second["median_aggregate_updates_per_second"]))
            noise_scale = max(float(first["median_absolute_deviation"]), float(second["median_absolute_deviation"]))
            report["primary_decision"]["top_two_median_gap"] = med_gap
            report["primary_decision"]["top_two_noise_scale_max_mad"] = noise_scale
            report["primary_decision"]["tie_break_triggered"] = med_gap <= noise_scale
            if med_gap <= noise_scale:
                tie_break_topologies = [
                    (first_key, int(first["native_workers"]), int(first["concurrency"])),
                    (second_key, int(second["native_workers"]), int(second["concurrency"])),
                ]
        report["tie_break_results"] = {}
        if tie_break_topologies:
            init_by_k = {1: None, 6: k6_init_path}
            for tie_k in (1, 6):
                if tie_k == 1:
                    path, digest = quality._prepare_common_initialization(ce, SEED, 1, init_dir)
                    init_by_k[1] = path
                    init_by_k[1] = (path, digest)
                else:
                    init_by_k[6] = (k6_init_path, k6_init_row["sha256"])
                for config_key, nw, concurrency in tie_break_topologies:
                    tie_key = f"K{tie_k}_{config_key}"
                    rows = []
                    for repetition in range(1, REPETITIONS + 1):
                        path, digest = init_by_k[tie_k]
                        item = _run_group(
                            output_dir=output_dir,
                            native_workers=nw,
                            concurrency=concurrency,
                            repetition=repetition,
                            rounds=tie_k,
                            seed=SEED,
                            init_path=path,
                            init_sha=digest,
                        )
                        rows.append(item)
                        quality._write_json(output_dir / "topology_progress.json", report)
                        if item["status"] != "COMPLETE":
                            break
                    valid_rows = [item for item in rows if item["status"] == "COMPLETE"]
                    report["tie_break_results"][tie_key] = {
                        "topology_from_K4": config_key,
                        "native_workers": nw,
                        "concurrency": concurrency,
                        "repetitions": rows,
                        "median_aggregate_updates_per_second": _median([float(row["throughput_aggregate_updates_per_second"]) for row in valid_rows]) if len(valid_rows) == REPETITIONS else None,
                    }

        selected = report["primary_decision"]
        selected_workers = int(selected["selected_native_workers"])
        selected_concurrency = int(selected["selected_concurrency"])
        safety_root = output_dir / "concurrency_safety"
        safety_root.mkdir(parents=True, exist_ok=False)
        solo = _run_group(
            output_dir=safety_root,
            native_workers=selected_workers,
            concurrency=1,
            repetition=1,
            rounds=PRIMARY_K,
            seed=SEED,
            init_path=init_path,
            init_sha=init_sha,
            capture_state_worker=0,
            group_label="safety_solo",
        )
        concurrent = _run_group(
            output_dir=safety_root,
            native_workers=selected_workers,
            concurrency=selected_concurrency,
            repetition=1,
            rounds=PRIMARY_K,
            seed=SEED,
            init_path=init_path,
            init_sha=init_sha,
            capture_state_worker=0,
            group_label="safety_loaded",
        )
        solo_path = Path(solo.get("workers", [{}])[0].get("final_state_path", ""))
        concurrent_path = Path(concurrent.get("workers", [{}])[0].get("final_state_path", ""))
        if solo.get("status") != "COMPLETE" or concurrent.get("status") != "COMPLETE":
            report["concurrency_safety"] = {"status": "HOLD_SAFETY_RUN_INCOMPLETE", "solo": solo, "concurrent": concurrent}
        else:
            report["concurrency_safety"] = _verify_safety_pair(solo_path, concurrent_path, local_gates)
            report["concurrency_safety"]["solo_group"] = solo
            report["concurrency_safety"]["concurrent_group"] = concurrent
        report["status"] = (
            "PASS_WORKERS8_REQUALIFICATION_REQUIRED"
            if adopt8 and report["concurrency_safety"]["status"] in ("BITWISE_EXACT_PASS", "NUMERIC_GATE_PASS")
            else "PASS_WORKERS4_CONCURRENCY"
            if not adopt8 and report["concurrency_safety"]["status"] in ("BITWISE_EXACT_PASS", "NUMERIC_GATE_PASS")
            else "HOLD_CONCURRENCY_SAFETY"
        )

    report["source_hashes_after"] = {key: _sha256_file(Path(path).resolve()) for key, path in source_ids.items()}
    if report["source_hashes_after"] != source_hashes:
        raise RuntimeError("a pinned source changed during the topology probe")
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_dir / "topology_probe_report.json", report)
    quality._write_json(output_dir / "topology_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-topology-probe-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="resume an existing progress report without repeating completed groups")
    parser.add_argument("--workers-only", type=int, choices=(4, 8))
    parser.add_argument("--concurrency-only", type=int, choices=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--partial", action="store_true", help="run selected topology groups and leave report open for a later resume")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--barrier-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--native-workers", type=int, choices=(4, 8), help=argparse.SUPPRESS)
    parser.add_argument("--K", type=int, choices=(1, 4, 6), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--init-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--capture-state", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        required = (args.worker_output, args.barrier_dir, args.worker_id, args.native_workers, args.K, args.seed, args.init_path, args.init_sha256)
        if any(value is None for value in required):
            parser.error("topology worker arguments incomplete")
        return _worker_main(args)
    if not args.confirm_topology_probe_go:
        parser.error("OMEGA-I7-EXECUTION-TOPOLOGY-PROBE requires the explicit MD/282 GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"i7_topology_probe_{stamp}"
    try:
        report = run_probe(
            output_dir,
            resume=args.resume,
            workers_only=args.workers_only,
            concurrency_only=args.concurrency_only,
            partial=args.partial,
        )
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "topology_probe_failure.json",
                {"status": "FAILED_RUNTIME", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "output_dir": str(output_dir)},
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report.get("primary_decision"),
                "concurrency_safety": report.get("concurrency_safety", {}).get("status"),
                "report": str(output_dir / "topology_probe_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
