"""Phase 2 ER64 cost-gate mechanics.

Phase 1 exposes the real candidate factory, corrected cost formulas, report
artifact self-hashing, and synthetic checks. Real corpus execution is reserved
for explicit follow-up authorization after review of this phase.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import hmac
import json
import math
import os
import platform
import secrets
import sys
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
ER64_ADAPTER_DIR = HERE
ER32_GATE_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er32_integration_and_cost_gate"
SCOPING_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
R1_SCRIPTS_DIR = CAMPAIGN_ROOT.parent / "scripts"
FASTPATH_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
sys.path.insert(0, str(ER64_ADAPTER_DIR))
sys.path.insert(0, str(ER32_GATE_DIR))
sys.path.insert(0, str(SCOPING_DIR))
sys.path.insert(0, str(FASTPATH_DIR))
sys.path.insert(0, str(R1_SCRIPTS_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402
from omega_fast_er64 import OmegaCoreLMFastER64  # noqa: E402
from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical  # noqa: E402
from run_scientific_scoping_a import configure_cpu_runtime  # noqa: E402


VOCAB_SIZE = 50257
DIMENSION = 128
SLOTS = 8
ER64_RANK = 64
TRAINING_UPDATES = 6
GATE_THRESHOLD = 1.25
RSS_ANOMALY_BYTES = 128 * 1024 * 1024
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"
CONFIGS = ("F", "ER64")
COMBINATIONS = (("F", 1), ("F", 4), ("ER64", 1), ("ER64", 4))
CHILD_TOKEN_ENV = "ER64_COST_GATE_CHILD_TOKEN"
er32_base: Any | None = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_self_hashed_json(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    snapshot = json_safe(payload)
    snapshot["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    unsigned = canonical_json(snapshot)
    digest = sha256_bytes(unsigned)
    snapshot["artifact_self_hash"] = digest
    encoded = canonical_json(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    persisted = path.read_bytes()
    parsed = json.loads(persisted.decode("utf-8"))
    stored = parsed["artifact_self_hash"]
    parsed["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    if persisted != encoded or canonical_json(parsed) != unsigned or sha256_bytes(canonical_json(parsed)) != stored:
        raise RuntimeError("artifact self-hash verification failed")
    return digest, sha256_bytes(persisted)


def make_f_reference(rounds: int, seed: int, *, vocab_size: int = VOCAB_SIZE, dimension: int = DIMENSION, slots: int = SLOTS) -> OmegaCoreLMFast:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = OmegaCoreLM0R1Technical(vocab_size=vocab_size, dimension=dimension, slots=slots, rounds=rounds, variant="shared").float()
        return OmegaCoreLMFast.from_reference(reference).float()


def make_candidate(config: str, rounds: int, f_reference: OmegaCoreLMFast, *, experimental_seed: int = 20260917) -> nn.Module:
    if config == "F":
        return copy.deepcopy(f_reference).float()
    if config == "ER64":
        return OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=experimental_seed, implementation="efficient").float()
    raise ValueError(f"unknown config: {config}")


def aggregate_training_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    by_combination: dict[str, dict[str, Any]] = {}
    incomplete = [
        {"config": result.get("config"), "rounds": result.get("rounds"), "status": result.get("status"), "updates_completed": result.get("updates_completed", 0)}
        for result in values
        if result.get("status") != "completed" or int(result.get("updates_completed", 0)) != TRAINING_UPDATES
    ]
    for result in values:
        key = f"{result['config']}_K{result['rounds']}"
        measured = [item for item in result.get("updates", []) if item.get("phase") == "measured"]
        seconds = [float(item["timing"]["total_seconds"]) for item in measured]
        total = sum(seconds)
        by_combination[key] = {"measured_updates": len(measured), "measured_total_seconds": total, "t_update_seconds": total / len(seconds) if seconds else float("inf")}
    f_total = sum(float(by_combination[f"F_K{k}"]["measured_total_seconds"]) for k in (1, 4))
    er64_total = sum(float(by_combination[f"ER64_K{k}"]["measured_total_seconds"]) for k in (1, 4))
    return {"by_combination": by_combination, "joint_K1_K4": {"F": f_total, "ER64": er64_total}, "R_train": float("inf") if incomplete else (er64_total / f_total if f_total > 0 else float("inf")), "R_train_formula": "T_ER64_total / T_F_total", "incomplete_results": incomplete}


def classify_training_cost(metrics: dict[str, Any], threshold: float = GATE_THRESHOLD) -> dict[str, Any]:
    ratio = float(metrics["R_train"])
    passed = ratio <= threshold
    return {"classification": "PASS" if passed else "COST_REGRESSION", "R_train": ratio, "threshold": threshold, "ratio_semantics": "ER64_time / F_time"}


def aggregate_inference_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_combination: dict[str, Any] = {}
    for result in results:
        key = f"{result['config']}_K{result['rounds']}"
        by_combination[key] = {
            "mode_a_ms_per_token": float(result["mode_a"]["ms_per_token"]),
            "mode_b_seconds_per_window": float(result["mode_b"]["mean_seconds_per_window"]),
            "steady_state_rss_bytes": int(result["steady_state_rss_bytes"]),
        }
    ratios: dict[str, float] = {}
    for rounds in (1, 4):
        baseline = by_combination[f"F_K{rounds}"]
        candidate = by_combination[f"ER64_K{rounds}"]
        ratios[f"K{rounds}_mode_a"] = candidate["mode_a_ms_per_token"] / baseline["mode_a_ms_per_token"]
        ratios[f"K{rounds}_mode_b"] = candidate["mode_b_seconds_per_window"] / baseline["mode_b_seconds_per_window"]
    return {"by_combination": by_combination, "ratio_semantics": "ER64_time / F_time", "time_cost_fields": {"mode_a": "ms_per_token", "mode_b": "mean_seconds_per_window"}, "ER64_over_F_time_cost": ratios}


def classify_inference_cost(metrics: dict[str, Any], threshold: float = GATE_THRESHOLD, rss_limit: int = RSS_ANOMALY_BYTES) -> dict[str, Any]:
    ratios = metrics["ER64_over_F_time_cost"]
    anomaly = any(metrics["by_combination"][f"ER64_K{k}"]["steady_state_rss_bytes"] - metrics["by_combination"][f"F_K{k}"]["steady_state_rss_bytes"] > rss_limit for k in (1, 4))
    if anomaly:
        classification = "MEMORY_RUNTIME_ANOMALY"
    elif any(float(value) > threshold for value in ratios.values()):
        classification = "INFERENCE_COST_REGRESSION"
    else:
        classification = "PASS"
    return {"classification": classification, "threshold": threshold, "ratio_semantics": "ER64_time / F_time", "ratios": ratios}


def classify_memory_safety(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    if any(item.get("status") == "INCONCLUSIVE_RESOURCE" for item in values):
        classification = "INCONCLUSIVE_RESOURCE"
    elif any(item.get("status") in {"OOM", "NONFINITE"} for item in values):
        classification = "FAIL"
    elif sum(int(item.get("updates_completed", 0)) for item in values) != 24:
        classification = "FAIL"
    else:
        classification = "PASS"
    return {"classification": classification, "updates_completed": sum(int(item.get("updates_completed", 0)) for item in values)}


def global_gate_status(gates: dict[str, Any]) -> str:
    return "PASS" if all(value.get("classification") == "PASS" for value in gates.values()) else "FAIL"


def source_hashes() -> dict[str, str]:
    paths = {
        "er64_runner": HERE / "run_er64_cost_gate.py",
        "er64_adapter": HERE / "omega_fast_er64.py",
        "er64_tests": HERE / "test_er64_cost_gate.py",
        "er32_runner": ER32_GATE_DIR / "run_er32_cost_gate.py",
        "step2_runner": FASTPATH_DIR / "run_step2_benchmark.py",
        "step2_tests": FASTPATH_DIR / "test_step2_benchmark.py",
        "technical_preflight": R1_SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def artifact_metadata() -> dict[str, Any]:
    base = get_er32_base()
    metadata = base.artifact_metadata()
    metadata["source_sha256s"] = source_hashes()
    metadata["threads"] = {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads(), "configured_intraop": 4, "configured_interop": 1}
    metadata["cpu_identity"] = platform.processor() or platform.uname().processor
    return metadata


def build_cost_report(training_results: Iterable[dict[str, Any]], inference_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    training_values = list(training_results)
    inference_values = list(inference_results)
    training = aggregate_training_metrics(training_values)
    inference = aggregate_inference_metrics(inference_values)
    gates = {"TRAINING_COST": classify_training_cost(training), "INFERENCE_COST": classify_inference_cost(inference), "MEMORY_SAFETY": classify_memory_safety(training_values)}
    return {"schema": "omega-core-lm-0-er64-cost-gate-v2", "candidate": "ER64", "benchmark_rerun": False, "metadata": artifact_metadata(), "training": training, "inference": inference, "gates": gates, "global_status": global_gate_status(gates), "phase1_only": False}


def get_er32_base() -> Any:
    global er32_base
    if er32_base is None:
        configure_cpu_runtime()
        import run_er32_cost_gate as loaded_er32_base  # noqa: E402

        er32_base = loaded_er32_base
    return er32_base


def run_training_child(config: str, rounds: int, run_dir: Path, ledger_path: Path) -> dict[str, Any]:
    """Run one six-update ER64/F combination in a fresh child process."""
    get_er32_base()
    snapshots: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    result: dict[str, Any] = {"kind": "training", "config": config, "rounds": rounds, "updates_requested": TRAINING_UPDATES, "snapshots": snapshots, "updates": updates, "status": "failed", "updates_completed": 0}
    try:
        er32_base.capture_snapshot(snapshots, "process_start")
        source, document_ids, manifest, tokenizer = er32_base.load_approved_source()
        del tokenizer
        teacher = er32_base.load_teacher()
        er32_base.capture_snapshot(snapshots, "teacher_loaded")
        seed = 20260913 if rounds == 1 else 20260914
        f_reference = make_f_reference(rounds, seed)
        er32_base.capture_snapshot(snapshots, "reference_created")
        candidate = make_candidate(config, rounds, f_reference)
        del f_reference
        gc.collect()
        er32_base.capture_snapshot(snapshots, "candidate_ready_after_reference_deleted")
        optimizer = er32_base.make_adamw(candidate)
        result["parameter_count"] = sum(parameter.numel() for parameter in candidate.parameters())
        state: torch.Tensor | None = None
        for update_index, window in enumerate(er32_base.WINDOW_SCHEDULE):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if window == 0:
                state = candidate.initial_state(er32_base.BATCH, device=torch.device("cpu"))
            if state is None:
                raise RuntimeError("window 1 has no detached state")
            input_ids = source[:, :er32_base.TOKENS_PER_WINDOW] if window == 0 else source[:, er32_base.TOKENS_PER_WINDOW : 2 * er32_base.TOKENS_PER_WINDOW]
            targets = source[:, 1 : 1 + er32_base.TOKENS_PER_WINDOW] if window == 0 else source[:, 1 + er32_base.TOKENS_PER_WINDOW : 1 + 2 * er32_base.TOKENS_PER_WINDOW]
            valid_mask = torch.ones((er32_base.BATCH, er32_base.TOKENS_PER_WINDOW), dtype=torch.bool)
            timings: dict[str, float] = {}
            teacher_started = time.perf_counter()
            er32_base.capture_snapshot(snapshots, "before_teacher")
            teacher_logits = er32_base.teacher_window_logits(teacher, source, window)
            timings["teacher_forward_seconds"] = time.perf_counter() - teacher_started
            er32_base.capture_snapshot(snapshots, "after_teacher")
            recurrence_started = time.perf_counter()
            next_state, _, _, readout_states = candidate.recur_states(input_ids, state)
            timings["student_recurrence_forward_seconds"] = time.perf_counter() - recurrence_started
            er32_base.capture_snapshot(snapshots, "after_student_forward")
            loss_started = time.perf_counter()
            losses = er32_base.chunked_original_loss(candidate, readout_states, teacher_logits, targets, valid_mask)
            timings["vocab_projection_loss_seconds"] = time.perf_counter() - loss_started
            er32_base.capture_snapshot(snapshots, "after_loss")
            backward_started = time.perf_counter()
            losses["total"].backward()
            timings["backward_seconds"] = time.perf_counter() - backward_started
            er32_base.capture_snapshot(snapshots, "after_backward")
            er32_base.assert_finite(candidate, losses)
            clip_started = time.perf_counter()
            pre_clip = er32_base.grad_norm(candidate)
            returned_clip = float(torch.nn.utils.clip_grad_norm_(candidate.parameters(), er32_base.CLIP_NORM).item())
            timings["clipping_seconds"] = time.perf_counter() - clip_started
            er32_base.capture_snapshot(snapshots, "after_clip")
            optimizer_started = time.perf_counter()
            optimizer.step()
            timings["adamw_seconds"] = time.perf_counter() - optimizer_started
            er32_base.capture_snapshot(snapshots, "after_optimizer_step")
            inventory = er32_base.tensor_inventory(candidate, optimizer) if update_index == 0 else None
            state = next_state.detach()
            del teacher_logits, readout_states, next_state, losses
            gc.collect()
            er32_base.capture_snapshot(snapshots, "after_update_cleanup")
            timings["total_seconds"] = time.perf_counter() - started
            timings["ledger_seconds"] = 0.0
            record = {"kind": "training_update", "config": config, "rounds": rounds, "update": update_index, "window": window, "phase": "warmup" if update_index < er32_base.WARMUP_UPDATES else "measured", "valid_tokens": er32_base.BATCH * er32_base.TOKENS_PER_WINDOW, "document_ids": document_ids, "manifest": manifest, "pre_clip_grad_norm": pre_clip, "clip_returned_norm": returned_clip, "timing": timings, "tensor_inventory_after_first_optimizer_step": inventory, "snapshots": [item for item in snapshots if item.get("phase") in er32_base.UPDATE_SNAPSHOT_PHASES][-8:]}
            ledger_started = time.perf_counter()
            append_seconds = er32_base.append_ledger(ledger_path, record)
            timings["ledger_seconds"] = time.perf_counter() - ledger_started + append_seconds
            timings["total_seconds"] = time.perf_counter() - started
            updates.append(record)
            result["updates_completed"] = len(updates)
        result["status"] = "completed"
        result["tensor_inventory"] = updates[0]["tensor_inventory_after_first_optimizer_step"]
        result["minimum_available_system_bytes"] = min(int(item["available_system_bytes"]) for item in snapshots if item.get("available_system_bytes") is not None)
    except er32_base.ResourceGuard as error:
        result["status"] = "INCONCLUSIVE_RESOURCE"
        result["resource_guard"] = {"phase": error.phase, "snapshot": error.snapshot}
    except er32_base.NonFinite as error:
        result["status"] = "NONFINITE"
        result["error"] = str(error)
    except RuntimeError as error:
        result["status"] = "OOM" if "out of memory" in str(error).lower() else "failed"
        result["error"] = str(error)
    except Exception as error:  # pragma: no cover
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def run_inference_child(config: str, rounds: int) -> dict[str, Any]:
    get_er32_base()
    source, _, _, tokenizer = er32_base.load_approved_source()
    del tokenizer
    seed = 20260913 if rounds == 1 else 20260914
    candidate = make_candidate(config, rounds, make_f_reference(rounds, seed)).eval()
    tokens = source[:1]
    with torch.inference_mode():
        state = candidate.initial_state(1, device=torch.device("cpu"))
        for index in range(32):
            state, _, _, _ = candidate.recur_states(tokens[:, index : index + 1], state)
        started = time.perf_counter()
        for index in range(32, 288):
            state, _, _, _ = candidate.recur_states(tokens[:, index : index + 1], state)
        mode_a_seconds = time.perf_counter() - started
        warmup_state = candidate.initial_state(1, device=torch.device("cpu"))
        candidate.recur_states(tokens[:, :256], warmup_state)
        mode_b_seconds: list[float] = []
        for _ in range(3):
            initial = candidate.initial_state(1, device=torch.device("cpu"))
            started = time.perf_counter()
            candidate.recur_states(tokens[:, :256], initial)
            mode_b_seconds.append(time.perf_counter() - started)
    mean_window = sum(mode_b_seconds) / len(mode_b_seconds)
    return {"kind": "inference", "config": config, "rounds": rounds, "mode_a": {"tokens": 256, "seconds": mode_a_seconds, "tokens_per_second": 256 / mode_a_seconds, "ms_per_token": mode_a_seconds * 1000 / 256}, "mode_b": {"tokens": 256, "seconds_per_window": mode_b_seconds, "mean_seconds_per_window": mean_window, "mean_tokens_per_second": 256 / mean_window}, "steady_state_rss_bytes": er32_base.memory_snapshot()["rss_bytes"]}


def child_authorized(token: str | None) -> bool:
    expected = os.environ.get(CHILD_TOKEN_ENV, "")
    return bool(token and expected and hmac.compare_digest(token, expected))


def spawn_child(command: list[str], token: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment[CHILD_TOKEN_ENV] = token
    completed = subprocess.run(command, cwd=HERE, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"child failed: {completed.stderr[-1000:]}")
    for line in reversed(completed.stdout.splitlines()):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("child output is not an object")
            return value
    raise RuntimeError("child produced no JSON result")


def run_training_children(run_dir: Path, ledger_path: Path, *, spawn: Any = spawn_child) -> list[dict[str, Any]]:
    token = secrets.token_hex(32)
    results = []
    for config, rounds in COMBINATIONS:
        command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--child-training", "--child-token", token, "--config", config, "--rounds", str(rounds), "--run-dir", str(run_dir), "--ledger", str(ledger_path)]
        results.append(spawn(command, token))
    return results


def run_inference_children(*, spawn: Any = spawn_child) -> list[dict[str, Any]]:
    token = secrets.token_hex(32)
    results = []
    for config, rounds in COMBINATIONS:
        command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--child-inference", "--child-token", token, "--config", config, "--rounds", str(rounds)]
        results.append(spawn(command, token))
    return results


def run_full(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = run_dir / "ledger.jsonl"
    training = run_training_children(run_dir, ledger)
    inference = run_inference_children()
    report = build_cost_report(training, inference)
    report.update({"run_dir": run_dir.as_posix(), "training_results": training, "inference_results": inference, "phase1_only": False})
    write_self_hashed_json(run_dir / "cost_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child-training", action="store_true")
    parser.add_argument("--child-inference", action="store_true")
    parser.add_argument("--child-token")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--rounds", type=int, choices=(1, 4))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-er64-cost-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.execute and child_authorized(args.child_token) and args.config and args.rounds:
        if args.child_training and args.run_dir and args.ledger:
            print(json.dumps(run_training_child(args.config, args.rounds, args.run_dir, args.ledger), sort_keys=True))
            return 0
        if args.child_inference:
            print(json.dumps(run_inference_child(args.config, args.rounds), sort_keys=True))
            return 0
    if args.smoke:
        parser.error("synthetic smoke execution is covered by tests; no corpus is loaded")
    if not (args.full and args.confirm_er64_cost_gate):
        parser.error("real ER64 cost-gate execution requires --full --confirm-er64-cost-gate")
    print(json.dumps(run_full(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
