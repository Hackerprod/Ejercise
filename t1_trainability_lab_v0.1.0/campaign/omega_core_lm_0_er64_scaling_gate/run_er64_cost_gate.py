"""Phase 2 ER64 cost-gate mechanics.

Phase 1 exposes the real candidate factory, corrected cost formulas, report
artifact self-hashing, and synthetic checks. Real corpus execution is reserved
for explicit follow-up authorization after review of this phase.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
ER64_ADAPTER_DIR = HERE
R1_SCRIPTS_DIR = CAMPAIGN_ROOT.parent / "scripts"
FASTPATH_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
sys.path.insert(0, str(ER64_ADAPTER_DIR))
sys.path.insert(0, str(FASTPATH_DIR))
sys.path.insert(0, str(R1_SCRIPTS_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402
from omega_fast_er64 import OmegaCoreLMFastER64  # noqa: E402
from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical  # noqa: E402


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
    by_combination: dict[str, dict[str, Any]] = {}
    for result in results:
        key = f"{result['config']}_K{result['rounds']}"
        measured = [item for item in result.get("updates", []) if item.get("phase") == "measured"]
        seconds = [float(item["timing"]["total_seconds"]) for item in measured]
        total = sum(seconds)
        by_combination[key] = {"measured_updates": len(measured), "measured_total_seconds": total, "t_update_seconds": total / len(seconds) if seconds else float("inf")}
    f_total = sum(float(by_combination[f"F_K{k}"]["measured_total_seconds"]) for k in (1, 4))
    er64_total = sum(float(by_combination[f"ER64_K{k}"]["measured_total_seconds"]) for k in (1, 4))
    return {"by_combination": by_combination, "joint_K1_K4": {"F": f_total, "ER64": er64_total}, "R_train": er64_total / f_total if f_total > 0 else float("inf"), "R_train_formula": "T_ER64_total / T_F_total"}


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


def build_cost_report(training_results: Iterable[dict[str, Any]], inference_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    training = aggregate_training_metrics(training_results)
    inference = aggregate_inference_metrics(inference_results)
    return {"schema": "omega-core-lm-0-er64-cost-gate-phase1-v1", "candidate": "ER64", "benchmark_rerun": False, "training": training, "inference": inference, "gates": {"TRAINING_COST": classify_training_cost(training), "INFERENCE_COST": classify_inference_cost(inference)}, "phase1_only": True}


def main(argv: list[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-er64-cost-gate", action="store_true")
    parser.error("Phase 2 real execution is reserved for explicit follow-up authorization; use tests for Phase 1")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
