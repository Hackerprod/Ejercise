"""OMEGA readout-cache survival audit.

This unit defines a paired, single-physical-core protocol. Real positive-control
and 256-block execution require separate explicit confirmations; tests and
``--smoke`` never load checkpoints or run the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

import psutil
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
LAB_ROOT = CAMPAIGN_ROOT.parent
REPO_ROOT = LAB_ROOT.parent
TOPOLOGY_SOURCE = REPO_ROOT / "Trash" / "HANDOFF_CONTEXT_2026-09-04.md"
FASTPATH_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
ER32_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er32_integration_and_cost_gate"
ER64_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er64_scaling_gate"
for path in (FASTPATH_DIR, ER32_DIR, ER64_DIR):
    sys.path.insert(0, str(path))


VARIANTS = ("F", "ER32", "ER64")
ARCHITECTURES = ("Zen5", "Zen5c")
BLOCK_COUNT = 256
WARMUP_CALLS = 16
TOKEN_BATCH = 1
TOKEN_COUNT = 1
DIMENSION = 128
SLOTS = 8
MIN_COLD_BYTES = 64 * 1024 * 1024
LLC_BYTES = 8 * 1024 * 1024
POSITIVE_CONTROL_BYTES = max(MIN_COLD_BYTES, 4 * LLC_BYTES)
BLOCK_ORDER_SEED = 20260918
STATE_ALLOWLIST = {
    "MEASUREMENT_COMPLETE",
    "UNRESOLVED_DENOMINATOR",
    "MEASUREMENT_RESOLUTION_INSUFFICIENT",
    "TECHNICAL_FAIL",
}
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"


@dataclass(frozen=True)
class PhysicalCoreTarget:
    architecture: str
    physical_core: int
    logical_processor: int
    efficiency_class: int


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_self_hashed_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    digest = sha256_bytes(canonical_json(unsigned))
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = canonical_json(written)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    persisted = path.read_bytes()
    parsed = json.loads(persisted.decode("utf-8"))
    stored = parsed.pop("artifact_self_hash", None)
    parsed["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    if stored != digest or persisted != encoded or sha256_bytes(canonical_json(parsed)) != digest:
        raise RuntimeError("audit artifact self-hash verification failed")
    return digest


def load_validated_topology(path: Path = TOPOLOGY_SOURCE) -> list[PhysicalCoreTarget]:
    """Parse the checked-in topology handoff; never invent CPU IDs."""
    text = path.read_text(encoding="utf-8")
    classic = re.search(r"core0\s*=\s*Zen5.*?\[(\d+),(\d+)\]", text)
    compact = re.search(r"cores\s+1-3\s*=\s*Zen5c.*?\[(\d+),(\d+)\],\[(\d+),(\d+)\],\[(\d+),(\d+)\]", text)
    if not classic or not compact:
        raise RuntimeError(f"validated Zen5/Zen5c topology not found in {path}")
    targets = [PhysicalCoreTarget("Zen5", 0, int(classic.group(1)), 1)]
    compact_values = [int(value) for value in compact.groups()]
    for core, pair_start in enumerate(range(0, len(compact_values), 2), start=1):
        targets.append(PhysicalCoreTarget("Zen5c", core, compact_values[pair_start], 0))
    if len(targets) != 4 or len({target.physical_core for target in targets}) != 4:
        raise RuntimeError("validated topology must expose one Classic and three Compact physical cores")
    return targets


def validate_single_core_target(target: PhysicalCoreTarget) -> None:
    if target.architecture not in ARCHITECTURES or target.logical_processor < 0:
        raise ValueError("invalid topology target")
    if target.architecture == "Zen5" and target.physical_core != 0:
        raise ValueError("Classic target must be physical core 0")
    if target.architecture == "Zen5c" and target.physical_core not in {1, 2, 3}:
        raise ValueError("Compact target must be physical core 1, 2, or 3")


def balanced_block_order(seed: int = BLOCK_ORDER_SEED) -> list[tuple[str, str, int]]:
    targets = load_validated_topology()
    rows = [(target.architecture, f"physical_{target.physical_core}", block) for target in targets for block in range(BLOCK_COUNT)]
    random.Random(seed).shuffle(rows)
    return rows


def differences(samples: Iterable[dict[str, float]]) -> dict[str, list[float]]:
    cold = [float(item["cold"]) for item in samples]
    warm = [float(item["warm"]) for item in samples]
    post = [float(item["posthead"]) for item in samples]
    return {"D_cold": [c - w for c, w in zip(cold, warm)], "D_post": [p - w for p, w in zip(post, warm)]}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def ratio_of_medians(d_cold: list[float], d_post: list[float], d_negative: list[float], *, bootstrap_seed: int = BLOCK_ORDER_SEED, bootstrap_rounds: int = 2000) -> dict[str, Any]:
    if len(d_cold) != len(d_post) or not d_cold:
        raise ValueError("paired block differences required")
    cold_median = float(median(d_cold))
    post_median = float(median(d_post))
    rng = random.Random(bootstrap_seed)
    bootstrap_cold: list[float] = []
    bootstrap_ratios: list[float] = []
    blocks = list(zip(d_cold, d_post))
    for _ in range(bootstrap_rounds):
        sample = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        cold = float(median([item[0] for item in sample]))
        post = float(median([item[1] for item in sample]))
        bootstrap_cold.append(cold)
        if cold != 0:
            bootstrap_ratios.append(post / cold)
    cold_lower = percentile(bootstrap_cold, 0.025)
    denominator_resolved = cold_median > 0 and cold_lower > 0 and cold_median > percentile([abs(value) for value in d_negative], 0.95)
    if not denominator_resolved:
        return {"R_eviction": None, "S_survival": None, "classification": "UNRESOLVED_DENOMINATOR", "median_D_cold": cold_median, "median_D_post": post_median, "bootstrap_median_D_cold_lower_95": cold_lower, "negative_control_abs_p95": percentile([abs(value) for value in d_negative], 0.95)}
    ratio = post_median / cold_median
    return {"R_eviction": ratio, "S_survival": 1.0 - ratio, "classification": "MEASUREMENT_COMPLETE", "median_D_cold": cold_median, "median_D_post": post_median, "bootstrap_median_D_cold_lower_95": cold_lower, "negative_control_abs_p95": percentile([abs(value) for value in d_negative], 0.95), "bootstrap_R_eviction_lower_95": percentile(bootstrap_ratios, 0.025), "bootstrap_R_eviction_upper_95": percentile(bootstrap_ratios, 0.975)}


def choose_resolution_state(probe_seconds: list[float], positive_control_separation: float, *, resolution_floor_seconds: float = 1e-6) -> str:
    if not probe_seconds or max(probe_seconds) < resolution_floor_seconds or positive_control_separation <= resolution_floor_seconds:
        return "MEASUREMENT_RESOLUTION_INSUFFICIENT"
    return "MEASUREMENT_COMPLETE"


def touch_cold_buffer(cold_buffer: bytearray) -> None:
    for index in range(0, len(cold_buffer), 64):
        cold_buffer[index] = (cold_buffer[index] + 1) & 0xFF


def core_probe(model: Any, tokens: Any, state: Any, *, cold_buffer: bytearray | None = None, tiny_operation: Callable[[], None] | None = None) -> float:
    """Time recurrent core only; optional flush/control runs before timer."""
    if cold_buffer is not None:
        touch_cold_buffer(cold_buffer)
    if tiny_operation is not None:
        tiny_operation()
    started = time.perf_counter_ns()
    model.recur_states(tokens, state)
    return (time.perf_counter_ns() - started) / 1_000_000_000


def run_readout_head(model: Any, state: Any) -> None:
    """Consume full variant readout so compiler cannot discard it."""
    logits = model.logits_from_states(state)
    logits.argmax(dim=-1)


def block_measurement(model: Any, tokens: Any, state_factory: Callable[[], Any], cold_bytes: int = POSITIVE_CONTROL_BYTES) -> dict[str, float]:
    cold_buffer = bytearray(cold_bytes)
    cold = core_probe(model, tokens, state_factory(), cold_buffer=cold_buffer)
    warm_state = state_factory()
    for _ in range(WARMUP_CALLS):
        core_probe(model, tokens, warm_state)
    warm = core_probe(model, tokens, warm_state)
    run_readout_head(model, warm_state)
    post = core_probe(model, tokens, warm_state)
    return {"cold": cold, "warm": warm, "posthead": post}


def positive_control(output_dir: Path) -> dict[str, Any]:
    """Real resolution control; caller must explicitly authorize it."""
    started = time.perf_counter_ns()
    buffer = bytearray(POSITIVE_CONTROL_BYTES)
    cold_samples: list[float] = []
    warm_samples: list[float] = []
    for _ in range(8):
        touch_started = time.perf_counter_ns()
        for index in range(0, len(buffer), 64):
            buffer[index] = (buffer[index] + 1) & 0xFF
        elapsed = (time.perf_counter_ns() - touch_started) / 1_000_000_000
        cold_samples.append(elapsed)
        touch_started = time.perf_counter_ns()
        for index in range(0, len(buffer), 64):
            buffer[index] = (buffer[index] + 1) & 0xFF
        warm_samples.append((time.perf_counter_ns() - touch_started) / 1_000_000_000)
    separation = abs(median(cold_samples) - median(warm_samples))
    result = {"schema": "omega-readout-cache-survival-positive-control-v1", "status": choose_resolution_state(cold_samples + warm_samples, separation), "cold_buffer_bytes": len(buffer), "cold_median_seconds": median(cold_samples), "warm_median_seconds": median(warm_samples), "separation_seconds": separation, "elapsed_seconds": (time.perf_counter_ns() - started) / 1_000_000_000, "topology_source": TOPOLOGY_SOURCE.as_posix()}
    write_self_hashed_json(output_dir / "positive_control.json", result)
    return result


def set_single_core_affinity(logical_processor: int) -> list[int] | None:
    process = psutil.Process()
    previous = process.cpu_affinity()
    process.cpu_affinity([logical_processor])
    if process.cpu_affinity() != [logical_processor]:
        raise RuntimeError("single-logical-processor affinity was not applied")
    return previous


def build_variants() -> dict[str, Any]:
    from run_er64_cost_gate import make_f_reference
    from omega_fast_er32 import OmegaCoreLMFastER32
    from omega_fast_er64 import OmegaCoreLMFastER64

    f_reference = make_f_reference(1, 20260918).eval()
    er32 = OmegaCoreLMFastER32.from_f_reference(f_reference, experimental_seed=20260917).eval()
    er64 = OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=20260917).eval()
    return {"F": f_reference, "ER32": er32, "ER64": er64}


def measure_variant_block(variant: str, block: int) -> dict[str, Any]:
    variants = build_variants()
    model = variants[variant]
    tokens = torch.tensor([[block % model.vocab_size]], dtype=torch.long)
    state_factory = lambda: model.initial_state(TOKEN_BATCH, device=torch.device("cpu"))
    samples = block_measurement(model, tokens, state_factory)
    negative_state = state_factory()
    negative = core_probe(model, tokens, negative_state, tiny_operation=lambda: None)
    return {"variant": variant, "block": block, **samples, "negative": negative, "rss_bytes": psutil.Process().memory_info().rss}


def spawn_block(command: list[str], token: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["OMEGA_CACHE_SURVIVAL_TOKEN"] = token
    completed = subprocess.run(command, cwd=HERE, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"cache-survival child failed: {completed.stderr[-1000:]}")
    for line in reversed(completed.stdout.splitlines()):
        if line.strip():
            result = json.loads(line)
            if not isinstance(result, dict):
                raise RuntimeError("cache-survival child result is not an object")
            return result
    raise RuntimeError("cache-survival child produced no JSON")


def child_authorized(token: str | None) -> bool:
    expected = os.environ.get("OMEGA_CACHE_SURVIVAL_TOKEN", "")
    return bool(token and expected and hmac.compare_digest(token, expected))


def aggregate_variant_samples(samples: list[dict[str, Any]], positive: dict[str, Any]) -> dict[str, Any]:
    differences_result = differences(samples)
    analysis = ratio_of_medians(differences_result["D_cold"], differences_result["D_post"], [float(item["negative"]) - float(item["warm"]) for item in samples])
    if positive.get("status") == "MEASUREMENT_RESOLUTION_INSUFFICIENT":
        analysis["classification"] = "MEASUREMENT_RESOLUTION_INSUFFICIENT"
        analysis["R_eviction"] = None
        analysis["S_survival"] = None
    return {"sample_count": len(samples), "statistics": analysis, "samples": samples}


def run_full(output_dir: Path, positive: dict[str, Any]) -> dict[str, Any]:
    targets = load_validated_topology()
    order = balanced_block_order()
    token = os.urandom(32).hex()
    raw_path = output_dir / "raw_samples.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for architecture, physical_name, block in order:
        target = next(item for item in targets if item.architecture == architecture and f"physical_{item.physical_core}" == physical_name)
        for variant in VARIANTS:
            command = [sys.executable, str(Path(__file__).resolve()), "--child-block", "--child-token", token, "--variant", variant, "--block", str(block), "--logical-processor", str(target.logical_processor), "--physical-core", str(target.physical_core), "--architecture", architecture]
            sample = spawn_block(command, token)
            key = f"{architecture}/physical_{target.physical_core}/{variant}"
            sample.update({"architecture": architecture, "physical_core": target.physical_core, "logical_processor": target.logical_processor})
            grouped.setdefault(key, []).append(sample)
            output_dir.mkdir(parents=True, exist_ok=True)
            with raw_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
    report = {"schema": "omega-readout-cache-survival-audit-v1", "status": "MEASUREMENT_COMPLETE", "architectures": ARCHITECTURES, "variants": VARIANTS, "block_count": BLOCK_COUNT, "warmup_calls": WARMUP_CALLS, "paired_conditions_same_child": True, "topology_source": TOPOLOGY_SOURCE.as_posix(), "positive_control": positive, "groups": {key: aggregate_variant_samples(value, positive) for key, value in grouped.items()}, "source_sha256": sha256_file(Path(__file__).resolve())}
    write_self_hashed_json(output_dir / "cache_survival_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--confirm-positive-control", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-omega-readout-cache-survival-audit", action="store_true")
    parser.add_argument("--child-block", action="store_true")
    parser.add_argument("--child-token")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--block", type=int)
    parser.add_argument("--logical-processor", type=int)
    parser.add_argument("--physical-core", type=int)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.child_block:
        if not child_authorized(args.child_token) or args.variant is None or args.block is None or args.logical_processor is None or args.physical_core is None or args.architecture is None:
            parser.error("unauthorized or incomplete child block")
        target = PhysicalCoreTarget(args.architecture, args.physical_core, args.logical_processor, 1 if args.architecture == "Zen5" else 0)
        validate_single_core_target(target)
        set_single_core_affinity(target.logical_processor)
        print(json.dumps(measure_variant_block(args.variant, args.block), sort_keys=True))
        return 0
    if args.smoke:
        print(json.dumps({"status": "MEASUREMENT_COMPLETE", "real_data_loaded": False, "blocks": 0}, sort_keys=True))
        return 0
    if args.positive_control:
        if not args.confirm_positive_control:
            parser.error("positive control requires --confirm-positive-control")
        print(json.dumps(positive_control(args.output_dir), indent=2, sort_keys=True))
        return 0
    if not (args.full and args.confirm_omega_readout_cache_survival_audit):
        parser.error("real audit requires --full --confirm-omega-readout-cache-survival-audit")
    positive_path = args.output_dir / "positive_control.json"
    if not positive_path.is_file():
        parser.error("real audit requires completed positive_control.json from prior explicit run")
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    if positive.get("status") != "MEASUREMENT_COMPLETE":
        parser.error("positive control did not establish measurement resolution")
    print(json.dumps(run_full(args.output_dir, positive), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
