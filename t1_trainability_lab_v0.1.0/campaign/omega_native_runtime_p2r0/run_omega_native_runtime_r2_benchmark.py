"""Fresh-process-compatible R2 end-to-end benchmark harness.

Default execution runs pure synthetic ratio canaries only. Real preflight and
stable phases load R1's model/data setup only after explicit authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from omega_native_runtime_r2_canaries import (
    K_NEUTRAL_LIMIT,
    K_PASS_LIMIT,
    JOINT_NEUTRAL_LIMIT,
    JOINT_PASS_LIMIT,
    STRONG_LIMIT,
    STOP_LIMIT,
    candidate_baseline_ratio,
    classify_joint,
    run_canaries,
    should_stop,
)


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
ROUTES = ("pytorch-k1", "pytorch-k4", "native-k1", "native-k4")
R1_SEED = 20260913
WINDOW_TOKENS = 256
DEFAULT_UPDATES = 6
DEFAULT_WARMUP_UPDATES = 1


class RealExecutionAuthorizationError(RuntimeError):
    pass


def schedule(updates: int) -> list[dict[str, int]]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    return [{"update": update, "window": update % 2, "pair": update // 2} for update in range(updates)]


def route_parts(route: str) -> tuple[bool, int]:
    if route not in ROUTES:
        raise ValueError(f"unsupported route: {route}")
    native = route.startswith("native-")
    return native, int(route.rsplit("k", 1)[1])


def require_real_authorization(confirmed: bool) -> None:
    if not confirmed:
        raise RealExecutionAuthorizationError(
            "R2 preflight/stable phases require --confirm-real-execution"
        )


def policy_metadata() -> dict[str, Any]:
    return {
        "ratio_definition": "candidate / baseline",
        "joint_limits": {
            "strong_r_joint_max": STRONG_LIMIT,
            "pass_r_joint_max": JOINT_PASS_LIMIT,
            "neutral_r_joint_max": JOINT_NEUTRAL_LIMIT,
        },
        "per_k_limits": {
            "strong_pass_max": K_PASS_LIMIT,
            "neutral_max": K_NEUTRAL_LIMIT,
        },
        "stop_ratio_strictly_greater_than": STOP_LIMIT,
        "uncovered_tradeoffs": "UNCLASSIFIED",
    }


def _route_total_seconds(report: Mapping[str, Any]) -> float | None:
    updates = report.get("updates")
    if not isinstance(updates, list) or not updates:
        return None
    values: list[float] = []
    for update in updates:
        if not isinstance(update, Mapping):
            return None
        timing = update.get("timing_seconds")
        if not isinstance(timing, Mapping) or not isinstance(timing.get("total_update"), (int, float)):
            return None
        values.append(float(timing["total_update"]))
    return sum(values)


def aggregate_performance(route_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate stable route reports without importing runtime dependencies."""
    required = ROUTES
    missing = [route for route in required if route not in route_reports]
    totals = {route: _route_total_seconds(route_reports[route]) for route in required if route in route_reports}
    invalid = [route for route, total in totals.items() if total is None]
    if missing or invalid:
        return {
            "status": "NOT_READY",
            "classification": None,
            "missing_routes": missing,
            "invalid_routes": invalid,
            "route_totals_seconds": totals,
            "r_k": {},
            "r_joint": None,
            "stop": None,
        }
    native_k1 = float(totals["native-k1"])
    native_k4 = float(totals["native-k4"])
    pytorch_k1 = float(totals["pytorch-k1"])
    pytorch_k4 = float(totals["pytorch-k4"])
    r_k = {
        "K1": candidate_baseline_ratio(native_k1, pytorch_k1),
        "K4": candidate_baseline_ratio(native_k4, pytorch_k4),
    }
    r_joint = candidate_baseline_ratio(native_k1 + native_k4, pytorch_k1 + pytorch_k4)
    return {
        "status": "READY",
        "classification": classify_joint(r_joint, tuple(r_k.values())),
        "missing_routes": [],
        "invalid_routes": [],
        "route_totals_seconds": totals,
        "r_k": r_k,
        "r_joint": r_joint,
        "stop": any(should_stop(value) for value in (r_joint, *r_k.values())),
    }


def _load_r1_modules() -> tuple[Any, Any, Any]:
    """Load runtime dependencies only from an explicitly requested real phase."""
    p0_dir = CAMPAIGN_ROOT / "omega_native_runtime_p0"
    ce_dir = CAMPAIGN_ROOT / "omega_ce_only_baseline"
    for path in (p0_dir, ce_dir, HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_omega_native_runtime_p0 as p0  # noqa: PLC0415
    import run_omega_ce_only_baseline as ce  # noqa: PLC0415
    import omega_recurrent_production_bridge as bridge  # noqa: PLC0415

    return p0, ce, bridge


def prepare_native_model(model: Any) -> Any:
    """Validate fused prelude state view once before measured updates."""
    dimension = int(model.dimension)
    state_part_weight = model.prelude.weight[:, dimension:]
    expected_shape = (int(model.slots) * dimension, dimension)
    if tuple(state_part_weight.shape) != expected_shape:
        raise RuntimeError(f"native R2 state view shape {tuple(state_part_weight.shape)} != {expected_shape}")
    expected_stride = (2 * dimension, 1)
    if tuple(state_part_weight.stride()) != expected_stride:
        raise RuntimeError(
            f"native R2 state view must be real fused view with stride {expected_stride}; "
            f"got {tuple(state_part_weight.stride())}"
        )
    expected_data_ptr = int(model.prelude.weight.data_ptr()) + dimension * int(model.prelude.weight.element_size())
    if int(state_part_weight.data_ptr()) != expected_data_ptr:
        raise RuntimeError("native R2 state view data_ptr is not fused prelude.weight[:, dimension:]")
    return state_part_weight


def _native_forward(model: Any, token_ids: Any, previous_state: Any, state_part_weight: Any, bridge: Any) -> tuple[Any, Any, Any]:
    import torch.nn.functional as F  # noqa: PLC0415

    dimension = int(model.dimension)
    token_part = F.linear(model.embedding(token_ids), model.prelude.weight[:, :dimension], model.prelude.bias)
    block = model.blocks[0]
    next_state, readout_states = bridge.apply(
        token_part,
        previous_state,
        state_part_weight,
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
        int(model.rounds),
    )
    student_logits = model.logits_from_projected(model.project(readout_states))
    return next_state, student_logits, readout_states


def _reference_forward(model: Any, token_ids: Any, previous_state: Any, p0: Any) -> tuple[Any, Any, Any]:
    next_state, student_logits, boundaries, _ = p0._student_forward_timed(model, token_ids, previous_state)
    return next_state, student_logits, boundaries["recurrent_readout_boundary"]


def _rss_bytes() -> int:
    try:
        import psutil  # noqa: PLC0415

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


def _dll_metadata(bridge: Any) -> dict[str, Any]:
    """Capture identity and build/profile metadata for exact loaded DLL."""
    path = bridge.loaded_library_path()
    stat = path.stat()
    profile_available = bridge.profile_abi_available()
    profile_snapshot = bridge.profile_stats() if profile_available else None
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "build_metadata": {
            "configuration": "Release",
            "compiler_flags": ["/O2", "/fp:precise", "/MD"],
            "profile_compile_flag": "OMEGA_PROFILE_INTERNAL",
            "profile_compiled": bool(profile_snapshot and profile_snapshot["compiled"]),
            "profile_runtime_enabled_at_capture": bool(profile_snapshot and profile_snapshot["enabled"]),
            "explicit_isa": None,
            "native_threading": "none (no OpenMP/std::thread)",
        },
    }


def _load_inputs(p0: Any, manifest_path: Path, cache_file: Path) -> tuple[list[dict[str, Any]], Any, Any, Any]:
    p0.integration.verify_provenance(manifest_path, cache_file=cache_file)
    manifest, resolved_cache = p0.hidden.hidden_cache_load_manifest(manifest_path)
    payload, _, _ = p0.hidden.preload_hidden_cache(resolved_cache, manifest)
    teacher_weight, teacher_bias = p0.hidden.load_lm_head(manifest_path.parent, manifest["lm_head"])
    _, documents, _ = p0.hidden.base.load_frozen_train_documents()
    return documents, payload, teacher_weight, teacher_bias


def _teacher_logits(p0: Any, payload: Any, teacher_weight: Any, teacher_bias: Any, positions: Sequence[int], window: int) -> Any:
    import torch  # noqa: PLC0415

    hidden_states = torch.from_numpy(payload[window, list(positions)].copy()).float()
    return p0.hidden.hidden_to_logits(hidden_states, teacher_weight, teacher_bias)


def _run_update(
    *,
    model: Any,
    optimizer: Any,
    inputs: Any,
    targets: Any,
    previous_state: Any,
    teacher_logits: Any,
    native: bool,
    p0: Any,
    ce: Any,
    bridge: Any,
    state_part_weight: Any,
) -> tuple[dict[str, Any], Any]:
    bridge_before = bridge.runtime_stats()
    rss_before = _rss_bytes()
    optimizer.zero_grad(set_to_none=True)
    total_started = time.perf_counter()
    if native:
        next_state, student_logits, _ = _native_forward(model, inputs, previous_state, state_part_weight, bridge)
    else:
        next_state, student_logits, _ = _reference_forward(model, inputs, previous_state, p0)
    losses = p0.hidden.base.distillation_loss(student_logits, teacher_logits, targets)
    losses["total"].backward()
    torch = __import__("torch")
    clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
    optimizer.step()
    total_seconds = time.perf_counter() - total_started
    bridge_after = bridge.runtime_stats()
    rss_after = _rss_bytes()
    delta = {
        name: bridge_after[name] - bridge_before[name]
        for name in (
            "workspace_allocations",
            "workspace_reuses",
            "native_forward_c_abi_seconds",
            "native_backward_c_abi_seconds",
            "native_forward_bridge_boundary_seconds",
            "native_backward_bridge_boundary_seconds",
        )
    }
    bridge_boundary = delta["native_forward_bridge_boundary_seconds"] + delta["native_backward_bridge_boundary_seconds"]
    c_abi = delta["native_forward_c_abi_seconds"] + delta["native_backward_c_abi_seconds"]
    return {
        "timing_seconds": {
            "total_update": total_seconds,
            "native_forward_bridge_boundary": delta["native_forward_bridge_boundary_seconds"] if native else None,
            "native_backward_bridge_boundary": delta["native_backward_bridge_boundary_seconds"] if native else None,
            "native_forward_c_abi": delta["native_forward_c_abi_seconds"] if native else None,
            "native_backward_c_abi": delta["native_backward_c_abi_seconds"] if native else None,
            "bridge_overhead": bridge_boundary - c_abi if native else None,
        },
        "workspace_allocation_reuse": {
            "allocations": int(delta["workspace_allocations"]),
            "reuses": int(delta["workspace_reuses"]),
        },
        "rss": {"before_bytes": rss_before, "after_bytes": rss_after},
        "loss": float(losses["total"].detach().item()),
        "clip_norm": clip_norm,
    }, next_state.detach()


def _profile_report(stats: Mapping[str, Any]) -> dict[str, Any]:
    totals_seconds: dict[str, dict[str, float]] = {}
    means_seconds: dict[str, dict[str, float]] = {}
    stage_calls: dict[str, dict[str, int]] = {}
    stage_shares: dict[str, dict[str, float]] = {}
    depth_share: dict[str, float] = {}
    for direction in ("forward", "backward"):
        direction_stats = stats[direction]
        stages = direction_stats["stages"]
        totals_seconds[direction] = {name: float(stage["seconds"]) for name, stage in stages.items()}
        stage_calls[direction] = {name: int(stage["calls"]) for name, stage in stages.items()}
        means_seconds[direction] = {
            name: (float(stage["seconds"]) / int(stage["calls"]) if int(stage["calls"]) else 0.0)
            for name, stage in stages.items()
        }
        total = sum(totals_seconds[direction].values())
        stage_shares[direction] = {
            name: (seconds / total if total else 0.0)
            for name, seconds in totals_seconds[direction].items()
        }
        depth_total = sum(
            totals_seconds[direction].get(name, 0.0)
            for name in ("depth_embedding_pairwise_reduction", "depth_embedding_carry")
        )
        depth_share[direction] = depth_total / total if total else 0.0
    return {
        "enabled": bool(stats["enabled"]),
        "compiled": bool(stats["compiled"]),
        "native_function_calls": {
            "forward": int(stats["forward"]["calls"]),
            "backward": int(stats["backward"]["calls"]),
        },
        "stage_calls": stage_calls,
        "stage_totals_seconds": totals_seconds,
        "stage_means_seconds_per_stage_call": means_seconds,
        "stage_shares": stage_shares,
        "depth_share": depth_share,
        "clock": "std::chrono::steady_clock",
        "overhead_note": "Profile build only; enabled collection adds clock reads and mutex-protected aggregation. Not performance evidence.",
    }


def run_preflight(*, route: str) -> dict[str, Any]:
    """Validate policy and fresh model construction without corpus loading."""
    p0, ce, bridge = _load_r1_modules()
    native, k = route_parts(route)
    policy = ce.validate_policy()
    model = ce.fresh_model(R1_SEED, k)
    state_part_weight = prepare_native_model(model) if native else None
    result = {
        "phase": "preflight",
        "route": route,
        "native": native,
        "K": k,
        "policy": policy,
        "model_setup": {"seed": R1_SEED, "rounds": k, "variant": "shared"},
        "schedule": "update -> window=update%2, pair=update//2",
        "p0_module_loaded": bool(p0),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if native:
        result["dll_metadata"] = _dll_metadata(bridge)
    return result


def run_stable(*, route: str, updates: int, warmup_updates: int, manifest_path: Path, cache_file: Path, profile: bool = False) -> dict[str, Any]:
    if updates <= 0 or warmup_updates < 0:
        raise ValueError("updates must be positive and warmup_updates non-negative")
    p0, ce, bridge = _load_r1_modules()
    native, k = route_parts(route)
    if profile and (not native or k != 1):
        raise ValueError("profile phase is restricted to native-k1")
    documents, payload, teacher_weight, teacher_bias = _load_inputs(p0, manifest_path, cache_file)
    model = ce.fresh_model(R1_SEED, k)
    state_part_weight = prepare_native_model(model) if native else None
    dll_metadata = _dll_metadata(bridge) if native else None
    optimizer = __import__("torch").optim.AdamW(
        model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY,
    )
    physical_batch = int(p0.PHYSICAL_BATCH)
    state = model.initial_state(physical_batch, device=__import__("torch").device("cpu"))
    bridge.reset_runtime_stats(clear_pool=True)
    if profile:
        bridge.reset_profile()
        bridge.set_profile_enabled(True)
        if not bool(bridge.profile_stats()["compiled"]):
            raise RuntimeError("profile phase requires a rebuild with -DOMEGA_PROFILE_INTERNAL=ON")
    frozen, _, _ = p0.hidden.base.load_frozen_train_documents()
    def run_items(items: list[dict[str, int]], records: list[dict[str, Any]]) -> None:
        nonlocal state
        for item in items:
            positions = [int(index) for index in frozen["cyclic_pairs"]["pairs"][item["pair"]]["document_indices"]]
            source = __import__("torch").tensor([documents[position]["tokens"] for position in positions], dtype=__import__("torch").long)
            offset = item["window"] * WINDOW_TOKENS
            inputs = source[:, offset : offset + WINDOW_TOKENS]
            targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
            if item["window"] == 0:
                state = model.initial_state(physical_batch, device=__import__("torch").device("cpu"))
            teacher = _teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, item["window"])
            measurement, state = _run_update(
                model=model, optimizer=optimizer, inputs=inputs, targets=targets,
                previous_state=state, teacher_logits=teacher, native=native,
                p0=p0, ce=ce, bridge=bridge, state_part_weight=state_part_weight,
            )
            records.append({"update": item["update"], **measurement})

    warmup_records: list[dict[str, Any]] = []
    run_items(schedule(warmup_updates) if warmup_updates else [], warmup_records)
    # Warmup changes model/optimizer state but is excluded from measured totals.
    bridge.reset_runtime_stats(clear_pool=False)
    if profile:
        bridge.reset_profile()
    state = model.initial_state(physical_batch, device=__import__("torch").device("cpu"))
    measured_records: list[dict[str, Any]] = []
    run_items(schedule(updates), measured_records)
    result: dict[str, Any] = {
        "phase": "profile" if profile else "stable",
        "route": route,
        "K": k,
        "native": native,
        "warmup": {"updates": warmup_updates, "measured": False},
        "measured": {"updates": updates, "measured": True},
        "updates": measured_records,
    }
    if dll_metadata is not None:
        result["dll_metadata"] = dll_metadata
    if profile:
        result["profile"] = _profile_report(bridge.profile_stats())
        bridge.set_profile_enabled(False)
    return result


def run_profile(*, manifest_path: Path, cache_file: Path) -> dict[str, Any]:
    """Run only the explicitly authorized bounded native K1 profile route."""
    return run_stable(
        route="native-k1",
        updates=2,
        warmup_updates=1,
        manifest_path=manifest_path,
        cache_file=cache_file,
        profile=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("canary", "preflight", "stable", "profile"), default="canary")
    parser.add_argument("--route", choices=ROUTES, action="append", default=None)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--warmup-updates", type=int, default=DEFAULT_WARMUP_UPDATES)
    parser.add_argument("--confirm-real-execution", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.phase == "canary":
        report: dict[str, Any] = {
            "campaign_id": "OMEGA-NATIVE-RUNTIME-P2-R2",
            "phase": "canary",
            "routes": list(args.route or ROUTES),
            "real_corpus_or_model_benchmark": False,
            "policy": policy_metadata(),
            "canaries": run_canaries(),
            "aggregation": {
                "status": "NOT_RUN",
                "classification": None,
                "r_k": {},
                "r_joint": None,
            },
        }
    else:
        require_real_authorization(args.confirm_real_execution)
        routes = args.route or list(ROUTES)
        if args.phase == "preflight":
            route_reports = {route: run_preflight(route=route) for route in routes}
            report = {
                "campaign_id": "OMEGA-NATIVE-RUNTIME-P2-R2",
                "phase": args.phase,
                "policy": policy_metadata(),
                "routes": route_reports,
                "aggregation": aggregate_performance(route_reports),
            }
        elif args.phase == "stable":
            if args.manifest is None or args.cache_file is None:
                raise ValueError("stable phase requires --manifest and --cache-file")
            route_reports = {
                route: run_stable(
                    route=route,
                    updates=args.updates,
                    warmup_updates=args.warmup_updates,
                    manifest_path=args.manifest,
                    cache_file=args.cache_file,
                )
                for route in routes
            }
            report = {
                "campaign_id": "OMEGA-NATIVE-RUNTIME-P2-R2",
                "phase": args.phase,
                "policy": policy_metadata(),
                "routes": route_reports,
                "aggregation": aggregate_performance(route_reports),
            }
        else:
            if args.manifest is None or args.cache_file is None:
                raise ValueError("profile phase requires --manifest and --cache-file")
            requested_routes = args.route or ["native-k1"]
            if requested_routes != ["native-k1"]:
                raise ValueError("profile phase accepts only --route native-k1")
            report = {
                "campaign_id": "OMEGA-NATIVE-RUNTIME-P2-R2",
                "phase": "profile",
                "real_corpus_or_model_benchmark": True,
                "authorization": "explicit --confirm-real-execution",
                "route": run_profile(manifest_path=args.manifest, cache_file=args.cache_file),
            }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.get("canaries", {}).get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
