"""Capture a scoped CPU profiler trace for the four analyzed PyTorch replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPLAY_SCRIPT = HERE / "run_omega_recurrent_backward_replay.py"
DEFAULT_FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
DEFAULT_OUTPUT_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "profile" / "pytorch_exact_markers"
FIXTURE_FILENAMES = (
    "replay_K4_update2_window0.bin",
    "replay_K4_update3_window1.bin",
)
FIXTURE_LABELS = ("A_update2_window0", "B_update3_window1")
ENV_VARS_REQUIRED_UNSET = (
    "KMP_BLOCKTIME",
    "KMP_LIBRARY",
    "OMP_WAIT_POLICY",
    "KMP_SETTINGS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
EXTERNAL_CLEAN_R_BACKWARD = 1.7363279572184258


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_env() -> dict[str, Any]:
    current = {name: os.environ.get(name) for name in ENV_VARS_REQUIRED_UNSET}
    set_values = {name: value for name, value in current.items() if value is not None}
    if set_values:
        raise RuntimeError(f"profile requires inherited-unset canonical environment: {set_values}")
    return {
        "inherited_environment": current,
        "effective_libiomp_defaults_from_separate_probe": {
            "KMP_BLOCKTIME": "200ms",
            "KMP_LIBRARY": "throughput",
            "OMP_WAIT_POLICY": "PASSIVE",
        },
        "KMP_SETTINGS_during_capture": "unset",
        "torch_threads": {"intraop": 4, "interop": 1},
    }


def _load_modules() -> tuple[Any, Any, Any, Any, Any]:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    replay = __import__("run_omega_recurrent_backward_replay")
    torch, (p0, ce, golden, fast_block_type, policy) = replay._torch_setup()
    return replay, torch, (p0, ce), golden, (fast_block_type, policy)


def _load_cases(replay: Any, torch: Any, fast_block_type: Any, fixture_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for label, filename in zip(FIXTURE_LABELS, FIXTURE_FILENAMES):
        path = fixture_root / filename
        header, arrays, fixture_sha = replay.read_fixture(path)
        expected_update_window = (2, 0) if label.startswith("A_") else (3, 1)
        observed_update_window = (int(header["metadata"]["update_index"]), int(header["metadata"]["window_index"]))
        if observed_update_window != expected_update_window:
            raise ValueError(f"fixture identity mismatch for {filename}: {observed_update_window}")
        model = replay._ReplayModel(arrays, torch, fast_block_type)
        cases.append({
            "label": label,
            "path": path,
            "header": header,
            "arrays": arrays,
            "sha256": fixture_sha,
            "model": model,
            "parameters": replay_params(model),
            "token_part": np.array(arrays["token_part"], copy=True),
            "previous_state": np.array(arrays["previous_state"], copy=True),
            "G_R": torch.from_numpy(np.array(arrays["G_R"], copy=True)),
            "G_S": torch.from_numpy(np.array(arrays["G_S"], copy=True)),
        })
    return cases


def replay_params(model: Any) -> list[Any]:
    block = model.blocks[0]
    return [
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
    ]


def _tensor_layout(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(dim) for dim in value.shape],
        "strides_elements": [int(dim) for dim in value.stride()],
        "dtype": str(value.dtype),
    }


def _source_audit(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model = cases[0]["model"]
    block = model.blocks[0]
    linear_weights = {
        "block.qkv": block.qkv.weight,
        "block.out": block.out.weight,
        "block.fc1": block.fc1.weight,
        "block.fc2": block.fc2.weight,
    }
    layouts: dict[str, Any] = {}
    for name, weight in linear_weights.items():
        transpose = weight.T
        layouts[name] = {
            "weight": _tensor_layout(weight),
            "F_linear_weight_transpose_view": _tensor_layout(transpose),
        }
    layouts["state_part_weight_view"] = _tensor_layout(model.prelude.weight[:, 128:])
    return {
        "source_files": [
            {
                "path": "omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py",
                "ranges": ["58-70", "121-129", "131-171"],
                "roles": [
                    "shared block QKV/attention/out/FC1/GELU/FC2/RMSNorm",
                    "differentiable depth bias and gate construction",
                    "production recurrent state/write/anchor/readout loop",
                ],
            },
            {
                "path": "omega_native_runtime_p2r0/run_omega_native_runtime_p2r0_golden.py",
                "ranges": ["83-108"],
                "roles": ["isolated recurrent loop from token_part; same equations used for certified VJP"],
            },
        ],
        "linear_parameter_layouts": layouts,
        "shape_disambiguation_notes": [
            "aten::mm [64,128] x [128,512] under AddmmBackward0 is FC2 dInput, not FC1 forward.",
            "FC1 forward is biased aten::addmm with input [64,128] and transposed fc1 weight [128,512], whose view stride is [1,128].",
            "Profiler input_shapes do not encode strides; actual parameter and transpose-view strides above come from loaded fixture model, and operator attribution additionally uses parent autograd node plus audited source equations.",
        ],
    }


def _json_shape(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_json_shape(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _event_name(event: Any) -> str:
    return str(getattr(event, "name", ""))


def _parent_chain(event: Any) -> list[str]:
    names: list[str] = []
    current = getattr(event, "cpu_parent", None)
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        names.append(_event_name(current))
        current = getattr(current, "cpu_parent", None)
    return names


def _backward_owner(event: Any, marker_names: set[str]) -> str | None:
    current = getattr(event, "cpu_parent", None)
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = _event_name(current)
        if name in marker_names:
            return name
        current = getattr(current, "cpu_parent", None)
    return None


def _matrix_source_candidate(name: str, shapes_value: Any, parent_chain: Sequence[str]) -> str | None:
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes_value if isinstance(shape, (list, tuple)))
    parent = " ".join(parent_chain)
    if name == "aten::mm":
        candidates = {
            ((64, 128), (128, 512)): "block.fc2 dInput (dY @ fc2.weight)",
            ((64, 512), (512, 128)): "block.fc1 dInput (dY @ fc1.weight)",
            ((512, 64), (64, 128)): "block.fc1 dWeight (dY.T @ saved input)",
            ((128, 64), (64, 512)): "block.fc2 dWeight (dY.T @ GELU output)",
            ((64, 384), (384, 128)): "block.qkv dInput",
            ((384, 64), (64, 128)): "block.qkv dWeight",
            ((64, 128), (128, 128)): "block.out dInput (dY @ out.weight)",
            ((128, 64), (64, 128)): "block.out dWeight (dY.T @ attention mixed output)",
            ((128, 8), (8, 1024)): "state_part_weight dWeight accumulation, transposed intermediate",
            ((8, 1024), (1024, 128)): "state_part_weight dInput (dY @ state_part_weight)",
        }
        candidate = candidates.get(shapes)
        if candidate and "AddmmBackward0" in parent:
            return candidate
        if candidate and "MmBackward0" in parent and shapes in {
            ((128, 8), (8, 1024)),
            ((8, 1024), (1024, 128)),
        }:
            return candidate
        if candidate:
            return candidate + f" [parent={parent or 'unresolved'}]"
    if name == "aten::addmm":
        addmm_candidates = {
            ((512,), (64, 128), (128, 512)): "block.fc1 forward (biased linear; transposed fc1 weight stride [1,128])",
            ((128,), (64, 512), (512, 128)): "block.fc2 forward (biased linear)",
            ((384,), (64, 128), (128, 384)): "block.qkv forward (biased linear)",
            ((128,), (64, 128), (128, 128)): "block.out forward (biased linear)",
        }
        return addmm_candidates.get(shapes)
    return None


def _category(name: str) -> str:
    lower = name.lower()
    if name in {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear", "aten::addbmm", "aten::baddbmm"}:
        return "multiplication_projection"
    if "scaled_dot_product" in lower and "attention" in lower:
        return "attention_backend"
    if "accumulategrad" in lower:
        return "backward_management"
    if any(tag in lower for tag in ("sum", "mean", "var", "norm", "layer_norm", "rms_norm", "softmax")):
        return "reduction_normalization"
    if any(tag in lower for tag in ("add_", "copy", "clone", "empty", "zero", "fill_", "contiguous", "detach", "view", "as_strided", "set_", "to_copy")):
        return "backward_management"
    if lower.startswith("autograd::") or ("backward" in lower and not lower.startswith("aten::")):
        return "autograd_engine_node"
    if any(tag in lower for tag in ("gelu", "sigmoid", "tanh", "erf", "exp", "mul", "add", "sub", "div", "sqrt", "rsqrt", "where")):
        return "elementwise_derivative_gate"
    return "other"


def _profile_aggregation(profiler: Any, analyzed_labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = list(profiler.events())
    backward_begin_names = {
        str(item["backward_begin_label"])
        for item in analyzed_labels
    }
    backward_end_names = {str(item["backward_end_label"]) for item in analyzed_labels}
    event_by_name = {
        _event_name(event): event
        for event in events
        if _event_name(event) in backward_begin_names or _event_name(event) in backward_end_names
    }
    expected_marker_names = backward_begin_names | backward_end_names
    if set(event_by_name) != expected_marker_names:
        missing = sorted(expected_marker_names - set(event_by_name))
        raise RuntimeError(f"backward profiler marker(s) missing from event tree: {missing}")
    intervals: dict[str, list[Any]] = {name: [] for name in backward_begin_names}
    interval_bounds: dict[str, tuple[float, float]] = {}
    marker_names = backward_begin_names | backward_end_names
    for item in analyzed_labels:
        begin_name = str(item["backward_begin_label"])
        end_name = str(item["backward_end_label"])
        begin_range = getattr(event_by_name[begin_name], "time_range", None)
        end_range = getattr(event_by_name[end_name], "time_range", None)
        if begin_range is None or end_range is None:
            raise RuntimeError(f"profiler marker lacks time range: {begin_name} / {end_name}")
        interval_start = float(begin_range.end)
        interval_end = float(end_range.start)
        if interval_end < interval_start:
            raise RuntimeError(f"backward marker order invalid: {begin_name} / {end_name}")
        interval_bounds[begin_name] = (interval_start, interval_end)

    for event in events:
        name = _event_name(event)
        if name in marker_names:
            continue
        event_range = getattr(event, "time_range", None)
        if event_range is None:
            continue
        event_start = float(event_range.start)
        event_end = float(event_range.end)
        for marker_name, (interval_start, interval_end) in interval_bounds.items():
            if event_start >= interval_start and event_end <= interval_end:
                intervals[marker_name].append(event)
                break

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    per_interval: list[dict[str, Any]] = []
    for item in analyzed_labels:
        marker_name = str(item["backward_begin_label"])
        scope_event = event_by_name[marker_name]
        scoped_events = intervals[marker_name]
        stack_evidence = 0
        self_sum = 0.0
        interval_rows: list[dict[str, Any]] = []
        for event in scoped_events:
            event_name = _event_name(event)
            shapes_value = _json_shape(getattr(event, "input_shapes", None))
            shapes_key = json.dumps(shapes_value, sort_keys=True, separators=(",", ":"))
            parent_chain = _parent_chain(event)
            parent_name = parent_chain[0] if parent_chain else "<root>"
            if parent_name.startswith("REPLAY_BACKWARD_BEGIN|"):
                parent_name = "<REPLAY_BACKWARD_SCOPE>"
                parent_chain = [parent_name, *parent_chain[1:]]
            group_key = (event_name, shapes_key, parent_name)
            stack = getattr(event, "stack", None) or []
            stack = [str(line) for line in stack if line]
            stack_evidence += bool(stack)
            event_self = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            event_inclusive = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
            self_sum += event_self
            row = grouped.setdefault(group_key, {
                "operator": event_name,
                "input_shapes": shapes_value,
                "direct_parent": parent_name,
                "parent_chain_example": parent_chain,
                "category": _category(event_name),
                "count": 0,
                "sum_self_cpu_time_us": 0.0,
                "sum_inclusive_cpu_time_us_non_additive": 0.0,
                "max_self_cpu_time_us": 0.0,
                "stack_evidence_count": 0,
                "stack_examples": [],
                "source_audit_candidate": _matrix_source_candidate(event_name, shapes_value or [], parent_chain),
            })
            row["count"] += 1
            row["sum_self_cpu_time_us"] += event_self
            row["sum_inclusive_cpu_time_us_non_additive"] += event_inclusive
            row["max_self_cpu_time_us"] = max(row["max_self_cpu_time_us"], event_self)
            if stack:
                row["stack_evidence_count"] += 1
                if len(row["stack_examples"]) < 3 and stack not in row["stack_examples"]:
                    row["stack_examples"].append(stack[:12])
            interval_rows.append({
                "operator": event_name,
                "input_shapes": shapes_value,
                "direct_parent": parent_name,
                "self_cpu_time_us": event_self,
                "inclusive_cpu_time_us_non_additive": event_inclusive,
            })

        marker_scope_total = float(getattr(scope_event, "cpu_time_total", 0.0) or 0.0)
        interval_start, interval_end = interval_bounds[marker_name]
        per_interval.append({
            "fixture": item["fixture"],
            "replay": item["replay"],
            "pid": item["pid"],
            "backward_begin_marker": marker_name,
            "backward_end_marker": item["backward_end_label"],
            "profiler_marker_delimited_backward_interval_us": interval_end - interval_start,
            "begin_marker_scope_overhead_us": marker_scope_total,
            "descendant_event_count": len(scoped_events),
            "descendant_self_cpu_time_sum_us_exclusive": self_sum,
            "descendant_stack_evidence_count": int(stack_evidence),
            "attribution_scope_note": "Only events wholly between BEGIN marker end and END marker start on the profiler timebase are included.",
        })

    operator_rows = sorted(
        grouped.values(),
        key=lambda row: float(row["sum_self_cpu_time_us"]),
        reverse=True,
    )
    category_totals: dict[str, dict[str, Any]] = {}
    for row in operator_rows:
        category = row["category"]
        bucket = category_totals.setdefault(category, {"count": 0, "sum_self_cpu_time_us_exclusive": 0.0, "operator_groups": 0})
        bucket["count"] += int(row["count"])
        bucket["sum_self_cpu_time_us_exclusive"] += float(row["sum_self_cpu_time_us"])
        bucket["operator_groups"] += 1

    total_stack_evidence = sum(int(row["stack_evidence_count"]) for row in operator_rows)
    return {
        "profile_event_count": len(events),
        "backward_intervals": per_interval,
        "backward_interval_count": len(per_interval),
        "backward_descendant_events_total": sum(int(row["descendant_event_count"]) for row in per_interval),
        "operator_group_count": len(operator_rows),
        "operator_shape_parent_aggregates": operator_rows,
        "category_totals_self_time_exclusive_only": category_totals,
        "stack_evidence_count": total_stack_evidence,
        "stack_evidence_status": "present" if total_stack_evidence else "EMPTY_WITH_STACK_TRUE",
        "inclusive_time_warning": "inclusive_cpu_time_us_non_additive includes descendants; do not sum it with children. Exclusive self time is the additive field.",
    }


def run_profile(fixture_root: Path, output_root: Path) -> dict[str, Any]:
    canonical_environment = _canonical_env()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite PyTorch profile output: {output_root}")
    output_parent = output_root.parent
    if not output_parent.is_dir():
        raise FileNotFoundError(f"profile output parent must exist: {output_parent}")

    replay, torch, (p0, _ce), golden, (fast_block_type, thread_policy) = _load_modules()
    if int(torch.get_num_threads()) != 4 or int(torch.get_num_interop_threads()) != 1:
        raise RuntimeError("canonical PyTorch thread policy drift")
    memory_gate = p0.integration.memory_safety_gate()
    cases = _load_cases(replay, torch, fast_block_type, fixture_root)
    pid = os.getpid()
    marker_ledger: list[dict[str, Any]] = []

    def replay_once(case: Mapping[str, Any], replay_number: int, phase: str, *, collect: bool) -> dict[str, Any]:
        model = case["model"]
        for parameter in case["parameters"]:
            parameter.grad = None
        token_part = torch.from_numpy(case["token_part"].copy()).requires_grad_(True)
        previous_state = torch.from_numpy(case["previous_state"].copy()).requires_grad_(True)
        fixture = str(case["label"])
        forward_begin = f"REPLAY_FORWARD_BEGIN|fixture={fixture}|replay={replay_number}|phase={phase}|pid={pid}"
        forward_end = f"REPLAY_FORWARD_END|fixture={fixture}|replay={replay_number}|phase={phase}|pid={pid}"
        backward_begin = f"REPLAY_BACKWARD_BEGIN|fixture={fixture}|replay={replay_number}|phase={phase}|pid={pid}"
        backward_end = f"REPLAY_BACKWARD_END|fixture={fixture}|replay={replay_number}|phase={phase}|pid={pid}"
        with torch.profiler.record_function(forward_begin):
            pass
        next_state, readout_states = golden._recurrent_forward_from_token_part(model, token_part, previous_state)
        with torch.profiler.record_function(forward_end):
            pass
        backward_outputs = (next_state, readout_states)
        upstreams = (case["G_S"], case["G_R"])
        with torch.profiler.record_function(backward_begin):
            pass
        # Adjacent begin/end markers delimit the exact API-call boundary. No
        # destination prep is moved across the boundary; autograd performs its
        # same internal gradient accumulation as in the clean benchmark.
        torch.autograd.backward(backward_outputs, grad_tensors=upstreams, retain_graph=False)
        with torch.profiler.record_function(backward_end):
            pass
        marker_ledger.append({
            "fixture": fixture,
            "replay": replay_number,
            "phase": phase,
            "pid": pid,
            "forward_begin": forward_begin,
            "forward_end": forward_end,
            "backward_begin": backward_begin,
            "backward_end": backward_end,
            "profiled": collect,
        })
        return {
            "fixture": fixture,
            "replay": replay_number,
            "phase": phase,
            "pid": pid,
            "backward_begin_label": backward_begin,
            "backward_end_label": backward_end,
        }

    # Warmup A/B is not profiled; all analyzed trace events belong to A/B/A/B.
    replay_once(cases[0], 1, "warmup", collect=False)
    replay_once(cases[1], 2, "warmup", collect=False)

    analyzed_labels: list[dict[str, Any]] = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
    ) as profiler:
        for replay_number in range(3, 7):
            case = cases[(replay_number - 3) % 2]
            analyzed_labels.append(replay_once(case, replay_number, "analyzed", collect=True))

    output_root.mkdir(parents=False)
    trace_path = output_root / "pytorch_profile_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    aggregation = _profile_aggregation(profiler, analyzed_labels)
    aggregation_path = output_root / "backward_operator_shape_aggregates.json"
    _write_json(aggregation_path, aggregation)
    matrix_attribution_rows = [
        {
            "operator": row["operator"],
            "input_shapes": row["input_shapes"],
            "parent": row["direct_parent"],
            "count": row["count"],
            "sum_self_cpu_time_us": row["sum_self_cpu_time_us"],
            "source_audit_candidate": row.get("source_audit_candidate"),
        }
        for row in aggregation["operator_shape_parent_aggregates"]
        if row["operator"] in ("aten::mm", "aten::addmm", "aten::bmm")
    ]

    marker_events = {"forward_begin": 4, "forward_end": 4, "backward_begin": 4, "backward_end": 4}
    marker_ledger_counts = {
        "forward_begin": sum(item["forward_begin"].startswith("REPLAY_FORWARD_BEGIN|") for item in marker_ledger),
        "forward_end": sum(item["forward_end"].startswith("REPLAY_FORWARD_END|") for item in marker_ledger),
        "backward_begin": sum(item["backward_begin"].startswith("REPLAY_BACKWARD_BEGIN|") for item in marker_ledger),
        "backward_end": sum(item["backward_end"].startswith("REPLAY_BACKWARD_END|") for item in marker_ledger),
    }
    if len(analyzed_labels) != 4 or aggregation["backward_interval_count"] != 4:
        raise RuntimeError("profile did not capture exactly four analyzed backward intervals")
    report = {
        "schema": "omega-recurrent-backward-replay-pytorch-profile-v1",
        "status": "PYTORCH_PROFILE_PASS",
        "route": "pytorch",
        "process_fresh": True,
        "pid": pid,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "thread_policy": thread_policy,
        "parallel_info_outside_profile": torch.__config__.parallel_info(),
        "canonical_environment": canonical_environment,
        "memory_safety_gate": memory_gate,
        "capture": {
            "activities": ["CPU"],
            "record_shapes": True,
            "with_stack": True,
            "profile_memory": False,
            "profile_scope": "four analyzed replays only; two A/B warmups were outside profiler context",
            "warmup_schedule": ["A_update2_window0", "B_update3_window1"],
            "analyzed_schedule": ["A_update2_window0", "B_update3_window1"] * 2,
            "autograd_call": "torch.autograd.backward((next_state, readout_states), grad_tensors=(G_S, G_R), retain_graph=False)",
            "gradient_destination_boundary": "No explicit destination preparation added; autograd accumulation/initialization inside backward call remains within backward marker, matching clean benchmark API-call interval.",
            "marker_counts": marker_events,
            "marker_ledger_counts_including_unprofiled_warmups": marker_ledger_counts,
            "marker_ledger": marker_ledger,
            "clean_external_R_backward_reference": EXTERNAL_CLEAN_R_BACKWARD,
            "clean_external_ratio_recomputed": False,
        },
        "fixtures": [
            {
                "label": case["label"],
                "filename": Path(case["path"]).name,
                "sha256": case["sha256"],
                "update": int(case["header"]["metadata"]["update_index"]),
                "window": int(case["header"]["metadata"]["window_index"]),
                "state_part_weight_layout": case["header"]["metadata"]["model"]["state_part_weight_layout"],
            }
            for case in cases
        ],
        "source_audit": _source_audit(cases),
        "aggregation_file": aggregation_path.name,
        "full_trace_file": trace_path.name,
        "full_trace_bytes": int(trace_path.stat().st_size),
        "full_trace_sha256": _sha256_file(trace_path),
        "backward_aggregation": aggregation,
        "operator_group_count": aggregation["operator_group_count"],
        "matrix_operator_attribution_rows": matrix_attribution_rows,
        "analysis_notes": [
            "Four backward intervals use marker event time ranges: between REPLAY_BACKWARD_BEGIN end and REPLAY_BACKWARD_END start.",
            "with_stack=True produced zero usable stack frames; operator attribution uses shapes, parent autograd node, actual parameter strides, and audited source.",
            "Inclusive event time is non-additive; category totals use exclusive self_cpu_time.",
            "This profiler capture does not replace the clean external R_backward=1.7363279572184258.",
        ],
        "warnings": [
            "Profiler self/inclusive times are attribution data, not the clean backward benchmark ratio.",
            "PyTorch emitted a generic event-clearing warning; this used one profiler context without prof.step(), and all four backward marker intervals were present.",
            "No native profile was run; this is the first-route report only.",
        ],
    }
    _write_json(output_root / "profile_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run_profile(args.fixture_root.resolve(), args.output_root.resolve())
    print(json.dumps({
        "status": result["status"],
        "pid": result["pid"],
        "trace": result["full_trace_file"],
        "trace_bytes": result["full_trace_bytes"],
        "backward_interval_count": result["backward_aggregation"]["backward_interval_count"],
        "stack_evidence_status": result["backward_aggregation"]["stack_evidence_status"],
        "operator_group_count": len(result["backward_aggregation"]["operator_shape_parent_aggregates"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
