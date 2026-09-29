"""Fresh-process 8+32 stable recaracterization with canonical R1 loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_backend_quality_qualification as quality  # noqa: E402
import run_loss_contract_causal_smoke as smoke  # noqa: E402


SEED = 20260913
WARMUP_UPDATES = 8
MEASURED_UPDATES = 32
WINDOW_TOKENS = 256
ROUTE_ORDER = (("pytorch", 1), ("native", 1), ("pytorch", 4), ("native", 4))
SMOKE_REPORT = HERE / "results" / "loss_contract_causal_smoke_20260924T115525" / "causal_smoke_report.json"
COMMON_ORIGIN_REPORT = HERE / "results" / "corrected_loss_window1_common_origin_20260924T121844" / "window1_common_origin_report.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(quality._canonical_bytes(event))
        stream.flush()
        os.fsync(stream.fileno())


def _loss_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    function = quality.r1_masked_token_mean_loss
    path = Path(function.__code__.co_filename).resolve()
    digest = _sha256_file(path)
    pinned = manifest["loss_contract"]["implementation"]
    if (
        path != Path(pinned["path"]).resolve()
        or function.__name__ != pinned["callable"]
        or digest != pinned["sha256"]
        or quality.LOSS_CONTRACT != "R1_MASKED_TOKEN_MEAN_V1"
    ):
        raise ValueError("actually loaded loss callable/path/hash differs from the pinned R1 contract")
    return {"callable": function.__name__, "path": str(path), "sha256": digest, "contract": quality.LOSS_CONTRACT}


def _policy(ce: Any, p0: Any) -> dict[str, Any]:
    result = ce.validate_policy()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("PyTorch threads must be intra-op 4 / inter-op 1")
    if result["intraop_threads"] != 4 or result["interop_threads"] != 1 or int(p0.PHYSICAL_BATCH) != 8:
        raise RuntimeError("runtime common policy differs from stable 8+32 contract")
    return result


def _verify_prerequisites(manifest: dict[str, Any]) -> dict[str, Any]:
    if not SMOKE_REPORT.is_file() or not COMMON_ORIGIN_REPORT.is_file():
        raise FileNotFoundError("required causal smoke/common-origin reports are missing")
    smoke_report = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    origin_report = json.loads(COMMON_ORIGIN_REPORT.read_text(encoding="utf-8"))
    if smoke_report.get("status") != "FAIL" or smoke_report.get("technical_optimizer_updates") != 8:
        raise PermissionError("stable step requires preserved original 8-update smoke report, including its FAIL")
    if (
        origin_report.get("status") != "LOCAL_PASS"
        or origin_report.get("classification") != "TRAJECTORY_PROXIMITY_DISCREPANCY_IN_THIS_CASE"
        or not origin_report.get("part2", {}).get("common_origin_copy_exact")
        or not origin_report.get("part2", {}).get("gates", {}).get("K4_common_origin_window1", {}).get("pass")
    ):
        raise PermissionError("corrected-loss common-origin diagnosis did not pass all local K4/window1 gates")
    if smoke_report.get("manifest_sha256") != manifest["manifest_sha256"] or origin_report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("prerequisite evidence uses a different sealed manifest")
    return {
        "original_smoke": {"path": str(SMOKE_REPORT), "sha256": _sha256_file(SMOKE_REPORT), "status": smoke_report["status"]},
        "common_origin": {"path": str(COMMON_ORIGIN_REPORT), "sha256": _sha256_file(COMMON_ORIGIN_REPORT), "status": origin_report["status"], "classification": origin_report["classification"]},
    }


def _route_child(
    *,
    backend: str,
    rounds: int,
    init_path: Path,
    init_sha256: str,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = quality._load_qualification_manifest()
    loss_identity = _loss_identity(manifest)
    r2, p0, ce, bridge, _modules = quality._load_real_dependencies()
    policy = _policy(ce, p0)
    if _sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("selected be376 DLL changed before stable recaracterization")

    bridge_path = Path(bridge.__file__).resolve()
    pinned_bridge = manifest["source_identities"]["r2_production_bridge"]
    if bridge_path != Path(pinned_bridge["path"]).resolve() or _sha256_file(bridge_path) != pinned_bridge["sha256"]:
        raise ValueError("loaded production bridge identity differs from sealed manifest")
    dll_identity = {
        "path": str(quality.BE376_DLL),
        "sha256": _sha256_file(quality.BE376_DLL),
        "expected_sha256": quality.BE376_SHA256,
    }
    if backend == "native":
        bridge.configure_library(quality.BE376_DLL)
        if not bridge.runtime_abi_available():
            raise RuntimeError("selected be376 DLL has no production runtime ABI")
        bridge.configure_runtime(4)
        bridge.reset_runtime_stats(clear_pool=True)

    documents, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    train_manifest = json.loads(Path(manifest["training_manifest"]["path"]).read_text(encoding="utf-8"))
    expected_doc_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in train_manifest["documents"]]
    actual_doc_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in documents]
    if len(documents) != 602 or actual_doc_keys != expected_doc_keys:
        raise ValueError("stable route training docs/order differ from the sealed 602-doc manifest")

    init_path = init_path.resolve()
    if _sha256_file(init_path) != init_sha256:
        raise ValueError("stable common-init bundle file hash mismatch")
    bundle = torch.load(init_path, map_location="cpu", weights_only=False)
    if int(bundle["seed"]) != SEED or int(bundle["K"]) != rounds:
        raise ValueError("stable common-init bundle seed/K mismatch")
    model = ce.fresh_model(SEED, rounds)
    model.load_state_dict(bundle["model_state"], strict=True)
    if quality._value_hash(model.state_dict()) != bundle["model_state_sha256"]:
        raise ValueError("stable route model differs from common initialization snapshot")
    torch.set_rng_state(bundle["torch_rng_state"])
    random.setstate(bundle["python_rng_state"])
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY,
    )
    if optimizer.state or quality._value_hash(optimizer.state_dict()) != bundle["fresh_adamw_state_sha256"]:
        raise AssertionError("each stable route must begin with a fresh empty AdamW state")
    physical_batch = int(p0.PHYSICAL_BATCH)
    state = model.initial_state(physical_batch, device=torch.device("cpu"))
    if quality._tensor_hash(state) != bundle["initial_recurrent_state_sha256"]:
        raise ValueError("stable route initial recurrent state differs from common bundle")
    state_part_weight = r2.prepare_native_model(model) if backend == "native" else None

    route_id = f"{backend}_K{rounds}_seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = run_dir / "updates.jsonl"
    frozen_pairs = train_manifest["pairs"]

    def run_schedule(count: int, phase: str, rows: list[dict[str, Any]]) -> None:
        nonlocal state
        for item in r2.schedule(count):
            window = int(item["window"])
            pair_index = int(item["pair"])
            if window == 0:
                state = model.initial_state(physical_batch, device=torch.device("cpu"))
            pair = frozen_pairs[pair_index]
            positions = [int(index) for index in pair["document_indices"]]
            source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
            offset = window * WINDOW_TOKENS
            inputs = source[:, offset : offset + WINDOW_TOKENS]
            targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, window)
            if tuple(inputs.shape) != (8, 256) or tuple(targets.shape) != (8, 256) or int(valid_mask.sum().item()) != 2048:
                raise ValueError("stable input/target/mask shape or valid-token count changed")

            bridge_before = bridge.runtime_stats()
            rss_before = r2._rss_bytes()
            optimizer.zero_grad(set_to_none=True)
            update_started = time.perf_counter()
            previous_state = state.detach() if window == 1 else state
            if backend == "native":
                next_state, student_logits, _readouts = r2._native_forward(
                    model, inputs, previous_state, state_part_weight, bridge
                )
            else:
                next_state, student_logits, _readouts = r2._reference_forward(
                    model, inputs, previous_state, p0
                )
            loss_terms = quality.r1_masked_token_mean_loss(student_logits, teacher_logits, targets, valid_mask)
            loss_terms["total"].backward()
            pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
            clip_coefficient = min(1.0, float(ce.CLIP_NORM) / (pre_clip_norm + 1.0e-6))
            optimizer.step()
            total_update_seconds = time.perf_counter() - update_started
            bridge_after = bridge.runtime_stats()
            rss_after = r2._rss_bytes()
            bridge_delta = {
                name: bridge_after[name] - bridge_before[name]
                for name in (
                    "workspace_allocations", "workspace_reuses",
                    "native_forward_c_abi_seconds", "native_backward_c_abi_seconds",
                    "native_forward_bridge_boundary_seconds", "native_backward_bridge_boundary_seconds",
                )
            }
            optimizer_step = max(
                int(slot["step"].item()) if torch.is_tensor(slot.get("step")) else int(slot.get("step", 0))
                for slot in optimizer.state.values()
            )
            state = next_state.detach()
            record = {
                "route_id": route_id,
                "backend": backend,
                "K": rounds,
                "seed": SEED,
                "phase": phase,
                "phase_update": int(item["update"]),
                "pair": pair_index,
                "window": window,
                "documents_positions": positions,
                "valid_tokens": int(loss_terms["valid_tokens"]),
                "loss_contract": quality.LOSS_CONTRACT,
                "loss_callable": loss_identity["callable"],
                "loss_source_path": loss_identity["path"],
                "loss_source_sha256": loss_identity["sha256"],
                "ce": float(loss_terms["ce"].detach().item()),
                "kl_tau_squared_applied": float(loss_terms["kl"].detach().item()),
                "total_loss": float(loss_terms["total"].detach().item()),
                "pre_clip_gradient_norm": pre_clip_norm,
                "clip_norm_limit": float(ce.CLIP_NORM),
                "clip_coefficient_applied": clip_coefficient,
                "clip_intervened": clip_coefficient < 1.0,
                "optimizer_step": optimizer_step,
                "optimizer_step_total": WARMUP_UPDATES + MEASURED_UPDATES,
                "total_update_seconds": total_update_seconds,
                "bridge_runtime_delta": bridge_delta,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "state_reset": window == 0,
                "state_source_update": None if window == 0 else optimizer_step - 2,
                "state_input_sha256": quality._tensor_hash(previous_state),
                "state_output_sha256": quality._tensor_hash(state),
                "teacher_logits_sha256": quality._tensor_hash(teacher_logits),
                "source_tokens_sha256": quality._tensor_hash(source),
                "physical_batch": 8,
                "effective_batch": 8,
                "gradient_accumulations": 1,
                "native_workers": 4 if backend == "native" else None,
                "torch_intraop_threads": 4,
                "torch_interop_threads": 1,
                "instrumentation": 0,
                "profile": False,
                "validation_or_test_loaded": False,
            }
            _append_jsonl(ledger_path, record)
            rows.append(record)

    warmup_rows: list[dict[str, Any]] = []
    run_schedule(WARMUP_UPDATES, "warmup_optimizer_updates", warmup_rows)
    # Match the existing R2 stable protocol: warmups mutate weights/AdamW, then
    # reset only recurrent state before measured schedule restarts at pair zero.
    state = model.initial_state(physical_batch, device=torch.device("cpu"))
    if backend == "native":
        bridge.reset_runtime_stats(clear_pool=False)
    measured_rows: list[dict[str, Any]] = []
    measured_start_utc_ns = time.time_ns()
    measured_start_perf_counter_ns = time.perf_counter_ns()
    run_schedule(MEASURED_UPDATES, "measured_updates", measured_rows)
    measured_end_perf_counter_ns = time.perf_counter_ns()
    measured_end_utc_ns = time.time_ns()

    report = {
        "schema": "corrected-loss-stable-route-report-v1",
        "status": "COMPLETE",
        "route_id": route_id,
        "backend": backend,
        "K": rounds,
        "seed": SEED,
        "phase": "STABLE_8_WARMUP_32_MEASURED",
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "optimizer_updates_total": WARMUP_UPDATES + MEASURED_UPDATES,
        "warmup_optimizer_steps_are_real": True,
        "route_process_id": os.getpid(),
        "common_initialization_path": str(init_path),
        "common_initialization_sha256": init_sha256,
        "common_initial_model_state_sha256": bundle["model_state_sha256"],
        "common_init_empty_adamw_sha256": bundle["fresh_adamw_state_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source": loss_identity,
        "historical_helper_called": False,
        "dll_identity": dll_identity if backend == "native" else None,
        "native_bridge_identity": {"path": str(bridge_path), "sha256": _sha256_file(bridge_path)},
        "hidden_cache_identity": {
            "manifest_path": manifest["hidden_cache"]["manifest"]["path"],
            "manifest_sha256": manifest["hidden_cache"]["manifest"]["sha256"],
            "cache_path": manifest["hidden_cache"]["cache_file"]["path"],
            "cache_sha256": manifest["hidden_cache"]["cache_file"]["sha256"],
            "teacher_revision": manifest["hidden_cache"]["teacher_revision"],
            "teacher_weight_sha256": quality._tensor_hash(teacher_weight),
            "teacher_bias_sha256": quality._tensor_hash(teacher_bias) if teacher_bias is not None else None,
        },
        "dataset_identity": {
            "train_manifest_sha256": manifest["training_manifest"]["manifest_sha256"],
            "source_train_manifest_sha256": manifest["training_manifest"]["source_sha256"],
            "dataset_revision": manifest["hidden_cache"]["dataset_revision"],
            "split": "train",
            "document_count": len(documents),
        },
        "runtime_policy": policy,
        "physical_batch": 8,
        "effective_batch": 8,
        "gradient_accumulations": 1,
        "native_workers": 4 if backend == "native" else None,
        "torch_intraop_threads": 4,
        "torch_interop_threads": 1,
        "instrumentation": 0,
        "diagnostic_statistics": "null/disabled",
        "profile": False,
        "measured_interval": {
            "start_utc_ns": measured_start_utc_ns,
            "start_perf_counter_ns": measured_start_perf_counter_ns,
            "end_perf_counter_ns": measured_end_perf_counter_ns,
            "end_utc_ns": measured_end_utc_ns,
            "wall_seconds": (measured_end_perf_counter_ns - measured_start_perf_counter_ns) / 1e9,
            "definition": "outer interval includes data/teacher materialization; primary total_update_seconds excludes it as R2 route timer does",
        },
        "warmup_rows": warmup_rows,
        "measured_rows": measured_rows,
        "primary_measured_update_seconds": sum(float(row["total_update_seconds"]) for row in measured_rows),
        "mean_primary_update_seconds": sum(float(row["total_update_seconds"]) for row in measured_rows) / MEASURED_UPDATES,
        "validation_or_test_loaded": False,
        "route_ledger_path": str(ledger_path),
        "route_ledger_sha256": _sha256_file(ledger_path),
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": _sha256_file(Path(__file__).resolve()),
    }
    report["report_self_sha256"] = quality._canonical_hash(report)
    _write_json(run_dir / "route_report.json", report)
    return report


def run_stable_8p32(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"stable corrected-loss output is immutable: {output_dir}")
    manifest = quality._load_qualification_manifest()
    preconditions = _verify_prerequisites(manifest)
    loss_identity = _loss_identity(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)

    r2, _p0, ce, _bridge, _modules = quality._load_real_dependencies()
    ce.validate_policy()
    common_init_rows: dict[int, tuple[Path, str, dict[str, Any]]] = {}
    for rounds in (1, 4):
        init_path, init_sha = quality._prepare_common_initialization(
            ce, SEED, rounds, output_dir / "common_initializations"
        )
        bundle = torch.load(init_path, map_location="cpu", weights_only=False)
        common_init_rows[rounds] = (init_path, init_sha, bundle)

    process_environment = os.environ.copy()
    for name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
        process_environment.pop(name, None)
    route_reports: list[dict[str, Any]] = []
    for backend, rounds in ROUTE_ORDER:
        init_path, init_sha, _bundle = common_init_rows[rounds]
        route_dir = output_dir / f"{backend}_K{rounds}_seed_{SEED}"
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()),
            "--_stable-child", "--backend", backend, "--K", str(rounds),
            "--init-path", str(init_path), "--init-sha256", init_sha,
            "--run-dir", str(route_dir), "--confirm-step3-preauthorized",
        ]
        completed = subprocess.run(command, cwd=quality.REPO_ROOT, env=process_environment, capture_output=True, text=True)
        (output_dir / f"{backend}_K{rounds}_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / f"{backend}_K{rounds}_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            failure = {
                "schema": "corrected-loss-stable-8p32-report-v1",
                "status": "INCOMPLETE",
                "reason": "stable route process failed; no automatic retry",
                "failed_route": {"backend": backend, "K": rounds, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
                "completed_route_reports": route_reports,
                "manifest_sha256": manifest["manifest_sha256"],
                "preconditions": preconditions,
                "updates_per_route": {"warmup": WARMUP_UPDATES, "measured": MEASURED_UPDATES},
                "output_dir": str(output_dir),
            }
            failure["report_self_sha256"] = quality._canonical_hash(failure)
            _write_json(output_dir / "stable_8p32_INCOMPLETE.json", failure)
            raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
        route_report_path = route_dir / "route_report.json"
        route_report = json.loads(route_report_path.read_text(encoding="utf-8"))
        if route_report.get("status") != "COMPLETE" or route_report.get("route_id") != f"{backend}_K{rounds}_seed_{SEED}":
            raise ValueError(f"stable child route report identity/status mismatch: {route_report_path}")
        route_reports.append({
            "backend": backend,
            "K": rounds,
            "seed": SEED,
            "route_report_path": str(route_report_path),
            "route_report_sha256": _sha256_file(route_report_path),
            "route_report": route_report,
        })

    per_route_mean = {(row["backend"], int(row["K"])): float(row["route_report"]["mean_primary_update_seconds"]) for row in route_reports}
    totals = {(row["backend"], int(row["K"])): float(row["route_report"]["primary_measured_update_seconds"]) for row in route_reports}
    ratios = {
        "K1_native_over_pytorch": per_route_mean[("native", 1)] / per_route_mean[("pytorch", 1)],
        "K4_native_over_pytorch": per_route_mean[("native", 4)] / per_route_mean[("pytorch", 4)],
        "joint_native_over_pytorch": (totals[("native", 1)] + totals[("native", 4)]) / (totals[("pytorch", 1)] + totals[("pytorch", 4)]),
    }
    results = {
        "schema": "corrected-loss-stable-8p32-report-v1",
        "status": "COMPLETE",
        "phase": "AUTHORIZED_STEP_3_CANONICAL_R1_LOSS_RECHARACTERIZATION",
        "preconditions": preconditions,
        "manifest_sha256": manifest["manifest_sha256"],
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source": loss_identity,
        "historical_helper_called": False,
        "dll": {"path": str(quality.BE376_DLL), "sha256": _sha256_file(quality.BE376_DLL), "expected_sha256": quality.BE376_SHA256},
        "seed": SEED,
        "route_order": [f"{backend}_K{rounds}" for backend, rounds in ROUTE_ORDER],
        "updates_per_route": {"warmup_optimizer_updates": WARMUP_UPDATES, "measured_updates": MEASURED_UPDATES},
        "route_processes_fresh": True,
        "route_reports": route_reports,
        "timing_ratios": ratios,
        "update_only_cost_estimate_seconds": {
            "per_route_2000_updates": {f"{backend}_K{rounds}": 2000.0 * per_route_mean[(backend, rounds)] for backend, rounds in ROUTE_ORDER},
            "two_seed_block_A_16000_updates": 4000.0 * sum(per_route_mean[(backend, rounds)] for backend, rounds in ROUTE_ORDER),
            "all_five_seed_matrix_40000_updates": 10000.0 * sum(per_route_mean[(backend, rounds)] for backend, rounds in ROUTE_ORDER),
            "unit": "CPU seconds; update-only estimate from corrected-loss measured means; validation/checkpoint/setup excluded",
        },
        "kernel_rewrite": False,
        "original_smoke_status_remains_fail_recorded": True,
        "strict_trajectory_proximity_reclassified": False,
        "output_dir": str(output_dir),
    }
    results["report_self_sha256"] = quality._canonical_hash(results)
    _write_json(output_dir / "stable_8p32_report.json", results)
    return results


def _stable_child_main(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_step3_preauthorized or args.backend not in ("pytorch", "native") or args.K not in (1, 4):
        raise PermissionError("stable child lacks preauthorized route identity")
    manifest = quality._load_qualification_manifest()
    return _route_child(
        backend=args.backend,
        rounds=args.K,
        init_path=args.init_path,
        init_sha256=args.init_sha256,
        run_dir=args.run_dir,
        output_dir=args.run_dir.parent.parent,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-step3-preauthorized", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--_stable-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("pytorch", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--K", type=int, choices=(1, 4), help=argparse.SUPPRESS)
    parser.add_argument("--init-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._stable_child:
        required = (args.backend, args.K, args.init_path, args.init_sha256, args.run_dir)
        if any(value is None for value in required):
            parser.error("incomplete stable child arguments")
        route = _stable_child_main(args)
        print(json.dumps({"route_id": route["route_id"], "status": route["status"], "mean_primary_update_seconds": route["mean_primary_update_seconds"]}, sort_keys=True))
        return 0
    if not args.confirm_step3_preauthorized:
        parser.error("stable 8+32 requires the explicit conditional Step-3 preauthorization")
    stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"corrected_loss_stable_8p32_{stamp}"
    report = run_stable_8p32(output_dir)
    print(json.dumps({
        "status": report["status"],
        "report": str(output_dir / "stable_8p32_report.json"),
        "timing_ratios": report["timing_ratios"],
        "update_only_cost_estimate_seconds": report["update_only_cost_estimate_seconds"],
        "dll_sha256": report["dll"]["sha256"],
        "loss_sha256": report["loss_source"]["sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
