"""Exercise real quality-runner checkpoint/resume semantics under a tiny budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


SEED = 20260913
TARGET_UPDATES = 4
PREFIX_UPDATES = 3
CHECKPOINT_INTERVAL = 3
ROUTE_ORDER = (("pytorch", 1), ("native", 1), ("pytorch", 4), ("native", 4))
LEDGER_FIELDS = (
    "run_id", "variant", "backend", "K", "seed", "update", "microbatch", "window", "pair",
    "document_positions", "document_id", "input_range", "target_range", "teacher_context_range",
    "valid_tokens", "loss_contract_id", "loss_callable", "loss_implementation_sha256",
    "state_reset", "state_source_update", "effective_batch", "physical_batch", "accumulations",
    "total_loss", "ce", "kl", "pre_clip_grad_norm", "clip_coefficient", "clip_intervened",
    "adamw_steps", "status",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _tensor_meta(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "stride": list(value.stride()) if value.layout == torch.strided else None,
        "storage_offset": int(value.storage_offset()) if value.layout == torch.strided else None,
    }


def _compare_tree_exact(reference: Any, candidate: Any, path: str = "root") -> dict[str, Any]:
    if torch.is_tensor(reference) and torch.is_tensor(candidate):
        meta_equal = _tensor_meta(reference) == _tensor_meta(candidate)
        same_shape = tuple(reference.shape) == tuple(candidate.shape)
        numeric_equal = bool(torch.equal(reference, candidate)) if same_shape else False
        max_abs: float | None = None
        different_count: int | None = None
        if same_shape and reference.is_floating_point() and candidate.is_floating_point():
            if numeric_equal:
                max_abs, different_count = 0.0, 0
            else:
                diff = (candidate.to(torch.float64) - reference.to(torch.float64)).abs()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                different_count = int(torch.count_nonzero(candidate != reference).item())
        elif same_shape:
            different_count = int(torch.count_nonzero(candidate != reference).item())
        return {
            "path": path,
            "kind": "tensor",
            "equal": numeric_equal and meta_equal,
            "torch_equal": numeric_equal,
            "metadata_equal": meta_equal,
            "reference_metadata": _tensor_meta(reference),
            "candidate_metadata": _tensor_meta(candidate),
            "different_element_count": different_count,
            "max_abs_diff": max_abs,
        }
    if isinstance(reference, dict) and isinstance(candidate, dict):
        keys_equal = set(reference) == set(candidate)
        children = [
            _compare_tree_exact(reference[key], candidate[key], f"{path}.{key}")
            for key in sorted(set(reference) & set(candidate), key=str)
        ]
        missing = sorted(set(reference) ^ set(candidate), key=str)
        return {"path": path, "kind": "mapping", "keys_equal": keys_equal, "missing_keys": [str(key) for key in missing], "equal": keys_equal and all(row["equal"] for row in children), "children": children}
    if isinstance(reference, (list, tuple)) and isinstance(candidate, type(reference)):
        same_length = len(reference) == len(candidate)
        children = [_compare_tree_exact(left, right, f"{path}[{index}]") for index, (left, right) in enumerate(zip(reference, candidate))]
        return {"path": path, "kind": type(reference).__name__, "length_equal": same_length, "equal": same_length and all(row["equal"] for row in children), "children": children}
    equal = reference == candidate
    return {"path": path, "kind": "scalar", "equal": bool(equal), "reference": reference, "candidate": candidate}


def _flatten_failures(report: dict[str, Any], output: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    output = output if output is not None else []
    if report.get("equal") is False:
        output.append({"path": report.get("path"), "kind": report.get("kind"), "max_abs_diff": report.get("max_abs_diff"), "different_element_count": report.get("different_element_count"), "missing_keys": report.get("missing_keys")})
    for child in report.get("children", []):
        _flatten_failures(child, output)
    return output


def _update_events(path: Path) -> dict[int, dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [event for event in events if event.get("phase") == "update_completed"]
    by_update = {int(event["update"]): event for event in rows}
    if len(by_update) != len(rows):
        raise ValueError(f"duplicate update_completed ledger row in {path}")
    return by_update


def _run_child(
    *,
    backend: str,
    rounds: int,
    init_path: Path,
    init_sha: str,
    run_dir: Path,
    phase: str,
    run_until: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
    command = [
        sys.executable, "-B", str(HERE / "run_backend_quality_qualification.py"),
        "--_technical-resume-child", "--confirm-technical-resume-test",
        "--technical-resume-phase", phase,
        "--backend", backend, "--K", str(rounds), "--seed", str(SEED),
        "--init-bundle", str(init_path), "--init-sha256", init_sha,
        "--run-dir", str(run_dir), "--run-until-update", str(run_until),
        "--target-updates", str(TARGET_UPDATES),
        "--technical-checkpoint-interval", str(CHECKPOINT_INTERVAL),
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    completed = subprocess.run(command, cwd=quality.REPO_ROOT, capture_output=True, text=True)
    (run_dir / f"{phase}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / f"{phase}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
    report_path = run_dir / f"technical_resume_{phase}_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"real runner omitted technical route report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_status = "TECHNICAL_SEGMENT_COMPLETE" if phase == "prefix" else "TECHNICAL_COMPLETE"
    if report.get("status") != expected_status:
        raise ValueError(f"real runner phase status mismatch: {phase} -> {report.get('status')}")
    return report


def _checkpoint_state_report(continuous: Path, split: Path, update: int) -> dict[str, Any]:
    continuous_path = continuous / f"checkpoint_{update:05d}.pt"
    split_path = split / f"checkpoint_{update:05d}.pt"
    continuous_sidecar = json.loads(continuous_path.with_suffix(continuous_path.suffix + ".identity.json").read_text(encoding="utf-8"))
    split_sidecar = json.loads(split_path.with_suffix(split_path.suffix + ".identity.json").read_text(encoding="utf-8"))
    if _sha256_file(continuous_path) != continuous_sidecar["sha256"] or _sha256_file(split_path) != split_sidecar["sha256"]:
        raise ValueError(f"checkpoint sidecar/file hash mismatch at completed update {update}")
    if continuous_sidecar["identity_sha256"] != split_sidecar["identity_sha256"]:
        raise ValueError(f"checkpoint execution identities differ at update {update}")
    left = torch.load(continuous_path, map_location="cpu", weights_only=False)
    right = torch.load(split_path, map_location="cpu", weights_only=False)
    if left.get("identity") != right.get("identity"):
        raise ValueError(f"checkpoint identity payload differs at update {update}")
    rows = {
        "model_parameters_and_buffers": _compare_tree_exact(left["model"], right["model"], "model"),
        "optimizer_moments_and_step": _compare_tree_exact(left["optimizer"], right["optimizer"], "optimizer"),
        "rng_states": _compare_tree_exact(left["rng_states"], right["rng_states"], "rng_states"),
        "data_cursor": _compare_tree_exact(left["data_cursor"], right["data_cursor"], "data_cursor"),
        "recurrent_state": _compare_tree_exact(left["recurrent_state"], right["recurrent_state"], "recurrent_state"),
        "completed_updates": _compare_tree_exact(left["completed_updates"], right["completed_updates"], "completed_updates"),
        "next_update": _compare_tree_exact(left["next_update"], right["next_update"], "next_update"),
    }
    equal = all(row["equal"] for row in rows.values())
    return {
        "completed_update": update,
        "continuous_checkpoint_path": str(continuous_path),
        "continuous_checkpoint_sha256": _sha256_file(continuous_path),
        "split_checkpoint_path": str(split_path),
        "split_checkpoint_sha256": _sha256_file(split_path),
        "identity_equal": True,
        "exact_state_equal": equal,
        "comparisons": rows,
        "failure_summaries": {name: _flatten_failures(row) for name, row in rows.items() if not row["equal"]},
        "excluded_metadata": ["checkpoint file hash", "last ledger event and its hash", "execution_segment", "timestamps", "elapsed time", "memory samples"],
    }


def _compare_ledger_paths(continuous: Path, split: Path) -> dict[str, Any]:
    left = _update_events(continuous / "ledger.jsonl")
    right = _update_events(split / "ledger.jsonl")
    if set(left) != set(range(TARGET_UPDATES)) or set(right) != set(range(TARGET_UPDATES)):
        raise ValueError("continuous/resumed ledger lacks one of four canonical update rows")
    rows = []
    for update in range(TARGET_UPDATES):
        left_event, right_event = left[update], right[update]
        fields = {name: {"continuous": left_event.get(name), "resumed": right_event.get(name), "equal": left_event.get(name) == right_event.get(name)} for name in LEDGER_FIELDS}
        rows.append({"update": update, "pass_exact": all(field["equal"] for field in fields.values()), "fields": fields})
    return {"pass_exact": all(row["pass_exact"] for row in rows), "canonical_event_count": TARGET_UPDATES, "updates": rows, "excluded_metadata": ["elapsed_seconds", "memory", "execution_segment"]}


def _verify_resume_start_event(split: Path, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    events = [json.loads(line) for line in (split / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    resume_event = next((event for event in events if event.get("phase") == "resume_segment_started"), None)
    if resume_event is None:
        raise ValueError("fresh resume process did not log resume_segment_started")
    restored = {
        "recurrent_state": resume_event.get("restored_recurrent_state_sha256") == quality._tensor_hash(checkpoint["recurrent_state"]),
        "model_state": resume_event.get("restored_model_state_sha256") == quality._value_hash(checkpoint["model"]),
        "optimizer_state": resume_event.get("restored_optimizer_state_sha256") == quality._value_hash(checkpoint["optimizer"]),
        "torch_rng": resume_event.get("restored_torch_rng_sha256") == quality._tensor_hash(checkpoint["rng_states"]["torch"]),
        "cursor": resume_event.get("data_cursor") == checkpoint["data_cursor"] == {"next_update": 3, "pair": 1, "window": 1},
        "window1": resume_event.get("window") == 1 and resume_event.get("state_reset") is False and resume_event.get("state_source_update") == 2,
    }
    return {"pass_exact": all(restored.values()), "resume_event": resume_event, "restored_components": restored}


def run_resume_contract_test(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"resume-test output is immutable: {output_dir}")
    manifest = quality._load_qualification_manifest()
    contract = manifest["technical_resume_test_contract"]
    if contract["total_technical_optimizer_updates"] != 32 or contract["checkpoint_after_completed_update"] != 3:
        raise ValueError("sealed resume test contract does not specify 32 updates / checkpoint at 3")
    r2, _p0, ce, _bridge, _modules = quality._load_real_dependencies()
    ce.validate_policy()
    output_dir.mkdir(parents=True, exist_ok=False)
    init_rows: dict[int, tuple[Path, str, dict[str, Any]]] = {}
    for rounds in (1, 4):
        init_path, init_sha = quality._prepare_common_initialization(ce, SEED, rounds, output_dir / "common_initializations")
        init_bundle = torch.load(init_path, map_location="cpu", weights_only=False)
        init_rows[rounds] = (init_path, init_sha, init_bundle)

    routes = []
    for backend, rounds in ROUTE_ORDER:
        init_path, init_sha, init_bundle = init_rows[rounds]
        route_root = output_dir / f"{backend}_K{rounds}_seed_{SEED}"
        continuous_dir = route_root / "continuous"
        split_dir = route_root / "segmented_resumed"
        route_root.mkdir(parents=True, exist_ok=False)
        continuous_report = _run_child(
            backend=backend, rounds=rounds, init_path=init_path, init_sha=init_sha,
            run_dir=continuous_dir, phase="continuous", run_until=4,
        )
        prefix_report = _run_child(
            backend=backend, rounds=rounds, init_path=init_path, init_sha=init_sha,
            run_dir=split_dir, phase="prefix", run_until=3,
        )
        checkpoint3 = split_dir / "checkpoint_00003.pt"
        if not checkpoint3.is_file() or not checkpoint3.with_suffix(".pt.identity.json").is_file():
            raise FileNotFoundError(f"three-update prefix omitted its own update-3 checkpoint: {checkpoint3}")
        resumed_report = _run_child(
            backend=backend, rounds=rounds, init_path=init_path, init_sha=init_sha,
            run_dir=split_dir, phase="resume", run_until=4, resume_checkpoint=checkpoint3,
        )
        continuous3_vs_prefix3 = _checkpoint_state_report(continuous_dir, split_dir, 3)
        continuous4_vs_resumed4 = _checkpoint_state_report(continuous_dir, split_dir, 4)
        ledger_comparison = _compare_ledger_paths(continuous_dir, split_dir)
        resume_start = _verify_resume_start_event(split_dir, checkpoint3)
        route_pass = (
            continuous3_vs_prefix3["exact_state_equal"]
            and continuous4_vs_resumed4["exact_state_equal"]
            and ledger_comparison["pass_exact"]
            and resume_start["pass_exact"]
            and continuous_report["updates"] == 4
            and prefix_report["updates"] == 3
            and resumed_report["updates"] == 4
        )
        routes.append({
            "backend": backend,
            "K": rounds,
            "seed": SEED,
            "route_pass": route_pass,
            "common_initialization_path": str(init_path),
            "common_initialization_sha256": init_sha,
            "common_initial_model_state_sha256": init_bundle["model_state_sha256"],
            "fresh_adamw_state_empty": init_bundle["fresh_adamw_state_empty"],
            "updates": {"continuous": 4, "prefix": 3, "resumed_suffix": 1, "optimizer_updates_total": 8},
            "child_reports": {
                "continuous": {"status": continuous_report["status"], "path": str(continuous_dir / "technical_resume_continuous_report.json"), "sha256": _sha256_file(continuous_dir / "technical_resume_continuous_report.json")},
                "prefix": {"status": prefix_report["status"], "path": str(split_dir / "technical_resume_prefix_report.json"), "sha256": _sha256_file(split_dir / "technical_resume_prefix_report.json")},
                "resumed": {"status": resumed_report["status"], "path": str(split_dir / "technical_resume_resume_report.json"), "sha256": _sha256_file(split_dir / "technical_resume_resume_report.json")},
            },
            "checkpoint3_continuous_vs_prefix": continuous3_vs_prefix3,
            "checkpoint4_continuous_vs_resumed": continuous4_vs_resumed4,
            "update_ledger_continuous_vs_resumed": ledger_comparison,
            "resume_checkpoint_restore": resume_start,
            "test_validation_loaded": any(bool(report.get("test_split_loaded", True)) for report in (continuous_report, prefix_report, resumed_report)),
        })

    status = "PASS" if len(routes) == 4 and all(route["route_pass"] and not route["test_validation_loaded"] for route in routes) else "FAIL"
    resume_report = {
        "schema": "omega-real-runner-resume-contract-test-v1",
        "status": status,
        "technical_resume_test": True,
        "scientific_block_started": False,
        "real_dll_used_for_native_routes": True,
        "real_optimizer_updates": sum(route["updates"]["optimizer_updates_total"] for route in routes),
        "expected_real_optimizer_updates": 32,
        "fresh_process_for_each_segment": True,
        "test_or_validation_loaded": False,
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source": {
            "callable": quality.r1_masked_token_mean_loss.__name__,
            "path": str(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
            "sha256": _sha256_file(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
        },
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "dll_sha256": _sha256_file(quality.BE376_DLL),
        "technical_resume_contract": contract,
        "routes": routes,
        "runner_source_identity": {"path": str(Path(quality.__file__).resolve()), "sha256": _sha256_file(Path(quality.__file__).resolve())},
        "test_harness_identity": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        "output_dir": str(output_dir),
    }
    resume_report["report_self_sha256"] = quality._canonical_hash(resume_report)
    _write_json(output_dir / "real_runner_resume_contract_report.json", resume_report)
    return resume_report


def _seal_block_a_preflight(resume_report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    manifest = quality._load_qualification_manifest()
    if resume_report.get("status") != "PASS" or resume_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
        raise PermissionError("Block A preflight report cannot be sealed without passing real-runner resume test")
    report = {
        "schema": "omega-block-a-preflight-v1",
        "status": "PASS",
        "block": "A",
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "resume_test_report_path": str(output_dir / "real_runner_resume_contract_report.json"),
        "resume_test_report_sha256": _sha256_file(output_dir / "real_runner_resume_contract_report.json"),
        "resume_test_report_self_sha256": resume_report["report_self_sha256"],
        "real_dll_sha256": quality.BE376_SHA256,
        "loss_contract": manifest["loss_contract"]["id"],
        "loss_source_sha256": manifest["loss_contract"]["implementation"]["sha256"],
        "technical_resume_updates": 32,
        "block_a_run_count": 8,
        "block_a_updates_per_run": 2000,
        "block_a_total_updates": 16000,
        "block_b_authorized": False,
        "original_causal_smoke_fail_preserved": manifest["prior_preflight_evidence"]["reports"]["original_causal_smoke"]["status"] == "FAIL",
        "strict_trajectory_proximity": "FAIL_RECORDED",
        "scientific_block_started": False,
    }
    report["report_self_sha256"] = quality._canonical_hash(report)
    _write_json(output_dir / "block_a_preflight_report.json", report)
    authorization = {
        "status": "APPROVED",
        "block": "A",
        "candidate_sha256": quality.BE376_SHA256,
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "preflight_report_sha256": _sha256_file(output_dir / "block_a_preflight_report.json"),
        "authorization_basis": "Sol DevMCP conditional GO after sealed protocol and real-runner resume PASS",
        "seeds": [20260913, 20260914],
        "K": [1, 4],
        "backends": ["pytorch", "native"],
        "updates_per_run": 2000,
        "total_updates": 16000,
        "block_b_authorized": False,
    }
    _write_json(output_dir / "block_a_authorization.json", authorization)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-sol-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_sol_go:
        parser.error("real-runner resume test requires the explicit Sol preflight authorization")
    stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"block_a_preflight_{stamp}"
    try:
        resume_report = run_resume_contract_test(output_dir)
        if resume_report["status"] == "PASS":
            preflight_report = _seal_block_a_preflight(resume_report, output_dir)
        else:
            preflight_report = None
    except Exception as error:
        if output_dir.exists():
            _write_json(output_dir / "block_a_preflight_failure.json", {
                "status": "FAILED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "output_dir": str(output_dir),
            })
        raise
    print(json.dumps({
        "status": resume_report["status"],
        "report": str(output_dir / "real_runner_resume_contract_report.json"),
        "preflight_report": str(output_dir / "block_a_preflight_report.json") if preflight_report else None,
        "authorization_file": str(output_dir / "block_a_authorization.json") if preflight_report else None,
        "technical_updates": resume_report["real_optimizer_updates"],
        "route_passes": sum(bool(route["route_pass"]) for route in resume_report["routes"]),
        "route_count": len(resume_report["routes"]),
        "dll_sha256": resume_report["dll_sha256"],
        "loss_sha256": resume_report["loss_source"]["sha256"],
        "scientific_block_started": False,
    }, indent=2, sort_keys=True))
    return 0 if resume_report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
