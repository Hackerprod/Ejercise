"""Run the authorized native-only OMEGA-R1-K-CURVE-A primary blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import psutil
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
P2R0 = HERE.parent / "omega_native_runtime_p2r0"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
for _path in (CAMPAIGN, SCRIPTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_backend_quality_qualification as quality  # noqa: E402
import run_k2_k6_preflight as k_preflight  # noqa: E402


EXPECTED_LOSS_SHA256 = "4f456775993c60dc58c63a160ecd292c37c5b53518e933cca64d209698927c88"
EXPECTED_DLL_SHA256 = "be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc"
SEEDS_BY_BLOCK = {"A": (20260913, 20260914), "B": (20260915, 20260916, 20260917)}
CURVE_DEPTHS = (1, 4, 6)
TARGET_UPDATES = 2000
CHECKPOINT_INTERVAL = 500
MAX_CONCURRENCY = 4
MIN_AVAILABLE_BYTES = 1 * 1024**3
WAIT_TIMEOUT_SECONDS = 24 * 60 * 60


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_model_factory(ce: Any, technical_model: Any):
    original = ce.fresh_model

    def fresh_model(seed: int, rounds: int, **kwargs: Any) -> torch.nn.Module:
        if rounds in getattr(ce, "KS", ()):
            return original(seed, rounds, **kwargs)
        ce.r1.configure_cpu_runtime()
        ce.set_seed(seed)
        reference = technical_model.OmegaCoreLM0R1Technical(
            vocab_size=kwargs.get("vocab_size", ce.TOKENIZER_VOCAB),
            dimension=kwargs.get("dimension", ce.DIMENSION),
            slots=kwargs.get("slots", ce.SLOTS),
            rounds=rounds,
            variant="shared",
        ).to(dtype=torch.float32)
        try:
            return ce.OmegaCoreLMFast.from_reference(reference).to(dtype=torch.float32)
        finally:
            del reference

    return fresh_model


def _child_run(args: argparse.Namespace) -> int:
    manifest = quality._load_qualification_manifest()
    r2, p0, ce, _bridge, _modules = quality._load_real_dependencies()
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("curve runs require the frozen PyTorch intraop/inter-op 4/1")
    if int(policy["physical_batch"]) != 8 or int(policy["effective_batch"]) != 8:
        raise RuntimeError("curve run batch contract is not 8/8")
    technical_model_path = SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415

    if _sha256_file(technical_model_path) != manifest["source_identities"]["r1_technical_preflight_current"]["sha256"]:
        raise ValueError("R1 technical architecture source identity drift")
    ce.fresh_model = _fresh_model_factory(ce, technical_model)
    init_path = Path(args.init_bundle).resolve()
    if _sha256_file(init_path) != args.init_sha256:
        raise ValueError("curve route common-initialization SHA mismatch")
    run_dir = Path(args.run_dir).resolve()
    resume_path = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    run_report = quality._quality_run_child(
        block=args.block,
        backend="native",
        rounds=args.K,
        seed=args.seed,
        init_path=init_path,
        init_sha=args.init_sha256,
        run_dir=run_dir,
        manifest=manifest,
        resume_path=resume_path,
        run_until_update=TARGET_UPDATES,
        target_updates=TARGET_UPDATES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        technical_resume_test=False,
    )
    print(json.dumps({"status": run_report["status"], "run_id": run_report["run_id"], "K": args.K, "seed": args.seed, "updates": run_report["updates"], "endpoint": run_report["endpoint_nll_validation"]}, sort_keys=True))
    return 0


def _latest_valid_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[-1]), reverse=True)
    for path in candidates:
        sidecar = path.with_suffix(path.suffix + ".identity.json")
        if not sidecar.is_file():
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if _sha256_file(path) == meta.get("sha256"):
            return path
    return None


def _initialize_route_bundles(output_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r2, _p0, ce, _bridge, _modules = quality._load_real_dependencies()
    ce.validate_policy()
    technical_model_path = SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415
    if _sha256_file(technical_model_path) != manifest["source_identities"]["r1_technical_preflight_current"]["sha256"]:
        raise ValueError("R1 technical architecture source identity drift")
    ce.fresh_model = _fresh_model_factory(ce, technical_model)
    init_dir = output_dir / "common_initializations"
    init_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict[str, Any]] = {}
    for seed in SEEDS_BY_BLOCK["A"] + SEEDS_BY_BLOCK["B"]:
        for rounds in CURVE_DEPTHS:
            label = f"K{rounds}_seed_{seed}"
            path = init_dir / f"common_init_{label}.pt"
            if path.is_file():
                digest = _sha256_file(path)
                bundle = torch.load(path, map_location="cpu", weights_only=False)
                if int(bundle.get("seed", -1)) != seed or int(bundle.get("K", -1)) != rounds:
                    raise ValueError(f"existing curve common init identity mismatch: {path}")
            elif rounds in getattr(ce, "KS", ()):
                path, digest = quality._prepare_common_initialization(ce, seed, rounds, init_dir)
                bundle = torch.load(path, map_location="cpu", weights_only=False)
            else:
                bundle, path, digest = k_preflight._make_common_initialization(
                    ce=ce,
                    technical_model=technical_model,
                    seed=seed,
                    rounds=rounds,
                    output_dir=init_dir,
                )
            specs[label] = {
                "seed": seed,
                "K": rounds,
                "path": str(path),
                "sha256": digest,
                "model_state_sha256": bundle["model_state_sha256"],
                "initial_state_sha256": bundle["initial_recurrent_state_sha256"],
                "fresh_adamw_state_empty": bool(bundle.get("fresh_adamw_state_empty", True)),
                "source": "fresh i7 common initialization; reference construction then F conversion",
            }
    sentinel_path = HERE / "results" / "k6_seed_sentinel_20260927_md281" / "k6_seed_sentinel_report.json"
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    sentinel_init = sentinel["common_initialization"]
    current_k6_13 = specs["K6_seed_20260913"]
    sentinel_bundle = torch.load(sentinel_init["path"], map_location="cpu", weights_only=False)
    fresh_bundle = torch.load(current_k6_13["path"], map_location="cpu", weights_only=False)
    specs["K6_seed_20260913"]["sentinel_model_state_bitwise_matches"] = (
        quality._value_hash(sentinel_bundle["model_state"]) == quality._value_hash(fresh_bundle["model_state"])
    )
    specs["K6_seed_20260913"]["sentinel_checkpoint_reused"] = False
    specs["K6_seed_20260913"]["sentinel_difference_reason"] = "new Block-A run identity/common-init path required; prior sentinel remains evidence only"
    return specs


def _run_block(output_dir: Path, block: str, *, resume: bool = False) -> dict[str, Any]:
    if block not in ("A", "B"):
        raise ValueError(block)
    block_dir = output_dir / f"block_{block}"
    progress_path = output_dir / f"block_{block}_progress.json"
    if block_dir.exists() and not resume:
        raise FileExistsError(f"Block {block} already exists; resume explicitly to continue")
    block_dir.mkdir(parents=True, exist_ok=True)
    manifest = quality._load_qualification_manifest()
    init_specs = _initialize_route_bundles(output_dir, manifest)
    seeds = SEEDS_BY_BLOCK[block]
    expected_specs = [(seed, rounds) for seed in seeds for rounds in CURVE_DEPTHS]
    if resume and progress_path.is_file():
        report = json.loads(progress_path.read_text(encoding="utf-8"))
        if report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"Block {block} progress manifest mismatch")
    else:
        report = {
            "schema": "omega-r1-k-curve-a-block-report-v1",
            "unit": "OMEGA-R1-K-CURVE-A",
            "block": block,
            "status": "IN_PROGRESS",
            "seeds": list(seeds),
            "K_values": list(CURVE_DEPTHS),
            "backend": "native",
            "qualification_manifest_sha256": manifest["manifest_sha256"],
            "loss_contract": quality.LOSS_CONTRACT,
            "loss_source_sha256": _sha256_file(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
            "native_dll_sha256": quality.BE376_SHA256,
            "torch_policy": {"intraop": 4, "interop": 1},
            "native_workers": 4,
            "physical_batch": 8,
            "effective_batch": 8,
            "updates_per_run": TARGET_UPDATES,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "initializations": {key: init_specs[key] for seed, k in expected_specs for key in [f"K{k}_seed_{seed}"]},
            "routes": {},
            "scientific_updates": 0,
            "test_split_loaded": False,
            "sentinel_reused": False,
            "output_dir": str(block_dir),
        }
        quality._write_json(progress_path, report)

    results_dir = output_dir / "route_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed, rounds in expected_specs:
        key = f"native_K{rounds}_seed_{seed}"
        run_dir = block_dir / key
        final_report_path = run_dir / "run_report.json"
        if final_report_path.is_file():
            final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
            report["routes"][key] = {
                "status": final_report["status"],
                "run_dir": str(run_dir),
                "run_report_path": str(final_report_path),
                "run_report_sha256": _sha256_file(final_report_path),
                "endpoint_nll_validation": final_report["endpoint_nll_validation"],
                "updates": final_report["updates"],
                "resumed_or_reused": True,
            }
            continue
        init = init_specs[f"K{rounds}_seed_{seed}"]
        resume_checkpoint = _latest_valid_checkpoint(run_dir) if run_dir.exists() and any(run_dir.iterdir()) else None
        if run_dir.exists() and any(run_dir.iterdir()) and resume_checkpoint is None:
            raise FileNotFoundError(f"nonempty route directory has no valid resumable checkpoint: {run_dir}")
        pending.append({
            "key": key,
            "seed": seed,
            "K": rounds,
            "run_dir": str(run_dir),
            "init_path": init["path"],
            "init_sha256": init["sha256"],
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        })

    for wave_index in range(0, len(pending), MAX_CONCURRENCY):
        wave = pending[wave_index : wave_index + MAX_CONCURRENCY]
        child_processes: list[tuple[dict[str, Any], subprocess.Popen[Any], Any, Any]] = []
        for spec in wave:
            route_output = results_dir / f"{block}_{spec['key']}.stdout.log"
            route_error = results_dir / f"{block}_{spec['key']}.stderr.log"
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--block",
                block,
                "--K",
                str(spec["K"]),
                "--seed",
                str(spec["seed"]),
                "--run-dir",
                spec["run_dir"],
                "--init-bundle",
                spec["init_path"],
                "--init-sha256",
                spec["init_sha256"],
            ]
            if spec["resume_checkpoint"]:
                command.extend(["--resume-checkpoint", spec["resume_checkpoint"]])
            stdout_stream = route_output.open("ab")
            stderr_stream = route_error.open("ab")
            process = subprocess.Popen(command, cwd=str(LAB_ROOT.parent), stdout=stdout_stream, stderr=stderr_stream)
            child_processes.append((spec, process, stdout_stream, stderr_stream))
            report["routes"][spec["key"]] = {
                "status": "RUNNING",
                "pid": process.pid,
                "run_dir": spec["run_dir"],
                "resume_checkpoint": spec["resume_checkpoint"],
            }
            quality._write_json(progress_path, report)

        wave_started = time.monotonic()
        wave_anomaly: str | None = None
        while any(process.poll() is None for _spec, process, _out, _err in child_processes):
            if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
                wave_anomaly = "available_memory_below_1GiB"
                break
            failed = [(spec, process.returncode) for spec, process, _out, _err in child_processes if process.poll() not in (None, 0)]
            if failed:
                wave_anomaly = f"child_route_failed:{failed}"
                break
            for spec, process, _out, _err in child_processes:
                route_dir = Path(spec["run_dir"])
                run_report_path = route_dir / "run_report.json"
                if run_report_path.is_file():
                    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
                    report["routes"][spec["key"]] = {
                        "status": run_report["status"],
                        "pid": process.pid,
                        "run_dir": str(route_dir),
                        "run_report_path": str(run_report_path),
                        "run_report_sha256": _sha256_file(run_report_path),
                        "endpoint_nll_validation": run_report["endpoint_nll_validation"],
                        "updates": run_report["updates"],
                    }
            quality._write_json(progress_path, report)
            time.sleep(2.0)
        for _spec, process, stdout_stream, stderr_stream in child_processes:
            if process.poll() is None and wave_anomaly:
                process.terminate()
        for _spec, process, stdout_stream, stderr_stream in child_processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
            stdout_stream.close()
            stderr_stream.close()
        if wave_anomaly:
            report["status"] = "HOLD_TECHNICAL_ANOMALY"
            report["technical_anomaly"] = {"reason": wave_anomaly, "wave_started_monotonic": wave_started}
            quality._write_json(progress_path, report)
            return report

        for spec, process, _out, _err in child_processes:
            route_dir = Path(spec["run_dir"])
            run_report_path = route_dir / "run_report.json"
            if process.returncode != 0 or not run_report_path.is_file():
                report["status"] = "HOLD_TECHNICAL_ANOMALY"
                report["technical_anomaly"] = {"route": spec["key"], "return_code": process.returncode, "run_report_exists": run_report_path.is_file()}
                quality._write_json(progress_path, report)
                return report
            run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
            endpoint = run_report["endpoint_nll_validation"]
            route_pass = (
                run_report.get("status") == "COMPLETE"
                and int(run_report.get("updates", -1)) == TARGET_UPDATES
                and int(endpoint.get("update", -1)) == TARGET_UPDATES
                and int(endpoint.get("tokens", -1)) == 30720
                and bool(math.isfinite(float(endpoint.get("nll", float("nan")))))
                and run_report.get("test_split_loaded") is False
                and run_report.get("scientific_quality_training") is True
            )
            report["routes"][spec["key"]] = {
                "status": run_report["status"],
                "run_dir": str(route_dir),
                "run_report_path": str(run_report_path),
                "run_report_sha256": _sha256_file(run_report_path),
                "endpoint_nll_validation": endpoint,
                "updates": run_report["updates"],
                "route_pass": route_pass,
                "scientific_quality_training": run_report["scientific_quality_training"],
            }
            if not route_pass:
                report["status"] = "HOLD_TECHNICAL_ANOMALY"
                report["technical_anomaly"] = {"route": spec["key"], "reason": "route contract/status/endpoint verification failed"}
                quality._write_json(progress_path, report)
                return report
            report["scientific_updates"] += TARGET_UPDATES
            quality._write_json(progress_path, report)

    route_passes = all(row.get("route_pass") is True for row in report["routes"].values())
    report["status"] = "COMPLETE" if route_passes and len(report["routes"]) == len(expected_specs) else "HOLD_TECHNICAL_ANOMALY"
    report["routes_expected"] = len(expected_specs)
    report["routes_completed"] = len(report["routes"])
    report["scientific_updates"] = sum(int(row.get("updates", 0)) for row in report["routes"].values())
    report["self_sha256"] = quality._canonical_hash(report)
    final_report_path = output_dir / f"block_{block}_report.json"
    quality._write_json(final_report_path, report)
    report["report_path"] = str(final_report_path)
    quality._write_json(progress_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-k-curve-block-go", action="store_true")
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--K", type=int, choices=CURVE_DEPTHS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-bundle", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        required = (args.K, args.seed, args.run_dir, args.init_bundle, args.init_sha256)
        if any(value is None for value in required):
            parser.error("curve worker arguments incomplete")
        return _child_run(args)
    if not args.confirm_k_curve_block_go:
        if not args.worker:
            parser.error("curve block runner requires explicit OMEGA-R1-K-CURVE-A GO")
    if args.output_dir is None and not args.worker:
        parser.error("curve parent invocation requires --output-dir")
    output_dir = args.output_dir.resolve()
    if args.block == "B":
        block_a_path = output_dir / "block_A_report.json"
        if not block_a_path.is_file():
            raise PermissionError("Block B is automatically authorized only after a complete Block A report")
        block_a = json.loads(block_a_path.read_text(encoding="utf-8"))
        unsigned = dict(block_a)
        signature = unsigned.pop("self_sha256", None)
        if (
            block_a.get("status") != "COMPLETE"
            or not signature
            or signature != quality._canonical_hash(unsigned)
            or any(route.get("route_pass") is not True for route in block_a.get("routes", {}).values())
        ):
            raise PermissionError("Block A did not complete cleanly; automatic Block B GO is unavailable")
    try:
        report = _run_block(output_dir, args.block, resume=args.resume)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / f"block_{args.block}_failure.json",
                {"status": "FAILED_RUNTIME", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "block": args.block},
            )
        raise
    print(json.dumps({"block": args.block, "status": report["status"], "routes_completed": report.get("routes_completed"), "scientific_updates": report.get("scientific_updates"), "report_path": report.get("report_path")}, indent=2, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
