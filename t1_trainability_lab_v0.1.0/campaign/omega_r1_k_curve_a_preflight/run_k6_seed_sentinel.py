"""Run the authorized paired K6 seed-20260913 sentinel to update 2000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_backend_quality_qualification as quality  # noqa: E402


SEED = 20260913
K = 6
TARGET_UPDATES = 2000
CHECKPOINT_INTERVAL = 500
DELTA_LIMIT = 0.02


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_report_identity(report: dict[str, Any]) -> None:
    unsigned = dict(report)
    signature = unsigned.pop("report_self_sha256", None)
    if not signature or signature != quality._canonical_hash(unsigned):
        raise ValueError("K2/K6 preflight report self-hash mismatch")
    k6 = report.get("per_depth", {}).get("6", {})
    common = k6.get("common_origin", {})
    if (
        common.get("common_origin_pass") is not True
        or common.get("window0", {}).get("native_vs_pytorch_gate", {}).get("pass") is not True
        or common.get("window1", {}).get("native_vs_pytorch_gate", {}).get("pass") is not True
    ):
        raise PermissionError("K6 common-origin W0/W1 gates have not passed")


def _install_k6_factory(ce: Any, technical_model: Any) -> None:
    def fresh_model(seed: int, rounds: int, **kwargs: Any) -> torch.nn.Module:
        if int(rounds) != K:
            raise ValueError(f"sentinel factory is scoped to K6, received K={rounds}")
        ce.r1.configure_cpu_runtime()
        ce.set_seed(seed)
        reference = technical_model.OmegaCoreLM0R1Technical(
            vocab_size=kwargs.get("vocab_size", ce.TOKENIZER_VOCAB),
            dimension=kwargs.get("dimension", ce.DIMENSION),
            slots=kwargs.get("slots", ce.SLOTS),
            rounds=K,
            variant="shared",
        ).to(dtype=torch.float32)
        try:
            return ce.OmegaCoreLMFast.from_reference(reference).to(dtype=torch.float32)
        finally:
            del reference

    ce.fresh_model = fresh_model


def run_sentinel(preflight_report_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"K6 sentinel output is immutable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    _check_report_identity(preflight)
    manifest = quality._load_qualification_manifest()
    if preflight.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("K6 sentinel preflight report uses a different qualification manifest")

    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("K6 sentinel requires torch intraop/inter-op 4/1")
    if quality._sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("certified be37623d DLL identity mismatch before sentinel")

    technical_path = SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415

    if _sha256_file(technical_path) != manifest["source_identities"]["r1_technical_preflight_current"]["sha256"]:
        raise ValueError("R1 technical architecture source SHA mismatch")
    _install_k6_factory(ce, technical_model)

    init_row = preflight["common_initializations"][str(K)]
    init_path = Path(init_row["path"])
    init_sha = str(init_row["sha256"])
    if _sha256_file(init_path) != init_sha:
        raise ValueError("K6 seed-centinel common initialization hash mismatch")
    init_bundle = torch.load(init_path, map_location="cpu", weights_only=False)
    if int(init_bundle.get("seed", -1)) != SEED or int(init_bundle.get("K", -1)) != K:
        raise ValueError("K6 sentinel common initialization seed/K mismatch")

    report: dict[str, Any] = {
        "schema": "omega-r1-k-curve-k6-seed-sentinel-v1",
        "unit": "OMEGA-R1-K-CURVE-A",
        "status": "IN_PROGRESS",
        "seed": SEED,
        "K": K,
        "target_updates": TARGET_UPDATES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source_sha256": quality._sha256_file(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
        "common_initialization": {"path": str(init_path), "sha256": init_sha, "model_state_sha256": init_bundle["model_state_sha256"]},
        "dll": {"path": str(quality.BE376_DLL), "sha256": quality._sha256_file(quality.BE376_DLL)},
        "runtime_policy": policy,
        "torch": {
            "version": str(torch.__version__),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cpu": {"processor": platform.processor(), "machine": platform.machine(), "platform": platform.platform()},
        "routes": {},
        "training_updates": 0,
        "scientific_quality_runs": 2,
        "sentinel_complete": False,
        "delta_limit_nats_per_token": DELTA_LIMIT,
        "sentinel_started": True,
        "output_dir": str(output_dir),
    }
    quality._write_json(output_dir / "sentinel_progress.json", report)

    for backend in ("pytorch", "native"):
        run_dir = output_dir / backend
        run_report = quality._quality_run_child(
            block="KCURVE_SENTINEL",
            backend=backend,
            rounds=K,
            seed=SEED,
            init_path=init_path,
            init_sha=init_sha,
            run_dir=run_dir,
            manifest=manifest,
            run_until_update=TARGET_UPDATES,
            target_updates=TARGET_UPDATES,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            technical_resume_test=False,
        )
        run_report_path = run_dir / "run_report.json"
        report["routes"][backend] = {
            "status": run_report["status"],
            "run_id": run_report["run_id"],
            "run_dir": str(run_dir),
            "run_report_path": str(run_report_path),
            "run_report_sha256": _sha256_file(run_report_path),
            "endpoint_nll_validation": run_report["endpoint_nll_validation"],
            "checkpoint_02000_sha256": _sha256_file(run_dir / "checkpoint_02000.pt"),
            "scientific_quality_training": run_report["scientific_quality_training"],
        }
        report["training_updates"] += int(run_report["updates"])
        quality._write_json(output_dir / "sentinel_progress.json", report)

    pytorch_nll = float(report["routes"]["pytorch"]["endpoint_nll_validation"]["nll"])
    native_nll = float(report["routes"]["native"]["endpoint_nll_validation"]["nll"])
    delta = native_nll - pytorch_nll
    report["backend_delta"] = {
        "delta_K6_native_minus_pytorch": delta,
        "absolute_delta": abs(delta),
        "limit": DELTA_LIMIT,
        "pass": abs(delta) <= DELTA_LIMIT,
    }
    report["status"] = "K6_SENTINEL_PASS" if report["backend_delta"]["pass"] else "K6_SENTINEL_BACKEND_DELTA_HOLD"
    report["sentinel_complete"] = True
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_dir / "k6_seed_sentinel_report.json", report)
    quality._write_json(output_dir / "sentinel_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-k6-sentinel-go", action="store_true")
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_k6_sentinel_go:
        parser.error("K6 seed sentinel requires the explicit MD/281 GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"k6_seed_sentinel_{stamp}"
    try:
        report = run_sentinel(args.preflight_report.resolve(), output_dir)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "sentinel_failure.json",
                {
                    "status": "FAILED_RUNTIME",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "sentinel_started": True,
                    "output_dir": str(output_dir),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_updates": report["training_updates"],
                "backend_delta": report["backend_delta"],
                "report": str(output_dir / "k6_seed_sentinel_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["backend_delta"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
