"""Capture fresh-process native K1/K4 end-to-end baseline/candidate pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKER = HERE / "run_omega_native_runtime_r2_benchmark.py"
BASELINE_SHA256 = "6f38e3b1dcada32c98aef50e870e201510bab9c75b875438823915ba53b854d5"
VARIANTS_BY_PAIR = {
    1: ("baseline", "candidate"),
    2: ("candidate", "baseline"),
    3: ("baseline", "candidate"),
}
DLLS: dict[str, dict[str, Any]] = {
    "baseline": {
        "path": Path(r"C:\Users\danil\AppData\Local\Temp\opencode\omega-backward-diagnostics-fastpath\python\omega_recurrent.dll"),
        "sha256": BASELINE_SHA256,
    },
    "candidate": {
        "path": Path(r"C:\Users\danil\bpf2a\t1_trainability_lab_v0.1.0\campaign\omega_native_runtime_p2r0\native\build-qkv-fastpath-2a-candidate\python\omega_recurrent.dll"),
        "sha256": "90d9aa3177b3fabdfb0f80a3360bebcaf9d52409f065f066213e3328dbd255c3",
    },
}
MANIFEST = Path(r"D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_teacher_hidden_cache_probe\results\cache_manifest.json")
CACHE = Path(r"C:\omega_cache\teacher_hidden.fp32")
DEFAULT_OUTPUT = HERE / "results" / "backward_diagnostics_fastpath_2a_e2e_matrix"
K_VALUES = (1, 4)
WARMUP_UPDATES = 2
MEASURED_UPDATES = 6
ENV_NAMES_AUDIT = ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _plan() -> list[dict[str, Any]]:
    return [
        {"K": k, "pair_id": pair_id, "position": position, "variant": variant}
        for k in K_VALUES
        for pair_id, variants in VARIANTS_BY_PAIR.items()
        for position, variant in enumerate(variants, start=1)
    ]


def _extract_route(report: Mapping[str, Any], route: str, dll: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("phase") != "stable" or report.get("runtime_threads") != 4 or set(report.get("routes", {})) != {route}:
        raise ValueError(f"unexpected end-to-end route envelope: {route}")
    row = report["routes"][route]
    if row.get("route") != route or row.get("phase") != "stable" or row.get("runtime_threads") != 4:
        raise ValueError(f"route config mismatch: {route}")
    if row.get("warmup") != {"updates": WARMUP_UPDATES, "measured": False}:
        raise ValueError(f"warmup count mismatch: {route}")
    if row.get("measured") != {"updates": MEASURED_UPDATES, "measured": True}:
        raise ValueError(f"measured update count mismatch: {route}")
    metadata = row.get("dll_metadata", {})
    if metadata.get("sha256") != dll["sha256"] or Path(metadata.get("path", "")).resolve() != Path(dll["path"]).resolve():
        raise ValueError(f"loaded DLL identity mismatch for {route}")
    updates = row.get("updates", [])
    if len(updates) != MEASURED_UPDATES:
        raise ValueError(f"measured update record count mismatch: {route}")
    times = []
    for index, item in enumerate(updates):
        seconds = item.get("timing_seconds", {}).get("total_update")
        if int(item.get("update", -1)) != index or not isinstance(seconds, (float, int)) or seconds <= 0:
            raise ValueError(f"invalid measured total_update for {route}, update {index}")
        times.append(float(seconds))
    return {
        "route": route,
        "K": int(row["K"]),
        "runtime_threads": int(row["runtime_threads"]),
        "warmup_updates": int(row["warmup"]["updates"]),
        "measured_updates": int(row["measured"]["updates"]),
        "dll_metadata": dict(metadata),
        "measured_total_update_seconds": times,
        "sum_measured_total_update_seconds": sum(times),
    }


def _aggregate(processes: Sequence[Mapping[str, Any]], dlls: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(processes) != len(K_VALUES) * len(VARIANTS_BY_PAIR) * 2:
        raise ValueError("expected 12 fresh route processes")
    per_k: dict[str, Any] = {}
    expected_plan = _plan()
    for item, expected in zip(processes, expected_plan):
        if any(item.get(key) != expected[key] for key in ("K", "pair_id", "position", "variant")) or item.get("status") != "PASS":
            raise ValueError(f"process order/status mismatch: {item} expected {expected}")

    for k in K_VALUES:
        matching = [item for item in processes if int(item["K"]) == k]
        baseline_total = sum(float(item["route_report"]["sum_measured_total_update_seconds"])
                             for item in matching if item["variant"] == "baseline")
        candidate_total = sum(float(item["route_report"]["sum_measured_total_update_seconds"])
                              for item in matching if item["variant"] == "candidate")
        pairs: dict[str, Any] = {}
        for pair_id in (1, 2, 3):
            pair = [item for item in matching if int(item["pair_id"]) == pair_id]
            values = {str(item["variant"]): float(item["route_report"]["sum_measured_total_update_seconds"]) for item in pair}
            ratio = values["candidate"] / values["baseline"]
            pairs[str(pair_id)] = {
                "baseline_seconds": values["baseline"],
                "candidate_seconds": values["candidate"],
                "R_candidate_over_baseline": ratio,
                "speedup_percent": (1.0 - ratio) * 100.0,
            }
        ratios = [float(value["R_candidate_over_baseline"]) for value in pairs.values()]
        per_k[f"K{k}"] = {
            "runtime_threads": 4,
            "warmup_updates_per_route_process": WARMUP_UPDATES,
            "measured_updates_per_route_process": MEASURED_UPDATES,
            "process_count": len(matching),
            "baseline_total_seconds": baseline_total,
            "candidate_total_seconds": candidate_total,
            "R_candidate_over_baseline": candidate_total / baseline_total,
            "speedup_percent": (1.0 - candidate_total / baseline_total) * 100.0,
            "pair_ratios": pairs,
            "favorable_pairs": sum(ratio < 1.0 for ratio in ratios),
        }
    return {
        "schema": "omega-backward-diagnostics-fastpath-2a-e2e-aggregate-v1",
        "status": "E2E_MATRIX_PASS",
        "timing_performed": True,
        "torch_route_measured": False,
        "runtime_threads": 4,
        "process_count": len(processes),
        "processes_per_K": 6,
        "protocol": {
            "pairs_per_K": 3,
            "pair_order": {"1": ["baseline", "candidate"], "2": ["candidate", "baseline"], "3": ["baseline", "candidate"]},
            "fresh_process_per_route": True,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
        },
        "dlls": dict(dlls),
        "results": per_k,
        "raw_process_reports": [str(item["report_path"]) for item in processes],
    }


def run_matrix(output_root: Path, timeout_seconds: int, dll_overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    dlls: dict[str, dict[str, Any]] = {
        name: dict(identity) for name, identity in (dll_overrides or DLLS).items()
    }
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"output parent must exist: {output_root.parent}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite end-to-end matrix: {output_root}")
    if not WORKER.is_file() or not MANIFEST.is_file() or not CACHE.is_file():
        raise FileNotFoundError("R2 worker or sealed manifest/cache missing")
    for name, identity in dlls.items():
        path = Path(identity["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != identity["sha256"]:
            raise ValueError(f"{name} DLL SHA mismatch: {actual}")

    output_root.mkdir(parents=False)
    raw_root = output_root / "raw_process_io"
    raw_root.mkdir()
    process_results: list[dict[str, Any]] = []
    manifest_path = output_root / "matrix_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "omega-backward-diagnostics-fastpath-2a-e2e-matrix-v1",
        "status": "RUNNING",
        "run_root": output_root.as_posix(),
        "dlls": {name: {**identity, "path": str(Path(identity["path"]).resolve())} for name, identity in dlls.items()},
        "manifest": {"path": MANIFEST.as_posix(), "sha256": _sha256(MANIFEST)},
        "cache": {"path": CACHE.as_posix(), "sha256": _sha256(CACHE)},
        "runtime_threads": 4,
        "protocol": {
            "fresh_process_per_route": True,
            "processes_per_K": 6,
            "pairs_per_K": 3,
            "pair_order": {"1": ["baseline", "candidate"], "2": ["candidate", "baseline"], "3": ["baseline", "candidate"]},
            "warmup_updates_per_route": WARMUP_UPDATES,
            "measured_updates_per_route": MEASURED_UPDATES,
        },
        "process_plan": _plan(),
        "process_results": process_results,
    }
    _write_json(manifest_path, manifest)

    for item in _plan():
        k = int(item["K"])
        pair_id = int(item["pair_id"])
        position = int(item["position"])
        variant = str(item["variant"])
        dll = Path(dlls[variant]["path"]).resolve()
        run_name = f"K{k}_pair{pair_id}_{position:02d}_{variant}"
        report_path = output_root / f"{run_name}.json"
        stdout_path = raw_root / f"{run_name}.stdout.txt"
        stderr_path = raw_root / f"{run_name}.stderr.txt"
        exit_path = raw_root / f"{run_name}.exit.json"
        argv = [
            sys.executable, "-B", str(WORKER),
            "--phase", "stable",
            "--route", f"native-k{k}",
            "--updates", str(MEASURED_UPDATES),
            "--warmup-updates", str(WARMUP_UPDATES),
            "--confirm-real-execution",
            "--manifest", str(MANIFEST),
            "--cache-file", str(CACHE),
            "--dll", str(dll),
            "--output", str(report_path),
        ]
        child_env = os.environ.copy()
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        timed_out = False
        try:
            completed = subprocess.run(
                argv, cwd=REPO_ROOT, env=child_env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout_seconds, check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        report: dict[str, Any] = {}
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        try:
            stdout_json = json.loads(stdout)
            stdout_valid = stdout_json == report
        except json.JSONDecodeError:
            stdout_valid = False
        route_report = report.get("routes", {}).get(f"native-k{k}", {})
        metadata = route_report.get("dll_metadata", {})
        process_status = {
            "K": k,
            "pair_id": pair_id,
            "position": position,
            "variant": variant,
            "expected_dll_sha256": dlls[variant]["sha256"],
            "argv": argv,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "stdout_json_valid": stdout_valid,
            "loaded_dll_sha256": metadata.get("sha256"),
            "report_path": report_path.as_posix(),
            "exit_record_path": exit_path.as_posix(),
            "status": "PASS" if exit_code == 0 and not timed_out and stdout_valid and report.get("phase") == "stable" and metadata.get("sha256") == dlls[variant]["sha256"] else "FAIL",
            "route_report": _extract_route(report, f"native-k{k}", dlls[variant]) if report else {},
        }
        _write_json(exit_path, process_status)
        process_results.append(process_status)
        manifest["status"] = "RUNNING" if process_status["status"] == "PASS" else "BLOCKED"
        _write_json(manifest_path, manifest)
        if process_status["status"] != "PASS":
            manifest["failure"] = process_status
            manifest["stop_reason"] = "route process identity/status/output validation failed"
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"end-to-end child blocked at {run_name}; inspect {exit_path}")

    aggregate = _aggregate(process_results, dlls)
    aggregate_path = output_root / "aggregate_report.json"
    _write_json(aggregate_path, aggregate)
    manifest["status"] = "E2E_MATRIX_PASS"
    manifest["aggregate_path"] = aggregate_path.as_posix()
    manifest["aggregate_status"] = aggregate["status"]
    _write_json(manifest_path, manifest)
    return {"matrix_manifest": manifest, "aggregate": aggregate}


def finalize_existing(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RUNNING" or len(manifest.get("process_results", [])) != 12:
        raise ValueError("existing matrix is not a complete unaggregated 12-process run")
    if any(result.get("status") != "PASS" for result in manifest["process_results"]):
        raise ValueError("refusing to aggregate a matrix with failed route processes")
    aggregate = _aggregate(manifest["process_results"], manifest["dlls"])
    aggregate_path = output_root / "aggregate_report.json"
    _write_json(aggregate_path, aggregate)
    manifest["status"] = "E2E_MATRIX_PASS"
    manifest["aggregate_path"] = aggregate_path.as_posix()
    manifest["aggregate_status"] = aggregate["status"]
    _write_json(manifest_path, manifest)
    return {"matrix_manifest": manifest, "aggregate": aggregate}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--baseline-dll", type=Path)
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--candidate-dll", type=Path)
    parser.add_argument("--candidate-sha256")
    args = parser.parse_args(argv)
    dll_overrides = None
    if any(value is not None for value in (args.baseline_dll, args.baseline_sha256, args.candidate_dll, args.candidate_sha256)):
        if any(value is None for value in (args.baseline_dll, args.baseline_sha256, args.candidate_dll, args.candidate_sha256)):
            parser.error("custom DLL pair requires --baseline-dll, --baseline-sha256, --candidate-dll, and --candidate-sha256")
        dll_overrides = {
            "baseline": {"path": str(args.baseline_dll.resolve()), "sha256": args.baseline_sha256.lower()},
            "candidate": {"path": str(args.candidate_dll.resolve()), "sha256": args.candidate_sha256.lower()},
        }
    result = (finalize_existing(args.output_root.resolve()) if args.finalize_existing else
              run_matrix(args.output_root.resolve(), args.timeout_seconds, dll_overrides))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
