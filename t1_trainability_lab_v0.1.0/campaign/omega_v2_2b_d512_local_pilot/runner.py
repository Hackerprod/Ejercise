"""Single-path runner for the OMEGA V2-2B d512 pilot."""

from __future__ import annotations

import argparse
import ast
import builtins
import compileall
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import symtable
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor

from . import CAMPAIGN_ROOT, D3Q_ROOT, PACKAGE_ROOT, V20_ROOT, V22A_ROOT
from .config import (
    ARTIFACT_CONTRACT,
    CALIBRATION_SEED,
    CAPACITY_REASONS,
    CONFORMANCE_BATCH,
    D,
    D1_ELEMENTWISE_ABS_TOL,
    D1_ELEMENTWISE_FLOOR,
    D1_ELEMENTWISE_REL_TOL,
    D1_K_VALUES,
    D1_NORMWISE_E_L2_LIMIT,
    D1_NORMWISE_FLOOR,
    D3_E_INF_LIMIT,
    D3_E_L2_LIMIT,
    D3_MAX_ABS_ULP_LIMIT,
    D6_BACKWARD_K_VALUES,
    D6_FORWARD_K_VALUES,
    D6_MASTER_SEED,
    D7_FINAL_LOSS_RATIO,
    D8_GC_BEFORE_CELL,
    D8_PRE_AND_POST_CELL_SYNCHRONIZE,
    D8_PRE_CELL_EMPTY_CACHE,
    D8_PRE_CELL_RESET_PEAK_STATS,
    D8_VRAM_BUDGET_BYTES,
    D8_WALL_LIMIT_SECONDS,
    K4,
    M,
    OFFICIAL_ID,
    OFFICIAL_LAUNCH_LOG_DIR_NAME,
    OFFICIAL_MASTER_SEEDS,
    OFFICIAL_RESULT_SLOT_NAME,
    TERMINAL_CALIBRATION_QA_CAPACITY_HOLD,
    TERMINAL_CALIBRATION_QA_HARNESS_HOLD,
    TERMINAL_CALIBRATION_QA_SCIENTIFIC_HOLD,
    TERMINAL_CALIBRATION_SMOKE_COMPLETE,
    TERMINAL_CAPACITY_HOLD,
    TERMINAL_FAIL,
    TERMINAL_PASS,
    TRAINABILITY_BATCH,
    SeedPlan,
    make_seed_plan,
    official_seed_plans,
)
from .core import (
    NonFinitePilotState,
    StructuralCorruption,
    configure_cuda_reference,
    run_calibration_qa_cpu,
    run_d1_cell,
    run_d2_cell,
    run_d3_cell,
    run_d4_cpu,
    run_d6_cell,
    run_d7_cell,
)
from .ledger import recompute_d512_ledger
from .metrics import tensor_raw_sha256


SPEC_PATH = PACKAGE_ROOT / "OMEGA_V2_2B_SPEC.md"
SOURCE_SEAL_PATH = PACKAGE_ROOT / "SOURCE_SEAL.json"
QA_ROOT = PACKAGE_ROOT / "results" / "qa"
QA_REPORT_PATH = QA_ROOT / "QA_REPORT.json"
CALIBRATION_SMOKE_ROOT = QA_ROOT / "calibration_seed_20260930_smoke_02"
CALIBRATION_SMOKE_LOG_ROOT = QA_ROOT / "calibration_seed_20260930_smoke_02_launch_logs"
CALIBRATION_SOURCE_SNAPSHOT_PATH = PACKAGE_ROOT / "CALIBRATION_SOURCE_SNAPSHOT.json"
CALIBRATION_SMOKE_01_ROOT = QA_ROOT / "calibration_seed_20260930_smoke_01"
CALIBRATION_SMOKE_INCIDENT_ROOT = PACKAGE_ROOT / "results" / "incidents" / "calibration_seed_20260930_smoke_02"
OFFICIAL_RESULTS_ROOT = PACKAGE_ROOT / "results" / OFFICIAL_RESULT_SLOT_NAME
OFFICIAL_LAUNCH_LOG_ROOT = PACKAGE_ROOT / "results" / OFFICIAL_LAUNCH_LOG_DIR_NAME
INCIDENT_ROOT = PACKAGE_ROOT / "results" / "incidents" / OFFICIAL_ID
CALIBRATION_SMOKE_01_INCIDENT = INCIDENT_ROOT / "calibration_smoke" / "PRE_SCIENTIFIC_OPERATIONAL_ABORT.json"
MD324_PATH = PACKAGE_ROOT.parents[2] / "MD" / "324.md"
MD325_PATH = PACKAGE_ROOT.parents[2] / "MD" / "325.md"
D3Q_SOURCE_SEAL = D3Q_ROOT / "SOURCE_SEAL.json"
D3Q_RESULT_ROOT = D3Q_ROOT / "results" / "omega_v2_2a_d3q_official_attempt01"


class PreScientificAbort(RuntimeError):
    """Operational stop before held-out scientific data is exposed."""


class PilotHardStop(RuntimeError):
    """Runtime failure that prevents the next registered cell."""


class LedgerPreSealHold(RuntimeError):
    """Frozen D512 FLOP constants differ from the V2-0 analytic ledger."""


@dataclass(frozen=True)
class SymbolicSeedPlan:
    mode: str
    master_seed: str
    weight_seed: str = "symbolic"
    input_seed: str = "symbolic"
    loss_w_seed: str = "symbolic"
    target_seed: str = "symbolic"
    d6_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "master_seed": self.master_seed,
            "weight_seed": self.weight_seed,
            "input_seed": self.input_seed,
            "loss_w_seed": self.loss_w_seed,
            "target_seed": self.target_seed,
            "symbolic_only": True,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def source_hashes() -> dict[str, str]:
    paths = [SPEC_PATH, PACKAGE_ROOT / ".gitattributes", PACKAGE_ROOT / "results" / ".gitattributes", *sorted(PACKAGE_ROOT.rglob("*.py"))]
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path) for path in paths if path.is_file()}


def write_calibration_source_snapshot() -> dict[str, Any]:
    if CALIBRATION_SOURCE_SNAPSHOT_PATH.exists():
        raise FileExistsError(f"calibration source snapshot is immutable: {CALIBRATION_SOURCE_SNAPSHOT_PATH}")
    qa = json.loads(QA_REPORT_PATH.read_text(encoding="utf-8"))
    current_sources = source_hashes()
    if qa.get("status") != "PASS" or qa.get("source_sha256") != current_sources:
        raise RuntimeError("V2_2B_CALIBRATION_SOURCE_SNAPSHOT_STOP: QA must PASS for current sources")
    snapshot = {
        "schema": "omega-v2-2b-calibration-source-snapshot-v1",
        "official_id": OFFICIAL_ID,
        "spec_sha256": sha256_file(SPEC_PATH),
        "source_sha256": current_sources,
        "executed_dependency_sha256": executed_dependency_hashes(),
    }
    write_json(CALIBRATION_SOURCE_SNAPSHOT_PATH, snapshot)
    return {**snapshot, "snapshot_sha256": sha256_file(CALIBRATION_SOURCE_SNAPSHOT_PATH)}


def verify_calibration_source_snapshot() -> dict[str, Any]:
    if not CALIBRATION_SOURCE_SNAPSHOT_PATH.is_file():
        raise RuntimeError("V2_2B_CALIBRATION_SOURCE_SNAPSHOT_MISSING")
    snapshot = json.loads(CALIBRATION_SOURCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    current_sources = source_hashes()
    current_dependencies = executed_dependency_hashes()
    current_spec = sha256_file(SPEC_PATH)
    if (
        snapshot.get("source_sha256") != current_sources
        or snapshot.get("executed_dependency_sha256") != current_dependencies
        or snapshot.get("spec_sha256") != current_spec
    ):
        raise RuntimeError("V2_2B_CALIBRATION_SOURCE_SNAPSHOT_STALE")
    return {
        "path": str(CALIBRATION_SOURCE_SNAPSHOT_PATH.resolve()),
        "snapshot_sha256": sha256_file(CALIBRATION_SOURCE_SNAPSHOT_PATH),
        "spec_sha256": current_spec,
        "source_sha256": current_sources,
        "executed_dependency_sha256": current_dependencies,
        "verified": True,
    }


def _module_name_for_path(path: Path, root: Path = PACKAGE_ROOT) -> tuple[str, str]:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    module_name = "omega_v2_2b_d512_local_pilot" + ("." + ".".join(parts) if parts else "")
    package_name = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    return module_name, package_name


def from_import_resolution_audit() -> dict[str, Any]:
    """Resolve every `from X import Y`, including deferred/local imports, via Python import semantics."""
    rows: list[dict[str, Any]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        _, package_name = _module_name_for_path(path)
        parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[int, Any] = {}
        for parent in ast.walk(parsed):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        for node in ast.walk(parsed):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported_module = node.module or ""
            if node.level:
                imported_module = importlib.util.resolve_name("." * node.level + imported_module, package_name)
            deferred_or_local = False
            current = node
            while id(current) in parents:
                current = parents[id(current)]
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    deferred_or_local = True
            for alias in node.names:
                if alias.name == "*":
                    rows.append({"file": path.relative_to(PACKAGE_ROOT).as_posix(), "line": node.lineno, "module": imported_module, "symbol": "*", "exists": True, "resolution": "wildcard-unexpanded", "deferred_or_local": deferred_or_local})
                    continue
                exists = False
                error = None
                try:
                    target_module = importlib.import_module(imported_module)
                    try:
                        getattr(target_module, alias.name)
                        exists = True
                    except AttributeError:
                        importlib.import_module(imported_module + "." + alias.name)
                        exists = True
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                rows.append({
                    "file": path.relative_to(PACKAGE_ROOT).as_posix(),
                    "line": node.lineno,
                    "module": imported_module,
                    "symbol": alias.name,
                    "exists": exists,
                    "error": error,
                    "deferred_or_local": deferred_or_local,
                })
    return {"pass": all(row["exists"] for row in rows), "checked_count": len(rows), "imports": rows, "missing": [row for row in rows if not row["exists"]]}
def package_import_sweep() -> dict[str, Any]:
    modules = []
    errors = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module_name, _ = _module_name_for_path(path)
        try:
            importlib.import_module(module_name)
            modules.append(module_name)
        except Exception as exc:
            errors.append({"module": module_name, "error": f"{type(exc).__name__}: {exc}"})
    return {"pass": not errors, "module_count": len(modules), "modules": modules, "errors": errors}


def reference_hashes() -> dict[str, str]:
    paths = [
        V20_ROOT / "omega_v2" / "core.py",
        V20_ROOT / "omega_v2" / "variants.py",
        V20_ROOT / "omega_v2" / "ledger.py",
        V20_ROOT / "V2_0_RESULT_SEAL.json",
        D3Q_ROOT / "OMEGA_V2_2A_D3Q_SPEC.md",
        D3Q_SOURCE_SEAL,
        D3Q_RESULT_ROOT / "d3q_metrics.json",
        D3Q_RESULT_ROOT / "OMEGA_V2_2A_D3Q_DIAGNOSTIC_REPORT.md",
        D3Q_RESULT_ROOT / "artifact_hashes.json",
        MD324_PATH,
        MD325_PATH,
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def executed_dependency_hashes() -> dict[str, str]:
    """Hash Python dependencies actually imported by smoke cells and the V2-0 counter."""
    paths = [
        *sorted((V20_ROOT / "omega_v2").rglob("*.py")),
        V22A_ROOT / "__init__.py",
        V22A_ROOT / "core.py",
        V22A_ROOT / "variants.py",
        V22A_ROOT / "checks.py",
        D3Q_ROOT / "__init__.py",
        D3Q_ROOT / "metrics.py",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def environment_build_record() -> dict[str, Any]:
    """CPU-safe framework metadata; deliberately does not query a CUDA device."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_device_query_performed": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
    }


def _nvidia_smi_record() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "query": None, "stderr": "nvidia-smi missing"}
    result = subprocess.run([executable, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
    return {"available": result.returncode == 0, "query": result.stdout.strip(), "stderr": result.stderr.strip()}


def environment_record(*, configure: bool) -> dict[str, Any]:
    if configure:
        configure_cuda_reference()
    available = bool(torch.cuda.is_available())
    device_name = capability = memory = None
    if available:
        props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
        memory = int(props.total_memory)
    return {
        **environment_build_record(),
        "cuda_available": available,
        "device_name": device_name,
        "compute_capability": capability,
        "device_total_memory_bytes": memory,
        "driver": _nvidia_smi_record(),
        "torch_num_threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "dtype": "torch.float32",
        "amp": False,
        "bf16": False,
        "fp16": False,
        "dropout": 0,
    }


def _environment_matches(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    return bool(
        record.get("torch_version") == expected.get("torch_version")
        and record.get("torch_cuda_runtime") == expected.get("torch_cuda_runtime")
        and record.get("cuda_available") is True
        and "GTX 1650 SUPER" in str(record.get("device_name"))
        and tuple(record.get("compute_capability") or ()) == tuple(expected.get("compute_capability") or (7, 5))
        and record.get("driver", {}).get("available") is True
        and record.get("driver", {}).get("query") == expected.get("nvidia_smi", {}).get("query")
        and record.get("cublas_workspace_config") == ":4096:8"
        and record.get("deterministic_algorithms") is True
        and record.get("torch_num_threads") == 1
        and record.get("cuda_autocast_enabled") is False
        and record.get("allow_tf32_matmul") is False
        and record.get("allow_tf32_cudnn") is False
        and record.get("cudnn_benchmark") is False
        and record.get("cudnn_deterministic") is True
        and record.get("amp") is False
        and record.get("bf16") is False
        and record.get("fp16") is False
        and record.get("dropout") == 0
    )


def full_environment_record_after_go() -> dict[str, Any]:
    reference = json.loads((V22A_ROOT / "SOURCE_SEAL_R1.json").read_text(encoding="utf-8"))["environment"]
    actual = environment_record(configure=True)
    if not _environment_matches(actual, reference):
        raise PreScientificAbort("V2-2B CUDA environment differs from sealed V2-2A reference")
    return actual


def unresolved_name_audit(package_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Find referenced global names absent from imports, assignments, and builtins."""
    root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    builtin_names = set(dir(builtins)) | {
        "__name__", "__file__", "__package__", "__spec__", "__loader__", "__cached__", "__builtins__",
    }
    issues = []
    for path in sorted(root.rglob("*.py")):
        table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        module_bound = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()}
        missing: set[str] = set()

        def visit(scope) -> None:
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in module_bound and symbol.get_name() not in builtin_names:
                    missing.add(symbol.get_name())
            for child in scope.get_children():
                visit(child)

        visit(table)
        if missing:
            issues.append({"path": path.relative_to(root).as_posix(), "unresolved": sorted(missing)})
    return issues


class PilotContext:
    def __init__(self, *, result_root: Path, official: bool, wall_start: float):
        self.result_root = result_root
        self.official = official
        self.wall_start = wall_start
        self.boundary_crossed = False
        self.cell_records: list[dict[str, Any]] = []
        self.capacity_issues: list[dict[str, Any]] = []
        self.wall_stop = False
        self.current_seed_state: dict[str, Any] | None = None

    def mark_boundary(self, *, master_seed: Any, cell_id: str) -> None:
        if self.boundary_crossed:
            return
        self.result_root.mkdir(parents=True, exist_ok=False)
        self.boundary_crossed = True
        marker = {
            "official_id": OFFICIAL_ID,
            "frontier": "first held-out CUDA numerical cell or earlier scientific-data exposure",
            "master_seed_or_symbolic_slot": master_seed,
            "cell_id": cell_id,
            "marked_before_numerical_cell": True,
            "symbolic_dry_run": isinstance(master_seed, str),
        }
        write_json(self.result_root / ("EXECUTION_BOUNDARY.json" if self.official else "QA_SMOKE_BOUNDARY.json"), marker)


def _append_capacity(context: PilotContext, reason: str, *, cell_id: str, master_seed: Any, **details: Any) -> None:
    context.capacity_issues.append({"reason": reason, "cell_id": cell_id, "master_seed": master_seed, **details})


def measure_cuda_cell(
    context: PilotContext,
    *,
    cell_id: str,
    master_seed: Any,
    gate: str,
    action: Callable[[Callable[[], None]], Any],
) -> tuple[Any | None, dict[str, Any]]:
    if context.wall_stop or time.perf_counter() - context.wall_start >= D8_WALL_LIMIT_SECONDS:
        context.wall_stop = True
        _append_capacity(context, "WALL_TIME", cell_id=cell_id, master_seed=master_seed)
        record = {"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": "NOT_RUN_WALL_TIME"}
        context.cell_records.append(record)
        return None, record
    if D8_GC_BEFORE_CELL:
        gc.collect()
    if D8_PRE_CELL_EMPTY_CACHE:
        torch.cuda.empty_cache()
    if D8_PRE_CELL_RESET_PEAK_STATS:
        torch.cuda.reset_peak_memory_stats()
    if D8_PRE_AND_POST_CELL_SYNCHRONIZE:
        torch.cuda.synchronize()
    started = time.perf_counter()

    def mark() -> None:
        context.mark_boundary(master_seed=master_seed, cell_id=cell_id)

    try:
        value = action(mark)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as error:
        if not context.boundary_crossed:
            raise PreScientificAbort(f"pre-frontier OOM in {cell_id}: {error}") from error
        record = {"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": "OOM_CAPACITY", "error": str(error), "wall_seconds": time.perf_counter() - started}
        context.cell_records.append(record)
        _append_capacity(context, "OOM", cell_id=cell_id, master_seed=master_seed)
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as recovery_error:
            raise PilotHardStop(f"unrecoverable CUDA OOM after {cell_id}: {recovery_error}") from recovery_error
        return None, record
    except (NonFinitePilotState, StructuralCorruption):
        context.cell_records.append({"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": "HARD_STOP", "wall_seconds": time.perf_counter() - started})
        raise
    except Exception as error:
        status = "PRE_FRONTIER_ABORT" if not context.boundary_crossed else "HARD_STOP"
        context.cell_records.append({"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": status, "error": str(error), "wall_seconds": time.perf_counter() - started})
        if not context.boundary_crossed:
            raise PreScientificAbort(f"pre-frontier failure in {cell_id}: {type(error).__name__}: {error}") from error
        raise PilotHardStop(f"cell {cell_id} hard-stop: {type(error).__name__}: {error}") from error

    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    elapsed = time.perf_counter() - started
    under_budget = allocated <= D8_VRAM_BUDGET_BYTES
    record = {
        "cell_id": cell_id,
        "master_seed": master_seed,
        "gate": gate,
        "peak_allocated": allocated,
        "peak_reserved": reserved,
        "wall_seconds": elapsed,
        "status": "PASS" if under_budget else "CAPACITY_ISSUE_VRAM_BUDGET",
    }
    context.cell_records.append(record)
    if not under_budget:
        _append_capacity(context, "VRAM_BUDGET", cell_id=cell_id, master_seed=master_seed, peak_allocated=allocated)
    if time.perf_counter() - context.wall_start >= D8_WALL_LIMIT_SECONDS:
        context.wall_stop = True
        _append_capacity(context, "WALL_TIME", cell_id=cell_id, master_seed=master_seed)
    return value, record


def pre_cuda_official_dry_run() -> dict[str, Any]:
    if OFFICIAL_RESULTS_ROOT.exists():
        raise FileExistsError(f"official result slot already exists: {OFFICIAL_RESULTS_ROOT}")
    return {
        "phase": "PRE_CUDA_OFFICIAL_DRY_RUN",
        "official_id": OFFICIAL_ID,
        "slot_created": False,
        "official_seed_plans_materialized": False,
        "cuda_device_query_performed": False,
        "cuda_kernels_launched": 0,
        "result_slot": str(OFFICIAL_RESULTS_ROOT.resolve()),
        "launch_log_dir": str(OFFICIAL_LAUNCH_LOG_ROOT.resolve()),
    }


def _serialize_tree_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _serialize_tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_tree_cpu(item) for item in value)
    return value


def _walk_tensors(value: Any, prefix: str = ""):
    if isinstance(value, Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            yield from _walk_tensors(item, name)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, f"{prefix}/{index}")


def _cell_id(gate: str, plan: Any, *, mode: str | None = None, k: int | None = None, variant: str | None = None) -> str:
    parts = [gate, str(plan.master_seed)]
    if mode:
        parts.append(mode)
    if variant:
        parts.append(variant)
    if k is not None:
        parts.append(f"K{k}")
    return "_".join(parts)


def _run_seed_gate_suite(
    plan: Any,
    context: PilotContext,
    *,
    smoke: bool,
    cell_impls: dict[str, Callable[..., Any]] | None = None,
    cell_measure: Callable[..., tuple[Any | None, dict[str, Any]]] = measure_cuda_cell,
) -> dict[str, Any]:
    implementations = cell_impls or {
        "D1": run_d1_cell,
        "D2": run_d2_cell,
        "D3": run_d3_cell,
        "D4": run_d4_cpu,
        "D6": run_d6_cell,
        "D7": run_d7_cell,
    }
    gates: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    seed_cell_records: list[dict[str, Any]] = []
    partial = {
        "master_seed": plan.master_seed,
        "seed_plan": plan.as_dict(),
        "gate_results": gates,
        "cell_records": seed_cell_records,
        "scientific_failures": failures,
        "tensor_bundle": tensors,
        "seed_pass": False,
        "terminal_status": "SEED_IN_PROGRESS",
    }
    context.current_seed_state = partial
    ledger = recompute_d512_ledger()
    if not ledger["exact_match"]:
        raise LedgerPreSealHold("FLOP_LEDGER_PRESEAL_HOLD")
    gates["D5"] = {"pass": True, "ledger": ledger, "cuda_cell": False}

    def run_cell(name: str, gate: str, action):
        try:
            value, record = cell_measure(context, cell_id=name, master_seed=plan.master_seed, gate=gate, action=action)
            if record not in seed_cell_records:
                seed_cell_records.append(record)
            return value, record
        except NonFinitePilotState:
            if context.cell_records and context.cell_records[-1].get("cell_id") == name and context.cell_records[-1] not in seed_cell_records:
                seed_cell_records.append(context.cell_records[-1])
            failures.append({"gate": f"{gate}_FINITENESS", "cell_id": name})
            raise
        except StructuralCorruption:
            if context.cell_records and context.cell_records[-1].get("cell_id") == name and context.cell_records[-1] not in seed_cell_records:
                seed_cell_records.append(context.cell_records[-1])
            failures.append({"gate": f"{gate}_STRUCTURE", "cell_id": name})
            raise
        except Exception:
            if context.cell_records and context.cell_records[-1].get("cell_id") == name and context.cell_records[-1] not in seed_cell_records:
                seed_cell_records.append(context.cell_records[-1])
            raise

    d1_rows: list[dict[str, Any]] = []
    gates["D1"] = {"cells": d1_rows, "pass": False}
    for k in (1, 4):
        name = _cell_id("D1", plan, k=k)
        value, d8 = run_cell(name, "D1", lambda mark, kk=k: implementations["D1"](plan, kk, mark))
        if value is None:
            d1_rows.append({"cell_id": name, "status": d8["status"], "pass": False, "D8": d8})
            continue
        tensors[name] = _serialize_tree_cpu(value.get("tensor_bundle", {}))
        row = {key: item for key, item in value.items() if key != "tensor_bundle"}
        row.update({"cell_id": name, "status": "PASS" if value.get("pass") else "FAIL", "D8": d8})
        d1_rows.append(row)
        if not value.get("pass"):
            failures.append({"gate": "D1", "cell_id": name})
    gates["D1"]["pass"] = len(d1_rows) == 2 and all(row.get("pass") for row in d1_rows)

    d2_name = _cell_id("D2", plan)
    d2, d2_d8 = run_cell(d2_name, "D2", lambda mark: implementations["D2"](plan, mark))
    if d2 is None:
        gates["D2"] = {"pass": False, "status": d2_d8["status"], "D8": d2_d8}
    else:
        tensors[d2_name] = _serialize_tree_cpu(d2.get("tensor_bundle", {}))
        gates["D2"] = {**{key: value for key, value in d2.items() if key != "tensor_bundle"}, "D8": d2_d8}
        if not d2.get("pass"):
            failures.append({"gate": "D2", "cell_id": d2_name})

    d3_name = _cell_id("D3", plan)
    d3, d3_d8 = run_cell(d3_name, "D3", lambda mark: implementations["D3"](plan, mark))
    if d3 is None:
        gates["D3"] = {"pass": False, "status": d3_d8["status"], "D8": d3_d8}
    else:
        tensors[d3_name] = _serialize_tree_cpu(d3.get("tensor_bundle", {}))
        gates["D3"] = {**{key: value for key, value in d3.items() if key != "tensor_bundle"}, "D8": d3_d8}
        if not d3.get("pass"):
            failures.append({"gate": "D3", "cell_id": d3_name})
        del d3

    try:
        d4 = implementations["D4"](plan)
    except StructuralCorruption:
        failures.append({"gate": "D4_STRUCTURE", "cell_id": f"D4_{plan.master_seed}"})
        gates["D4"] = {"pass": False, "status": "STRUCTURAL_CORRUPTION"}
        raise
    gates["D4"] = d4
    if not d4.get("pass"):
        failures.append({"gate": "D4", "cell_id": f"D4_{plan.master_seed}"})
        if not d4.get("storage", {}).get("pass") or not d4.get("state_dict_schema_conformant"):
            raise StructuralCorruption(f"D4 storage/schema corruption at {plan.master_seed}")

    d6_required = smoke or plan.master_seed == D6_MASTER_SEED or bool(getattr(plan, "d6_required", False))
    if d6_required:
        d6_rows: list[dict[str, Any]] = []
        gates["D6"] = {"applicable": True, "cells": d6_rows, "pass": False}
        for mode, ks in (("forward", D6_FORWARD_K_VALUES), ("backward", D6_BACKWARD_K_VALUES)):
            for k in ks:
                name = _cell_id("D6", plan, mode=mode, k=k)
                value, d8 = run_cell(name, "D6", lambda mark, mm=mode, kk=k: implementations["D6"](plan, mm, kk, mark))
                if value is None:
                    d6_rows.append({"cell_id": name, "status": d8["status"], "pass": False, "D8": d8})
                    continue
                if value.get("tensor_bundle"):
                    tensors[name] = _serialize_tree_cpu(value["tensor_bundle"])
                row = {key: item for key, item in value.items() if key not in ("tensor_bundle", "gate", "seed_plan")}
                row.update({"cell_id": name, "D8": d8})
                d6_rows.append(row)
                if not value.get("pass"):
                    failures.append({"gate": "D6", "cell_id": name})
        gates["D6"]["pass"] = len(d6_rows) == len(D6_FORWARD_K_VALUES) + len(D6_BACKWARD_K_VALUES) and all(row.get("pass") for row in d6_rows)
    else:
        gates["D6"] = {"applicable": False, "status": "NOT_APPLICABLE_MD324_SEED_SCOPE", "pass": True}

    d7_rows: list[dict[str, Any]] = []
    gates["D7"] = {"variants": d7_rows, "pass": False}
    for variant in ("R4", "U4"):
        name = _cell_id("D7", plan, variant=variant)
        value, d8 = run_cell(name, "D7", lambda mark, vv=variant: implementations["D7"](plan, vv, mark))
        if value is None:
            d7_rows.append({"cell_id": name, "variant": variant, "status": d8["status"], "pass": False, "D8": d8})
            continue
        row = {**value, "cell_id": name, "D8": d8}
        d7_rows.append(row)
        if not value.get("all_steps_finite"):
            failures.append({"gate": "D7_FINITE", "cell_id": name})
            raise NonFinitePilotState(f"D7 non-finite optimizer/model state at {name}")
        if not value.get("loss_reduction_pass"):
            failures.append({"gate": "D7_LOSS_REDUCTION", "cell_id": name})
    gates["D7"]["pass"] = len(d7_rows) == 2 and all(row.get("loss_reduction_pass") and row.get("all_steps_finite") for row in d7_rows)

    partial["seed_pass"] = not failures and all(gates.get(name, {}).get("pass", True) for name in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"))
    partial["terminal_status"] = "SEED_PASS" if partial["seed_pass"] else "SEED_FAIL"
    return partial


def _persist_seed_bundle(root: Path, seed_result: dict[str, Any]) -> dict[str, Any]:
    label = str(seed_result.get("master_seed", "unknown")).replace("/", "_").replace("\\", "_")
    path = root / f"seed_{label}_D1_D2_D3_D6_D7_bundles.pt"
    tensor_tree = _serialize_tree_cpu(seed_result.get("tensor_bundle", {}))
    torch.save({"schema": "omega-v2-2b-seed-bundle-v1", "official_id": OFFICIAL_ID, "seed_slot": label, "gates": tensor_tree}, path)
    hashes = {
        f"seed_{label}/{name}": tensor_raw_sha256(f"seed_{label}/{name}", tensor)
        for name, tensor in _walk_tensors(tensor_tree)
    }
    return {"path": path.name, "size_bytes": path.stat().st_size, "file_sha256": sha256_file(path), "tensor_raw_sha256": hashes}


def _terminal_decision(
    scientific_failures: list[dict[str, Any]],
    capacity_issues: list[dict[str, Any]],
    hard_stop: dict[str, Any] | None,
    *,
    all_seed_results_pass: bool = True,
) -> tuple[str, bool, str | None]:
    reasons = [item.get("reason") for item in capacity_issues if item.get("reason") in CAPACITY_REASONS]
    capacity_reason = next((reason for reason in CAPACITY_REASONS if reason in reasons), None)
    capacity_issue = capacity_reason is not None
    if scientific_failures:
        return TERMINAL_FAIL, capacity_issue, capacity_reason
    resource_hard_stop = bool(
        hard_stop
        and (
            hard_stop.get("kind") == "UNRECOVERABLE_OOM"
            or "OOM" in str(hard_stop.get("error", "")).upper()
        )
    )
    if hard_stop is not None and not resource_hard_stop:
        return TERMINAL_FAIL, capacity_issue, capacity_reason
    if capacity_issue:
        return TERMINAL_CAPACITY_HOLD, True, capacity_reason
    if hard_stop is not None or not all_seed_results_pass:
        return TERMINAL_FAIL, False, None
    return TERMINAL_PASS, False, None


def _render_pilot_report(result: dict[str, Any]) -> str:
    rows = [
        "# OMEGA-V2-2B d512 Local Contractual Pilot", "",
        f"- Official ID: `{OFFICIAL_ID}`",
        f"- terminal_status: `{result['terminal_status']}`",
        f"- capacity_issue: `{result.get('capacity_issue', False)}`",
        f"- capacity_reason: `{result.get('capacity_reason')}`",
        f"- official_wall_gate_seconds: `{result.get('official_wall_gate_seconds')}`", "",
        "## Seed results", "", "```json", json.dumps(result, sort_keys=True, indent=2), "```", "",
    ]
    return "\n".join(rows)


def _conformance_block(result: dict[str, Any]) -> str:
    logs = result.get("launch_logs", {})
    lines = [
        "# V2-2B Conformance Block", "", "```text",
        "V2_2A_attempt_00: HARNESS_ABORT_PRE_CUDA_SCIENCE / consumed=false / result=NONE",
        "V2_2A_r1: OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL / D3_GRADIENT_SHARING / elementwise_max_rel",
        "D3_DIAGNOSTIC: DIAGNOSTIC_COMPLETE / CALIBRATION_DIAGNOSTIC_ONLY / may_rescue_V2_2A=false",
        "D3Q: OMEGA_V2_2A_D3Q_PASS / INDEPENDENT_HELDOUT_VALIDATION / 5 of 5 held-out seeds",
        "D6_reporting_limitation: hashes_computed=true / per_cell_persistence=false / gate=PASS / rerun=false",
        f"V2_2B: {result['terminal_status']}",
        "RunPod: HOLD", "T3: HOLD", "CONFORMANCE_HOLD: unchanged",
        f"official_wall_gate_seconds: {result.get('official_wall_gate_seconds')}",
        f"launch_log_directory: {logs.get('directory')}",
        f"external_launch_log_hash_manifest: {logs.get('hash_manifest')}",
    ]
    for name, metadata in sorted(logs.get("files", {}).items()):
        lines.append(f"launch_log_{name}_sha256: {metadata.get('sha256')}")
    lines.extend(["```", ""])
    return "\n".join(lines)


def _persist_result_files(
    root: Path,
    result: dict[str, Any],
    *,
    official: bool,
    launch_root: Path | None = None,
    include_package_sources: bool = True,
) -> dict[str, Any]:
    result_name = "OFFICIAL_RESULT.json" if official else "mocked_result.json"
    report_name = "OMEGA_V2_2B_REPORT.md" if official else "mocked_report.md"
    report_hash_name = f"{report_name}.sha256"
    write_json(root / result_name, result)
    report_path = root / report_name
    report_path.write_text(_render_pilot_report(result), encoding="utf-8", newline="\n")
    (root / report_hash_name).write_text(sha256_file(report_path) + "\n", encoding="ascii", newline="\n")
    block_path = root / "OMEGA_V2_2B_CONFORMANCE_BLOCK.md"
    block_path.write_text(_conformance_block(result), encoding="utf-8", newline="\n")
    evidence = [*root.glob("*.pt"), *root.glob("*_PRIMARY_*.json"), root / result_name, report_path, root / report_hash_name, block_path]
    boundary = root / "EXECUTION_BOUNDARY.json"
    if boundary.is_file():
        evidence.append(boundary)
    if include_package_sources:
        evidence.extend([SPEC_PATH, PACKAGE_ROOT / ".gitattributes", PACKAGE_ROOT / "results" / ".gitattributes", *sorted(PACKAGE_ROOT.rglob("*.py"))])
        if SOURCE_SEAL_PATH.is_file():
            evidence.append(SOURCE_SEAL_PATH)
        if QA_REPORT_PATH.is_file():
            if official:
                qa_copy = root / "QA_REPORT.json"
                shutil.copy2(QA_REPORT_PATH, qa_copy)
                evidence.append(qa_copy)
            evidence.append(QA_REPORT_PATH)
    evidence = [path for path in evidence if path.is_file()]
    manifest = {
        "schema": "omega-v2-2b-artifact-hashes-v1",
        "official_id": OFFICIAL_ID,
        "terminal_status": result.get("terminal_status"),
        "artifacts": {str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in evidence},
    }
    if launch_root is not None:
        manifest["external_launch_logs"] = {
            "directory": str(launch_root.resolve()),
            "hash_manifest": "external_launch_logs_artifact_hashes.json",
            "hashed_after_runner_exit": True,
        }
    manifest_path = root / "artifact_hashes.json"
    write_json(manifest_path, manifest)
    verified = all(Path(name).is_file() and Path(name).stat().st_size == row["size_bytes"] and sha256_file(name) == row["sha256"] for name, row in manifest["artifacts"].items())
    verified = verified and sha256_file(report_path) == (root / report_hash_name).read_text(encoding="ascii").strip()
    write_json(root / "artifact_hashes_verified.json", {"verified": bool(verified), "artifact_count": len(evidence)})
    return {"verified": bool(verified), "artifact_count": len(evidence), "manifest_sha256": sha256_file(manifest_path)}


def _persist_primary_evidence(root: Path, prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    metrics_path = root / f"{prefix}_PRIMARY_METRICS.json"
    write_json(metrics_path, metrics)
    evidence = [metrics_path, *sorted(root.glob("seed_*.pt")), *sorted(root.glob("calibration_seed_*.pt"))]
    manifest_path = root / f"{prefix}_PRIMARY_ARTIFACT_HASHES.json"
    manifest = {
        "schema": "omega-v2-2b-primary-artifact-hashes-v1",
        "artifacts": {str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in evidence},
    }
    write_json(manifest_path, manifest)
    verified = all(Path(name).is_file() and Path(name).stat().st_size == item["size_bytes"] and sha256_file(name) == item["sha256"] for name, item in manifest["artifacts"].items())
    verification_path = root / f"{prefix}_PRIMARY_ARTIFACTS_VERIFIED.json"
    write_json(verification_path, {"verified": bool(verified), "artifact_count": len(evidence)})
    return {
        "verified": bool(verified),
        "artifact_count": len(evidence),
        "metrics_path": metrics_path.name,
        "manifest_path": manifest_path.name,
        "verification_path": verification_path.name,
    }


def _execute_seed_plans(
    plans: list[Any],
    *,
    result_root: Path,
    official: bool,
    wall_start: float,
    smoke: bool = False,
    cell_impls: dict[str, Callable[..., Any]] | None = None,
    cell_measure: Callable[..., tuple[Any | None, dict[str, Any]]] = measure_cuda_cell,
) -> tuple[list[dict[str, Any]], PilotContext, dict[str, Any] | None]:
    context = PilotContext(result_root=result_root, official=official, wall_start=wall_start)
    completed: set[Any] = set()
    seed_results: list[dict[str, Any]] = []
    hard_stop = None
    for plan in plans:
        if context.wall_stop:
            break
        context.current_seed_state = None
        try:
            seed_result = _run_seed_gate_suite(plan, context, smoke=smoke, cell_impls=cell_impls, cell_measure=cell_measure)
            if context.boundary_crossed:
                seed_result["bundle"] = _persist_seed_bundle(result_root, seed_result)
            seed_results.append(seed_result)
            completed.add(plan.master_seed)
        except PreScientificAbort as error:
            if not context.boundary_crossed:
                return seed_results, context, {"kind": "PRE_SCIENTIFIC_ABORT", "error": str(error), "pre_frontier": True}
            hard_stop = {"master_seed": plan.master_seed, "kind": "PRE_SCIENTIFIC_ABORT_AFTER_FRONTIER", "error": str(error)}
            partial = context.current_seed_state or {"master_seed": plan.master_seed, "seed_plan": plan.as_dict(), "gate_results": {}, "cell_records": context.cell_records, "scientific_failures": [], "tensor_bundle": {}, "seed_pass": False}
            partial["hard_stop"] = hard_stop
            partial["terminal_status"] = "SEED_HARD_STOP"
            partial["bundle"] = _persist_seed_bundle(result_root, partial)
            seed_results.append(partial)
            break
        except torch.cuda.OutOfMemoryError as error:
            if not context.boundary_crossed:
                return seed_results, context, {"kind": "PRE_SCIENTIFIC_ABORT", "error": str(error), "pre_frontier": True}
            _append_capacity(context, "OOM", cell_id="unknown", master_seed=plan.master_seed)
            hard_stop = {"master_seed": plan.master_seed, "kind": "UNRECOVERABLE_OOM", "error": str(error)}
            partial = context.current_seed_state or {"master_seed": plan.master_seed, "seed_plan": plan.as_dict(), "gate_results": {}, "cell_records": context.cell_records, "scientific_failures": [], "tensor_bundle": {}, "seed_pass": False}
            partial["hard_stop"] = hard_stop
            partial["terminal_status"] = "SEED_HARD_STOP"
            partial["bundle"] = _persist_seed_bundle(result_root, partial)
            seed_results.append(partial)
            break
        except Exception as error:
            if not context.boundary_crossed:
                return seed_results, context, {"kind": "PRE_SCIENTIFIC_ABORT", "error": f"{type(error).__name__}: {error}", "pre_frontier": True}
            hard_stop = {"master_seed": plan.master_seed, "kind": type(error).__name__, "error": str(error)}
            partial = context.current_seed_state or {"master_seed": plan.master_seed, "seed_plan": plan.as_dict(), "gate_results": {}, "cell_records": context.cell_records, "scientific_failures": [], "tensor_bundle": {}, "seed_pass": False}
            partial["hard_stop"] = hard_stop
            partial["terminal_status"] = "SEED_HARD_STOP"
            partial["bundle"] = _persist_seed_bundle(result_root, partial)
            seed_results.append(partial)
            break
    for plan in plans:
        if plan.master_seed not in completed and not any(row.get("master_seed") == plan.master_seed for row in seed_results):
            seed_results.append({
                "master_seed": plan.master_seed,
                "seed_plan": plan.as_dict(),
                "terminal_status": "NOT_RUN_AFTER_HARD_STOP" if hard_stop else "NOT_RUN_WALL_TIME",
                "scientific_failures": [],
                "gate_results": {},
                "cell_records": [],
                "tensor_bundle": {},
                "seed_pass": False,
            })
    return seed_results, context, hard_stop


def _make_result(seed_results: list[dict[str, Any]], context: PilotContext, ledger: dict[str, Any], hard_stop: dict[str, Any] | None, plans: list[Any]) -> dict[str, Any]:
    failures = [failure for seed in seed_results for failure in seed.get("scientific_failures", [])]
    all_pass = len(seed_results) == len(plans) and all(seed.get("seed_pass") is True for seed in seed_results)
    terminal, capacity, reason = _terminal_decision(failures, context.capacity_issues, hard_stop, all_seed_results_pass=all_pass)
    persisted_seed_results = [{key: value for key, value in seed.items() if key != "tensor_bundle"} for seed in seed_results]
    return {
        "schema": "omega-v2-2b-official-result-v1" if context.official else "omega-v2-2b-mocked-result-v1",
        "official_id": OFFICIAL_ID,
        "classification": "LOCAL_CONTRACTUAL_PILOT" if context.official else "MOCKED_CONTROL_FLOW_ONLY",
        "terminal_status": terminal,
        "seed_results": persisted_seed_results,
        "planned_seeds": [plan.as_dict() for plan in plans],
        "completed_seed_count": sum(seed.get("terminal_status") == "SEED_PASS" or seed.get("terminal_status") == "SEED_FAIL" for seed in seed_results),
        "scientific_failures": failures,
        "capacity_issue": capacity,
        "capacity_reason": reason,
        "capacity_events": context.capacity_issues,
        "cell_records": context.cell_records,
        "hard_stop": hard_stop,
        "D5_ledger": ledger,
        "per_seed_retries": 0,
        "majority_or_average_used": False,
    }


def _finalize_wall_in_process(result: dict[str, Any], context: PilotContext, root: Path, *, launch_root: Path | None = None) -> dict[str, Any]:
    """Verify primary scientific evidence, freeze Q8 gate time, then seal admin artifacts once."""
    primary = _persist_primary_evidence(root, "OFFICIAL", result)
    if not primary["verified"]:
        raise RuntimeError("OFFICIAL_PRIMARY_ARTIFACT_HASH_VERIFICATION_FAILED")

    gate_end = time.perf_counter()
    gate_seconds = gate_end - context.wall_start
    if gate_seconds > D8_WALL_LIMIT_SECONDS and not any(item.get("reason") == "WALL_TIME" for item in context.capacity_issues):
        _append_capacity(context, "WALL_TIME", cell_id="official_primary_verification", master_seed="all")
        all_seed_pass = len(result.get("seed_results", [])) == len(result.get("planned_seeds", [])) and all(
            seed.get("seed_pass") is True for seed in result.get("seed_results", [])
        )
        terminal, capacity_issue, capacity_reason = _terminal_decision(
            result.get("scientific_failures", []),
            context.capacity_issues,
            result.get("hard_stop"),
            all_seed_results_pass=all_seed_pass,
        )
        result.update({
            "terminal_status": terminal,
            "capacity_issue": capacity_issue,
            "capacity_reason": capacity_reason,
            "capacity_events": context.capacity_issues,
        })
        primary = _persist_primary_evidence(root, "OFFICIAL", result)
        if not primary["verified"]:
            raise RuntimeError("OFFICIAL_PRIMARY_ARTIFACT_HASH_VERIFICATION_FAILED_AFTER_WALL_HOLD")
        gate_end = time.perf_counter()
        gate_seconds = gate_end - context.wall_start

    result.update({
        "official_wall_start": context.wall_start,
        "official_wall_gate_end": gate_end,
        "official_wall_start_perf_counter": context.wall_start,
        "official_wall_gate_end_perf_counter": gate_end,
        "official_wall_gate_seconds": gate_seconds,
        "official_wall_gate_limit_seconds": D8_WALL_LIMIT_SECONDS,
        "primary_evidence": primary,
        "launch_logs": {
            "directory": str(launch_root.resolve()) if launch_root is not None else None,
            "hash_manifest": "external_launch_logs_artifact_hashes.json" if launch_root is not None else None,
            "finalized": False,
        },
    })
    final_hashes = _persist_result_files(root, result, official=context.official, launch_root=launch_root)
    if not final_hashes["verified"]:
        raise RuntimeError("OFFICIAL_FINAL_ARTIFACT_HASH_VERIFICATION_FAILED")
    return result


def _pre_scientific_abort(error: BaseException, *, wall_start: float, ledger: dict[str, Any] | None = None, smoke: bool = False) -> dict[str, Any]:
    root = INCIDENT_ROOT if not smoke else CALIBRATION_SMOKE_INCIDENT_ROOT
    record = {
        "schema": "omega-v2-2b-pre-scientific-abort-v1",
        "official_id": OFFICIAL_ID,
        "terminal_status": "PRE_SCIENTIFIC_OPERATIONAL_ABORT",
        "consumed": False,
        "heldout_cuda_cells_started": 0,
        "heldout_results_persisted": 0,
        "error_type": type(error).__name__,
        "error": str(error),
        "duration_seconds": time.perf_counter() - wall_start,
        "ledger_preflight": ledger,
    }
    write_json(root / "PRE_SCIENTIFIC_OPERATIONAL_ABORT.json", record)
    return record


def full_mocked_control_flow_dry_run(output_root: str | Path | None = None) -> dict[str, Any]:
    """Run the real seed suite/orchestrator with cell and D8 measurement stubs."""
    temp = tempfile.TemporaryDirectory(prefix="omega_v2b_real_mock_") if output_root is None else None
    base = Path(temp.name) if temp else Path(output_root)
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"mock dry-run root is not empty: {base}")
    base.mkdir(parents=True, exist_ok=True)
    scenarios = ("PASS", "SCIENTIFIC_FAIL", "VRAM_BUDGET", "OOM", "WALL_TIME", "HARD_STOP", "UNRECOVERABLE_OOM")
    expected = {
        "PASS": TERMINAL_PASS,
        "SCIENTIFIC_FAIL": TERMINAL_FAIL,
        "VRAM_BUDGET": TERMINAL_CAPACITY_HOLD,
        "OOM": TERMINAL_CAPACITY_HOLD,
        "WALL_TIME": TERMINAL_CAPACITY_HOLD,
        "HARD_STOP": TERMINAL_FAIL,
        "UNRECOVERABLE_OOM": TERMINAL_CAPACITY_HOLD,
    }
    cases = {}
    for scenario in scenarios:
        root = base / scenario.lower()
        plans = [SymbolicSeedPlan("mock", f"symbolic_slot_{index:02d}", d6_required=index == 1) for index in range(1, 4)]
        context = PilotContext(result_root=root, official=True, wall_start=time.perf_counter())
        counter = {"cells": 0}

        def d1(plan, k, mark):
            mark()
            return {"pass": True, "tensor_bundle": {"output": torch.tensor([float(k)])}}

        def d2(plan, mark):
            mark()
            return {"pass": True, "structural": {"round_trace_equalities": [True] * 4}, "tensor_bundle": {}}

        def d3(plan, mark):
            mark()
            failed = scenario == "SCIENTIFIC_FAIL" and plan.master_seed == "symbolic_slot_01"
            return {"pass": not failed, "families": {"W_Q": {"pass": not failed}}, "tensor_bundle": {"W_Q": {"gR_fp32": torch.tensor([1.0])}}}

        def d4(plan):
            return {"pass": True, "storage": {"pass": True}, "state_dict_schema_conformant": True}

        def d6(plan, mode, k, mark):
            mark()
            return {"pass": True, "mode": mode, "K": k, "equality": {"pass": True}, "tensor_bundle": {}}

        def d7(plan, variant, mark):
            mark()
            return {"loss_reduction_pass": True, "all_steps_finite": True, "variant": variant}

        impls = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D6": d6, "D7": d7}

        def mock_measure(ctx, *, cell_id, master_seed, gate, action):
            counter["cells"] += 1
            if scenario == "WALL_TIME" and counter["cells"] > 2:
                ctx.wall_stop = True
                _append_capacity(ctx, "WALL_TIME", cell_id=cell_id, master_seed=master_seed)
                row = {"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": "NOT_RUN_WALL_TIME"}
                ctx.cell_records.append(row)
                return None, row
            if scenario in ("OOM", "UNRECOVERABLE_OOM") and gate == "D7" and cell_id.endswith("R4"):
                _append_capacity(ctx, "OOM", cell_id=cell_id, master_seed=master_seed)
                row = {"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": "OOM_CAPACITY"}
                ctx.cell_records.append(row)
                if scenario == "UNRECOVERABLE_OOM":
                    raise PilotHardStop("symbolic unrecoverable OOM")
                return None, row
            try:
                value = action(lambda: ctx.mark_boundary(master_seed=master_seed, cell_id=cell_id))
            except Exception:
                row = {"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "status": "HARD_STOP"}
                ctx.cell_records.append(row)
                raise
            row = {"cell_id": cell_id, "master_seed": master_seed, "gate": gate, "peak_allocated": 0, "peak_reserved": 0, "wall_seconds": 0.001, "status": "PASS"}
            if scenario == "VRAM_BUDGET" and gate == "D3":
                row.update({"peak_allocated": D8_VRAM_BUDGET_BYTES + 1, "status": "CAPACITY_ISSUE_VRAM_BUDGET"})
                _append_capacity(ctx, "VRAM_BUDGET", cell_id=cell_id, master_seed=master_seed, peak_allocated=row["peak_allocated"])
            ctx.cell_records.append(row)
            return value, row

        def hard_stop_d2(plan, mark):
            mark()
            if scenario == "HARD_STOP" and plan.master_seed == "symbolic_slot_01":
                raise PilotHardStop("symbolic runtime crash")
            return d2(plan, lambda: None)

        if scenario == "HARD_STOP":
            impls["D2"] = hard_stop_d2
        seed_results, context, hard_stop = _execute_seed_plans(
            plans,
            result_root=root,
            official=True,
            wall_start=context.wall_start,
            smoke=False,
            cell_impls=impls,
            cell_measure=mock_measure,
        )
        result = _make_result(seed_results, context, recompute_d512_ledger(), hard_stop, plans)
        hashes = _persist_result_files(root, result, official=True, include_package_sources=False)
        result["artifact_seal"] = hashes
        _persist_result_files(root, result, official=True, include_package_sources=False)
        verified = json.loads((root / "artifact_hashes_verified.json").read_text(encoding="utf-8"))["verified"]
        passed = result["terminal_status"] == expected[scenario] and verified
        cases[scenario] = {
            "pass": passed,
            "terminal_status": result["terminal_status"],
            "expected_terminal_status": expected[scenario],
            "seed_count": len(seed_results),
            "not_run_seed_count": sum(str(seed.get("terminal_status", "")).startswith("NOT_RUN") for seed in seed_results),
            "partial_seed_persisted": any(seed.get("hard_stop") and seed.get("bundle") for seed in seed_results),
            "capacity_events": context.capacity_issues,
            "hard_stop": hard_stop,
            "artifact_hashes_verified": verified,
            "cuda_kernels_launched": 0,
            "seed_values_materialized": False,
        }
    if temp is not None:
        temp.cleanup()
    return {
        "schema": "omega-v2-2b-real-mocked-control-flow-v1",
        "classification": "MOCKED_CONTROL_FLOW_ONLY",
        "official_id": OFFICIAL_ID,
        "cases": cases,
        "artifact_hashes_verified": all(case["artifact_hashes_verified"] for case in cases.values()),
        "all_scenarios_pass": all(case["pass"] for case in cases.values()),
        "seed_values_materialized": False,
        "cuda_kernels_launched": 0,
    }


def create_source_seal() -> dict[str, Any]:
    if SOURCE_SEAL_PATH.exists():
        raise FileExistsError(f"source seal is immutable: {SOURCE_SEAL_PATH}")
    smoke_path = CALIBRATION_SMOKE_ROOT / "CALIBRATION_SMOKE_RESULT.json"
    smoke_verification = CALIBRATION_SMOKE_ROOT / "artifact_hashes_verified.json"
    if not smoke_path.is_file() or not smoke_verification.is_file():
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: verified smoke_02 required")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    verified = json.loads(smoke_verification.read_text(encoding="utf-8"))
    qa = json.loads(QA_REPORT_PATH.read_text(encoding="utf-8"))
    snapshot = verify_calibration_source_snapshot()
    external_logs_path = CALIBRATION_SMOKE_ROOT / "external_launch_logs_artifact_hashes.json"
    if not external_logs_path.is_file():
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: smoke_02 external log hash manifest missing")
    external_logs = json.loads(external_logs_path.read_text(encoding="utf-8"))
    if verified.get("verified") is not True or smoke.get("terminal_status") != TERMINAL_CALIBRATION_SMOKE_COMPLETE or smoke.get("V2_2B_verdict") is not None:
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: smoke_02 must be COMPLETE, verified, and verdict-null")
    if external_logs.get("verified") is not True:
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: smoke_02 external launch logs not verified")
    current = source_hashes()
    if (
        smoke.get("source_sha256") != current
        or smoke.get("spec_sha256") != sha256_file(SPEC_PATH)
        or smoke.get("calibration_source_snapshot_sha256") != snapshot["snapshot_sha256"]
        or smoke.get("executed_dependency_sha256") != executed_dependency_hashes()
        or qa.get("source_sha256") != current
        or qa.get("executed_dependency_sha256") != executed_dependency_hashes()
        or qa.get("status") != "PASS"
    ):
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: QA/source snapshot/smoke hashes stale or failed")
    ledger = recompute_d512_ledger()
    if not ledger["exact_match"]:
        raise LedgerPreSealHold("FLOP_LEDGER_PRESEAL_HOLD")
    reference = json.loads((V22A_ROOT / "SOURCE_SEAL_R1.json").read_text(encoding="utf-8"))["environment"]
    if not _environment_matches(smoke.get("environment", {}), reference):
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: smoke_02 environment mismatch")

    smoke01_result = CALIBRATION_SMOKE_01_ROOT / "CALIBRATION_SMOKE_RESULT.json"
    smoke01_report = CALIBRATION_SMOKE_01_ROOT / "CALIBRATION_SMOKE_REPORT.md"
    smoke01_manifest = CALIBRATION_SMOKE_01_ROOT / "artifact_hashes.json"
    smoke01_verified = CALIBRATION_SMOKE_01_ROOT / "artifact_hashes_verified.json"
    if not all(path.is_file() for path in (smoke01_result, smoke01_report, smoke01_manifest, smoke01_verified)):
        raise RuntimeError("V2_2B_SOURCE_SEAL_STOP: immutable smoke_01 evidence incomplete")
    smoke01_record = json.loads(smoke01_result.read_text(encoding="utf-8"))
    smoke01_refs = {
        "incident_path": str(CALIBRATION_SMOKE_01_INCIDENT.resolve()),
        "incident_sha256": sha256_file(CALIBRATION_SMOKE_01_INCIDENT) if CALIBRATION_SMOKE_01_INCIDENT.is_file() else None,
        "result_sha256": sha256_file(smoke01_result),
        "report_sha256": sha256_file(smoke01_report),
        "artifact_manifest_sha256": sha256_file(smoke01_manifest),
        "artifact_verification_sha256": sha256_file(smoke01_verified),
        "terminal_status": smoke01_record.get("terminal_status"),
        "official_attempt_consumed": False,
    }
    smoke02_report = CALIBRATION_SMOKE_ROOT / "CALIBRATION_SMOKE_REPORT.md"
    smoke02_manifest = CALIBRATION_SMOKE_ROOT / "artifact_hashes.json"
    seal = {
        "schema": "omega-v2-2b-source-seal-v1",
        "official_id": OFFICIAL_ID,
        "spec_sha256": sha256_file(SPEC_PATH),
        "source_sha256": current,
        "executed_dependency_sha256": executed_dependency_hashes(),
        "calibration_source_snapshot_sha256": snapshot["snapshot_sha256"],
        "smoke_01_references": smoke01_refs,
        "smoke_02_references": {
            "result_sha256": sha256_file(smoke_path),
            "report_sha256": sha256_file(smoke02_report),
            "artifact_manifest_sha256": sha256_file(smoke02_manifest),
            "artifact_verification_sha256": sha256_file(smoke_verification),
            "external_launch_logs_manifest_sha256": sha256_file(external_logs_path),
            "terminal_status": smoke["terminal_status"],
        },
        "v2_0_reference_sha256": {str(path.resolve()): sha256_file(path) for path in (V20_ROOT / "omega_v2" / "core.py", V20_ROOT / "omega_v2" / "variants.py", V20_ROOT / "omega_v2" / "ledger.py")},
        "d3q_and_spec_reference_sha256": reference_hashes(),
        "environment_identity": {key: smoke["environment"].get(key) for key in ("python_version", "torch_version", "torch_cuda_runtime", "driver", "device_name", "compute_capability", "cublas_workspace_config")},
        "environment": smoke["environment"],
        "all_seeds": {"calibration_only": CALIBRATION_SEED, "official": list(OFFICIAL_MASTER_SEEDS), "D6_seed": D6_MASTER_SEED},
        "thresholds": {
            "D1_elementwise_abs": D1_ELEMENTWISE_ABS_TOL,
            "D1_elementwise_rel": D1_ELEMENTWISE_REL_TOL,
            "D1_elementwise_floor": D1_ELEMENTWISE_FLOOR,
            "D1_normwise_E_L2": D1_NORMWISE_E_L2_LIMIT,
            "D1_normwise_floor": D1_NORMWISE_FLOOR,
            "D3_E_L2": D3_E_L2_LIMIT,
            "D3_E_inf": D3_E_INF_LIMIT,
            "D3_max_abs_ulps": D3_MAX_ABS_ULP_LIMIT,
            "D7_L20_over_L0": D7_FINAL_LOSS_RATIO,
            "D8_allocated_bytes": D8_VRAM_BUDGET_BYTES,
            "D8_wall_seconds": D8_WALL_LIMIT_SECONDS,
        },
        "batch_regimes": {"conformance": CONFORMANCE_BATCH, "trainability": TRAINABILITY_BATCH},
        "flop_ledger": ledger,
        "artifact_contract": ARTIFACT_CONTRACT,
        "open_questions_accepted_by_MD325": {f"Q{number}": "ACCEPTED" for number in range(1, 10)},
        "pending_limits_changed": False,
        "qa_report_sha256": sha256_file(QA_REPORT_PATH),
        "calibration_smoke_sha256": sha256_file(smoke_path),
        "official_started": False,
    }
    write_json(SOURCE_SEAL_PATH, seal)
    return seal


def verify_source_seal() -> dict[str, Any]:
    seal = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
    if seal.get("spec_sha256") != sha256_file(SPEC_PATH) or seal.get("source_sha256") != source_hashes():
        raise RuntimeError("V2_2B_SOURCE_SEAL_MISMATCH")
    snapshot = verify_calibration_source_snapshot()
    if seal.get("calibration_source_snapshot_sha256") != snapshot["snapshot_sha256"]:
        raise RuntimeError("V2_2B_CALIBRATION_SOURCE_SNAPSHOT_SEAL_MISMATCH")
    if seal.get("executed_dependency_sha256") != executed_dependency_hashes():
        raise RuntimeError("V2_2B_EXECUTED_DEPENDENCY_SEAL_MISMATCH")
    if seal.get("d3q_and_spec_reference_sha256") != reference_hashes():
        raise RuntimeError("V2_2B_REFERENCE_SEAL_MISMATCH")
    if not recompute_d512_ledger()["exact_match"]:
        raise LedgerPreSealHold("FLOP_LEDGER_PRESEAL_HOLD")
    return seal


def run_qa_calibration() -> dict[str, Any]:
    ledger = recompute_d512_ledger()
    cpu_qa = run_calibration_qa_cpu(CALIBRATION_SEED)
    static = unresolved_name_audit()
    compile_pass = compileall.compile_dir(str(PACKAGE_ROOT), quiet=1, force=True)
    import_audit = from_import_resolution_audit()
    import_sweep = package_import_sweep()
    pre_cuda = pre_cuda_official_dry_run()
    mocked = full_mocked_control_flow_dry_run()
    build = environment_build_record()
    reference = json.loads((V22A_ROOT / "SOURCE_SEAL_R1.json").read_text(encoding="utf-8"))["environment"]
    build_match = build["torch_version"] == reference.get("torch_version") and build["torch_cuda_runtime"] == reference.get("torch_cuda_runtime")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(PACKAGE_ROOT / "tests"), "-v"],
        cwd=str(CAMPAIGN_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    cpu_checks_pass = bool(
        ledger["exact_match"] and cpu_qa["pass"] and not static
        and pre_cuda["slot_created"] is False
        and compile_pass and import_audit["pass"] and import_sweep["pass"]
        and mocked["all_scenarios_pass"] and mocked["artifact_hashes_verified"]
        and tests.returncode == 0
    )
    passed = cpu_checks_pass and build_match
    report = {
        "schema": "omega-v2-2b-qa-report-v1",
        "official_id": OFFICIAL_ID,
        "status": "PASS" if passed else ("ENVIRONMENT_HOLD" if cpu_checks_pass else "FAIL"),
        "cpu_qa_status": "PASS" if cpu_checks_pass else "FAIL",
        "classification": "CALIBRATION_QA_ONLY",
        "V2_2B_verdict": None,
        "heldout_seed_data_touched": False,
        "cuda_device_query_performed": False,
        "cuda_kernels_launched": 0,
        "unit_tests": {"returncode": tests.returncode, "passed": tests.returncode == 0, "stdout": tests.stdout, "stderr": tests.stderr},
        "static_unresolved_names": len(static),
        "static_issues": static,
        "from_import_resolution_audit": import_audit,
        "package_compile_sweep": {"pass": bool(compile_pass)},
        "package_import_sweep": import_sweep,
        "cpu_calibration_qa": cpu_qa,
        "pre_cuda_official_dry_run": pre_cuda,
        "full_mocked_control_flow_dry_run": mocked,
        "environment_build_validation": {"record": build, "matches_V2_2A_reference": build_match},
        "flop_ledger_crosscheck": ledger,
        "source_sha256": source_hashes(),
        "executed_dependency_sha256": executed_dependency_hashes(),
        "spec_sha256": sha256_file(SPEC_PATH),
        "official_seed_plans_materialized": False,
    }
    write_json(QA_REPORT_PATH, report)
    return report


def _render_smoke_report(smoke: dict[str, Any]) -> str:
    return "\n".join([
        "# OMEGA-V2-2B calibration-seed harness smoke", "",
        "- classification: `CALIBRATION_QA_ONLY`",
        "- V2_2B_verdict: `null`",
        f"- terminal_status: `{smoke['terminal_status']}`",
        f"- master_seed: `{smoke['seed_result']['seed_plan']['master_seed']}`",
        f"- structural gates: `{smoke['structural_gates_pass']}`",
        f"- D3 A/B/C gates: `{smoke['D3_A_B_C_pass']}`",
        f"- external launch log hash manifest: `{smoke.get('launch_logs', {}).get('hash_manifest')}`",
        f"- wall_gate_seconds: `{smoke.get('wall_gate_seconds')}`",
        "- official held-out seeds used: `False`",
        "- scientific verdict: `NONE`", "", "```json",
        json.dumps(smoke["seed_result"]["gate_results"], sort_keys=True, indent=2), "```", "",
    ])


def _persist_smoke_artifacts(smoke: dict[str, Any]) -> dict[str, Any]:
    root = CALIBRATION_SMOKE_ROOT
    result_path = root / "CALIBRATION_SMOKE_RESULT.json"
    report_path = root / "CALIBRATION_SMOKE_REPORT.md"
    sidecar_path = root / "CALIBRATION_SMOKE_REPORT.md.sha256"
    write_json(result_path, smoke)
    report_path.write_text(_render_smoke_report(smoke), encoding="utf-8", newline="\n")
    sidecar_path.write_text(sha256_file(report_path) + "\n", encoding="ascii", newline="\n")
    evidence = [
        result_path,
        report_path,
        sidecar_path,
        *root.glob("*.pt"),
        *root.glob("*_PRIMARY_*.json"),
        root / "QA_SMOKE_BOUNDARY.json",
        SPEC_PATH,
        CALIBRATION_SOURCE_SNAPSHOT_PATH,
        PACKAGE_ROOT / ".gitattributes",
        PACKAGE_ROOT / "results" / ".gitattributes",
        *sorted(PACKAGE_ROOT.rglob("*.py")),
    ]
    evidence = [path for path in evidence if path.is_file()]
    manifest = {
        "schema": "omega-v2-2b-calibration-smoke-artifact-hashes-v1",
        "artifacts": {str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in evidence},
        "external_launch_logs": {
            "directory": str(CALIBRATION_SMOKE_LOG_ROOT.resolve()),
            "hash_manifest": "external_launch_logs_artifact_hashes.json",
            "hashed_after_process_exit": True,
        },
    }
    manifest_path = root / "artifact_hashes.json"
    write_json(manifest_path, manifest)
    verified = all(Path(name).is_file() and Path(name).stat().st_size == row["size_bytes"] and sha256_file(name) == row["sha256"] for name, row in manifest["artifacts"].items())
    write_json(root / "artifact_hashes_verified.json", {"verified": bool(verified), "artifact_count": len(evidence)})
    return {"verified": bool(verified), "artifact_count": len(evidence), "manifest_sha256": sha256_file(manifest_path)}


def _calibration_smoke_terminal(
    seed: dict[str, Any],
    context: PilotContext,
    hard_stop: dict[str, Any] | None,
    *,
    required_d8_cells: int,
    wall_gate_seconds: float | None = None,
) -> str:
    gates = seed.get("gate_results", {})
    failures = seed.get("scientific_failures", [])
    executed_structural_failure = any(
        gates.get(name, {}).get("pass") is False
        and gates.get(name, {}).get("status") not in ("NOT_RUN_WALL_TIME", "NOT_RUN_CAPACITY", "OOM_CAPACITY")
        for name in ("D2", "D4")
    )
    if failures or executed_structural_failure:
        return TERMINAL_CALIBRATION_QA_SCIENTIFIC_HOLD
    resource_hard_stop = bool(
        hard_stop
        and (
            hard_stop.get("kind") == "UNRECOVERABLE_OOM"
            or "OOM" in str(hard_stop.get("error", "")).upper()
        )
    )
    if hard_stop is not None and not resource_hard_stop:
        return TERMINAL_CALIBRATION_QA_HARNESS_HOLD
    if context.capacity_issues or (wall_gate_seconds is not None and wall_gate_seconds > D8_WALL_LIMIT_SECONDS):
        return TERMINAL_CALIBRATION_QA_CAPACITY_HOLD
    gates_pass = all(gates.get(name, {}).get("pass") is True for name in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"))
    d8_complete = len(context.cell_records) == required_d8_cells and all(row.get("status") == "PASS" for row in context.cell_records)
    if not gates_pass or not d8_complete or seed.get("seed_pass") is not True:
        return TERMINAL_CALIBRATION_QA_HARNESS_HOLD
    return TERMINAL_CALIBRATION_SMOKE_COMPLETE


def run_calibration_smoke(*, go_calibration_seed_smoke: bool) -> dict[str, Any]:
    wall_start = time.perf_counter()
    if not go_calibration_seed_smoke:
        raise RuntimeError("V2_2B_SMOKE_STOP: explicit --go-calibration-seed-smoke required")
    if CALIBRATION_SMOKE_ROOT.exists():
        raise FileExistsError(f"calibration smoke slot is immutable: {CALIBRATION_SMOKE_ROOT}")
    logs = [CALIBRATION_SMOKE_LOG_ROOT / name for name in ("command.txt", "stdout.log", "stderr.log")]
    if not all(path.is_file() for path in logs):
        return _pre_scientific_abort(PreScientificAbort("smoke command/stdout/stderr logs missing"), wall_start=wall_start, smoke=True)
    try:
        qa = json.loads(QA_REPORT_PATH.read_text(encoding="utf-8"))
        if qa.get("status") != "PASS" or qa.get("source_sha256") != source_hashes():
            raise PreScientificAbort("calibration QA report is not PASS for current sources")
        if qa.get("executed_dependency_sha256") != executed_dependency_hashes():
            raise PreScientificAbort("calibration QA dependency snapshot is stale")
        source_snapshot = verify_calibration_source_snapshot()
        ledger = recompute_d512_ledger()
        if not ledger["exact_match"]:
            raise LedgerPreSealHold("FLOP_LEDGER_PRESEAL_HOLD")
        environment = full_environment_record_after_go()
    except Exception as error:
        return _pre_scientific_abort(error, wall_start=wall_start, smoke=True)

    plan = make_seed_plan(CALIBRATION_SEED, mode="smoke")
    seeds, context, hard_stop = _execute_seed_plans(
        [plan],
        result_root=CALIBRATION_SMOKE_ROOT,
        official=False,
        wall_start=wall_start,
        smoke=True,
    )
    if not context.boundary_crossed:
        return _pre_scientific_abort(PreScientificAbort("smoke exited before first numerical cell"), wall_start=wall_start, ledger=ledger, smoke=True)

    seed = seeds[0]
    gates = seed.get("gate_results", {})
    structural = bool(gates.get("D2", {}).get("pass") and gates.get("D4", {}).get("pass"))
    d3_pass = bool(gates.get("D3", {}).get("pass"))
    required_d8_cells = len(D1_K_VALUES) + 1 + 1 + len(D6_FORWARD_K_VALUES) + len(D6_BACKWARD_K_VALUES) + 2
    preliminary_status = _calibration_smoke_terminal(seed, context, hard_stop, required_d8_cells=required_d8_cells)
    smoke: dict[str, Any] = {
        "schema": "omega-v2-2b-calibration-smoke-v2",
        "official_id": OFFICIAL_ID,
        "calibration_attempt": "smoke_02",
        "classification": "CALIBRATION_QA_ONLY",
        "terminal_status": preliminary_status,
        "V2_2B_verdict": None,
        "seed_result": {key: value for key, value in seed.items() if key != "tensor_bundle"},
        "structural_gates_pass": structural,
        "D3_A_B_C_pass": d3_pass,
        "harness_gates_pass": all(gates.get(name, {}).get("pass") is True for name in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")),
        "D8_cell_records": context.cell_records,
        "capacity_issue": bool(context.capacity_issues),
        "capacity_reason": next((item.get("reason") for item in context.capacity_issues if item.get("reason") in CAPACITY_REASONS), None),
        "capacity_events": context.capacity_issues,
        "hard_stop": hard_stop,
        "D5_ledger": ledger,
        "environment": environment,
        "source_sha256": source_hashes(),
        "executed_dependency_sha256": executed_dependency_hashes(),
        "spec_sha256": sha256_file(SPEC_PATH),
        "calibration_source_snapshot_sha256": source_snapshot["snapshot_sha256"],
        "heldout_seed_values_materialized": False,
        "official_seed_plans_materialized": False,
        "required_D8_cell_count": required_d8_cells,
        "launch_logs": {
            "directory": str(CALIBRATION_SMOKE_LOG_ROOT.resolve()),
            "hash_manifest": "external_launch_logs_artifact_hashes.json",
            "finalized": False,
        },
    }
    bundle_path = CALIBRATION_SMOKE_ROOT / "calibration_seed_20260930_smoke_bundle.pt"
    torch.save({"schema": "omega-v2-2b-calibration-smoke-bundle-v2", "gates": _serialize_tree_cpu(seed.get("tensor_bundle", {}))}, bundle_path)
    smoke["bundle"] = {"path": bundle_path.name, "size_bytes": bundle_path.stat().st_size, "sha256": sha256_file(bundle_path)}

    primary = _persist_primary_evidence(CALIBRATION_SMOKE_ROOT, "CALIBRATION_SMOKE", smoke)
    if not primary["verified"]:
        hard_stop = {"kind": "PRIMARY_ARTIFACT_HASH_FAILURE", "error": "calibration primary bundle/metrics hashes failed verification"}
        smoke["hard_stop"] = hard_stop
        smoke["terminal_status"] = TERMINAL_CALIBRATION_QA_HARNESS_HOLD
        primary = _persist_primary_evidence(CALIBRATION_SMOKE_ROOT, "CALIBRATION_SMOKE", smoke)

    wall_gate_end = time.perf_counter()
    wall_gate_seconds = wall_gate_end - wall_start
    if wall_gate_seconds > D8_WALL_LIMIT_SECONDS and not any(item.get("reason") == "WALL_TIME" for item in context.capacity_issues):
        _append_capacity(context, "WALL_TIME", cell_id="calibration_primary_verification", master_seed=CALIBRATION_SEED)
        smoke["capacity_issue"] = True
        smoke["capacity_reason"] = "WALL_TIME"
        smoke["capacity_events"] = context.capacity_issues
        smoke["terminal_status"] = _calibration_smoke_terminal(seed, context, hard_stop, required_d8_cells=required_d8_cells, wall_gate_seconds=wall_gate_seconds)
        primary = _persist_primary_evidence(CALIBRATION_SMOKE_ROOT, "CALIBRATION_SMOKE", smoke)
        wall_gate_end = time.perf_counter()
        wall_gate_seconds = wall_gate_end - wall_start
    smoke.update({
        "wall_gate_start": wall_start,
        "wall_gate_end": wall_gate_end,
        "wall_gate_start_perf_counter": wall_start,
        "wall_gate_end_perf_counter": wall_gate_end,
        "wall_gate_seconds": wall_gate_seconds,
        "wall_gate_limit_seconds": D8_WALL_LIMIT_SECONDS,
        "primary_evidence": primary,
    })
    _persist_smoke_artifacts(smoke)
    return smoke


def run_official_pilot(*, go_v2_2b_official: bool, official_wall_start: float) -> dict[str, Any]:
    if not go_v2_2b_official:
        raise RuntimeError("V2_2B_STOP: --go-v2-2b-official is required")
    if OFFICIAL_RESULTS_ROOT.exists():
        raise FileExistsError(f"official result slot is immutable: {OFFICIAL_RESULTS_ROOT}")
    log_paths = [OFFICIAL_LAUNCH_LOG_ROOT / name for name in ("command.txt", "stdout.log", "stderr.log")]
    if not all(path.is_file() for path in log_paths):
        return _pre_scientific_abort(PreScientificAbort("external official command/stdout/stderr logs missing"), wall_start=official_wall_start)
    try:
        verify_source_seal()
        ledger = recompute_d512_ledger()
        if not ledger["exact_match"]:
            raise LedgerPreSealHold("FLOP_LEDGER_PRESEAL_HOLD")
        environment = full_environment_record_after_go()
    except Exception as error:
        return _pre_scientific_abort(error, wall_start=official_wall_start, ledger=recompute_d512_ledger())
    plans = list(official_seed_plans())
    seed_results, context, hard_stop = _execute_seed_plans(
        plans, result_root=OFFICIAL_RESULTS_ROOT, official=True, wall_start=official_wall_start,
    )
    if not context.boundary_crossed:
        return _pre_scientific_abort(PreScientificAbort("official execution ended before held-out frontier"), wall_start=official_wall_start, ledger=ledger)
    result = _make_result(seed_results, context, ledger, hard_stop, plans)
    result.update({
        "source_seal_sha256": sha256_file(SOURCE_SEAL_PATH),
        "environment": environment,
        "source_seal_verified": True,
        "launch_logs": {"directory": str(OFFICIAL_LAUNCH_LOG_ROOT.resolve()), "finalized": False},
    })
    _finalize_wall_in_process(result, context, OFFICIAL_RESULTS_ROOT, launch_root=OFFICIAL_LAUNCH_LOG_ROOT)
    return result


def finalize_external_launch_logs(*, smoke: bool = False) -> dict[str, Any]:
    """Write a post-process log-hash sidecar; never rewrite frozen result/report/manifests."""
    if smoke:
        root = CALIBRATION_SMOKE_ROOT
        result_path = root / "CALIBRATION_SMOKE_RESULT.json"
        launch_root = CALIBRATION_SMOKE_LOG_ROOT
        incident_path = CALIBRATION_SMOKE_INCIDENT_ROOT / "PRE_SCIENTIFIC_OPERATIONAL_ABORT.json"
    else:
        root = OFFICIAL_RESULTS_ROOT
        result_path = root / "OFFICIAL_RESULT.json"
        launch_root = OFFICIAL_LAUNCH_LOG_ROOT
        incident_path = INCIDENT_ROOT / "PRE_SCIENTIFIC_OPERATIONAL_ABORT.json"
    paths = {name: launch_root / name for name in ("command.txt", "stdout.log", "stderr.log")}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("external command/stdout/stderr logs incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    incident = json.loads(incident_path.read_text(encoding="utf-8")) if incident_path.is_file() else None
    if result is None and incident is None:
        raise FileNotFoundError("no completed result or pre-scientific incident to finalize")
    gate_start = None
    gate_seconds = None
    terminal = None
    if result is not None:
        gate_start = result.get("wall_gate_start", result.get("official_wall_start"))
        gate_seconds = result.get("wall_gate_seconds", result.get("official_wall_gate_seconds"))
        terminal = result.get("terminal_status")
    elif incident is not None:
        terminal = incident.get("terminal_status")
    files = {
        name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    process_total = time.perf_counter() - float(gate_start) if gate_start is not None else None
    payload = {
        "schema": "omega-v2-2b-external-launch-log-hashes-v1",
        "official_id": OFFICIAL_ID,
        "attempt": "smoke_02" if smoke else "official",
        "terminal_status": terminal,
        "directory": str(launch_root.resolve()),
        "files": files,
        "gate_wall_seconds_frozen": gate_seconds,
        "process_total_wall_seconds": process_total,
        "process_total_wall_is_diagnostic_only": True,
        "verified": all(path.is_file() and path.stat().st_size == files[name]["size_bytes"] and sha256_file(path) == files[name]["sha256"] for name, path in paths.items()),
    }
    manifest_path = root / "external_launch_logs_artifact_hashes.json" if result is not None else incident_path.parent / "external_launch_logs_artifact_hashes.json"
    write_json(manifest_path, payload)
    core_verified = True
    if result is not None:
        core_verified_path = root / "artifact_hashes_verified.json"
        core_verified = core_verified_path.is_file() and json.loads(core_verified_path.read_text(encoding="utf-8")).get("verified") is True
    return {
        "terminal_status": terminal,
        "launch_logs_finalized": True,
        "artifact_hashes_verified": bool(payload["verified"] and core_verified),
        "gate_wall_seconds_unchanged": True,
        "process_total_wall_seconds": process_total,
        "external_log_manifest": str(manifest_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--qa-calibration", action="store_true")
    modes.add_argument("--pre-cuda-official-dry-run", action="store_true")
    modes.add_argument("--full-control-flow-mocked-dry-run", action="store_true")
    modes.add_argument("--calibration-seed-smoke", action="store_true")
    modes.add_argument("--write-calibration-source-snapshot", action="store_true")
    modes.add_argument("--seal-source", action="store_true")
    modes.add_argument("--run-official", action="store_true")
    modes.add_argument("--finalize-launch-logs", action="store_true")
    modes.add_argument("--finalize-smoke-launch-logs", action="store_true")
    parser.add_argument("--go-calibration-seed-smoke", action="store_true")
    parser.add_argument("--go-v2-2b-official", action="store_true")
    args = parser.parse_args(argv)
    if args.run_official:
        if not args.go_v2_2b_official:
            raise RuntimeError("V2_2B_STOP: --go-v2-2b-official is required")
        start = time.perf_counter()
        result = run_official_pilot(go_v2_2b_official=True, official_wall_start=start)
    elif args.qa_calibration:
        result = run_qa_calibration()
    elif args.pre_cuda_official_dry_run:
        result = pre_cuda_official_dry_run()
    elif args.full_control_flow_mocked_dry_run:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_mocked_cli_") as temporary:
            result = full_mocked_control_flow_dry_run(Path(temporary) / "dry_run")
    elif args.calibration_seed_smoke:
        result = run_calibration_smoke(go_calibration_seed_smoke=args.go_calibration_seed_smoke)
    elif args.write_calibration_source_snapshot:
        result = write_calibration_source_snapshot()
    elif args.finalize_launch_logs:
        result = finalize_external_launch_logs(smoke=False)
    elif args.finalize_smoke_launch_logs:
        result = finalize_external_launch_logs(smoke=True)
    else:
        result = create_source_seal()
    print(json.dumps(result, sort_keys=True, indent=2))
    if args.qa_calibration:
        return 0 if result.get("status") == "PASS" else 1
    if args.full_control_flow_mocked_dry_run:
        return 0 if result.get("all_scenarios_pass") and result.get("artifact_hashes_verified") else 1
    if args.calibration_seed_smoke:
        return 0 if result.get("terminal_status") == TERMINAL_CALIBRATION_SMOKE_COMPLETE else 1
    if args.run_official:
        return 0 if result.get("terminal_status") in (TERMINAL_PASS, TERMINAL_FAIL, TERMINAL_CAPACITY_HOLD) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
