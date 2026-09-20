"""OMEGA hidden-cache production integration gate.

All real work is explicitly guarded. Tests use tiny synthetic stores and tiny
models; they never load the 0.88 GiB cache or DistilGPT2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
PROBE_DIR = CAMPAIGN_ROOT / "omega_teacher_hidden_cache_probe"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))
import run_omega_teacher_hidden_cache_probe as hidden  # noqa: E402


CAMPAIGN_ID = "OMEGA-HIDDEN-CACHE-PRODUCTION-INTEGRATION"
SEALED_MANIFEST = PROBE_DIR / "results" / "cache_manifest.json"
DEFAULT_CACHE_FILE = Path(r"C:\omega_cache\teacher_hidden.fp32")
MIN_AVAILABLE_BYTES = 1 << 30
SMOKE_UPDATES = 8
SMOKE_CHECKPOINT_UPDATE = 4
BOUNDARY_UPDATES = (0, 1, 2, 3, 4, 5, 6, 7)
K_VALUES = (1, 4)
PHYSICAL_BATCH = hidden.PHYSICAL_BATCH
BASE_LR = hidden.BASE_LR
ADAMW_BETAS = hidden.ADAMW_BETAS
ADAMW_EPS = hidden.ADAMW_EPS
WEIGHT_DECAY = hidden.WEIGHT_DECAY

# Pinned values from the sealed, independently verified hidden-cache unit.
SEALED_EXPECTATIONS = {
    "manifest_self_hash": "5472ef9a4f3fa38e9c3436dede9a8eb08d7d860eb0f7ce915cfbb452d1651abc",
    "cache_file_sha256": "b43e9c37f9585e1c5d605df5cc58caba84a10480b4f8f06e747cb8d2f3cd16ff",
    "teacher_model_id": "distilbert/distilgpt2",
    "teacher_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "teacher_parameter_sha256": "73d9f64fb946a490e711457193add57a30110663277ff0936d9bcd540b2f5370",
    "tokenizer_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "tokenizer_sha256": "d657e96ec92d9fc9eb130258eda9efc68eadc499ebcca184cb79416b4827bf77",
    "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "source_manifest_sha256": "a114b42252cef45dddca18bd4d1d31eebb0fd966363265539bc69703a00546ef",
    "lm_head_weight_sha256": "d502923f9730f0fa6644f847162f403dc4e40a403cf4b47853a9d65f248cb776",
}


class ProvenanceError(RuntimeError):
    pass


class RealExecutionAuthorizationError(RuntimeError):
    pass


class MemorySafetyError(RuntimeError):
    pass


class IntegrationLockError(RuntimeError):
    pass


class OracleMismatchError(RuntimeError):
    pass


def require_real_authorization(confirmed: bool, operation: str) -> None:
    if not confirmed:
        raise RealExecutionAuthorizationError(f"{operation} requires --confirm-real-execution")


def memory_snapshot() -> dict[str, int]:
    info = psutil.virtual_memory()
    return {"rss_bytes": int(psutil.Process().memory_info().rss), "available_bytes": int(info.available)}


def verify_provenance(manifest_path: Path = SEALED_MANIFEST, *, cache_file: Path | None = None, expected: Mapping[str, str] = SEALED_EXPECTATIONS) -> dict[str, Any]:
    """Verify every sealed identity before loading cached training data."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not hidden.verify_self_hash(manifest, "manifest_self_hash"):
        raise ProvenanceError("manifest self-hash mismatch")
    for field, value in expected.items():
        if field == "manifest_self_hash":
            actual = manifest.get("manifest_self_hash")
        elif field == "cache_file_sha256":
            actual = manifest.get("cache_file_sha256")
        elif field == "teacher_model_id":
            actual = manifest.get("teacher", {}).get("model_id")
        elif field == "teacher_revision":
            actual = manifest.get("teacher", {}).get("revision")
        elif field == "teacher_parameter_sha256":
            actual = manifest.get("teacher", {}).get("parameter_sha256")
        elif field == "tokenizer_revision":
            actual = manifest.get("tokenizer", {}).get("revision")
        elif field == "tokenizer_sha256":
            actual = manifest.get("tokenizer", {}).get("sha256")
        elif field == "dataset_revision":
            actual = manifest.get("source", {}).get("dataset_revision")
        elif field == "source_manifest_sha256":
            actual = manifest.get("source", {}).get("manifest_sha256")
        elif field == "lm_head_weight_sha256":
            actual = manifest.get("lm_head", {}).get("weight_sha256")
        else:
            raise ProvenanceError(f"unknown expected provenance field: {field}")
        if actual != value:
            raise ProvenanceError(f"provenance mismatch for {field}: expected {value}, got {actual}")
    if manifest.get("status") != "CACHE_SEALED" or manifest.get("access") != "READ_ONLY":
        raise ProvenanceError("cache is not sealed read-only")
    if manifest.get("shape") != [2, 602, 256, 768] or manifest.get("dtype") != "float32" or manifest.get("representation") != "raw_fp32_hidden":
        raise ProvenanceError("cache shape/dtype/representation mismatch")
    resolved_cache = Path(manifest["cache_file"]).resolve() if cache_file is None else cache_file.resolve()
    if resolved_cache != Path(manifest["cache_file"]).resolve():
        raise ProvenanceError("cache path differs from sealed manifest")
    if resolved_cache.stat().st_size != hidden.EXPECTED_CACHE_BYTES:
        raise ProvenanceError("cache file size mismatch")
    actual_cache_hash = hidden.base.file_hash(resolved_cache)
    if actual_cache_hash != manifest["cache_file_sha256"]:
        raise ProvenanceError("cache file hash mismatch")
    lm_head_path = manifest_path.parent / manifest["lm_head"]["weight_file"]
    if hidden.base.file_hash(lm_head_path) != manifest["lm_head"]["weight_sha256"]:
        raise ProvenanceError("lm_head hash mismatch")
    return {"passed": True, "manifest_path": manifest_path.resolve().as_posix(), "cache_file": resolved_cache.as_posix(), "manifest_self_hash": manifest["manifest_self_hash"], "cache_file_sha256": actual_cache_hash, "shape": manifest["shape"], "dtype": manifest["dtype"], "teacher_transformer_loaded": False, "teacher_transformer_forward_calls": 0}


class CacheLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "CacheLock":
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode("ascii"))
        except FileExistsError as exc:
            raise IntegrationLockError(f"cache integration lock already exists: {self.path}") from exc
        return self

    def __exit__(self, *_: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


def memory_safety_gate(*, minimum_available_bytes: int = MIN_AVAILABLE_BYTES) -> dict[str, Any]:
    snapshot = memory_snapshot()
    if snapshot["available_bytes"] < minimum_available_bytes:
        raise MemorySafetyError(f"available memory {snapshot['available_bytes']} below minimum {minimum_available_bytes}")
    return {"passed": True, "minimum_available_bytes": minimum_available_bytes, **snapshot}


class ProductionHiddenRoute:
    """RAM-only hidden route. Deliberately has no teacher object."""

    def __init__(self, payload: np.ndarray, weight: Tensor, bias: Tensor | None) -> None:
        self.payload = payload
        self.weight = weight
        self.bias = bias
        self.teacher_transformer_loaded = False
        self.teacher_transformer_forward_calls = 0

    def logits(self, document_positions: Sequence[int], window: int) -> Tensor:
        hidden_states = torch.from_numpy(np.array(self.payload[window, list(document_positions)], copy=True)).float()
        return hidden.hidden_to_logits(hidden_states, self.weight, self.bias)

    def accounting(self) -> dict[str, Any]:
        return {"teacher_transformer_loaded": self.teacher_transformer_loaded, "teacher_transformer_forward_calls": self.teacher_transformer_forward_calls}


def smoke_schedule(updates: int = SMOKE_UPDATES) -> list[dict[str, int]]:
    if updates <= 0 or updates % 2:
        raise ValueError("smoke updates must be positive and even")
    return [{"update": update, "pair": update // 2, "window": update % 2} for update in range(updates)]


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, state: Tensor, next_update: int, *, accounting: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "omega-hidden-cache-production-checkpoint-v1", "update": next_update, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state.detach().clone(), "accounting": dict(accounting), "schedule_cursor": {"next_update": next_update, "next_pair": next_update // 2, "next_window": next_update % 2}}


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "omega-hidden-cache-production-checkpoint-v1":
        raise ProvenanceError("checkpoint schema mismatch")
    return payload


def checkpoint_fingerprint(payload: Mapping[str, Any]) -> str:
    return hidden.base.canonical_hash({"update": payload["update"], "model": hidden.base.parameter_hash(_StateModel(payload["model"])), "optimizer": payload["optimizer"], "state": hidden.tensor_hash(payload["state"])})


class _StateModel(nn.Module):
    """Only used to hash a state dict in tests/diagnostics."""

    def __init__(self, state: Mapping[str, Tensor]) -> None:
        super().__init__()
        for name, value in state.items():
            self.register_buffer(name.replace(".", "_"), value)


def compare_results(direct: Mapping[str, Any], cached: Mapping[str, Any]) -> dict[str, Any]:
    exact = direct["model_state_hash"] == cached["model_state_hash"] and direct["optimizer_state_hash"] == cached["optimizer_state_hash"] and torch.equal(direct["next_state"], cached["next_state"])
    exact = exact and all(torch.equal(direct["losses"][key], cached["losses"][key]) for key in direct["losses"])
    exact = exact and direct["clip_norm"] == cached["clip_norm"]
    return {"passed": exact, "model_state_hash": direct["model_state_hash"], "optimizer_state_hash": direct["optimizer_state_hash"]}


def _nested_exact(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left) == list(right) and all(_nested_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_exact(a, b) for a, b in zip(left, right))
    return left == right


def _state_dict_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return list(left) == list(right) and all(torch.equal(left[key], right[key]) for key in left)


def _load_student_from_checkpoint(payload: Mapping[str, Any], k: int) -> tuple[nn.Module, torch.optim.Optimizer, Tensor, int]:
    import run_omega_ce_only_baseline as ce

    model = ce.fresh_model(20260913, k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, payload["state"].clone(), int(payload["update"])


def _run_sequence(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Tensor,
    start_update: int,
    end_update: int,
    documents: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    mode: str,
    teacher: nn.Module | None = None,
    route: ProductionHiddenRoute | None = None,
) -> tuple[dict[str, Any], Tensor]:
    if mode not in {"direct", "cached"}:
        raise ValueError(mode)
    rows: list[dict[str, Any]] = []
    for update in range(start_update, end_update):
        pair_index = update // 2
        window = update % 2
        positions = [int(index) for index in pairs[pair_index]["document_indices"]]
        source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
        offset = window * 256
        inputs = source[:, offset : offset + 256]
        targets = source[:, offset + 1 : offset + 257]
        current_state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if window == 0 else state
        if mode == "direct":
            if teacher is None:
                raise RuntimeError("direct sequence missing teacher")
            teacher_logits = hidden.teacher_direct_logits(teacher, source, window)
        else:
            if route is None:
                raise RuntimeError("cached sequence missing hidden route")
            teacher_logits = route.logits(positions, window)
        result = hidden.base.training_update(model, optimizer, inputs, targets, current_state, teacher_logits)
        rows.append({"update": update, "pair": pair_index, "window": window, "losses": {key: float(value.item()) for key, value in result["losses"].items()}, "clip_norm": result["clip_norm"], "model_state_hash": result["model_state_hash"], "optimizer_state_hash": result["optimizer_state_hash"]})
        state = result["next_state"] if window == 0 else model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
    return {"rows": rows, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state, "next_update": end_update}, state


def _resume_worker(*, manifest_path: Path, cache_file: Path, checkpoint_path: Path, output_path: Path, k: int, end_update: int) -> dict[str, Any]:
    provenance = verify_provenance(manifest_path, cache_file=cache_file)
    memory = memory_safety_gate()
    manifest, resolved_cache = hidden.hidden_cache_load_manifest(manifest_path)
    frozen, documents, _ = hidden.base.load_frozen_train_documents()
    payload = load_checkpoint(checkpoint_path)
    model, optimizer, state, start_update = _load_student_from_checkpoint(payload, k)
    hidden_states, load_seconds, _ = hidden.preload_hidden_cache(resolved_cache, manifest)
    weight, bias = hidden.load_lm_head(manifest_path.parent, manifest["lm_head"])
    route = ProductionHiddenRoute(hidden_states, weight, bias)
    result, final_state = _run_sequence(model=model, optimizer=optimizer, state=state, start_update=start_update, end_update=end_update, documents=documents, pairs=frozen["cyclic_pairs"]["pairs"], mode="cached", route=route)
    final_payload = checkpoint_payload(model, optimizer, final_state, end_update, accounting=route.accounting())
    save_checkpoint(output_path, final_payload)
    return {"provenance": provenance, "memory": memory, "cache_load_seconds": load_seconds, "resume_start_update": start_update, "resume_end_update": end_update, "rows": result["rows"], "final_checkpoint": output_path.as_posix(), "accounting": route.accounting()}


def run_production_smoke(*, manifest_path: Path, cache_file: Path, output_root: Path, confirm_real_execution: bool) -> dict[str, Any]:
    """Run K1/K4 direct oracle and cached fresh-process resume smoke."""
    require_real_authorization(confirm_real_execution, "production integration smoke")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_file.with_suffix(cache_file.suffix + ".integration.lock")
    memory = memory_safety_gate()
    with CacheLock(lock_path):
        provenance = verify_provenance(manifest_path, cache_file=cache_file)
        manifest, resolved_cache = hidden.hidden_cache_load_manifest(manifest_path)
        frozen, documents, _ = hidden.base.load_frozen_train_documents()
        hidden_states, cache_load_seconds, _ = hidden.preload_hidden_cache(resolved_cache, manifest)
        weight, bias = hidden.load_lm_head(manifest_path.parent, manifest["lm_head"])
        route = ProductionHiddenRoute(hidden_states, weight, bias)
        combinations: dict[str, Any] = {}
        for k in K_VALUES:
            import run_omega_ce_only_baseline as ce

            initial_model = ce.fresh_model(20260913, k)
            initial_optimizer = torch.optim.AdamW(initial_model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
            initial_state = initial_model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
            initial_payload = checkpoint_payload(initial_model, initial_optimizer, initial_state, 0, accounting={"teacher_transformer_loaded": False, "teacher_transformer_forward_calls": 0})
            initial_path = output_root / f"initial_K{k}.pt"
            save_checkpoint(initial_path, initial_payload)

            direct_model, direct_optimizer, direct_state, _ = _load_student_from_checkpoint(load_checkpoint(initial_path), k)
            teacher = hidden.base.load_real_teacher()
            direct_result, direct_final_state = _run_sequence(model=direct_model, optimizer=direct_optimizer, state=direct_state, start_update=0, end_update=SMOKE_UPDATES, documents=documents, pairs=frozen["cyclic_pairs"]["pairs"], mode="direct", teacher=teacher)
            direct_final = checkpoint_payload(direct_model, direct_optimizer, direct_final_state, SMOKE_UPDATES, accounting={"teacher_transformer_loaded": True, "teacher_transformer_forward_calls": SMOKE_UPDATES})
            direct_path = output_root / f"direct_final_K{k}.pt"
            save_checkpoint(direct_path, direct_final)

            cached_model, cached_optimizer, cached_state, _ = _load_student_from_checkpoint(load_checkpoint(initial_path), k)
            cached_first, cached_state_after = _run_sequence(model=cached_model, optimizer=cached_optimizer, state=cached_state, start_update=0, end_update=SMOKE_CHECKPOINT_UPDATE, documents=documents, pairs=frozen["cyclic_pairs"]["pairs"], mode="cached", route=route)
            resume_path = output_root / f"resume_input_K{k}.pt"
            save_checkpoint(resume_path, checkpoint_payload(cached_model, cached_optimizer, cached_state_after, SMOKE_CHECKPOINT_UPDATE, accounting=route.accounting()))
            resume_output = output_root / f"cached_final_K{k}.pt"
            command = [sys.executable, str(Path(__file__).resolve()), "--resume-worker", "--manifest", str(manifest_path.resolve()), "--cache-file", str(cache_file.resolve()), "--checkpoint", str(resume_path.resolve()), "--output-checkpoint", str(resume_output.resolve()), "--K", str(k), "--end-update", str(SMOKE_UPDATES), "--confirm-real-execution", "--lock-held"]
            completed = subprocess.run(command, cwd=hidden.base.REPO_ROOT, capture_output=True, text=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"fresh resume worker failed for K{k}: {completed.stderr}")
            cached_worker = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
            cached_final = load_checkpoint(resume_output)
            exact = _state_dict_exact(direct_final["model"], cached_final["model"]) and _nested_exact(direct_final["optimizer"], cached_final["optimizer"]) and torch.equal(direct_final["state"], cached_final["state"])
            if not exact:
                raise OracleMismatchError(f"production oracle mismatch for K{k}")
            combinations[f"K{k}"] = {"smoke_updates": SMOKE_UPDATES, "checkpoint_boundary": SMOKE_CHECKPOINT_UPDATE, "direct_rows": direct_result["rows"], "cached_first_rows": cached_first["rows"], "fresh_resume": cached_worker, "oracle_bit_exact": exact, "cached_accounting": route.accounting(), "direct_teacher_transformer_loaded": True, "direct_teacher_transformer_forward_calls": SMOKE_UPDATES, "direct_checkpoint": direct_path.as_posix(), "cached_checkpoint": resume_output.as_posix()}
    report = {"schema": "omega-hidden-cache-production-integration-v1", "campaign_id": CAMPAIGN_ID, "status": "INTEGRATION_PASS", "provenance": provenance, "memory": memory, "cache_load_seconds": cache_load_seconds, "schedule": {"updates": SMOKE_UPDATES, "boundaries": BOUNDARY_UPDATES, "window_pair_pattern": "update%2: window0/window1; update//2: cyclic pair"}, "combinations": combinations}
    return hidden.write_self_hashed(output_root / "integration_report.json", report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-worker", action="store_true")
    parser.add_argument("--manifest", type=Path, default=SEALED_MANIFEST)
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--K", type=int)
    parser.add_argument("--end-update", type=int, default=SMOKE_UPDATES)
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--confirm-real-execution", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke and not args.resume_worker:
        parser.error("select --smoke")
    if args.resume_worker:
        if args.checkpoint is None or args.output_checkpoint is None or args.K not in K_VALUES:
            parser.error("resume worker requires --checkpoint, --output-checkpoint, and --K 1|4")
        require_real_authorization(args.confirm_real_execution, "production integration resume")
        result = _resume_worker(manifest_path=args.manifest, cache_file=args.cache_file, checkpoint_path=args.checkpoint, output_path=args.output_checkpoint, k=args.K, end_update=args.end_update)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run_production_smoke(manifest_path=args.manifest, cache_file=args.cache_file, output_root=args.output_root, confirm_real_execution=args.confirm_real_execution)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
