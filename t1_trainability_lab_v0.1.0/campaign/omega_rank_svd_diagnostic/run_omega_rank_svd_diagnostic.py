"""OMEGA rank-SVD diagnostic.

The diagnostic is documentary only.  Synthetic smoke execution is local and
CPU-only.  Real checkpoint/corpus evaluation requires both ``--full`` and
``--confirm-omega-rank-svd-diagnostic``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
R1_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
FASTPATH_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(FASTPATH_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402
from run_scientific_scoping_a import evaluate_validation  # noqa: E402


VOCAB_SIZE = 50257
DIMENSION = 128
SLOTS = 8
RANKS = (32, 64)
SEEDS = (20260913, 20260914)
K_VALUES = (1, 4)
CHECKPOINT_UPDATE = 2000
PRIMARY_VALIDATION_DOCUMENTS = 8
WINDOW_TOKENS = 256
RETAINED_TOKENS = 513
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"


@dataclass(frozen=True)
class CheckpointSpec:
    architecture: str
    k: int
    seed: int
    path: Path


@dataclass(frozen=True)
class SVDResult:
    left: Tensor
    singular_values: Tensor
    right_transpose: Tensor


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_self_hashed_json(path: Path, payload: dict[str, Any]) -> str:
    """Write once and verify hash over canonical bytes with placeholder field."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    snapshot = json_safe(payload)
    snapshot["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    unsigned = canonical_json(snapshot)
    digest = sha256_bytes(unsigned)
    snapshot["artifact_self_hash"] = digest
    encoded = canonical_json(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    persisted = path.read_bytes()
    parsed = json.loads(persisted.decode("utf-8"))
    stored = parsed["artifact_self_hash"]
    parsed["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    if persisted != encoded or canonical_json(parsed) != unsigned or sha256_bytes(canonical_json(parsed)) != stored:
        raise RuntimeError("artifact self-hash verification failed")
    return digest


def checkpoint_specs(root: Path = CAMPAIGN_ROOT) -> list[CheckpointSpec]:
    checkpoint_root = root / "omega_core_lm_0_r1_scientific_scoping_a" / "results" / "full_campaign" / "runs"
    return [
        CheckpointSpec("R1", k, seed, checkpoint_root / f"shared_K{k}_seed_{seed}" / "checkpoint_02000.pt")
        for seed in SEEDS
        for k in K_VALUES
    ]


def primary_validation_protocol() -> dict[str, Any]:
    return {
        "name": "R1 scientific scoping A primary validation",
        "document_count": PRIMARY_VALIDATION_DOCUMENTS,
        "document_selection": "same frozen eight-document validation manifest",
        "document_tokens": RETAINED_TOKENS,
        "windows": [0, 1],
        "window_tokens": WINDOW_TOKENS,
        "metric": "mean token cross-entropy NLL",
        "evaluator": "run_scientific_scoping_a.evaluate_validation",
        "test_split": False,
    }


def full_svd(weight: Tensor) -> SVDResult:
    """Compute every singular direction of a 2-D weight matrix.

    ``full_matrices=False`` is the economy representation: for [V, D] with
    V >= D it retains all D singular values and all D usable left directions,
    without allocating an otherwise useless [V, V] orthogonal matrix.
    """
    if weight.ndim != 2:
        raise ValueError(f"tied embedding must be 2-D, got {tuple(weight.shape)}")
    left, singular_values, right_transpose = torch.linalg.svd(weight.detach(), full_matrices=False)
    usable_directions = min(weight.shape)
    if left.shape != (weight.shape[0], usable_directions) or singular_values.shape != (usable_directions,) or right_transpose.shape != (usable_directions, weight.shape[1]):
        raise RuntimeError("full SVD did not retain all embedding dimensions")
    return SVDResult(left, singular_values, right_transpose)


def svd_factors(svd: SVDResult, rank: int) -> tuple[Tensor, Tensor]:
    if rank not in RANKS:
        raise ValueError(f"diagnostic rank must be one of {RANKS}, got {rank}")
    if rank > svd.singular_values.numel():
        raise ValueError(f"rank {rank} exceeds available singular directions")
    root_sigma = svd.singular_values[:rank].sqrt()
    c = svd.left[:, :rank] * root_sigma.unsqueeze(0)
    u = root_sigma.unsqueeze(1) * svd.right_transpose[:rank, :]
    return c, u


class FrozenSVDVocabulary(nn.Module):
    """Tied input/output vocabulary backed by frozen SVD factors."""

    def __init__(self, c: Tensor, u: Tensor) -> None:
        super().__init__()
        if c.ndim != 2 or u.ndim != 2 or c.shape[1] != u.shape[0]:
            raise ValueError("invalid C/U factor shapes")
        self.vocab_size = c.shape[0]
        self.rank = c.shape[1]
        self.dimension = u.shape[1]
        self.register_buffer("C", c.detach().clone())
        self.register_buffer("U", u.detach().clone())

    def forward(self, tokens: Tensor) -> Tensor:
        return self.C[tokens] @ self.U


class FrozenSVDModel(OmegaCoreLMFast):
    """F model with only tied lexical input/output replaced by frozen C/U."""

    def __init__(self, source: OmegaCoreLMFast, c: Tensor, u: Tensor) -> None:
        super().__init__(
            vocab_size=1,
            dimension=source.dimension,
            slots=source.slots,
            rounds=source.rounds,
            variant=source.variant,
        )
        self.vocab_size = source.vocab_size
        target = self.state_dict()
        for name, value in source.state_dict().items():
            if name == "embedding.weight":
                continue
            if name not in target:
                raise RuntimeError(f"missing frozen-copy destination state: {name}")
            target[name].copy_(value)
        self.embedding = FrozenSVDVocabulary(c, u)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def logits_from_projected(self, projected: Tensor) -> Tensor:
        scale = self.dimension**-0.5
        return (projected @ self.embedding.U.transpose(-2, -1)) @ self.embedding.C.transpose(-2, -1) * scale


def replace_tied_embedding(source: OmegaCoreLMFast, c: Tensor, u: Tensor) -> FrozenSVDModel:
    if tuple(source.embedding.weight.shape) != (source.vocab_size, source.dimension):
        raise ValueError("source model does not expose expected tied [V,D] embedding")
    if tuple(c.shape) != (source.vocab_size, u.shape[0]) or tuple(u.shape) != (u.shape[0], source.dimension):
        raise ValueError("SVD factor dimensions do not match source model")
    return FrozenSVDModel(source, c, u)


def reconstruction_metrics(weight: Tensor, svd: SVDResult, rank: int) -> dict[str, float | int]:
    c, u = svd_factors(svd, rank)
    reconstruction = c @ u
    residual = torch.linalg.vector_norm(weight - reconstruction)
    norm = torch.linalg.vector_norm(weight)
    energy = (svd.singular_values[:rank].square().sum() / svd.singular_values.square().sum()).item()
    return {
        "rank": rank,
        "relative_frobenius_error": float((residual / norm).item()),
        "retained_squared_singular_value_energy": float(energy),
    }


def evaluate_primary(model: nn.Module, documents: list[dict[str, Any]]) -> dict[str, Any]:
    if len(documents) != PRIMARY_VALIDATION_DOCUMENTS:
        raise ValueError(f"primary validation requires exactly {PRIMARY_VALIDATION_DOCUMENTS} documents")
    return evaluate_validation(model, documents)


def run_diagnostic(
    checkpoints: Iterable[CheckpointSpec],
    validation_documents: list[dict[str, Any]],
    model_loader: Callable[[CheckpointSpec], tuple[OmegaCoreLMFast, dict[str, Any]]],
    output_dir: Path,
    *,
    execution_mode: str,
    real_evaluation_executed: bool,
    primary_validation_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint_values = list(checkpoints)
    records: list[dict[str, Any]] = []
    for spec in checkpoint_values:
        source, metadata = model_loader(spec)
        weight = source.embedding.weight.detach()
        if tuple(weight.shape) != (VOCAB_SIZE, DIMENSION) and execution_mode == "real":
            raise ValueError(f"real tied embedding must have shape {(VOCAB_SIZE, DIMENSION)}, got {tuple(weight.shape)}")
        svd = full_svd(weight)
        baseline = evaluate_primary(source, validation_documents)
        for rank in RANKS:
            c, u = svd_factors(svd, rank)
            replacement = replace_tied_embedding(source, c, u)
            result = evaluate_primary(replacement, validation_documents)
            records.append({
                **metadata,
                "rank": rank,
                "baseline_r1_nll": baseline["nll"],
                "svd_nll": result["nll"],
                "delta_svd_minus_r1": result["nll"] - baseline["nll"],
                "tokens": result["tokens"],
                "finite": result["finite"],
                "reconstruction": reconstruction_metrics(weight, svd, rank),
                "replacement": "frozen tied input/output C/U only",
            })
    report = {
        "schema": "omega-rank-svd-diagnostic-v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "training_executed": False,
        "execution_mode": execution_mode,
        "real_checkpoint_evaluation": real_evaluation_executed,
        "checkpoint_update": CHECKPOINT_UPDATE,
        "ranks": list(RANKS),
        "checkpoint_count": len(checkpoint_values),
        "primary_validation_protocol": primary_validation_protocol(),
        "results": records,
        "source_hashes": {"runner": sha256_file(Path(__file__).resolve())},
    }
    if primary_validation_manifest_sha256 is not None:
        report["primary_validation_manifest_sha256"] = primary_validation_manifest_sha256
    write_self_hashed_json(output_dir / "diagnostic_report.json", report)
    report["artifact_self_hash"] = json.loads((output_dir / "diagnostic_report.json").read_text(encoding="utf-8"))["artifact_self_hash"]
    return report


def verify_checkpoint_payload(payload: dict[str, Any], spec: CheckpointSpec) -> None:
    if int(payload.get("update", -1)) != CHECKPOINT_UPDATE:
        raise ValueError(f"checkpoint update mismatch for {spec.path}")
    if int(payload.get("seed", -1)) != spec.seed:
        raise ValueError(f"checkpoint seed mismatch for {spec.path}")
    config = payload.get("config")
    if not isinstance(config, dict) or int(config.get("seed", -1)) != spec.seed:
        raise ValueError(f"checkpoint config seed mismatch for {spec.path}")
    if config.get("variant") != f"shared_K{spec.k}":
        raise ValueError(f"checkpoint K/variant mismatch for {spec.path}")
    identity = payload.get("implementation_identity")
    if not isinstance(identity, dict) or identity.get("implementation") != "F":
        raise ValueError(f"R1/F implementation identity missing for {spec.path}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"checkpoint model state missing for {spec.path}")


def load_checkpoint_model(spec: CheckpointSpec) -> tuple[OmegaCoreLMFast, dict[str, Any]]:
    actual_sha256 = sha256_file(spec.path)
    payload = torch.load(spec.path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be an object: {spec.path}")
    verify_checkpoint_payload(payload, spec)
    model = OmegaCoreLMFast(vocab_size=VOCAB_SIZE, dimension=DIMENSION, slots=SLOTS, rounds=spec.k, variant="shared").float()
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "architecture": "R1",
        "K": spec.k,
        "seed": spec.seed,
        "checkpoint_path": spec.path.as_posix(),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_update": CHECKPOINT_UPDATE,
    }


def run_real(output_dir: Path) -> dict[str, Any]:
    from run_scientific_scoping_a import _load_real_documents

    _, validation_documents, _, validation_manifest, teacher = _load_real_documents()
    del teacher
    if len(validation_documents) != PRIMARY_VALIDATION_DOCUMENTS:
        raise ValueError("frozen primary validation did not contain exactly eight documents")
    report = run_diagnostic(
        checkpoint_specs(),
        validation_documents,
        load_checkpoint_model,
        output_dir,
        execution_mode="real",
        real_evaluation_executed=True,
        primary_validation_manifest_sha256=validation_manifest["manifest_sha256"],
    )
    return report


def synthetic_documents(vocab_size: int) -> list[dict[str, Any]]:
    return [{"tokens": [((index + 1) * 7 + position) % vocab_size for position in range(RETAINED_TOKENS)]} for index in range(PRIMARY_VALIDATION_DOCUMENTS)]


def run_smoke(output_dir: Path) -> dict[str, Any]:
    torch.manual_seed(20260918)
    source = OmegaCoreLMFast(vocab_size=70, dimension=DIMENSION, slots=1, rounds=1, variant="shared").float()
    documents = synthetic_documents(70)

    def loader(spec: CheckpointSpec) -> tuple[OmegaCoreLMFast, dict[str, Any]]:
        del spec
        return copy.deepcopy(source), {"architecture": "synthetic-R1", "K": 1, "seed": 0, "checkpoint_sha256": "synthetic"}

    return run_diagnostic(
        [CheckpointSpec("synthetic-R1", 1, 0, Path("synthetic"))],
        documents,
        loader,
        output_dir,
        execution_mode="synthetic_smoke",
        real_evaluation_executed=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-omega-rank-svd-diagnostic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "omega_rank_svd_diagnostic")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))
        return 0
    if not (args.full and args.confirm_omega_rank_svd_diagnostic):
        parser.error("real diagnostic requires --full --confirm-omega-rank-svd-diagnostic")
    print(json.dumps(run_real(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
