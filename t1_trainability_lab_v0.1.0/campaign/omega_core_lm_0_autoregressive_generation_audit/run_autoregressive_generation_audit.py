"""Documentary autoregressive audit for frozen R1/F and ER32 checkpoints.

The default CLI path is blocked. ``--smoke`` uses tiny synthetic models; real
execution requires an explicit confirmation flag and performs inference only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
R1_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
R1_FASTPATH_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
ER32_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er32_integration_and_cost_gate"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(R1_FASTPATH_DIR))
sys.path.insert(0, str(ER32_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402
from omega_fast_er32 import OmegaCoreLMFastER32  # noqa: E402
from run_scientific_scoping_a import (  # noqa: E402
    FULL_PAIRS,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_VOCAB,
    _load_real_documents,
    file_hash,
)


SEEDS = (20260913, 20260914)
KS = (1, 4)
CHECKPOINT_UPDATE = 2000
PROMPT_LENGTH = 32
MAX_NEW_TOKENS = 64
EOS_TOKEN_ID = 50256
EXPECTED_GENERATIONS = 64
CPU_INTRAOP_THREADS = 4
CPU_INTEROP_THREADS = 1
_CPU_RUNTIME_CONFIGURED = False


@dataclass(frozen=True)
class CheckpointSpec:
    architecture: str
    k: int
    seed: int
    path: Path


@dataclass(frozen=True)
class PromptSpec:
    document_id: str
    document_hash: str
    token_ids: tuple[int, ...]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def configure_cpu() -> None:
    global _CPU_RUNTIME_CONFIGURED
    if _CPU_RUNTIME_CONFIGURED:
        return
    torch.set_num_threads(CPU_INTRAOP_THREADS)
    torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    torch.set_float32_matmul_precision("highest")
    _CPU_RUNTIME_CONFIGURED = True


def checkpoint_specs(root: Path = CAMPAIGN_ROOT) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    r1_root = root / "omega_core_lm_0_r1_scientific_scoping_a" / "results" / "full_campaign" / "runs"
    er32_root = root / "omega_core_lm_0_er32_quality_scoping_a" / "results" / "quality_scoping_a" / "runs"
    for seed in SEEDS:
        for k in KS:
            specs.append(CheckpointSpec("R1", k, seed, r1_root / f"shared_K{k}_seed_{seed}" / "checkpoint_02000.pt"))
            specs.append(CheckpointSpec("ER32", k, seed, er32_root / f"ER32-K{k}_seed_{seed}" / "checkpoint_02000.pt"))
    return specs


def prompt_specs(documents: Iterable[dict[str, Any]]) -> list[PromptSpec]:
    prompts: list[PromptSpec] = []
    for document in documents:
        tokens = tuple(int(token) for token in document["tokens"][:PROMPT_LENGTH])
        if len(tokens) != PROMPT_LENGTH:
            raise ValueError("validation document has fewer than 32 tokens")
        prompts.append(PromptSpec(str(document["document_index"]), str(document["full_text_sha256"]), tokens))
    if len(prompts) != 8:
        raise ValueError(f"expected exactly 8 frozen validation prompts, got {len(prompts)}")
    return prompts


def verify_checkpoint_payload(payload: dict[str, Any], spec: CheckpointSpec, actual_sha256: str) -> dict[str, Any]:
    if int(payload.get("update", -1)) != CHECKPOINT_UPDATE:
        raise ValueError(f"checkpoint update mismatch for {spec.path}")
    if int(payload.get("seed", -1)) != spec.seed:
        raise ValueError(f"checkpoint seed mismatch for {spec.path}")
    config = payload.get("config")
    if not isinstance(config, dict) or int(config.get("seed", -1)) != spec.seed:
        raise ValueError(f"checkpoint config seed mismatch for {spec.path}")
    expected_variant = f"shared_K{spec.k}" if spec.architecture == "R1" else f"ER32-K{spec.k}"
    if config.get("variant") != expected_variant:
        raise ValueError(f"checkpoint K/variant mismatch for {spec.path}")
    if spec.architecture == "R1":
        identity = payload.get("implementation_identity", {})
        if not isinstance(identity, dict) or identity.get("implementation") != "F":
            raise ValueError(f"R1/F implementation identity missing for {spec.path}")
    else:
        identity = payload.get("implementation_identity", {})
        if not isinstance(identity, dict) or identity.get("er32") != "omega_fast_er32.OmegaCoreLMFastER32":
            raise ValueError(f"ER32 implementation identity missing for {spec.path}")
        if int(config.get("experimental_seed", -1)) != spec.seed:
            raise ValueError(f"ER32 experimental seed mismatch for {spec.path}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"checkpoint model state missing for {spec.path}")
    return {
        "architecture": spec.architecture,
        "K": spec.k,
        "seed": spec.seed,
        "checkpoint_sha256": actual_sha256,
        "checkpoint_update": CHECKPOINT_UPDATE,
        "checkpoint_path": spec.path.as_posix(),
        "implementation_identity": payload.get("implementation_identity"),
    }


def load_checkpoint_model(spec: CheckpointSpec) -> tuple[torch.nn.Module, dict[str, Any]]:
    actual_sha256 = file_hash(spec.path)
    payload = torch.load(spec.path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be an object: {spec.path}")
    metadata = verify_checkpoint_payload(payload, spec, actual_sha256)
    rounds = spec.k
    if spec.architecture == "R1":
        model: torch.nn.Module = OmegaCoreLMFast(vocab_size=TOKENIZER_VOCAB, dimension=128, slots=8, rounds=rounds, variant="shared")
    else:
        model = OmegaCoreLMFastER32(vocab_size=TOKENIZER_VOCAB, rank=32, dimension=128, slots=8, rounds=rounds, variant="shared", implementation="efficient", _initialize_factors=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, metadata


def _ngram_frequencies(tokens: list[int], n: int) -> dict[str, int]:
    counts = collections.Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))
    return {" ".join(map(str, gram)): count for gram, count in sorted(counts.items()) if count > 1}


def _cycle_lengths(tokens: list[int]) -> list[int]:
    cycles: list[int] = []
    for period in range(1, min(8, len(tokens)) + 1):
        if len(tokens) >= 2 * period and tokens[-2 * period : -period] == tokens[-period:]:
            cycles.append(period)
    return cycles


def generation_metrics(tokens: list[int], eos_position: int | None) -> dict[str, Any]:
    total = len(tokens)
    ngrams = {str(n): _ngram_frequencies(tokens, n) for n in (2, 3, 4)}
    max_run = 0
    current = 0
    previous: int | None = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        max_run = max(max_run, current)
        previous = token
    return {
        "unique_token_proportion": len(set(tokens)) / total if total else 0.0,
        "distinct_1": len(set(tokens)) / total if total else 0.0,
        "distinct_2": len(set(zip(tokens, tokens[1:]))) / max(1, total - 1),
        "max_same_token_run": max_run,
        "repeated_ngram_frequencies": ngrams,
        "exact_cycle_lengths_1_to_8": _cycle_lengths(tokens),
        "eos_position": eos_position,
    }


def generate_one(model: torch.nn.Module, prompt: PromptSpec, tokenizer: Any, *, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    prompt_tensor = torch.tensor([list(prompt.token_ids)], dtype=torch.long)
    generated: list[int] = []
    eos_position: int | None = None
    with torch.inference_mode():
        state = model.initial_state(1, device=torch.device("cpu"))
        result = model.forward_window(prompt_tensor, state)
        state, logits = result[0], result[1]
        for index in range(max_new_tokens):
            token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(token)
            if token == EOS_TOKEN_ID:
                eos_position = index + 1
                break
            next_input = torch.tensor([[token]], dtype=torch.long)
            result = model.forward_window(next_input, state)
            state, logits = result[0], result[1]
    stop_reason = "EOS" if eos_position is not None else "MAX_LENGTH"
    decoded = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
    return {
        "prompt_document_id": prompt.document_id,
        "prompt_document_hash": prompt.document_hash,
        "prompt_token_ids": list(prompt.token_ids),
        "generated_token_ids": generated,
        "decoded_generation": decoded,
        "generated_length": len(generated),
        "stop_reason": stop_reason,
        "metrics": generation_metrics(generated, eos_position),
    }


def write_blind_outputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    identity: dict[str, Any] = {}
    lines = ["# Blind autoregressive generation review", "", "Annotate each opaque continuation without consulting model identity.", ""]
    for index, record in enumerate(records, start=1):
        opaque_id = f"generation-{index:02d}"
        identity[opaque_id] = {
            "architecture": record["architecture"],
            "K": record["K"],
            "seed": record["seed"],
            "prompt_document_id": record["prompt_document_id"],
            "prompt_document_hash": record["prompt_document_hash"],
        }
        lines.extend([f"## {opaque_id}", "", record["decoded_generation"], ""])
    (output_dir / "blind_review.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(output_dir / "blind_identity_map.json", identity)


def run_audit(checkpoints: list[CheckpointSpec], prompts: list[PromptSpec], output_dir: Path, tokenizer: Any, model_loader: Callable[[CheckpointSpec], tuple[torch.nn.Module, dict[str, Any]]] = load_checkpoint_model, *, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    configure_cpu()
    records: list[dict[str, Any]] = []
    for spec in checkpoints:
        model, metadata = model_loader(spec)
        for prompt in prompts:
            record = {**metadata, **generate_one(model, prompt, tokenizer, max_new_tokens=max_new_tokens)}
            records.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "omega-core-lm-0-autoregressive-generation-audit-v1",
        "audit_status": "AUDIT_COMPLETE" if len(records) == EXPECTED_GENERATIONS else None,
        "documentary_only": True,
        "quality_gate": None,
        "checkpoint_count": len(checkpoints),
        "prompt_count": len(prompts),
        "generation_count": len(records),
        "max_new_tokens": max_new_tokens,
        "eos_token_id": EOS_TOKEN_ID,
        "decoding": "greedy_argmax_free_running",
        "generations": records,
    }
    write_json(output_dir / "generation_results.json", report)
    write_blind_outputs(output_dir, records)
    return report


def load_real_context() -> tuple[list[PromptSpec], Any]:
    _, validation, _, validation_manifest, _ = _load_real_documents(pair_count=FULL_PAIRS)
    frozen_manifest_path = R1_DIR / "results" / "full_campaign" / "runs" / "shared_K1_seed_20260913" / "validation_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if validation_manifest.get("manifest_sha256") != frozen_manifest.get("manifest_sha256"):
        raise ValueError("real validation manifest differs from frozen R1 validation manifest")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    return prompt_specs(validation), tokenizer


def run_real(output_dir: Path) -> dict[str, Any]:
    configure_cpu()
    prompts, tokenizer = load_real_context()
    return run_audit(checkpoint_specs(), prompts, output_dir, tokenizer)


class TinyTokenizer:
    def decode(self, tokens: list[int], *, clean_up_tokenization_spaces: bool = False) -> str:
        return " ".join(str(token) for token in tokens)


class TinyAuditModel(torch.nn.Module):
    def __init__(self, sequence: list[int]) -> None:
        super().__init__()
        self.sequence = sequence
        self.position = 0

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, 1, 1, device=device)

    def forward_window(self, tokens: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del tokens
        vocab = 50257
        token = self.sequence[min(self.position, len(self.sequence) - 1)]
        self.position += 1
        logits = torch.full((1, 1, vocab), -1.0)
        logits[:, :, token] = 1.0
        return state, logits


def run_smoke(output_dir: Path) -> dict[str, Any]:
    prompts = [PromptSpec("synthetic-0", "synthetic-hash", tuple(range(PROMPT_LENGTH)))]
    specs = [CheckpointSpec("R1", 1, 1, Path("synthetic-r1"))]

    def loader(spec: CheckpointSpec) -> tuple[torch.nn.Module, dict[str, Any]]:
        return TinyAuditModel([11, 12, EOS_TOKEN_ID]), {"architecture": spec.architecture, "K": spec.k, "seed": spec.seed, "checkpoint_sha256": "synthetic", "checkpoint_update": CHECKPOINT_UPDATE}

    return run_audit(specs, prompts, output_dir, TinyTokenizer(), loader, max_new_tokens=8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-autoregressive-generation-audit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "autoregressive_generation_audit")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))
        return 0
    if not (args.full and args.confirm_autoregressive_generation_audit):
        parser.error("real audit requires --full --confirm-autoregressive-generation-audit")
    print(json.dumps(run_real(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
