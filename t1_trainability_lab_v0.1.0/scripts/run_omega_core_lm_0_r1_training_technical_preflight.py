"""OMEGA CORE-LM-0 R1 technical training preflight.

This is a bounded engineering measurement, not the 20k-step scientific pilot.
It downloads only the pinned GPT-2 teacher/tokenizer and WikiText-2 train split,
then runs at most 21 optimizer updates (7 per student variant).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omega_core_lm_0_design import RMSNorm, WorkspaceUpdateBlock  # noqa: E402


PREVIOUS_OUTPUT = ROOT / "campaign" / "omega_core_lm_0_r1_training_technical_preflight" / "training_technical_preflight.json"
OUTPUT = ROOT / "campaign" / "omega_core_lm_0_r1_training_technical_preflight" / "training_technical_preflight_v2.json"
MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_VOCAB = 50257
SEED = 20260913
TEMPERATURE = 2.0
BASE_LR = 3e-4
MAX_UPDATES_PER_VARIANT = 7
MAX_TOTAL_UPDATES = 21


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def tensor_hash(value: Tensor) -> str:
    return sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_self_hashed(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = sha256_text(json.dumps(unsigned, indent=2, sort_keys=True) + "\n")
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return digest, sha256_bytes(encoded)


def memory_snapshot() -> dict[str, int | None]:
    process = psutil.Process()
    result: dict[str, int | None] = {
        "rss_bytes": process.memory_info().rss,
        "available_bytes": psutil.virtual_memory().available,
        "gpu_allocated_bytes": None,
        "gpu_reserved_bytes": None,
    }
    if torch.cuda.is_available():
        result["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
        result["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
    return result


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class OmegaCoreLM0R1Technical(nn.Module):
    """R1 student adapter with device-safe state and explicit window forward."""

    def __init__(self, *, vocab_size: int, dimension: int = 128, slots: int = 8, rounds: int = 4, variant: str = "shared") -> None:
        super().__init__()
        if variant not in {"shared", "untied"}:
            raise ValueError(variant)
        self.vocab_size = vocab_size
        self.dimension = dimension
        self.slots = slots
        self.rounds = rounds
        self.variant = variant
        self.embedding = nn.Embedding(vocab_size, dimension)
        self.prelude = nn.Linear(2 * dimension, slots * dimension)
        self.prelude_norm = RMSNorm(slots * dimension)
        block_count = 1 if variant == "shared" else rounds
        self.blocks = nn.ModuleList(WorkspaceUpdateBlock(dimension, slots) for _ in range(block_count))
        self.depth_embedding = nn.Embedding(rounds, dimension)
        self.gate_logits = nn.Parameter(torch.full((rounds, dimension), -2.1972245773362196))
        self.readout_norm = RMSNorm(slots * dimension)
        self.output_projection = nn.Linear(slots * dimension, dimension, bias=False)

    def initial_state(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dimension, device=device)

    def prelude_step(self, token: Tensor, previous_state: Tensor) -> Tensor:
        token_state = self.embedding(token)
        summary = previous_state.mean(dim=1)
        write = self.prelude(torch.cat((token_state, summary), dim=-1))
        anchor = self.prelude_norm(previous_state.flatten(start_dim=1) + write)
        return anchor.view(token.shape[0], self.slots, self.dimension)

    def update_step(self, anchor: Tensor) -> Tensor:
        state = anchor
        for round_index in range(self.rounds):
            block = self.blocks[0] if self.variant == "shared" else self.blocks[round_index]
            state = block(state, anchor, self.depth_embedding.weight[round_index], torch.sigmoid(self.gate_logits[round_index]))
        return state

    def readout_from_state(self, state: Tensor) -> Tensor:
        normalized = self.readout_norm(state.flatten(start_dim=1))
        projected = self.output_projection(normalized)
        return projected @ self.embedding.weight.transpose(0, 1) / math.sqrt(self.dimension)

    def forward_token(self, token: Tensor, previous_state: Tensor) -> tuple[Tensor, Tensor]:
        anchor = self.prelude_step(token, previous_state)
        final_state = self.update_step(anchor)
        return final_state, self.readout_from_state(final_state)

    def forward_window(self, tokens: Tensor, previous_state: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        state = previous_state
        outputs: list[Tensor] = []
        for position in range(tokens.shape[1]):
            next_state, logits = self.forward_token(tokens[:, position], state)
            if valid_mask is None:
                state = next_state
            else:
                active = valid_mask[:, position].view(-1, 1, 1)
                state = torch.where(active, next_state, state)
            outputs.append(logits)
        return state, torch.stack(outputs, dim=1)


def model_parameter_breakdown(model: nn.Module) -> dict[str, int]:
    categories = {"P_shell": 0, "P_core": 0, "P_modulation": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("blocks."):
            categories["P_core"] += parameter.numel()
        elif name.startswith("depth_embedding") or name.startswith("gate_logits"):
            categories["P_modulation"] += parameter.numel()
        else:
            categories["P_shell"] += parameter.numel()
    categories["P_total"] = sum(categories.values())
    return categories


def masked_mean(value: Tensor, valid_mask: Tensor) -> Tensor:
    weights = valid_mask.to(dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, targets: Tensor, valid_mask: Tensor) -> dict[str, Tensor]:
    flat_targets = targets.reshape(-1)
    ce_per_token = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), flat_targets, reduction="none").view_as(targets)
    student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * TEMPERATURE**2
    ce = masked_mean(ce_per_token, valid_mask)
    kl = masked_mean(kl_per_token, valid_mask)
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}


def is_level_one_header(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= =") and not stripped.endswith("= =")


def reconstruct_documents(dataset: Any) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    current_start: int | None = None
    current_lines: list[str] = []
    current_header = ""
    for row_index, row in enumerate(dataset):
        text = str(row["text"])
        if is_level_one_header(text):
            if current_start is not None:
                documents.append({"row_range": [current_start, row_index], "header": current_header, "text": "\n".join(current_lines)})
            current_start = row_index
            current_header = text.strip()
            current_lines = [text]
        elif current_start is not None:
            current_lines.append(text)
    if current_start is not None:
        documents.append({"row_range": [current_start, len(dataset)], "header": current_header, "text": "\n".join(current_lines)})
    return documents


def select_documents(dataset: Any, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    documents = reconstruct_documents(dataset)
    scanned = len(documents)
    for document_index, document in enumerate(documents):
        text = document["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 513:
            continue
        chosen = token_ids[:513]
        selected.append(
            {
                "document_index": document_index,
                "row_range": document["row_range"],
                "header": document["header"],
                "token_count": len(token_ids),
                "selected_token_count": len(chosen),
                "text_sha256": sha256_text(text),
                "token_sha256": sha256_bytes(bytes().join(int(token).to_bytes(4, "little") for token in chosen)),
                "tokens": chosen,
            }
        )
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise RuntimeError(f"only found {len(selected)} reconstructed train documents with >=513 tokens after scanning {scanned} documents")
    manifest = {
        "selection_rule": "reconstruct documents from level-one '= Title =' headers; select first 8 reconstructed documents with >=513 GPT-2 tokens; retain first 513 tokens",
        "document_count_scanned": scanned,
        "header_rule": "level-one header is '= Title ='; '= = Subsection = =' and deeper headings do not start new documents",
        "selected_documents": [
            {key: value for key, value in item.items() if key != "tokens"}
            for item in selected
        ],
        "windows": [
            {"window": 0, "input_range": [0, 256], "target_range": [1, 257], "teacher_context_range": [0, 256]},
            {"window": 1, "input_range": [256, 512], "target_range": [257, 513], "teacher_context_range": [0, 512]},
        ],
    }
    return selected, manifest


def teacher_window_logits(teacher: nn.Module, tokens: Tensor, window: int) -> Tensor:
    if window == 0:
        context = tokens[:, :256]
        start = 0
    else:
        context = tokens[:, :512]
        start = 256
    with torch.no_grad():
        output = teacher(input_ids=context)
    return output.logits[:, start : start + 256].detach()


def student_window_loss(student: OmegaCoreLM0R1Technical, teacher: nn.Module, token_ids: list[int], *, length: int, carry_state: bool = False) -> tuple[dict[str, Tensor], Tensor]:
    device = next(student.parameters()).device
    source = torch.tensor(token_ids[: length + 1], dtype=torch.long, device=device).unsqueeze(0)
    state = student.initial_state(1, device=device)
    input_ids = source[:, :length]
    targets = source[:, 1 : length + 1]
    student_state, student_logits = student.forward_window(input_ids, state)
    teacher_logits = teacher_window_logits(teacher, source, 0)[:, :length]
    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    losses = distillation_loss(student_logits, teacher_logits, targets, valid_mask)
    return losses, student_state.detach() if carry_state else student_state


def full_article_loss(student: OmegaCoreLM0R1Technical, teacher: nn.Module, token_ids: list[int]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    device = next(student.parameters()).device
    source = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    state = student.initial_state(1, device=device)
    losses: list[dict[str, Tensor]] = []
    timing: dict[str, float] = {}
    for window in range(2):
        input_ids = source[:, window * 256 : (window + 1) * 256]
        targets = source[:, window * 256 + 1 : (window + 1) * 256 + 1]
        started = time.perf_counter()
        state, student_logits = student.forward_window(input_ids, state.detach() if window else state)
        synchronize()
        timing[f"student_window_{window}_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        teacher_logits = teacher_window_logits(teacher, source, window)
        synchronize()
        timing[f"teacher_window_{window}_seconds"] = time.perf_counter() - started
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        losses.append(distillation_loss(student_logits, teacher_logits, targets, valid_mask))
    combined = {key: torch.stack([item[key] for item in losses]).mean() for key in losses[0]}
    return combined, {"timing": timing, "valid_tokens": 512, "state_hash": tensor_hash(state)}


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def run_variant(student: OmegaCoreLM0R1Technical, teacher: nn.Module, selected: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(student.parameters(), lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    student.train()
    integrity_before_hash = parameter_hash(student)
    integrity_losses, _ = student_window_loss(student, teacher, selected[0]["tokens"], length=32)
    integrity_losses["total"].backward()
    integrity_grad_norm = grad_norm(student)
    integrity_finite = all(torch.isfinite(value).item() for value in integrity_losses.values()) and math.isfinite(integrity_grad_norm)
    student.zero_grad(set_to_none=True)
    integrity_after_hash = parameter_hash(student)

    mid_before_hash = integrity_after_hash
    mid_losses, _ = student_window_loss(student, teacher, selected[0]["tokens"], length=128)
    mid_losses["total"].backward()
    mid_grad_norm = grad_norm(student)
    mid_finite = all(torch.isfinite(value).item() for value in mid_losses.values()) and math.isfinite(mid_grad_norm)
    student.zero_grad(set_to_none=True)
    mid_after_hash = parameter_hash(student)

    updates: list[dict[str, Any]] = []
    before_hash = parameter_hash(student)
    for update_index in range(1, MAX_UPDATES_PER_VARIANT + 1):
        optimizer.zero_grad(set_to_none=True)
        article = selected[(update_index - 1) % len(selected)]
        started = time.perf_counter()
        losses, forward_meta = full_article_loss(student, teacher, article["tokens"])
        losses["total"].backward()
        pre_clip = grad_norm(student)
        returned_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item())
        post_clip = grad_norm(student)
        clipping_intervened = pre_clip > 1.0
        warmup_lr = BASE_LR
        for group in optimizer.param_groups:
            group["lr"] = warmup_lr
        optimizer.step()
        synchronize()
        elapsed = time.perf_counter() - started
        updates.append(
            {
                "update": update_index,
                "document_index": article["document_index"],
                "document_row_range": article["row_range"],
                "document_header": article["header"],
                "phase": "warmup" if update_index <= 2 else "measured",
                "learning_rate": warmup_lr,
                "ce": float(losses["ce"].detach().item()),
                "kl": float(losses["kl"].detach().item()),
                "total_loss": float(losses["total"].detach().item()),
                "pre_clip_grad_norm": pre_clip,
                "clip_returned_norm": returned_norm,
                "post_clip_grad_norm": post_clip,
                "clipping_intervened": clipping_intervened,
                "finite": all(torch.isfinite(losses[key]).item() for key in ("ce", "kl", "total")) and math.isfinite(pre_clip) and math.isfinite(post_clip),
                "parameter_hash_after": parameter_hash(student),
                "elapsed_seconds": elapsed,
                "valid_tokens_per_second": 512.0 / elapsed if elapsed > 0 else None,
                "memory_after": memory_snapshot(),
                **forward_meta,
            }
        )
    after_hash = parameter_hash(student)
    return {
        "variant": variant,
        "parameter_breakdown": model_parameter_breakdown(student),
        "fp32_parameter_bytes": model_parameter_breakdown(student)["P_total"] * 4,
        "integrity_no_update": {"length": 32, "loss": float(integrity_losses["total"].detach().item()), "grad_norm": integrity_grad_norm, "finite": integrity_finite, "optimizer_step": False, "parameters_unchanged": integrity_before_hash == integrity_after_hash},
        "mid_scale_no_update": {"length": 128, "loss": float(mid_losses["total"].detach().item()), "grad_norm": mid_grad_norm, "finite": mid_finite, "optimizer_step": False, "parameters_unchanged": mid_before_hash == mid_after_hash},
        "nominal": {"warmup_updates": 2, "measured_updates": 5, "updates": updates},
        "parameter_hash_before_nominal": before_hash,
        "parameter_hash_after_nominal": after_hash,
        "parameters_changed": before_hash != after_hash,
        "integrity_parameters_unchanged": integrity_before_hash == integrity_after_hash,
        "mid_parameters_unchanged": mid_before_hash == mid_after_hash,
        "all_updates_finite": all(item["finite"] for item in updates),
        "mid_integrity_finite": mid_finite and integrity_finite,
        "peak_rss_bytes": max(item["memory_after"]["rss_bytes"] for item in updates),
    }


def main() -> int:
    torch.manual_seed(SEED)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": None,
        "datasets": None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "memory_initial": memory_snapshot(),
    }
    result: dict[str, Any] = {
        "schema": "omega-core-lm-0-r1-training-technical-preflight-v2",
        "status": "blocked",
        "classification": "BLOCKED_BEFORE_EXTERNAL_LOAD",
        "technical_optimizer_updates": 0,
        "scientific_pilot_started": False,
        "full_training_campaign_started": False,
        "environment": environment,
        "configuration": {
            "seed": SEED,
            "device": str(device),
            "dtype": "float32",
            "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "frozen": True, "eval": True, "no_grad": True, "context_limit": 1024},
            "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION, "type": "GPT-2 byte-level BPE", "vocab_size_expected": TOKENIZER_VOCAB},
            "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "split_loaded": "train", "local_repartition": False},
            "variants": ["shared_K1", "shared_K4", "untied_K4"],
            "loss": "0.5*CE + 0.5*T^2*KL; T=2; KL sum over vocabulary and mean over valid positions",
        "optimizer": {"type": "AdamW", "lr": BASE_LR, "learning_rate_policy": "constant 3e-4 for all 7 updates; first 2 are discardable budget warmups, not LR ramp steps", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "clip_norm": 1.0, "warmup_updates": 2, "measured_updates": 5},
            "update_budget": {"per_variant": MAX_UPDATES_PER_VARIANT, "total": MAX_TOTAL_UPDATES},
            "licenses": {"teacher": "Apache-2.0", "dataset": "WikiText metadata reports CC BY-SA 3.0; source/ficha discrepancy remains documented, not resolved"},
        },
        "selection_manifest": None,
        "corrections_from_v1": {"learning_rate": "v1 ramped the first two updates; v2 fixes LR to constant 3e-4 for all updates", "document_selection": "v1 treated dataset rows as documents; v2 reconstructs documents from level-one '= Title =' headers and records [start,end) row ranges", "previous_artifact": str(PREVIOUS_OUTPUT.relative_to(ROOT)).replace("\\", "/") if PREVIOUS_OUTPUT.is_file() else None, "previous_artifact_sha256": hashlib.sha256(PREVIOUS_OUTPUT.read_bytes()).hexdigest() if PREVIOUS_OUTPUT.is_file() else None},
        "variants": [],
        "errors": [],
    }
    try:
        import datasets
        import transformers
        from datasets import load_dataset, load_dataset_builder
        from transformers import AutoModelForCausalLM, AutoTokenizer

        environment["transformers"] = transformers.__version__
        environment["datasets"] = datasets.__version__
        builder = load_dataset_builder(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION)
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if len(tokenizer) != TOKENIZER_VOCAB:
            raise RuntimeError(f"tokenizer vocab mismatch: expected {TOKENIZER_VOCAB}, got {len(tokenizer)}")
        if getattr(teacher.config, "n_positions", 1024) > 1024:
            raise RuntimeError(f"teacher context configuration unexpectedly exceeds R1 limit: {teacher.config.n_positions}")
        selected, manifest = select_documents(dataset, tokenizer)
        manifest["dataset_builder"] = {"description_sha256": sha256_text(builder.info.description or ""), "features": str(builder.info.features)}
        result["selection_manifest"] = manifest
        for variant, rounds, model_variant in (("shared_K1", 1, "shared"), ("shared_K4", 4, "shared"), ("untied_K4", 4, "untied")):
            student = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, rounds=rounds, variant=model_variant).to(device)
            variant_result = run_variant(student, teacher, selected, variant)
            result["variants"].append(variant_result)
            result["technical_optimizer_updates"] += MAX_UPDATES_PER_VARIANT
        result["status"] = "passed"
        result["classification"] = "PASS_TECHNICAL_PREFLIGHT"
    except Exception as exc:  # bounded artifact records failure without retrying or changing scope
        result["errors"].append({"type": type(exc).__name__, "message": str(exc), "stage": "external_load_or_bounded_run"})
        result["status"] = "blocked" if not result["variants"] else "partial"
        result["classification"] = "BLOCKED_EXTERNAL_LOAD_OR_RUN" if not result["variants"] else "PARTIAL_BOUNDED_RUN"
    result["elapsed_seconds"] = time.perf_counter() - started
    result["environment"]["memory_final"] = memory_snapshot()
    result["update_budget_respected"] = result["technical_optimizer_updates"] <= MAX_TOTAL_UPDATES
    digest, file_sha = write_self_hashed(OUTPUT, result)
    print(json.dumps({"status": result["status"], "classification": result["classification"], "artifact": OUTPUT.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": file_sha, "technical_optimizer_updates": result["technical_optimizer_updates"], "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
