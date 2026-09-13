"""Pure T4 structural-cause diagnostic; never trains or updates weights."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import execute_t4_nobypass1_three_active_roles as t4  # noqa: E402
from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7101, 7102, 7103, 7104, 7105)
OUTPUT_ROOT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "training"
MANIFEST_ROOT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "manifests"
DIAGNOSTIC_OUTPUT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "struct_cause_diagnostic.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_runtime() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.nn.Module, torch.nn.Module, torch.Tensor]:
    observations, labels = load_source()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    for parameter in supervisor.parameters():
        parameter.requires_grad_(False)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    return observations, labels, supervisor, executor, codebook


def load_encoder(seed: int, manifest: dict[str, Any]) -> tuple[t4.T4RelKeyEncoder, Path]:
    checkpoint = OUTPUT_ROOT / f"seed_{seed}" / "final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = t4.T4RelKeyEncoder(manifest, seed)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    return encoder, checkpoint


def part_a(seed: int, manifest: dict[str, Any], observations: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], supervisor: torch.nn.Module, executor: torch.nn.Module, codebook: torch.Tensor) -> dict[str, Any]:
    train_link_rows = sum(
        int(row.get("operator") == "LINK" or row.get("argument") == "LINK" or "LINK" in row.get("tokens", []))
        for row in manifest["train"]
    )
    encoder, checkpoint = load_encoder(seed, manifest)
    batch = t4.build_training_batch(encoder, manifest, labels)
    encoder.zero_grad(set_to_none=True)
    before = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    losses = t4.objective(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    losses["total"].backward()
    link_index = encoder.vocab.encode("LINK")
    gradient = encoder.embedding.weight.grad
    link_gradient_norm = 0.0 if gradient is None else float(gradient[link_index].norm().item())
    unchanged = all(torch.equal(before[name], value) for name, value in encoder.state_dict().items())
    rcsep_source = inspect.getsource(t4.rcsep3)
    probe_scores = {query: {source: torch.zeros(t4.VALUE_COUNT) for source in t4.ROLES} for query in t4.ROLES}
    _, terms = t4.rcsep3(probe_scores)
    return {
        "train_rows": len(manifest["train"]),
        "train_link_rows": train_link_rows,
        "train_link_rows_exact_zero": train_link_rows == 0,
        "rcsep3_source_sha256": sha256_file(SCRIPT_DIR / "execute_t4_nobypass1_three_active_roles.py"),
        "rcsep3_source_contains_LINK": "LINK" in rcsep_source,
        "rcsep3_terms": list(terms),
        "rcsep3_terms_exact_six_active_role_relations": tuple(terms) == ("F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A"),
        "complete_objective": {
            "batch_rows": 96,
            "losses": {name: float(value.detach().item()) for name, value in losses.items() if name in ("behavior", "ref", "sep", "total")},
            "link_embedding_index": link_index,
            "dL_total_d_e_LINK_norm": link_gradient_norm,
            "exact_zero": link_gradient_norm == 0.0,
            "weights_unchanged": unchanged,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    }


def structural_contexts(encoder: t4.T4RelKeyEncoder, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    arg00 = encoder.vocab.encode("ARG_00")
    link = encoder.vocab.encode("LINK")
    for source_role in t4.ROLES:
        operator = manifest["operator_for_role"][source_role]
        contexts.append({"kind": "START→OP", "context": f"START→{operator}", "tokens": [operator, "ARG_00"], "score_position": 0, "token_ids": [encoder.vocab.encode(operator), arg00]})
    for index in range(t4.VALUE_COUNT):
        argument = t4.arg_name(index)
        contexts.append({"kind": "ARG→LINK", "context": f"{argument}→LINK", "tokens": [argument, "LINK"], "score_position": 1, "token_ids": [encoder.vocab.encode(argument), link]})
    for target_role in t4.ROLES:
        operator = manifest["operator_for_role"][target_role]
        contexts.append({"kind": "LINK→OP", "context": f"LINK→{operator}", "tokens": ["LINK", operator], "score_position": 1, "token_ids": [link, encoder.vocab.encode(operator)]})
    assert len(contexts) == 38
    return contexts


def context_scores(encoder: t4.T4RelKeyEncoder, contexts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ids = torch.tensor([context["token_ids"] for context in contexts], dtype=torch.long)
    lengths = torch.full((len(contexts),), 2, dtype=torch.long)
    with torch.no_grad():
        details = encoder(ids, lengths)
    return {
        query: [
            {"kind": context["kind"], "context": context["context"], "score": float(details["scores"][query][index, context["score_position"]].item())}
            for index, context in enumerate(contexts)
        ]
        for query in t4.ROLES
    }


def part_b(seed: int, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder, checkpoint = load_encoder(seed, manifest)
    contexts = structural_contexts(encoder, manifest)
    candidates = context_scores(encoder, contexts)
    catalog = t4.catalog_scores(encoder, manifest)
    report: dict[str, Any] = {"context_count": len(contexts), "contexts": [context["context"] for context in contexts], "queries": {}, "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}}
    for query in t4.ROLES:
        self_scores = catalog[query][query]
        self_value, self_index = torch.min(self_scores, dim=0)
        winner = max(candidates[query], key=lambda item: (item["score"], item["context"]))
        report["queries"][query] = {
            "self_min": float(self_value.item()),
            "self_arg": t4.arg_name(int(self_index.item())),
            "struct_max": winner["score"],
            "struct_winner": {"kind": winner["kind"], "context": winner["context"]},
            "G_star_r_STRUCT": float(self_value.item() - winner["score"]),
        }
    return report, {query: candidates[query] for query in t4.ROLES}


def part_c(seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    encoder, checkpoint = load_encoder(seed, manifest)
    catalog = t4.catalog_scores(encoder, manifest)
    contexts = structural_contexts(encoder, manifest)
    candidates = context_scores(encoder, contexts)
    direct: dict[str, float] = {}
    for query in t4.ROLES:
        self_scores = catalog[query][query]
        for source in t4.ROLES:
            if source == query:
                continue
            direct[f"G_{query[:1]},{source[:1]}"] = float((self_scores.unsqueeze(1) - catalog[query][source].unsqueeze(0)).min().item())

    def scenario(name: str, selected: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        structural: dict[str, dict[str, Any]] = {}
        for query in t4.ROLES:
            self_min = float(catalog[query][query].min().item())
            winner = max(selected[query], key=lambda item: (item["score"], item["context"])) if selected[query] else None
            structural[f"G_{query[:1]},STRUCT"] = {"margin": None if winner is None else self_min - winner["score"], "winner": None if winner is None else {"kind": winner["kind"], "context": winner["context"]}}
        positive_direct = all(value > 0.0 for value in direct.values())
        positive_structural = all(item["margin"] is not None and item["margin"] > 0.0 for item in structural.values())
        return {"name": name, "structural": structural, "directed": direct, "D2_pass": positive_direct and positive_structural, "directed_all_positive": positive_direct, "structural_all_positive": positive_structural}

    without_link_positions = {query: [item for item in candidates[query] if item["kind"] != "ARG→LINK"] for query in t4.ROLES}
    without_all_structural = {query: [] for query in t4.ROLES}
    first = scenario("exclude ARG_i→LINK positions only", without_link_positions)
    second = scenario("exclude all 38 structural contexts", without_all_structural)
    second["D2_definition"] = "six directed certificates only; structural certificate omitted"
    second["D2_pass"] = second["directed_all_positive"]
    return {"seed": seed, "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}, "scenarios": [first, second], "d3_executed": False}


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    artifact["artifact_self_hash"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifests = {seed: t4.load_manifest(seed)[0] for seed in SEEDS}
    observations, labels, supervisor, executor, codebook = load_frozen_runtime()
    result: dict[str, Any] = {
        "status": "completed",
        "task": "T4-STRUCT-CAUSE-ALG",
        "training_performed": False,
        "weight_updates_performed": False,
        "d3_executed": False,
        "part_a": {str(seed): part_a(seed, manifests[seed], observations, labels, supervisor, executor, codebook) for seed in SEEDS},
        "part_b": {},
        "part_c": part_c(7102, manifests[7102]),
        "provenance": {"runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}, "executor_checkpoint": {"path": str(BASE_CHECKPOINT), "sha256": sha256_file(BASE_CHECKPOINT)}, "supervisor_checkpoint": {"path": str(CTRL7_CHECKPOINT), "sha256": sha256_file(CTRL7_CHECKPOINT)}},
    }
    for seed in SEEDS:
        report, _ = part_b(seed, manifests[seed])
        result["part_b"][str(seed)] = report
    digest = write_self_hashed(DIAGNOSTIC_OUTPUT, result)
    print(json.dumps({"status": "completed", "artifact": str(DIAGNOSTIC_OUTPUT), "artifact_self_hash": digest, "training_performed": False, "weight_updates_performed": False, "d3_executed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
