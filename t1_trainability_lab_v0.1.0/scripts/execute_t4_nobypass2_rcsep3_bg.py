"""Execute T4-NOBYPASS-2 with local structural background negatives."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import execute_t4_nobypass1_three_active_roles as t4  # noqa: E402
from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7101, 7102, 7103, 7104, 7105)
ROLES = ("FLOOR", "AVOID", "MATCH")
UPDATES = 5000
BACKGROUND_COUNT = 38
BACKGROUND_SPEC = {"START→OP": 3, "ARG→LINK": 32, "LINK→OP": 3}
TERM_ORDER = ("F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A", "F<-BG", "A<-BG", "M<-BG")
OUTPUT_ROOT = ROOT / "campaign" / "t4_nobypass2_rcsep3_bg" / "training"
MANIFEST_ROOT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "manifests"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_bytes(payload)
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    return digest


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T4-nobypass1-three-active-roles-manifest-v1":
        raise ValueError(f"unexpected manifest schema for {seed}")
    if len(manifest.get("train", [])) != 96 or len(manifest.get("test", [])) != 5952:
        raise ValueError(f"manifest cardinality mismatch for {seed}")
    if manifest.get("multi_clause_train") != 0 or manifest.get("joint_train_examples") != 0:
        raise ValueError(f"manifest admits multi-clause training for {seed}")
    return manifest, path


def background_contexts(encoder: t4.T4RelKeyEncoder, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    arg00 = encoder.vocab.encode("ARG_00")
    link = encoder.vocab.encode("LINK")
    for role in ROLES:
        operator = manifest["operator_for_role"][role]
        contexts.append({"kind": "START→OP", "context": f"START→{operator}", "token_ids": [encoder.vocab.encode(operator), arg00], "score_position": 0})
    for index in range(VALUE_COUNT):
        argument = t4.arg_name(index)
        contexts.append({"kind": "ARG→LINK", "context": f"{argument}→LINK", "token_ids": [encoder.vocab.encode(argument), link], "score_position": 1})
    for role in ROLES:
        operator = manifest["operator_for_role"][role]
        contexts.append({"kind": "LINK→OP", "context": f"LINK→{operator}", "token_ids": [link, encoder.vocab.encode(operator)], "score_position": 1})
    if len(contexts) != BACKGROUND_COUNT:
        raise AssertionError(f"background context count: {len(contexts)}")
    return contexts


def background_scores(encoder: t4.T4RelKeyEncoder, manifest: dict[str, Any]) -> dict[str, Tensor]:
    contexts = background_contexts(encoder, manifest)
    token_ids = torch.tensor([context["token_ids"] for context in contexts], dtype=torch.long)
    lengths = torch.full((BACKGROUND_COUNT,), 2, dtype=torch.long)
    details = encoder(token_ids, lengths)
    return {query: torch.stack(tuple(details["scores"][query][index, context["score_position"]] for index, context in enumerate(contexts))) for query in ROLES}


def separation_with_background(encoder: t4.T4RelKeyEncoder, manifest: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
    catalog = t4.catalog_scores(encoder, manifest)
    backgrounds = background_scores(encoder, manifest)
    terms: dict[str, Tensor] = {}
    for query in ROLES:
        self_scores = catalog[query][query]
        for source in ROLES:
            if source == query:
                continue
            relation = f"{query[:1]}<-{source[:1]}"
            terms[relation] = F.softplus(catalog[query][source].unsqueeze(0) - self_scores.unsqueeze(1)).mean()
    for query in ROLES:
        self_scores = catalog[query][query]
        background_matrix = F.softplus(backgrounds[query].unsqueeze(0) - self_scores.unsqueeze(1))
        if background_matrix.shape != (VALUE_COUNT, BACKGROUND_COUNT):
            raise AssertionError(f"unexpected BG matrix shape: {background_matrix.shape}")
        terms[f"{query[:1]}<-BG"] = background_matrix.mean()
    if tuple(terms) != TERM_ORDER:
        raise AssertionError(f"unexpected separation term order: {tuple(terms)}")
    return torch.stack(tuple(terms.values())).mean(), terms


def objective_with_background(encoder: t4.T4RelKeyEncoder, supervisor: nn.Module, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], batch: dict[str, Any], observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
    details = encoder(batch["token_ids"], batch["lengths"])
    values = details["values"][:, 1]
    floor = torch.tensor([index for index, role in enumerate(batch["roles"]) if role == "FLOOR"], dtype=torch.long)
    avoid = torch.tensor([index for index, role in enumerate(batch["roles"]) if role == "AVOID"], dtype=torch.long)
    match = torch.tensor([index for index, role in enumerate(batch["roles"]) if role == "MATCH"], dtype=torch.long)
    behavior_indices = torch.cat((floor, avoid))
    behavior = F.cross_entropy(supervisor(observations["features"][batch["source_rows"][behavior_indices]], values[behavior_indices]), labels["action"][batch["source_rows"][behavior_indices]])
    ref_losses = []
    for indices in (floor, avoid, match):
        logits = executor.register_decoder(torch.cat((values[indices], torch.zeros_like(values[indices])), dim=-1), codebook)
        ref_losses.append(F.cross_entropy(logits, batch["targets"][indices]))
    reference = torch.stack(ref_losses).mean()
    separation, terms = separation_with_background(encoder, manifest)
    return {"behavior": behavior, "ref": reference, "sep": separation, "total": behavior + reference + separation, **{f"sep_{name}": value for name, value in terms.items()}}


def bg_certificates(encoder: t4.T4RelKeyEncoder, manifest: dict[str, Any]) -> dict[str, Any]:
    catalog = t4.catalog_scores(encoder, manifest)
    contexts = background_contexts(encoder, manifest)
    token_ids = torch.tensor([context["token_ids"] for context in contexts], dtype=torch.long)
    lengths = torch.full((BACKGROUND_COUNT,), 2, dtype=torch.long)
    with torch.no_grad():
        details = encoder(token_ids, lengths)
    certificates: dict[str, Any] = {}
    for query in ROLES:
        self_scores = catalog[query][query]
        for source in ROLES:
            if source == query:
                continue
            matrix = self_scores.unsqueeze(1) - catalog[query][source].unsqueeze(0)
            value, flat = matrix.reshape(-1).min(0)
            i, j = divmod(int(flat), VALUE_COUNT)
            certificates[f"G_{query[:1]},{source[:1]}"] = {"margin": finite(value.item()), "positive": bool(value > 0), "self_arg_index": i, "cross_role": source, "cross_arg_index": j}
        scores = torch.stack(tuple(details["scores"][query][index, context["score_position"]] for index, context in enumerate(contexts)))
        matrix = self_scores.unsqueeze(1) - scores.unsqueeze(0)
        value, flat = matrix.reshape(-1).min(0)
        i, j = divmod(int(flat), BACKGROUND_COUNT)
        certificates[f"G_{query[:1]},BG"] = {"margin": finite(value.item()), "positive": bool(value > 0), "self_arg_index": i, "background_index": j, "background_context": contexts[j]["context"], "background_kind": contexts[j]["kind"]}
        arg_link_indices = [index for index, context in enumerate(contexts) if context["kind"] == "ARG→LINK"]
        arg_link_scores = scores[arg_link_indices]
        value, flat = (self_scores.unsqueeze(1) - arg_link_scores.unsqueeze(0)).reshape(-1).min(0)
        i, j = divmod(int(flat), len(arg_link_indices))
        certificates[f"G_{query[:1]},ARG→LINK"] = {"margin": finite(value.item()), "positive": bool(value > 0), "self_arg_index": i, "background_index": arg_link_indices[j], "background_context": contexts[arg_link_indices[j]]["context"]}
    formal_names = tuple(f"G_{query[:1]},{source[:1]}" for query in ROLES for source in ROLES if source != query) + tuple(f"G_{query[:1]},BG" for query in ROLES)
    return {"formal": {name: certificates[name] for name in formal_names}, "arg_link": {f"G_{query[:1]},ARG→LINK": certificates[f"G_{query[:1]},ARG→LINK"] for query in ROLES}, "all_formal_positive": all(certificates[name]["positive"] for name in formal_names), "six_role_and_three_bg": len(formal_names) == 9}


def link_gradient_norm(encoder: t4.T4RelKeyEncoder, supervisor: nn.Module, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], batch: dict[str, Any], observations: dict[str, Tensor], labels: dict[str, Tensor]) -> float:
    encoder.zero_grad(set_to_none=True)
    losses = objective_with_background(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    losses["total"].backward()
    link_index = encoder.vocab.encode("LINK")
    gradient = encoder.embedding.weight.grad
    return 0.0 if gradient is None else float(gradient[link_index].norm().item())


def train_seed(seed: int, manifest: dict[str, Any], manifest_path: Path, observations: dict[str, Tensor], labels: dict[str, Tensor], supervisor: nn.Module, executor: nn.Module, codebook: Tensor, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("final.pt", "results.json")):
        raise FileExistsError(f"refusing to overwrite T4-NOBYPASS-2 seed output {destination}")
    encoder = t4.T4RelKeyEncoder(manifest, seed)
    batch = t4.build_training_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last: dict[str, float] = {}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = objective_with_background(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
        last = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * (step - 1) / (UPDATES - 1)
    encoder.eval()
    with torch.no_grad():
        final_losses_tensor = objective_with_background(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    final_losses = {name: finite(final_losses_tensor[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    torch.save({"encoder": encoder.state_dict(), "seed": seed, "updates": UPDATES, "manifest_sha256": sha256_file(manifest_path), "fresh_init": True, "prior_t2_t3_checkpoint_loaded": False, "shared_w_v": True, "query_roles": list(ROLES), "objective": "L_behavior^(F,A) + L_ref^(F,A,M) + L_sep^(3+BG)", "L_sep_coefficient": 1.0, "multi_clause_train": 0, "local_background_negative_contexts": True, "test_rows_used": 0}, checkpoint)
    atomic = t4.atomic_gate(encoder, executor, codebook, manifest)
    certificates = bg_certificates(encoder, manifest)
    gradient = link_gradient_norm(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    result = {"status": "trained", "seed": seed, "updates": UPDATES, "fresh_init": {"constructor": "T4RelKeyEncoder(manifest_710X_v1, seed)", "seed": seed, "prior_t2_t3_checkpoint_loaded": False, "shared_w_v": True}, "training": {"batch_rows": {role: 32 for role in ROLES} | {"total": 96}, "multi_clause_train": 0, "joint_train_examples": 0, "local_background_negative_contexts": True, "test_rows_used": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A) + L_ref^(F,A,M) + L_sep^(3+BG)", "L_sep_coefficient": 1.0, "last_update_losses": last, "final_losses": final_losses, "background_spec": BACKGROUND_SPEC, "background_count": BACKGROUND_COUNT}, "architecture": {"name": "RELKEY", "d_model": t4.DMODEL, "queries": list(ROLES), "address_key": t4.T4RelKeyEncoder.address_key_formula, "scores": t4.T4RelKeyEncoder.score_formula, "selected_state": t4.T4RelKeyEncoder.selected_state_formula, "forward_inputs": ["token_ids", "lengths"], "shared_w_v": True, "gate": False, "stage_b": False, "soft_mixture": False, "hard_pointer": "unchanged pure argmax over valid positions"}, "atomic": atomic, "certificates": certificates, "gradient_sanity": {"dL_total_d_e_LINK_norm": gradient, "strictly_positive": gradient > 0.0}, "manifest": source_record(manifest_path), "checkpoint": source_record(checkpoint)}
    write_self_hashed(destination / "results.json", result)
    return {"encoder": encoder, "result": result}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty T4-NOBYPASS-2 output root {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifests = {seed: load_manifest(seed) for seed in SEEDS}
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
    per_seed: list[dict[str, Any]] = []
    encoders: dict[int, t4.T4RelKeyEncoder] = {}
    for seed in SEEDS:
        destination = OUTPUT_ROOT / f"seed_{seed}"
        try:
            manifest, manifest_path = manifests[seed]
            trained = train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, destination)
            item = trained["result"]
            encoders[seed] = trained["encoder"]
            per_seed.append(item)
        except Exception as error:
            failure = {"status": "execution_failed", "seed": seed, "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, "development_gate": {"D1": False, "D2": False, "D3": False}, "gradient_sanity": {"strictly_positive": False}}
            write_self_hashed(destination / "results.json", failure)
            per_seed.append(failure)
    for item in per_seed:
        if item.get("status") == "trained":
            item["development_gate"] = {"D1": bool(item["atomic"]["pass"]), "D2": bool(item["atomic"]["pass"] and item["certificates"]["all_formal_positive"] and item["certificates"]["six_role_and_three_bg"] and item["gradient_sanity"]["strictly_positive"]), "D3_eligible": False}
    all_d1 = len(per_seed) == 5 and all(item.get("development_gate", {}).get("D1") is True for item in per_seed)
    all_d2 = all_d1 and all(item.get("development_gate", {}).get("D2") is True for item in per_seed)
    all_gradient = len(per_seed) == 5 and all(item.get("gradient_sanity", {}).get("strictly_positive") is True for item in per_seed)
    all_d3 = False
    for item in per_seed:
        seed = item["seed"]
        manifest, _ = manifests[seed]
        if all_d1 and all_d2 and all_gradient and seed in encoders:
            item["development_gate"]["D3_eligible"] = True
            item["test"] = t4.evaluate_test(encoders[seed], executor, codebook, manifest, seed)
            item["development_gate"]["D3"] = bool(item["test"]["cases"] == 5952 and item["test"]["all_required_outcomes"] and item["test"]["all_nine_margins_positive"] and item["test"]["six_order_outcome_equality"]["equal"])
        else:
            item["test"] = {"status": "HALTED_BY_D1_D2_OR_GRADIENT", "reason": "D1, D2, or gradient sanity failed for at least one seed; D3 remained sealed", "cases": 0}
            item["development_gate"]["D3"] = False
        write_self_hashed(OUTPUT_ROOT / f"seed_{seed}" / "results.json", item)
    all_d3 = all_d1 and all_d2 and all_gradient and all(item.get("development_gate", {}).get("D3") is True for item in per_seed)
    consolidated = {"status": "completed", "classification": "T4-NOBYPASS-2 CLOSURE/PASS" if all_d1 and all_d2 and all_d3 else "T4-NOBYPASS-2 DEVELOPMENT: VALID FAIL", "task": "T4-NOBYPASS-2-RCSEP3-BG", "executive_summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "D1_all_seeds": all_d1, "D2_all_seeds": all_d2, "gradient_sanity_all_seeds": all_gradient, "D3_opened": all_d1 and all_d2 and all_gradient, "D3_all_seeds": all_d3, "test_cases_evaluated": sum(item.get("test", {}).get("cases", 0) for item in per_seed), "no_test_training": True, "no_sequential_stopping": True, "no_tuning_between_seeds": True}, "recipe": {"fresh_init": True, "prior_t2_t3_checkpoint_loaded": False, "architecture": t4.T4RelKeyEncoder.address_key_formula, "three_queries": list(ROLES), "shared_w_v": True, "objective": "L_behavior^(F,A) + L_ref^(F,A,M) + L_sep^(3+BG)", "L_sep_relations": list(TERM_ORDER), "background_spec": BACKGROUND_SPEC, "background_count": BACKGROUND_COUNT, "local_background_negative_contexts": True, "L_sep_coefficient": 1.0, "updates": UPDATES, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5}, "batch_rows": {"FLOOR": 32, "AVOID": 32, "MATCH": 32, "total": 96}, "multi_clause_train": 0, "joint_train_examples": 0, "test_rows_used": 0, "stage_b": False, "gate": False, "hard_pointer": "unchanged pure argmax over valid positions"}, "provenance": {"executor": {"identity": "approved C1JointModel via ctrl2_common.load_executor", "path": str(BASE_CHECKPOINT), "sha256": sha256_file(BASE_CHECKPOINT)}, "supervisor": {"identity": "frozen CTRL7 LatentConditionedSupervisor", "path": str(CTRL7_CHECKPOINT), "sha256": sha256_file(CTRL7_CHECKPOINT)}, "runner": source_record(Path(__file__).resolve()), "manifests": {str(seed): source_record(manifests[seed][1]) for seed in SEEDS}}, "artifacts": {"results": str(OUTPUT_ROOT / "results.json"), "per_seed_results": [str(OUTPUT_ROOT / f"seed_{seed}" / "results.json") for seed in SEEDS], "checkpoints": [str(OUTPUT_ROOT / f"seed_{seed}" / "final.pt") for seed in SEEDS]}, "per_seed": per_seed, "next_recommended": "Preserve T4-NOBYPASS-2 evidence; do not tune or reopen D3 after failure." if not all_d3 else "Preserve T4-NOBYPASS-2 closure evidence."}
    digest = write_self_hashed(OUTPUT_ROOT / "results.json", consolidated)
    print(json.dumps({"status": "completed", "classification": consolidated["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": digest, "seeds": len(per_seed), "test_cases": consolidated["executive_summary"]["test_cases_evaluated"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
