"""Execute authorized T5 N=4 development training and sealed gates.

Each encoder is freshly initialized. The frozen supervisor, executor, and
codebook are external read-only dependencies; no T2/T3/T4 encoder checkpoint
is loaded. The test rows are not used by training or checkpoint selection.
"""

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
from t5_nrole_design_audit import GenericNRoleBinder, generic_background_contexts, generic_background_scores, generic_catalog_scores, generic_separation  # noqa: E402
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7301, 7302, 7303, 7304, 7305)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
UPDATES = 5000
VALUE_COUNT = 32
BACKGROUND_COUNT = 40
OUTPUT_ROOT = ROOT / "campaign" / "t5_n4_development" / "training"
MANIFEST_ROOT = ROOT / "campaign" / "t5_n4_preparation" / "manifests"
TEST_CASES_PER_SEED = 23808
TOTAL_TEST_CASES = TEST_CASES_PER_SEED * len(SEEDS)


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
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    placeholder = written.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if sha256_bytes(placeholder) != digest:
        raise RuntimeError("self-hash verification failed")
    return digest


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T5-n4-fresh-manifest-v1":
        raise ValueError(f"unexpected N=4 manifest schema for {seed}")
    if len(manifest.get("train", [])) != 128 or len(manifest.get("test", [])) != TEST_CASES_PER_SEED:
        raise ValueError(f"manifest cardinality mismatch for {seed}")
    if manifest.get("multi_clause_train") != 0 or manifest.get("joint_train_examples") != 0:
        raise ValueError(f"manifest admits joint training for {seed}")
    return manifest, path


def build_training_batch(encoder: GenericNRoleBinder, manifest: dict[str, Any], labels: dict[str, Tensor]) -> dict[str, Any]:
    inverse = {int(value): index for index, value in enumerate(manifest["permutation"])}
    token_rows: list[list[int]] = []
    roles: list[str] = []
    source_rows: list[int] = []
    targets: list[int] = []
    for row in manifest["train"]:
        role = str(row["role"])
        value = int(row["value"])
        argument = str(row["argument"])
        if inverse[value] != int(argument[4:]):
            raise ValueError(f"training permutation mismatch for {role}/{value}")
        token_rows.append([encoder.vocab.encode(manifest["operator_for_role"][role]), encoder.vocab.encode(argument)])
        roles.append(role)
        targets.append(value)
        if role == "FLOOR":
            mask = (labels["constraints"][:, 0] == 1) & (labels["constraints"][:, 1] == 0) & (labels["lower"] == value)
        elif role == "AVOID":
            mask = (labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 1) & (labels["forbidden"] == value)
        else:
            source_rows.append(-1)
            continue
        candidates = torch.where(mask)[0]
        if not len(candidates):
            raise ValueError(f"missing behavior source row for {role}/{value}")
        source_rows.append(int(candidates[0]))
    counts = {role: roles.count(role) for role in ROLES}
    if counts != {role: 32 for role in ROLES}:
        raise ValueError(f"training batch counts: {counts}")
    return {"token_ids": torch.tensor(token_rows, dtype=torch.long), "lengths": torch.full((128,), 2, dtype=torch.long), "roles": roles, "source_rows": torch.tensor(source_rows, dtype=torch.long), "targets": torch.tensor(targets, dtype=torch.long)}


def objective_n4(encoder: GenericNRoleBinder, supervisor: nn.Module, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], batch: dict[str, Any], observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
    details = encoder(batch["token_ids"], batch["lengths"])
    values = details["values"][:, 1]
    role_indices = {role: torch.tensor([index for index, item in enumerate(batch["roles"]) if item == role], dtype=torch.long) for role in ROLES}
    behavior_indices = torch.cat((role_indices["FLOOR"], role_indices["AVOID"]))
    behavior = F.cross_entropy(supervisor(observations["features"][batch["source_rows"][behavior_indices]], values[behavior_indices]), labels["action"][batch["source_rows"][behavior_indices]])
    ref_losses = []
    for role in ROLES:
        indices = role_indices[role]
        logits = executor.register_decoder(torch.cat((values[indices], torch.zeros_like(values[indices])), dim=-1), codebook)
        ref_losses.append(F.cross_entropy(logits, batch["targets"][indices]))
    reference = torch.stack(ref_losses).mean()
    separation, terms, _ = generic_separation(encoder, manifest)
    return {"behavior": behavior, "ref": reference, "sep": separation, "total": behavior + reference + separation, **{f"sep_{name}": value for name, value in terms.items()}}


def d1_atomic(encoder: GenericNRoleBinder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ROLES:
        rows = [row for row in manifest["train"] if row["role"] == role]
        ids = torch.tensor([[encoder.vocab.encode(row["operator"]), encoder.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
        lengths = torch.full((len(rows),), 2, dtype=torch.long)
        with torch.no_grad():
            details = encoder(ids, lengths)
        pointers = details["scores"][ROLES.index(role)].argmax(dim=1)
        decoded = []
        for index, row in enumerate(rows):
            selected = details["values"][index, pointers[index]]
            decoded.append(t4.decode_state(executor, codebook, selected, int(row["value"])))
        result[role] = {"pointer_exact": int((pointers == 1).sum()), "decode_exact": sum(item["raw_pass"] for item in decoded), "raw_exact": sum(item["raw_pass"] for item in decoded), "canonical_exact": sum(item["canonical_pass"] for item in decoded), "raw_canon_equal": sum(item["raw_canon_equal"] for item in decoded), "total": 32, "rows": [{"argument": rows[index]["argument"], "pointer": int(pointers[index]), **decoded[index]} for index in range(32)]}
    result["pass"] = all(result[role]["pointer_exact"] == 32 and result[role]["decode_exact"] == 32 and result[role]["raw_exact"] == 32 and result[role]["canonical_exact"] == 32 and result[role]["raw_canon_equal"] == 32 for role in ROLES)
    return result


def d2_certificates(encoder: GenericNRoleBinder, manifest: dict[str, Any]) -> dict[str, Any]:
    catalog = generic_catalog_scores(encoder, manifest)
    backgrounds, contexts = generic_background_scores(encoder, manifest)
    formal: dict[str, Any] = {}
    arg_link: dict[str, Any] = {}
    for query_index, query in enumerate(ROLES):
        self_scores = catalog[query_index][query_index]
        for source_index, source in enumerate(ROLES):
            if source == query:
                continue
            cross = catalog[query_index][source_index]
            value, flat = (self_scores.unsqueeze(1) - cross.unsqueeze(0)).reshape(-1).min(0)
            i, j = divmod(int(flat), VALUE_COUNT)
            formal[f"G_{query},{source}"] = {"margin": finite(value.item()), "positive": bool(value > 0), "query_role": query, "cross_role": source, "self_arg_index": i, "cross_arg_index": j, "self_score": finite(self_scores[i].item()), "cross_score": finite(cross[j].item())}
        bg = backgrounds[query_index]
        value, flat = (self_scores.unsqueeze(1) - bg.unsqueeze(0)).reshape(-1).min(0)
        i, j = divmod(int(flat), len(contexts))
        formal[f"G_{query},BG"] = {"margin": finite(value.item()), "positive": bool(value > 0), "query_role": query, "self_arg_index": i, "background_index": j, "background_context": contexts[j]["context"], "background_kind": contexts[j]["kind"], "self_score": finite(self_scores[i].item()), "background_score": finite(bg[j].item())}
        indices = [index for index, context in enumerate(contexts) if context["kind"] == "ARG→LINK"]
        arg_scores = bg[indices]
        value, flat = (self_scores.unsqueeze(1) - arg_scores.unsqueeze(0)).reshape(-1).min(0)
        i, j = divmod(int(flat), len(indices))
        arg_link[f"G_{query},ARG→LINK"] = {"margin": finite(value.item()), "positive": bool(value > 0), "query_role": query, "self_arg_index": i, "background_index": indices[j], "background_context": contexts[indices[j]]["context"], "self_score": finite(self_scores[i].item()), "background_score": finite(arg_scores[j].item())}
    return {"formal": formal, "arg_link": arg_link, "formal_count": len(formal), "arg_link_count": len(arg_link), "all_formal_positive": all(item["positive"] for item in formal.values()), "all_arg_link_positive": all(item["positive"] for item in arg_link.values()), "role_role_count": 12, "role_bg_count": 4, "background_context_count": len(contexts)}


def gradient_sanity(encoder: GenericNRoleBinder, supervisor: nn.Module, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], batch: dict[str, Any], observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Any]:
    encoder.zero_grad(set_to_none=True)
    losses = objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    losses["total"].backward()
    link_index = encoder.vocab.encode("LINK")
    gradient = encoder.embedding.weight.grad
    if gradient is None:
        return {"dL_de_LINK_norm": 0.0, "dL_de_LINK_max_abs": 0.0, "strictly_positive": False, "gradient_present": False}
    link = gradient[link_index]
    return {"dL_de_LINK_norm": finite(link.norm().item()), "dL_de_LINK_max_abs": finite(link.abs().max().item()), "dL_de_LINK_nonzero": int((link != 0).sum()), "strictly_positive": bool(link.norm() > 0), "gradient_present": True}


def train_seed(seed: int, manifest: dict[str, Any], manifest_path: Path, observations: dict[str, Tensor], labels: dict[str, Tensor], supervisor: nn.Module, executor: nn.Module, codebook: Tensor, destination: Path) -> tuple[GenericNRoleBinder, dict[str, Any], Path]:
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("final.pt", "results.json")):
        raise FileExistsError(f"refusing to overwrite N=4 seed output {destination}")
    encoder = GenericNRoleBinder(manifest, seed, list(ROLES))
    batch = build_training_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last: dict[str, float] = {}
    for step in range(1, UPDATES + 1):
        progress = (step - 1) / (UPDATES - 1)
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.zero_grad(set_to_none=True)
        losses = objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
        last = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        final_losses_tensor = objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    final_losses = {name: finite(final_losses_tensor[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    torch.save({"encoder": encoder.state_dict(), "seed": seed, "updates": UPDATES, "manifest_sha256": sha256_file(manifest_path), "fresh_init": True, "prior_t2_t3_t4_encoder_checkpoint_loaded": False, "shared_w_v": True, "query_roles": list(ROLES), "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_sep_coefficient": 1.0, "L_sep_terms": 16, "background_contexts": BACKGROUND_COUNT, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0}, checkpoint)
    atomic = d1_atomic(encoder, executor, codebook, manifest)
    certificates = d2_certificates(encoder, manifest)
    gradient = gradient_sanity(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    result = {"status": "trained", "seed": seed, "updates": UPDATES, "fresh_init": {"constructor": "GenericNRoleBinder(manifest_730X_v1, seed)", "seed": seed, "prior_t2_t3_t4_encoder_checkpoint_loaded": False, "shared_w_v": True}, "training": {"batch_rows": {role: 32 for role in ROLES} | {"total": 128}, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_manifest_used_for_training": False, "test_rows_used": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_behavior_roles": ["FLOOR", "AVOID"], "L_ref_roles": list(ROLES), "L_sep": {"coefficient": 1.0, "role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "background_contexts": 40, "catalog_queries": 128}, "last_update_losses": last, "final_losses": final_losses}, "architecture": {"name": "RELKEY", "d_model": 16, "queries": list(ROLES), "query_shape": [4, 16], "address_key": "k_t=LN(SiLU(W_k[e_{t-1},e_t]))", "scores": "s_{r,t}=Q_r^T k_t/4", "selected_state": "r_r=W_v(e_{j_r}), j_r=argmax_{t<length}s_{r,t}", "shared_w_v": True, "gate": False, "stage_b": False, "soft_mixture": False, "hard_pointer": "pure argmax over valid positions"}, "atomic": atomic, "certificates": certificates, "gradient_sanity": gradient, "manifest": source_record(manifest_path), "checkpoint": source_record(checkpoint)}
    return encoder, result, checkpoint


def update_minimum(store: dict[str, Any], name: str, margin: float, evidence: dict[str, Any]) -> None:
    if name not in store or margin < store[name]["margin"]:
        store[name] = {"margin": finite(margin), **evidence}


def d3_test(encoder: GenericNRoleBinder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    fields = tuple(f"{prefix}_{role}" for prefix in ("pointer", "decode", "RAW", "CANON", "RAW_CANON") for role in ROLES)
    counts = {field: 0 for field in fields}
    minima: dict[str, Any] = {}
    outcomes_by_quad: dict[tuple[int, int, int, int], tuple[bool, ...]] = {}
    mismatch_samples: list[dict[str, Any]] = []
    mismatch_count = 0
    order_groups: dict[tuple[int, int, int, int], int] = {}
    cases = manifest["test"]
    for start in range(0, len(cases), 512):
        chunk = cases[start : start + 512]
        token_ids = torch.tensor([[encoder.vocab.encode(token) for token in case["tokens"]] for case in chunk], dtype=torch.long)
        lengths = torch.tensor([len(case["tokens"]) for case in chunk], dtype=torch.long)
        with torch.no_grad():
            details = encoder(token_ids, lengths)
        positions = {role: torch.tensor([case["tokens"].index(next(item["argument"] for item in case["clauses"] if item["role"] == role)) for case in chunk], dtype=torch.long) for role in ROLES}
        bg_mask = torch.tensor([[not token.startswith("ARG_") for token in case["tokens"]] for case in chunk], dtype=torch.bool)
        case_outcomes: list[dict[str, bool]] = [{} for _ in chunk]
        for role_index, role in enumerate(ROLES):
            scores = details["scores"][role_index]
            target_position = positions[role]
            target_score = scores.gather(1, target_position.unsqueeze(1)).squeeze(1)
            pointer = scores.argmax(dim=1)
            selected = details["values"][torch.arange(len(chunk)), pointer]
            logits = executor.register_decoder(torch.cat((selected, torch.zeros_like(selected)), dim=-1), codebook)
            raw = logits.argmax(dim=1)
            canonical_logits = executor.register_decoder(codebook[raw], codebook)
            canonical = canonical_logits.argmax(dim=1)
            targets = torch.tensor([next(item["value"] for item in case["clauses"] if item["role"] == role) for case in chunk], dtype=torch.long)
            outcomes = {f"pointer_{role}": pointer == target_position, f"decode_{role}": raw == targets, f"RAW_{role}": raw == targets, f"CANON_{role}": canonical == targets, f"RAW_CANON_{role}": raw == canonical}
            for field, passed in outcomes.items():
                counts[field] += int(passed.sum())
                for index, value in enumerate(passed.tolist()):
                    case_outcomes[index][field] = bool(value)
            for index, case in enumerate(chunk):
                for other in ROLES:
                    if other == role:
                        continue
                    other_position = positions[other][index]
                    margin = target_score[index] - scores[index, other_position]
                    evidence = {"seed": seed, "case_index": start + index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "H": int(case["H"]), "role": role, "target": case["tokens"][int(target_position[index])], "competitor": case["tokens"][int(other_position)], "target_score": finite(target_score[index].item()), "competitor_score": finite(scores[index, other_position].item())}
                    update_minimum(minima, f"M_{role}^{other}", margin.item(), evidence)
                bg_scores = scores[index].masked_fill(~bg_mask[index], float("-inf"))
                bg_value, bg_position = bg_scores.max(dim=0)
                bg_margin = target_score[index] - bg_value
                evidence = {"seed": seed, "case_index": start + index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "H": int(case["H"]), "role": role, "target": case["tokens"][int(target_position[index])], "competitor": case["tokens"][int(bg_position)], "target_score": finite(target_score[index].item()), "competitor_score": finite(bg_value.item())}
                update_minimum(minima, f"M_{role}^BG", bg_margin.item(), evidence)
        for index, case in enumerate(chunk):
            quad = (int(case["L"]), int(case["F"]), int(case["E"]), int(case["H"]))
            order_groups[quad] = order_groups.get(quad, 0) + 1
            outcome_tuple = tuple(case_outcomes[index][field] for field in fields)
            existing = outcomes_by_quad.get(quad)
            if existing is None:
                outcomes_by_quad[quad] = outcome_tuple
            elif existing != outcome_tuple:
                mismatch_count += 1
                if len(mismatch_samples) < 20:
                    mismatch_samples.append({"seed": seed, "case_index": start + index, "quadruple": quad, "order": case["order_name"], "expected": existing, "actual": outcome_tuple})
    rates = {field: counts[field] / len(cases) for field in fields}
    families = {name: value for name, value in sorted(minima.items()) if name.startswith("M_")}
    return {"cases": len(cases), "expected_cases": TEST_CASES_PER_SEED, "counts": counts, "rates": rates, "all_required_outcomes": all(counts[field] == len(cases) for field in fields), "margin_families": families, "margin_family_count": len(families), "all_16_margin_families_positive": len(families) == 16 and all(item["margin"] > 0.0 for item in families.values()), "order_groups": len(order_groups), "orders_per_quadruple": sorted(set(order_groups.values())), "24_order_outcome_equality": {"equal": mismatch_count == 0, "mismatch_count": mismatch_count, "mismatch_samples": mismatch_samples}}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty output root {OUTPUT_ROOT}")
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
    encoders: dict[int, GenericNRoleBinder] = {}
    for seed in SEEDS:
        destination = OUTPUT_ROOT / f"seed_{seed}"
        try:
            manifest, manifest_path = manifests[seed]
            encoder, item, checkpoint = train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, destination)
            item["development_gate"] = {"D1": bool(item["atomic"]["pass"]), "D2": bool(item["atomic"]["pass"] and item["certificates"]["all_formal_positive"] and item["certificates"]["all_arg_link_positive"] and item["gradient_sanity"]["strictly_positive"]), "D3": False}
            encoders[seed] = encoder
            per_seed.append(item)
        except Exception as error:
            failure = {"status": "execution_failed", "seed": seed, "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, "development_gate": {"D1": False, "D2": False, "D3": False}, "gradient_sanity": {"strictly_positive": False}}
            write_self_hashed(destination / "results.json", failure)
            per_seed.append(failure)
    all_d1 = len(per_seed) == 5 and all(item.get("development_gate", {}).get("D1") is True for item in per_seed)
    all_gradient = len(per_seed) == 5 and all(item.get("gradient_sanity", {}).get("strictly_positive") is True for item in per_seed)
    all_d2 = all_d1 and all_gradient and all(item.get("development_gate", {}).get("D2") is True for item in per_seed)
    for item in per_seed:
        seed = item["seed"]
        if all_d2 and seed in encoders:
            manifest, _ = manifests[seed]
            item["test"] = d3_test(encoders[seed], executor, codebook, manifest, seed)
            item["development_gate"]["D3"] = bool(item["test"]["cases"] == TEST_CASES_PER_SEED and item["test"]["all_required_outcomes"] and item["test"]["all_16_margin_families_positive"] and item["test"]["24_order_outcome_equality"]["equal"])
        else:
            item["test"] = {"status": "HALTED_BY_D1_D2_OR_GRADIENT", "reason": "D1, D2, or LINK gradient failed for at least one seed; D3 remained sealed", "cases": 0}
            item["development_gate"]["D3"] = False
        write_self_hashed(OUTPUT_ROOT / f"seed_{seed}" / "results.json", item)
    all_d3 = all_d2 and all(item.get("development_gate", {}).get("D3") is True for item in per_seed)
    consolidated = {"status": "completed", "classification": "T5-N4 DEVELOPMENT CLOSURE/PASS" if all_d1 and all_d2 and all_d3 else "T5-N4 DEVELOPMENT: VALID FAIL", "task": "T5-N4-DEVELOPMENT TRAINING", "executive_summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "D1_all_seeds": all_d1, "D2_all_seeds": all_d2, "gradient_sanity_all_seeds": all_gradient, "D3_opened": all_d2, "D3_all_seeds": all_d3, "test_cases_evaluated": sum(item.get("test", {}).get("cases", 0) for item in per_seed), "no_sequential_stopping": True, "no_tuning_between_seeds": True, "training_performed": True, "new_checkpoints": True}, "recipe": {"fresh_init": True, "prior_t2_t3_t4_encoder_checkpoint_loaded": False, "architecture": "RELKEY; Q∈R^{4×16}; shared W_v; pure HARDPTR; no masks/gates/Stage B/soft mixture", "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "behavior_roles": ["FLOOR", "AVOID"], "reference_roles": list(ROLES), "L_sep_terms": {"role_role": 12, "role_bg": 4, "total": 16, "coefficient": 1.0}, "background_contexts": {"START→OP": 4, "ARG→LINK": 32, "LINK→OP": 4, "total": 40}, "updates": UPDATES, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "batch_rows": {role: 32 for role in ROLES} | {"total": 128}, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "test_manifest_used_for_training": False}, "provenance": {"executor": {"identity": "frozen approved C1JointModel decoder/codebook", **source_record(BASE_CHECKPOINT)}, "supervisor": {"identity": "frozen CTRL7 LatentConditionedSupervisor", **source_record(CTRL7_CHECKPOINT)}, "runner": source_record(Path(__file__).resolve()), "manifests": {str(seed): source_record(manifests[seed][1]) for seed in SEEDS}}, "artifacts": {"results": str(OUTPUT_ROOT / "results.json"), "per_seed_results": [str(OUTPUT_ROOT / f"seed_{seed}" / "results.json") for seed in SEEDS], "checkpoints": [str(OUTPUT_ROOT / f"seed_{seed}" / "final.pt") for seed in SEEDS]}, "per_seed": [{key: value for key, value in item.items() if key != "_encoder"} for item in per_seed], "next_recommended": "Preserve development evidence; no 740x proposal from a failed N=4 development run." if not (all_d1 and all_d2 and all_d3) else "Preserve development PASS; no scientific PASS_STRONG claim."}
    digest = write_self_hashed(OUTPUT_ROOT / "results.json", consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": digest, "seeds": len(per_seed), "test_cases": consolidated["executive_summary"]["test_cases_evaluated"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
