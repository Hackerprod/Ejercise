"""Execute authorized T4 development training and gated evaluation.

This runner uses fresh three-query RELKEY encoders only.  It never loads a
T2/T3 writer/core/checkpoint and never uses the sealed test rows for training.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
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

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7101, 7102, 7103, 7104, 7105)
ROLES = ("FLOOR", "AVOID", "MATCH")
UPDATES = 5000
DMODEL = 16
OUTPUT_ROOT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "training"
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
    stored = json.loads(written.decode("utf-8"))["artifact_self_hash"]
    needle = f'"artifact_self_hash": "{stored}"'.encode("utf-8")
    placeholder = written.replace(needle, b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if sha256_bytes(placeholder) != stored:
        raise RuntimeError(f"self-hash verification failed: {path}")
    return digest


class ManifestVocab:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.names = tuple(manifest["token_ids"])
        self.name_to_index = {name: index for index, name in enumerate(self.names)}

    def encode(self, name: str) -> int:
        return self.name_to_index[name]


class T4RelKeyEncoder(nn.Module):
    """Fresh RELKEY encoder with three queries and one shared W_v."""

    address_key_formula = "k_t=LN(SiLU(W_k[e_prev,e_cur]))"
    score_formula = "s_r=q_r^T k/4"
    selected_state_formula = "r_r=W_v(e_j), j=argmax_t s_r,t"

    def __init__(self, manifest: dict[str, Any], seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = ManifestVocab(manifest)
        self.embedding = nn.Embedding(len(self.vocab.names), DMODEL)
        self.local_binding = nn.Linear(2 * DMODEL, DMODEL)
        self.local_norm = nn.LayerNorm(DMODEL)
        self.q_f = nn.Parameter(torch.randn(DMODEL) * 0.02)
        self.q_a = nn.Parameter(torch.randn(DMODEL) * 0.02)
        self.q_m = nn.Parameter(torch.randn(DMODEL) * 0.02)
        self.w_v = nn.Linear(DMODEL, 32)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> dict[str, Any]:
        if token_ids.ndim != 2 or lengths.ndim != 1 or token_ids.shape[0] != lengths.shape[0]:
            raise ValueError("token_ids must be [B,T] and lengths must be [B]")
        batch, token_count = token_ids.shape
        valid = torch.arange(token_count).unsqueeze(0) < lengths.unsqueeze(1)
        embeddings = self.embedding(token_ids)
        zero = torch.zeros((batch, 1, DMODEL), dtype=embeddings.dtype)
        previous = torch.cat((zero, embeddings[:, :-1]), dim=1)
        keys = self.local_norm(self.local_binding(torch.cat((previous, embeddings), dim=-1)))
        keys = keys.masked_fill(~valid.unsqueeze(-1), 0.0)
        values = self.w_v(embeddings).masked_fill(~valid.unsqueeze(-1), 0.0)
        scores = {
            "FLOOR": (keys @ self.q_f) / 4.0,
            "AVOID": (keys @ self.q_a) / 4.0,
            "MATCH": (keys @ self.q_m) / 4.0,
        }
        scores = {role: score.masked_fill(~valid, float("-inf")) for role, score in scores.items()}
        return {"scores": scores, "keys": keys, "values": values, "valid": valid}


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T4-nobypass1-three-active-roles-manifest-v1":
        raise ValueError(f"unexpected manifest schema for {seed}")
    if len(manifest.get("train", [])) != 96 or len(manifest.get("test", [])) != 5952:
        raise ValueError(f"manifest cardinality mismatch for {seed}")
    if manifest.get("multi_clause_train") != 0 or manifest.get("joint_train_examples") != 0:
        raise ValueError(f"manifest admits multi-clause training for {seed}")
    if set(manifest.get("operator_for_role", {})) != set(ROLES):
        raise ValueError(f"manifest role mapping malformed for {seed}")
    return manifest, path


def build_training_batch(encoder: T4RelKeyEncoder, manifest: dict[str, Any], labels: dict[str, Tensor]) -> dict[str, Any]:
    inverse = {int(value): index for index, value in enumerate(manifest["permutation"])}
    token_rows: list[list[int]] = []
    roles: list[str] = []
    source_rows: list[int] = []
    targets: list[int] = []
    operator_for_role = manifest["operator_for_role"]
    for row in manifest["train"]:
        role = str(row["role"])
        value = int(row["value"])
        argument = str(row["argument"])
        if inverse[value] != int(argument[4:]):
            raise ValueError(f"training permutation mismatch for {role}/{value}")
        token_rows.append([encoder.vocab.encode(operator_for_role[role]), encoder.vocab.encode(argument)])
        roles.append(role)
        targets.append(value)
        if role == "FLOOR":
            mask = (labels["constraints"][:, 0] == 1) & (labels["constraints"][:, 1] == 0) & (labels["lower"] == value)
            candidates = torch.where(mask)[0]
            if not len(candidates):
                raise ValueError(f"missing FLOOR source row for {value}")
            source_rows.append(int(candidates[0]))
        elif role == "AVOID":
            mask = (labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 1) & (labels["forbidden"] == value)
            candidates = torch.where(mask)[0]
            if not len(candidates):
                raise ValueError(f"missing AVOID source row for {value}")
            source_rows.append(int(candidates[0]))
        else:
            source_rows.append(-1)
    if {role: roles.count(role) for role in ROLES} != {role: 32 for role in ROLES}:
        raise ValueError("training batch is not 32/32/32")
    return {
        "token_ids": torch.tensor(token_rows, dtype=torch.long),
        "lengths": torch.full((96,), 2, dtype=torch.long),
        "roles": roles,
        "source_rows": torch.tensor(source_rows, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def catalog_inputs(encoder: T4RelKeyEncoder, manifest: dict[str, Any]) -> tuple[Tensor, Tensor, list[tuple[str, int]]]:
    rows: list[list[int]] = []
    metadata: list[tuple[str, int]] = []
    for operator_role in ROLES:
        for index in range(VALUE_COUNT):
            rows.append([encoder.vocab.encode(manifest["operator_for_role"][operator_role]), encoder.vocab.encode(arg_name(index))])
            metadata.append((operator_role, index))
    return torch.tensor(rows, dtype=torch.long), torch.full((96,), 2, dtype=torch.long), metadata


def catalog_scores(encoder: T4RelKeyEncoder, manifest: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
    token_ids, lengths, metadata = catalog_inputs(encoder, manifest)
    details = encoder(token_ids, lengths)
    result: dict[str, dict[str, Tensor]] = {query: {} for query in ROLES}
    for query in ROLES:
        for source in ROLES:
            start = ROLES.index(source) * VALUE_COUNT
            result[query][source] = details["scores"][query][start : start + VALUE_COUNT, 1]
    return result


def rcsep3(scores: dict[str, dict[str, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
    terms: dict[str, Tensor] = {}
    for query in ROLES:
        self_scores = scores[query][query]
        for source in ROLES:
            if source == query:
                continue
            relation = f"{query[:1]}<-{source[:1]}"
            terms[relation] = F.softplus(scores[query][source].unsqueeze(0) - self_scores.unsqueeze(1)).mean()
    if tuple(terms) != ("F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A"):
        raise AssertionError(f"unexpected RCSEP-3 relation order: {tuple(terms)}")
    return torch.stack(tuple(terms.values())).mean(), terms


def objective(encoder: T4RelKeyEncoder, supervisor: nn.Module, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], batch: dict[str, Any], observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
    details = encoder(batch["token_ids"], batch["lengths"])
    values = details["values"][:, 1]  # Atomic rows are exactly [OP_role, ARG_i]; no soft mixture.
    floor = torch.tensor([index for index, role in enumerate(batch["roles"]) if role == "FLOOR"], dtype=torch.long)
    avoid = torch.tensor([index for index, role in enumerate(batch["roles"]) if role == "AVOID"], dtype=torch.long)
    match = torch.tensor([index for index, role in enumerate(batch["roles"]) if role == "MATCH"], dtype=torch.long)
    behavior_indices = torch.cat((floor, avoid))
    behavior = F.cross_entropy(
        supervisor(observations["features"][batch["source_rows"][behavior_indices]], values[behavior_indices]),
        labels["action"][batch["source_rows"][behavior_indices]],
    )
    ref_losses = []
    for role, indices in (("FLOOR", floor), ("AVOID", avoid), ("MATCH", match)):
        logits = executor.register_decoder(torch.cat((values[indices], torch.zeros_like(values[indices])), dim=-1), codebook)
        ref_losses.append(F.cross_entropy(logits, batch["targets"][indices]))
    reference = torch.stack(ref_losses).mean()
    scores = catalog_scores(encoder, manifest)
    separation, terms = rcsep3(scores)
    return {"behavior": behavior, "ref": reference, "sep": separation, "total": behavior + reference + separation, **{f"sep_{name}": value for name, value in terms.items()}}


def decode_state(executor: nn.Module, codebook: Tensor, state: Tensor, target: int) -> dict[str, Any]:
    zeros = torch.zeros((1, state.shape[-1]), dtype=state.dtype)
    logits = executor.register_decoder(torch.cat((state.unsqueeze(0), zeros), dim=-1), codebook)[0]
    raw = int(logits.argmax())
    canonical_logits = executor.register_decoder(codebook[raw].unsqueeze(0), codebook)[0]
    canonical = int(canonical_logits.argmax())
    return {"raw": raw, "canonical": canonical, "target": target, "raw_pass": raw == target, "canonical_pass": canonical == target, "raw_canon_equal": raw == canonical, "target_score": finite(logits[target].item()), "raw_score": finite(logits[raw].item())}


def atomic_gate(encoder: T4RelKeyEncoder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ROLES:
        rows = [row for row in manifest["train"] if row["role"] == role]
        ids = torch.tensor([[encoder.vocab.encode(row["operator"]), encoder.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
        lengths = torch.full((len(rows),), 2, dtype=torch.long)
        details = encoder(ids, lengths)
        pointer = details["scores"][role].argmax(dim=1)
        decoded = []
        for index in range(len(rows)):
            selected = details["values"][index, pointer[index]]
            decoded.append(decode_state(executor, codebook, selected, int(rows[index]["value"])))
        result[role] = {"pointer_exact": int((pointer == 1).sum()), "decode_exact": sum(item["raw_pass"] for item in decoded), "canonical_exact": sum(item["canonical_pass"] for item in decoded), "total": 32, "rows": [{"arg": rows[index]["argument"], "pointer": int(pointer[index]), **decoded[index]} for index in range(32)]}
    result["pass"] = all(result[role]["pointer_exact"] == 32 and result[role]["decode_exact"] == 32 and result[role]["canonical_exact"] == 32 for role in ROLES)
    return result


def structural_scores(encoder: T4RelKeyEncoder, manifest: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    token_rows: list[list[int]] = []
    names: list[str] = []
    for operator_role in ROLES:
        token_rows.append([encoder.vocab.encode(manifest["operator_for_role"][operator_role]), encoder.vocab.encode("ARG_00")])
        names.append(manifest["operator_for_role"][operator_role])
    token_rows.append([encoder.vocab.encode(manifest["operator_for_role"]["FLOOR"]), encoder.vocab.encode("ARG_00"), encoder.vocab.encode("LINK")])
    names.append("LINK")
    padded = [row + [encoder.vocab.encode("LINK")] * (3 - len(row)) for row in token_rows]
    details = encoder(torch.tensor(padded, dtype=torch.long), torch.tensor([len(row) for row in token_rows], dtype=torch.long))
    return {role: [(name, finite(details["scores"][role][index, 0 if name != "LINK" else 2].item())) for index, name in enumerate(names)] for role in ROLES}


def certificate_gate(encoder: T4RelKeyEncoder, manifest: dict[str, Any]) -> dict[str, Any]:
    scores = catalog_scores(encoder, manifest)
    certificates: dict[str, Any] = {}
    for query in ROLES:
        self_scores = scores[query][query]
        for source in ROLES:
            if source == query:
                continue
            matrix = self_scores.unsqueeze(1) - scores[query][source].unsqueeze(0)
            value, flat = matrix.reshape(-1).min(0)
            i, j = divmod(int(flat), VALUE_COUNT)
            name = f"G_{query[:1]},{source[:1]}"
            certificates[name] = {"margin": finite(value.item()), "positive": bool(value > 0), "query_role": query, "self_role": query, "self_arg_index": i, "cross_role": source, "cross_arg_index": j, "self_score": finite(self_scores[i].item()), "cross_score": finite(scores[query][source][j].item())}
    structures = structural_scores(encoder, manifest)
    for query in ROLES:
        value, (competitor, competitor_score) = min(((scores[query][query][index].item() - candidate_score, (f"ARG_{index:02d}", candidate_score)) for index in range(VALUE_COUNT) for _, candidate_score in structures[query]), key=lambda item: item[0])
        # Recompute exact winning evidence without losing self argument identity.
        best_margin = float("inf")
        evidence: dict[str, Any] = {}
        for index in range(VALUE_COUNT):
            for candidate, candidate_score in structures[query]:
                margin = scores[query][query][index].item() - candidate_score
                if margin < best_margin:
                    best_margin = margin
                    evidence = {"self_arg_index": index, "self_score": finite(scores[query][query][index].item()), "competitor": candidate, "competitor_score": finite(candidate_score)}
        certificates[f"G_{query[:1]},STRUCT"] = {"margin": finite(best_margin), "positive": best_margin > 0.0, "query_role": query, **evidence}
    return {"definitions": {"directed": "min_i s_r(OP_r,ARG_i) - max_j s_r(OP_r',ARG_j)", "structural": "min self ARG score - max score over OP_X, OP_Y, OP_Z, LINK positions", "catalog_queries": 96}, "certificates": certificates, "all_positive": all(item["positive"] for item in certificates.values()), "six_directed_and_three_structural": len(certificates) == 9}


def margin_minima_update(store: dict[str, Any], name: str, margin: float, evidence: dict[str, Any]) -> None:
    if name not in store or margin < store[name]["margin"]:
        store[name] = {"margin": finite(margin), **evidence}


def evaluate_case(encoder: T4RelKeyEncoder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], case: dict[str, Any], seed: int, case_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens = list(case["tokens"])
    ids = torch.tensor([[encoder.vocab.encode(token) for token in tokens]], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    target_positions = {clause["role"]: tokens.index(clause["argument"]) for clause in case["clauses"]}
    with torch.no_grad():
        details = encoder(ids, lengths)
        pointers = {role: int(details["scores"][role][0].argmax()) for role in ROLES}
        decodes = {role: decode_state(executor, codebook, details["values"][0, pointers[role]], int(next(clause["value"] for clause in case["clauses"] if clause["role"] == role))) for role in ROLES}
        scores = {role: details["scores"][role][0] for role in ROLES}
    outcomes: dict[str, bool] = {}
    for role in ROLES:
        outcomes[f"pointer_{role[:1]}"] = pointers[role] == target_positions[role]
        outcomes[f"decode_{role[:1]}"] = decodes[role]["raw_pass"]
        outcomes[f"RAW_{role[:1]}"] = decodes[role]["raw_pass"]
        outcomes[f"CANON_{role[:1]}"] = decodes[role]["canonical_pass"]
        outcomes[f"RAW_CANON_{role[:1]}"] = decodes[role]["raw_canon_equal"]
    minima: dict[str, tuple[float, dict[str, Any]]] = {}
    structural_positions = [(index, token) for index, token in enumerate(tokens) if not token.startswith("ARG_")]
    for role in ROLES:
        target_position = target_positions[role]
        target_score = finite(scores[role][target_position].item())
        for other in ROLES:
            if other == role:
                continue
            name = f"M_{role[:1]}^{other[:1]}"
            competitor_position = target_positions[other]
            competitor_score = finite(scores[role][competitor_position].item())
            minima[name] = (target_score - competitor_score, {"seed": seed, "case_index": case_index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "role": role, "target": tokens[target_position], "competitor": tokens[competitor_position], "target_score": target_score, "competitor_score": competitor_score})
        structural_position, structural_token = min(structural_positions, key=lambda item: (-finite(scores[role][item[0]].item()), item[0]))
        name = f"M_{role[:1]}^STRUCT"
        minima[name] = (target_score - finite(scores[role][structural_position].item()), {"seed": seed, "case_index": case_index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "role": role, "target": tokens[target_position], "competitor": structural_token, "target_score": target_score, "competitor_score": finite(scores[role][structural_position].item())})
    row = {"seed": seed, "case_index": case_index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "pointers": pointers, "decodes": decodes, "outcomes": outcomes, "margins": {name: {"margin": finite(value), **evidence} for name, (value, evidence) in minima.items()}}
    failures = [{"seed": seed, "case_index": case_index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "failure": name, **evidence, "margin": finite(value)} for name, (value, evidence) in minima.items() if value <= 0.0]
    failures.extend({"seed": seed, "case_index": case_index, "order": case["order_name"], "L": int(case["L"]), "F": int(case["F"]), "E": int(case["E"]), "failure": name, "margin": None} for name, passed in outcomes.items() if not passed)
    return row, failures


def evaluate_test(encoder: T4RelKeyEncoder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    fields = tuple(f"{prefix}_{role[:1]}" for prefix in ("pointer", "decode", "RAW", "CANON", "RAW_CANON") for role in ROLES)
    counts = {field: 0 for field in fields}
    minima: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    outcomes_by_pair: dict[tuple[int, int, int], dict[str, tuple[bool, ...]]] = {}
    predicate_total = 0
    predicate_pass = 0
    for case_index, case in enumerate(manifest["test"]):
        row, case_failures = evaluate_case(encoder, executor, codebook, manifest, case, seed, case_index)
        for field, passed in row["outcomes"].items():
            counts[field] += int(passed)
        if int(case["E"]) >= int(case["L"]) and int(case["E"]) != int(case["F"]):
            predicate_pass += 1
        predicate_total += 1
        for name, margin in row["margins"].items():
            margin_minima_update(minima, name, margin["margin"], {key: margin[key] for key in ("seed", "case_index", "order", "L", "F", "E", "role", "target", "competitor", "target_score", "competitor_score")})
        outcomes_by_pair.setdefault((row["L"], row["F"], row["E"]), {})[row["order"]] = tuple(row["outcomes"][field] for field in fields)
        if len(failures) < 100:
            failures.extend(case_failures[: 100 - len(failures)])
    mismatches = []
    for pair, order_results in sorted(outcomes_by_pair.items()):
        values = list(order_results.values())
        if len(order_results) != 6 or any(value != values[0] for value in values[1:]):
            mismatches.append({"L": pair[0], "F": pair[1], "E": pair[2], "outcomes": order_results})
    return {"cases": len(manifest["test"]), "expected_cases": 5952, "counts": counts, "rates": {field: counts[field] / max(1, len(manifest["test"])) for field in fields}, "all_required_outcomes": all(counts[field] == 5952 for field in fields), "global_minima": minima, "all_nine_margins_positive": len(minima) == 9 and all(item["margin"] > 0.0 for item in minima.values()), "six_order_outcome_equality": {"equal": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches}, "joint_predicate_non_gate": {"expression": "E>=L and E!=F", "pass": predicate_pass, "total": predicate_total}, "failures_sample": failures}


def train_seed(seed: int, manifest: dict[str, Any], manifest_path: Path, observations: dict[str, Tensor], labels: dict[str, Tensor], supervisor: nn.Module, executor: nn.Module, codebook: Tensor, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("final.pt", "results.json")):
        raise FileExistsError(f"refusing to overwrite T4 seed output {destination}")
    encoder = T4RelKeyEncoder(manifest, seed)
    batch = build_training_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last: dict[str, float] = {}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = objective(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
        last = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * (step - 1) / (UPDATES - 1)
    encoder.eval()
    with torch.no_grad():
        final_losses_tensor = objective(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    final_losses = {name: finite(final_losses_tensor[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    torch.save({"encoder": encoder.state_dict(), "seed": seed, "updates": UPDATES, "manifest_sha256": sha256_file(manifest_path), "fresh_init": True, "prior_t2_t3_checkpoint_loaded": False, "shared_w_v": True, "query_roles": list(ROLES), "objective": "L_behavior^(F,A) + L_ref^(F,A,M) + L_sep^(3)", "L_sep_coefficient": 1.0, "multi_clause_train": 0, "test_rows_used": 0}, checkpoint)
    atomic = atomic_gate(encoder, executor, codebook, manifest)
    certificates = certificate_gate(encoder, manifest)
    training_result = {"seed": seed, "updates": UPDATES, "batch_rows": {role: 32 for role in ROLES} | {"total": 96}, "multi_clause_train": 0, "test_rows_used": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A) + L_ref^(F,A,M) + L_sep^(3)", "L_behavior_roles": ["FLOOR", "AVOID"], "L_ref_roles": list(ROLES), "L_sep": {"coefficient": 1.0, "relations": ["F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A"], "catalog_queries": 96, "noop_included": False}, "last_update_losses": last, "final_losses": final_losses}
    result = {"status": "trained", "seed": seed, "fresh_init": {"constructor": "T4RelKeyEncoder(manifest_710X_v1, seed)", "seed": seed, "prior_t2_t3_checkpoint_loaded": False, "shared_w_v": True}, "training": training_result, "architecture": {"name": "RELKEY", "d_model": DMODEL, "queries": list(ROLES), "address_key": T4RelKeyEncoder.address_key_formula, "scores": T4RelKeyEncoder.score_formula, "selected_state": T4RelKeyEncoder.selected_state_formula, "forward_inputs": ["token_ids", "lengths"], "shared_w_v": True, "gate": False, "stage_b": False, "soft_mixture": False}, "atomic": atomic, "certificates": certificates, "manifest": source_record(manifest_path), "checkpoint": source_record(checkpoint)}
    write_self_hashed(destination / "results.json", result)
    return {"encoder": encoder, "result": result, "checkpoint": checkpoint}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty T4 output root {OUTPUT_ROOT}")
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
    for seed in SEEDS:
        destination = OUTPUT_ROOT / f"seed_{seed}"
        try:
            manifest, manifest_path = manifests[seed]
            trained = train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, destination)
            item = trained["result"]
            item["_encoder"] = trained["encoder"]
            item["_manifest"] = manifest
            item["_manifest_path"] = manifest_path
            item["_checkpoint_path"] = trained["checkpoint"]
            per_seed.append(item)
        except Exception as error:
            failure = {"status": "execution_failed", "seed": seed, "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, "development_gate": {"D1": False, "D2": False, "D3": False}, "test": {"status": "HALTED_BY_D1_D2", "cases": 0}}
            write_self_hashed(destination / "results.json", failure)
            per_seed.append(failure)
    all_d1 = len(per_seed) == 5 and all(item.get("development_gate", {}).get("D1") is True for item in per_seed)
    for item in per_seed:
        if "atomic" in item and "certificates" in item:
            item["development_gate"] = {
                "D1": item["atomic"]["pass"],
                "D2": bool(item["atomic"]["pass"] and item["certificates"]["all_positive"] and item["certificates"]["six_directed_and_three_structural"]),
            }
    all_d1 = len(per_seed) == 5 and all(item.get("development_gate", {}).get("D1") is True for item in per_seed)
    all_d2 = all_d1 and all(item.get("development_gate", {}).get("D2") is True for item in per_seed)
    # Gate D3 is global: do not open any triple-joint test until every seed closes D1+D2.
    for item in per_seed:
        if all_d2 and "_encoder" in item:
            item["test"] = evaluate_test(item["_encoder"], executor, codebook, item["_manifest"], item["seed"])
            item["development_gate"]["D3"] = bool(item["test"]["cases"] == 5952 and item["test"]["all_required_outcomes"] and item["test"]["all_nine_margins_positive"] and item["test"]["six_order_outcome_equality"]["equal"])
        else:
            item["test"] = {"status": "HALTED_BY_D1_D2", "reason": "D1 or D2 failed for at least one development seed; no triple-joint test was opened", "cases": 0}
            item["development_gate"]["D3"] = False
        for transient in ("_encoder", "_manifest", "_manifest_path", "_checkpoint_path"):
            item.pop(transient, None)
        seed_path = OUTPUT_ROOT / f"seed_{item['seed']}" / "results.json"
        write_self_hashed(seed_path, item)
    all_d3 = all_d2 and all(item.get("development_gate", {}).get("D3") is True for item in per_seed)
    consolidated: dict[str, Any] = {
        "status": "completed",
        "classification": "T4 DEVELOPMENT CLOSURE/PASS" if all_d1 and all_d2 and all_d3 else "T4 DEVELOPMENT: VALID FAIL",
        "task": "T4-NOBYPASS-1-THREE-ACTIVE-ROLES",
        "executive_summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "D1_all_seeds": all_d1, "D2_all_seeds": all_d2, "D3_opened": all_d2, "D3_all_seeds": all_d3, "test_cases_evaluated": sum(item.get("test", {}).get("cases", 0) for item in per_seed), "no_test_training": True, "no_sequential_stopping": True, "no_tuning_between_seeds": True},
        "recipe": {"fresh_init": True, "prior_t2_t3_checkpoint_loaded": False, "architecture": T4RelKeyEncoder.address_key_formula, "three_queries": list(ROLES), "shared_w_v": True, "objective": "L_behavior^(F,A) + L_ref^(F,A,M) + L_sep^(3)", "L_sep_relations": ["F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A"], "L_sep_catalog_queries": 96, "L_sep_coefficient": 1.0, "updates": UPDATES, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5}, "batch_rows": {"FLOOR": 32, "AVOID": 32, "MATCH": 32, "total": 96}, "multi_clause_train": 0, "test_rows_used": 0, "stage_b": False, "gate": False, "soft_mixture": False},
        "provenance": {"executor": {"identity": "approved C1JointModel via ctrl2_common.load_executor", **source_record(BASE_CHECKPOINT)}, "supervisor": {"identity": "frozen CTRL7 LatentConditionedSupervisor", **source_record(CTRL7_CHECKPOINT)}, "runner": source_record(Path(__file__).resolve()), "manifests": {str(seed): source_record(manifests[seed][1]) for seed in SEEDS}},
        "per_seed": per_seed,
        "artifacts": {"results": str(OUTPUT_ROOT / "results.json"), "per_seed_results": [str(OUTPUT_ROOT / f"seed_{seed}" / "results.json") for seed in SEEDS], "checkpoints": [str(OUTPUT_ROOT / f"seed_{seed}" / "final.pt") for seed in SEEDS]},
        "next_recommended": "Preserve development evidence; T4 fresh PASS_STRONG requires 7201-7205." if all_d1 and all_d2 and all_d3 else "Preserve VALID FAIL evidence; do not tune RCSEP-3 or reopen D3 after D1/D2 failure.",
    }
    write_self_hashed(OUTPUT_ROOT / "results.json", consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": consolidated["artifact_self_hash"], "seeds": len(per_seed), "test_cases": consolidated["executive_summary"]["test_cases_evaluated"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
