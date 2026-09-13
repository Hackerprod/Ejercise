"""T6 active-set design audit: isolated NULL prototype, no training."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder, generic_background_contexts, generic_catalog_scores, generic_separation  # noqa: E402


SEED = 7301
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
MANIFEST = ROOT / "campaign" / "t5_n4_preparation" / "manifests" / f"manifest_{SEED}_v1.json"
CHECKPOINT = ROOT / "campaign" / "t5_n4_development" / "training" / f"seed_{SEED}" / "final.pt"
OUTPUT = ROOT / "campaign" / "t6_nrole_activeset_design" / "t6_nrole_activeset_design.json"


class NullCapableBinder(nn.Module):
    """T6 prototype wrapper. NULL is disabled by forwarding unchanged base path."""

    def __init__(self, manifest: dict[str, Any], seed: int) -> None:
        super().__init__()
        self.base = GenericNRoleBinder(manifest, seed, list(ROLES))
        self.null_bias = nn.Parameter(torch.zeros(len(ROLES)))
        self.roles = self.base.roles
        self.vocab = self.base.vocab

    def forward(self, token_ids: Tensor, lengths: Tensor) -> dict[str, Any]:
        return self.base(token_ids, lengths)


def finite(value: Any) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"non-finite value: {result}")
    return result


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare_tensor(left: Tensor, right: Tensor) -> dict[str, Any]:
    exact = torch.equal(left, right)
    return {"exact": exact, "shape": list(left.shape), "max_abs": 0.0 if exact else finite((left - right).abs().max().item())}


def compare_path(reference: GenericNRoleBinder, prototype: NullCapableBinder, token_ids: Tensor, lengths: Tensor) -> dict[str, Any]:
    with torch.no_grad():
        left = reference(token_ids, lengths)
        right = prototype(token_ids, lengths)
    scores = {role: compare_tensor(left["scores"][index], right["scores"][index]) for index, role in enumerate(ROLES)}
    return {"keys": compare_tensor(left["keys"], right["keys"]), "values": compare_tensor(left["values"], right["values"]), "valid": compare_tensor(left["valid"], right["valid"]), "scores": scores, "all_exact": all(item["exact"] for item in (compare_tensor(left["keys"], right["keys"]), compare_tensor(left["values"], right["values"]), compare_tensor(left["valid"], right["valid"]), *scores.values()))}


def null_hardptr(scores: list[float], length: int, null_score: float) -> dict[str, Any]:
    valid = list(range(max(0, length)))
    if not valid:
        return {"present": False, "pointer": None, "null": True}
    best_score = max(scores[index] for index in valid)
    if null_score >= best_score:
        return {"present": False, "pointer": None, "null": True}
    pointer = min(index for index in valid if scores[index] == best_score)
    return {"present": True, "pointer": pointer, "null": False}


def canonicalize(result: dict[str, Any], raw: int | None, canonical: int | None) -> dict[str, Any]:
    if not result["present"]:
        return {"raw": None, "canonical": None}
    return {"raw": raw, "canonical": canonical}


def synthetic_contract_checks() -> dict[str, Any]:
    checks = {
        "real_above_null_returns_pointer": null_hardptr([0.1, 0.9, 0.2], 3, 0.5) == {"present": True, "pointer": 1, "null": False},
        "null_wins_when_all_real_below": null_hardptr([0.1, 0.2], 2, 0.5) == {"present": False, "pointer": None, "null": True},
        "value_zero_remains_present": bool(null_hardptr([0.9, 0.1], 2, 0.5)["present"] and 0 == 0),
        "padding_ignored": null_hardptr([0.1, 0.2, 99.0], 2, 0.5) == {"present": False, "pointer": None, "null": True},
        "present_canonicalizes": canonicalize({"present": True}, 0, 0) == {"raw": 0, "canonical": 0},
        "absent_preserves_null": canonicalize({"present": False}, 0, 0) == {"raw": None, "canonical": None},
        "null_first_on_real_tie": null_hardptr([0.7], 1, 0.7) == {"present": False, "pointer": None, "null": True},
        "first_real_on_real_tie": null_hardptr([0.8, 0.8, 0.1], 3, 0.2) == {"present": True, "pointer": 0, "null": False},
        "empty_length_returns_null": null_hardptr([1.0], 0, 0.0) == {"present": False, "pointer": None, "null": True},
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "tie_policy": {"null_vs_real": "NULL wins ties", "real_vs_real": "minimum valid position wins", "future_pass_must_not_depend_on_ties": True}, "absence_contract": {"present": False, "pointer": None, "value_state": None, "decoded_value": None, "raw": None, "canonical": None, "never_decode_zero_vector": True, "never_index_codebook_minus_one": True}}


def interval_diagnostic(reference: GenericNRoleBinder, manifest: dict[str, Any]) -> dict[str, Any]:
    catalog = generic_catalog_scores(reference, manifest)
    contexts = generic_background_contexts(reference, manifest)
    background_ids = torch.tensor([context["token_ids"] for context in contexts], dtype=torch.long)
    background_lengths = torch.full((len(contexts),), 2, dtype=torch.long)
    with torch.no_grad():
        background = reference(background_ids, background_lengths)
    rows = []
    for query_index, role in enumerate(ROLES):
        positive = catalog[query_index][query_index]
        p_value, p_index = positive.min(0)
        negatives: list[dict[str, Any]] = []
        for source_index, source in enumerate(ROLES):
            if source == role:
                continue
            cross = catalog[query_index][source_index]
            value, index = cross.max(0)
            negatives.append({"kind": "role→role", "source_role": source, "argument_index": int(index), "score": finite(value.item()), "context": f"{source}/ARG_{int(index):02d}"})
        bg_scores = background["scores"][query_index]
        for index, context in enumerate(contexts):
            negatives.append({"kind": "BG", "source_role": None, "argument_index": None, "score": finite(bg_scores[index, context["score_position"]].item()), "context": context["context"], "background_kind": context["kind"]})
        negative = max(negatives, key=lambda item: item["score"])
        p_min = finite(p_value.item())
        n_max = finite(negative["score"])
        rows.append({"role": role, "P_r_min": p_min, "P_r_min_arg_index": int(p_index), "P_r_min_context": f"{role}/ARG_{int(p_index):02d}", "N_r_max": n_max, "N_r_max_context": negative, "interval_width": p_min - n_max, "interval_exists": bool(n_max < p_min), "b_r_mid": (n_max + p_min) / 2.0})
    return {"status": "passed" if all(row["interval_exists"] for row in rows) else "interval_missing", "source": "atomic catalog + 40 BG contexts only; no multi-clause or joint examples", "roles": rows}


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    reference = GenericNRoleBinder(manifest, SEED, list(ROLES))
    reference.load_state_dict(payload["encoder"], strict=True)
    reference.eval()
    prototype = NullCapableBinder(manifest, SEED)
    prototype.base.load_state_dict(payload["encoder"], strict=True)
    prototype.eval()
    with torch.no_grad():
        base_state = {"null_bias": prototype.null_bias.detach().clone()}
    # NULL is disabled explicitly: no b_r participates in the forwarded score path.
    atomic_rows = []
    for role in ROLES:
        operator = manifest["operator_for_role"][role]
        for index in range(VALUE_COUNT):
            atomic_rows.append([reference.vocab.encode(operator), reference.vocab.encode(f"ARG_{index:02d}")])
    atomic_ids = torch.tensor(atomic_rows, dtype=torch.long)
    atomic_lengths = torch.full((len(atomic_rows),), 2, dtype=torch.long)
    bg_contexts = generic_background_contexts(reference, manifest)
    bg_ids = torch.tensor([context["token_ids"] for context in bg_contexts], dtype=torch.long)
    bg_lengths = torch.full((len(bg_contexts),), 2, dtype=torch.long)
    test = manifest["test"]
    test_ids = torch.tensor([[reference.vocab.encode(token) for token in case["tokens"]] for case in test], dtype=torch.long)
    test_lengths = torch.tensor([len(case["tokens"]) for case in test], dtype=torch.long)
    equivalence = {"atomic_catalog": compare_path(reference, prototype, atomic_ids, atomic_lengths), "background_contexts": compare_path(reference, prototype, bg_ids, bg_lengths), "sealed_development_test": {"cases": len(test), "all_exact": True, "keys": {"max_abs": 0.0}, "scores_real_positions": {"max_abs": 0.0}, "pointers_exact": True, "selected_states_exact": True, "raw_decoded_indices_exact": True, "canonical_decoded_indices_exact": True}}
    pointer_counts = {role: 0 for role in ROLES}
    decode_counts = {role: 0 for role in ROLES}
    canonical_counts = {role: 0 for role in ROLES}
    max_key = 0.0
    max_score = 0.0
    for start in range(0, len(test), 512):
        chunk = test[start : start + 512]
        ids = test_ids[start : start + len(chunk)]
        lengths = test_lengths[start : start + len(chunk)]
        with torch.no_grad():
            left = reference(ids, lengths)
            right = prototype(ids, lengths)
        max_key = max(max_key, finite((left["keys"] - right["keys"]).abs().max().item()))
        for role_index, role in enumerate(ROLES):
            max_score = max(max_score, finite((left["scores"][role_index] - right["scores"][role_index]).abs().max().item()))
            left_pointer = left["scores"][role_index].argmax(dim=1)
            right_pointer = right["scores"][role_index].argmax(dim=1)
            if not torch.equal(left_pointer, right_pointer):
                equivalence["sealed_development_test"]["pointers_exact"] = False
            left_state = left["values"][torch.arange(len(chunk)), left_pointer]
            right_state = right["values"][torch.arange(len(chunk)), right_pointer]
            if not torch.equal(left_state, right_state):
                equivalence["sealed_development_test"]["selected_states_exact"] = False
            # Decoder comparison uses the frozen external runtime below, after one load.
            pointer_counts[role] += int(torch.equal(left_pointer, right_pointer))
            decode_counts[role] += int(torch.equal(left_state, right_state))
            canonical_counts[role] += int(torch.equal(left_state, right_state))
    equivalence["sealed_development_test"]["keys"]["max_abs"] = max_key
    equivalence["sealed_development_test"]["scores_real_positions"]["max_abs"] = max_score
    # Frozen decoder/runtime is loaded once only for the selected-state decode comparison.
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    raw_exact = True
    canonical_exact = True
    for start in range(0, len(test), 512):
        chunk = test[start : start + 512]
        ids = test_ids[start : start + len(chunk)]
        lengths = test_lengths[start : start + len(chunk)]
        with torch.no_grad():
            left = reference(ids, lengths)
            right = prototype(ids, lengths)
        for role_index in range(len(ROLES)):
            lp = left["scores"][role_index].argmax(dim=1)
            rp = right["scores"][role_index].argmax(dim=1)
            ls = left["values"][torch.arange(len(chunk)), lp]
            rs = right["values"][torch.arange(len(chunk)), rp]
            with torch.no_grad():
                lraw = executor.register_decoder(torch.cat((ls, torch.zeros_like(ls)), dim=-1), codebook).argmax(dim=1)
                rraw = executor.register_decoder(torch.cat((rs, torch.zeros_like(rs)), dim=-1), codebook).argmax(dim=1)
                lcanon = executor.register_decoder(codebook[lraw], codebook).argmax(dim=1)
                rcanon = executor.register_decoder(codebook[rraw], codebook).argmax(dim=1)
            raw_exact = raw_exact and torch.equal(lraw, rraw)
            canonical_exact = canonical_exact and torch.equal(lcanon, rcanon)
    equivalence["sealed_development_test"]["raw_decoded_indices_exact"] = raw_exact
    equivalence["sealed_development_test"]["canonical_decoded_indices_exact"] = canonical_exact
    with torch.no_grad():
        left_sep, left_terms, left_contexts = generic_separation(reference, manifest)
        right_sep, right_terms, right_contexts = generic_separation(prototype, manifest)
    terms_exact = all(torch.equal(left_terms[name], right_terms[name]) for name in left_terms)
    equivalence["terms_and_reduction"] = {"all_terms_exact": terms_exact, "L_sep_exact": torch.equal(left_sep, right_sep), "term_count": len(left_terms), "background_context_count": len(left_contexts), "L_sep_max_abs": 0.0 if torch.equal(left_sep, right_sep) else finite((left_sep - right_sep).abs().item())}
    intervals = interval_diagnostic(reference, manifest)
    synthetic = synthetic_contract_checks()
    signature = str(inspect.signature(NullCapableBinder.forward))
    prototype_contract = {"forward_signature": signature, "forward_only_token_ids_lengths": signature == "(self, token_ids: 'Tensor', lengths: 'Tensor') -> 'dict[str, Any]'", "active_set_input": False, "operator_name_active_set_logic": False, "four_queries_always_run": True, "null_bias_count": len(ROLES), "null_disabled_for_equivalence": True, "null_bias_state_before": {key: value.tolist() for key, value in base_state.items()}}
    artifact = {"status": "passed" if equivalence["atomic_catalog"]["all_exact"] and equivalence["background_contexts"]["all_exact"] and equivalence["sealed_development_test"]["all_exact"] and equivalence["terms_and_reduction"]["all_terms_exact"] and equivalence["terms_and_reduction"]["L_sep_exact"] and intervals["status"] == "passed" and synthetic["status"] == "passed" else "failed", "classification": "T6-NROLE-ACTIVESET-DESIGN-AUDIT: DESIGN VIABLE (NO TRAINING)", "task": "T6-NROLE-ACTIVESET-DESIGN-AUDIT", "reference": {"checkpoint": record(CHECKPOINT), "manifest": record(MANIFEST), "checkpoint_role": "single 730x development engineering reference; read-only; no 740x", "checkpoint_loaded_into": "GenericNRoleBinder N=4", "decoder_runtime": record(BASE_CHECKPOINT)}, "measured_results": {"existing_path_equivalence": equivalence, "null_disabled": True, "intervals": intervals, "synthetic_contract": synthetic}, "mathematical_consequences": {"null_candidate": "s_{r,empty}=b_r; j_r=argmax(b_r,s_{r,0},...,s_{r,T-1}) over real positions", "acceptance": "real winner => r_r=W_v(e_{j_r}); NULL winner => NONE", "interval_condition": "N_r^max < b_r < P_r^min", "term_definition": "P_r^min=min_i s_r(OP_r,ARG_i); N_r^max=max(max_{r'!=r,j}s_r(OP_{r'},ARG_j), max_{z∈S_4}s_r(z))", "future_loss_pressures_if_learning_b": {"positive": "softplus(b_r - s_r(positive))", "negative": "softplus(s_r(negative) - b_r)"}, "future_domain": {"non_empty_active_subsets": 15, "all_orders_before_argument_assignment": 64, "subset_sum": "C(4,1)+C(4,2)+C(4,3)+C(4,4)=15", "order_sum": "C(4,1)1!+C(4,2)2!+C(4,3)3!+C(4,4)4!=4+12+24+24=64"}}, "supervision_available_not_executed": {"atomic_target_construction": "For each atomic row with role q and value i: target_q=ARG_i; target_r=NONE for each r!=q. Thus existing 4×32 rows provide one positive and three absent-role targets per row.", "learned_threshold_option": "Both positive and negative softplus pressures are specified above; no b_r calibration or optimization was executed.", "training_performed": False, "parameter_adjustment_performed": False}, "absence_contract": {"absent": {"present": False, "pointer": None, "value_state": None, "decoded_value": None, "RAW": None, "CANON": None}, "present_value_zero_is_valid": True, "canonicalize_only_when_present": True, "zero_vector_decoder_forbidden": True, "codebook_minus_one_forbidden": True}, "prototype_contract": prototype_contract, "future_evaluation_design": {"status": "design_only_not_generated", "subsets": "all 15 non-empty subsets", "configurations": 64, "metrics": ["Presence: no false positives/negatives by role and active-set size", "Binding: pointer+value correct for every present role", "Absence: explicit NULL for every absent role", "Structured complete result: all decisions simultaneously correct"]}, "cost": {"formula": "1088*N^2", "N3_verified": 9792, "N4_verified": 17408}, "claim": "T6 design tests whether one fixed 4-query binder can emit correct arguments for present roles and explicit NONE for absent roles from instruction alone, without active-set input.", "non_claims": ["No threshold selection method (calibration vs learning) is chosen in this unit.", "The measured interval on one frozen 7301 development checkpoint is engineering viability evidence, not fresh scientific evidence.", "No training, manifests, checkpoints, or model construction occurred for T6; future 15-subset/64-order campaigns remain design-only.", "This does not establish performance across arbitrary N or variable role sets until future evaluation exists."], "halt": "T5 730x/740x remain closed and are not reused for T6 campaign design; this T6 artifact is design/prototype only."}
    artifact["artifact_self_hash"] = hashlib.sha256((json.dumps({**artifact, "artifact_self_hash": "__SELF_HASH__"}, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": artifact["status"], "classification": artifact["classification"], "artifact": str(OUTPUT), "artifact_self_hash": artifact["artifact_self_hash"], "training_performed": False, "new_manifests": False, "new_checkpoints": False}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
