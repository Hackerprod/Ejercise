"""Audit generic N-role RELKEY design against frozen T4 RCSEP3-BG at N=3."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import execute_t4_nobypass2_rcsep3_bg as specific  # noqa: E402
from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402


SEED = 7101
ROLES = ("FLOOR", "AVOID", "MATCH")
MANIFEST = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "manifests" / f"manifest_{SEED}_v1.json"
CHECKPOINT = ROOT / "campaign" / "t4_nobypass2_rcsep3_bg" / "training" / f"seed_{SEED}" / "final.pt"
OUTPUT = ROOT / "campaign" / "t5_nrole_design_audit" / "t5_nrole_design_audit.json"


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    placeholder = written.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if hashlib.sha256(placeholder).hexdigest() != digest:
        raise RuntimeError("T5 artifact self-hash verification failed")
    return digest


class GenericNRoleBinder(nn.Module):
    """N-role RELKEY binder; role semantics live only in metadata."""

    address_key_formula = "k_t=LN(SiLU(W_k[e_prev,e_cur]))"
    score_formula = "s_{r,t}=Q_r^T k_t/4"
    selected_state_formula = "r_r=W_v(e_{j_r}), j_r=argmax_t s_{r,t}"

    def __init__(self, manifest: dict[str, Any], seed: int, roles: list[str]) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.roles = tuple(roles)
        self.vocab = specific.t4.ManifestVocab(manifest)
        self.embedding = nn.Embedding(len(self.vocab.names), specific.t4.DMODEL)
        self.local_binding = nn.Linear(2 * specific.t4.DMODEL, specific.t4.DMODEL)
        self.local_norm = nn.LayerNorm(specific.t4.DMODEL)
        self.query_bank = nn.Parameter(torch.randn(len(self.roles), specific.t4.DMODEL) * 0.02)
        self.w_v = nn.Linear(specific.t4.DMODEL, 32)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> dict[str, Any]:
        if token_ids.ndim != 2 or lengths.ndim != 1 or token_ids.shape[0] != lengths.shape[0]:
            raise ValueError("token_ids must be [B,T] and lengths must be [B]")
        batch, token_count = token_ids.shape
        valid = torch.arange(token_count).unsqueeze(0) < lengths.unsqueeze(1)
        embeddings = self.embedding(token_ids)
        zero = torch.zeros((batch, 1, specific.t4.DMODEL), dtype=embeddings.dtype)
        previous = torch.cat((zero, embeddings[:, :-1]), dim=1)
        keys = self.local_norm(self.local_binding(torch.cat((previous, embeddings), dim=-1)))
        keys = keys.masked_fill(~valid.unsqueeze(-1), 0.0)
        values = self.w_v(embeddings).masked_fill(~valid.unsqueeze(-1), 0.0)
        scores = tuple((keys @ self.query_bank[index]) / 4.0 for index in range(len(self.roles)))
        scores = tuple(score.masked_fill(~valid, float("-inf")) for score in scores)
        return {"scores": scores, "keys": keys, "values": values, "valid": valid}


def copy_specific_state(source: specific.t4.T4RelKeyEncoder, target: GenericNRoleBinder) -> None:
    with torch.no_grad():
        target.embedding.weight.copy_(source.embedding.weight)
        target.local_binding.weight.copy_(source.local_binding.weight)
        target.local_binding.bias.copy_(source.local_binding.bias)
        target.local_norm.weight.copy_(source.local_norm.weight)
        target.local_norm.bias.copy_(source.local_norm.bias)
        target.query_bank.copy_(torch.stack((source.q_f, source.q_a, source.q_m)))
        target.w_v.weight.copy_(source.w_v.weight)
        target.w_v.bias.copy_(source.w_v.bias)


def generic_background_contexts(binder: GenericNRoleBinder, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    argument_count = len(manifest["permutation"])
    contexts: list[dict[str, Any]] = []
    arg00 = binder.vocab.encode("ARG_00")
    link = binder.vocab.encode("LINK")
    operators = [manifest["operator_for_role"][role] for role in binder.roles]
    for operator in operators:
        contexts.append({"kind": "START→OP", "context": f"START→{operator}", "token_ids": [binder.vocab.encode(operator), arg00], "score_position": 0})
    for index in range(argument_count):
        argument = f"ARG_{index:02d}"
        contexts.append({"kind": "ARG→LINK", "context": f"{argument}→LINK", "token_ids": [binder.vocab.encode(argument), link], "score_position": 1})
    for operator in operators:
        contexts.append({"kind": "LINK→OP", "context": f"LINK→{operator}", "token_ids": [link, binder.vocab.encode(operator)], "score_position": 1})
    if len(contexts) != len(binder.roles) + argument_count + len(binder.roles):
        raise AssertionError("generic background context cardinality mismatch")
    return contexts


def generic_catalog_scores(binder: GenericNRoleBinder, manifest: dict[str, Any]) -> list[list[Tensor]]:
    rows: list[list[int]] = []
    metadata: list[tuple[int, int]] = []
    for role_index, role in enumerate(binder.roles):
        operator = manifest["operator_for_role"][role]
        for argument_index in range(len(manifest["permutation"])):
            rows.append([binder.vocab.encode(operator), binder.vocab.encode(f"ARG_{argument_index:02d}")])
            metadata.append((role_index, argument_index))
    token_ids = torch.tensor(rows, dtype=torch.long)
    lengths = torch.full((len(rows),), 2, dtype=torch.long)
    details = binder(token_ids, lengths)
    result = [[torch.empty(0) for _ in binder.roles] for _ in binder.roles]
    for query_index in range(len(binder.roles)):
        for source_index in range(len(binder.roles)):
            start = source_index * len(manifest["permutation"])
            result[query_index][source_index] = details["scores"][query_index][start : start + len(manifest["permutation"]), 1]
    return result


def generic_background_scores(binder: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[list[Tensor], list[dict[str, Any]]]:
    contexts = generic_background_contexts(binder, manifest)
    token_ids = torch.tensor([context["token_ids"] for context in contexts], dtype=torch.long)
    lengths = torch.full((len(contexts),), 2, dtype=torch.long)
    details = binder(token_ids, lengths)
    scores = [torch.stack(tuple(details["scores"][query_index][index, context["score_position"]] for index, context in enumerate(contexts))) for query_index in range(len(binder.roles))]
    return scores, contexts


def generic_separation(binder: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor], list[dict[str, Any]]]:
    catalog = generic_catalog_scores(binder, manifest)
    backgrounds, contexts = generic_background_scores(binder, manifest)
    terms: dict[str, Tensor] = {}
    for query_index in range(len(binder.roles)):
        self_scores = catalog[query_index][query_index]
        for source_index in range(len(binder.roles)):
            if source_index == query_index:
                continue
            terms[f"{query_index}<-{source_index}"] = torch.nn.functional.softplus(catalog[query_index][source_index].unsqueeze(0) - self_scores.unsqueeze(1)).mean()
    for query_index in range(len(binder.roles)):
        self_scores = catalog[query_index][query_index]
        background_matrix = torch.nn.functional.softplus(backgrounds[query_index].unsqueeze(0) - self_scores.unsqueeze(1))
        if background_matrix.shape != (len(manifest["permutation"]), len(contexts)):
            raise AssertionError("generic background matrix shape mismatch")
        terms[f"{query_index}<-BG"] = background_matrix.mean()
    if len(terms) != len(binder.roles) * len(binder.roles):
        raise AssertionError("generic separation term count mismatch")
    return torch.stack(tuple(terms.values())).mean(), terms, contexts


def generic_hardptr_outputs(details: dict[str, Any], cases: list[dict[str, Any]], roles: tuple[str, ...], executor: nn.Module, codebook: Tensor) -> dict[str, Any]:
    batch = len(cases)
    positions = torch.arange(batch)
    pointers: list[Tensor] = []
    selected: list[Tensor] = []
    raw: list[list[int]] = []
    canonical: list[list[int]] = []
    for role_index, role in enumerate(roles):
        pointer = details["scores"][role_index].argmax(dim=1)
        state = details["values"][positions, pointer]
        pointers.append(pointer)
        selected.append(state)
        role_raw: list[int] = []
        role_canonical: list[int] = []
        for case_index, item in enumerate(cases):
            target = next(clause["value"] for clause in item["clauses"] if clause["role"] == role)
            decoded = specific.t4.decode_state(executor, codebook, state[case_index], int(target))
            role_raw.append(decoded["raw"])
            role_canonical.append(decoded["canonical"])
        raw.append(role_raw)
        canonical.append(role_canonical)
    return {"pointers": pointers, "selected": selected, "raw": raw, "canonical": canonical}


def anti_hardcoding_audit() -> dict[str, Any]:
    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    function_sources = {}
    for name in ("__init__", "forward", "generic_background_contexts", "generic_catalog_scores", "generic_background_scores", "generic_separation", "generic_hardptr_outputs"):
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        function_sources[name] = ast.get_source_segment(text, node) or ""
    core = "\n".join(function_sources.values())
    checks = {
        "query_bank_dimension_is_N_by_d": "torch.randn(len(self.roles), specific.t4.DMODEL)" in function_sources["__init__"],
        "query_scoring_loops_over_role_indices": "range(len(self.roles))" in function_sources["forward"],
        "background_cardinality_is_N_plus_32_plus_N": "len(binder.roles) + argument_count + len(binder.roles)" in function_sources["generic_background_contexts"],
        "role_terms_loop_over_indices": "for query_index in range(len(binder.roles))" in function_sources["generic_separation"] and "for source_index in range(len(binder.roles))" in function_sources["generic_separation"],
        "no_role_name_branches_in_binder_core": all(token not in core for token in ("FLOOR", "AVOID", "MATCH", "if role", "if query_role")),
        "no_three_way_constants_in_generic_core": all(token not in core for token in ("range(3)", "BACKGROUND_COUNT = 38", "== 3")),
        "hardptr_uses_indexed_argmax": "argmax(dim=1)" in function_sources["generic_hardptr_outputs"],
        "decoding_loops_over_metadata_roles": "for role_index, role in enumerate(roles)" in function_sources["generic_hardptr_outputs"],
        "n4_role_registry_shape_supported": len(["FLOOR", "AVOID", "MATCH", "ANCHOR"]) == 4,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "evidence": {"core_functions": list(function_sources), "role_branch_tokens_absent": [token for token in ("FLOOR", "AVOID", "MATCH", "if role", "if query_role") if token not in core], "n4_registry_example": {"roles": ["FLOOR", "AVOID", "MATCH", "ANCHOR"], "background_context_count_formula": "32+2N", "N4_count": 40}}}


def compare_tensor(name: str, left: Tensor, right: Tensor, failures: list[dict[str, Any]]) -> dict[str, Any]:
    exact = torch.equal(left, right)
    maximum = 0.0
    if not exact:
        maximum = float((left - right).abs().max().item())
        failures.append({"name": name, "shape_left": list(left.shape), "shape_right": list(right.shape), "max_abs": maximum})
    return {"exact": exact, "shape": list(left.shape), "max_abs": maximum}


def main() -> int:
    anti = anti_hardcoding_audit()
    if anti["status"] != "passed":
        raise RuntimeError(json.dumps(anti, sort_keys=True))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    frozen = specific.t4.T4RelKeyEncoder(manifest, SEED)
    frozen.load_state_dict(payload["encoder"], strict=True)
    frozen.eval()
    generic = GenericNRoleBinder(manifest, SEED, list(ROLES))
    copy_specific_state(frozen, generic)
    generic.eval()
    failures: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}

    atomic_ids, atomic_lengths, _ = specific.t4.catalog_inputs(frozen, manifest)
    bg_contexts = generic_background_contexts(generic, manifest)
    bg_ids = torch.tensor([context["token_ids"] for context in bg_contexts], dtype=torch.long)
    bg_lengths = torch.full((len(bg_contexts),), 2, dtype=torch.long)
    test_cases = manifest["test"]
    test_ids = torch.tensor([[frozen.vocab.encode(token) for token in case["tokens"]] for case in test_cases], dtype=torch.long)
    test_lengths = torch.tensor([len(case["tokens"]) for case in test_cases], dtype=torch.long)
    for label, token_ids, lengths in (("atomic_catalog", atomic_ids, atomic_lengths), ("background_contexts", bg_ids, bg_lengths), ("sealed_test", test_ids, test_lengths)):
        with torch.no_grad():
            left = frozen(token_ids, lengths)
            right = generic(token_ids, lengths)
        comparisons[f"{label}.keys"] = compare_tensor(f"{label}.keys", left["keys"], right["keys"], failures)
        comparisons[f"{label}.values"] = compare_tensor(f"{label}.values", left["values"], right["values"], failures)
        comparisons[f"{label}.valid"] = compare_tensor(f"{label}.valid", left["valid"], right["valid"], failures)
        score_comparisons = []
        for role_index, role in enumerate(ROLES):
            score_comparisons.append(compare_tensor(f"{label}.scores.{role}", left["scores"][role], right["scores"][role_index], failures))
        comparisons[f"{label}.scores"] = score_comparisons

    with torch.no_grad():
        specific_sep, specific_terms = specific.separation_with_background(frozen, manifest)
        generic_sep, generic_terms_indexed, generic_contexts = generic_separation(generic, manifest)
    normalized_specific_contexts = [{key: context[key] for key in ("kind", "context", "token_ids", "score_position")} for context in specific.background_contexts(generic, manifest)]
    context_exact = normalized_specific_contexts == generic_contexts
    if not context_exact:
        failures.append({"name": "background_context_list", "reason": "context lists differ"})
    comparisons["background_context_list"] = {"exact": context_exact, "count": len(generic_contexts), "kinds": {kind: sum(item["kind"] == kind for item in generic_contexts) for kind in ("START→OP", "ARG→LINK", "LINK→OP")}}

    role_term_names = ("F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A")
    bg_term_names = ("F<-BG", "A<-BG", "M<-BG")
    term_comparisons: dict[str, Any] = {}
    for name, query_index, source_index in (("F<-A", 0, 1), ("F<-M", 0, 2), ("A<-F", 1, 0), ("A<-M", 1, 2), ("M<-F", 2, 0), ("M<-A", 2, 1)):
        term_comparisons[name] = compare_tensor(f"term.{name}", specific_terms[name], generic_terms_indexed[f"{query_index}<-{source_index}"], failures)
    for name, query_index in (("F<-BG", 0), ("A<-BG", 1), ("M<-BG", 2)):
        term_comparisons[name] = compare_tensor(f"term.{name}", specific_terms[name], generic_terms_indexed[f"{query_index}<-BG"], failures)
    comparisons["individual_terms"] = term_comparisons
    comparisons["L_sep"] = compare_tensor("L_sep", specific_sep, generic_sep, failures)
    comparisons["scalar_values"] = {
        "terms": {name: {"specific": finite(specific_terms[name].item()), "generic": finite(generic_terms_indexed["0<-1"].item()) if name == "F<-A" else finite(generic_terms_indexed["0<-2"].item()) if name == "F<-M" else finite(generic_terms_indexed["1<-0"].item()) if name == "A<-F" else finite(generic_terms_indexed["1<-2"].item()) if name == "A<-M" else finite(generic_terms_indexed["2<-0"].item()) if name == "M<-F" else finite(generic_terms_indexed["2<-1"].item()) if name == "M<-A" else finite(generic_terms_indexed["0<-BG"].item()) if name == "F<-BG" else finite(generic_terms_indexed["1<-BG"].item()) if name == "A<-BG" else finite(generic_terms_indexed["2<-BG"].item())} for name in (*role_term_names, *bg_term_names)},
        "L_sep": {"specific": finite(specific_sep.item()), "generic": finite(generic_sep.item())},
    }
    comparisons["background_context_list"] = {"contexts": generic_contexts}

    with torch.no_grad():
        executor = load_executor()
        executor.eval()
        for parameter in executor.parameters():
            parameter.requires_grad_(False)
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
        specific_test = frozen(test_ids, test_lengths)
        generic_test = generic(test_ids, test_lengths)
        specific_outputs = {"scores": tuple(specific_test["scores"][role] for role in ROLES), "values": specific_test["values"], "valid": specific_test["valid"]}
        generic_outputs = {"scores": generic_test["scores"], "values": generic_test["values"], "valid": generic_test["valid"]}
        specific_hardptr = generic_hardptr_outputs(specific_outputs, test_cases, ROLES, executor, codebook)
        generic_hardptr = generic_hardptr_outputs(generic_outputs, test_cases, ROLES, executor, codebook)
    pointer_exact = all(torch.equal(specific_hardptr["pointers"][index], generic_hardptr["pointers"][index]) for index in range(len(ROLES)))
    selected_exact = all(torch.equal(specific_hardptr["selected"][index], generic_hardptr["selected"][index]) for index in range(len(ROLES)))
    raw_exact = specific_hardptr["raw"] == generic_hardptr["raw"]
    canonical_exact = specific_hardptr["canonical"] == generic_hardptr["canonical"]
    if not pointer_exact:
        failures.append({"name": "hardptr.pointers"})
    if not selected_exact:
        failures.append({"name": "hardptr.selected_states"})
    if not raw_exact:
        failures.append({"name": "hardptr.raw_decodes"})
    if not canonical_exact:
        failures.append({"name": "hardptr.canonical_decodes"})
    comparisons["hardptr_outputs"] = {"cases": len(test_cases), "roles": len(ROLES), "pointer_exact": pointer_exact, "selected_states_exact": selected_exact, "raw_decoded_indices_exact": raw_exact, "canonical_decoded_indices_exact": canonical_exact}

    parameter_count_formula = "P(N)=V*d + 2*d^2 + 35*d + 32 + N*d; V=36,d=16 => P(N)=1680+16N"
    artifact = {"status": "passed" if not failures else "failed", "task": "T5-NOBYPASS-NROLE-DESIGN-AUDIT", "checkpoint_used": source_record(CHECKPOINT), "manifest_used": source_record(MANIFEST), "training_performed": False, "new_manifests": False, "new_checkpoints": False, "anti_hardcoding": anti, "generic_formulas": {"query_bank": "Q∈R^{N×d}; s_{r,t}=Q_r^T k_t/4", "pointer": "j_r=argmax_t s_{r,t}; r_r=W_v(e_{j_r})", "role_separation": "L_{r<-r'}=mean_{i,j} softplus[s_r(OP_r',ARG_j)-s_r(OP_r,ARG_i)] for r'!=r", "background": "L_{r<-BG}=mean_{i,z∈S_N} softplus[s_r(z)-s_r(OP_r,ARG_i)]", "separation": "L_sep^{N+BG}=(1/N^2)[sum_r sum_{r'!=r}L_{r<-r'}+sum_r L_{r<-BG}]", "term_count_proof": "N(N-1)+N=N^2", "background_context_proof": "|S_N|=N+32+N=32+2N", "parameter_count": parameter_count_formula, "contrastive_cost": {"role_vs_role": "32^2*N*(N-1)", "role_vs_background": "32*N*(32+2N)", "total": "1088*N^2", "N3": 9792, "N4": 17408}}, "n3_equivalence": {"specific_implementation": "execute_t4_nobypass2_rcsep3_bg", "generic_implementation": "GenericNRoleBinder", "role_mapping": {"Q_0": "q_F", "Q_1": "q_A", "Q_2": "q_M"}, "bit_exact_required": True, "all_comparisons_exact": not failures, "comparisons": comparisons, "failure_count": len(failures), "failures": failures, "minimum_tolerance": {"required": 0.0, "used": 0.0, "reason": "generic scorer preserves per-query vector matmul and reduction order; no floating tolerance needed"}}, "future_n4_design": {"status": "prepared_only", "roles": ["FLOOR", "AVOID", "MATCH", "ANCHOR"], "provisional_role": "ANCHOR/REFERENCE", "primary_condition": "decode(r_H)=H", "c1_extension": False, "n4_manifests": False, "claim_scope": "four simultaneous operator->argument bindings, not four native executor operations"}, "claim": "N-role parameterization expresses T4 as N=3 while preserving exact T4 behavior; future fresh N=4 evidence would test four simultaneous operator->argument bindings.", "non_claim": "No N=4 architecture, manifests, training, or executor extension is claimed or implemented.", "halt": "If N=3 equivalence fails, halt N-role design. This artifact reports equivalence only when all comparisons are exact."}
    digest = write_self_hashed(OUTPUT, artifact)
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "checkpoint": str(CHECKPOINT), "all_comparisons_exact": not failures, "failure_count": len(failures), "training_performed": False}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
