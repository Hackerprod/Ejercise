"""Execute only T7 Stage-C NOOP embedding development on frozen Stage-A cores."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from execute_t6_activeset_midpoint_development import decode_states  # noqa: E402
from execute_t7_noop_none_stage_a_development import (  # noqa: E402
    ROLES,
    SEEDS,
    load_t7_manifest,
    source_record,
    state_hash,
    write_self_hashed,
)
from t1_trainability.t7_production_core_noop_integration import AppendedNoopEmbedding  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder, generic_background_scores, generic_catalog_scores  # noqa: E402


STAGE_A_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_development" / "training"
STAGE_B_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_development"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_c_development"
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER = "__SELF_HASH__"
UPDATES = 5000
DMODEL = 16


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite Stage-C value: {result}")
    return result


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_self_hash(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    stored = parsed.get("artifact_self_hash")
    parsed["artifact_self_hash"] = PLACEHOLDER
    actual = hashlib.sha256((json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    if actual != stored:
        raise RuntimeError(f"invalid self-hash: {path}")
    return json.loads(data.decode("utf-8")), str(stored)


def load_stage_c_inputs(seed: int) -> tuple[dict[str, Any], Path, Path, dict[str, Any], Path]:
    base_manifest, manifest_path = load_t7_manifest(seed)
    checkpoint_path = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
    calibration_path = STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
    if not checkpoint_path.exists() or not calibration_path.exists():
        raise RuntimeError(f"missing Stage-A/B input for seed {seed}")
    calibration, calibration_self_hash = verify_self_hash(calibration_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if calibration["manifest"]["sha256"] != manifest_hash or calibration["core_checkpoint"]["sha256"] != checkpoint_hash:
        raise RuntimeError(f"Stage-A/B association mismatch for seed {seed}")
    intervals = calibration["calibration"]["intervals"]
    thresholds = [float(intervals[role]["b_r_stored"]) for role in ROLES]
    if thresholds != [float(value) for value in calibration["calibration"]["threshold_vector"]]:
        raise RuntimeError(f"role-named threshold order mismatch for seed {seed}")
    if any(intervals[role]["stored_strictly_inside"] is not True for role in ROLES):
        raise RuntimeError(f"reused calibration interval is not strict for seed {seed}")
    return base_manifest, manifest_path, checkpoint_path, {"payload": calibration, "self_hash": calibration_self_hash, "path": calibration_path, "thresholds": thresholds, "intervals": intervals}, checkpoint_path


def load_core(seed: int, checkpoint_path: Path, base_manifest: dict[str, Any]) -> tuple[GenericNRoleBinder, dict[str, Any], str, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("noop_in_stage_a_vocab") is not False or payload.get("noop_in_batch") is not False:
        raise RuntimeError(f"invalid Stage-A checkpoint provenance for seed {seed}")
    core = GenericNRoleBinder(base_manifest, seed, list(ROLES))
    core.load_state_dict(payload["encoder"], strict=True)
    if tuple(core.embedding.weight.shape) != (37, DMODEL):
        raise RuntimeError(f"Stage-A core shape is not [37,16] for seed {seed}")
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    return core, payload, state_hash(core.state_dict()), hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()


def append_noop(core: GenericNRoleBinder, manifest: dict[str, Any], seed: int) -> tuple[GenericNRoleBinder, dict[str, Any], torch.Tensor, torch.Tensor]:
    noop_operator = manifest["noop_operator"]
    base_tokens = list(manifest["token_order"])
    if noop_operator in base_tokens:
        raise RuntimeError(f"NOOP already present in base vocabulary for seed {seed}")
    physical_noop_id = int(manifest["id_block"][1]) + 1
    augmented = dict(manifest)
    augmented["token_order"] = [*base_tokens, noop_operator]
    augmented["token_ids"] = {**manifest["token_ids"], noop_operator: physical_noop_id}
    old_weights = core.embedding.weight.detach().clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(100000 + seed)
    initial_noop = torch.randn((DMODEL,), generator=generator)
    core.embedding = AppendedNoopEmbedding(old_weights, initial_noop)
    for name, parameter in core.named_parameters():
        parameter.requires_grad_(name == "embedding.noop_embedding")
    if [name for name, parameter in core.named_parameters() if parameter.requires_grad] != ["embedding.noop_embedding"]:
        raise RuntimeError(f"Stage-C trainable parameter set mismatch for seed {seed}")
    return core, augmented, initial_noop, torch.tensor(physical_noop_id, dtype=torch.long)


def noop_contexts(core: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if len(manifest["token_order"]) != 38 or tuple(manifest["token_order"][:-1]) != tuple(core.vocab.names):
        raise RuntimeError("Stage-C vocabulary is not append-only [37 base rows + 1 NOOP row]")
    noop = len(manifest["token_order"]) - 1
    link = core.vocab.encode("LINK")
    arguments = [core.vocab.encode(f"ARG_{index:02d}") for index in range(VALUE_COUNT)]
    rows: list[list[int]] = []
    lengths: list[int] = []
    metadata: list[dict[str, Any]] = []
    for index, argument in enumerate(arguments):
        rows.append([noop, argument])
        lengths.append(2)
        metadata.append({"family": "NOOP→ARG", "context": f"{manifest['noop_operator']}→ARG_{index:02d}", "position": 1, "position_token": f"ARG_{index:02d}", "argument_index": index})
    rows.append([noop, arguments[0]])
    lengths.append(1)
    metadata.append({"family": "START→NOOP", "context": f"START→{manifest['noop_operator']}", "position": 0, "position_token": manifest["noop_operator"]})
    rows.append([link, noop])
    lengths.append(2)
    metadata.append({"family": "LINK→NOOP", "context": f"LINK→{manifest['noop_operator']}", "position": 1, "position_token": manifest["noop_operator"]})
    if len(rows) != 34:
        raise RuntimeError("NOOP context cardinality mismatch")
    return torch.tensor(rows, dtype=torch.long), torch.tensor(lengths, dtype=torch.long), metadata


def noop_scores(core: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    ids, lengths, metadata = noop_contexts(core, manifest)
    details = core(ids, lengths)
    scores = torch.stack(tuple(details["scores"][query][index, metadata[index]["position"]] for query in range(len(ROLES)) for index in range(len(metadata)))).reshape(len(ROLES), len(metadata))
    return scores, metadata


def score_table_and_certificates(scores: torch.Tensor, metadata: list[dict[str, Any]], thresholds: list[float]) -> tuple[list[list[float]], dict[str, Any]]:
    values = [[finite(value) for value in row] for row in scores.detach().tolist()]
    certificates: dict[str, Any] = {}
    families = {"NOOP→ARG": list(range(32)), "START→NOOP": [32], "LINK→NOOP": [33]}
    threshold_tensor = torch.tensor(thresholds, dtype=scores.dtype)
    for query_index, role in enumerate(ROLES):
        row = scores[query_index].detach()
        maximum, maximum_index = row.max(dim=0)
        certificate = {"role": role, "G": finite(threshold_tensor[query_index].item() - maximum.item()), "positive": bool(threshold_tensor[query_index].item() - maximum.item() > 0.0), "max_score": finite(maximum.item()), "max_context": metadata[int(maximum_index.item())]["context"], "max_position": metadata[int(maximum_index.item())]["position"], "max_context_index": int(maximum_index.item()), "threshold": finite(threshold_tensor[query_index].item()), "slack": finite(threshold_tensor[query_index].item() - maximum.item()), "family_maxima": {}}
        for family, indices in families.items():
            family_tensor = row[indices]
            family_max, family_local = family_tensor.max(dim=0)
            family_index = indices[int(family_local.item())]
            certificate["family_maxima"][family] = {"max_score": finite(family_max.item()), "G": finite(threshold_tensor[query_index].item() - family_max.item()), "positive": bool(threshold_tensor[query_index].item() - family_max.item() > 0.0), "context": metadata[family_index]["context"], "position": metadata[family_index]["position"], "context_index": family_index}
        certificates[role] = certificate
    return values, {"certificates": certificates, "certificate_count": len(certificates), "comparisons": len(ROLES) * len(metadata), "all_positive": len(certificates) == 4 and all(item["positive"] for item in certificates.values()), "families": list(families)}


def objective(core: GenericNRoleBinder, manifest: dict[str, Any], thresholds: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    scores, _ = noop_scores(core, manifest)
    threshold_tensor = torch.tensor(thresholds, dtype=scores.dtype).view(4, 1)
    differences = scores - threshold_tensor
    if differences.shape != (4, 34):
        raise RuntimeError(f"Stage-C difference matrix shape is {differences.shape}")
    return F.softplus(differences).mean(), scores


def gradient_record(core: GenericNRoleBinder, manifest: dict[str, Any], thresholds: list[float], phase: str) -> dict[str, Any]:
    core.zero_grad(set_to_none=True)
    loss, _ = objective(core, manifest, thresholds)
    loss.backward()
    gradient = core.embedding.noop_embedding.grad
    if gradient is None:
        raise RuntimeError(f"e_N gradient disconnected at {phase}")
    base_gradients = {name: parameter.grad for name, parameter in core.named_parameters() if name != "embedding.noop_embedding" and parameter.grad is not None}
    if base_gradients:
        raise RuntimeError(f"base parameter gradient appeared at {phase}: {list(base_gradients)}")
    result = {"phase": phase, "loss": finite(loss.item()), "e_N_norm": finite(gradient.norm().item()), "e_N_max_abs": finite(gradient.abs().max().item()), "e_N_nonzero": int((gradient != 0).sum().item()), "base_gradients_present": list(base_gradients), "trainable_parameter_names": [name for name, parameter in core.named_parameters() if parameter.requires_grad]}
    core.zero_grad(set_to_none=True)
    return result


def base_snapshot(core: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = manifest["train"]
    atomic_ids = torch.tensor([[core.vocab.encode(row["operator"]), core.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
    atomic_lengths = torch.full((len(rows),), 2, dtype=torch.long)
    bg_scores, bg_contexts = generic_background_scores(core, manifest)
    bg_ids = torch.tensor([context["token_ids"] for context in bg_contexts], dtype=torch.long)
    bg_lengths = torch.full((len(bg_contexts),), 2, dtype=torch.long)
    with torch.no_grad():
        atomic = core(atomic_ids, atomic_lengths)
        background = core(bg_ids, bg_lengths)
    tensors = {"atomic.keys": atomic["keys"], "atomic.values": atomic["values"], "atomic.valid": atomic["valid"], "background.keys": background["keys"], "background.values": background["values"], "background.valid": background["valid"]}
    for index, role in enumerate(ROLES):
        tensors[f"atomic.scores.{role}"] = atomic["scores"][index]
        tensors[f"background.scores.{role}"] = background["scores"][index]
    return tensors, {"atomic_ids": atomic_ids, "atomic_lengths": atomic_lengths, "bg_ids": bg_ids, "bg_lengths": bg_lengths, "bg_contexts": bg_contexts}


def compare_snapshots(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, Any]:
    mismatches = []
    for name in before:
        if not torch.equal(before[name], after[name]):
            mismatches.append({"name": name, "max_abs": finite((before[name] - after[name]).abs().max().item())})
    return {"tensor_count": len(before), "bit_exact": not mismatches, "mismatches": mismatches}


def atomic_controls(core: GenericNRoleBinder, manifest: dict[str, Any], executor: torch.nn.Module, codebook: torch.Tensor, thresholds: list[float], noop: bool) -> dict[str, Any]:
    if noop:
        rows = [{"role": "NOOP", "operator": manifest["noop_operator"], "argument": f"ARG_{index:02d}", "value": None} for index in range(VALUE_COUNT)]
    else:
        rows = manifest["train"]
    noop_id = len(manifest["token_order"]) - 1 if noop else None
    ids = torch.tensor([[(noop_id if noop else core.vocab.encode(row["operator"])), core.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
    lengths = torch.full((len(rows),), 2, dtype=torch.long)
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32)
    with torch.no_grad():
        details = core(ids, lengths)
    present = absent = pointer_correct = raw_correct = canon_correct = complete = 0
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_complete = True
        query_records: dict[str, Any] = {}
        for query_index, query_role in enumerate(ROLES):
            scores = details["scores"][query_index][row_index, :2]
            best, pointer = scores.max(dim=0)
            is_present = bool(best.item() > threshold_tensor[query_index].item())
            expected_present = (not noop) and query_role == row["role"]
            if is_present:
                present += 1
                selected = details["values"][row_index, int(pointer.item())].unsqueeze(0)
                with torch.no_grad():
                    raw, canon = decode_states(executor, codebook, selected)
                raw_value = int(raw[0].item())
                canon_value = int(canon[0].item())
                pointer_ok = expected_present and int(pointer.item()) == 1
                raw_ok = pointer_ok and raw_value == int(row["value"])
                canon_ok = pointer_ok and canon_value == int(row["value"])
                pointer_correct += int(pointer_ok)
                raw_correct += int(raw_ok)
                canon_correct += int(canon_ok)
                correct = pointer_ok and raw_ok and canon_ok and raw_value == canon_value
                row_complete = row_complete and correct
                query_records[query_role] = {"presence": True, "pointer": int(pointer.item()), "score": finite(best.item()), "expected_present": expected_present, "pointer_correct": pointer_ok, "RAW": raw_value, "CANON": canon_value, "RAW_correct": raw_ok, "CANON_correct": canon_ok, "RAW_CANON_equal": raw_value == canon_value, "correct": correct}
            else:
                absent += 1
                expected_absent = noop or query_role != row["role"]
                null_ok = expected_absent
                row_complete = row_complete and null_ok
                query_records[query_role] = {"presence": False, "pointer": None, "score": finite(best.item()), "expected_absent": expected_absent, "NULL": True, "correct": null_ok}
        complete += int(row_complete)
        records.append({"row_index": row_index, "role": row["role"], "argument": row["argument"], "expected_value": row["value"], "queries": query_records, "structured_complete": row_complete})
    expected_present = 0 if noop else 128
    expected_absent = len(rows) * 4 if noop else 384
    expected_complete = len(rows)
    return {"rows": len(rows), "decisions": len(rows) * 4, "present_decisions": present, "absent_decisions": absent, "present_pointer_correct": pointer_correct, "present_RAW_correct": raw_correct, "present_CANON_correct": canon_correct, "absent_NULL_correct": absent if noop else absent, "structured_complete": complete, "expected_present_decisions": expected_present, "expected_absent_decisions": expected_absent, "expected_structured_complete": expected_complete, "passed": present == expected_present and absent == expected_absent and pointer_correct == expected_present and raw_correct == expected_present and canon_correct == expected_present and complete == expected_complete, "outputs": records}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-C output root: {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    runtime_file_before = hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()
    runtime_state_before = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_before = tensor_digest({"codebook": codebook})
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        base_manifest, manifest_path, checkpoint_path, calibration_binding, _ = load_stage_c_inputs(seed)
        core, checkpoint_payload, base_state_before, checkpoint_hash_before = load_core(seed, checkpoint_path, base_manifest)
        base_scores_before, base_context_input = base_snapshot(core, base_manifest)
        core, stage_c_manifest, initial_noop, noop_physical_id = append_noop(core, base_manifest, seed)
        thresholds = calibration_binding["thresholds"]
        initial_scores, context_metadata = noop_scores(core, stage_c_manifest)
        initial_table, initial_certificates = score_table_and_certificates(initial_scores, context_metadata, thresholds)
        initial_gradient = gradient_record(core, stage_c_manifest, thresholds, "initial")
        optimizer = torch.optim.AdamW([core.embedding.noop_embedding], lr=1e-3, weight_decay=0.0)
        last_loss = None
        for step in range(1, UPDATES + 1):
            progress = (step - 1) / (UPDATES - 1)
            optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(core, stage_c_manifest, thresholds)
            last_loss = finite(loss.item())
            loss.backward()
            if core.embedding.noop_embedding.grad is None:
                raise RuntimeError(f"e_N gradient disconnected during update for {seed}")
            optimizer.step()
        core.eval()
        final_gradient = gradient_record(core, stage_c_manifest, thresholds, "final")
        with torch.no_grad():
            final_loss, final_scores = objective(core, stage_c_manifest, thresholds)
        final_loss_value = finite(final_loss.item())
        final_table, final_certificates = score_table_and_certificates(final_scores, context_metadata, thresholds)
        final_noop = core.embedding.noop_embedding.detach().clone()
        base_scores_after, _ = base_snapshot(core, base_manifest)
        base_preservation = compare_snapshots(base_scores_before, base_scores_after)
        base_controls = atomic_controls(core, base_manifest, executor, codebook, thresholds, noop=False)
        noop_controls = atomic_controls(core, stage_c_manifest, executor, codebook, thresholds, noop=True)
        base_state_after = tensor_digest({name.replace("embedding.old_embeddings", "embedding.weight"): value for name, value in core.state_dict().items() if name != "embedding.noop_embedding"})
        checkpoint = OUTPUT_ROOT / "training" / f"seed_{seed}" / "final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload_c = {"encoder": core.state_dict(), "seed": seed, "updates": UPDATES, "stage": "C", "stage_a_checkpoint_sha256": checkpoint_hash_before, "calibration_sha256": hashlib.sha256(calibration_binding["path"].read_bytes()).hexdigest(), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "noop_operator": base_manifest["noop_operator"], "noop_physical_id": int(noop_physical_id.item()), "base_token_order": base_manifest["token_order"], "augmented_token_order": stage_c_manifest["token_order"], "base_vocab_rows": 37, "augmented_vocab_rows": 38, "trainable_parameter_names": ["embedding.noop_embedding"], "initialization_seed_e_N": 100000 + seed, "objective": "(1/136)*sum_r sum_z softplus(s_r(z;e_N)-b_r)", "contexts_per_update": 34, "threshold_vector_role_order": thresholds}
        torch.save(checkpoint_payload_c, checkpoint)
        checkpoint_hash_c = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        core_projection_before = {name: value for name, value in core.state_dict().items() if name != "embedding.noop_embedding"}
        e_n_initial_hash = tensor_digest({"e_N": initial_noop})
        e_n_final_hash = tensor_digest({"e_N": final_noop})
        result = {"schema": "T7-noop-none-lexical-stage-c-development-seed-v1", "seed": seed, "status": "trained", "manifest": source_record(manifest_path), "stage_a_checkpoint": {**source_record(checkpoint_path), "sha256_before": checkpoint_hash_before, "sha256_after": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "unchanged": checkpoint_hash_before == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}, "calibration_reused": {**source_record(calibration_binding["path"]), "self_hash": calibration_binding["self_hash"], "recalculated": False, "threshold_vector_role_order": thresholds, "intervals_by_role": calibration_binding["intervals"]}, "mapping_augmented": {"noop_operator": base_manifest["noop_operator"], "base_token_order": base_manifest["token_order"], "augmented_token_order": stage_c_manifest["token_order"], "base_physical_to_internal": [{"token": token, "physical_id": int(base_manifest["token_ids"][token]), "internal_index": index} for index, token in enumerate(base_manifest["token_order"])], "noop_mapping": {"token": base_manifest["noop_operator"], "physical_id": int(noop_physical_id.item()), "internal_index": 37}}, "initialization": {"seed_e_N": 100000 + seed, "distribution": "normal standard", "initial_e_N": initial_noop.tolist(), "initial_e_N_hash": e_n_initial_hash, "unique_per_seed": True}, "C0": {"base_state_hash_before": base_state_before, "base_state_projection_hash_after": base_state_after, "base_state_projection_bit_exact": base_state_before == base_state_after, "initial_scores": initial_table, "initial_certificates": initial_certificates, "initial_gradient": initial_gradient, "optimizer_parameter_names": [name for name, parameter in core.named_parameters() if parameter.requires_grad], "optimizer_parameter_count": sum(parameter.numel() for parameter in optimizer.param_groups[0]["params"])}, "training": {"updates": UPDATES, "last_update_loss": last_loss, "final_loss": final_loss_value, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "contexts_per_update": 34, "difference_matrix_shape": [4, 34], "terms": 136, "selection": "last update; no early stopping"}, "C1": {"final_scores": final_table, "final_certificates": final_certificates, "all_4_positive": final_certificates["all_positive"], "final_gradient": final_gradient}, "C2": {"base_scores_and_values_before_after": base_preservation, "atomic_base_controls": base_controls, "decoder_codebook_used": True}, "C3": {"noop_atomic_controls": noop_controls}, "invariants": {"base_state_hash_before": base_state_before, "base_state_hash_after": base_state_after, "base_state_bit_exact": base_state_before == base_state_after, "old_embedding_hash": tensor_digest({"old_embeddings": core.embedding.old_embeddings}), "e_N_final_hash": e_n_final_hash, "threshold_vector_before": thresholds, "threshold_vector_after": thresholds, "thresholds_bit_exact": thresholds == [float(value) for value in calibration_binding["thresholds"]], "stage_c_checkpoint": source_record(checkpoint), "stage_c_checkpoint_sha256": checkpoint_hash_c}, "forbidden_operations": {"binder_retrained": False, "base_embeddings_updated": False, "queries_updated": False, "key_network_updated": False, "w_v_updated": False, "decoder_updated": False, "codebook_updated": False, "thresholds_recalculated": False, "multi_clause_evaluation": False, "stage_b_recalibration": False, "noop_scores_before_initialization": False}}
        result["C1"]["final_e_N"] = final_noop.tolist()
        result_path = OUTPUT_ROOT / "results" / f"result_{seed}_v1.json"
        write_self_hashed(result_path, result)
        result["result"] = source_record(result_path)
        per_seed.append(result)
    runtime_file_after = hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()
    runtime_state_after = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook_after_tensor = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_after = tensor_digest({"codebook": codebook_after_tensor})
    runtime_integrity = {"checkpoint": source_record(BASE_CHECKPOINT), "file_sha256_before": runtime_file_before, "file_sha256_after": runtime_file_after, "file_sha256_identical": runtime_file_before == runtime_file_after, "tensor_state_hash_before": runtime_state_before, "tensor_state_hash_after": runtime_state_after, "tensor_state_hash_identical": runtime_state_before == runtime_state_after, "codebook_hash_before": codebook_before, "codebook_hash_after": codebook_after, "codebook_hash_identical": codebook_before == codebook_after}
    all_pass = all(item["C1"]["all_4_positive"] and item["C2"]["base_scores_and_values_before_after"]["bit_exact"] and item["C2"]["atomic_base_controls"]["passed"] and item["C3"]["noop_atomic_controls"]["passed"] and item["invariants"]["base_state_bit_exact"] for item in per_seed) and runtime_integrity["file_sha256_identical"] and runtime_integrity["tensor_state_hash_identical"] and runtime_integrity["codebook_hash_identical"]
    consolidated = {"schema": "T7-noop-none-lexical-stage-c-development-v1", "status": "completed", "classification": "T7 STAGE-C DEVELOPMENT: CLOSED/PASS" if all_pass else "T7 STAGE-C DEVELOPMENT: VALID FAIL", "task": "T7-NOOP-NONE-LEXICAL / STAGE-C DEVELOPMENT", "authorization": "Sol-authorized Stage C only; multi-clause evaluation held", "summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "updates_per_seed": UPDATES, "trainable_dimension_per_seed": 16, "initial_certificates": 20, "final_certificates": 20, "final_certificates_positive": all(item["C1"]["all_4_positive"] for item in per_seed), "base_scores_values_bit_exact_all_seeds": all(item["C2"]["base_scores_and_values_before_after"]["bit_exact"] for item in per_seed), "base_atomic_present": sum(item["C2"]["atomic_base_controls"]["present_pointer_correct"] for item in per_seed), "base_atomic_absent": sum(item["C2"]["atomic_base_controls"]["absent_decisions"] for item in per_seed), "base_atomic_complete": sum(item["C2"]["atomic_base_controls"]["structured_complete"] for item in per_seed), "noop_atomic_null_decisions": sum(item["C3"]["noop_atomic_controls"]["absent_decisions"] for item in per_seed), "noop_atomic_complete": sum(item["C3"]["noop_atomic_controls"]["structured_complete"] for item in per_seed), "training": True, "multi_clause_evaluation": False, "NOOP_shared_between_seeds": False}, "runner": source_record(SCRIPT_PATH), "core_source": source_record(SCRIPT_DIR / "t5_nrole_design_audit.py"), "decoder_runtime": runtime_integrity, "stage_a_checkpoints": [item["stage_a_checkpoint"] for item in per_seed], "calibrations_reused": [item["calibration_reused"] for item in per_seed], "stage_c_checkpoints": [item["invariants"]["stage_c_checkpoint"] for item in per_seed], "per_seed_results": [item["result"] for item in per_seed], "per_seed": per_seed, "elapsed_seconds": time.perf_counter() - started, "next_stage": "Multi-clause evaluation remains held; no automatic continuation."}
    artifact_path = OUTPUT_ROOT / "results.json"
    digest = write_self_hashed(artifact_path, consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(), "artifact_self_hash": digest, "seeds": len(per_seed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
