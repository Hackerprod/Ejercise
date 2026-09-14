"""Execute authorized T7 Stage-C fresh NOOP learning on frozen A/B artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from execute_t6_activeset_midpoint_development import (  # noqa: E402
    ROLES,
    decode_states,
    generic_background_contexts,
    sha256_file,
    state_hash,
)
from execute_t7_noop_none_stage_a_fresh import fresh_manifest_view, verify_raw_self_hash  # noqa: E402
from repair_t7_noop_none_stage_b_atomic_sanity import atomic_sanity as repaired_atomic_sanity  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402


SEEDS = (7801, 7802, 7803, 7804, 7805)
UPDATES = 5000
DMODEL = 16
STAGE_A_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_fresh" / "training"
STAGE_B_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_fresh"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_c_fresh"
SCRIPT_PATH = Path(__file__).resolve()
REPAIR_PATH = SCRIPT_DIR / "repair_t7_noop_none_stage_b_atomic_sanity.py"
PLACEHOLDER = "__SELF_HASH__"


class HarnessError(RuntimeError):
    """Implementation or provenance failure; never classify as scientific failure."""


class FrozenBasePlusNoop(nn.Module):
    """Append-only embedding wrapper: frozen 37-row base plus trainable e_N."""

    def __init__(self, old_weights: Tensor, noop_weights: Tensor) -> None:
        super().__init__()
        self.old_embeddings = nn.Parameter(old_weights.detach().clone(), requires_grad=False)
        self.noop_embedding = nn.Parameter(noop_weights.detach().clone())

    def forward(self, token_ids: Tensor) -> Tensor:
        table = torch.cat((self.old_embeddings, self.noop_embedding.unsqueeze(0)), dim=0)
        return table[token_ids]


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"non-finite value: {result}")
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_digest(tensors: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json_bytes(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_self_hashed(path: Path, value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned["artifact_self_hash"] = PLACEHOLDER
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    value["artifact_self_hash"] = digest
    write_json_bytes(path, value)
    written = path.read_bytes()
    stored = json.loads(written.decode("utf-8"))["artifact_self_hash"]
    if sha256_bytes(written.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != stored:
        raise HarnessError(f"self-hash verification failed: {path}")
    return digest


def verify_json_self_hash(path: Path) -> tuple[dict[str, Any], str, str]:
    data = path.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    stored = parsed.get("artifact_self_hash")
    if not isinstance(stored, str):
        raise HarnessError(f"missing self-hash: {path}")
    if sha256_bytes(data.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != stored:
        raise HarnessError(f"invalid self-hash: {path}")
    return parsed, stored, sha256_bytes(data)


def load_stage_c_inputs(seed: int) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, str, str, str]:
    manifest_path = ROOT / "campaign" / "t7_noop_none_lexical_fresh_preparation" / "manifests" / f"manifest_{seed}_v1.json"
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("seed") != seed or raw_manifest.get("schema") != "T7-noop-none-lexical-fresh-preparation-manifest-v1":
        raise HarnessError(f"fresh manifest mismatch: {seed}")
    manifest = dict(raw_manifest)
    permutation = manifest.get("permutation")
    if not isinstance(permutation, list) or len(permutation) != VALUE_COUNT or sorted(permutation) != list(range(VALUE_COUNT)):
        raise HarnessError(f"fresh manifest permutation mismatch: {seed}")
    manifest["train"] = [{"argument": f"ARG_{index:02d}", "constraints": role.lower(), "kind": "atomic", "operator": manifest["operator_for_role"][role], "role": role, "value": int(permutation[index])} for role in ROLES for index in range(VALUE_COUNT)]
    manifest["multi_clause_train"] = 0
    manifest["joint_examples_in_training"] = 0
    manifest["test_rows_used"] = 0
    view = fresh_manifest_view(manifest)
    checkpoint_path = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
    calibration_path = STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
    a_result_path = STAGE_A_ROOT / f"seed_{seed}" / "results.json"
    if not checkpoint_path.exists() or not calibration_path.exists() or not a_result_path.exists():
        raise HarnessError(f"missing Stage-A/B input: {seed}")
    a_result, _, _ = verify_json_self_hash(a_result_path)
    a3 = a_result.get("A3", {})
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    if a3.get("checkpoint", {}).get("sha256") != checkpoint_sha or a3.get("checkpoint_sha256_after_audits") != checkpoint_sha:
        raise HarnessError(f"Stage-A checkpoint binding mismatch: {seed}")
    calibration, calibration_self_hash, calibration_file_sha = verify_json_self_hash(calibration_path)
    if calibration.get("seed") != seed or calibration.get("manifest", {}).get("sha256") != manifest_sha or calibration.get("core_checkpoint", {}).get("sha256") != checkpoint_sha:
        raise HarnessError(f"Stage-B association mismatch: {seed}")
    intervals = calibration.get("calibration", {}).get("intervals", {})
    thresholds = [float(intervals[role]["b_r_stored"]) for role in ROLES]
    if set(intervals) != set(ROLES) or thresholds != [float(value) for value in calibration["calibration"]["threshold_vector"]] or any(intervals[role]["stored_strictly_inside"] is not True for role in ROLES):
        raise HarnessError(f"Stage-B threshold binding mismatch: {seed}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("manifest_sha256") != manifest_sha or payload.get("base_token_order") != manifest["token_order"] or payload.get("base_token_ids_physical") != manifest["token_ids"]:
        raise HarnessError(f"Stage-A core mapping mismatch: {seed}")
    return manifest, view, manifest_path, checkpoint_path, calibration_path, calibration_self_hash, calibration_file_sha, a3["decoder_checkpoint"]["sha256"]


def load_core(seed: int, view: dict[str, Any], manifest: dict[str, Any], checkpoint_path: Path) -> tuple[GenericNRoleBinder, dict[str, Any], str, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    core = GenericNRoleBinder(view, seed, list(ROLES))
    core.load_state_dict(payload["encoder"], strict=True)
    if tuple(core.embedding.weight.shape) != (37, DMODEL):
        raise HarnessError(f"Stage-A core shape is not 37x16: {seed}")
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    return core, payload, state_hash(core.state_dict()), sha256_file(checkpoint_path)


def append_noop(core: GenericNRoleBinder, manifest: dict[str, Any], seed: int) -> tuple[GenericNRoleBinder, dict[str, Any], Tensor, int]:
    noop_operator = manifest["noop_operator"]
    base_order = list(manifest["token_order"])
    if noop_operator in base_order or manifest["stage_c_noop_internal_id"] != 37 or manifest["stage_c_token_order"] != [*base_order, noop_operator]:
        raise HarnessError(f"sealed append-only NOOP mapping mismatch: {seed}")
    physical_noop_id = int(manifest["stage_c_noop_external_id"])
    if physical_noop_id != int(manifest["id_block"][1]) + 1:
        raise HarnessError(f"sealed NOOP physical ID mismatch: {seed}")
    augmented = dict(manifest)
    augmented["token_order"] = list(manifest["stage_c_token_order"])
    augmented["token_ids"] = {token: manifest["token_ids"][token] for token in base_order} | {noop_operator: physical_noop_id}
    if augmented["token_order"][-1] != noop_operator:
        raise HarnessError(f"NOOP is not appended: {seed}")
    old_weights = core.embedding.weight.detach().clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(100000 + seed)
    initial_noop = torch.randn((DMODEL,), generator=generator)
    core.embedding = FrozenBasePlusNoop(old_weights, initial_noop)
    for name, parameter in core.named_parameters():
        parameter.requires_grad_(name == "embedding.noop_embedding")
    trainable = [name for name, parameter in core.named_parameters() if parameter.requires_grad]
    if trainable != ["embedding.noop_embedding"]:
        raise HarnessError(f"Stage-C trainable parameter set mismatch: {seed}")
    return core, augmented, initial_noop, physical_noop_id


def noop_contexts(core: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    if len(manifest["token_order"]) != 38 or tuple(manifest["token_order"][:-1]) != tuple(core.vocab.names) or manifest["token_order"][37] != manifest["noop_operator"]:
        raise HarnessError("Stage-C vocabulary is not append-only 37+1")
    noop = 37
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
    metadata.append({"family": "START→NOOP", "context": f"START→{manifest['noop_operator']}", "position": 0, "position_token": manifest["noop_operator"], "start_prefix": "zero; no BOS"})
    rows.append([link, noop])
    lengths.append(2)
    metadata.append({"family": "LINK→NOOP", "context": f"LINK→{manifest['noop_operator']}", "position": 1, "position_token": manifest["noop_operator"], "supervised_position_is_operator": True})
    if len(rows) != 34:
        raise HarnessError("Stage-C context cardinality is not 34")
    return torch.tensor(rows, dtype=torch.long), torch.tensor(lengths, dtype=torch.long), metadata


def noop_scores(core: GenericNRoleBinder, manifest: dict[str, Any]) -> tuple[Tensor, list[dict[str, Any]]]:
    ids, lengths, metadata = noop_contexts(core, manifest)
    details = core(ids, lengths)
    scores = torch.stack(tuple(details["scores"][query][index, metadata[index]["position"]] for query in range(len(ROLES)) for index in range(len(metadata)))).reshape(len(ROLES), len(metadata))
    if scores.shape != (4, 34):
        raise HarnessError(f"Stage-C score matrix shape is {scores.shape}")
    return scores, metadata


def score_table_and_certificates(scores: Tensor, metadata: list[dict[str, Any]], thresholds: list[float]) -> tuple[list[list[float]], dict[str, Any]]:
    table = [[finite(value) for value in row] for row in scores.detach().tolist()]
    threshold_tensor = torch.tensor(thresholds, dtype=scores.dtype)
    certificates: dict[str, Any] = {}
    families = {"NOOP→ARG": list(range(32)), "START→NOOP": [32], "LINK→NOOP": [33]}
    for query_index, role in enumerate(ROLES):
        row = scores[query_index].detach()
        maximum, maximum_index = row.max(dim=0)
        maximum_index_value = int(maximum_index.item())
        slack = finite(threshold_tensor[query_index].item() - maximum.item())
        certificate = {"role": role, "G": slack, "positive": bool(slack > 0.0), "max_score": finite(maximum.item()), "max_context": metadata[maximum_index_value]["context"], "max_position": metadata[maximum_index_value]["position"], "max_context_index": maximum_index_value, "threshold": finite(threshold_tensor[query_index].item()), "slack": slack, "family_maxima": {}}
        for family, indices in families.items():
            family_values = row[indices]
            family_max, family_local = family_values.max(dim=0)
            family_index = indices[int(family_local.item())]
            family_slack = finite(threshold_tensor[query_index].item() - family_max.item())
            certificate["family_maxima"][family] = {"max_score": finite(family_max.item()), "G": family_slack, "positive": bool(family_slack > 0.0), "context": metadata[family_index]["context"], "position": metadata[family_index]["position"], "context_index": family_index}
        certificates[role] = certificate
    return table, {"certificates": certificates, "certificate_count": len(certificates), "comparisons": len(ROLES) * len(metadata), "all_positive": len(certificates) == 4 and all(item["positive"] for item in certificates.values()), "families": list(families)}


def objective(core: GenericNRoleBinder, manifest: dict[str, Any], thresholds: list[float]) -> tuple[Tensor, Tensor]:
    scores, _ = noop_scores(core, manifest)
    differences = scores - torch.tensor(thresholds, dtype=scores.dtype).view(4, 1)
    if differences.shape != (4, 34):
        raise HarnessError(f"Stage-C difference matrix shape is {differences.shape}")
    return F.softplus(differences).sum() / 136.0, scores


def gradient_record(core: GenericNRoleBinder, manifest: dict[str, Any], thresholds: list[float], phase: str) -> dict[str, Any]:
    core.zero_grad(set_to_none=True)
    loss, _ = objective(core, manifest, thresholds)
    loss.backward()
    gradient = core.embedding.noop_embedding.grad
    if gradient is None or not bool(gradient.norm() > 0.0):
        raise HarnessError(f"e_N gradient disconnected or zero at {phase}")
    base_gradients = {name: parameter.grad for name, parameter in core.named_parameters() if name != "embedding.noop_embedding" and parameter.grad is not None}
    if base_gradients:
        raise HarnessError(f"base gradient appeared at {phase}: {list(base_gradients)}")
    result = {"phase": phase, "loss": finite(loss.item()), "e_N_norm": finite(gradient.norm().item()), "e_N_max_abs": finite(gradient.abs().max().item()), "e_N_nonzero": int((gradient != 0).sum().item()), "base_gradients_present": list(base_gradients), "only_nonzero_gradient_parameter": "embedding.noop_embedding", "trainable_parameter_names": [name for name, parameter in core.named_parameters() if parameter.requires_grad]}
    core.zero_grad(set_to_none=True)
    return result


def base_snapshot(core: GenericNRoleBinder, manifest: dict[str, Any]) -> dict[str, Tensor]:
    rows = manifest["train"]
    atomic_ids = torch.tensor([[core.vocab.encode(row["operator"]), core.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
    atomic_lengths = torch.full((len(rows),), 2, dtype=torch.long)
    contexts = generic_background_contexts(core, manifest)
    bg_ids = torch.tensor([context["token_ids"] for context in contexts], dtype=torch.long)
    bg_lengths = torch.full((len(contexts),), 2, dtype=torch.long)
    with torch.no_grad():
        atomic = core(atomic_ids, atomic_lengths)
        background = core(bg_ids, bg_lengths)
    tensors: dict[str, Tensor] = {"atomic.keys": atomic["keys"], "atomic.values": atomic["values"], "atomic.valid": atomic["valid"], "background.keys": background["keys"], "background.values": background["values"], "background.valid": background["valid"]}
    for index, role in enumerate(ROLES):
        tensors[f"atomic.scores.{role}"] = atomic["scores"][index]
        tensors[f"background.scores.{role}"] = background["scores"][index]
    return tensors


def compare_snapshots(before: dict[str, Tensor], after: dict[str, Tensor]) -> dict[str, Any]:
    mismatches = []
    for name in before:
        if not torch.equal(before[name], after[name]):
            mismatches.append({"name": name, "max_abs": finite((before[name] - after[name]).abs().max().item())})
    return {"tensor_count": len(before), "bit_exact": not mismatches, "mismatches": mismatches}


def noop_atomic_controls(core: GenericNRoleBinder, manifest: dict[str, Any], thresholds: list[float]) -> dict[str, Any]:
    rows = [[37, core.vocab.encode(f"ARG_{index:02d}")] for index in range(VALUE_COUNT)]
    ids = torch.tensor(rows, dtype=torch.long)
    lengths = torch.full((VALUE_COUNT,), 2, dtype=torch.long)
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32)
    with torch.no_grad():
        details = core(ids, lengths)
    decisions = 0
    null_decisions = 0
    complete = 0
    records: list[dict[str, Any]] = []
    for row_index in range(VALUE_COUNT):
        queries: dict[str, Any] = {}
        row_complete = True
        for query_index, role in enumerate(ROLES):
            scores = details["scores"][query_index][row_index, :2]
            best, pointer = scores.max(dim=0)
            present = bool(best.item() > threshold_tensor[query_index].item())
            decisions += 1
            null_ok = not present
            null_decisions += int(null_ok)
            row_complete = row_complete and null_ok
            queries[role] = {"presence": present, "pointer_if_present": int(pointer.item()) if present else None, "best_score": finite(best.item()), "threshold": finite(threshold_tensor[query_index].item()), "NULL": null_ok, "correct": null_ok}
        complete += int(row_complete)
        records.append({"row_index": row_index, "argument": f"ARG_{row_index:02d}", "queries": queries, "structured_complete": row_complete})
    return {"rows": VALUE_COUNT, "expected_rows": VALUE_COUNT, "decisions": decisions, "expected_decisions": 128, "NULL_decisions": null_decisions, "expected_NULL_decisions": 128, "structured_complete": complete, "expected_structured_complete": VALUE_COUNT, "passed": decisions == 128 and null_decisions == 128 and complete == VALUE_COUNT, "outputs": records}


def writer_probe() -> dict[str, Any]:
    path = OUTPUT_ROOT / "writer_byte_exact_probe.json"
    value = {"schema": "t7-stage-c-fresh-byte-exact-writer-probe-v1", "artifact_self_hash": PLACEHOLDER}
    digest = sha256_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    value["artifact_self_hash"] = digest
    written = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    if sha256_bytes(written.replace(digest.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != digest:
        raise HarnessError("Stage-C writer probe failed")
    return {"path": path.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": sha256_bytes(written), "verified": True, "write_method": "write_bytes(UTF-8)", "line_endings": {"CRLF": written.count(b"\r\n"), "LF": written.count(b"\n")}}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-C fresh root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    runner_hash = sha256_file(SCRIPT_PATH)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in executor.parameters()):
        raise HarnessError("decoder runtime not fully frozen")
    runtime_file_before = sha256_file(BASE_CHECKPOINT)
    runtime_state_before = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_before = tensor_digest({"codebook": codebook})
    per_seed: list[dict[str, Any]] = []
    decoder_a3_hashes: set[str] = set()
    try:
        for seed in SEEDS:
            manifest, view, manifest_path, checkpoint_path, calibration_path, calibration_self_hash, calibration_file_sha, decoder_a3_hash = load_stage_c_inputs(seed)
            decoder_a3_hashes.add(decoder_a3_hash)
            if decoder_a3_hash != runtime_file_before:
                raise HarnessError(f"decoder A3 hash mismatch: {seed}")
            core, payload_a, base_state_before, checkpoint_before = load_core(seed, view, manifest, checkpoint_path)
            base_manifest_state_before = {name: value.detach().clone() for name, value in core.state_dict().items()}
            base_scores_before = base_snapshot(core, view)
            thresholds = [float(json.loads(calibration_path.read_text(encoding="utf-8"))["calibration"]["intervals"][role]["b_r_stored"]) for role in ROLES]
            core, stage_c_manifest, initial_noop, noop_physical_id = append_noop(core, manifest, seed)
            initial_scores, context_metadata = noop_scores(core, stage_c_manifest)
            initial_table, initial_certificates = score_table_and_certificates(initial_scores, context_metadata, thresholds)
            initial_gradient = gradient_record(core, stage_c_manifest, thresholds, "initial")
            optimizer = torch.optim.AdamW([core.embedding.noop_embedding], lr=1e-3, weight_decay=0.0)
            last_loss = None
            nonzero_update_gradients = 0
            for step in range(1, UPDATES + 1):
                progress = (step - 1) / (UPDATES - 1)
                optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
                optimizer.zero_grad(set_to_none=True)
                loss, _ = objective(core, stage_c_manifest, thresholds)
                last_loss = finite(loss.item())
                loss.backward()
                gradient = core.embedding.noop_embedding.grad
                if gradient is None or not bool(gradient.norm() > 0.0):
                    raise HarnessError(f"e_N update gradient disconnected: {seed}/{step}")
                if any(parameter.grad is not None for name, parameter in core.named_parameters() if name != "embedding.noop_embedding"):
                    raise HarnessError(f"base update gradient appeared: {seed}/{step}")
                nonzero_update_gradients += 1
                optimizer.step()
            core.eval()
            final_gradient = gradient_record(core, stage_c_manifest, thresholds, "final")
            with torch.no_grad():
                final_loss, final_scores = objective(core, stage_c_manifest, thresholds)
            final_table, final_certificates = score_table_and_certificates(final_scores, context_metadata, thresholds)
            final_noop = core.embedding.noop_embedding.detach().clone()
            base_scores_after = base_snapshot(core, view)
            base_preservation = compare_snapshots(base_scores_before, base_scores_after)
            base_controls = repaired_atomic_sanity(core, view, executor, codebook, thresholds)
            noop_controls = noop_atomic_controls(core, stage_c_manifest, thresholds)
            projected_base_state = {name.replace("embedding.old_embeddings", "embedding.weight"): value for name, value in core.state_dict().items() if name != "embedding.noop_embedding"}
            base_state_after = state_hash(projected_base_state)
            base_state_bit_exact = all(name in projected_base_state and torch.equal(value, projected_base_state[name]) for name, value in base_manifest_state_before.items())
            checkpoint = OUTPUT_ROOT / "training" / f"seed_{seed}" / "final.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = {"encoder": core.state_dict(), "seed": seed, "updates": UPDATES, "stage": "C", "stage_a_checkpoint_sha256": checkpoint_before, "calibration_sha256": calibration_file_sha, "manifest_sha256": sha256_file(manifest_path), "noop_operator": manifest["noop_operator"], "noop_physical_id": noop_physical_id, "noop_internal_id": 37, "base_token_order": manifest["token_order"], "augmented_token_order": stage_c_manifest["token_order"], "base_vocab_rows": 37, "augmented_vocab_rows": 38, "trainable_parameter_names": ["embedding.noop_embedding"], "initialization_seed_e_N": 100000 + seed, "objective": "(1/136)*sum_r sum_z softplus(s_r(z;e_N)-b_r)", "contexts_per_update": 34, "difference_matrix_shape": [4, 34], "threshold_vector_role_order": thresholds, "stage_a_core_reused": True, "stage_b_calibration_reused": True, "old_buggy_stage_c_runner_used": False}
            torch.save(checkpoint_payload, checkpoint)
            checkpoint_c_sha = sha256_file(checkpoint)
            threshold_after = [float(value) for value in thresholds]
            result = {"schema": "t7-noop-none-lexical-stage-c-fresh-seed-v1", "seed": seed, "status": "trained", "manifest": source_record(manifest_path), "stage_a_input": {**source_record(checkpoint_path), "sha256_before": checkpoint_before, "sha256_after": sha256_file(checkpoint_path), "unchanged": checkpoint_before == sha256_file(checkpoint_path)}, "stage_b_input": {**source_record(calibration_path), "self_hash": calibration_self_hash, "file_sha256": calibration_file_sha, "threshold_vector_role_order": thresholds, "recalculated": False}, "mapping_augmented": {"noop_operator": manifest["noop_operator"], "base_token_order": manifest["token_order"], "stage_c_token_order_from_manifest": manifest["stage_c_token_order"], "augmented_token_order": stage_c_manifest["token_order"], "base_physical_to_internal": [{"token": token, "physical_id": int(manifest["token_ids"][token]), "internal_index": index} for index, token in enumerate(manifest["token_order"])], "noop_mapping": {"token": manifest["noop_operator"], "physical_id": noop_physical_id, "internal_index": 37}}, "initialization": {"seed_e_N": 100000 + seed, "distribution": "N(0,I)", "initial_e_N": initial_noop.tolist(), "initial_e_N_hash": tensor_digest({"e_N": initial_noop}), "single_initialization": True, "best_of_multiple_initializations": False}, "C0": {"initial_scores": initial_table, "initial_certificates": initial_certificates, "initial_gradient": initial_gradient, "optimizer_parameter_names": [name for name, parameter in core.named_parameters() if parameter.requires_grad], "optimizer_parameter_count": sum(parameter.numel() for parameter in optimizer.param_groups[0]["params"])}, "training": {"updates": UPDATES, "last_update_loss": last_loss, "final_loss": finite(final_loss.item()), "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "contexts_per_update": 34, "difference_matrix_shape": [4, 34], "terms": 136, "selection": "last update; no early stopping", "nonzero_e_N_gradient_updates": nonzero_update_gradients, "base_parameters_updated": False}, "C1": {"final_scores": final_table, "final_certificates": final_certificates, "certificate_count": 4, "all_4_positive": final_certificates["all_positive"], "final_e_N": final_noop.tolist(), "final_e_N_hash": tensor_digest({"e_N": final_noop}), "final_gradient": final_gradient}, "C2": {"base_scores_and_values_before_after": base_preservation, "base_state_hash_before": base_state_before, "base_state_projection_hash_after": base_state_after, "base_state_projection_bit_exact": base_state_bit_exact, "atomic_base_controls": base_controls, "decoder_codebook_used": True}, "C3": {"noop_atomic_controls": noop_controls}, "invariants": {"base_embeddings_updated": False, "queries_updated": False, "key_network_updated": False, "w_v_updated": False, "decoder_updated": False, "codebook_updated": False, "threshold_vector_before": thresholds, "threshold_vector_after": threshold_after, "thresholds_bit_exact": thresholds == threshold_after, "stage_c_checkpoint": {**source_record(checkpoint), "sha256": checkpoint_c_sha}}, "sanity_implementation": {"path": "scripts/repair_t7_noop_none_stage_b_atomic_sanity.py", "entrypoint": "atomic_sanity", "decoder_path": "decode_states(executor, codebook, selected)", "buggy_stage_b_atomic_sanity_executed": False}, "stage_c_gate": {"C0_gradient": initial_gradient["only_nonzero_gradient_parameter"] == "embedding.noop_embedding" and initial_gradient["e_N_norm"] > 0.0, "C1": final_certificates["all_positive"], "C2": base_preservation["bit_exact"] and base_state_bit_exact and base_controls["passed"], "C3": noop_controls["passed"], "invariance": checkpoint_before == sha256_file(checkpoint_path) and thresholds == threshold_after, "pass": final_certificates["all_positive"] and base_preservation["bit_exact"] and base_state_bit_exact and base_controls["passed"] and noop_controls["passed"] and checkpoint_before == sha256_file(checkpoint_path) and thresholds == threshold_after}}
            result_path = OUTPUT_ROOT / "results" / f"result_{seed}_v1.json"
            write_self_hashed(result_path, result)
            result["result"] = source_record(result_path)
            per_seed.append(result)
    except HarnessError as error:
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7 STAGE-C FRESH", "error_type": type(error).__name__, "error": str(error), "runner_sha256": runner_hash, "completed_seed_results": per_seed}
        write_json_bytes(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2
    runtime_file_after = sha256_file(BASE_CHECKPOINT)
    runtime_state_after = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook_after = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_hash_after = tensor_digest({"codebook": codebook_after})
    runtime_integrity = {"file_sha256_before": runtime_file_before, "file_sha256_after": runtime_file_after, "file_sha256_identical": runtime_file_before == runtime_file_after, "tensor_state_hash_before": runtime_state_before, "tensor_state_hash_after": runtime_state_after, "tensor_state_hash_identical": runtime_state_before == runtime_state_after, "codebook_hash_before": codebook_before, "codebook_hash_after": codebook_hash_after, "codebook_hash_identical": codebook_before == codebook_hash_after}
    probe = writer_probe()
    all_pass = len(per_seed) == 5 and len(decoder_a3_hashes) == 1 and all(item["stage_c_gate"]["pass"] for item in per_seed) and all(runtime_integrity[key] for key in ("file_sha256_identical", "tensor_state_hash_identical", "codebook_hash_identical"))
    source_paths = (SCRIPT_PATH, SCRIPT_DIR / "execute_t7_noop_none_stage_a_fresh.py", SCRIPT_DIR / "execute_t7_noop_none_stage_b_fresh.py", SCRIPT_DIR / "repair_t7_noop_none_stage_b_atomic_sanity.py", SCRIPT_DIR / "execute_t6_activeset_midpoint_development.py", SCRIPT_DIR / "t5_nrole_design_audit.py", SCRIPT_DIR / "ctrl2_common.py")
    consolidated = {"schema": "t7-noop-none-lexical-stage-c-fresh-v1", "status": "completed", "classification": "T7 STAGE-C FRESH: CLOSED/PASS" if all_pass else "T7 STAGE-C FRESH: VALID SCIENTIFIC FAIL", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION / STAGE-C FRESH", "authorization": "Sol-authorized Stage C fresh only; paired evaluation held", "summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "C1_certificates": len(per_seed) * 4, "C1_certificates_expected": 20, "C1_all_positive": all(item["C1"]["all_4_positive"] for item in per_seed), "C2_base_atomic_present": sum(item["C2"]["atomic_base_controls"]["present_decisions"] for item in per_seed), "C2_base_atomic_NULL_absent": sum(item["C2"]["atomic_base_controls"]["absent_decisions"] for item in per_seed), "C2_base_structured_complete": sum(item["C2"]["atomic_base_controls"]["structured_complete"] for item in per_seed), "C3_NOOP_instructions": sum(item["C3"]["noop_atomic_controls"]["structured_complete"] for item in per_seed), "C3_NOOP_decisions_NULL": sum(item["C3"]["noop_atomic_controls"]["NULL_decisions"] for item in per_seed), "C2_base_bit_exact": all(item["C2"]["base_scores_and_values_before_after"]["bit_exact"] and item["C2"]["base_state_projection_bit_exact"] for item in per_seed), "training_updates_per_seed": UPDATES, "only_e_N_optimized": True, "paired_evaluation_opened": False, "stage_c_completed": True}, "runner": {**source_record(SCRIPT_PATH), "sha256_used_for_run": runner_hash}, "inputs": {"stage_a_core_root": STAGE_A_ROOT.relative_to(ROOT).as_posix(), "stage_b_calibration_root": STAGE_B_ROOT.relative_to(ROOT).as_posix(), "fresh_manifests": [item["manifest"] for item in per_seed], "stage_a_checkpoints": [item["stage_a_input"] for item in per_seed], "stage_b_calibrations": [item["stage_b_input"] for item in per_seed]}, "sanity_implementation": {"path": "scripts/repair_t7_noop_none_stage_b_atomic_sanity.py", "commit": "fea9ea43f255893451230db4714360d409e0363f", "sha256": source_record(REPAIR_PATH)["sha256"], "entrypoint": "atomic_sanity", "decoder_path": "decode_states(executor, codebook, selected)", "buggy_stage_b_atomic_sanity_executed": False, "old_buggy_stage_c_development_runner_imported": False}, "runtime_integrity": runtime_integrity, "byte_exact_writer_probe": probe, "source_registry": [source_record(path) for path in source_paths], "per_seed_results": [item["result"] for item in per_seed], "per_seed": per_seed, "forbidden_operations_confirmed": ["paired multi-clause evaluation", "Stage D/next stage", "770x inputs", "Stage-B recalibration", "NOOP context measurement before initialization", "more than one e_N initialization"], "next_stage": "Paired evaluation remains held; no automatic continuation.", "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unavailable"}, "elapsed_seconds": time.perf_counter() - started}
    artifact_path = OUTPUT_ROOT / "results.json"
    write_self_hashed(artifact_path, consolidated)
    stored, _, file_sha = verify_json_self_hash(artifact_path)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_self_hash": stored["artifact_self_hash"], "file_sha256": file_sha, "seeds": len(per_seed), "C1_certificates": 20}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
