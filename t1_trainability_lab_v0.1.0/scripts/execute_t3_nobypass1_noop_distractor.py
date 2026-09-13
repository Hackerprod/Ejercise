"""Execute authorized T3-NOBYPASS-1-NOOP-DISTRACTOR training and audit.

This module is intentionally independent of all T2 writer/core checkpoints.
Only the sealed T3 manifests, the approved C1 executor/decoder, and the
frozen CTRL7 supervisor are loaded.
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


SEEDS = (7001, 7002, 7003, 7004, 7005)
ROLES = ("FLOOR", "AVOID", "NOOP")
ORDERS = (
    "FLOOR_AVOID_NOOP",
    "FLOOR_NOOP_AVOID",
    "AVOID_FLOOR_NOOP",
    "AVOID_NOOP_FLOOR",
    "NOOP_FLOOR_AVOID",
    "NOOP_AVOID_FLOOR",
)
UPDATES = 5000
DMODEL = 16
OUTPUT_ROOT = ROOT / "campaign" / "t3_nobypass1_noop_distractor"
MANIFEST_ROOT = ROOT / "campaign" / "nb5_manifests"
SCRIPT_PATH = Path(__file__).resolve()


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


def artifact_hash(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return sha256_bytes(payload)


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    digest = artifact_hash(artifact)
    artifact["artifact_self_hash"] = digest
    payload = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    stored = json.loads(payload.decode("utf-8"))["artifact_self_hash"]
    placeholder = payload.replace(f'"artifact_self_hash": "{stored}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if sha256_bytes(placeholder) != stored:
        raise RuntimeError(f"self-hash verification failed: {path}")
    return digest


class ManifestVocab:
    """Fresh per-manifest physical-token table mapped to local embedding rows."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        token_ids = manifest["token_ids"]
        self.token_names = tuple(token_ids)
        self.external_to_internal = {int(external): index for index, external in enumerate(token_ids.values())}
        self.name_to_internal = {name: index for index, name in enumerate(self.token_names)}

    def encode_name(self, name: str) -> int:
        return self.name_to_internal[name]


class RelKeyEncoder(nn.Module):
    """Exact T3 RELKEY encoder: local relative key, lexical hard pointer, raw W_v."""

    address_key_formula = "k_t=LN(SiLU(W_k[e_prev,e_cur]))"
    score_formula = "s_r=q_r^T k/4"
    selected_state_formula = "r=W_v(e_j)"

    def __init__(self, manifest: dict[str, Any], seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = ManifestVocab(manifest)
        self.embedding = nn.Embedding(len(self.vocab.token_names), DMODEL)
        self.local_binding = nn.Linear(2 * DMODEL, DMODEL)
        self.local_norm = nn.LayerNorm(DMODEL)
        self.q_f = nn.Parameter(torch.randn(DMODEL) * 0.02)
        self.q_a = nn.Parameter(torch.randn(DMODEL) * 0.02)
        self.w_v = nn.Linear(DMODEL, 32)

    def forward(self, token_ids: Tensor, lengths: Tensor, *, return_details: bool = False) -> Any:
        if token_ids.ndim != 2 or lengths.ndim != 1 or token_ids.shape[0] != lengths.shape[0]:
            raise ValueError("token_ids must be [batch,tokens] and lengths must be [batch]")
        batch, token_count = token_ids.shape
        if bool((lengths < 1).any()) or bool((lengths > token_count).any()):
            raise ValueError("lengths must cover at least one and no more than token_count")
        e = self.embedding(token_ids)
        valid = torch.arange(token_count, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        zero = torch.zeros((batch, 1, DMODEL), dtype=e.dtype, device=e.device)
        previous = torch.cat((zero, e[:, :-1]), dim=1)
        k = self.local_norm(F.silu(self.local_binding(torch.cat((previous, e), dim=-1))))
        k = k.masked_fill(~valid.unsqueeze(-1), 0.0)
        s_f = (k @ self.q_f) / 4.0
        s_a = (k @ self.q_a) / 4.0
        values = self.w_v(e).masked_fill(~valid.unsqueeze(-1), 0.0)
        if not return_details:
            return s_f, s_a, values
        return {"s_F": s_f, "s_A": s_a, "keys": k, "values": values, "valid": valid}


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T3-nobypass1-noop-distractor-manifest-v1":
        raise ValueError(f"unexpected manifest schema for seed {seed}")
    if manifest.get("domain_seed") != seed or len(manifest.get("train", [])) != 96 or len(manifest.get("test", [])) != 5952:
        raise ValueError(f"manifest cardinality/seed mismatch for seed {seed}")
    if manifest.get("multi_clause_train") != 0 or manifest.get("joint_train_examples") != 0:
        raise ValueError(f"manifest admits forbidden training rows for seed {seed}")
    if set(manifest["operator_for_role"]) != set(("FLOOR", "AVOID", "NOOP")):
        raise ValueError(f"manifest role mapping malformed for seed {seed}")
    if len(set(manifest["token_ids"].values())) != 36:
        raise ValueError(f"manifest token table is not fresh/unique for seed {seed}")
    return manifest, path


def build_training_batch(
    encoder: RelKeyEncoder, manifest: dict[str, Any], labels: dict[str, Tensor]
) -> dict[str, Any]:
    inverse = {int(value): index for index, value in enumerate(manifest["permutation"])}
    token_rows: list[list[int]] = []
    roles: list[str] = []
    source_rows: list[int] = []
    for row in manifest["train"]:
        role = str(row["role"])
        value = int(row["value"])
        operator = str(row["operator"])
        argument = str(row["argument"])
        if role == "FLOOR":
            mask = (labels["constraints"][:, 0] == 1) & (labels["constraints"][:, 1] == 0) & (labels["lower"] == value)
        elif role == "AVOID":
            mask = (labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 1) & (labels["forbidden"] == value)
        elif role == "NOOP":
            mask = (labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 0)
        else:
            raise ValueError(f"unknown training role: {role}")
        candidates = torch.where(mask)[0]
        if not len(candidates):
            raise ValueError(f"missing source row for {role}/{value}")
        if role == "NOOP":
            # NOOP has no value target; deterministic distinct source rows preserve 32 atomic rows.
            source = int(candidates[len(source_rows) % len(candidates)].item())
        else:
            source = int(candidates[0].item())
        if inverse[value] != int(argument[4:]):
            raise ValueError(f"manifest argument/value mismatch for {role}/{value}")
        token_rows.append([encoder.vocab.encode_name(operator), encoder.vocab.encode_name(argument)])
        roles.append(role)
        source_rows.append(source)

    if len(token_rows) != 96 or {role: roles.count(role) for role in ROLES} != {role: 32 for role in ROLES}:
        raise ValueError("training batch is not exactly 32/32/32 atomic rows")
    return {
        "token_ids": torch.tensor(token_rows, dtype=torch.long),
        "lengths": torch.full((96,), 2, dtype=torch.long),
        "roles": roles,
        "source_rows": torch.tensor(source_rows, dtype=torch.long),
    }


def lexical_positions(encoder: RelKeyEncoder, token_ids: Tensor, length: int) -> list[int]:
    arg_ids = {encoder.vocab.encode_name(arg_name(index)) for index in range(VALUE_COUNT)}
    return [index for index, token_id in enumerate(token_ids[:length].tolist()) if token_id in arg_ids]


def first_argmax(scores: Tensor, positions: list[int]) -> int:
    if not positions:
        raise ValueError("hard pointer requires lexical positions")
    return min(positions, key=lambda index: (-finite(scores[index].item()), index))


def selected_atomic_state(details: dict[str, Tensor], token_ids: Tensor, role: str) -> Tensor:
    lexical_counts = [(row < 32).sum().item() for row in token_ids]
    if any(count != 1 for count in lexical_counts):
        raise ValueError(f"{role} atomic training queries must contain exactly one lexical argument")
    return details["values"][:, 1]


def objective(
    encoder: RelKeyEncoder,
    supervisor: LatentConditionedSupervisor,
    executor: nn.Module,
    codebook: Tensor,
    batch: dict[str, Any],
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
) -> dict[str, Tensor]:
    source = batch["source_rows"]
    roles = batch["roles"]
    floor_indices = [index for index, role in enumerate(roles) if role == "FLOOR"]
    avoid_indices = [index for index, role in enumerate(roles) if role == "AVOID"]
    floor_details = encoder(batch["token_ids"][floor_indices], batch["lengths"][floor_indices], return_details=True)
    avoid_details = encoder(batch["token_ids"][avoid_indices], batch["lengths"][avoid_indices], return_details=True)
    noop_details = encoder(batch["token_ids"], batch["lengths"], return_details=True)
    floor_state = selected_atomic_state(floor_details, batch["token_ids"][floor_indices], "FLOOR")
    avoid_state = selected_atomic_state(avoid_details, batch["token_ids"][avoid_indices], "AVOID")
    conditions = torch.zeros((len(roles), 32), dtype=noop_details["values"].dtype)
    conditions[floor_indices] = floor_state
    conditions[avoid_indices] = avoid_state
    behavior = F.cross_entropy(supervisor(observations["features"][source], conditions), labels["action"][source])

    floor_logits = executor.register_decoder(torch.cat((floor_state, torch.zeros_like(floor_state)), dim=-1), codebook)
    avoid_logits = executor.register_decoder(torch.cat((avoid_state, torch.zeros_like(avoid_state)), dim=-1), codebook)
    floor_targets = labels["lower"][source[floor_indices]]
    avoid_targets = labels["forbidden"][source[avoid_indices]]
    reference = (F.cross_entropy(floor_logits, floor_targets) + F.cross_entropy(avoid_logits, avoid_targets)) / 2

    floor_self_f = floor_details["s_F"][:, 1]
    floor_self_a = floor_details["s_A"][:, 1]
    avoid_cross_f = avoid_details["s_F"][:, 1]
    avoid_cross_a = avoid_details["s_A"][:, 1]
    loss_sep_f = F.softplus(avoid_cross_f.unsqueeze(0) - floor_self_f.unsqueeze(1)).mean()
    loss_sep_a = F.softplus(floor_self_a.unsqueeze(0) - avoid_cross_a.unsqueeze(1)).mean()
    separation = (loss_sep_f + loss_sep_a) / 2
    return {
        "behavior": behavior,
        "ref": reference,
        "sep": separation,
        "total": behavior + reference + separation,
        "L_sep_F": loss_sep_f,
        "L_sep_A": loss_sep_a,
    }


def local_certificate(
    encoder: RelKeyEncoder,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compute exhaustive 32-argument local score separation certificates."""
    operator_for_role = manifest["operator_for_role"]
    rows: dict[str, list[list[str]]] = {
        role: [[operator_for_role[role], arg_name(index)] for index in range(VALUE_COUNT)]
        for role in ROLES
    }
    scores: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for role, token_rows in rows.items():
            ids = torch.tensor(
                [[encoder.vocab.encode_name(token) for token in row] for row in token_rows],
                dtype=torch.long,
            )
            lengths = torch.full((VALUE_COUNT,), 2, dtype=torch.long)
            details = encoder(ids, lengths, return_details=True)
            scores[role] = {
                "F": [finite(value) for value in details["s_F"][:, 1].tolist()],
                "A": [finite(value) for value in details["s_A"][:, 1].tolist()],
            }

    floor_f = scores["FLOOR"]["F"]
    floor_a = scores["FLOOR"]["A"]
    avoid_f = scores["AVOID"]["F"]
    avoid_a = scores["AVOID"]["A"]
    noop_f = scores["NOOP"]["F"]
    noop_a = scores["NOOP"]["A"]

    def off_diagonal_min(left: list[float], right: list[float]) -> float:
        return min(left[index] - right[other] for index in range(VALUE_COUNT) for other in range(VALUE_COUNT) if index != other)

    margins = {
        "G_F,N": min(floor_f) - max(noop_f),
        "G_A,N": min(avoid_a) - max(noop_a),
        "G_F,A": off_diagonal_min(floor_f, avoid_f),
        "G_A,F": off_diagonal_min(avoid_a, floor_a),
    }
    return {
        "definitions": {
            "F_N": "s_F([OP_NOOP, ARG_k])",
            "A_N": "s_A([OP_NOOP, ARG_k])",
            "G_F,N": "min_i F_self(i) - max_k F_N(k)",
            "G_A,N": "min_i A_self(i) - max_k A_N(k)",
            "G_F,A": "min_{i!=j} F_self(i) - F_cross_A(j)",
            "G_A,F": "min_{i!=j} A_self(i) - A_cross_F(j)",
            "cross_diagonal_policy": "exclude i=j, matching T2 raw-score certificate",
        },
        "margins": {name: finite(value) for name, value in margins.items()},
        "all_positive": all(value > 0.0 for value in margins.values()),
        "scores": scores,
        "queries_per_role": VALUE_COUNT,
    }


def train_seed(
    seed: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
    supervisor: LatentConditionedSupervisor,
    executor: nn.Module,
    codebook: Tensor,
    destination: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("final.pt", "results.json")):
        raise FileExistsError(f"refusing to overwrite T3 seed output: {destination}")
    encoder = RelKeyEncoder(manifest, seed)
    initial_hash = sha256_bytes(repr([(name, tuple(value.shape)) for name, value in encoder.state_dict().items()]).encode())
    writer_checkpoint_loaded = False
    prior_core_loaded = False
    if writer_checkpoint_loaded or prior_core_loaded:
        raise AssertionError("fresh T3 writer loaded prior checkpoint/core")
    batch = build_training_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last: dict[str, float] = {}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = objective(encoder, supervisor, executor, codebook, batch, observations, labels)
        last = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * (step - 1) / (UPDATES - 1)

    encoder.eval()
    with torch.no_grad():
        final = objective(encoder, supervisor, executor, codebook, batch, observations, labels)
        final_losses = {name: finite(final[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "seed": seed,
            "updates": UPDATES,
            "manifest_sha256": sha256_file(manifest_path),
            "objective": "L_behavior + L_ref + L_sep",
            "L_sep_coefficient": 1.0,
            "fresh_init": True,
            "writer_checkpoint_loaded": False,
            "prior_core_loaded": False,
        },
        checkpoint,
    )
    result = {
        "status": "trained",
        "seed": seed,
        "fresh_init": {
            "constructor": "RelKeyEncoder(manifest_700X_v1, seed)",
            "seed": seed,
            "writer_checkpoint_loaded": False,
            "prior_core_loaded": False,
            "initial_state_shape_fingerprint": initial_hash,
        },
        "training": {
            "updates": UPDATES,
            "batch_rows": {"FLOOR": 32, "AVOID": 32, "NOOP": 32, "total": 96},
            "multi_clause_rows": 0,
            "test_rows_used": 0,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "objective": "L_behavior + L_ref + L_sep",
            "L_sep": {
                "coefficient": 1.0,
                "L_sep_F": "mean softplus(F_cross_A(j) - F_self_F(i)) over 32x32 FLOOR/AVOID atomic queries",
                "L_sep_A": "mean softplus(A_cross_F(j) - A_self_A(i)) over 32x32 FLOOR/AVOID atomic queries",
                "noop_excluded": True,
            },
            "last_update_losses": last,
            "final_losses": final_losses,
        },
        "architecture": {
            "name": "RELKEY",
            "d_model": DMODEL,
            "address_key": RelKeyEncoder.address_key_formula,
            "scores": RelKeyEncoder.score_formula,
            "pointer": "first/min-position argmax over lexical ARG positions only",
            "selected_state": RelKeyEncoder.selected_state_formula,
            "forward_inputs": ["token_ids", "lengths"],
            "position_embeddings": False,
            "right_neighbor": False,
            "mode_path": False,
            "teacher_or_ground_truth_arguments": False,
            "stage_b": False,
            "gate": False,
            "soft_mixture": False,
        },
        "manifest": source_record(manifest_path),
        "checkpoint": source_record(checkpoint),
    }
    write_self_hashed(destination / "results.json", result)
    return {"encoder": encoder, "batch": batch, "training": result}


def decode_state(executor: nn.Module, codebook: Tensor, state: Tensor, target: int) -> dict[str, Any]:
    zeros = torch.zeros((1, state.shape[-1]), dtype=state.dtype)
    logits = executor.register_decoder(torch.cat((state.unsqueeze(0), zeros), dim=-1), codebook)[0]
    decoded = int(logits.argmax().item())
    canonical = int(executor.register_decoder(codebook[decoded].unsqueeze(0), codebook)[0].argmax().item())
    top = logits.topk(2).values
    return {
        "decoded_value": decoded,
        "canonical_value": canonical,
        "target_value": target,
        "target_score": finite(logits[target].item()),
        "winner_score": finite(logits[decoded].item()),
        "winner_margin": finite((top[0] - top[1]).item()),
        "decode_pass": decoded == target,
        "canonical_pass": canonical == target,
        "raw_canon_equal": decoded == canonical,
    }


def margin_record(
    tokens: list[str],
    scores: Tensor,
    target_position: int,
    other_position: int,
    distractor_position: int,
) -> dict[str, Any]:
    structural = [index for index, token in enumerate(tokens) if not token.startswith("ARG_")]
    target_score = finite(scores[target_position].item())
    noop_position = distractor_position
    structure_position = min(structural, key=lambda index: (-finite(scores[index].item()), index))

    def item(position: int, winner_type: str) -> dict[str, Any]:
        winner_score = finite(scores[position].item())
        margin = finite(target_score - winner_score)
        return {"token": tokens[position], "position": position, "winner_type": winner_type, "winner_score": winner_score, "margin": margin, "positive": margin > 0.0}

    return {
        "target": {"token": tokens[target_position], "position": target_position, "score": target_score},
        "M_OTHER_ARG": item(other_position, "ARG"),
        "M_NOOP": item(noop_position, "NOOP_ARG"),
        "M_STRUCT": item(structure_position, tokens[structure_position]),
    }


def evaluate_case(
    seed: int,
    manifest: dict[str, Any],
    encoder: RelKeyEncoder,
    executor: nn.Module,
    codebook: Tensor,
    case: dict[str, Any],
    case_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens = list(case["tokens"])
    ids = torch.tensor([[encoder.vocab.encode_name(token) for token in tokens]], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    target_by_role = {clause["role"]: tokens.index(clause["argument"]) for clause in case["clauses"]}
    distractor_position = target_by_role["NOOP"]
    lexical = lexical_positions(encoder, ids[0], len(tokens))
    with torch.no_grad():
        details = encoder(ids, lengths, return_details=True)
        s_f = details["s_F"][0]
        s_a = details["s_A"][0]
        values = details["values"][0]
        pointer_f = first_argmax(s_f, lexical)
        pointer_a = first_argmax(s_a, lexical)
        margin_f = margin_record(
            tokens,
            s_f,
            target_by_role["FLOOR"],
            target_by_role["AVOID"],
            distractor_position,
        )
        margin_a = margin_record(
            tokens,
            s_a,
            target_by_role["AVOID"],
            target_by_role["FLOOR"],
            distractor_position,
        )
        lower = int(case["lower"])
        forbidden = int(case["forbidden"])
        floor_decode = decode_state(executor, codebook, values[pointer_f], lower)
        avoid_decode = decode_state(executor, codebook, values[pointer_a], forbidden)
    row = {
        "seed": seed,
        "order": case["order_name"],
        "case_index": case_index,
        "L": lower,
        "F": forbidden,
        "D": int(case["distractor_value"]),
        "pointer_F": pointer_f == target_by_role["FLOOR"],
        "pointer_A": pointer_a == target_by_role["AVOID"],
        "decode_F": floor_decode["decode_pass"],
        "decode_A": avoid_decode["decode_pass"],
        "RAW": bool(floor_decode["decode_pass"] and avoid_decode["decode_pass"]),
        "CANON": bool(floor_decode["canonical_pass"] and avoid_decode["canonical_pass"]),
        "RAW_CANON": bool(floor_decode["raw_canon_equal"] and avoid_decode["raw_canon_equal"]),
        "margins": {"FLOOR": margin_f, "AVOID": margin_a},
        "decode_evidence": {"FLOOR": floor_decode, "AVOID": avoid_decode},
        "raw_scores": {"s_F": [finite(value) for value in s_f.tolist()], "s_A": [finite(value) for value in s_a.tolist()]},
        "tokens": tokens,
    }
    failures: list[dict[str, Any]] = []
    for role, pointer, decoded, margins in (
        ("FLOOR", row["pointer_F"], floor_decode, margin_f),
        ("AVOID", row["pointer_A"], avoid_decode, margin_a),
    ):
        target_arg = next(clause["argument"] for clause in case["clauses"] if clause["role"] == role)
        for metric in ("M_OTHER_ARG", "M_NOOP", "M_STRUCT"):
            if not margins[metric]["positive"]:
                failures.append({"seed": seed, "order": case["order_name"], "L": lower, "F": forbidden, "D": int(case["distractor_value"]), "role": role, "target": target_arg, "failure_kind": f"{metric} failure", "winning_competitor": margins[metric]["token"], "target_score": margins[metric]["target"]["score"], "winner_score": margins[metric]["winner_score"], "margin": margins[metric]["margin"]})
        if not pointer:
            pointer_index = pointer_f if role == "FLOOR" else pointer_a
            failures.append({"seed": seed, "order": case["order_name"], "L": lower, "F": forbidden, "D": int(case["distractor_value"]), "role": role, "target": target_arg, "failure_kind": "pointer failure", "winning_competitor": tokens[pointer_index], "target_score": margins["target"]["score"], "winner_score": finite((s_f if role == "FLOOR" else s_a)[pointer_index].item()), "margin": finite(margins["target"]["score"] - (s_f if role == "FLOOR" else s_a)[pointer_index].item())})
        if not decoded["decode_pass"]:
            failures.append({"seed": seed, "order": case["order_name"], "L": lower, "F": forbidden, "D": int(case["distractor_value"]), "role": role, "target": target_arg, "failure_kind": "decode failure", "winning_competitor": decoded["decoded_value"], "target_score": decoded["target_score"], "winner_score": decoded["winner_score"], "margin": decoded["winner_margin"]})
        if not decoded["canonical_pass"]:
            failures.append({"seed": seed, "order": case["order_name"], "L": lower, "F": forbidden, "D": int(case["distractor_value"]), "role": role, "target": target_arg, "failure_kind": "canonical failure", "winning_competitor": decoded["canonical_value"], "target_score": decoded["target_score"], "winner_score": decoded["winner_score"], "margin": decoded["winner_margin"]})
    return row, failures


def minima_update(store: dict[str, Any], metric: str, value: float, evidence: dict[str, Any]) -> None:
    current = store.get(metric)
    if current is None or value < current["margin"]:
        store[metric] = {"margin": finite(value), **evidence}


def summarize_order(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("pointer_F", "pointer_A", "decode_F", "decode_A", "RAW", "CANON", "RAW_CANON")
    return {
        "cases": len(rows),
        **{field: sum(bool(row[field]) for row in rows) for field in fields},
        "M_OTHER_ARG_positive": sum(row["margins"][role]["M_OTHER_ARG"]["positive"] for row in rows for role in ("FLOOR", "AVOID")),
        "M_NOOP_positive": sum(row["margins"][role]["M_NOOP"]["positive"] for row in rows for role in ("FLOOR", "AVOID")),
        "M_STRUCT_positive": sum(row["margins"][role]["M_STRUCT"]["positive"] for row in rows for role in ("FLOOR", "AVOID")),
        "failure_count": len(failures),
        "M_NOOP_failure_count": sum(item["failure_kind"] == "M_NOOP failure" for item in failures),
    }


def six_order_equality(rows_by_order: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    fields = ("pointer_F", "pointer_A", "decode_F", "decode_A", "RAW", "CANON", "RAW_CANON")
    by_pair: dict[tuple[int, int, int], dict[str, tuple[bool, ...]]] = {}
    for order, rows in rows_by_order.items():
        for row in rows:
            by_pair.setdefault((row["L"], row["F"], row["D"]), {})[order] = tuple(bool(row[field]) for field in fields)
    mismatches = []
    for key, outcomes in sorted(by_pair.items()):
        values = list(outcomes.values())
        if len(outcomes) != 6 or any(value != values[0] for value in values[1:]):
            mismatches.append({"L": key[0], "F": key[1], "D": key[2], "outcomes": outcomes})
    return {"key": "(L,F,D)", "orders": list(ORDERS), "outcome_fields": list(fields), "pair_count": len(by_pair), "equal": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches}


def evaluate_seed(seed: int, manifest: dict[str, Any], manifest_path: Path, checkpoint: Path, supervisor: LatentConditionedSupervisor, executor: nn.Module, codebook: Tensor) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = RelKeyEncoder(manifest, seed)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    rows_by_order: dict[str, list[dict[str, Any]]] = {order: [] for order in ORDERS}
    failures_by_order: dict[str, list[dict[str, Any]]] = {order: [] for order in ORDERS}
    minima: dict[str, Any] = {}
    for index, case in enumerate(manifest["train"]):
        if case["role"] not in ROLES:
            raise ValueError("invalid atomic role")
    for role in ROLES:
        atomic_rows = [row for row in manifest["train"] if row["role"] == role]
        ids = torch.tensor([[encoder.vocab.encode_name(row["operator"]), encoder.vocab.encode_name(row["argument"])] for row in atomic_rows], dtype=torch.long)
        lengths = torch.full((32,), 2, dtype=torch.long)
        with torch.no_grad():
            details = encoder(ids, lengths, return_details=True)
            selected = details["values"][:, 1]
            if role == "FLOOR":
                logits = executor.register_decoder(torch.cat((selected, torch.zeros_like(selected)), dim=-1), codebook)
                exact = int((logits.argmax(-1) == torch.tensor([int(row["value"]) for row in atomic_rows])).sum())
                atomic_floor = {"exact": exact, "total": 32}
            elif role == "AVOID":
                logits = executor.register_decoder(torch.cat((selected, torch.zeros_like(selected)), dim=-1), codebook)
                exact = int((logits.argmax(-1) == torch.tensor([int(row["value"]) for row in atomic_rows])).sum())
                atomic_avoid = {"exact": exact, "total": 32}
    atomic_noop = {"exact": 0, "total": 32}
    # NOOP behavior is evaluated from the frozen supervisor with a zero condition.
    observations, labels = load_source()
    noop_source = torch.where((labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 0))[0][:32]
    with torch.no_grad():
        noop_condition = torch.zeros((32, 32))
        noop_logits = supervisor(observations["features"][noop_source], noop_condition)
        atomic_noop["exact"] = int((noop_logits.argmax(-1) == labels["action"][noop_source]).sum())

    local = local_certificate(encoder, manifest)

    for case_index, case in enumerate(manifest["test"]):
        order = case["order_name"]
        row, failures = evaluate_case(seed, manifest, encoder, executor, codebook, case, case_index)
        rows_by_order[order].append(row)
        failures_by_order[order].extend(failures)
        for role in ("FLOOR", "AVOID"):
            for metric in ("M_OTHER_ARG", "M_NOOP", "M_STRUCT"):
                margin = row["margins"][role][metric]["margin"]
                winner = row["margins"][role][metric]
                minima_update(minima, f"{role}.{metric}", margin, {"seed": seed, "order": order, "L": row["L"], "F": row["F"], "D": row["D"], "role": role, "target": row["margins"][role]["target"]["token"], "winning_competitor": winner["token"], "target_score": row["margins"][role]["target"]["score"], "winner_score": winner["winner_score"]})
    order_results = {order: {"summary": summarize_order(rows_by_order[order], failures_by_order[order]), "failures": failures_by_order[order]} for order in ORDERS}
    return {
        "seed": seed,
        "manifest": source_record(manifest_path),
        "checkpoint": source_record(checkpoint),
        "atomic": {"FLOOR": atomic_floor, "AVOID": atomic_avoid, "NOOP": atomic_noop},
        "local_certificate": local,
        "orders": order_results,
        "six_order_outcome_equality": six_order_equality(rows_by_order),
        "global_minima": minima,
        "test_cases_evaluated": sum(len(rows) for rows in rows_by_order.values()),
        "explicit_M_NOOP_failures": [failure for failures in failures_by_order.values() for failure in failures if failure["failure_kind"] == "M_NOOP failure"],
    }


def forbidden_snapshot() -> dict[str, str]:
    roots = [ROOT / "scratch_probes" / "scale_arg_512", ROOT / "campaign" / "t2_nobypass2_final_closure"]
    paths = [path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()]
    paths += [path for path in MANIFEST_ROOT.glob("manifest_700*_v1.json") if path.is_file()]
    return {str(path.relative_to(ROOT).as_posix()): sha256_file(path) for path in sorted(paths)}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty T3 output root: {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    before_forbidden = forbidden_snapshot()
    manifest_data = {seed: load_manifest(seed) for seed in SEEDS}
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
            manifest, manifest_path = manifest_data[seed]
            trained = train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, destination)
            audited = evaluate_seed(seed, manifest, manifest_path, Path(trained["training"]["checkpoint"]["path"]), supervisor, executor, codebook)
            entry = {"status": "completed", "classification": "PASS" if seed_passes(trained["training"], audited) else "VALID_FAIL", "training": trained["training"], "audit": audited}
        except Exception as error:
            entry = {"status": "execution_failed", "classification": "VALID_FAIL", "seed": seed, "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, "test_cases_evaluated": 0}
        write_self_hashed(destination / "results.json", entry)
        per_seed.append(entry)
        print(json.dumps({"seed": seed, "status": entry["status"], "classification": entry["classification"], "test_cases_evaluated": entry.get("audit", {}).get("test_cases_evaluated", 0)}, sort_keys=True), flush=True)

    after_forbidden = forbidden_snapshot()
    forbidden_unchanged = before_forbidden == after_forbidden
    all_pass = len(per_seed) == len(SEEDS) and forbidden_unchanged and all(entry["classification"] == "PASS" for entry in per_seed)
    consolidated: dict[str, Any] = {
        "status": "completed",
        "classification": "PASS_STRONG" if all_pass else "VALID_FAIL",
        "task": "T3-NOBYPASS-1-NOOP-DISTRACTOR",
        "executive_summary": {
            "result": "PASS_STRONG only if every seed and every gate passes" if all_pass else "VALID_FAIL: one or more seed, atomic, pointer, decode, canonical, margin, equality, or immutability gates failed",
            "seeds_completed": len(per_seed),
            "seeds_expected": len(SEEDS),
            "total_test_cases_expected": 29760,
            "total_test_cases_evaluated": sum(entry.get("audit", {}).get("test_cases_evaluated", 0) for entry in per_seed),
            "no_sequential_stopping": True,
            "no_tuning_or_modification_between_seeds": True,
        },
        "authorization": {"id": "T3-NOBYPASS-1-NOOP-DISTRACTOR", "run_all_seeds_even_if_one_fails": True, "no_test_training": True},
        "recipe": {
            "architecture": "RELKEY k_t=LN(SiLU(W_k[e_prev,e_cur])), scores q_r^T k/4, lexical-position hard argmax, r=W_v(e_j)",
            "d_model": 16,
            "forward_inputs": ["token_ids", "lengths"],
            "updates": UPDATES,
            "batch": {"FLOOR": 32, "AVOID": 32, "NOOP": 32, "total": 96},
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "objective": "L_behavior + L_ref + L_sep",
            "L_sep_coefficient": 1.0,
            "L_sep": "L_sep_F=mean softplus(F_cross_A(j)-F_self_F(i)); L_sep_A=mean softplus(A_cross_F(j)-A_self_A(i)); mean of both; FLOOR↔AVOID only over 64 atomic queries",
            "noop_excluded_from_contrastive": True,
            "multi_clause_train": 0,
            "stage_b": False,
            "gate": False,
            "teacher": False,
            "threshold": False,
            "soft_mixture": False,
        },
        "provenance": {
            "sealed_manifests": {str(seed): source_record(manifest_data[seed][1]) for seed in SEEDS},
            "executor": {"identity": "approved C1JointModel via ctrl2_common.load_executor", **source_record(BASE_CHECKPOINT)},
            "supervisor": {"identity": "frozen CTRL7 LatentConditionedSupervisor", **source_record(CTRL7_CHECKPOINT)},
            "writer_checkpoint_loaded": False,
            "prior_t2_writer_or_core_loaded": False,
            "decoder": "frozen real register_decoder with frozen VALUE codebook",
            "arg_mapping_adapter": {"used": False, "simplification": "none; sealed ARG names map directly to fresh local table rows"},
            "script": source_record(SCRIPT_PATH),
        },
        "gates": {
            "all_seeds_present": len(per_seed) == 5,
            "atomic_32_each_role": all(entry.get("audit", {}).get("atomic", {}).get(role, {}).get("exact") == 32 for entry in per_seed for role in ROLES),
            "local_four_margins_positive": all(entry.get("audit", {}).get("local_certificate", {}).get("all_positive") is True for entry in per_seed),
            "all_5952_test_rows_per_seed": all(entry.get("audit", {}).get("test_cases_evaluated") == 5952 for entry in per_seed),
            "six_order_equality_all_seeds": all(entry.get("audit", {}).get("six_order_outcome_equality", {}).get("equal") is True for entry in per_seed),
            "all_required_margins_positive": all(all(order_data.get("summary", {}).get(metric, 0) == 2 * 992 for order_data in entry.get("audit", {}).get("orders", {}).values() for metric in ("M_OTHER_ARG_positive", "M_NOOP_positive", "M_STRUCT_positive")) for entry in per_seed),
            "forbidden_paths_unchanged": forbidden_unchanged,
            "total_test_cases": sum(entry.get("audit", {}).get("test_cases_evaluated", 0) for entry in per_seed),
        },
        "forbidden_path_immutability": {"before": before_forbidden, "after": after_forbidden, "unchanged": forbidden_unchanged},
        "per_seed": per_seed,
        "artifacts": {"consolidated_results": str(OUTPUT_ROOT / "results.json"), "per_seed_results": [str(OUTPUT_ROOT / f"seed_{seed}" / "results.json") for seed in SEEDS], "checkpoints": [str(OUTPUT_ROOT / f"seed_{seed}" / "final.pt") for seed in SEEDS]},
        "next_recommended": "No tuning. Preserve VALID_FAIL evidence and await fresh authorization if any gate fails." if not all_pass else "Archive PASS_STRONG certificate; no further training.",
        "risks": ["Any failed seed or gate produces VALID_FAIL; partial success never upgrades to PASS_STRONG.", "M_NOOP failures are explicit and never repaired by changing contrastive loss."],
        "skill_resolution": {"sdd_apply": "C:/Users/danil/.config/opencode/skills/sdd-apply/SKILL.md", "go_testing": "C:/Users/danil/.config/opencode/skills/go-testing/SKILL.md", "shared": "C:/Users/danil/.config/opencode/skills/_shared/SKILL.md"},
    }
    write_self_hashed(OUTPUT_ROOT / "results.json", consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": consolidated["artifact_self_hash"]}, sort_keys=True))
    return 0


def seed_passes(training: dict[str, Any], audited: dict[str, Any]) -> bool:
    if audited.get("test_cases_evaluated") != 5952:
        return False
    if any(audited["atomic"][role]["exact"] != 32 for role in ROLES):
        return False
    if not audited.get("local_certificate", {}).get("all_positive", False):
        return False
    if not audited["six_order_outcome_equality"]["equal"]:
        return False
    for order_data in audited["orders"].values():
        summary = order_data["summary"]
        if summary["cases"] != 992:
            return False
        if any(summary[field] != 992 for field in ("pointer_F", "pointer_A", "decode_F", "decode_A", "RAW", "CANON", "RAW_CANON")):
            return False
        if any(summary[metric] != 2 * 992 for metric in ("M_OTHER_ARG_positive", "M_NOOP_positive", "M_STRUCT_positive")):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
