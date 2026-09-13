"""Train fresh NB5 Stage A encoders with raw role-separation supervision."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ctrl2_common import BASE_CHECKPOINT, load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from nb5_fresh import NB5CoreEncoder
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor
from train_t2_i2_r2 import load_source


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
MANIFEST_ROOT = CAMPAIGN / "nb5_manifests"
SCRIPT_PATH = Path(__file__).resolve()
SEEDS = (6801, 6802, 6803, 6804, 6805)
UPDATES = 5000
CATALOG_SIZE = 32


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: float) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"non-finite metric: {result}")
    return result


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def inverse_permutation(permutation: list[int]) -> dict[int, int]:
    inverse = {value: index for index, value in enumerate(permutation)}
    if len(inverse) != VALUE_COUNT or set(inverse) != set(range(VALUE_COUNT)):
        raise ValueError("manifest permutation is not a bijection over 0..31")
    return inverse


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "NB5-fresh-lexical-cipher-v1":
        raise ValueError(f"unexpected manifest schema for seed {seed}")
    if int(manifest.get("domain_seed")) != seed:
        raise ValueError(f"manifest domain seed mismatch for seed {seed}")
    if len(manifest.get("train", [])) != 96:
        raise ValueError(f"manifest train row count is not 96 for seed {seed}")
    if len(manifest.get("test", [])) != 1984:
        raise ValueError(f"manifest test row count is not 1984 for seed {seed}")
    natural = [row for row in manifest["test"] if row["order"] == "natural"]
    reverse = [row for row in manifest["test"] if row["order"] == "reverse"]
    if len(natural) != 992 or len(reverse) != 992:
        raise ValueError(f"manifest natural/reverse counts are not 992/992 for seed {seed}")
    inverse_permutation([int(value) for value in manifest["permutation"]])
    kinds = {
        kind: sum((row["constraints"] if row["constraints"] != "none" else "noop") == kind for row in manifest["train"])
        for kind in ("floor", "avoid", "noop")
    }
    if kinds != {"floor": 32, "avoid": 32, "noop": 32}:
        raise ValueError(f"manifest atomic train counts are {kinds}, expected 32/32/32")
    if any(row.get("kind") == "joint" or row.get("constraints") == "joint" for row in manifest["train"]):
        raise ValueError("manifest training rows contain forbidden joint data")
    return manifest, path


def encode_tokens(encoder: NB5CoreEncoder, tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [encoder.vocab.encode_name(token) for token in tokens]
    return torch.tensor([ids], dtype=torch.long), torch.tensor([len(ids)], dtype=torch.long)


def build_behavior_batch(
    encoder: NB5CoreEncoder,
    manifest: dict[str, Any],
    labels: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    inverse = inverse_permutation([int(value) for value in manifest["permutation"]])
    token_rows: list[list[int]] = []
    lengths: list[int] = []
    source_rows: list[int] = []
    floor_mask: list[bool] = []
    avoid_mask: list[bool] = []
    source_by_role: dict[str, list[int]] = {"FLOOR": [], "AVOID": []}

    for row in manifest["train"]:
        kind = row["constraints"] if row["constraints"] != "none" else "noop"
        if kind == "noop":
            tokens = ["NOOP"]
            candidates = torch.where(
                (labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 0)
            )[0]
        elif kind in ("floor", "avoid"):
            value = int(row["value"])
            operator = manifest["operator_floor"] if kind == "floor" else manifest["operator_avoid"]
            tokens = [operator, arg_name(inverse[value])]
            constraints = (1, 0) if kind == "floor" else (0, 1)
            value_field = labels["lower"] if kind == "floor" else labels["forbidden"]
            candidates = torch.where(
                (labels["constraints"][:, 0] == constraints[0])
                & (labels["constraints"][:, 1] == constraints[1])
                & (value_field == value)
            )[0]
        else:
            raise ValueError(f"unexpected training row kind: {kind}")

        if not len(candidates):
            raise ValueError(f"missing source row for manifest training row {row}")
        source = int(candidates[0].item())
        if kind in ("floor", "avoid"):
            value_field = labels["lower"] if kind == "floor" else labels["forbidden"]
            if int(value_field[source].item()) != int(row["value"]):
                raise ValueError("source VALUE target does not match manifest VALUE")
            source_by_role[kind.upper()].append(source)
        token_rows.append([encoder.vocab.encode_name(token) for token in tokens])
        lengths.append(len(tokens))
        source_rows.append(source)
        floor_mask.append(kind == "floor")
        avoid_mask.append(kind == "avoid")

    tokens = torch.tensor([row + [0] * (2 - len(row)) for row in token_rows], dtype=torch.long)
    return (
        tokens,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(source_rows, dtype=torch.long),
        torch.tensor(floor_mask, dtype=torch.bool),
        torch.tensor(avoid_mask, dtype=torch.bool),
        source_by_role,
    )


def build_catalog(encoder: NB5CoreEncoder, manifest: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows = [
        [encoder.vocab.encode_name(manifest["operator_floor"]), encoder.vocab.encode_name(arg_name(index))]
        for index in range(CATALOG_SIZE)
    ] + [
        [encoder.vocab.encode_name(manifest["operator_avoid"]), encoder.vocab.encode_name(arg_name(index))]
        for index in range(CATALOG_SIZE)
    ]
    return torch.tensor(token_rows, dtype=torch.long), torch.full((2 * CATALOG_SIZE,), 2, dtype=torch.long)


def catalog_geometry(
    encoder: NB5CoreEncoder, manifest: dict[str, Any], *, retain_grad: bool = False
) -> dict[str, torch.Tensor]:
    token_ids, lengths = build_catalog(encoder, manifest)
    output = encoder(token_ids, lengths, return_details=True)
    rf, ra, mode, _, _, key, sf, sa, _ = output
    sf_scores = sf[:, 1]
    sa_scores = sa[:, 1]
    if retain_grad:
        sf_scores.retain_grad()
        sa_scores.retain_grad()
    floor_sf = sf_scores[:CATALOG_SIZE]
    avoid_sf = sf_scores[CATALOG_SIZE:]
    avoid_sa = sa_scores[CATALOG_SIZE:]
    floor_sa = sa_scores[:CATALOG_SIZE]
    matrix_f = avoid_sf.unsqueeze(0) - floor_sf.unsqueeze(1)
    matrix_a = floor_sa.unsqueeze(0) - avoid_sa.unsqueeze(1)
    loss_sep_f = F.softplus(matrix_f).mean()
    loss_sep_a = F.softplus(matrix_a).mean()
    return {
        "rf": rf,
        "ra": ra,
        "mode": mode,
        "key": key,
        "sf_scores": sf_scores,
        "sa_scores": sa_scores,
        "F_self": floor_sf,
        "F_cross": avoid_sf,
        "A_self": avoid_sa,
        "A_cross": floor_sa,
        "matrix_F": matrix_f,
        "matrix_A": matrix_a,
        "L_sep_F": loss_sep_f,
        "L_sep_A": loss_sep_a,
        "L_sep": (loss_sep_f + loss_sep_a) / 2,
    }


def reference_losses(
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    rf: torch.Tensor,
    ra: torch.Tensor,
    floor_targets: torch.Tensor,
    avoid_targets: torch.Tensor,
    floor_mask: torch.Tensor,
    avoid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    floor_logits = executor.register_decoder(torch.cat((rf[floor_mask], torch.zeros_like(rf[floor_mask])), -1), codebook)
    avoid_logits = executor.register_decoder(torch.cat((ra[avoid_mask], torch.zeros_like(ra[avoid_mask])), -1), codebook)
    return F.cross_entropy(floor_logits, floor_targets), F.cross_entropy(avoid_logits, avoid_targets)


def objective(
    encoder: NB5CoreEncoder,
    supervisor: LatentConditionedSupervisor,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    manifest: dict[str, Any],
    behavior_tokens: torch.Tensor,
    behavior_lengths: torch.Tensor,
    source_rows: torch.Tensor,
    floor_mask: torch.Tensor,
    avoid_mask: torch.Tensor,
    observations: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    rf, ra, mode = encoder(behavior_tokens, behavior_lengths)
    behavior_loss = F.cross_entropy(supervisor(observations["features"][source_rows], mode), labels["action"][source_rows])
    ref_floor, ref_avoid = reference_losses(
        executor,
        codebook,
        rf,
        ra,
        labels["lower"][source_rows][floor_mask],
        labels["forbidden"][source_rows][avoid_mask],
        floor_mask,
        avoid_mask,
    )
    catalog = catalog_geometry(encoder, manifest)
    ref_loss = (ref_floor + ref_avoid) / 2
    total = behavior_loss + ref_loss + catalog["L_sep"]
    return {"behavior": behavior_loss, "ref": ref_loss, "sep": catalog["L_sep"], "total": total, "catalog": catalog}


def atomic_evidence(
    encoder: NB5CoreEncoder,
    manifest: dict[str, Any],
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    catalog: dict[str, torch.Tensor],
) -> dict[str, Any]:
    permutation = [int(value) for value in manifest["permutation"]]
    floor_logits = executor.register_decoder(
        torch.cat((catalog["rf"][:CATALOG_SIZE], torch.zeros_like(catalog["rf"][:CATALOG_SIZE])), -1), codebook
    )
    avoid_logits = executor.register_decoder(
        torch.cat((catalog["ra"][CATALOG_SIZE:], torch.zeros_like(catalog["ra"][CATALOG_SIZE:])), -1), codebook
    )
    floor_decoded = floor_logits.argmax(-1)
    avoid_decoded = avoid_logits.argmax(-1)
    rows = []
    for index, value in enumerate(permutation):
        rows.append(
            {
                "arg_index": index,
                "arg": arg_name(index),
                "value": value,
                "floor_decoded_value": int(floor_decoded[index].item()),
                "avoid_decoded_value": int(avoid_decoded[index].item()),
                "floor_exact": bool(int(floor_decoded[index].item()) == value),
                "avoid_exact": bool(int(avoid_decoded[index].item()) == value),
            }
        )
    return {
        "rows": rows,
        "floor_exact": sum(row["floor_exact"] for row in rows),
        "avoid_exact": sum(row["avoid_exact"] for row in rows),
        "floor_total": CATALOG_SIZE,
        "avoid_total": CATALOG_SIZE,
    }


def gradient_probe(
    encoder: NB5CoreEncoder,
    supervisor: LatentConditionedSupervisor,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    manifest: dict[str, Any],
    role: str,
    arg_index: int,
    source_row: int,
    observations: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> dict[str, Any]:
    before = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    encoder.zero_grad(set_to_none=True)
    operator = manifest["operator_avoid"] if role == "AVOID" else manifest["operator_floor"]
    token_ids, lengths = encode_tokens(encoder, [operator, arg_name(arg_index)])
    rf, ra, mode, _, _, _, sf, sa, _ = encoder(token_ids, lengths, return_details=True)
    sf.retain_grad()
    sa.retain_grad()
    behavior_loss = F.cross_entropy(supervisor(observations["features"][source_row : source_row + 1], mode), labels["action"][source_row : source_row + 1])
    target = labels["forbidden"][source_row : source_row + 1] if role == "AVOID" else labels["lower"][source_row : source_row + 1]
    state = ra if role == "AVOID" else rf
    ref_logits = executor.register_decoder(torch.cat((state, torch.zeros_like(state)), -1), codebook)
    ref_loss = F.cross_entropy(ref_logits, target)
    catalog = catalog_geometry(encoder, manifest, retain_grad=True)
    total = behavior_loss + ref_loss + catalog["L_sep"]
    total.backward()
    present_q = encoder.q_a if role == "AVOID" else encoder.q_f
    absent_q = encoder.q_f if role == "AVOID" else encoder.q_a
    present_score = sa[0, 1] if role == "AVOID" else sf[0, 1]
    absent_score = catalog["F_cross"][arg_index] if role == "AVOID" else catalog["A_cross"][arg_index]
    present_score_grad = sa.grad[0, 1] if role == "AVOID" else sf.grad[0, 1]
    absent_score_grad = catalog["sf_scores"].grad[CATALOG_SIZE + arg_index] if role == "AVOID" else catalog["sa_scores"].grad[arg_index]
    unchanged = all(torch.equal(before[name], parameter.detach()) for name, parameter in encoder.named_parameters())
    result = {
        "role": role,
        "arg_index": arg_index,
        "arg": arg_name(arg_index),
        "source_row_index": source_row,
        "value": int(target.item()),
        "loss_provenance": "single atomic row plus full 64-query separation catalog; frozen supervisor/executor; backward diagnostic only",
        "L_behavior": finite(behavior_loss.detach().item()),
        "L_ref": finite(ref_loss.detach().item()),
        "L_sep": finite(catalog["L_sep"].detach().item()),
        "L_actual_modified_objective": finite(total.detach().item()),
        "present_score_raw": finite(present_score.detach().item()),
        "absent_cross_score_raw": finite(absent_score.detach().item()),
        "q_gradients": {
            "F": {"norm": finite(encoder.q_f.grad.norm().item()) if encoder.q_f.grad is not None else 0.0, "present": role == "FLOOR"},
            "A": {"norm": finite(encoder.q_a.grad.norm().item()) if encoder.q_a.grad is not None else 0.0, "present": role == "AVOID"},
            "selected_present_norm": finite(present_q.grad.norm().item()) if present_q.grad is not None else 0.0,
            "selected_absent_norm": finite(absent_q.grad.norm().item()) if absent_q.grad is not None else 0.0,
        },
        "score_gradients_at_ARG": {
            "present_self": finite(present_score_grad.item()) if present_score_grad is not None else 0.0,
            "absent_cross": finite(absent_score_grad.item()) if absent_score_grad is not None else 0.0,
        },
        "absent_role_strictly_nonzero": bool(
            absent_q.grad is not None
            and absent_q.grad.norm().item() > 0.0
            and absent_score_grad is not None
            and abs(absent_score_grad.item()) > 0.0
        ),
        "parameters_unchanged": unchanged,
        "optimizer_or_update_called": False,
    }
    encoder.zero_grad(set_to_none=True)
    return result


def raw_score_evidence(manifest: dict[str, Any], catalog: dict[str, torch.Tensor]) -> dict[str, Any]:
    tables = {name: [finite(value) for value in catalog[name].detach().cpu().tolist()] for name in ("F_self", "F_cross", "A_self", "A_cross")}
    pairs_f = [tables["F_self"][i] - tables["F_cross"][j] for i in range(CATALOG_SIZE) for j in range(CATALOG_SIZE) if i != j]
    pairs_a = [tables["A_self"][i] - tables["A_cross"][j] for i in range(CATALOG_SIZE) for j in range(CATALOG_SIZE) if i != j]
    rows = [
        {
            "arg_index": index,
            "arg": arg_name(index),
            "value": int(manifest["permutation"][index]),
            **{name: tables[name][index] for name in tables},
        }
        for index in range(CATALOG_SIZE)
    ]
    return {
        "tables": tables,
        "rows": rows,
        "G_F": finite(min(pairs_f)),
        "G_A": finite(min(pairs_a)),
        "off_diagonal_pairs": CATALOG_SIZE * (CATALOG_SIZE - 1),
    }


def pointer_matrix_evidence(manifest: dict[str, Any], raw: dict[str, Any], order: str) -> dict[str, Any]:
    inverse = inverse_permutation([int(value) for value in manifest["permutation"]])
    rows = []
    for case in manifest["test"]:
        if case["order"] != order:
            continue
        lower = int(case["lower"])
        forbidden = int(case["forbidden"])
        lower_index = inverse[lower]
        forbidden_index = inverse[forbidden]
        delta_f = raw["tables"]["F_self"][lower_index] - raw["tables"]["F_cross"][forbidden_index]
        delta_a = raw["tables"]["A_self"][forbidden_index] - raw["tables"]["A_cross"][lower_index]
        rows.append(
            {
                "lower": lower,
                "forbidden": forbidden,
                "lower_arg_index": lower_index,
                "forbidden_arg_index": forbidden_index,
                "delta_F": finite(delta_f),
                "delta_A": finite(delta_a),
                "floor_comparison": bool(delta_f > 0.0),
                "avoid_comparison": bool(delta_a > 0.0),
            }
        )
    return {
        "order": order,
        "cases": len(rows),
        "floor_comparison_count": sum(row["floor_comparison"] for row in rows),
        "avoid_comparison_count": sum(row["avoid_comparison"] for row in rows),
        "floor_992_of_992": len(rows) == 992 and sum(row["floor_comparison"] for row in rows) == 992,
        "avoid_992_of_992": len(rows) == 992 and sum(row["avoid_comparison"] for row in rows) == 992,
        "rows": rows,
    }


def run_seed(
    seed: int,
    observations: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    supervisor: LatentConditionedSupervisor,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
) -> dict[str, Any]:
    manifest, manifest_path = load_manifest(seed)
    output = CAMPAIGN / f"nb5_rcsep_a_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    encoder = NB5CoreEncoder(manifest, seed)
    encoder.train()
    behavior_tokens, behavior_lengths, source_rows, floor_mask, avoid_mask, source_by_role = build_behavior_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last: dict[str, float] = {}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = objective(
            encoder,
            supervisor,
            executor,
            codebook,
            manifest,
            behavior_tokens,
            behavior_lengths,
            source_rows,
            floor_mask,
            avoid_mask,
            observations,
            labels,
        )
        last = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * (step - 1) / (UPDATES - 1)

    encoder.eval()
    with torch.no_grad():
        final_losses = objective(
            encoder,
            supervisor,
            executor,
            codebook,
            manifest,
            behavior_tokens,
            behavior_lengths,
            source_rows,
            floor_mask,
            avoid_mask,
            observations,
            labels,
        )
        final_loss_values = {name: finite(final_losses[name].item()) for name in ("behavior", "ref", "sep", "total")}
        final_catalog = {name: value for name, value in final_losses["catalog"].items()}

    probes = [
        gradient_probe(encoder, supervisor, executor, codebook, manifest, "AVOID", 0, source_by_role["AVOID"][0], observations, labels),
        gradient_probe(encoder, supervisor, executor, codebook, manifest, "FLOOR", 0, source_by_role["FLOOR"][0], observations, labels),
    ]
    blockers = [probe for probe in probes if not probe["absent_role_strictly_nonzero"]]
    if blockers:
        raise RuntimeError(f"zero absent-role gradient blocker for seed {seed}: {json.dumps(blockers, sort_keys=True)}")

    with torch.no_grad():
        atomic = atomic_evidence(encoder, manifest, executor, codebook, final_catalog)
        noop_source = torch.where(~floor_mask & ~avoid_mask)[0]
        _, _, noop_mode = encoder(behavior_tokens[noop_source], behavior_lengths[noop_source])
        noop_logits = supervisor(observations["features"][source_rows[noop_source]], noop_mode)
        noop_exact = int((noop_logits.argmax(-1) == labels["action"][source_rows[noop_source]]).sum().item())
    raw = raw_score_evidence(manifest, final_catalog)
    natural_pointer = pointer_matrix_evidence(manifest, raw, "natural")
    reverse_pointer = pointer_matrix_evidence(manifest, raw, "reverse")
    reverse_sanity = {
        "matrix_only": True,
        "natural_cases": natural_pointer["cases"],
        "reverse_cases": reverse_pointer["cases"],
        "floor_failure_sets_equal": {
            "value_pairs": {
                (row["lower"], row["forbidden"])
                for row in natural_pointer["rows"]
                if not row["floor_comparison"]
            }
            == {
                (row["lower"], row["forbidden"])
                for row in reverse_pointer["rows"]
                if not row["floor_comparison"]
            }
        },
        "avoid_failure_sets_equal": {
            "value_pairs": {
                (row["lower"], row["forbidden"])
                for row in natural_pointer["rows"]
                if not row["avoid_comparison"]
            }
            == {
                (row["lower"], row["forbidden"])
                for row in reverse_pointer["rows"]
                if not row["avoid_comparison"]
            }
        },
        "no_joint_forward": True,
    }
    reverse_sanity["pass"] = bool(
        reverse_sanity["natural_cases"] == 992
        and reverse_sanity["reverse_cases"] == 992
        and reverse_sanity["floor_failure_sets_equal"]["value_pairs"]
        and reverse_sanity["avoid_failure_sets_equal"]["value_pairs"]
    )

    checkpoint = output / "stage_a.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "seed": seed,
            "updates": UPDATES,
            "manifest_sha256": sha256_file(manifest_path),
            "objective": "L_behavior + L_ref + L_sep",
            "L_sep_coefficient": 1.0,
            "joint_train_examples": 0,
        },
        checkpoint,
    )
    script_paths = {
        "train_nb5_rcsep_a.py": SCRIPT_PATH,
        "nb5_fresh.py": ROOT / "scripts" / "nb5_fresh.py",
        "train_nb5_stage_a_6801.py": ROOT / "scripts" / "train_nb5_stage_a_6801.py",
        "train_t2_i0_baseline_b.py": ROOT / "scripts" / "train_t2_i0_baseline_b.py",
        "train_t2_i2_r2.py": ROOT / "scripts" / "train_t2_i2_r2.py",
        "ctrl2_common.py": ROOT / "scripts" / "ctrl2_common.py",
    }
    source_files = {
        name: source_record(path)
        for name, path in {
            **script_paths,
            "manifest_v2.json": manifest_path,
            "supervisor_checkpoint": CTRL7_CHECKPOINT,
            "executor_checkpoint": BASE_CHECKPOINT,
        }.items()
    }
    gates = {
        "atomic": {
            "floor": {"exact": atomic["floor_exact"], "total": atomic["floor_total"]},
            "avoid": {"exact": atomic["avoid_exact"], "total": atomic["avoid_total"]},
            "noop": {"exact": noop_exact, "total": int(len(noop_source))},
        },
        "gradient_audit": {"probes": probes, "all_absent_role_strictly_nonzero": True},
        "raw_score_separation": {"G_F": raw["G_F"], "G_A": raw["G_A"], "both_positive": raw["G_F"] > 0.0 and raw["G_A"] > 0.0},
        "raw_score_pointer_reconstruction": {
            "natural": natural_pointer,
            "reverse_matrix_sanity": reverse_sanity,
        },
    }
    all_gates = bool(
        atomic["floor_exact"] == 32
        and atomic["avoid_exact"] == 32
        and noop_exact == 32
        and raw["G_F"] > 0.0
        and raw["G_A"] > 0.0
        and natural_pointer["floor_992_of_992"]
        and natural_pointer["avoid_992_of_992"]
        and reverse_sanity["pass"]
    )
    result = {
        "status": "completed" if all_gates else "trained_gates_failed",
        "task": "T2-NOBYPASS-2-RCSEP-A",
        "seed": seed,
        "fresh_init": {"constructor": "NB5CoreEncoder(manifest, seed)", "seed": seed, "manifest_v2_unchanged": True},
        "objective": {
            "total": "L_behavior + L_ref + L_sep",
            "L_sep": "(mean(softplus(F_cross[j] - F_self[i])) + mean(softplus(A_cross[j] - A_self[i]))) / 2",
            "L_sep_coefficient": 1.0,
            "catalog_queries_per_update": 64,
            "catalog_layout": "32 [OP_X, ARG_i] followed by 32 [OP_Y, ARG_i]",
            "catalog_matrices": "full [32,32], all-to-all including diagonal",
            "score_source": "same encoder q_F/q_A; raw score at argument position 1",
            "reference_targets": "external VALUE labels from original 96-row source batch",
            "joint_train_examples": 0,
            "ground_truth_arguments_in_forward": False,
            "teacher_in_inference": False,
            "optimizer": "AdamW",
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "weight_decay": 0.0,
            "gradient_clipping": False,
            "updates": UPDATES,
            "batch_rows": {"floor": 32, "avoid": 32, "noop": 32, "total": 96},
        },
        "last_update_losses": last,
        "final_losses": final_loss_values,
        "gates": gates,
        "checkpoint": source_record(checkpoint),
        "manifest": source_record(manifest_path),
        "script_hashes": {name: record["sha256"] for name, record in source_files.items() if name.endswith(".py")},
        "source_hashes": source_files,
        "artifacts": {"stage_a": str(checkpoint), "results": str(output / "results.json")},
        "risks": [
            "Seeds 6801-6805 are development/diagnostic evidence for this causal objective change; fresh validation requires a new battery.",
            "Raw-score evidence intentionally uses no selector or threshold rule.",
        ],
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"seed": seed, "status": result["status"], "checkpoint_sha256": result["checkpoint"]["sha256"], "final_losses": final_loss_values, "G_F": raw["G_F"], "G_A": raw["G_A"], "natural_pointer": [natural_pointer["floor_comparison_count"], natural_pointer["avoid_comparison_count"]]}, sort_keys=True))
    return result


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    random.seed(0)
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
    for seed in SEEDS:
        run_seed(seed, observations, labels, supervisor, executor, codebook)


if __name__ == "__main__":
    main()
