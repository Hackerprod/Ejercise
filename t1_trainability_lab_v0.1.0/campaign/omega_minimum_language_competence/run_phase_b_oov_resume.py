"""Resume Phase B using the approved full-vocabulary OOV base; never loads OMEGA."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parents[2]
CAMPAIGN_ROOT = LAB_ROOT / "t1_trainability_lab_v0.1.0" / "campaign"
R1_CAMPAIGN = CAMPAIGN_ROOT / "omega_r1_k_curve_a_preflight"
QUALITY_CAMPAIGN = CAMPAIGN_ROOT / "omega_backend_quality_qualification"
INPUTS = QUALITY_CAMPAIGN / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
HOLD_ROOT = HERE / "results" / "phase_b_20260929_md298"
HOLD_MANIFEST_PATH = HOLD_ROOT / "phase_b_manifest.json"
HOLD_STREAM_PATH = HOLD_ROOT / "training_stream_manifest.json"
HOLD_OOV_PATH = HOLD_ROOT / "absolute_calibration_oov_hold.json"
HOLD_UNIGRAM_COUNTS_PATH = HOLD_ROOT / "unigram_train_target_counts.json"
HOLD_VALL_COUNTS_PATH = HOLD_ROOT / "vall_target_counts.json"
HOLD_ABSOLUTE_REPORT_PATH = HOLD_ROOT / "absolute_lm_calibration.json"
HOLD_PHASE_REPORT_PATH = HOLD_ROOT / "PHASE_B_REPORT.md"
HOLD_PHASE_REPORT_SHA_PATH = HOLD_ROOT / "PHASE_B_REPORT.md.sha256"
HOLD_HUMAN_STATUS_PATH = HOLD_ROOT / "human_rater_status.json"
DEFAULT_OUTPUT_DIR = HERE / "results" / "phase_b_20260929_md298_oov_resume4"

STAGE_A_CHUNKS_PATH = R1_CAMPAIGN / "results" / "train_data_coverage_stage0_20260927_md288_retry2" / "multichunk_train_chunks.jsonl"
VALL_CHUNKS_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_validation_chunks.jsonl"
VALL_MANIFEST_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_validation_manifest.json"
STAGE_A_TRAIN_MANIFEST_PATH = R1_CAMPAIGN / "results" / "train_data_coverage_stage0_20260927_md288_retry2" / "multichunk_train_manifest.json"
HISTORICAL_TRAIN_MANIFEST_PATH = INPUTS / "train_manifest_1000_pairs.json"
VALL_PAYLOAD_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_validation_chunks.jsonl"
STAGE_A_REPORT_PATH = R1_CAMPAIGN / "results" / "train_data_coverage_stage0_20260927_md288_retry2" / "stage0_report.json"
STAGE_C0_REPORT_PATH = R1_CAMPAIGN / "results" / "coverage_c_stage0_document_balanced_20260928_md291" / "coverage_c_stage0_report.json"
STAGE_C_ADAPTER_PATH = R1_CAMPAIGN / "results" / "coverage_c_document_balanced_20260928_md291" / "input_manifests" / "coverage_c_document_balanced_execution_manifest.json"
STAGE_C_TRAINING_REPORT_PATH = R1_CAMPAIGN / "results" / "coverage_c_document_balanced_20260928_md291" / "coverage_c_training_report.json"
STAGE_C_EVAL_REPORT_PATH = R1_CAMPAIGN / "results" / "coverage_c_distribution_evaluation_20260928_md291" / "coverage_c_distribution_evaluation_report.json"
B_REPORT_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_distribution_report.json"

BASE_PROTOCOL_COMMIT = "e4d4322dc082b2c7de6e06fb0d3076234afbeffa"
BASE_HOLD_COMMIT = "375149498d0a888f7d0e720e77567be8b6cbcb13"
PROTOCOL_BLOB = "fc750a2ae9fb7d3933c54fb91f09ce5568d035ec"
FIXED_GENERATOR_SEED = 20260929
DATASET_REVISION_EXPECTED = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
FROZEN_TOKENIZER_SHA256 = "d657e96ec92d9fc9eb130258eda9efc68eadc499ebcca184cb79416b4827bf77"
DISTILGPT2_WEIGHTS_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DISTILGPT2_WEIGHTS_FILENAME = "model.safetensors"
DISTILGPT2_WEIGHTS_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
DISTILGPT2_WEIGHTS_SIZE_BYTES = 352824413
VOCAB_SIZE = 50257
NGRAM_ORDER = 5
CHUNK_TOKENS = 513
TARGETS_PER_CHUNK = 512
VALL_CHUNKS = 452
VALL_TARGETS = 231424
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
MINIMUM_LANGUAGE_PROTOCOL = "Conversacion LN.md"

for _path in (
    QUALITY_CAMPAIGN,
    CAMPAIGN_ROOT / "omega_teacher_hidden_cache_probe",
    CAMPAIGN_ROOT / "omega_teacher_logit_cache",
    CAMPAIGN_ROOT.parent / "scripts",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_linguistic_instrument import (  # noqa: E402
    FIXED_GENERATOR_SEED as BANK_SEED,
    L1_CATEGORY_ORDER,
    L0_TEMPLATE_VERSION,
    L1_TEMPLATE_VERSION,
    LEXICON_VERSION,
    build_candidate_bank,
    canonical_hash as bank_canonical_hash,
    validate_l1_twins,
)
from human_pilot_harness import pilot_summary  # noqa: E402
from ngram_calibration import (  # noqa: E402
    BOS_ID,
    CalibrationHold,
    FullVocabularyUnigram,
    ModifiedKneserNey5,
    build_raw_5gram_counts,
    score_unigram_chunk,
    validate_frozen_oov_mask,
)
from phase_b_utils import (  # noqa: E402
    LongestTokenOverlapIndex,
    clopper_pearson_upper_one_sided,
    corpus_extreme_degenerate,
    instrument_invalid_l1,
    reject_omega_checkpoint_access,
    select_nonoverlapping_vall_windows,
    threshold_auto_from_u_fp,
    type7_quantile,
    window_degeneration_metrics,
    WINDOW_TOKENS,
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _seal_json(path: Path, value: dict[str, Any], signature_field: str = "report_self_sha256") -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(signature_field, None)
    value[signature_field] = _canonical_hash(unsigned)
    _write_json(path, value)
    return value


def _verify_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")
    return str(signature)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=LAB_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _install_omega_checkpoint_fail_closed_guard() -> None:
    if torch is None:
        return
    original_load = torch.load
    if getattr(original_load, "_phase_b_omega_guard", False):
        return

    def guarded_load(file: Any, *args: Any, **kwargs: Any):
        candidate = getattr(file, "name", file)
        try:
            reject_omega_checkpoint_access(str(candidate))
        except PermissionError:
            raise
        return original_load(file, *args, **kwargs)

    setattr(guarded_load, "_phase_b_omega_guard", True)
    torch.load = guarded_load  # type: ignore[assignment]


def _missing_runtime_dependencies() -> list[str]:
    return [name for name in ("numpy", "torch", "transformers", "datasets", "huggingface_hub") if importlib.util.find_spec(name) is None]


def _teacher_weight_file_identity() -> dict[str, Any]:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    if DISTILGPT2_WEIGHTS_REVISION != TOKENIZER_REVISION:
        raise RuntimeError("FAIL_PROVENANCE: pinned teacher-weight and tokenizer revisions are not identical")
    path = Path(
        hf_hub_download(
            repo_id="distilbert/distilgpt2",
            filename=DISTILGPT2_WEIGHTS_FILENAME,
            revision=DISTILGPT2_WEIGHTS_REVISION,
            local_files_only=True,
        )
    )
    if path.name != DISTILGPT2_WEIGHTS_FILENAME or DISTILGPT2_WEIGHTS_REVISION not in path.parts:
        raise RuntimeError(f"FAIL_PROVENANCE: teacher weights are not from the pinned revision snapshot: {path}")
    size_bytes = path.stat().st_size
    file_sha256 = _sha256_file(path)
    if size_bytes != DISTILGPT2_WEIGHTS_SIZE_BYTES or file_sha256 != DISTILGPT2_WEIGHTS_SHA256:
        raise RuntimeError(
            "FAIL_PROVENANCE: pinned DistilGPT2 weight identity mismatch; "
            f"observed size={size_bytes}, sha256={file_sha256}"
        )
    return {
        "distilgpt2_weights_filename": DISTILGPT2_WEIGHTS_FILENAME,
        "distilgpt2_weights_revision": DISTILGPT2_WEIGHTS_REVISION,
        "distilgpt2_weights_sha256": file_sha256,
        "distilgpt2_weights_size_bytes": size_bytes,
        "distilgpt2_weights_cache_path": str(path),
    }


def _verify_implementation_provenance() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    implementation_commit = _git("rev-parse", "HEAD")
    parents = _git("show", "-s", "--format=%P", "HEAD").split()
    protocol_blob = _git("rev-parse", "HEAD:Conversacion LN.md")
    unit_status = _git("status", "--porcelain", "--", "t1_trainability_lab_v0.1.0/campaign/omega_minimum_language_competence")
    if (
        branch != "main"
        or len(parents) != 1
        or parents[0] != BASE_HOLD_COMMIT
        or protocol_blob != PROTOCOL_BLOB
        or unit_status
    ):
        raise RuntimeError(
            "FAIL_PROVENANCE: Phase-B resume must execute from one clean local implementation commit parented by the frozen HOLD; "
            f"observed branch={branch}, commit={implementation_commit}, parents={parents}, protocol={protocol_blob}, unit_status={unit_status!r}"
        )
    return {
        "repository": "Hackerprod/Ejercise",
        "branch": branch,
        "base_protocol_commit": BASE_PROTOCOL_COMMIT,
        "base_hold_commit": BASE_HOLD_COMMIT,
        "implementation_commit": implementation_commit,
        "implementation_parent": parents[0],
        "protocol_authority": MINIMUM_LANGUAGE_PROTOCOL,
        "protocol_blob_sha256": protocol_blob,
        "unit_worktree_clean_before_run": True,
        **_teacher_weight_file_identity(),
    }


def _hold_file_identities() -> dict[str, dict[str, str]]:
    expected = {
        "phase_b_manifest.json": ("82f39162af13646e99e78a680d79205d2c9fd872eb83dfabdf390d2f561cf4c3", "phase_b_manifest_sha256", "69a376d08efa7d02513fb1b4e865c02c589a0727a3ce27491aa5e69d81e6791d"),
        "training_stream_manifest.json": ("a51e694314cacd090c991b26e35f0bb426b6945f38d9d64a8a1a407784f6eb30", "manifest_sha256", "6f50fed30fc2de6adc9e8466932dde7f637e35a4af431544033420a62a03ebc0"),
        "absolute_calibration_oov_hold.json": ("d82b9e24c6282e8a181116a1702558c82d45ac6b86d165c874c558c7be7d97bb", "report_self_sha256", "fe53771638570db6200373363d8bd00803ef9fe2173bf8e6f68f7fd24a038170"),
        "unigram_train_target_counts.json": ("7cdc8ff2f2c8e0acf880b91bc69e35ec78a8a74ff663f0f1e8d36498f277a37a", None, None),
        "vall_target_counts.json": ("405f5ac2cc88ae9e2fe5a7c26d8799c5d36c18eca458363015594065f2617934", None, None),
        "absolute_lm_calibration.json": ("a7634c6c6987c2ab069fdcd9b011c9e6b087c5e16f60cd43b5ebb9eec24376a4", "report_self_sha256", "4ef01bf61b34d5a9a12c0cc1b9b7fa03543cec00be59198ea8c9fb5178037450"),
        "PHASE_B_REPORT.md": ("f52eb7fda6b12248de63c9997186e53952f174cd7d9716a123346ba8e198af74", None, None),
        "human_rater_status.json": ("fd60f22bb611f1ce7ddb1265a364d366e99cd4adecb825e11aaa065ab71ac6ca", "report_self_sha256", "fad5f2744869a7e397d2fdd9f9907e9961b35a808306a5d413e0cd6cd8b942f0"),
    }
    identities: dict[str, dict[str, str]] = {}
    for name, (expected_file_hash, self_field, expected_self_hash) in expected.items():
        path = HOLD_ROOT / name
        actual_file_hash = _sha256_file(path)
        if actual_file_hash != expected_file_hash:
            raise RuntimeError(f"FAIL_PROVENANCE: immutable HOLD artifact hash changed: {name}")
        actual_self = ""
        if self_field:
            value = json.loads(path.read_text(encoding="utf-8"))
            actual_self = _verify_self_hash(value, self_field, name)
            if actual_self != expected_self_hash:
                raise RuntimeError(f"FAIL_PROVENANCE: immutable HOLD artifact self-hash changed: {name}")
        identities[name] = {"file_sha256": actual_file_hash, "self_sha256": actual_self}
    sidecar = HOLD_PHASE_REPORT_SHA_PATH.read_text(encoding="ascii").strip()
    if sidecar != expected["PHASE_B_REPORT.md"][0]:
        raise RuntimeError("FAIL_PROVENANCE: HOLD PHASE_B_REPORT.md sidecar changed")
    if (HOLD_ROOT / "PHASE_B_REPORT.md").read_text(encoding="utf-8").splitlines()[-1] != "ABSOLUTE_CALIBRATION_HOLD":
        raise RuntimeError("FAIL_PROVENANCE: frozen HOLD status changed")
    return identities


def _load_hold_data() -> dict[str, Any]:
    identities = _hold_file_identities()
    manifest = json.loads(HOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
    stream = json.loads(HOLD_STREAM_PATH.read_text(encoding="utf-8"))
    oov_report = json.loads(HOLD_OOV_PATH.read_text(encoding="utf-8"))
    train_count_doc = json.loads(HOLD_UNIGRAM_COUNTS_PATH.read_text(encoding="utf-8"))
    vall_count_doc = json.loads(HOLD_VALL_COUNTS_PATH.read_text(encoding="utf-8"))
    absolute_doc = json.loads(HOLD_ABSOLUTE_REPORT_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "ABSOLUTE_CALIBRATION_HOLD" or oov_report.get("status") != "ABSOLUTE_CALIBRATION_OOV_POLICY_REQUIRED":
        raise RuntimeError("FAIL_PROVENANCE: expected frozen absolute-calibration OOV HOLD not found")
    if absolute_doc.get("unigram") is not None or absolute_doc.get("modified_kneser_ney_5gram") is not None or absolute_doc.get("distilgpt2") is not None:
        raise RuntimeError("FAIL_PROVENANCE: HOLD must contain no prior NLL scores")
    train_counts = {int(token): int(count) for token, count in train_count_doc["counts"].items()}
    vall_counts = {int(token): int(count) for token, count in vall_count_doc["counts"].items()}
    if _canonical_hash({str(token): count for token, count in sorted(train_counts.items())}) != train_count_doc["count_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: frozen train unigram count table hash mismatch")
    if _canonical_hash({str(token): count for token, count in sorted(vall_counts.items())}) != vall_count_doc["count_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: frozen VALL target count table hash mismatch")
    if sum(train_counts.values()) != int(stream["target_token_presentations"]) or len(train_counts) != int(stream["vocab_observed_by_unigram_target_stream"]):
        raise RuntimeError("FAIL_PROVENANCE: held unigram counts do not match the frozen training stream")
    oov_rows = oov_report["vall_oov_types"]
    oov_ids = validate_frozen_oov_mask(
        vall_counts,
        train_counts,
        oov_rows,
        expected_types=int(oov_report["vall_oov_type_count"]),
        expected_occurrences=int(oov_report["vall_oov_target_occurrences"]),
    )
    if len(oov_ids) != 560 or sum(vall_counts[token] for token in oov_ids) != 1198 or sum(vall_counts.values()) != 231424:
        raise RuntimeError("FAIL_PROVENANCE: frozen HOLD OOV mask/target counts differ from the verified values")
    return {
        "hold_manifest": manifest,
        "stream_manifest": stream,
        "oov_report": oov_report,
        "train_counts": train_counts,
        "vall_counts": vall_counts,
        "oov_ids": oov_ids,
        "hold_file_identities": identities,
    }


def _verify_resume_sources(hold: dict[str, Any]) -> dict[str, Any]:
    stream = hold["stream_manifest"]
    source = stream["source_hashes"]
    file_checks = (
        ("coverage_a_chunk_payload_path", "coverage_a_chunk_payload_sha256"),
        ("coverage_a_stage0_report_path", None),
        ("coverage_a_train_manifest_path", None),
        ("coverage_c_stage0_report_path", None),
        ("coverage_c_execution_manifest_path", None),
        ("coverage_c_training_report_path", "coverage_c_training_report_file_sha256"),
        ("coverage_c_evaluation_report_path", "coverage_c_evaluation_report_file_sha256"),
        ("historical_train_manifest_path", "historical_train_manifest_file_sha256"),
        ("vall_manifest_path", None),
        ("vall_payload_path", "vall_payload_sha256"),
        ("coverage_b_report_path", None),
    )
    actual_paths = {}
    for path_key, sha_key in file_checks:
        path = Path(source[path_key])
        if not path.is_file():
            raise RuntimeError(f"FAIL_PROVENANCE: frozen source file missing: {path}")
        actual_sha = _sha256_file(path)
        if sha_key and actual_sha != source[sha_key]:
            raise RuntimeError(f"FAIL_PROVENANCE: frozen source file hash mismatch: {path}")
        actual_paths[path_key] = str(path)

    stage_a = json.loads(Path(source["coverage_a_stage0_report_path"]).read_text(encoding="utf-8"))
    stage_a_sig = _verify_self_hash(stage_a, "report_self_sha256", "Coverage-A Stage-0 report")
    stage_a_manifest = json.loads(Path(source["coverage_a_train_manifest_path"]).read_text(encoding="utf-8"))
    stage_a_manifest_sig = _verify_self_hash(stage_a_manifest, "manifest_sha256", "Coverage-A train manifest")
    if stage_a_sig != source["coverage_a_stage0_report_self_sha256"] or stage_a_manifest_sig != source["coverage_a_train_manifest_self_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-A frozen source self-hashes differ from HOLD stream")
    if _sha256_file(Path(source["coverage_a_train_manifest_path"])) != stage_a["manifest"]["sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-A train manifest file SHA differs from Stage-0 report")
    if _sha256_file(Path(source["coverage_a_chunk_payload_path"])) != stage_a_manifest["chunk_payload"]["sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-A chunk payload SHA differs from manifest")

    stage_c0 = json.loads(Path(source["coverage_c_stage0_report_path"]).read_text(encoding="utf-8"))
    stage_c0_sig = _verify_self_hash(stage_c0, "report_self_sha256", "Coverage-C Stage-0 report")
    adapter_path = Path(source["coverage_c_execution_manifest_path"])
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_sig = _verify_self_hash(adapter, "manifest_sha256", "Coverage-C execution manifest")
    if stage_c0_sig != source["coverage_c_stage0_report_self_sha256"] or adapter_sig != source["coverage_c_execution_manifest_self_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-C Stage-0/execution identities differ from HOLD stream")
    if adapter.get("coverage_a_stage0_manifest_sha256") != stage_a_manifest_sig or adapter.get("coverage_c_stage0_self_sha256") != stage_c0_sig:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-C adapter does not bind the frozen A/C Stage-0 manifests")

    stage_c = json.loads(Path(source["coverage_c_training_report_path"]).read_text(encoding="utf-8"))
    stage_c_sig = _verify_self_hash(stage_c, "report_self_sha256", "Coverage-C training report")
    stage_c_eval_path = Path(source["coverage_c_evaluation_report_path"])
    stage_c_eval = json.loads(stage_c_eval_path.read_text(encoding="utf-8"))
    stage_c_eval_sig = _verify_self_hash(stage_c_eval, "report_self_sha256", "Coverage-C evaluation report")
    if stage_c_sig != source["coverage_c_training_report_self_sha256"] or stage_c_eval_sig != source["coverage_c_evaluation_report_self_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-C report self-hashes differ from HOLD stream")
    if (
        stage_c.get("status") != "TRAINING_COMPLETE_AWAITING_C_DISTRIBUTION_EVALUATION"
        or stage_c.get("source_identity", {}).get("execution_manifest_sha256") != adapter_sig
        or stage_c_eval.get("status") != "COVERAGE_C_COMPLETE"
    ):
        raise RuntimeError("FAIL_PROVENANCE: Coverage-C is not closed in the frozen source")

    history = json.loads(Path(source["historical_train_manifest_path"]).read_text(encoding="utf-8"))
    history_sig = _verify_self_hash(history, "manifest_sha256", "historical train manifest")
    if history_sig != source["historical_train_manifest_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: historical train-manifest self-hash mismatch")
    vall_manifest_path = Path(source["vall_manifest_path"])
    vall_manifest = json.loads(vall_manifest_path.read_text(encoding="utf-8"))
    vall_sig = _verify_self_hash(vall_manifest, "manifest_sha256", "VALL validation manifest")
    if vall_sig != source["vall_manifest_self_sha256"] or _sha256_file(vall_manifest_path) != _sha256_file(VALL_MANIFEST_PATH):
        raise RuntimeError("FAIL_PROVENANCE: VALL manifest differs from the frozen Coverage-B source")
    b_report = json.loads(Path(source["coverage_b_report_path"]).read_text(encoding="utf-8"))
    b_sig = _verify_self_hash(b_report, "report_self_sha256", "Coverage-B report")
    if b_sig != source["coverage_b_report_self_sha256"] or b_report.get("status") != "DISTRIBUTION_DIAGNOSTIC_COMPLETE":
        raise RuntimeError("FAIL_PROVENANCE: Coverage-B VALL source report identity/status mismatch")

    events = stream["presentations"]
    pairs = adapter["pairs"]
    if len(events) != 8000 or len(pairs) != 1000:
        raise RuntimeError("FAIL_PROVENANCE: frozen C stream does not contain 8000 visits/1000 pairs")
    for pair_index, pair in enumerate(pairs):
        for slot in range(8):
            event = events[pair_index * 8 + slot]
            if (
                int(event["pair_index"]) != pair_index
                or int(event["batch_slot"]) != slot
                or int(event["chunk_pool_index"]) != int(pair["document_indices"][slot])
                or int(event["document_order_index"]) != int(pair["source_document_order_indices"][slot])
                or int(event["chunk_depth"]) != int(pair["selected_chunk_depths"][slot])
            ):
                raise RuntimeError(f"FAIL_PROVENANCE: HOLD stream/adapter schedule mismatch at pair={pair_index}, slot={slot}")
    return {
        "stage_a_report_self_sha256": stage_a_sig,
        "stage_a_manifest_self_sha256": stage_a_manifest_sig,
        "stage_a_chunk_payload_sha256": stage_a_manifest["chunk_payload"]["sha256"],
        "stage_c0_report_self_sha256": stage_c0_sig,
        "stage_c_adapter_self_sha256": adapter_sig,
        "stage_c_training_report_self_sha256": stage_c_sig,
        "stage_c_evaluation_report_self_sha256": stage_c_eval_sig,
        "historical_train_manifest_self_sha256": history_sig,
        "vall_manifest_self_sha256": vall_sig,
        "vall_payload_sha256": source["vall_payload_sha256"],
        "coverage_b_report_self_sha256": b_sig,
        "presentation_count": len(events),
        "presentation_schedule_matches_adapter": True,
        "paths": actual_paths,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _aggregate_chunk_metrics(rows_by_document: Mapping[int, Sequence[dict[str, Any]]], scorer: Any) -> dict[str, Any]:
    total_targets = 0
    total_loss = 0.0
    seen_loss = 0.0
    oov_loss = 0.0
    seen_targets = 0
    oov_targets = 0
    document_nlls: list[dict[str, Any]] = []
    for document_order in sorted(rows_by_document):
        doc_targets = 0
        doc_loss = 0.0
        for row in rows_by_document[document_order]:
            metric = scorer(row["tokens"])
            if not metric["finite"] or int(metric["total_target_count"]) != TARGETS_PER_CHUNK:
                raise FloatingPointError(f"invalid calibration chunk score at document order {document_order}")
            count = int(metric["total_target_count"])
            doc_targets += count
            doc_loss += float(metric["nll_total_token_weighted"]) * count
            total_targets += count
            total_loss += float(metric["nll_total_token_weighted"]) * count
            seen_targets += int(metric["seen_target_count"])
            oov_targets += int(metric["oov_target_count"])
            seen_loss += float(metric["nll_contribution_seen_to_total"]) * count
            oov_loss += float(metric["nll_contribution_oov_to_total"]) * count
        document_nlls.append({"document_order_index": document_order, "target_tokens": doc_targets, "nll": doc_loss / doc_targets})
    total_nll = total_loss / total_targets
    seen_mean = seen_loss / seen_targets if seen_targets else None
    oov_mean = oov_loss / oov_targets if oov_targets else None
    seen_contribution = seen_loss / total_targets
    oov_contribution = oov_loss / total_targets
    if not math.isclose(total_nll, seen_contribution + oov_contribution, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError("seen/OOV contributions do not reproduce full token-weighted NLL")
    macro = sum(row["nll"] for row in document_nlls) / len(document_nlls)
    return {
        "nll_total_token_weighted": total_nll,
        "ppl_total": math.exp(total_nll),
        "nll_seen_targets": seen_mean,
        "nll_oov_targets": oov_mean,
        "nll_contribution_seen_to_total": seen_contribution,
        "nll_contribution_oov_to_total": oov_contribution,
        "seen_target_count": seen_targets,
        "oov_target_count": oov_targets,
        "total_target_count": total_targets,
        "document_macro_nll": macro,
        "document_macro_ppl": math.exp(macro),
        "document_count": len(document_nlls),
        "per_document": document_nlls,
        "decomposition_sum_matches_total": True,
        "finite": all(math.isfinite(value) for value in (total_nll, math.exp(total_nll), macro, math.exp(macro))),
    }


def _load_vall_documents(source_hashes: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], list[list[int]], list[list[int]]]:
    vall_manifest_path = Path(source_hashes["vall_manifest_path"])
    vall_payload_path = Path(source_hashes["vall_payload_path"])
    vall_manifest = json.loads(vall_manifest_path.read_text(encoding="utf-8"))
    vall_sig = _verify_self_hash(vall_manifest, "manifest_sha256", "VALL validation manifest")
    if vall_sig != source_hashes["vall_manifest_self_sha256"]:
        raise ValueError("HOLD and VALL manifest self-hashes differ")
    if _sha256_file(vall_manifest_path) != _sha256_file(VALL_MANIFEST_PATH):
        raise ValueError("HOLD VALL manifest path differs from the canonical frozen manifest")
    if _sha256_file(vall_payload_path) != vall_manifest["chunk_payload"]["sha256"] or _sha256_file(vall_payload_path) != source_hashes["vall_payload_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: frozen VALL payload SHA mismatch")
    if (
        int(vall_manifest.get("document_count", -1)) != 60
        or int(vall_manifest.get("chunk_policy", {}).get("chunk_count_all", -1)) != 452
        or vall_manifest.get("dataset", {}).get("revision") != DATASET_REVISION_EXPECTED
        or vall_manifest.get("dataset", {}).get("split") != "validation"
        or vall_manifest.get("leakage", {}).get("train_validation_full_text_sha_intersection") != 0
        or vall_manifest.get("leakage", {}).get("train_validation_dedupe_key_intersection") != 0
        or vall_manifest.get("leakage", {}).get("test_split_loaded") is not False
    ):
        raise ValueError("FAIL_PROVENANCE: VALL manifest document/chunk/leakage/test identity mismatch")
    grouped: dict[int, list[dict[str, Any]]] = {index: [] for index in range(60)}
    for row in _read_jsonl(Path(vall_manifest["chunk_payload"]["path"])):
        doc_order = int(row["validation_document_order_index"])
        if doc_order not in grouped or len(row["tokens"]) != 513:
            raise RuntimeError("FAIL_PROVENANCE: malformed VALL chunk row")
        if row["full_text_sha256"] != vall_manifest["documents"][doc_order]["full_text_sha256"]:
            raise RuntimeError(f"FAIL_PROVENANCE: VALL chunk/document identity mismatch at {doc_order}")
        grouped[doc_order].append(row)
    full_documents: list[list[int]] = []
    for doc_order in range(60):
        rows = grouped[doc_order]
        rows.sort(key=lambda row: int(row["chunk_index"]))
        if [int(row["chunk_index"]) for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"FAIL_PROVENANCE: VALL document chunk order gap at {doc_order}")
        tokens = list(rows[0]["tokens"])
        for previous, current in zip(rows, rows[1:]):
            if int(previous["token_end_exclusive"]) - 1 != int(current["token_start"]):
                raise RuntimeError(f"FAIL_PROVENANCE: VALL chunk overlap mismatch at document {doc_order}")
            tokens.extend(int(value) for value in current["tokens"][1:])
        if len(tokens) - 1 != len(rows) * TARGETS_PER_CHUNK:
            raise RuntimeError(f"FAIL_PROVENANCE: VALL target position count mismatch at document {doc_order}")
        full_documents.append(tokens)
    target_sequences = [tokens[1:] for tokens in full_documents]
    if sum(len(row) for row in target_sequences) != 231424:
        raise RuntimeError("FAIL_PROVENANCE: reconstructed VALL target count is not 231424")
    return vall_manifest, grouped, full_documents, target_sequences


def _verify_frozen_tokenizer_equivalence(
    tokenizer: Any,
    stream_manifest: Mapping[str, Any],
    train_documents: Sequence[Mapping[str, Any]],
    train_chunks: Sequence[Mapping[str, Any]],
    vall_manifest: Mapping[str, Any],
    vall_groups: Mapping[int, Sequence[Mapping[str, Any]]],
    vall_dataset: Any,
    reconstruct_documents: Any,
    sha256_text: Any,
) -> dict[str, Any]:
    from run_omega_teacher_logit_cache import tokenizer_hash  # noqa: PLC0415

    frozen_tokenizer = stream_manifest.get("tokenizer", {})
    if (
        frozen_tokenizer.get("id") != "distilbert/distilgpt2"
        or frozen_tokenizer.get("revision") != TOKENIZER_REVISION
        or frozen_tokenizer.get("sha256") != FROZEN_TOKENIZER_SHA256
        or int(frozen_tokenizer.get("vocab_size", -1)) != VOCAB_SIZE
        or len(tokenizer) != VOCAB_SIZE
    ):
        raise RuntimeError("FAIL_PROVENANCE: tokenizer id/revision/vocabulary differs from frozen training stream")

    runtime_tokenizer_sha256 = tokenizer_hash(tokenizer)
    train_chunk_checks = 0
    if len(train_documents) != 602 or len(train_chunks) != 4378:
        raise RuntimeError("FAIL_PROVENANCE: tokenizer equivalence requires all 602 train documents/4378 chunks")
    for row in train_chunks:
        doc_order = int(row["source_document_order_index"])
        depth = int(row["chunk_depth"])
        if not 0 <= doc_order < len(train_documents):
            raise RuntimeError("FAIL_PROVENANCE: train chunk references an out-of-range document order")
        ids = train_documents[doc_order]["tokens"]
        expected = [int(value) for value in ids[depth * TARGETS_PER_CHUNK : depth * TARGETS_PER_CHUNK + CHUNK_TOKENS]]
        if list(row["tokens"]) != expected:
            raise RuntimeError(
                f"FAIL_PROVENANCE: tokenizer output differs from frozen train chunk tokens at document={doc_order}, depth={depth}"
            )
        train_chunk_checks += 1

    vall_sources = {index: source for index, source in enumerate(reconstruct_documents(vall_dataset))}
    vall_chunk_checks = 0
    if len(vall_manifest.get("documents", [])) != 60 or sum(len(rows) for rows in vall_groups.values()) != VALL_CHUNKS:
        raise RuntimeError("FAIL_PROVENANCE: tokenizer equivalence requires all 60 VALL documents/452 chunks")
    for doc_order, metadata in enumerate(vall_manifest["documents"]):
        source_index = int(metadata["source_dataset_document_index"])
        source = vall_sources.get(source_index)
        if source is None:
            raise RuntimeError(f"FAIL_PROVENANCE: VALL source document missing during tokenizer equivalence: {source_index}")
        text = str(source["text"])
        if sha256_text(text) != metadata["full_text_sha256"]:
            raise RuntimeError(f"FAIL_PROVENANCE: VALL full-text identity mismatch during tokenizer equivalence: {doc_order}")
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) != int(metadata["token_count"]):
            raise RuntimeError(f"FAIL_PROVENANCE: VALL token count differs from frozen tokenizer stream: {doc_order}")
        rows = sorted(vall_groups[doc_order], key=lambda item: int(item["chunk_index"]))
        for row in rows:
            chunk_index = int(row["chunk_index"])
            expected = ids[chunk_index * TARGETS_PER_CHUNK : chunk_index * TARGETS_PER_CHUNK + CHUNK_TOKENS]
            if list(row["tokens"]) != expected:
                raise RuntimeError(
                    f"FAIL_PROVENANCE: tokenizer output differs from frozen VALL chunk tokens at document={doc_order}, chunk={chunk_index}"
                )
            vall_chunk_checks += 1

    if train_chunk_checks != 4378 or vall_chunk_checks != 452:
        raise RuntimeError("FAIL_PROVENANCE: incomplete exact tokenizer token-ID equivalence check")
    return {
        "tokenizer_id": str(frozen_tokenizer["id"]),
        "tokenizer_revision": TOKENIZER_REVISION,
        "vocabulary_size": len(tokenizer),
        "frozen_tokenizer_metadata_sha256": str(frozen_tokenizer["sha256"]),
        "runtime_tokenizer_metadata_sha256": runtime_tokenizer_sha256,
        "tokenizer_metadata_sha256_exact_match": runtime_tokenizer_sha256 == frozen_tokenizer["sha256"],
        "exact_frozen_train_chunk_token_ids_verified": train_chunk_checks,
        "exact_frozen_vall_chunk_token_ids_verified": vall_chunk_checks,
        "all_frozen_corpus_token_ids_exact_match": True,
    }


def _reconstruct_full_train_documents(tokenizer: Any, historical_train: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import DownloadConfig, load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: PLC0415
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_REVISION,
        reconstruct_documents,
        sha256_bytes,
        sha256_text,
    )

    if len(tokenizer) != 50257 or DATASET_REVISION != historical_train["dataset"]["revision"]:
        raise RuntimeError("FAIL_PROVENANCE: pinned train tokenizer/dataset revision mismatch")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=DownloadConfig(local_files_only=True))
    expected = historical_train["documents"]
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source_index, source in enumerate(reconstruct_documents(dataset)):
        text = str(source["text"])
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(token_ids) < CHUNK_TOKENS:
            continue
        full_hash = sha256_text(text)
        first_hash = sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in token_ids[:CHUNK_TOKENS]))
        key = (full_hash, first_hash)
        if key in seen:
            continue
        seen.add(key)
        recovered.append(
            {
                "document_index": source_index,
                "row_range": list(source["row_range"]),
                "header": str(source["header"]),
                "token_count": len(token_ids),
                "selected_token_count": CHUNK_TOKENS,
                "full_text_sha256": full_hash,
                "retained_513_token_sha256": first_hash,
                "tokens": token_ids,
            }
        )
    if len(recovered) != len(expected) or len(expected) != 602:
        raise RuntimeError(f"FAIL_PROVENANCE: train document reconstruction count={len(recovered)}, expected=602")
    fields = ("document_index", "row_range", "header", "token_count", "selected_token_count", "full_text_sha256", "retained_513_token_sha256")
    for index, (actual, frozen) in enumerate(zip(recovered, expected)):
        for field in fields:
            if actual[field] != frozen[field]:
                raise RuntimeError(f"FAIL_PROVENANCE: train reconstruction drift at document={index} field={field}")
    return recovered


def _instrument_leakage_records(bank: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in bank["L0_minimal_pairs"]:
        for name, ids in (
            ("prompt", row["prompt_token_ids"]),
            ("prompt_preferred", row["preferred_prompt_continuation_token_ids"]),
            ("prompt_foil", row["foil_prompt_continuation_token_ids"]),
        ):
            records.append({"item_id": row["pair_id"], "sequence_kind": name, "token_ids": ids})
    for row in bank["L1_twin_pairs"]:
        for name, ids in (
            ("prompt_a", row["prompt_a_token_ids"]),
            ("prompt_b", row["prompt_b_token_ids"]),
            ("prompt_a_preferred", row["preferred_a_prompt_continuation_token_ids"]),
            ("prompt_a_foil", row["foil_a_prompt_continuation_token_ids"]),
            ("prompt_b_preferred", row["preferred_b_prompt_continuation_token_ids"]),
            ("prompt_b_foil", row["foil_b_prompt_continuation_token_ids"]),
        ):
            records.append({"item_id": row["twin_id"], "sequence_kind": name, "token_ids": ids})
    return records


def _seal_instrument_bank(
    tokenizer: Any,
    train_documents: list[dict[str, Any]],
    vall_documents: list[list[int]],
    output_dir: Path,
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_index = LongestTokenOverlapIndex([row["tokens"] for row in train_documents])
    vall_index = LongestTokenOverlapIndex(vall_documents)
    rejected_attempts = []
    for attempt in range(256):
        bank = build_candidate_bank(tokenizer, attempt=attempt, seed=FIXED_GENERATOR_SEED)
        invariant_checks = validate_l1_twins(bank)
        if len(bank["L0_minimal_pairs"]) != 64 or len(bank["L1_twin_pairs"]) != 64 or any(row["pass"] is not True for row in invariant_checks):
            raise ValueError("FAIL_PROVENANCE: deterministic instrument bank violates frozen construction invariants")
        leakage_rows = []
        for sequence in _instrument_leakage_records(bank):
            ids = [int(value) for value in sequence["token_ids"]]
            train_longest = train_index.longest_match(ids)
            vall_longest = vall_index.longest_match(ids)
            train_exact = bool(ids) and train_longest == len(ids)
            vall_exact = bool(ids) and vall_longest == len(ids)
            leakage_rows.append(
                {
                    "item_id": sequence["item_id"],
                    "sequence_kind": sequence["sequence_kind"],
                    "sequence_token_count": len(ids),
                    "train_exact_match": train_exact,
                    "vall_exact_match": vall_exact,
                    "train_longest_contiguous_token_match": train_longest,
                    "vall_longest_contiguous_token_match": vall_longest,
                    "overlap_threshold_applied_to_prompt": sequence["sequence_kind"] in {"prompt", "prompt_a", "prompt_b"},
                    "reject": train_exact
                    or vall_exact
                    or (
                        sequence["sequence_kind"] in {"prompt", "prompt_a", "prompt_b"}
                        and (train_longest >= 16 or vall_longest >= 16)
                    ),
                }
            )
        rejected = sum(row["reject"] for row in leakage_rows)
        if rejected:
            rejected_attempts.append({"attempt": attempt, "rejected_sequence_count": rejected})
            continue
        bank["status"] = "SEALED"
        bank["generation_attempt"] = attempt
        bank["generator_seed"] = FIXED_GENERATOR_SEED
        bank["leakage_report_source_hashes"] = source_hashes
        leakage_report = {
            "schema": "omega-minimum-language-competence-instrument-leakage-v1",
            "status": "PASS",
            "threshold_max_contiguous_overlap_tokens": 16,
            "overlap_threshold_applies_to_prompt_sequences_only": True,
            "exact_prompt_and_prompt_continuation_forbidden": True,
            "train_document_count": len(train_documents),
            "vall_document_count": len(vall_documents),
            "rejected_generation_attempts": rejected_attempts,
            "accepted_attempt": attempt,
            "source_hashes": source_hashes,
            "checks": leakage_rows,
            "report_self_sha256": "",
        }
        leakage_report["report_self_sha256"] = _canonical_hash({key: value for key, value in leakage_report.items() if key != "report_self_sha256"})
        bank["leakage_report_self_sha256"] = leakage_report["report_self_sha256"]
        bank["bank_self_sha256"] = _canonical_hash({key: value for key, value in bank.items() if key != "bank_self_sha256"})
        bank_manifest = {
            "schema": "omega-minimum-language-competence-instrument-bank-manifest-v1",
            "status": "SEALED",
            "generator_seed": FIXED_GENERATOR_SEED,
            "generation_attempt": attempt,
            "templates_version": bank["templates_version"],
            "lexicons_version": bank["lexicons_version"],
            "item_counts": {"L0_minimal_pairs": 64, "L1_twin_pairs": 64, "L1_items": 128},
            "bank_self_sha256": bank["bank_self_sha256"],
            "leakage_report_self_sha256": leakage_report["report_self_sha256"],
            "source_hashes": source_hashes,
            "manifest_self_sha256": "",
        }
        bank_manifest["manifest_self_sha256"] = _canonical_hash({key: value for key, value in bank_manifest.items() if key != "manifest_self_sha256"})
        return bank, bank_manifest, leakage_report
    raise CalibrationHold("INSTRUMENT_LEAKAGE_HOLD: deterministic reserve exhausted before a clean sealed bank was found")


def _score_teacher_candidate(model: Any, prompt_ids: Sequence[int], candidate_ids: Sequence[int]) -> float:
    if not prompt_ids or not candidate_ids:
        raise ValueError("teacher candidate scoring requires nonempty prompt and continuation")
    full = [int(value) for value in prompt_ids] + [int(value) for value in candidate_ids]
    prefix_len = len(prompt_ids)
    if full[:prefix_len] != list(prompt_ids) or prefix_len >= len(full):
        raise ValueError("invalid prompt/continuation token boundary")
    input_ids = torch.tensor(full[:-1], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(full[1:], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits.squeeze(0).float()
        start = prefix_len - 1
        stop = start + len(candidate_ids)
        candidate_logits = logits[start:stop]
        candidate_targets = targets[prefix_len - 1 : prefix_len - 1 + len(candidate_ids)]
        losses = F.cross_entropy(candidate_logits, candidate_targets, reduction="none")
    if len(losses) != len(candidate_ids) or not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("non-finite or misaligned DistilGPT2 candidate loss")
    return -float(losses.double().mean().item())


def _candidate_choice(preferred_score: float, foil_score: float) -> tuple[bool, float, bool]:
    if preferred_score > foil_score:
        return True, 1.0, False
    if preferred_score < foil_score:
        return False, 0.0, False
    return False, 0.5, True


def _score_l0_l1(bank: dict[str, Any], mkn: ModifiedKneserNey5, teacher: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    l0_rows = []
    l0_scores: dict[str, list[float]] = {"KN5": [], "DistilGPT2": []}
    for row in bank["L0_minimal_pairs"]:
        item = {"pair_id": row["pair_id"], "category": row["category"]}
        for system in ("KN5", "DistilGPT2"):
            if system == "KN5":
                pref = mkn.score_continuation(row["prompt_token_ids"], row["preferred_token_ids"])
                foil = mkn.score_continuation(row["prompt_token_ids"], row["foil_token_ids"])
            else:
                pref = _score_teacher_candidate(teacher, row["prompt_token_ids"], row["preferred_token_ids"])
                foil = _score_teacher_candidate(teacher, row["prompt_token_ids"], row["foil_token_ids"])
            correct, accuracy, tie = _candidate_choice(pref, foil)
            l0_scores[system].append(accuracy)
            item[system] = {"preferred_mean_logprob": pref, "foil_mean_logprob": foil, "correct": correct, "accuracy_contribution": accuracy, "tie": tie}
        l0_rows.append(item)
    l0_summary = {
        system: {"accuracy": sum(values) / len(values), "items": len(values), "tie_count": sum(row[system]["tie"] for row in l0_rows)}
        for system, values in l0_scores.items()
    }

    l1_rows = []
    pair_scores: dict[str, dict[str, list[float]]] = {
        "KN5": {"full": [], "trunc5": []},
        "DistilGPT2": {"full": [], "trunc5": []},
    }
    item_accuracy: dict[str, dict[str, list[float]]] = {
        "KN5": {"full": [], "trunc5": []},
        "DistilGPT2": {"full": [], "trunc5": []},
    }
    for row in bank["L1_twin_pairs"]:
        result: dict[str, Any] = {"twin_id": row["twin_id"], "category": row["category"]}
        for system in ("KN5", "DistilGPT2"):
            modes = ("full", "trunc5")
            for mode in modes:
                prompt_a = row["prompt_a_token_ids"] if mode == "full" else row["prompt_a_last5_token_ids"]
                prompt_b = row["prompt_b_token_ids"] if mode == "full" else row["prompt_b_last5_token_ids"]
                if system == "KN5":
                    pref_a = mkn.score_continuation(prompt_a, row["preferred_a_token_ids"])
                    foil_a = mkn.score_continuation(prompt_a, row["foil_a_token_ids"])
                    pref_b = mkn.score_continuation(prompt_b, row["preferred_b_token_ids"])
                    foil_b = mkn.score_continuation(prompt_b, row["foil_b_token_ids"])
                else:
                    pref_a = _score_teacher_candidate(teacher, prompt_a, row["preferred_a_token_ids"])
                    foil_a = _score_teacher_candidate(teacher, prompt_a, row["foil_a_token_ids"])
                    pref_b = _score_teacher_candidate(teacher, prompt_b, row["preferred_b_token_ids"])
                    foil_b = _score_teacher_candidate(teacher, prompt_b, row["foil_b_token_ids"])
                correct_a, acc_a, tie_a = _candidate_choice(pref_a, foil_a)
                correct_b, acc_b, tie_b = _candidate_choice(pref_b, foil_b)
                pair_success = bool(correct_a and correct_b)
                pair_scores[system][mode].append(float(pair_success))
                item_accuracy[system][mode].extend((acc_a, acc_b))
                result[f"{system}_{mode}"] = {
                    "preferred_a_mean_logprob": pref_a,
                    "foil_a_mean_logprob": foil_a,
                    "preferred_b_mean_logprob": pref_b,
                    "foil_b_mean_logprob": foil_b,
                    "correct_a": correct_a,
                    "correct_b": correct_b,
                    "pair_success": pair_success,
                    "item_accuracy_contributions": [acc_a, acc_b],
                    "tie_a": tie_a,
                    "tie_b": tie_b,
                }
        l1_rows.append(result)

    l1_summary = {
        "twin_pairs": len(l1_rows),
        "items": 2 * len(l1_rows),
        "per_category_descriptive_only": {
            category: {
                "twin_pairs": sum(row["category"] == category for row in l1_rows),
                "gate_applied": False,
            }
            for category in L1_CATEGORY_ORDER
        },
        "references": {
            system: {
                mode: {
                    "pair_success": sum(pair_scores[system][mode]) / len(pair_scores[system][mode]),
                    "pair_success_count": sum(pair_scores[system][mode]),
                    "pair_count": len(pair_scores[system][mode]),
                    "item_accuracy": sum(item_accuracy[system][mode]) / len(item_accuracy[system][mode]),
                    "item_count": len(item_accuracy[system][mode]),
                }
                for mode in pair_scores[system]
            }
            for system in pair_scores
        },
        "pair_rows": l1_rows,
    }
    teacher_full = l1_summary["references"]["DistilGPT2"]["full"]["pair_success"]
    teacher_trunc5 = l1_summary["references"]["DistilGPT2"]["trunc5"]["pair_success"]
    kn5_full = l1_summary["references"]["KN5"]["full"]["pair_success"]
    invalid, reasons = instrument_invalid_l1(teacher_full, teacher_trunc5, kn5_full)
    l1_summary["validity"] = {
        "status": "INSTRUMENT_INVALID_L1" if invalid else "INSTRUMENT_VALID_L1",
        "invalid": invalid,
        "reasons": reasons,
        "distilgpt2_pair_success_full": teacher_full,
        "distilgpt2_pair_success_trunc5": teacher_trunc5,
        "distilgpt2_full_minus_trunc5": teacher_full - teacher_trunc5,
        "kn5_pair_success_full": kn5_full,
        "frozen_conditions": {
            "distilgpt2_full_lt_0_60": teacher_full < 0.60,
            "distilgpt2_trunc5_ge_0_40": teacher_trunc5 >= 0.40,
            "distilgpt2_full_minus_trunc5_le_0": teacher_full - teacher_trunc5 <= 0.0,
            "kn5_full_ge_0_40": kn5_full >= 0.40,
        },
    }
    return {"L0_LOCAL_LANGUAGE_SANITY": {"items": l0_rows, "references": l0_summary, "gate_status": "SANITY_ONLY_NO_POSITIVE_COMPETENCE_CLAIM"}, "L1_CONTEXTUAL_COMPETENCE": l1_summary}, {"invalid": invalid, "reasons": reasons}


def _aggregate_calibration_chunk_results(grouped: Mapping[int, Sequence[dict[str, Any]]], chunk_scorer: Any) -> dict[str, Any]:
    total_targets = 0
    total_loss = 0.0
    seen_loss = 0.0
    oov_loss = 0.0
    seen_count = 0
    oov_count = 0
    doc_rows = []
    for doc_order in sorted(grouped):
        doc_total_targets = 0
        doc_total_loss = 0.0
        for chunk in grouped[doc_order]:
            metric = chunk_scorer(chunk["tokens"])
            if not metric["finite"] or int(metric["total_target_count"]) != TARGETS_PER_CHUNK:
                raise FloatingPointError(f"invalid baseline chunk score for VALL document {doc_order}")
            count = int(metric["total_target_count"])
            loss_sum = float(metric["nll_total_token_weighted"]) * count
            doc_total_targets += count
            doc_total_loss += loss_sum
            total_targets += count
            total_loss += loss_sum
            seen_count += int(metric["seen_target_count"])
            oov_count += int(metric["oov_target_count"])
            seen_loss += float(metric["nll_contribution_seen_to_total"]) * count
            oov_loss += float(metric["nll_contribution_oov_to_total"]) * count
        doc_rows.append({"document_order_index": doc_order, "targets": doc_total_targets, "nll": doc_total_loss / doc_total_targets})
    nll = total_loss / total_targets
    macro = sum(row["nll"] for row in doc_rows) / len(doc_rows)
    result = {
        "nll_total_token_weighted": nll,
        "ppl_total": math.exp(nll),
        "nll_seen_targets": seen_loss / seen_count if seen_count else None,
        "nll_oov_targets": oov_loss / oov_count if oov_count else None,
        "nll_contribution_seen_to_total": seen_loss / total_targets,
        "nll_contribution_oov_to_total": oov_loss / total_targets,
        "seen_target_count": seen_count,
        "oov_target_count": oov_count,
        "total_target_count": total_targets,
        "document_macro_nll": macro,
        "document_macro_ppl": math.exp(macro),
        "document_count": len(doc_rows),
        "per_document": doc_rows,
        "decomposition_sum_matches_total": math.isclose(nll, seen_loss / total_targets + oov_loss / total_targets, rel_tol=0.0, abs_tol=1e-10),
        "finite": math.isfinite(nll) and math.isfinite(macro) and math.isfinite(math.exp(nll)),
    }
    if not result["decomposition_sum_matches_total"]:
        raise AssertionError("NLL seen/OOV contributions do not sum to total NLL")
    return result


def _score_distilgpt2_vall(model: Any, grouped: Mapping[int, Sequence[dict[str, Any]]], oov_ids: set[int]) -> dict[str, Any]:
    import torch.nn.functional as F  # noqa: PLC0415

    total = 0.0
    seen_sum = 0.0
    oov_sum = 0.0
    seen_count = 0
    oov_count = 0
    doc_rows = []
    with torch.inference_mode():
        for doc_order in sorted(grouped):
            doc_sum = 0.0
            doc_count = 0
            for row in grouped[doc_order]:
                tokens = [int(value) for value in row["tokens"]]
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0)
                targets = torch.tensor(tokens[1:], dtype=torch.long)
                logits = model(input_ids=input_ids, use_cache=False).logits.squeeze(0).float()
                losses = F.cross_entropy(logits, targets, reduction="none").double()
                if losses.numel() != TARGETS_PER_CHUNK or not bool(torch.isfinite(losses).all()):
                    raise FloatingPointError(f"invalid DistilGPT2 chunk score at VALL document {doc_order}")
                per_token = losses.tolist()
                target_ids = tokens[1:]
                for token_id, loss in zip(target_ids, per_token):
                    total += loss
                    doc_sum += loss
                    doc_count += 1
                    if token_id in oov_ids:
                        oov_sum += loss
                        oov_count += 1
                    else:
                        seen_sum += loss
                        seen_count += 1
            doc_rows.append({"document_order_index": doc_order, "targets": doc_count, "nll": doc_sum / doc_count})
    token_count = seen_count + oov_count
    nll = total / token_count
    macro = sum(row["nll"] for row in doc_rows) / len(doc_rows)
    if token_count != VALL_TARGETS or oov_count != 1198 or seen_count != VALL_TARGETS - 1198:
        raise ValueError("DistilGPT2 VALL token/OOV mask accounting mismatch")
    result = {
        "nll_total_token_weighted": nll,
        "ppl_total": math.exp(nll),
        "nll_seen_targets": seen_sum / seen_count,
        "nll_oov_targets": oov_sum / oov_count,
        "nll_contribution_seen_to_total": seen_sum / token_count,
        "nll_contribution_oov_to_total": oov_sum / token_count,
        "seen_target_count": seen_count,
        "oov_target_count": oov_count,
        "total_target_count": token_count,
        "document_macro_nll": macro,
        "document_macro_ppl": math.exp(macro),
        "document_count": len(doc_rows),
        "per_document": doc_rows,
        "decomposition_sum_matches_total": math.isclose(nll, seen_sum / token_count + oov_sum / token_count, rel_tol=0.0, abs_tol=1e-7),
        "finite": math.isfinite(nll) and math.isfinite(macro) and math.isfinite(math.exp(nll)),
    }
    if not result["decomposition_sum_matches_total"]:
        raise AssertionError("DistilGPT2 seen/OOV contributions do not sum to total NLL")
    return result


def _type7_values(values: Sequence[float]) -> dict[str, float]:
    return {"q05": type7_quantile(values, 0.05), "q95": type7_quantile(values, 0.95)}


def _main_and_pilot_spans(
    target_sequences: Sequence[Sequence[int]],
    vall_documents: Sequence[Mapping[str, Any]],
    train_leakage_index: LongestTokenOverlapIndex,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[tuple[int, int]]]]:
    main_rows: list[dict[str, Any]] = []
    pilot_rows: list[dict[str, Any]] = []
    reserved: dict[int, list[tuple[int, int]]] = {}
    for doc_order, target_sequence in enumerate(target_sequences):
        metadata = vall_documents[doc_order]
        header = str(metadata.get("header", ""))
        header_ids = list(tokenizer.encode(header, add_special_tokens=False))
        start0 = max(0, len(header_ids) - 1)
        condition = "main" if doc_order < 48 else "pilot"
        selected = None
        for start in range(start0, max(start0 + 1, len(target_sequence) - 144 + 1), 48):
            prompt = list(target_sequence[start : start + 48])
            continuation = list(target_sequence[start + 48 : start + 144])
            if len(prompt) != 48 or len(continuation) != 96:
                continue
            combined = prompt + continuation
            if train_leakage_index.exact_match(combined):
                continue
            selected = (start, prompt, continuation)
            break
        if selected is None:
            raise CalibrationHold(f"HUMAN_SPAN_SELECTION_HOLD: no train-disjoint 48+96 span in VALL document order {doc_order}")
        start, prompt, continuation = selected
        row = {
            "document_order_index": doc_order,
            "source_document_index": int(metadata["document_index"]),
            "full_text_sha256": metadata["full_text_sha256"],
            "partition": condition,
            "token_start_in_vall_target_sequence": start,
            "prompt_token_count": len(prompt),
            "gold_continuation_token_count": len(continuation),
            "prompt_token_ids": prompt,
            "gold_continuation_token_ids": continuation,
            "prompt_text": tokenizer.decode(prompt, clean_up_tokenization_spaces=False),
            "gold_continuation_text": tokenizer.decode(continuation, clean_up_tokenization_spaces=False),
            "prompt_plus_gold_train_exact_match": False,
        }
        if len(row["prompt_text"]) == 0 or len(row["gold_continuation_text"]) == 0:
            raise ValueError(f"invalid human span rendering in VALL document {doc_order}")
        reserved[doc_order] = [(start, start + 144)]
        (main_rows if condition == "main" else pilot_rows).append(row)
    if len(main_rows) != 48 or len(pilot_rows) != 12 or len({row["document_order_index"] for row in main_rows + pilot_rows}) != 60:
        raise AssertionError("human main/pilot manifest is not the frozen 48+12 distinct-document partition")
    return main_rows, pilot_rows, reserved


def _balanced_seed_assignment(main_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    import itertools  # noqa: PLC0415

    pairs = list(itertools.combinations(SEEDS, 2))
    # Two disjoint pairs occur four times; the other eight occur five times.
    # This gives every seed 19 or 20 appearances and pair frequencies 4/5.
    reduced_pairs = {(SEEDS[0], SEEDS[1]), (SEEDS[2], SEEDS[3])}
    schedule = [pair for pair in pairs for _ in range(4 if pair in reduced_pairs else 5)]
    if len(schedule) != 48:
        raise AssertionError("BIBD seed-pair schedule must contain exactly 48 assignments")
    rng = random.Random(FIXED_GENERATOR_SEED)
    rng.shuffle(schedule)
    assignments = []
    for index, row in enumerate(main_rows):
        first, second = schedule[index]
        assignments.append({"main_prompt_id": f"MAIN-{index + 1:02d}", "seed_ids": [first, second], "document_order_index": row["document_order_index"]})
    counts = Counter(seed for row in assignments for seed in row["seed_ids"])
    pair_counts = Counter(tuple(row["seed_ids"]) for row in assignments)
    if any(count not in (19, 20) for count in counts.values()) or max(pair_counts.values()) - min(pair_counts.values()) > 1:
        raise AssertionError("precomputed 48-prompt OMEGA seed assignment is not balanced")
    return assignments


def _prepare_human_manifests(
    output_dir: Path,
    vall_target_sequences: Sequence[Sequence[int]],
    vall_manifest: dict[str, Any],
    train_overlap: LongestTokenOverlapIndex,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[tuple[int, int]]]]:
    main_rows, pilot_rows, reserved = _main_and_pilot_spans(vall_target_sequences, vall_manifest["documents"], train_overlap, tokenizer)
    assignments = _balanced_seed_assignment(main_rows)
    prompt_manifest = {
        "schema": "omega-human-prompt-manifest-v1",
        "status": "SEALED_NOT_RUN",
        "main_prompt_count": 48,
        "pilot_prompt_count": 12,
        "prompt_tokens": 48,
        "continuation_tokens": 96,
        "main_prompts": main_rows,
        "pilot_prompts": pilot_rows,
        "source_vall_manifest_self_sha256": vall_manifest["manifest_sha256"],
        "report_self_sha256": "",
    }
    prompt_manifest["report_self_sha256"] = _canonical_hash({key: value for key, value in prompt_manifest.items() if key != "report_self_sha256"})
    _write_json(output_dir / "human_prompt_manifest.json", prompt_manifest)
    main_manifest = {
        "schema": "omega-human-main-manifest-v1",
        "status": "PREPARED_NOT_RUN_RATER_PANEL_PENDING",
        "prompt_count": len(main_rows),
        "prompts": main_rows,
        "balanced_incomplete_seed_assignment": assignments,
        "rater_panel_ready": False,
        "report_self_sha256": "",
    }
    main_manifest["report_self_sha256"] = _canonical_hash({key: value for key, value in main_manifest.items() if key != "report_self_sha256"})
    _write_json(output_dir / "main_human_manifest.json", main_manifest)
    pilot_manifest = {
        "schema": "omega-human-pilot-manifest-v1",
        "status": "NOT_RUN_RATER_PANEL_PENDING",
        "prompt_count": len(pilot_rows),
        "prompts": pilot_rows,
        "planned_output_conditions": ["REAL_HUMAN", "OMEGA_UPDATE0"],
        "planned_raters": 3,
        "report_self_sha256": "",
    }
    pilot_manifest["report_self_sha256"] = _canonical_hash({key: value for key, value in pilot_manifest.items() if key != "report_self_sha256"})
    _write_json(output_dir / "human_pilot_manifest.json", pilot_manifest)
    rater_status = {
        "schema": "omega-minimum-language-competence-human-rater-status-v1",
        "H1": "user",
        "H2": "TBD",
        "H3": "TBD",
        "RATER_PANEL_READY": False,
        "human_pilot_status": "NOT_RUN_RATER_PANEL_PENDING",
        "main_human_evaluation_status": "NOT_RUN",
        "report_self_sha256": "",
    }
    rater_status["report_self_sha256"] = _canonical_hash({key: value for key, value in rater_status.items() if key != "report_self_sha256"})
    _write_json(output_dir / "human_rater_status.json", rater_status)
    protocol = (
        "# HUMAN_RATER_PROTOCOL — Phase B preparation\n\n"
        "Panel: three independent human raters; H1=user, H2/H3=TBD; RATER_PANEL_READY=false. "
        "Use opaque IDs; do not substitute AI judges. Rate G/R/C/E/N on 1–5 independently. "
        "Pilot plan: 12 REAL-HUMAN and 12 OMEGA update0 outputs. Main plan: 96 OMEGA trained, "
        "48 MiniMind, 48 OMEGA update0, and 48 REAL-HUMAN outputs, anonymized and randomized. "
        "No pilot or main human evaluation has been run; wait until the panel is ready.\n"
    )
    protocol_path = output_dir / "HUMAN_RATER_PROTOCOL.md"
    protocol_path.write_text(protocol, encoding="utf-8", newline="\n")
    (output_dir / "HUMAN_RATER_PROTOCOL.md.sha256").write_text(_sha256_file(protocol_path) + "\n", encoding="ascii", newline="\n")
    return main_rows, pilot_rows, reserved


def _finish_phase_b(
    output_dir: Path,
    provenance: dict[str, Any],
    phase_manifest: dict[str, Any],
    status: str,
    result_fields: dict[str, Any],
    *,
    extra_files: Sequence[str] = (),
) -> dict[str, Any]:
    artifact_names = sorted(set(extra_files) | {"phase_b_manifest.json", "OOV_POLICY_DECISION.json", "phase_b_execution_identity.json"})
    artifact_hashes = {}
    for name in artifact_names:
        path = output_dir / name
        if path.is_file():
            artifact_hashes[name] = {"file_sha256": _sha256_file(path)}
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                signature_field = next((key for key in ("report_self_sha256", "manifest_self_sha256", "bank_self_sha256") if key in payload), None)
                if signature_field:
                    artifact_hashes[name]["self_sha256"] = payload[signature_field]
    final_manifest = {
        **phase_manifest,
        "status": status,
        "implementation_commit": provenance["implementation_commit"],
        "base_protocol_commit": BASE_PROTOCOL_COMMIT,
        "base_hold_commit": BASE_HOLD_COMMIT,
        "protocol_blob_sha256": PROTOCOL_BLOB,
        "artifacts": artifact_hashes,
        "phase_b_manifest_sha256": "",
    }
    final_manifest["phase_b_manifest_sha256"] = _canonical_hash({key: value for key, value in final_manifest.items() if key != "phase_b_manifest_sha256"})
    _write_json(output_dir / "phase_b_manifest.json", final_manifest)
    report_data = {
        "status": status,
        "implementation_commit": provenance["implementation_commit"],
        "base_protocol_commit": BASE_PROTOCOL_COMMIT,
        "base_hold_commit": BASE_HOLD_COMMIT,
        "protocol_blob_sha256": PROTOCOL_BLOB,
        "phase_b_manifest_self_sha256": final_manifest["phase_b_manifest_sha256"],
        **result_fields,
        "artifact_hashes": artifact_hashes,
    }
    lines = ["# PHASE B — ABSOLUTE-LM-CALIBRATION + MINIMUM-LANGUAGE-COMPETENCE", "", "## Provenance"]
    for key in ("status", "implementation_commit", "base_protocol_commit", "base_hold_commit", "protocol_blob_sha256", "phase_b_manifest_self_sha256"):
        lines.append(f"- {key}: `{report_data[key]}`")
    lines.extend(["", "## Raw results"])
    for key, value in result_fields.items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True, separators=(',', ':'))}`")
    lines.extend(["", "## Artifact hashes"])
    for name, record in sorted(artifact_hashes.items()):
        lines.append(f"- {name}: file_sha256=`{record['file_sha256']}`" + (f" self_sha256=`{record['self_sha256']}`" if record.get("self_sha256") else ""))
    lines.extend(["", status, ""])
    report_path = output_dir / "PHASE_B_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    report_sha = _sha256_file(report_path)
    (output_dir / "PHASE_B_REPORT.md.sha256").write_text(report_sha + "\n", encoding="ascii", newline="\n")
    final_manifest["phase_b_report_sha256"] = report_sha
    # Keep the report hash in a separate small terminal index to avoid self-reference.
    index = {
        "schema": "omega-phase-b-output-index-v1",
        "phase_b_manifest_self_sha256": final_manifest["phase_b_manifest_sha256"],
        "phase_b_report_sha256": report_sha,
        "status": status,
        "index_self_sha256": "",
    }
    index["index_self_sha256"] = _canonical_hash({key: value for key, value in index.items() if key != "index_self_sha256"})
    _write_json(output_dir / "phase_b_output_index.json", index)
    return {"status": status, "phase_b_manifest_self_sha256": final_manifest["phase_b_manifest_sha256"], "phase_b_report_sha256": report_sha, **result_fields}


def _load_teacher(tokenizer: Any) -> Any:
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(
        "distilbert/distilgpt2",
        revision=DISTILGPT2_WEIGHTS_REVISION,
        local_files_only=True,
        dtype=torch.float32,
    )
    if int(model.config.vocab_size) != 50257 or len(tokenizer) != 50257:
        raise RuntimeError("DistilGPT2/tokenizer vocabulary is not the frozen GPT-2 BPE V=50257")
    model.to("cpu", dtype=torch.float32)
    model.eval()
    return model


def _score_l0_l1_references(bank: dict[str, Any], mkn: ModifiedKneserNey5, teacher: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    l0_items = []
    l0_accuracy = {"KN5": [], "DistilGPT2": []}
    for row in bank["L0_minimal_pairs"]:
        scored = {"pair_id": row["pair_id"], "category": row["category"]}
        for model_name in ("KN5", "DistilGPT2"):
            if model_name == "KN5":
                preferred = mkn.score_continuation(row["prompt_token_ids"], row["preferred_token_ids"])
                foil = mkn.score_continuation(row["prompt_token_ids"], row["foil_token_ids"])
            else:
                preferred = _score_teacher_candidate(teacher, row["prompt_token_ids"], row["preferred_token_ids"])
                foil = _score_teacher_candidate(teacher, row["prompt_token_ids"], row["foil_token_ids"])
            if preferred > foil:
                contribution, correct, tie = 1.0, True, False
            elif preferred < foil:
                contribution, correct, tie = 0.0, False, False
            else:
                contribution, correct, tie = 0.5, False, True
            l0_accuracy[model_name].append(contribution)
            scored[model_name] = {"preferred_mean_logprob": preferred, "foil_mean_logprob": foil, "correct": correct, "tie": tie, "accuracy_contribution": contribution}
        l0_items.append(scored)
    l0 = {
        "status": "PASS_SANITY" if all(math.isfinite(value) for values in l0_accuracy.values() for value in values) else "FAIL_SANITY",
        "gate_role": "local sanity only; passing does not establish contextual competence",
        "items": l0_items,
        "reference_accuracy": {name: sum(values) / len(values) for name, values in l0_accuracy.items()},
        "tie_count": {name: sum(row[name]["tie"] for row in l0_items) for name in l0_accuracy},
        "pair_count": len(l0_items),
    }

    twin_rows = []
    pair_success = {"KN5": {"full": [], "trunc5": []}, "DistilGPT2": {"full": [], "trunc5": []}}
    item_accuracy = {name: {mode: [] for mode in ("full", "trunc5")} for name in ("KN5", "DistilGPT2")}

    def choice(preferred: float, foil: float) -> tuple[bool, float, bool]:
        if preferred > foil:
            return True, 1.0, False
        if preferred < foil:
            return False, 0.0, False
        return False, 0.5, True

    for row in bank["L1_twin_pairs"]:
        scored_twin: dict[str, Any] = {"twin_id": row["twin_id"], "category": row["category"]}
        for model_name in ("KN5", "DistilGPT2"):
            modes = ("full", "trunc5")
            for mode in modes:
                prompt_a = row["prompt_a_token_ids"] if mode == "full" else row["prompt_a_last5_token_ids"]
                prompt_b = row["prompt_b_token_ids"] if mode == "full" else row["prompt_b_last5_token_ids"]
                if model_name == "KN5":
                    pa = mkn.score_continuation(prompt_a, row["preferred_a_token_ids"])
                    fa = mkn.score_continuation(prompt_a, row["foil_a_token_ids"])
                    pb = mkn.score_continuation(prompt_b, row["preferred_b_token_ids"])
                    fb = mkn.score_continuation(prompt_b, row["foil_b_token_ids"])
                else:
                    pa = _score_teacher_candidate(teacher, prompt_a, row["preferred_a_token_ids"])
                    fa = _score_teacher_candidate(teacher, prompt_a, row["foil_a_token_ids"])
                    pb = _score_teacher_candidate(teacher, prompt_b, row["preferred_b_token_ids"])
                    fb = _score_teacher_candidate(teacher, prompt_b, row["foil_b_token_ids"])
                ca, acc_a, tie_a = choice(pa, fa)
                cb, acc_b, tie_b = choice(pb, fb)
                success = bool(ca and cb)
                pair_success[model_name][mode].append(float(success))
                item_accuracy[model_name][mode].extend((acc_a, acc_b))
                scored_twin[f"{model_name}_{mode}"] = {
                    "preferred_a_mean_logprob": pa,
                    "foil_a_mean_logprob": fa,
                    "preferred_b_mean_logprob": pb,
                    "foil_b_mean_logprob": fb,
                    "correct_a": ca,
                    "correct_b": cb,
                    "pair_success": success,
                    "item_accuracy_contributions": [acc_a, acc_b],
                    "tie_a": tie_a,
                    "tie_b": tie_b,
                }
        twin_rows.append(scored_twin)
    l1_refs = {
        system: {
            mode: {
                "pair_success": sum(pair_success[system][mode]) / len(pair_success[system][mode]),
                "pair_success_count": sum(pair_success[system][mode]),
                "pair_count": len(pair_success[system][mode]),
                "item_accuracy": sum(item_accuracy[system][mode]) / len(item_accuracy[system][mode]),
                "item_count": len(item_accuracy[system][mode]),
            }
            for mode in pair_success[system]
        }
        for system in pair_success
    }
    teacher_full = l1_refs["DistilGPT2"]["full"]["pair_success"]
    teacher_trunc5 = l1_refs["DistilGPT2"]["trunc5"]["pair_success"]
    kn5_full = l1_refs["KN5"]["full"]["pair_success"]
    invalid, reasons = instrument_invalid_l1(teacher_full, teacher_trunc5, kn5_full)
    l1 = {
        "status": "INSTRUMENT_INVALID_L1" if invalid else "INSTRUMENT_VALID_L1",
        "twin_pair_count": len(twin_rows),
        "item_count": 2 * len(twin_rows),
        "categories_descriptive_only": {category: 16 for category in L1_CATEGORY_ORDER},
        "category_gates_applied": False,
        "reference_scores": l1_refs,
        "validity": {
            "invalid": invalid,
            "reasons": reasons,
            "distilgpt2_pair_success_full": teacher_full,
            "distilgpt2_pair_success_trunc5": teacher_trunc5,
            "distilgpt2_full_minus_trunc5": teacher_full - teacher_trunc5,
            "kn5_pair_success_full": kn5_full,
        },
        "twin_rows": twin_rows,
    }
    return {"L0_LOCAL_LANGUAGE_SANITY": l0, "L1_CONTEXTUAL_COMPETENCE": l1}, {"invalid": invalid, "reasons": reasons}


def _write_signed_json(output_dir: Path, name: str, data: dict[str, Any], field: str = "report_self_sha256") -> dict[str, Any]:
    data[field] = ""
    data[field] = _canonical_hash({key: value for key, value in data.items() if key != field})
    _write_json(output_dir / name, data)
    return data


def _report_md(output_dir: Path, status: str, fields: dict[str, Any], hashes: dict[str, Any]) -> str:
    lines = ["# PHASE B — OMEGA-ABSOLUTE-LM-CALIBRATION + MINIMUM-LANGUAGE-COMPETENCE instrument preflight", "", "## Provenance"]
    for key in (
        "base_protocol_commit", "base_hold_commit", "implementation_commit", "protocol_blob_sha256",
        "distilgpt2_weights_revision", "distilgpt2_weights_sha256", "distilgpt2_weights_size_bytes",
    ):
        if key in fields["provenance"]:
            lines.append(f"- {key}: `{fields['provenance'][key]}`")
    lines.extend(["", "## Results"])
    for key, value in fields["raw_results"].items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True, separators=(',', ':'))}`")
    lines.extend(["", "## File hashes"])
    for name, record in sorted(hashes.items()):
        lines.append(f"- {name}: file=`{record['file_sha256']}`" + (f" self=`{record['self_sha256']}`" if record.get("self_sha256") else ""))
    lines.extend(["", status, ""])
    path = output_dir / "PHASE_B_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    digest = _sha256_file(path)
    (output_dir / "PHASE_B_REPORT.md.sha256").write_text(digest + "\n", encoding="ascii", newline="\n")
    return digest


def _finish(
    output_dir: Path,
    status: str,
    provenance: dict[str, Any],
    result_fields: dict[str, Any],
    *,
    output_names: Sequence[str],
    source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hashes = {}
    for name in sorted(set(output_names)):
        path = output_dir / name
        if not path.is_file():
            continue
        record = {"file_sha256": _sha256_file(path)}
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            field = next(
                (
                    key
                    for key in (
                        "report_self_sha256",
                        "manifest_self_sha256",
                        "bank_self_sha256",
                        "decision_self_sha256",
                        "phase_b_manifest_sha256",
                        "index_self_sha256",
                    )
                    if key in payload
                ),
                None,
            )
            if field:
                record["self_sha256"] = payload[field]
        hashes[name] = record
    phase_manifest = {
        "schema": "omega-minimum-language-competence-phase-b-resume-manifest-v1",
        "unit": "OMEGA-ABSOLUTE-LM-CALIBRATION+MINIMUM-LANGUAGE-COMPETENCE-PHASE-B",
        "status": status,
        "provenance": provenance,
        "source_identity": source_identity or {},
        "oov_policy_decision_self_sha256": json.loads((output_dir / "OOV_POLICY_DECISION.json").read_text(encoding="utf-8"))["decision_self_sha256"] if (output_dir / "OOV_POLICY_DECISION.json").is_file() else None,
        "raw_results": result_fields,
        "artifact_hashes": hashes,
        "phase_b_manifest_sha256": "",
    }
    phase_manifest["phase_b_manifest_sha256"] = _canonical_hash({key: value for key, value in phase_manifest.items() if key != "phase_b_manifest_sha256"})
    _write_json(output_dir / "phase_b_manifest.json", phase_manifest)
    report = {
        "status": status,
        "provenance": provenance,
        "source_identity": source_identity or {},
        "phase_b_manifest_self_sha256": phase_manifest["phase_b_manifest_sha256"],
        "raw_results": result_fields,
        "artifact_hashes": hashes,
    }
    report["report_self_sha256"] = _canonical_hash(report)
    _write_json(output_dir / "phase_b_completion_report.json", report)
    report_md_sha = _report_md(output_dir, status, {"provenance": provenance, "raw_results": result_fields}, hashes)
    output_index = {
        "schema": "omega-minimum-language-competence-phase-b-output-index-v1",
        "status": status,
        "phase_b_manifest_self_sha256": phase_manifest["phase_b_manifest_sha256"],
        "phase_b_completion_report_self_sha256": report["report_self_sha256"],
        "phase_b_report_md_sha256": report_md_sha,
        "artifact_hashes": hashes,
        "index_self_sha256": "",
    }
    output_index["index_self_sha256"] = _canonical_hash({key: value for key, value in output_index.items() if key != "index_self_sha256"})
    _write_json(output_dir / "phase_b_output_index.json", output_index)
    return {
        "status": status,
        "phase_b_manifest_self_sha256": phase_manifest["phase_b_manifest_sha256"],
        "phase_b_report_sha256": report_md_sha,
        "raw_results": result_fields,
        "artifact_hashes": hashes,
    }


def _write_hold_completion(
    output_dir: Path,
    provenance: dict[str, Any],
    hold: dict[str, Any],
    result_fields: dict[str, Any],
    status: str,
    *,
    source_identity: dict[str, Any] | None = None,
    extra_files: Sequence[str] = (),
) -> dict[str, Any]:
    if not (output_dir / "human_rater_status.json").is_file():
        rater_status = {
            "schema": "omega-minimum-language-competence-human-rater-status-v1",
            "H1": "user",
            "H2": "TBD",
            "H3": "TBD",
            "RATER_PANEL_READY": False,
            "human_pilot_status": "NOT_RUN_RATER_PANEL_PENDING",
            "main_human_evaluation_status": "NOT_RUN",
            "phase_b_terminal_status": status,
            "report_self_sha256": "",
        }
        rater_status["report_self_sha256"] = _canonical_hash({key: value for key, value in rater_status.items() if key != "report_self_sha256"})
        _write_json(output_dir / "human_rater_status.json", rater_status)
    if not (output_dir / "human_prompt_manifest.json").is_file():
        prompt_manifest = {
            "schema": "omega-human-prompt-manifest-v1",
            "status": "NOT_PREPARED_PHASE_B_TERMINAL_HOLD",
            "main_prompt_count": 48,
            "pilot_prompt_count": 12,
            "prompts": [],
            "report_self_sha256": "",
        }
        prompt_manifest["report_self_sha256"] = _canonical_hash({key: value for key, value in prompt_manifest.items() if key != "report_self_sha256"})
        _write_json(output_dir / "human_prompt_manifest.json", prompt_manifest)
    if not (output_dir / "main_human_manifest.json").is_file():
        main_manifest = {
            "schema": "omega-human-main-manifest-v1",
            "status": "NOT_PREPARED_PHASE_B_TERMINAL_HOLD",
            "prompt_count": 48,
            "prompts": [],
            "rater_panel_ready": False,
            "report_self_sha256": "",
        }
        main_manifest["report_self_sha256"] = _canonical_hash({key: value for key, value in main_manifest.items() if key != "report_self_sha256"})
        _write_json(output_dir / "main_human_manifest.json", main_manifest)
    if not (output_dir / "human_pilot_manifest.json").is_file():
        pilot_manifest = {
            "schema": "omega-human-pilot-manifest-v1",
            "status": "NOT_RUN_RATER_PANEL_PENDING",
            "preparation_status": "NOT_PREPARED_PHASE_B_TERMINAL_HOLD",
            "prompt_count": 12,
            "prompts": [],
            "planned_raters": 3,
            "report_self_sha256": "",
        }
        pilot_manifest["report_self_sha256"] = _canonical_hash({key: value for key, value in pilot_manifest.items() if key != "report_self_sha256"})
        _write_json(output_dir / "human_pilot_manifest.json", pilot_manifest)
    if not (output_dir / "HUMAN_RATER_PROTOCOL.md").is_file():
        protocol = (
            "# HUMAN_RATER_PROTOCOL — Phase B preparation\n\n"
            "Panel status: H1=user, H2=TBD, H3=TBD, RATER_PANEL_READY=false. "
            "Use three independent human raters; do not substitute AI judges. Rate G/R/C/E/N on a 1–5 scale. "
            "Human pilot is NOT_RUN_RATER_PANEL_PENDING; main human evaluation is NOT_RUN. "
            "No human evaluation or OMEGA trained generation has been run.\n"
        )
        protocol_path = output_dir / "HUMAN_RATER_PROTOCOL.md"
        protocol_path.write_text(protocol, encoding="utf-8", newline="\n")
        (output_dir / "HUMAN_RATER_PROTOCOL.md.sha256").write_text(_sha256_file(protocol_path) + "\n", encoding="ascii", newline="\n")
    if not (output_dir / "OOV_POLICY_DECISION.json").is_file():
        counts = hold["train_counts"]
        total = sum(counts.values())
        types = len(counts)
        beta_u = types / (total + types)
        decision = {
            "schema": "omega-absolute-calibration-oov-policy-decision-v1",
            "status": "OOV_POLICY_FROZEN_APPLIED",
            "policy_ids": ["UNIGRAM_FULLVOCAB_UNIFORM_BASE_V1", "MKN5_FULLVOCAB_UNIFORM_BASE_V1"],
            "vocabulary_size": VOCAB_SIZE,
            "unknown_mapping": "NONE; retain every GPT-2 token ID as a distinct event",
            "unk_token": "NONE",
            "add_k": "FORBIDDEN",
            "epsilon_or_manual_floor": "FORBIDDEN",
            "vall_oov_mask": {
                "definition": "the frozen HOLD token-ID set absent from the train target count table",
                "hold_oov_report_self_sha256": hold["hold_file_identities"]["absolute_calibration_oov_hold.json"]["self_sha256"],
                "type_count": len(hold["oov_ids"]),
                "target_occurrences": int(hold["oov_report"]["vall_oov_target_occurrences"]),
            },
            "unigram": {
                "formula": "beta_U=T/(N+T); P_U(w)=(1-beta_U)*count(w)/N+beta_U/V",
                "N_train_target_presentations": total,
                "T_observed_types": types,
                "V": VOCAB_SIZE,
                "beta_U_calculated": beta_u,
                "beta_U_expected_crosscheck": 0.008670502743944868,
            },
            "mkn5": {
                "formula": "beta_KN=T_C/(C_total+T_C); P1_KN(w)=(1-beta_KN)*C(w)/C_total+beta_KN/V",
                "V": VOCAB_SIZE,
                "beta_KN": None,
                "calculation_status": result_fields.get("hold_component_status", status),
                "higher_order": "standard interpolated modified Kneser-Ney backoff towards the terminal full-vocabulary base",
                "discount_fallback": "NONE; hold if estimation is undefined",
            },
            "base_hold_file_identities": hold["hold_file_identities"],
            "implementation_commit": provenance["implementation_commit"],
            "base_protocol_commit": BASE_PROTOCOL_COMMIT,
            "base_hold_commit": BASE_HOLD_COMMIT,
            "protocol_blob_sha256": PROTOCOL_BLOB,
            "decision_self_sha256": "",
        }
        decision["decision_self_sha256"] = _canonical_hash({key: value for key, value in decision.items() if key != "decision_self_sha256"})
        _write_json(output_dir / "OOV_POLICY_DECISION.json", decision)
    if not (output_dir / "absolute_lm_calibration.json").is_file():
        _seal_json(
            output_dir / "absolute_lm_calibration.json",
            {
                "schema": "omega-absolute-lm-calibration-v1",
                "status": "ABSOLUTE_CALIBRATION_HOLD",
                "reason": result_fields.get("hold_reason", "Phase-B calibration stopped before comparable NLL scoring"),
                "unigram": None,
                "modified_kneser_ney_5gram": None,
                "distilgpt2": None,
            },
        )
    names = tuple(extra_files) + (
        "phase_b_execution_identity.json", "OOV_POLICY_DECISION.json", "absolute_lm_calibration.json",
        "human_prompt_manifest.json", "main_human_manifest.json", "human_pilot_manifest.json",
        "human_rater_status.json", "HUMAN_RATER_PROTOCOL.md", "HUMAN_RATER_PROTOCOL.md.sha256",
    )
    return _finish(output_dir, status, provenance, result_fields, output_names=names, source_identity=source_identity)


def run_resume(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Phase-B OOV resume output is immutable: {output_dir}")
    provenance = _verify_implementation_provenance()
    hold = _load_hold_data()
    resume_sources = _verify_resume_sources(hold)
    provenance["hold_artifact_identities"] = hold["hold_file_identities"]
    output_dir.mkdir(parents=True, exist_ok=False)
    original_manifest = hold["hold_manifest"]
    stream = hold["stream_manifest"]
    oov_ids = hold["oov_ids"]
    train_counts = hold["train_counts"]
    execution_identity = {
        "schema": "omega-phase-b-oov-resume-execution-identity-v1",
        "provenance": provenance,
        "resume_sources": resume_sources,
        "hold_manifest_self_sha256": original_manifest["phase_b_manifest_sha256"],
        "execution_identity_self_sha256": "",
    }
    execution_identity["execution_identity_self_sha256"] = _canonical_hash({key: value for key, value in execution_identity.items() if key != "execution_identity_self_sha256"})
    _write_json(output_dir / "phase_b_execution_identity.json", execution_identity)

    missing_dependencies = _missing_runtime_dependencies()
    if missing_dependencies:
        return _write_hold_completion(
            output_dir,
            provenance,
            hold,
            {
                "hold_reason": "REQUIRED_LOCAL_RUNTIME_DEPENDENCIES_MISSING",
                "missing_dependencies": missing_dependencies,
                "hold_component_status": "DISTILGPT2_AND_LEAKAGE_RUNTIME_UNAVAILABLE",
                "NLLs_computed": False,
                "unigram_NLL_PPL": "NOT_COMPUTED",
                "MKN5_NLL_PPL": "NOT_COMPUTED",
                "DistilGPT2_NLL_PPL": "NOT_COMPUTED",
                "oov_type_count": len(oov_ids),
                "oov_target_occurrences": int(hold["oov_report"]["vall_oov_target_occurrences"]),
                "omega_checkpoint_loaded": False,
                "omega_trained_generation_or_scoring": False,
                "test_split_loaded": False,
                "training_updates": 0,
            },
            "ABSOLUTE_CALIBRATION_HOLD",
            source_identity=resume_sources,
        )

    from transformers import AutoTokenizer  # noqa: PLC0415
    from datasets import DownloadConfig, load_dataset  # noqa: PLC0415
    from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: PLC0415
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_REVISION,
        MODEL_ID,
        MODEL_REVISION,
        reconstruct_documents,
        sha256_bytes,
        sha256_text,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    if len(tokenizer) != VOCAB_SIZE or MODEL_REVISION != TOKENIZER_REVISION or DATASET_REVISION != DATASET_REVISION_EXPECTED:
        raise RuntimeError("FAIL_PROVENANCE: local tokenizer/dataset revision differs from frozen Phase-B inputs")
    if MODEL_ID != "distilbert/distilgpt2":
        raise RuntimeError(f"FAIL_PROVENANCE: unexpected teacher/tokenizer model id {MODEL_ID}")
    train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=DownloadConfig(local_files_only=True))
    vall_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION, download_config=DownloadConfig(local_files_only=True))
    train_manifest = json.loads(HISTORICAL_TRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    train_unsigned = dict(train_manifest)
    train_sig = train_unsigned.pop("manifest_sha256", None)
    if not train_sig or train_sig != _canonical_hash(train_unsigned) or train_sig != stream["source_hashes"]["historical_train_manifest_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: historical train manifest hash mismatch during leakage reconstruction")
    train_documents = []
    seen_train: set[tuple[str, str]] = set()
    for source_index, source in enumerate(reconstruct_documents(train_dataset)):
        text = str(source["text"])
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) < CHUNK_TOKENS:
            continue
        full_hash = sha256_text(text)
        first_hash = sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in ids[:CHUNK_TOKENS]))
        dedupe_key = (full_hash, first_hash)
        if dedupe_key in seen_train:
            continue
        seen_train.add(dedupe_key)
        train_documents.append(
            {
                "document_index": source_index,
                "row_range": list(source["row_range"]),
                "header": str(source["header"]),
                "full_text_sha256": full_hash,
                "retained_513_token_sha256": first_hash,
                "token_count": len(ids),
                "selected_token_count": CHUNK_TOKENS,
                "tokens": ids,
            }
        )
    expected_train = train_manifest["documents"]
    if len(train_documents) != 602 or len(expected_train) != 602:
        raise RuntimeError(f"FAIL_PROVENANCE: train leakage corpus count {len(train_documents)} != 602")
    for index, (actual, expected) in enumerate(zip(train_documents, expected_train)):
        for field in ("document_index", "row_range", "header", "token_count", "selected_token_count", "full_text_sha256", "retained_513_token_sha256"):
            if actual[field] != expected[field]:
                raise RuntimeError(f"FAIL_PROVENANCE: full train document identity/order mismatch at {index}, field={field}")

    vall_manifest, vall_groups, vall_documents, vall_target_sequences = _load_vall_documents(stream["source_hashes"])

    leakage_sources = {
        "train_manifest_self_sha256": train_sig,
        "train_document_order_sha256": train_manifest["dataset"]["document_order_sha256"],
        "vall_manifest_self_sha256": vall_manifest["manifest_sha256"],
        "vall_payload_sha256": stream["source_hashes"]["vall_payload_sha256"],
    }
    train_overlap = LongestTokenOverlapIndex([row["tokens"] for row in train_documents])

    stage_a_chunks = _read_jsonl(Path(stream["source_hashes"]["coverage_a_chunk_payload_path"]))
    stage_a_payload_path = Path(stream["source_hashes"]["coverage_a_chunk_payload_path"])
    if _sha256_file(stage_a_payload_path) != stream["source_hashes"]["coverage_a_chunk_payload_sha256"]:
        raise RuntimeError("FAIL_PROVENANCE: Coverage-A training chunk payload changed before KN5 counting")
    if len(stage_a_chunks) != 4378 or len(stream.get("unique_chunk_ids", [])) != 3869:
        raise RuntimeError("FAIL_PROVENANCE: frozen stream/chunk pool cardinality mismatch")
    presentations = stream["presentations"]
    if len(presentations) != 8000:
        raise RuntimeError("FAIL_PROVENANCE: frozen presentation stream no longer has 8000 chunk visits")
    for event in presentations:
        index = int(event["chunk_pool_index"])
        if index < 0 or index >= len(stage_a_chunks) or stage_a_chunks[index]["chunk_token_sha256"] != event["chunk_token_sha256"]:
            raise RuntimeError("FAIL_PROVENANCE: frozen presentation references an unexpected chunk pool record")

    tokenizer_identity = _verify_frozen_tokenizer_equivalence(
        tokenizer,
        stream,
        train_documents,
        stage_a_chunks,
        vall_manifest,
        vall_groups,
        vall_dataset,
        reconstruct_documents,
        sha256_text,
    )
    provenance["tokenizer_identity"] = tokenizer_identity
    execution_identity["provenance"] = provenance
    execution_identity["tokenizer_identity"] = tokenizer_identity
    execution_identity["execution_identity_self_sha256"] = ""
    execution_identity["execution_identity_self_sha256"] = _canonical_hash(
        {key: value for key, value in execution_identity.items() if key != "execution_identity_self_sha256"}
    )
    _write_json(output_dir / "phase_b_execution_identity.json", execution_identity)

    try:
        teacher = _load_teacher(tokenizer)
    except Exception as error:
        result_fields = {
            "hold_reason": f"DISTILGPT2_LOCAL_BASELINE_UNAVAILABLE: {type(error).__name__}: {error}",
            "hold_component_status": "DISTILGPT2_LOCAL_BASELINE_UNAVAILABLE",
            "NLLs_computed": False,
            "unigram_NLL_PPL": "NOT_COMPUTED",
            "MKN5_NLL_PPL": "NOT_COMPUTED",
            "DistilGPT2_NLL_PPL": "NOT_COMPUTED",
            "oov_type_count": len(oov_ids),
            "oov_target_occurrences": int(hold["oov_report"]["vall_oov_target_occurrences"]),
            "omega_checkpoint_loaded": False,
            "omega_trained_generation_or_scoring": False,
            "test_split_loaded": False,
            "training_updates": 0,
        }
        return _write_hold_completion(output_dir, provenance, hold, result_fields, "ABSOLUTE_CALIBRATION_HOLD", source_identity=resume_sources)

    presentation_tokens = (stage_a_chunks[int(event["chunk_pool_index"])]["tokens"] for event in presentations)
    try:
        mkn, mkn_metadata = ModifiedKneserNey5.from_presentations(presentation_tokens)
    except CalibrationHold as error:
        hold_reason = str(error)
        phase_status = "ABSOLUTE_CALIBRATION_HOLD"
        result_fields = {
            "hold_reason": hold_reason,
            "hold_component_status": "MKN_DISCOUNT_ESTIMATION_HOLD",
            "unigram_nll_ppl": "NOT_COMPUTED",
            "mkn5_nll_ppl": "NOT_COMPUTED",
            "distilgpt2_nll_ppl": "NOT_COMPUTED",
            "oov_type_count": len(oov_ids),
            "oov_target_occurrences": int(hold["oov_report"]["vall_oov_target_occurrences"]),
        }
        return _write_hold_completion(output_dir, provenance, hold, result_fields, phase_status, source_identity=resume_sources)
    if mkn_metadata["target_events"] != int(stream["target_token_presentations"]) or mkn_metadata["target_events"] != 4_096_000:
        raise RuntimeError("FAIL_PROVENANCE: MKN event counts differ from the frozen Coverage-C target-presentation stream")

    beta_u = len(train_counts) / (sum(train_counts.values()) + len(train_counts))
    if not math.isclose(beta_u, 0.008670502743944868, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"FAIL_PROVENANCE: beta_U calculation differs from approved full-vocabulary policy: {beta_u}")
    unigram = FullVocabularyUnigram.from_counts(train_counts, total_targets=sum(train_counts.values()), vocabulary_size=VOCAB_SIZE)
    if not math.isclose(unigram.full_normalizer(), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CalibrationHold("UNIGRAM_FULLVOCAB_NORMALIZATION_HOLD")
    if any(unigram.probability(token) <= 0 for token in range(VOCAB_SIZE)):
        raise CalibrationHold("UNIGRAM_FULLVOCAB_POSITIVITY_HOLD")
    unseen_probs = {unigram.probability(token) for token in range(VOCAB_SIZE) if token not in train_counts}
    if len(unseen_probs) != 1 or not unseen_probs:
        raise CalibrationHold("UNIGRAM_UNSEEN_FLOOR_HOLD")

    oov_policy_decision = {
        "schema": "omega-absolute-calibration-oov-policy-decision-v1",
        "status": "OOV_POLICY_FROZEN_APPLIED",
        "policy_ids": ["UNIGRAM_FULLVOCAB_UNIFORM_BASE_V1", "MKN5_FULLVOCAB_UNIFORM_BASE_V1"],
        "vocabulary_size": VOCAB_SIZE,
        "unknown_mapping": "NONE; retain every GPT-2 token ID as a distinct event",
        "unk_token": "NONE",
        "add_k": "FORBIDDEN",
        "epsilon_or_manual_floor": "FORBIDDEN",
        "vall_oov_mask": {
            "definition": "token id absent from raw train target vocabulary in the frozen Coverage-C presentation stream",
            "hold_report_self_sha256": hold["hold_file_identities"]["absolute_calibration_oov_hold.json"]["self_sha256"],
            "type_count": len(oov_ids),
            "target_occurrences": int(hold["oov_report"]["vall_oov_target_occurrences"]),
        },
        "unigram": {
            "formula": "beta_U=T/(N+T); P_U(w)=(1-beta_U)*count(w)/N+beta_U/V",
            "N_train_target_presentations": sum(train_counts.values()),
            "T_observed_types": len(train_counts),
            "V": VOCAB_SIZE,
            "beta_U_calculated": beta_u,
            "beta_U_expected_crosscheck": 0.008670502743944868,
            "all_vocabulary_probabilities_positive": True,
            "normalization_sum": unigram.full_normalizer(),
        },
        "mkn5": {
            "formula": "beta_KN=T_C/(C_total+T_C); P1_KN(w)=(1-beta_KN)*C(w)/C_total+beta_KN/V",
            "V": VOCAB_SIZE,
            "beta_KN_calculated_from_frozen_train_stream": mkn.beta_kn,
            "C_total": mkn.c_total,
            "T_C": mkn.t_c,
            "discounts_by_order": mkn_metadata["count_of_counts_and_discounts_by_order"],
            "higher_order": "standard interpolated modified Kneser-Ney backoff towards this terminal full-vocabulary base",
            "no_vall_counts_used_for_discounts_or_beta": True,
        },
        "base_hold_artifact_hashes": hold["hold_file_identities"],
        "implementation_commit": provenance["implementation_commit"],
        "tokenizer_identity": tokenizer_identity,
        "distilgpt2_weights_sha256": provenance["distilgpt2_weights_sha256"],
        "distilgpt2_weights_revision": provenance["distilgpt2_weights_revision"],
        "distilgpt2_weights_size_bytes": provenance["distilgpt2_weights_size_bytes"],
        "base_protocol_commit": BASE_PROTOCOL_COMMIT,
        "base_hold_commit": BASE_HOLD_COMMIT,
        "protocol_blob_sha256": PROTOCOL_BLOB,
        "decision_self_sha256": "",
    }
    oov_policy_decision["decision_self_sha256"] = _canonical_hash({key: value for key, value in oov_policy_decision.items() if key != "decision_self_sha256"})
    _write_json(output_dir / "OOV_POLICY_DECISION.json", oov_policy_decision)

    mkn_meta_file = {
        "schema": "omega-mkn5-train-metadata-v1",
        "stream_manifest_self_sha256": stream["manifest_sha256"],
        "stream_target_presentations": mkn_metadata["target_events"],
        "raw_5gram_types": mkn_metadata["raw_5gram_types"],
        "effective_ngram_types_by_order": mkn_metadata["effective_ngram_types_by_order"],
        "count_of_counts_and_discounts_by_order": mkn_metadata["count_of_counts_and_discounts_by_order"],
        "C_total": mkn_metadata["C_total"],
        "T_C": mkn_metadata["T_C"],
        "beta_KN": mkn_metadata["beta_KN"],
        "cross_document_or_chunk_ngrams": 0,
        "normalization_checks": [],
        "report_self_sha256": "",
    }
    normalization_rows = []
    random_ctx = random.Random(FIXED_GENERATOR_SEED)
    for order in range(1, 6):
        for sample_index in range(3):
            doc_order = random_ctx.randrange(60)
            rows = vall_groups[doc_order]
            chunk = rows[random_ctx.randrange(len(rows))]
            target_position = random_ctx.randrange(max(1, order - 1), 513)
            hist = [int(token) for token in chunk["tokens"][max(0, target_position - (order - 1)) : target_position]] if order > 1 else []
            hist = [BOS_ID] * (order - 1 - len(hist)) + hist
            probability_sum = math.fsum(mkn.probability(hist, token_id) for token_id in range(VOCAB_SIZE))
            probabilities = [mkn.probability(hist, token_id) for token_id in range(VOCAB_SIZE)]
            if not all(math.isfinite(value) and value > 0.0 for value in probabilities) or not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-10):
                raise CalibrationHold(f"MKN_REAL_CONTEXT_NORMALIZATION_HOLD order={order} doc={doc_order} sum={probability_sum}")
            normalization_rows.append({"order": order, "sample_index": sample_index, "document_order_index": doc_order, "chunk_index": int(chunk["chunk_index"]), "history_token_ids": hist, "probability_sum_fp64": probability_sum, "minimum_probability": min(probabilities), "all_finite_positive": True})
    mkn_meta_file["normalization_checks"] = normalization_rows
    mkn_meta_file["report_self_sha256"] = _canonical_hash({key: value for key, value in mkn_meta_file.items() if key != "report_self_sha256"})
    _write_json(output_dir / "mkn5_training_metadata.json", mkn_meta_file)

    # Seal/reject the generated bank before scoring any L0/L1 model responses.
    bank, bank_manifest, leakage_report = _seal_instrument_bank(
        tokenizer, train_documents, vall_documents, output_dir, leakage_sources
    )
    _write_json(output_dir / "instrument_bank.json", bank)
    bank_manifest["bank_file_sha256"] = _sha256_file(output_dir / "instrument_bank.json")
    bank_manifest["manifest_self_sha256"] = _canonical_hash({key: value for key, value in bank_manifest.items() if key != "manifest_self_sha256"})
    _write_json(output_dir / "instrument_bank_manifest.json", bank_manifest)
    _write_json(output_dir / "leakage_report.json", leakage_report)
    main_rows, pilot_rows, reserved_spans = _prepare_human_manifests(
        output_dir, vall_target_sequences, vall_manifest, train_overlap, tokenizer
    )

    # Absolute VALL NLLs, all including the same 1,198-token frozen OOV mask.
    unigram = _aggregate_chunk_metrics(vall_groups, lambda tokens: score_unigram_chunk(unigram, tokens, oov_ids))
    mkn5 = _aggregate_chunk_metrics(vall_groups, lambda tokens: mkn.score_chunk(tokens, oov_ids))
    teacher_vall = _score_distilgpt2_vall(teacher, vall_groups, oov_ids)
    absolute_calibration = {
        "schema": "omega-absolute-lm-calibration-v1",
        "status": "ABSOLUTE_CALIBRATION_COMPLETE",
        "vocabulary": {
            "tokenizer": "GPT-2 BPE",
            "revision": TOKENIZER_REVISION,
            "vocab_size": VOCAB_SIZE,
            "frozen_tokenizer_metadata_sha256": tokenizer_identity["frozen_tokenizer_metadata_sha256"],
            "runtime_tokenizer_metadata_sha256": tokenizer_identity["runtime_tokenizer_metadata_sha256"],
            "exact_frozen_train_chunk_token_ids_verified": tokenizer_identity["exact_frozen_train_chunk_token_ids_verified"],
            "exact_frozen_vall_chunk_token_ids_verified": tokenizer_identity["exact_frozen_vall_chunk_token_ids_verified"],
        },
        "vall": {"chunks": VALL_CHUNKS, "targets": VALL_TARGETS, "primary_weighting": "token-weighted", "secondary_weighting": "document-macro"},
        "oov_policy_decision_self_sha256": oov_policy_decision["decision_self_sha256"],
        "models": {
            "UNIGRAM_FULLVOCAB_UNIFORM_BASE_V1": unigram,
            "MKN5_FULLVOCAB_UNIFORM_BASE_V1": {**mkn5, "training_metadata_self_sha256": mkn_meta_file["report_self_sha256"]},
            "DISTILGPT2": {
                **teacher_vall,
                "model_id": "distilbert/distilgpt2",
                "revision": TOKENIZER_REVISION,
                "distilgpt2_weights_revision": provenance["distilgpt2_weights_revision"],
                "distilgpt2_weights_sha256": provenance["distilgpt2_weights_sha256"],
                "distilgpt2_weights_size_bytes": provenance["distilgpt2_weights_size_bytes"],
                "same_frozen_oov_mask_for_diagnostics": True,
            },
        },
        "opus_prediction": {"kn5_nll_lt_5_5": "prediction_only_not_gate", "omega_k4_2k_minus_kn5_gt_1_1": "prediction_only_not_gate"},
        "report_self_sha256": "",
    }
    absolute_calibration["report_self_sha256"] = _canonical_hash({key: value for key, value in absolute_calibration.items() if key != "report_self_sha256"})
    _write_json(output_dir / "absolute_lm_calibration.json", absolute_calibration)

    l0_l1, l1_validity = _score_l0_l1_references(bank, mkn, teacher)
    l0_l1_report = {
        "schema": "omega-l0-l1-reference-calibration-v1",
        "status": "INSTRUMENT_INVALID_L1" if l1_validity["invalid"] else "INSTRUMENT_VALID_L1",
        "instrument_bank_self_sha256": bank["bank_self_sha256"],
        "instrument_bank_manifest_self_sha256": bank_manifest["manifest_self_sha256"],
        "leakage_report_self_sha256": leakage_report["report_self_sha256"],
        "L0_LOCAL_LANGUAGE_SANITY": l0_l1["L0_LOCAL_LANGUAGE_SANITY"],
        "L1_CONTEXTUAL_COMPETENCE": l0_l1["L1_CONTEXTUAL_COMPETENCE"],
        "report_self_sha256": "",
    }
    l0_l1_report["report_self_sha256"] = _canonical_hash({key: value for key, value in l0_l1_report.items() if key != "report_self_sha256"})
    _write_json(output_dir / "l0_l1_reference_calibration.json", l0_l1_report)

    if l1_validity["invalid"]:
        result_fields = {
            "absolute_calibration": {key: value["nll_total_token_weighted"] for key, value in absolute_calibration["models"].items()},
            "L0_accuracy": l0_l1_report["L0_LOCAL_LANGUAGE_SANITY"]["reference_accuracy"],
            "L1_validity": l0_l1_report["L1_CONTEXTUAL_COMPETENCE"]["validity"],
            "tokenizer_identity": tokenizer_identity,
            "distilgpt2_weights": {
                "revision": provenance["distilgpt2_weights_revision"],
                "sha256": provenance["distilgpt2_weights_sha256"],
                "size_bytes": provenance["distilgpt2_weights_size_bytes"],
            },
            "FP_U_FP_threshold_auto": "NOT_RUN_L1_INSTRUMENT_INVALID",
            "RATER_PANEL_READY": False,
        }
        return _finish(
            output_dir,
            "INSTRUMENT_INVALID_L1",
            provenance,
            result_fields,
            output_names=(
                "OOV_POLICY_DECISION.json", "mkn5_training_metadata.json", "absolute_lm_calibration.json",
                "instrument_bank.json", "instrument_bank_manifest.json", "leakage_report.json",
                "phase_b_execution_identity.json",
                "l0_l1_reference_calibration.json", "human_prompt_manifest.json", "main_human_manifest.json",
                "human_pilot_manifest.json", "human_rater_status.json", "HUMAN_RATER_PROTOCOL.md",
                "HUMAN_RATER_PROTOCOL.md.sha256",
            ),
            source_identity=resume_sources,
        )

    reference_spans, control_spans = select_nonoverlapping_vall_windows(vall_target_sequences, reserved_spans, reference_count=256, control_count=256)
    def window_rows(spans: Sequence[dict[str, int]], prefix: str) -> list[dict[str, Any]]:
        result = []
        for index, span in enumerate(spans):
            doc = int(span["document_order_index"])
            ids = vall_target_sequences[doc][int(span["start"]) : int(span["end"])]
            if len(ids) != WINDOW_TOKENS:
                raise ValueError("degeneration instrument span is not 96 BPE tokens")
            result.append({"window_id": f"{prefix}-{index + 1:03d}", "document_order_index": doc, "start": span["start"], "end": span["end"], "token_ids": ids})
        return result

    reference_windows = window_rows(reference_spans, "REF")
    control_windows = window_rows(control_spans, "CTRL")
    ref_metrics = [{"window_id": row["window_id"], **window_degeneration_metrics(row["token_ids"])} for row in reference_windows]
    metrics = ("distinct_1", "distinct_2", "distinct_4", "max_4gram_recurrence")
    intervals = {metric: _type7_values([float(row[metric]) for row in ref_metrics]) for metric in metrics}
    deg_control_rows = []
    for row in control_windows:
        values = window_degeneration_metrics(row["token_ids"])
        deg_control_rows.append({"window_id": row["window_id"], **values, "corpus_extreme_degenerate": corpus_extreme_degenerate(values, intervals)})
    fp_count = sum(row["corpus_extreme_degenerate"] for row in deg_control_rows)
    fp = fp_count / len(deg_control_rows)
    u_fp = clopper_pearson_upper_one_sided(fp_count, len(deg_control_rows), confidence=0.90)
    threshold_auto = threshold_auto_from_u_fp(u_fp)
    degeneracy_invalid = fp > 0.20
    reference_report = {
        "schema": "omega-degeneration-reference-v1",
        "status": "COMPLETE",
        "window_tokens": WINDOW_TOKENS,
        "reference_window_count": len(reference_windows),
        "source_vall_manifest_self_sha256": vall_manifest["manifest_sha256"],
        "metrics": metrics,
        "quantile_method": "Type-7 linear interpolation",
        "intervals_q05_q95": intervals,
        "exact_cycle_length_distribution_1_to_8": {
            str(length): sum(row["exact_cycle"] is not None and int(row["exact_cycle"]["cycle_length"]) == length for row in ref_metrics)
            for length in range(1, 9)
        },
        "no_qualifying_exact_cycle_count": sum(row["exact_cycle"] is None for row in ref_metrics),
        "windows": ref_metrics,
        "report_self_sha256": "",
    }
    reference_report["report_self_sha256"] = _canonical_hash({key: value for key, value in reference_report.items() if key != "report_self_sha256"})
    _write_json(output_dir / "degeneration_reference.json", reference_report)
    control_report = {
        "schema": "omega-degeneration-control-v1",
        "status": "DEGENERATION_INSTRUMENT_INVALID" if degeneracy_invalid else "CONTROL_VALID",
        "control_window_count": len(control_windows),
        "fp_count": fp_count,
        "fp_rate": fp,
        "u_fp_exact_one_sided_90": u_fp,
        "clopper_pearson_confidence": 0.90,
        "threshold_auto": threshold_auto,
        "threshold_formula": "max(0.10, 2 * U_FP)",
        "threshold_applied_to_omega": False,
        "instrument_invalid_condition": "FP > 0.20",
        "exact_cycle_length_distribution_1_to_8": {
            str(length): sum(row["exact_cycle"] is not None and int(row["exact_cycle"]["cycle_length"]) == length for row in deg_control_rows)
            for length in range(1, 9)
        },
        "no_qualifying_exact_cycle_count": sum(row["exact_cycle"] is None for row in deg_control_rows),
        "windows": deg_control_rows,
        "report_self_sha256": "",
    }
    control_report["report_self_sha256"] = _canonical_hash({key: value for key, value in control_report.items() if key != "report_self_sha256"})
    _write_json(output_dir / "degeneration_control.json", control_report)
    if degeneracy_invalid:
        return _finish(output_dir, "DEGENERATION_INSTRUMENT_INVALID", provenance, {
            "absolute_calibration_nll_ppl": {name: {"nll_total": row["nll_total_token_weighted"], "ppl_total": row["ppl_total"]} for name, row in absolute_calibration["models"].items()},
            "tokenizer_identity": tokenizer_identity,
            "distilgpt2_weights": {
                "revision": provenance["distilgpt2_weights_revision"],
                "sha256": provenance["distilgpt2_weights_sha256"],
                "size_bytes": provenance["distilgpt2_weights_size_bytes"],
            },
            "L0_reference_accuracy": l0_l1["L0_LOCAL_LANGUAGE_SANITY"]["reference_accuracy"],
            "L1_reference_validity": l0_l1["L1_CONTEXTUAL_COMPETENCE"]["validity"],
            "FP": fp,
            "U_FP": u_fp,
            "threshold_auto": threshold_auto,
            "human_panel_ready": False,
        }, output_names=("phase_b_execution_identity.json", "OOV_POLICY_DECISION.json", "mkn5_training_metadata.json", "absolute_lm_calibration.json", "instrument_bank.json", "instrument_bank_manifest.json", "leakage_report.json", "l0_l1_reference_calibration.json", "degeneration_reference.json", "degeneration_control.json", "human_prompt_manifest.json", "human_pilot_manifest.json", "main_human_manifest.json", "human_rater_status.json", "HUMAN_RATER_PROTOCOL.md", "HUMAN_RATER_PROTOCOL.md.sha256"), source_identity=resume_sources)

    result_fields = {
        "absolute_calibration_nll_ppl": {name: {"nll_total": row["nll_total_token_weighted"], "ppl_total": row["ppl_total"], "document_macro_nll": row["document_macro_nll"], "seen_targets": row["seen_target_count"], "oov_targets": row["oov_target_count"], "nll_contribution_seen": row["nll_contribution_seen_to_total"], "nll_contribution_oov": row["nll_contribution_oov_to_total"]} for name, row in absolute_calibration["models"].items()},
        "tokenizer_identity": tokenizer_identity,
        "distilgpt2_weights": {
            "revision": provenance["distilgpt2_weights_revision"],
            "sha256": provenance["distilgpt2_weights_sha256"],
            "size_bytes": provenance["distilgpt2_weights_size_bytes"],
        },
        "L0_reference_accuracy": l0_l1["L0_LOCAL_LANGUAGE_SANITY"]["reference_accuracy"],
        "L1_reference_validity": l0_l1["L1_CONTEXTUAL_COMPETENCE"]["validity"],
        "degeneration_FP": fp,
        "degeneration_U_FP": u_fp,
        "threshold_auto": threshold_auto,
        "human_panel": {"H1": "user", "H2": "TBD", "H3": "TBD", "RATER_PANEL_READY": False},
        "human_pilot": "NOT_RUN_RATER_PANEL_PENDING",
        "main_human_evaluation": "NOT_RUN",
        "test_split_loaded": False,
        "omega_checkpoint_loaded": False,
        "omega_trained_scoring_or_generation": False,
        "training_updates": 0,
    }
    output_names = (
        "phase_b_execution_identity.json", "OOV_POLICY_DECISION.json", "mkn5_training_metadata.json", "absolute_lm_calibration.json",
        "instrument_bank.json", "instrument_bank_manifest.json", "leakage_report.json",
        "l0_l1_reference_calibration.json", "degeneration_reference.json", "degeneration_control.json",
        "human_prompt_manifest.json", "human_pilot_manifest.json", "main_human_manifest.json",
        "human_rater_status.json", "HUMAN_RATER_PROTOCOL.md", "HUMAN_RATER_PROTOCOL.md.sha256",
    )
    return _finish(output_dir, "PHASE_B_AUTOMATIC_PASS_HUMAN_PANEL_PENDING", provenance, result_fields, output_names=output_names, source_identity=resume_sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-phase-b-oov-resume-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.confirm_phase_b_oov_resume_go:
        parser.error("Phase-B OOV resume requires the explicit MD/299 GO")
    output_dir = args.output_dir.resolve()
    _install_omega_checkpoint_fail_closed_guard()
    try:
        report = run_resume(output_dir)
    except FileExistsError:
        raise
    except CalibrationHold as error:
        if not output_dir.exists():
            raise
        hold = _load_hold_data()
        execution_identity_path = output_dir / "phase_b_execution_identity.json"
        if not execution_identity_path.is_file():
            raise
        execution_identity = json.loads(execution_identity_path.read_text(encoding="utf-8"))
        _verify_self_hash(execution_identity, "execution_identity_self_sha256", "Phase-B execution identity")
        provenance = execution_identity["provenance"]
        source_identity = execution_identity["resume_sources"]
        reason = str(error)
        if reason.startswith("INSTRUMENT_LEAKAGE_HOLD"):
            terminal_status = "INSTRUMENT_INVALID_L1"
        elif reason.startswith("HUMAN_SPAN_SELECTION_HOLD"):
            terminal_status = "ABSOLUTE_CALIBRATION_HOLD"
        else:
            terminal_status = "ABSOLUTE_CALIBRATION_HOLD"
        report = _write_hold_completion(
            output_dir,
            provenance,
            hold,
            {
                "hold_reason": reason,
                "NLLs_computed": False,
                "optimizer_updates": 0,
                "omega_checkpoint_loaded": False,
                "omega_trained_generation_or_scoring": False,
                "test_split_loaded": False,
            },
            terminal_status,
            source_identity=source_identity,
            extra_files=tuple(path.name for path in output_dir.iterdir() if path.is_file()),
        )
    except Exception as error:
        if not output_dir.exists() or (output_dir / "phase_b_completion_report.json").is_file():
            raise
        hold = _load_hold_data()
        execution_identity_path = output_dir / "phase_b_execution_identity.json"
        if not execution_identity_path.is_file():
            raise
        execution_identity = json.loads(execution_identity_path.read_text(encoding="utf-8"))
        _verify_self_hash(execution_identity, "execution_identity_self_sha256", "Phase-B execution identity")
        provenance = execution_identity["provenance"]
        source_identity = execution_identity["resume_sources"]
        report = _write_hold_completion(
            output_dir,
            provenance,
            hold,
            {
                "hold_reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "NLLs_computed": False,
                "optimizer_updates": 0,
                "omega_checkpoint_loaded": False,
                "omega_trained_generation_or_scoring": False,
                "test_split_loaded": False,
            },
            "FAIL_PROVENANCE",
            source_identity=source_identity,
            extra_files=tuple(path.name for path in output_dir.iterdir() if path.is_file()),
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "phase_b_manifest_self_sha256": report.get("phase_b_manifest_self_sha256"),
                "phase_b_report_sha256": report.get("phase_b_report_sha256"),
                "raw_results": report.get("raw_results", {}),
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] in (
        "PHASE_B_AUTOMATIC_PASS_HUMAN_PANEL_PENDING",
        "PHASE_B_READY_FOR_HUMAN_PILOT",
        "INSTRUMENT_INVALID_L1",
        "DEGENERATION_INSTRUMENT_INVALID",
        "ABSOLUTE_CALIBRATION_HOLD",
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
