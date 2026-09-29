"""Verify and seal OMEGA backend-quality inputs; never starts training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parents[1]
CAMPAIGN_ROOT = LAB_ROOT / "campaign"
REPO_ROOT = LAB_ROOT.parent
R1_FREEZE = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_pilot_freeze"
R1_SCOPE = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
R1_FAST = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
P2R0 = CAMPAIGN_ROOT / "omega_native_runtime_p2r0"
HIDDEN_CACHE = CAMPAIGN_ROOT / "omega_teacher_hidden_cache_probe"
CE = CAMPAIGN_ROOT / "omega_ce_only_baseline"
EXPANDED_VALIDATION = CAMPAIGN_ROOT / "omega_expanded_frozen_validation"
BPF2A = Path(r"C:\Users\danil\bpf2a")

BE376_DLL = BPF2A / "t1_trainability_lab_v0.1.0" / "campaign" / "omega_native_runtime_p2r0" / "native" / "build-diagnostics-out-state-candidate-verified" / "python" / "omega_recurrent.dll"
EXPECTED_DLL_SHA256 = "be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc"
R1_NATIVE_CODE_COMMIT = "18e7225b819af61304189eff8537d602f729cf2f"
R1_COVERAGE_COMMIT = "d993b3c1f1e19e4bffa32d7c4be35687257b28c6"
R1_ARCHITECTURE_COMMIT = "269a4d79b5a1e6df8c230962e2b8c9e237095f18"
R1_NATIVE_SOURCE_BLOB_SHA256 = "f7659402751c3e578757ddbcef5fb500495e18e47b83ef135d1aab16685bab96"
VALIDATION_MANIFEST_SHA256 = "857800cd725a5648dac54f1a16285e3437be7732f372f179c5b5981480cd3019"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TEACHER_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
CACHE_SHA256 = "b43e9c37f9585e1c5d605df5cc58caba84a10480b4f8f06e747cb8d2f3cd16ff"
DEFAULT_SYNTHETIC_SENSITIVITY = HERE / "results" / "preparation" / "statistical_sensitivity.json"
LOSS_CONTRACT_ID = "R1_MASKED_TOKEN_MEAN_V1"
LOSS_IMPLEMENTATION = HERE / "r1_masked_token_mean_loss.py"
LOSS_TESTS = HERE / "test_r1_masked_token_mean_loss.py"
R1_LOSS_REFERENCE = LAB_ROOT / "scripts" / "run_omega_core_lm_0_r1_training_technical_preflight.py"
HISTORICAL_R2_LOSS = CAMPAIGN_ROOT / "omega_teacher_logit_cache" / "run_omega_teacher_logit_cache.py"
RESUME_TEST_RUNNER = HERE / "run_quality_resume_contract_test.py"
CAUSAL_SMOKE_RUNNER = HERE / "run_loss_contract_causal_smoke.py"
COMMON_ORIGIN_RUNNER = HERE / "run_corrected_loss_window1_common_origin.py"
STABLE_RECHARACTERIZATION_RUNNER = HERE / "run_corrected_loss_stable_8p32.py"
PREVIOUS_QUALIFICATION_MANIFEST = HERE / "inputs_r1_masked_token_mean_v1_final" / "qualification_manifest.json"
ORIGINAL_SMOKE_REPORT = HERE / "results" / "loss_contract_causal_smoke_20260924T115525" / "causal_smoke_report.json"
COMMON_ORIGIN_REPORT = HERE / "results" / "corrected_loss_window1_common_origin_20260924T121844" / "window1_common_origin_report.json"
STABLE_RECHARACTERIZATION_REPORT = HERE / "results" / "corrected_loss_stable_8p32_20260924T123036" / "stable_8p32_report.json"

EXPECTED_HASHES = {
    "r1_freeze_doc": "64e8fdd46b3fdafb61724e94701e9bd26749ef5fc6e2a99cb335f45bdbeadd61",
    "r1_freeze_manifest": "6fc9c6ba62730a837c79a27bf121daf301a53664050e4776fd5f47a09c69710c",
    "r1_model": "bc0250593dd7db03cc185f140c9603a3e63b70afb565bc548ceba21a9eb075a6",
    "r1_fast_candidate": "044f342fce9544df955afe7600017967d639f6f882c7d4d0e25da7dbeb18bf11",
    "r1_technical_preflight_current": "bb6d86ca2d1185efd1f359d187c0207384aaf301f17fb7290845c31dc091f2d4",
    "r1_scope_runner": "d9f8795a36fa57781f7735ff69b3b139155f448c95f900357a53ab5805e4a3f7",
    "expanded_validation_runner": "e3d67db5fbb995a5854b8f91e7fa83f4c8a768236f086a672a8f12356850704e",
    "nominal_runner": "e19e000a6f040d26e6413d4137712d227c3ac32f7780e728b2ce5b63bc8e5fdb",
    "r2_runner": "e52cccd91ae9aabf13e06fb90eeb91c6a45471a85fccb9882d4ff15e736f07fd",
    "r2_production_bridge": "7b359d49b093509aab3db6ccc8492c74abc1a3e035b8103e8d27e42813626697",
    "r2_p0_helper": "a7ec530f94d4cf1f98c2650a01da6f3beb609680c5cb6914fbf0b26f1e3cab67",
    "teacher_logit_cache_base": "ce025569ea1f3788c25f25d36afc4ae0a053a71698786983b95bb0689074fa1a",
    "hidden_cache_production_integration": "063cf6f1b8fe66969b3bff115b1d2c8eadb51c76be080c07d1d162bc6fb40df4",
    "teacher_hidden_cache_probe": "9a9a6d4ca356d325a23aba04b0159d76fba46d7352708a256ef6752410c4bdb4",
    "ce_model_factory": "9141e3e071cddcb3a078c77b8eab080712b94fe2b26691f664023f788256701c",
    "cache_manifest": "be3e1d2193670be281ef9ef59d6daa52b4569b9ded6c7080081f59e37e94d9f7",
    "source_train_manifest": "57ffc59374f99e35b28cd3726218f682539f7407bd97442bcab4da590d5c6477",
}
SOURCE_TRAIN_INTERNAL_SHA256 = "a114b42252cef45dddca18bd4d1d31eebb0fd966363265539bc69703a00546ef"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _file_identity(label: str, path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"{label} hash mismatch: expected {expected_sha}, got {digest}")
    return {"path": str(path), "sha256": digest}


def _verify_train_manifest(source_path: Path, cache_manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    docs = source.get("documents")
    cyclic = source.get("cyclic_pairs")
    if source.get("split") != "train" or not isinstance(docs, list) or len(docs) != 602:
        raise ValueError("source train manifest is not the expected 602-document training split")
    if source.get("document_count") != 602 or not isinstance(cyclic, dict):
        raise ValueError("source training document count/cyclic-pairs metadata mismatch")
    if source.get("manifest_sha256") != SOURCE_TRAIN_INTERNAL_SHA256:
        raise ValueError("source train manifest's self-identity hash changed")
    if cyclic.get("documents_available") != 602 or cyclic.get("pair_count") != 1000:
        raise ValueError("source training manifest does not contain exact Block-A 1000-pair schedule")
    if cyclic.get("documents_consumed") != 8000 or len(cyclic.get("pairs", [])) != 1000:
        raise ValueError("source 1000-pair schedule is incomplete")
    document_keys: set[tuple[str, str]] = set()
    previous_index = -1
    for document in docs:
        index = int(document["document_index"])
        key = (str(document["full_text_sha256"]), str(document["retained_513_token_sha256"]))
        if index <= previous_index:
            raise ValueError("training documents are not in strictly increasing source order")
        previous_index = index
        if key in document_keys:
            raise ValueError("training manifest contains duplicate document key")
        if int(document["selected_token_count"]) != 513 or int(document["token_count"]) < 513:
            raise ValueError("training manifest violates 513-token eligibility")
        document_keys.add(key)

    expected_pairs: list[dict[str, Any]] = []
    pair_rows = cyclic["pairs"]
    for pair_index, pair in enumerate(pair_rows):
        start = 8 * pair_index
        expected_indices = [(start + offset) % len(docs) for offset in range(8)]
        expected_wraps = ((start + 8 - 1) // len(docs)) - (start // len(docs))
        expected_keys = [
            [docs[index]["full_text_sha256"], docs[index]["retained_513_token_sha256"]]
            for index in expected_indices
        ]
        if pair.get("pair") != pair_index or pair.get("document_indices") != expected_indices:
            raise ValueError(f"cyclic pair {pair_index} differs from frozen formula")
        if pair.get("document_keys") != expected_keys or pair.get("start_offset") != start % len(docs):
            raise ValueError(f"cyclic pair {pair_index} keys/start offset mismatch")
        if int(pair.get("wraps", -1)) != expected_wraps:
            raise ValueError(f"cyclic pair {pair_index} wrap count mismatch")
        if pair.get("window_0_and_window_1_share_documents") is not True:
            raise ValueError(f"cyclic pair {pair_index} does not bind both causal windows")
        expected_pairs.append({
            "pair": pair_index,
            "document_indices": expected_indices,
            "document_keys": expected_keys,
            "start_offset": start % len(docs),
            "wraps": expected_wraps,
            "window_0_and_window_1_share_documents": True,
        })

    cache_entries = cache_manifest.get("entries")
    if not isinstance(cache_entries, dict):
        raise ValueError("hidden-cache manifest entries must be a mapping")
    cache_keys: set[tuple[str, str, int]] = set()
    cache_contexts: dict[int, set[tuple[int, int]]] = {0: set(), 1: set()}
    for entry in cache_entries.values():
        fields = entry["cache_key"]["fields"]
        document = fields["document"]
        window = int(fields["window"])
        key = (str(document["full_text_sha256"]), str(document["retained_513_token_sha256"]), window)
        cache_keys.add(key)
        cache_contexts[window].add(tuple(int(value) for value in fields["teacher_context_range"]))
        if list(entry.get("shape", [])) != [256, 768] or int(entry.get("window", -1)) != window:
            raise ValueError("hidden-cache entry shape/window mismatch")
    expected_cache_keys = {(full_hash, token_hash, window) for full_hash, token_hash in document_keys for window in (0, 1)}
    if cache_keys != expected_cache_keys:
        raise ValueError("hidden-cache keys do not exactly cover the 602 training docs × two windows")
    if not cache_contexts[0] or not cache_contexts[1]:
        raise ValueError("hidden-cache missing context ranges for a causal window")
    if any(len(ranges) != 1 for ranges in cache_contexts.values()):
        raise ValueError("hidden-cache context range is inconsistent within a window")
    if cache_contexts[0] != {(0, 256)} or cache_contexts[1] != {(0, 512)}:
        raise ValueError(f"hidden-cache causal context ranges differ from R1 windows: {cache_contexts}")
    if cache_manifest.get("source", {}).get("dataset_revision") != DATASET_REVISION:
        raise ValueError("hidden-cache dataset revision mismatch")
    if cache_manifest.get("source", {}).get("manifest_sha256") != source.get("manifest_sha256"):
        raise ValueError("hidden-cache source manifest identity differs from training manifest")
    if cache_manifest.get("teacher", {}).get("revision") != TEACHER_REVISION:
        raise ValueError("hidden-cache teacher revision mismatch")
    if cache_manifest.get("cache_file_sha256") != CACHE_SHA256:
        raise ValueError("hidden-cache manifest records unexpected cache file hash")
    if list(cache_manifest.get("shape", [])) != [2, 602, 256, 768] or len(cache_entries) != 1204:
        raise ValueError("hidden-cache shape or entry count mismatch")

    source_hash = sha256_file(source_path)
    derived = {
        "schema": "omega-backend-quality-train-manifest-v1",
        "derived_from": {
            "path": str(source_path),
            "file_sha256": source_hash,
            "source_manifest_sha256": source.get("manifest_sha256"),
            "split": "train",
        },
        "dataset": {
            "id": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "revision": DATASET_REVISION,
            "document_count": len(docs),
            "document_order_sha256": canonical_hash([
                [int(doc["document_index"]), doc["full_text_sha256"], doc["retained_513_token_sha256"]]
                for doc in docs
            ]),
        },
        "schedule": {
            "pair_count": 1000,
            "window_updates": 2000,
            "documents_consumed": 8000,
            "formula": "doc_position=((8*p+i) mod 602), p=0..999, i=0..7",
            "both_windows_share_pair_documents": True,
            "shuffle": False,
            "prefix_formula_verified_for_all_pairs": True,
        },
        "documents": docs,
        "pairs": expected_pairs,
        "hidden_cache": {
            "entry_count": len(cache_entries),
            "window_counts": {str(window): sum(1 for key in cache_keys if key[2] == window) for window in (0, 1)},
            "context_ranges": {str(window): [list(item) for item in sorted(cache_contexts[window])] for window in (0, 1)},
            "all_document_window_keys_match": True,
        },
    }
    derived["manifest_sha256"] = canonical_hash(derived)
    verification = {
        "source_manifest_path": str(source_path),
        "source_file_sha256": source_hash,
        "source_manifest_sha256": source.get("manifest_sha256"),
        "document_count": len(docs),
        "pair_count": len(expected_pairs),
        "documents_consumed": 8000,
        "pair_formula_verified": True,
        "cache_entries": len(cache_entries),
        "cache_keys_exact": True,
        "cache_context_ranges": {str(window): [list(item) for item in sorted(cache_contexts[window])] for window in (0, 1)},
        "derived_manifest_sha256": derived["manifest_sha256"],
    }
    return derived, verification


def _verify_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = CAMPAIGN_ROOT / "omega_expanded_frozen_validation" / "results" / "secondary_frozen_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("evaluation") != {
        "dataset": "WikiText-2",
        "split": "validation",
        "update": 2000,
        "all_eligible_documents": True,
        "document_count": 60,
    }:
        raise ValueError("existing 60-document validation report identity mismatch")
    existing_manifest = report.get("manifest", {})
    expected_manifest_hash = "857800cd725a5648dac54f1a16285e3437be7732f372f179c5b5981480cd3019"
    if existing_manifest.get("sha256") != expected_manifest_hash or existing_manifest.get("document_count") != 60:
        raise ValueError("existing validation manifest digest/count mismatch")

    expanded_dir = CAMPAIGN_ROOT / "omega_expanded_frozen_validation"
    sys.path.insert(0, str(expanded_dir))
    import run_omega_expanded_frozen_validation as expanded  # noqa: PLC0415

    documents, manifest, _tokenizer = expanded._load_real_validation()
    if len(documents) != 60 or manifest["manifest_sha256"] != expected_manifest_hash:
        raise ValueError("regenerated 60-document validation manifest differs from sealed artifact")
    if manifest["dataset"]["split"] != "validation" or manifest["selection_scope"] != "all eligible validation documents; no eight-document cap":
        raise ValueError("validation split/selection policy mismatch")
    if any(int(document["selected_token_count"]) != 513 or int(document["token_count"]) < 513 for document in documents):
        raise ValueError("validation document violates token eligibility")
    if any(
        int(left["document_index"]) >= int(right["document_index"])
        for left, right in zip(documents, documents[1:])
    ):
        raise ValueError("validation document order is not strictly increasing")

    validation_source_path = expanded_dir / "run_omega_expanded_frozen_validation.py"
    r1_source_path = R1_SCOPE / "run_scientific_scoping_a.py"
    expected_sources = report.get("source_hashes", {})
    actual_expanded_hash = sha256_file(validation_source_path)
    actual_r1_hash = sha256_file(r1_source_path)
    if expected_sources.get("runner") != actual_expanded_hash or expected_sources.get("r1_runner") != actual_r1_hash:
        raise ValueError("validation manifest source hashes differ from current source bytes")

    validation_copy = dict(manifest)
    verification = {
        "report_path": str(report_path),
        "report_file_sha256": sha256_file(report_path),
        "report_self_hash": report.get("artifact_self_hash"),
        "manifest_sha256": manifest["manifest_sha256"],
        "document_count": len(documents),
        "dataset": manifest["dataset"],
        "selection_scope": manifest["selection_scope"],
        "ordered_indices": [int(doc["document_index"]) for doc in documents],
        "unique_document_keys": True,
        "all_eligible": True,
        "train_key_exclusion": True,
        "causal_evaluation": "R1 evaluate_validation: reset initial state per document; window0 then detached state into window1",
        "expanded_validation_runner_sha256": actual_expanded_hash,
        "expanded_validation_runner_path": str(validation_source_path),
        "r1_evaluator_runner_sha256": actual_r1_hash,
        "r1_evaluator_runner_path": str(r1_source_path),
        "primary_gate_reclassification": False,
    }
    return validation_copy, verification


def _verify_hidden_cache_with_production_checker(manifest_path: Path, cache_path: Path) -> dict[str, Any]:
    r2_dir = P2R0
    sys.path.insert(0, str(r2_dir))
    import run_omega_native_runtime_r2_benchmark as r2  # noqa: PLC0415

    p0, _ce, _bridge = r2._load_r1_modules()
    result = p0.integration.verify_provenance(manifest_path, cache_file=cache_path)
    return {
        "status": result.get("status", "VERIFIED"),
        "manifest_path": str(manifest_path),
        "cache_path": str(cache_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cache_sha256": sha256_file(cache_path),
        "production_provenance_checker": str(P2R0 / "run_omega_native_runtime_r2_benchmark.py"),
    }


def _verify_preparation_evidence(synthetic_resume_path: Path, sensitivity_path: Path) -> dict[str, Any]:
    resume = json.loads(synthetic_resume_path.read_text(encoding="utf-8"))
    if resume.get("schema") != "omega-backend-quality-synthetic-resume-v1" or resume.get("status") != "PASS":
        raise ValueError("synthetic checkpoint/resume qualification failed")
    if resume.get("synthetic_only") is not True or resume.get("real_training") is not False or resume.get("real_dll_called") is not False:
        raise ValueError("checkpoint/resume evidence is not synthetic-only")
    expected_cases = {(backend, rounds) for backend in ("pytorch", "native") for rounds in (1, 4)}
    observed_cases = {(item.get("backend"), int(item.get("K", -1))) for item in resume.get("cases", [])}
    if observed_cases != expected_cases or any(not item.get("continuous_vs_resume_exact") or not item.get("wrong_backend_checkpoint_rejected") for item in resume["cases"]):
        raise ValueError("synthetic resume cases missing/failing")

    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    if sensitivity.get("schema") != "omega-backend-quality-synthetic-tost-sensitivity-v1":
        raise ValueError("synthetic sensitivity schema mismatch")
    if sensitivity.get("synthetic_only") is not True or sensitivity.get("real_candidate_results_read") is not False or sensitivity.get("real_training") is not False:
        raise ValueError("statistical sensitivity must use synthetic data only")
    if len(sensitivity.get("scenarios", [])) < 8 or sensitivity.get("paired_seed_count") != 5:
        raise ValueError("statistical sensitivity grid is incomplete")
    return {
        "synthetic_resume": {"path": str(synthetic_resume_path), "sha256": sha256_file(synthetic_resume_path), "status": resume["status"], "cases": resume["cases"]},
        "statistical_sensitivity": {"path": str(sensitivity_path), "sha256": sha256_file(sensitivity_path), "synthetic_only": True, "scenario_count": len(sensitivity["scenarios"]), "paired_seed_count": 5},
    }


def _verify_prior_preflight_evidence() -> dict[str, Any]:
    if not PREVIOUS_QUALIFICATION_MANIFEST.is_file():
        raise FileNotFoundError(f"prior preparation manifest missing: {PREVIOUS_QUALIFICATION_MANIFEST}")
    basis_manifest = json.loads(PREVIOUS_QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    basis_signature = basis_manifest.get("manifest_sha256")
    basis_unsigned = dict(basis_manifest)
    basis_unsigned.pop("manifest_sha256", None)
    if not basis_signature or basis_signature != canonical_hash(basis_unsigned):
        raise ValueError("prior preparation manifest self-hash mismatch")

    reports = {
        "original_causal_smoke": (ORIGINAL_SMOKE_REPORT, "FAIL"),
        "common_origin_diagnosis": (COMMON_ORIGIN_REPORT, "LOCAL_PASS"),
        "corrected_loss_stable_8p32": (STABLE_RECHARACTERIZATION_REPORT, "COMPLETE"),
    }
    verified: dict[str, Any] = {}
    for label, (path, expected_status) in reports.items():
        if not path.is_file():
            raise FileNotFoundError(f"required previous preflight evidence missing: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        self_hash_key = "report_self_sha256"
        report_self_hash = report.get(self_hash_key)
        unsigned = dict(report)
        unsigned.pop(self_hash_key, None)
        if not report_self_hash or report_self_hash != canonical_hash(unsigned):
            raise ValueError(f"preflight evidence self-hash mismatch: {path}")
        if report.get("status") != expected_status or report.get("manifest_sha256") != basis_signature:
            raise ValueError(f"preflight evidence status/basis-manifest mismatch: {path}")
        verified[label] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "report_self_sha256": report_self_hash,
            "status": report["status"],
            "basis_manifest_sha256": basis_signature,
        }

    smoke = json.loads(ORIGINAL_SMOKE_REPORT.read_text(encoding="utf-8"))
    if smoke.get("technical_optimizer_updates") != 8 or smoke.get("historical_helper_called") is not False:
        raise ValueError("original smoke evidence identity/route count changed")
    origin = json.loads(COMMON_ORIGIN_REPORT.read_text(encoding="utf-8"))
    if origin.get("part2", {}).get("common_origin_copy_exact") is not True or origin.get("part2", {}).get("gates", {}).get("K4_common_origin_window1", {}).get("pass") is not True:
        raise ValueError("common-origin local gates are no longer passing")
    stable = json.loads(STABLE_RECHARACTERIZATION_REPORT.read_text(encoding="utf-8"))
    if len(stable.get("route_reports", [])) != 4 or any(row.get("route_report", {}).get("status") != "COMPLETE" for row in stable.get("route_reports", [])):
        raise ValueError("corrected-loss 8+32 route evidence is incomplete")
    return {
        "basis_qualification_manifest": {
            "path": str(PREVIOUS_QUALIFICATION_MANIFEST),
            "file_sha256": sha256_file(PREVIOUS_QUALIFICATION_MANIFEST),
            "manifest_sha256": basis_signature,
        },
        "reports": verified,
        "original_smoke_fail_preserved": True,
        "common_origin_is_local_diagnosis_only": True,
        "stable_8p32_is_performance_evidence_only": True,
        "strict_trajectory_proximity_reclassified": False,
    }


def _latest_passing_synthetic_resume() -> Path:
    root = HERE / "results" / "preparation" / "synthetic_resume"
    candidates = sorted(root.glob("run_*/synthetic_resume_check.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("status") == "PASS" and report.get("synthetic_only") is True and report.get("real_training") is False and report.get("real_dll_called") is False:
            return path
    raise FileNotFoundError("no passing synthetic resume report found; run --synthetic-resume-check first")


def prepare(output_dir: Path, synthetic_resume_path: Path, sensitivity_path: Path) -> dict[str, Any]:
    for output_file in (output_dir / "train_manifest_1000_pairs.json", output_dir / "validation_manifest_60.json", output_dir / "qualification_manifest.json"):
        if output_file.exists():
            raise FileExistsError(f"prepared quality manifests are immutable: {output_file}")
    source_train_path = CE / "results" / "runs" / "CE-K4_seed_20260913" / "train_manifest.json"
    hidden_manifest_path = HIDDEN_CACHE / "results" / "cache_manifest.json"
    hidden_cache_path = Path(r"C:\omega_cache\teacher_hidden.fp32")
    source_train_hash = sha256_file(source_train_path)
    if source_train_hash != EXPECTED_HASHES["source_train_manifest"]:
        raise ValueError(f"source train manifest file hash changed: {source_train_hash}")
    hidden_manifest_hash = sha256_file(hidden_manifest_path)
    actual_cache_hash = sha256_file(hidden_cache_path)
    if actual_cache_hash != CACHE_SHA256:
        raise ValueError(f"hidden cache bytes changed: {actual_cache_hash}")
    hidden_manifest = json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
    if hidden_manifest_hash != EXPECTED_HASHES["cache_manifest"]:
        raise ValueError(f"hidden cache manifest hash changed: {hidden_manifest_hash}")
    derived_train, train_verification = _verify_train_manifest(source_train_path, hidden_manifest)
    hidden_cache_verification = _verify_hidden_cache_with_production_checker(hidden_manifest_path, hidden_cache_path)
    validation_manifest, validation_verification = _verify_validation()
    preparation_evidence = _verify_preparation_evidence(synthetic_resume_path, sensitivity_path)

    model_files = {
        "r1_freeze_doc": R1_FREEZE / "SCIENTIFIC_PILOT_FREEZE.md",
        "r1_freeze_manifest": R1_FREEZE / "freeze_manifest.json",
        "r1_model": LAB_ROOT / "t1_trainability" / "model.py",
        "r1_fast_candidate": CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation" / "omega_fast_candidate.py",
        "r1_technical_preflight_current": LAB_ROOT / "scripts" / "run_omega_core_lm_0_r1_training_technical_preflight.py",
        "r1_scope_runner": R1_SCOPE / "run_scientific_scoping_a.py",
        "expanded_validation_runner": EXPANDED_VALIDATION / "run_omega_expanded_frozen_validation.py",
        "nominal_runner": CAMPAIGN_ROOT / "omega_core_lm_0_gpu_environment_preparation" / "omega_nominal_microbatch_runner.py",
        "r2_runner": P2R0 / "run_omega_native_runtime_r2_benchmark.py",
        "r2_production_bridge": P2R0 / "omega_recurrent_production_bridge.py",
        "r2_p0_helper": CAMPAIGN_ROOT / "omega_native_runtime_p0" / "run_omega_native_runtime_p0.py",
        "ce_model_factory": CE / "run_omega_ce_only_baseline.py",
        "teacher_logit_cache_base": CAMPAIGN_ROOT / "omega_teacher_logit_cache" / "run_omega_teacher_logit_cache.py",
        "hidden_cache_production_integration": CAMPAIGN_ROOT / "omega_hidden_cache_production_integration" / "run_omega_hidden_cache_production_integration.py",
        "teacher_hidden_cache_probe": HIDDEN_CACHE / "run_omega_teacher_hidden_cache_probe.py",
    }
    source_identities = {key: _file_identity(key, path, EXPECTED_HASHES[key]) for key, path in model_files.items()}
    candidate_identity = _file_identity("selected_native_dll", BE376_DLL, EXPECTED_DLL_SHA256)
    protocol_identity = _file_identity("qualification_protocol", HERE / "BACKEND_QUALITY_QUALIFICATION_PROTOCOL.md")
    impact_audit_identity = _file_identity("loss_consumer_impact_audit", HERE / "LOSS_CONSUMER_IMPACT_AUDIT.md")
    kernel_scope_note_identity = _file_identity("kernel_closeout_scope_note", HERE / "KERNEL_CLOSEOUT_SCOPE_NOTE.md")
    runner_identity = _file_identity("isolated_quality_runner", HERE / "run_backend_quality_qualification.py")
    sensitivity_identity = _file_identity("synthetic_sensitivity_runner", HERE / "synthetic_statistical_sensitivity.py")
    loss_implementation_identity = _file_identity("canonical_r1_masked_token_mean_loss", LOSS_IMPLEMENTATION)
    loss_test_identity = _file_identity("canonical_loss_contract_tests", LOSS_TESTS)
    resume_test_identity = _file_identity("real_runner_resume_contract_test", RESUME_TEST_RUNNER)
    causal_smoke_identity = _file_identity("causal_smoke_harness", CAUSAL_SMOKE_RUNNER)
    common_origin_identity = _file_identity("common_origin_diagnostic_harness", COMMON_ORIGIN_RUNNER)
    stable_recharacterization_identity = _file_identity("corrected_loss_stable_8p32_harness", STABLE_RECHARACTERIZATION_RUNNER)
    prior_preflight_evidence = _verify_prior_preflight_evidence()

    cache_identity = {
        "manifest": _file_identity("hidden_cache_manifest", hidden_manifest_path, EXPECTED_HASHES["cache_manifest"]),
        "cache_file": _file_identity("hidden_cache_file", hidden_cache_path, CACHE_SHA256),
        "dataset_revision": DATASET_REVISION,
        "teacher_revision": TEACHER_REVISION,
        "document_count": 602,
        "windows_per_document": 2,
        "context_policy_verified": train_verification["cache_context_ranges"],
    }
    qualification = {
        "schema": "omega-backend-quality-qualification-manifest-v1",
        "status": "PREPARED_NO_SCIENTIFIC_TRAINING",
        "real_training_authorized": False,
        "unit": "OMEGA-BACKEND-QUALITY-QUALIFICATION",
        "phase": "DESIGN_PREPARATION",
        "block_a": {
            "seeds": [20260913, 20260914],
            "rounds": [1, 4],
            "backends": ["pytorch", "native"],
            "independent_run_count": 8,
            "updates_per_run": 2000,
            "total_updates": 16000,
            "one_common_init_per_seed_K_shared_by_backends": True,
            "fresh_empty_adamw_per_arm": True,
            "optimizer_warmup_updates": 0,
            "no_benchmark_moments_reused": True,
        },
        "block_b": {"seeds": [20260915, 20260916, 20260917], "rounds": [1, 4], "backends": ["pytorch", "native"], "updates_per_run": 2000, "total_updates": 24000, "requires_separate_authorization": True},
        "technical_resume_test_contract": {
            "test_id": "REAL_RUNNER_CHECKPOINT_RESUME_V1",
            "seed": 20260913,
            "K_values": [1, 4],
            "backends": ["pytorch", "native"],
            "route_count": 4,
            "continuous_updates_per_route": 4,
            "segmented_prefix_updates_per_route": 3,
            "checkpoint_after_completed_update": 3,
            "fresh_process_resume_updates_per_route": 1,
            "checkpoint_interval": 3,
            "total_technical_optimizer_updates": 32,
            "actual_quality_runner_and_actual_selected_dll": True,
            "canonical_loss_contract": LOSS_CONTRACT_ID,
            "loss_callable": "r1_masked_token_mean_loss",
            "continuous_and_resumed_compared_within_same_backend_K": True,
            "exact_compare": ["model_parameters_and_buffers", "adamw_moments_and_step", "rng", "data_cursor", "recurrent_state", "per_update_loss_clip_and_data_ledger"],
            "exclude_from_exact_compare": ["timestamps", "elapsed_runtime", "memory_samples", "execution_segment_number"],
            "validation_or_test_loaded": False,
            "warmup_optimizer_updates": 0,
            "checkpoint_must_restore_detached_causal_state": True,
        },
        "architectural_reference": {
            "source_commit": "269a4d79b5a1e6df8c230962e2b8c9e237095f18",
            "model_source_sha256": EXPECTED_HASHES["r1_model"],
            "architecture": "R1 shared core; K1/K4 only; no factorization/untied variant",
            "original_freeze": "declarative R1 contract remains intact; this qualification does not claim B2x4 execution conformity",
        },
        "qualified_backend_config": {
            "dll": candidate_identity,
            "native_source_commit": "18e7225b819af61304189eff8537d602f729cf2f",
            "native_cpp_blob_sha256": R1_NATIVE_SOURCE_BLOB_SHA256,
            "native_coverage_commit": "d993b3c1f1e19e4bffa32d7c4be35687257b28c6",
            "physical_batch": 8,
            "effective_batch": 8,
            "gradient_accumulations": 1,
            "native_workers": 4,
            "pytorch_intraop_threads": 4,
            "pytorch_interop_threads": 1,
            "instrumentation": 0,
            "diagnostic_statistics": "null/disabled",
            "profile_compile_and_runtime": False,
            "dtype": "float32",
            "objective": {
                "contract_id": LOSS_CONTRACT_ID,
                "formula": "0.5*CE_valid_token_mean + 0.5*KL_teacher||student_valid_token_mean",
                "temperature": 2.0,
                "teacher_distribution": "softmax(teacher_logits/tau)",
                "student_distribution": "log_softmax(student_logits/tau)",
                "kl_direction": "teacher||student",
                "kl_reduction": "vocab_sum_then_valid_token_mean",
                "ce_weight": 0.5,
                "kl_weight": 0.5,
                "valid_tokens_per_update": 2048,
                "valid_token_denominator": "valid_mask.sum(); require > 0",
                "interface": {
                    "student_logits": "[B,L,V]",
                    "teacher_logits": "[B,L,V]",
                    "targets": "[B,L]",
                    "valid_mask": "[B,L] bool",
                },
            },
            "optimizer": {"type": "AdamW", "lr": 0.0003, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "gradient_clip": {"norm": 1.0, "once_after_backward_before_adamw": True},
            "bptt_tokens": 256,
            "causal_state": {"window0_reset": True, "window1_source": "detached immediately preceding window0"},
        },
        "source_identities": source_identities,
        "loss_contract": {
            "id": LOSS_CONTRACT_ID,
            "teacher_distribution": "softmax(teacher_logits/tau)",
            "student_distribution": "log_softmax(student_logits/tau)",
            "kl_direction": "teacher||student",
            "kl_reduction": "vocab_sum_then_valid_token_mean",
            "temperature": 2.0,
            "temperature_squared_applied_once": True,
            "ce_weight": 0.5,
            "kl_weight": 0.5,
            "valid_tokens_per_update": 2048,
            "denominator": "valid_mask.sum(); require > 0",
            "interface": {
                "student_logits": "[B,L,V]",
                "teacher_logits": "[B,L,V]",
                "targets": "[B,L]",
                "valid_mask": "[B,L] bool",
            },
            "implementation": {
                **loss_implementation_identity,
                "callable": "r1_masked_token_mean_loss",
            },
            "r1_reference": {
                **source_identities["r1_technical_preflight_current"],
                "callable": "distillation_loss",
            },
            "historical_nonconforming_helper": {
                **source_identities["teacher_logit_cache_base"],
                "callable": "distillation_loss",
                "classification": "historical_nonconforming_for_R1_MASKED_TOKEN_MEAN_V1; preserved unchanged",
                "reason": "KL batchmean on [B,L,V] scales KL by sequence length relative to valid-token mean",
            },
            "contract_tests": loss_test_identity,
        },
        "protocol_identity": protocol_identity,
        "impact_audit_identity": impact_audit_identity,
        "kernel_scope_note_identity": kernel_scope_note_identity,
        "input_preparation_runner": _file_identity("input_preparation_runner", Path(__file__)),
        "runner_identity": runner_identity,
        "synthetic_sensitivity_runner": sensitivity_identity,
        "loss_contract_test_identity": loss_test_identity,
        "resume_test_harness_identity": resume_test_identity,
        "preflight_tool_identities": {
            "causal_smoke": causal_smoke_identity,
            "common_origin_diagnostic": common_origin_identity,
            "corrected_loss_stable_8p32": stable_recharacterization_identity,
        },
        "prior_preflight_evidence": prior_preflight_evidence,
        "preparation_evidence": preparation_evidence,
        "hidden_cache": cache_identity,
        "hidden_cache_production_verification": hidden_cache_verification,
        "training_manifest": {
            "path": str(output_dir / "train_manifest_1000_pairs.json"),
            "manifest_sha256": derived_train["manifest_sha256"],
            "source_path": str(source_train_path),
            "source_sha256": source_train_hash,
            "source_manifest_sha256": json.loads(source_train_path.read_text(encoding="utf-8"))["manifest_sha256"],
            "verification": train_verification,
        },
        "validation_manifest": {
            "path": str(output_dir / "validation_manifest_60.json"),
            "manifest_sha256": validation_manifest["manifest_sha256"],
            "verification": validation_verification,
            "qualification_role": "primary_validation_for_backend_quality_only",
            "source_report_role": "SECONDARY_FROZEN_VALIDATION",
            "source_report_primary_gate_reclassification": False,
            "evaluator": "pure-PyTorch R1 F reference; same implementation for native and PyTorch checkpoint scoring; no DLL call",
            "test_split_loaded": False,
        },
        "metrics": {
            "delta_K": "NLL_native,K - NLL_pytorch,K",
            "eta_depth": "(NLL_native,K1-NLL_native,K4)-(NLL_pytorch,K1-NLL_pytorch,K4)",
            "primary_endpoint": "update_2000_only",
            "validation_points": [0, 500, 1000, 1500, 2000],
            "test_set_used": False,
            "formal_analysis_after_all_5_paired_seeds_only": True,
            "delta_margin_nats_per_token": 0.02,
            "eta_margin_nats_per_token": 0.01,
            "analysis": "paired TOST with 90% CI; all seed-level values reported",
        },
        "preparation_only": True,
        "scientific_training_started": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_manifest_1000_pairs.json", derived_train)
    write_json(output_dir / "validation_manifest_60.json", validation_manifest)
    qualification["prepared_artifacts"] = {
        "train_manifest_file_sha256": sha256_file(output_dir / "train_manifest_1000_pairs.json"),
        "validation_manifest_file_sha256": sha256_file(output_dir / "validation_manifest_60.json"),
    }
    qualification["manifest_sha256"] = canonical_hash(qualification)
    write_json(output_dir / "qualification_manifest.json", qualification)
    return qualification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs", action="store_true", help="verify frozen input identity and write derived manifests only")
    parser.add_argument("--output-dir", type=Path, default=HERE / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2")
    parser.add_argument("--synthetic-resume-report", type=Path)
    parser.add_argument("--sensitivity-report", type=Path, default=DEFAULT_SYNTHETIC_SENSITIVITY)
    args = parser.parse_args()
    if not args.prepare_inputs:
        parser.error("no operation selected; use --prepare-inputs (never starts training)")
    synthetic_resume_report = args.synthetic_resume_report.resolve() if args.synthetic_resume_report is not None else _latest_passing_synthetic_resume()
    report = prepare(args.output_dir.resolve(), synthetic_resume_report, args.sensitivity_report.resolve())
    print(json.dumps({
        "status": report["status"],
        "manifest_sha256": report["manifest_sha256"],
        "training_manifest": report["prepared_artifacts"]["train_manifest_file_sha256"],
        "validation_manifest": report["prepared_artifacts"]["validation_manifest_file_sha256"],
        "training_pair_verification": report["training_manifest"]["verification"],
        "validation_verification": report["validation_manifest"]["verification"],
        "scientific_training_started": report["scientific_training_started"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
