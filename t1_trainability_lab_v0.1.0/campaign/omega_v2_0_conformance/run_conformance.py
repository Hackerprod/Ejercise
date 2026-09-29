"""Build immutable V2-0 conformance artifacts and run the nine acceptance tests."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = UNIT_ROOT / "results" / "omega_v2_0_conformance"
SNAPSHOT_COMMIT = "38061477d4c2b0c5c20d75d902b21dd1ef0a2611"
CONVERSACION_BLOB = "7027e2ac9d1ba89db08dda73c81e244f3b9b19db"
LN_BLOB = "fc750a2ae9fb7d3933c54fb91f09ce5568d035ec"
AUDIT_SHA256 = "a7e82cddfcacf17f163e7d046853046595f08e532086fb3c2a577432807933dc"
CONTRACT_RANGES = ["2390-2485", "2585-2630", "2688-2860", "2945-3075", "3078-3240"]

if str(UNIT_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIT_ROOT))

from omega_v2.core import configure_reference_execution  # noqa: E402
from omega_v2.ledger import DIMENSIONS, SLOT_COUNTS, build_flop_ledger, build_parameter_ledger  # noqa: E402
from omega_v2.reference import toy_acceptance_report  # noqa: E402
from omega_v2.variants import run_bitwise_acceptance  # noqa: E402


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=UNIT_ROOT.parents[2], check=True, capture_output=True, text=True).stdout.strip()


def _candidate_block_rows(parameter_ledger: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in parameter_ledger["rows"]:
        is_shared = row["sharing_mode"] == "shared"
        result.append(
            {
                "candidate_id": row["candidate_id"],
                "d": row["d"],
                "m": row["supported_m"],
                "K_train": "NOT_APPLICABLE_V2_0",
                "K_infer_policy": "runtime_integer_K_ge_1" if is_shared else "untied_control_fixed_K4",
                "recurrent_block_formula": "Q/K/V/O=4d^2; SwiGLU hidden=4d=12d^2; total=16d^2",
                "P_core_unique": row["P_core_unique"],
                "P_shell": row["P_shell"],
                "P_retriever": row["P_retriever"],
                "P_memory_learned": row["P_memory_learned"],
                "B_memory_static": row["B_memory_static"],
                "B_index": row["B_index"],
                "q4_logical_total_bytes": row["q4_logical_total_bytes"],
                "sharing": "R4_SHARED" if is_shared else "U4_UNTIED_CONTROL",
                "state_dict_sha256": row["state_dict_sha256"],
                "values_obtained_by_introspection": True,
            }
        )
    return result


def _contract_block(parameter_ledger: dict[str, Any], implementation_commit: str) -> dict[str, Any]:
    shared_rows = [row for row in parameter_ledger["rows"] if row["sharing_mode"] == "shared"]
    targets = []
    for d in DIMENSIONS:
        row = next(item for item in shared_rows if item["d"] == d)
        targets.append(
            {
                "id": f"V2-{d}",
                "d": d,
                "m": list(SLOT_COUNTS),
                "K_train": "NOT_APPLICABLE_V2_0",
                "K_infer_policy": "runtime_integer_K_ge_1",
                "recurrent_block_formula": "4d^2 + 12d^2 = 16d^2",
                "P_core_unique": 16 * d * d,
                "P_shell": 0,
                "P_retriever": 0,
                "P_memory_learned": 0,
                "B_memory_static": 0,
                "B_index": 0,
                "quantized_core_logical_bytes": row["q4_logical_total_bytes"],
                "sharing": "shared",
                "data_recipe": "synthetic_deterministic_only",
            }
        )
    return {
        "OMEGA_CONFORMANCE_BLOCK": {
            "authority": {
                "repository": "Hackerprod/Ejercise",
                "snapshot_commit": SNAPSHOT_COMMIT,
                "conversacion_md_blob": CONVERSACION_BLOB,
                "conversacion_ln_blob": LN_BLOB,
                "audit_report_sha256": AUDIT_SHA256,
                "contract_line_ranges": CONTRACT_RANGES,
            },
            "phase": "ENGINEERING",
            "implementation_commit": implementation_commit,
            "contract_target": {"candidates": targets},
            "actual_candidate": {
                "status": "PENDING_ACCEPTANCE",
                "values_obtained_by_introspection": True,
                "candidates": _candidate_block_rows(parameter_ledger),
            },
            "deviations": [],
            "authorized_deviation_ids": [],
            "claim_scope": {
                "allowed": [
                    "implementation_conformance",
                    "exact_parameter_ledger",
                    "K_parameter_independence",
                    "R4_U4_compute_parity_at_init",
                    "readiness_for_V2_1_if_all_acceptance_pass",
                ],
                "forbidden": [
                    "cache_residency_claim",
                    "CPU_speedup_claim",
                    "language_quality_claim",
                    "sharing_quality_claim",
                    "K_scaling_claim",
                    "T3_release",
                ],
            },
            "status": "CONFORMANCE_HOLD",
        }
    }


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rows: list[dict[str, str]] = []

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self.rows.append({"test": test.id(), "status": "PASS"})

    def addFailure(self, test: unittest.case.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self.rows.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test: unittest.case.TestCase, err: Any) -> None:
        super().addError(test, err)
        self.rows.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self.rows.append({"test": test.id(), "status": "SKIP", "detail": reason})


def _run_test_suite() -> tuple[RecordingResult, str]:
    tests_dir = UNIT_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(tests_dir),
        pattern="test_v2_*.py",
        top_level_dir=str(tests_dir),
    )
    text = io.StringIO()
    runner = unittest.TextTestRunner(stream=text, verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    return result, text.getvalue()


def _all_pretests_pass(parameter_ledger: dict[str, Any], flop_ledger: dict[str, Any], bitwise: dict[str, Any], toy: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for row in parameter_ledger["rows"]:
        if row["P_core_unique"] != (16 * row["d"] ** 2 if row["sharing_mode"] == "shared" else 64 * row["d"] ** 2):
            failures.append(f"parameter-count:{row['candidate_id']}")
    if not all(row.get("exact_compute_match") for row in flop_ledger["rows"] if row.get("variant_pair") == "R4_vs_U4"):
        failures.append("r4-u4-flop-parity")
    if bitwise.get("status") != "PASS":
        failures.append("r4-u4-bitwise")
    if toy.get("status") != "PASS":
        failures.append("toy-math-or-gradcheck")
    return not failures, failures


def _write_report(status: str, implementation_commit: str, parameter_ledger: dict[str, Any], flop_ledger: dict[str, Any], bitwise: dict[str, Any], toy: dict[str, Any], test_rows: list[dict[str, str]], artifact_paths: list[Path], failures: list[str]) -> str:
    test_failures = [row for row in test_rows if row["status"] != "PASS"]
    hashes = {path.name: _sha256_file(path) for path in artifact_paths if path.is_file()}
    lines = [
        "# OMEGA-V2-0 Conformance Report",
        "",
        f"- terminal_status: `{status}`",
        f"- implementation_commit: `{implementation_commit}`",
        f"- snapshot_commit: `{SNAPSHOT_COMMIT}`",
        f"- Conversacion.md blob: `{CONVERSACION_BLOB}`",
        f"- audit report SHA256: `{AUDIT_SHA256}`",
        "",
        "## Actual parameter ledger (introspection)",
        "",
        "| candidate | d | sharing | unique core params | FP32 bytes | Q4 logical bytes | state_dict SHA256 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in parameter_ledger["rows"]:
        lines.append(
            f"| {row['candidate_id']} | {row['d']} | {row['sharing_mode']} | {row['P_core_unique']} | {row['fp32_core_bytes']} | {row['q4_logical_total_bytes']} | `{row['state_dict_sha256']}` |"
        )
    lines.extend(["", "## FLOP ledger", "", "Forward matmul FLOPs are counted from live matrix shapes with 1 MAC = 2 FLOPs. R4 and U4 comparisons use K=4.", ""])
    for row in flop_ledger["rows"]:
        if row.get("variant_pair") == "R4_vs_U4":
            lines.append(f"- d={row['d']} m={row['m']} K=4: R4={row['R4_forward_flops']}; U4={row['U4_forward_flops']}; exact_match={row['exact_compute_match']}")
    lines.extend(["", "## Acceptance tests", ""])
    for row in test_rows:
        lines.append(f"- {row['status']}: `{row['test']}`" + (f" — {row.get('detail', '')}" if row.get("detail") else ""))
    lines.extend(["", "## Bitwise and toy reports", "", f"- bitwise_status: `{bitwise.get('status')}`; cases: {bitwise.get('row_count')}", f"- toy_status: `{toy.get('status')}`"])
    if failures:
        lines.extend(["", "## HOLD causes", *[f"- `{failure}`" for failure in failures]])
    if test_failures:
        lines.extend(["", "## Failed test details", *[f"- `{row['test']}`: {row.get('detail', '')}" for row in test_failures]])
    lines.extend(["", "## Artifact SHA256", *[f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())], "", "Scope: implementation conformance only. No language, quality, cache-residency, CPU-speed, K-scaling, or T3 claim.", ""])
    path = RESULTS_ROOT / "OMEGA_V2_0_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return _sha256_file(path)


def main() -> int:
    if RESULTS_ROOT.exists():
        raise FileExistsError(f"V2-0 results are immutable: {RESULTS_ROOT}")
    branch = _git("branch", "--show-current")
    implementation_commit = _git("rev-parse", "HEAD")
    parent = _git("show", "-s", "--format=%P", "HEAD").split()
    unit_status = _git("status", "--porcelain", "--", "t1_trainability_lab_v0.1.0/campaign/omega_v2_0_conformance")
    conversacion_blob = _git("rev-parse", "HEAD:Conversacion.md")
    if branch != "main" or len(parent) != 1 or parent[0] != SNAPSHOT_COMMIT or conversacion_blob != CONVERSACION_BLOB or unit_status:
        raise RuntimeError(
            "CONFORMANCE_HOLD: require a clean local implementation commit directly atop the authorized snapshot; "
            f"branch={branch}, commit={implementation_commit}, parent={parent}, conversacion_blob={conversacion_blob}, unit_status={unit_status!r}"
        )

    configure_reference_execution()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=False)
    parameter_ledger = build_parameter_ledger()
    flop_ledger = build_flop_ledger()
    pretest_errors: list[str] = []
    try:
        bitwise = run_bitwise_acceptance()
    except Exception as error:
        bitwise = {"schema": "omega-v2-bitwise-report-v1", "status": "ERROR", "error": f"{type(error).__name__}: {error}", "checks": []}
        pretest_errors.append(f"bitwise-exception:{type(error).__name__}:{error}")
    try:
        toy = toy_acceptance_report()
    except Exception as error:
        toy = {"status": "ERROR", "error": f"{type(error).__name__}: {error}"}
        pretest_errors.append(f"toy-exception:{type(error).__name__}:{error}")

    block_document = _contract_block(parameter_ledger, implementation_commit)
    block_path = RESULTS_ROOT / "OMEGA_CONFORMANCE_BLOCK.yaml"
    ledger_path = RESULTS_ROOT / "omega_v2_ledger.json"
    flop_path = RESULTS_ROOT / "omega_v2_flop_ledger.json"
    bitwise_path = RESULTS_ROOT / "omega_v2_bitwise_report.json"
    toy_path = RESULTS_ROOT / "omega_v2_toy_report.json"
    _write_json(ledger_path, parameter_ledger)
    _write_json(flop_path, flop_ledger)
    _write_json(bitwise_path, bitwise)
    _write_json(toy_path, toy)
    _write_json(block_path, block_document)

    test_result, test_output = _run_test_suite()
    test_rows = sorted(test_result.rows, key=lambda row: row["test"])
    prechecks_pass, precheck_failures = _all_pretests_pass(parameter_ledger, flop_ledger, bitwise, toy)
    test_failures = [row for row in test_rows if row["status"] != "PASS"]
    failures = pretest_errors + precheck_failures + [f"test:{row['test']}:{row['status']}" for row in test_failures]
    passed = prechecks_pass and not failures and test_result.wasSuccessful()
    terminal_status = "OMEGA_V2_0_CONFORMANT_PASS" if passed else "OMEGA_V2_0_CONFORMANCE_HOLD"
    block = block_document["OMEGA_CONFORMANCE_BLOCK"]
    block["actual_candidate"]["status"] = "CONFORMANT" if passed else "CONFORMANCE_HOLD"
    block["status"] = "CONFORMANT" if passed else "CONFORMANCE_HOLD"
    block["acceptance_tests"] = {row["test"]: row["status"] for row in test_rows}
    block["hold_causes"] = failures
    _write_json(block_path, block_document)

    artifact_paths = [block_path, ledger_path, flop_path, bitwise_path, toy_path]
    report_sha = _write_report(
        terminal_status,
        implementation_commit,
        parameter_ledger,
        flop_ledger,
        bitwise,
        toy,
        test_rows,
        artifact_paths,
        failures,
    )
    report_path = RESULTS_ROOT / "OMEGA_V2_0_REPORT.md"
    (RESULTS_ROOT / "OMEGA_V2_0_REPORT.md.sha256").write_text(report_sha + "\n", encoding="ascii", newline="\n")
    print(
        json.dumps(
            {
                "status": terminal_status,
                "implementation_commit": implementation_commit,
                "parent": parent[0],
                "results_dir": str(RESULTS_ROOT),
                "report_sha256": report_sha,
                "test_count": test_result.testsRun,
                "tests_passed": len([row for row in test_rows if row["status"] == "PASS"]),
                "failures": failures,
                "unittest_output": test_output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
