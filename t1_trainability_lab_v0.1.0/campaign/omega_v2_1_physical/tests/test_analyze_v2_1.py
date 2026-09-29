from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

UNIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIT_ROOT))

from scripts import analyze_v2_1 as analysis  # noqa: E402


class AnalyzerContractTests(unittest.TestCase):
    def test_required_test_names_match_md304_section_46(self) -> None:
        md304 = UNIT_ROOT.parents[2] / "MD" / "304.md"
        content = md304.read_text(encoding="utf-8")
        section = content.split("46. Tests obligatorios V2-1 — por nombre", 1)[1].split("47. Correctness thresholds", 1)[0]
        required = [line.strip() for line in section.splitlines() if line.strip().startswith("test_v2_1_")]
        self.assertEqual(required, analysis.TEST_NAMES)
        self.assertEqual(len(required), 37)

    def test_native_csv_parser_returns_rows_and_exact_header(self) -> None:
        header = "d,m,K,variant,is_warmup,block_id,sample_id,round_index,qpc_ticks,qpc_seconds,effective_macs,effective_flops,timed_heap_allocation_count,valid,invalid_reason\n"
        sample = "512,4,1,A,false,0,0,0,100,0.0001,10,20,0,true,\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text(header + sample, encoding="utf-8", newline="\n")
            rows, fields = analysis.parse_raw(path)
        self.assertEqual(fields, header.strip().split(","))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["d"], 512)
        self.assertTrue(rows[0]["valid"])

    def test_gate_metrics_use_median_derived_marginal_costs(self) -> None:
        cells = {}
        for m in (1, 4, 8, 16):
            for K in (1, 4, 8):
                for variant in ("A", "B", "C"):
                    cells[(512, m, K, variant)] = {
                        "median_seconds": 10.0,
                        "effective_MAC_per_s": 100.0,
                        "R_MAD": 0.0,
                    }
        cells[(512, 4, 1, "A")]["median_seconds"] = 10.0
        cells[(512, 4, 8, "A")]["median_seconds"] = 31.0
        cells[(512, 4, 1, "B")]["median_seconds"] = 12.0
        cells[(512, 4, 8, "B")]["median_seconds"] = 61.0
        cells[(512, 4, 1, "C")]["median_seconds"] = 10.5
        cells[(512, 4, 8, "C")]["median_seconds"] = 57.0
        cells[(512, 4, 8, "A")]["effective_MAC_per_s"] = 200.0
        cells[(512, 8, 8, "A")]["effective_MAC_per_s"] = 210.0
        cells[(512, 16, 8, "A")]["effective_MAC_per_s"] = 180.0

        gates = analysis.compute_gate_metrics(cells)
        self.assertAlmostEqual(gates["c_A_K8"], 3.0)
        self.assertAlmostEqual(gates["c_B_K8"], 7.0)
        self.assertAlmostEqual(gates["c_C_K8"], 6.642857142857143)
        self.assertAlmostEqual(gates["rho_resident"], 3.0 / 7.0)
        self.assertAlmostEqual(gates["G_matrix"], 2.1)
        self.assertTrue(gates["rho_minimum_pass"])
        self.assertTrue(gates["causal_minimum_pass"])
        self.assertEqual(gates["noisy_primary_stability_cells"], 0)

    def test_summary_reports_each_measurement_block(self) -> None:
        rows = []
        for sample_id, seconds in ((0, 1.0), (21, 1.2)):
            block_id = sample_id // 21
            rows.append({
                "d": 512, "m": 4, "K": 1, "variant": "A", "is_warmup": False,
                "block_id": block_id, "sample_id": sample_id, "round_index": 0,
                "valid": True, "invalid_reason": "", "qpc_seconds": seconds,
                "effective_macs": 100, "effective_flops": 200,
            })
        summaries, cells, discarded = analysis.summarize_samples(rows)
        cell = cells[(512, 4, 1, "A")]
        self.assertEqual(cell["n_valid"], 2)
        self.assertEqual(len(cell["blocks"]), 5)
        self.assertEqual(cell["blocks"][0]["median_seconds"], 1.0)
        self.assertEqual(cell["blocks"][1]["median_seconds"], 1.2)
        self.assertEqual(discarded, [])
        self.assertEqual(len(summaries), 72)

    def test_bootstrap_gate_diagnostics_are_seeded_and_present(self) -> None:
        rows = []
        cells = {
            (4, 1, "A"): 10.0, (4, 8, "A"): 31.0,
            (4, 1, "B"): 12.0, (4, 8, "B"): 61.0,
            (4, 1, "C"): 10.5, (4, 8, "C"): 57.0,
            (1, 8, "A"): 1.0, (8, 8, "A"): 0.8, (16, 8, "A"): 0.7,
        }
        for (m, K, variant), total_seconds in cells.items():
            for sample_id in range(21):
                for round_index in range(K):
                    rows.append({
                        "d": 512, "m": m, "K": K, "variant": variant,
                        "is_warmup": False, "block_id": sample_id // 5,
                        "sample_id": sample_id, "round_index": round_index,
                        "valid": True, "qpc_seconds": total_seconds / K,
                        "effective_macs": 100, "invalid_reason": "",
                    })
        first = analysis.bootstrap_gate_diagnostics(rows, reps=50)
        second = analysis.bootstrap_gate_diagnostics(rows, reps=50)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"rho_resident", "delta_CB", "G_matrix"})
        self.assertTrue(all(row["replicates"] == 50 for row in first.values()))

    def test_contractual_skip_forces_acceptance_hold(self) -> None:
        terminal, reason = analysis.choose_terminal(
            {"status": "MEASUREMENT_INVALID", "reason": "synthetic preflight stop"},
            [{"name": "required", "status": "SKIP"}],
            {},
        )
        self.assertEqual(terminal, "V2_1_ACCEPTANCE_HOLD")
        self.assertIn("SKIP", reason)

    def test_test_report_builder_emits_every_required_name_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "omega_v2_1_bench.exe"
            executable.write_bytes(b"native-test-fixture")
            (root / "native_run_status.json").write_text('{"status":"MEASUREMENT_INVALID"}', encoding="utf-8")
            rows = analysis.build_test_rows({}, executable, {}, {}, {}, {}, [], root)
        self.assertEqual([row["name"] for row in rows], analysis.TEST_NAMES)
        self.assertEqual(len({row["name"] for row in rows}), 37)

    def test_conformance_block_does_not_claim_measurement_before_sweep(self) -> None:
        families = [
            {"d": d, "matrices": [{} for _ in range(7)]}
            for d in (512, 640)
        ]
        block = analysis.build_conformance_block(
            {"implementation_commit": "fixture", "implementation_parent": "parent"},
            {"terminal_status": "OMEGA_V2_0_CONFORMANT_PASS", "execution_commit": analysis.V2_0_IMPLEMENTATION_COMMIT, "artifact_sha256": {}},
            {}, {"core_families": families}, {}, {"native_status": {"status": "MEASUREMENT_INVALID"}}, [], "V2_1_ACCEPTANCE_HOLD",
        )["OMEGA_CONFORMANCE_BLOCK"]
        self.assertFalse(block["actual_candidate"]["values_obtained_by_introspection_and_native_measurement"])

    def test_report_renders_complete_absolute_hash_records(self) -> None:
        digest = "a" * 64
        report = analysis.format_report(
            "V2_1_ACCEPTANCE_HOLD", "fixture", {
                "implementation_commit": "commit", "implementation_parent": "parent", "branch": "main",
                "git_status_porcelain": "",
            }, {}, None, [], {"C:\\fixture\\artifact.bin": {
                "sha256": digest, "size_bytes": 17, "absolute_path": "C:\\fixture\\artifact.bin",
            }},
        )
        self.assertIn(digest, report)
        self.assertIn("C:\\fixture\\artifact.bin", report)

    def test_sweep_completeness_checks_all_cells_blocks_and_rounds(self) -> None:
        rows = []
        for d in (512, 640):
            for m in (1, 4, 8, 16):
                for K in (1, 4, 8):
                    for variant in ("A", "B", "C"):
                        for warmup_index in range(10):
                            for round_index in range(K):
                                rows.append({
                                    "d": d, "m": m, "K": K, "variant": variant,
                                    "is_warmup": True, "block_id": -1,
                                    "sample_id": warmup_index - 10, "round_index": round_index,
                                })
                        for sample_id in range(105):
                            for round_index in range(K):
                                rows.append({
                                    "d": d, "m": m, "K": K, "variant": variant,
                                    "is_warmup": False, "block_id": sample_id // 21,
                                    "sample_id": sample_id, "round_index": round_index,
                                })
        self.assertTrue(analysis.check_sweep_completeness(rows, "SWEEP_COMPLETE"))
        self.assertFalse(analysis.check_sweep_completeness(rows[:-1], "SWEEP_COMPLETE"))
        self.assertFalse(analysis.check_sweep_completeness(rows, "MEASUREMENT_INVALID"))


if __name__ == "__main__":
    unittest.main()
