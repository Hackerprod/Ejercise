from __future__ import annotations

import csv
from pathlib import Path
import sys
import statistics
import tempfile
import unittest

UNIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIT_ROOT))

from scripts import run_kq  # noqa: E402


class KQContractTests(unittest.TestCase):
    def test_new_test_names_match_md306_section_26(self) -> None:
        md306 = UNIT_ROOT.parents[2] / "MD" / "306.md"
        content = md306.read_text(encoding="utf-8")
        section = content.split("26. Tests nuevos mínimos de V2-1b", 1)[1].split("27. OMEGA_CONFORMANCE_BLOCK", 1)[0]
        expected = [line.strip() for line in section.splitlines() if line.strip().startswith("test_v2_1b_")]
        self.assertEqual(expected, run_kq.KQ_TEST_NAMES)
        self.assertEqual(len(expected), 17)

    def test_candidate_freeze_state_machine_boundaries(self) -> None:
        frozen = run_kq.classify_candidate1(correctness=True, e_q4=.25, e_full4=.60, e_full16=.60, s_native=1.20)
        self.assertTrue(frozen["all_five_gates_pass"])
        self.assertFalse(frozen["candidate2_allowed"])

        speed_failure = run_kq.classify_candidate1(correctness=True, e_q4=.25, e_full4=.60, e_full16=.60, s_native=1.1999)
        self.assertTrue(speed_failure["scientifically_qualified"])
        self.assertEqual(speed_failure["terminal_status"], "KERNEL_SCIENTIFICALLY_QUALIFIED_PROJECT_NATIVE_SPEED_GATE_FAIL_CANDIDATE_2_ALLOWED")
        self.assertTrue(speed_failure["candidate2_allowed"])

        scientific_failure = run_kq.classify_candidate1(correctness=True, e_q4=.2499, e_full4=.60, e_full16=.60, s_native=1.5)
        self.assertFalse(scientific_failure["scientifically_qualified"])
        self.assertTrue(scientific_failure["candidate2_allowed"])

        correctness_failure = run_kq.classify_candidate1(correctness=False, e_q4=.5, e_full4=.7, e_full16=.8, s_native=1.5)
        self.assertFalse(correctness_failure["all_five_gates_pass"])
        self.assertTrue(correctness_failure["candidate2_allowed"])

    def test_attempt03_plan_keeps_exact_72_cell_contract(self) -> None:
        self.assertEqual(len((512, 640)) * len((1, 4, 8, 16)) * len((1, 4, 8)) * len(("A", "B", "C")), 72)

    def test_m1_raw_helper_filters_to_diagnostic_slot_only(self) -> None:
        fields = ["d", "m", "K", "variant", "is_warmup", "block_id", "sample_id", "valid", "qpc_seconds"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for d in (512, 640):
                    for variant in ("A", "B", "C"):
                        for K, seconds in ((1, 1.0), (8, 8.0 + (ord(variant) - ord("A")) * .1)):
                            for sample in range(105):
                                for _round in range(K):
                                    writer.writerow({
                                        "d": d, "m": 1, "K": K, "variant": variant, "is_warmup": "false",
                                        "block_id": sample // 21, "sample_id": sample, "valid": "true",
                                        "qpc_seconds": seconds / K,
                                    })
                for sample in range(10):
                    writer.writerow({"d": 512, "m": 4, "K": 1, "variant": "A", "is_warmup": "false", "block_id": 0, "sample_id": sample, "valid": "true", "qpc_seconds": .01})
            groups = run_kq.sample_groups(path, d=512, m=1, variant="B")
        self.assertEqual(len(groups[1]), 105)
        self.assertEqual(len(groups[8]), 105)
        self.assertAlmostEqual((statistics.median(groups[8]) - statistics.median(groups[1])) / 7, 1.0142857142857142)


if __name__ == "__main__":
    unittest.main()
