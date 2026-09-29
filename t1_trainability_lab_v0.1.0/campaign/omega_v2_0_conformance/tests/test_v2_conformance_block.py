import json
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from omega_v2.ledger import build_parameter_ledger


UNIT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = UNIT_ROOT / "results" / "omega_v2_0_conformance"


class ConformanceBlockTests(unittest.TestCase):
    def test_v2_conformance_block(self):
        path = RESULTS / "OMEGA_CONFORMANCE_BLOCK.yaml"
        self.assertTrue(path.is_file(), "run_conformance.py must materialize the block before tests")
        document = json.loads(path.read_text(encoding="utf-8"))
        block = document["OMEGA_CONFORMANCE_BLOCK"]
        self.assertEqual(block["authority"]["snapshot_commit"], "38061477d4c2b0c5c20d75d902b21dd1ef0a2611")
        self.assertEqual(block["authority"]["conversacion_md_blob"], "7027e2ac9d1ba89db08dda73c81e244f3b9b19db")
        self.assertIn(block["status"], ("CONFORMANT", "CONFORMANCE_HOLD"))
        self.assertEqual(block["global_conformance_status"], "CONFORMANCE_HOLD")
        self.assertEqual(block["permitted_follow_on_if_v2_0_passes"], "OMEGA-V2-1 T0 PHYSICAL ONLY")
        self.assertEqual(block["deviations"], [])
        self.assertEqual(block["authorized_deviation_ids"], [])
        self.assertTrue(block["actual_candidate"]["values_obtained_by_introspection"])

        expected = {
            (row["d"], row["candidate_id"].rsplit("-", 1)[-1]): row["P_core_unique"]
            for row in build_parameter_ledger()["rows"]
        }
        actual = {
            (row["d"], row["candidate_id"].rsplit("-", 1)[-1]): row["P_core_unique"]
            for row in block["actual_candidate"]["candidates"]
        }
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
