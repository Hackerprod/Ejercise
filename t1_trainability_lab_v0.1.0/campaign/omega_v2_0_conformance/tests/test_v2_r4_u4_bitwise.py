import unittest

import _bootstrap  # noqa: F401
from omega_v2.variants import run_bitwise_acceptance


class R4U4BitwiseTests(unittest.TestCase):
    def test_v2_r4_u4_bitwise_three_seeds_all_d_m(self):
        report = run_bitwise_acceptance()
        self.assertEqual(report["seeds"], [20260929, 20260930, 20260931])
        self.assertEqual(report["row_count"], 18)
        self.assertEqual(report["status"], "PASS")
        for row in report["checks"]:
            self.assertTrue(row["initial_values_bitwise_equal"])
            self.assertTrue(all(row["round_trace_bitwise_equal"]))
            self.assertTrue(row["final_output_bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
