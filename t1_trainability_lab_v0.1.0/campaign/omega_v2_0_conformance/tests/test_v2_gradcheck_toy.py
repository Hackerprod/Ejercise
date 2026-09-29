import unittest

import _bootstrap  # noqa: F401
from omega_v2.reference import toy_gradcheck_report, zero_weight_residual_report


class ToyGradcheckTests(unittest.TestCase):
    def test_v2_gradcheck_toy(self):
        report = toy_gradcheck_report()["gradcheck"]
        self.assertEqual({row["family"] for row in report["families"]}, {"W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"})
        self.assertLessEqual(report["max_relative_error"], 1e-5)
        self.assertTrue(report["pass"])

    def test_v2_zero_weight_residual_bitwise(self):
        report = zero_weight_residual_report()["zero_weight_residual"]
        self.assertTrue(report["step_bitwise_identity"])
        self.assertTrue(report["K2_bitwise_identity"])
        self.assertTrue(report["pass"])


if __name__ == "__main__":
    unittest.main()
