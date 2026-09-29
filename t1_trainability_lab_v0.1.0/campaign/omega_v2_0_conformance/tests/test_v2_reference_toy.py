import unittest

import _bootstrap  # noqa: F401
from omega_v2.reference import toy_reference_report


class ToyReferenceTests(unittest.TestCase):
    def test_v2_reference_toy(self):
        report = toy_reference_report()["naive_reference"]
        self.assertEqual((report["dtype"], report["d"], report["m"], report["K"]), ("FP64", 8, 2, 2))
        self.assertLessEqual(report["max_abs_error"], 1e-12)
        self.assertLessEqual(report["max_rel_error"], 1e-10)
        self.assertTrue(report["pass"])


if __name__ == "__main__":
    unittest.main()
