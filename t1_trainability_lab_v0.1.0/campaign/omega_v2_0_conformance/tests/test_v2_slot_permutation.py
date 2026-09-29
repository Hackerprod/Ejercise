import unittest

import _bootstrap  # noqa: F401
from omega_v2.reference import slot_permutation_report


class SlotPermutationTests(unittest.TestCase):
    def test_v2_slot_permutation_equivariance(self):
        report = slot_permutation_report()["slot_permutation_equivariance"]
        self.assertEqual((report["d"], report["m"], report["K"]), (8, 4, 2))
        self.assertTrue(report["pass"])
        self.assertLessEqual(report["max_abs_error"], report["abs_tolerance"])


if __name__ == "__main__":
    unittest.main()
