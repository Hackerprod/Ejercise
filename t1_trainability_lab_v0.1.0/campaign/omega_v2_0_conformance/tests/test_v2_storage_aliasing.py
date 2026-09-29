import unittest

import _bootstrap  # noqa: F401
from omega_v2.variants import R4Shared, U4Untied, clone_value_and_storage_report


class StorageAliasingTests(unittest.TestCase):
    def test_v2_storage_aliasing(self):
        for d in (512, 640):
            r4 = R4Shared.seeded(d, 20260929)
            u4 = U4Untied.from_shared(r4)
            report = clone_value_and_storage_report(r4, u4)
            self.assertTrue(report["initial_values_bitwise_equal"])
            self.assertTrue(report["r4_reuses_one_block_object"])
            self.assertTrue(report["u4_has_four_block_objects"])
            self.assertTrue(report["u4_block_storages_pairwise_disjoint"])
            self.assertTrue(report["u4_storage_disjoint_from_r4"])


if __name__ == "__main__":
    unittest.main()
