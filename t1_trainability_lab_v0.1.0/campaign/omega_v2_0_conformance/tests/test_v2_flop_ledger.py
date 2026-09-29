import unittest

import _bootstrap  # noqa: F401
from omega_v2.ledger import build_flop_ledger


class FlopLedgerTests(unittest.TestCase):
    def test_v2_flop_ledger_r4_u4_exact_compute_match(self):
        ledger = build_flop_ledger()
        pairs = [row for row in ledger["rows"] if row.get("variant_pair") == "R4_vs_U4"]
        self.assertEqual(len(pairs), 6)
        for row in pairs:
            d, m = row["d"], row["m"]
            expected = 2 * (4 * (16 * m * d * d + 2 * m * m * d))
            self.assertEqual(row["R4_forward_flops"], expected)
            self.assertEqual(row["U4_forward_flops"], expected)
            self.assertTrue(row["exact_compute_match"])


if __name__ == "__main__":
    unittest.main()
