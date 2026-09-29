import unittest

import _bootstrap  # noqa: F401
from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES
from omega_v2.ledger import build_parameter_ledger
from omega_v2.variants import R4Shared, U4Untied


class ParameterCountTests(unittest.TestCase):
    def test_v2_parameter_counts(self):
        ledger = build_parameter_ledger()
        rows = {(row["d"], row["candidate_id"].rsplit("-", 1)[-1]): row for row in ledger["rows"]}
        for d in (512, 640):
            expected = 16 * d * d
            self.assertEqual(rows[(d, "R4")]["P_core_unique"], expected)
            self.assertEqual(rows[(d, "U4")]["P_core_unique"], 4 * expected)
            self.assertEqual(rows[(d, "R4")]["P_attention_projection_total"], 4 * d * d)
            self.assertEqual(rows[(d, "R4")]["P_swiglu_total"], 12 * d * d)
            self.assertEqual(rows[(d, "U4")]["P_core_unique"] / rows[(d, "R4")]["P_core_unique"], 4.0)

            block = ContractualCoreBlock(d, seed=20260929)
            self.assertEqual(set(dict(block.named_parameters())), set(MATRIX_FAMILIES))
            self.assertEqual(dict(block.named_buffers()), {})
            self.assertEqual(sum(p.numel() for p in block.parameters()), expected)

            r4 = R4Shared.seeded(d, 20260929)
            u4 = U4Untied.from_shared(r4)
            self.assertEqual(sum(p.numel() for p in r4.parameters()), expected)
            self.assertEqual(sum(p.numel() for p in u4.parameters()), 4 * expected)


if __name__ == "__main__":
    unittest.main()
