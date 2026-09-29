import unittest

import torch

import _bootstrap  # noqa: F401
from omega_v2.core import SharedRecurrentCore, make_seeded_core, state_dict_values_equal, tensor_shapes
from omega_v2.ledger import state_dict_sha256
from omega_v2.variants import fixed_input


class KFlexTests(unittest.TestCase):
    def test_v2_k_flex_same_object_and_state_dict(self):
        for d in (512, 640):
            core = SharedRecurrentCore(make_seeded_core(d, 20260929))
            reference = SharedRecurrentCore(make_seeded_core(d, 20260929))
            initial_keys = tuple(core.state_dict())
            initial_shapes = tensor_shapes(core)
            initial_hash = state_dict_sha256(core)
            initial_count = sum(parameter.numel() for parameter in core.parameters())
            self.assertTrue(state_dict_values_equal(core, reference))
            for m in (4, 8, 16):
                state = fixed_input(d, m, 20260929 + m)
                for k in (1, 2, 4, 8, 16):
                    output = core(state, K=k)
                    self.assertEqual(tuple(output.shape), (1, m, d))
                    self.assertTrue(torch.isfinite(output).all().item())
                    explicit = state
                    for _ in range(k):
                        explicit = core.block.step(explicit)
                    self.assertTrue(torch.equal(output, explicit), f"K={k} differed from explicit repeated step loop")
                    self.assertEqual(tuple(core.state_dict()), initial_keys)
                    self.assertEqual(tensor_shapes(core), initial_shapes)
                    self.assertEqual(sum(parameter.numel() for parameter in core.parameters()), initial_count)
                    self.assertEqual(state_dict_sha256(core), initial_hash)


if __name__ == "__main__":
    unittest.main()
