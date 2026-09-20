from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_omega_native_runtime_p2r0_golden as golden  # noqa: E402


def test_small_oracle_case_has_forward_and_autograd_tensors(tmp_path: Path) -> None:
    model = golden.ce.fresh_model(20260913, 1, vocab_size=31, dimension=8, slots=2)
    tokens = torch.randint(0, 31, (2, 3), dtype=torch.long)
    tensors, metadata = golden.make_golden_case(model, tokens, k=1, window=0, seed=55, source_pair=0)
    assert metadata["config"]["sequence_length"] == 3
    assert tensors["token_part"].shape == (2, 3, 16)
    assert tensors["readout_states"].shape == (2, 3, 2, 8)
    assert tensors["d_token_part"].shape == tensors["token_part"].shape
    assert tensors["d_previous_state"].shape == tensors["previous_state"].shape
    assert tensors["d_block_qkv_weight"].shape == tensors["block_qkv_weight"].shape


def test_binary_layout_round_trip(tmp_path: Path) -> None:
    model = golden.ce.fresh_model(20260913, 1, vocab_size=31, dimension=8, slots=2)
    tokens = torch.randint(0, 31, (2, 3), dtype=torch.long)
    tensors, metadata = golden.make_golden_case(model, tokens, k=1, window=0, seed=55, source_pair=0)
    path = tmp_path / "golden.bin"
    golden.write_golden_case(path, tensors, metadata)
    header, restored = golden.read_golden_case(path)
    assert header["magic"] == golden.MAGIC
    assert header["tensor_order"] == list(golden.TENSOR_ORDER)
    assert header["metadata"]["case"] == "K1_window0"
    for name in golden.TENSOR_ORDER:
        assert torch.equal(restored[name], tensors[name])


def test_tensor_order_is_complete_and_stable() -> None:
    assert golden.TENSOR_ORDER[:3] == ("token_part", "previous_state", "state_part_weight")
    assert golden.TENSOR_ORDER[-3:] == ("d_block_norm_weight", "d_depth_embedding_weight", "d_gate_logits")
    assert len(set(golden.TENSOR_ORDER)) == len(golden.TENSOR_ORDER)
