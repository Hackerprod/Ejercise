# P2-R0 golden binary layout

Each `golden_K{K}_window{window}.bin` is a little-endian file:

1. `uint32_le` header byte length.
2. UTF-8 JSON header of that length.
3. Raw tensor payload blobs in `header.tensor_order` order.

Header contains `magic`, `format_version`, case metadata, configuration, and a
descriptor for every tensor: `shape`, `dtype`, `byte_offset`, and
`byte_length`. Offsets are relative to the start of the payload (after JSON).
Every tensor is contiguous C-order `float32`, serialized as little-endian
IEEE-754 bytes. No padding or compression exists between blobs.

Tensor order:

1. `token_part [B,T,S*D]`
2. `previous_state [B,S,D]`
3. `state_part_weight [S*D,D]`
4. `prelude_norm_weight [S*D]`
5. Shared block parameters: `qkv_weight`, `qkv_bias`, `out_weight`, `out_bias`,
   `fc1_weight`, `fc1_bias`, `fc2_weight`, `fc2_bias`, `norm_weight`.
6. `depth_embedding_weight [K,D]`, `gate_logits [K,D]`.
7. Expected forward outputs: `next_state [B,S,D]`,
   `readout_states [B,T,S,D]`.
8. Upstreams: `d_readout_states`, `d_next_state`.
9. Expected PyTorch autograd gradients: `d_token_part`, `d_previous_state`,
   and `d_` for every recurrent parameter in steps 3–6.

The gradient objective is:

```text
sum(readout_states * d_readout_states) + sum(next_state * d_next_state)
```

Cases are generated from the P0 production forward oracle for K1/K4 and
window0/window1. They contain no PyTorch or model-runtime objects; the later
standalone C-ABI test can parse them without Python.
