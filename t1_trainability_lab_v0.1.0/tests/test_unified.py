from __future__ import annotations

import torch
import pytest

from t1_trainability.unified import (
    OPCODE_IDS,
    ROW_ASSIGN,
    ROW_ATTR,
    ROW_PAIR,
    ROW_REL,
    ROW_VEC,
    SLOT_P,
    SLOT_R,
    SLOT_W,
    SLOT_COUNT,
    CandidateState,
    ReadResult,
    READ_MODE_BLEND,
    READ_MODE_SELECT,
    SharedMemoryReader,
    TypedCommit,
    UnifiedT1U0,
    _select_payload,
)


def one_hot(index: int, dimension: int) -> torch.Tensor:
    value = torch.zeros(dimension)
    value[index] = 1.0
    return value


def read_fixture(reader: SharedMemoryReader, state_value: torch.Tensor, memory_keys: torch.Tensor, memory_values: torch.Tensor, memory_types: int, opcode: str) -> torch.Tensor:
    state = torch.zeros(1, 4, reader.dimension)
    state[0, SLOT_P] = state_value
    result = reader(
        state,
        memory_keys.unsqueeze(0),
        memory_values.unsqueeze(0),
        torch.tensor([[memory_types]], dtype=torch.long),
        torch.tensor([[True]]),
        torch.tensor([OPCODE_IDS[opcode]]),
        torch.tensor([0]),
        torch.tensor([SLOT_P]),
    )
    assert result.valid.tolist() == [True]
    return result.payload[0]


def test_shared_reader_equivalence_for_rel_pair_assign_attr_and_vec() -> None:
    dimension = 8
    reader = SharedMemoryReader(dimension, attention_temperature=40.0)
    key = one_hot(1, dimension)
    value = one_hot(2, dimension)
    assert torch.cosine_similarity(read_fixture(reader, key, key.unsqueeze(0), value.unsqueeze(0), ROW_REL, "READ_P"), value.unsqueeze(0)).item() > 0.99

    pair_key = one_hot(3, dimension)
    pair_value = one_hot(4, dimension)
    assert torch.cosine_similarity(read_fixture(reader, pair_key, pair_key.unsqueeze(0), pair_value.unsqueeze(0), ROW_PAIR, "READ_E"), pair_value.unsqueeze(0)).item() > 0.99

    reference = one_hot(5, dimension)
    assigned = one_hot(6, dimension)
    assert torch.cosine_similarity(read_fixture(reader, reference, reference.unsqueeze(0), assigned.unsqueeze(0), ROW_ASSIGN, "READ_P"), assigned.unsqueeze(0)).item() > 0.99

    attribute = one_hot(7, dimension)
    color = one_hot(0, dimension)
    assert torch.cosine_similarity(read_fixture(reader, attribute, attribute.unsqueeze(0), color.unsqueeze(0), ROW_ATTR, "READ_E"), color.unsqueeze(0)).item() > 0.99

    vector_key = one_hot(2, dimension)
    vector_value = one_hot(1, dimension)
    assert torch.cosine_similarity(read_fixture(reader, vector_key, vector_key.unsqueeze(0), vector_value.unsqueeze(0), ROW_VEC, "ACCUM_W"), vector_value.unsqueeze(0)).item() > 0.99


def test_reader_noop_immediate_is_exactly_unconditioned() -> None:
    dimension = 8
    reader = SharedMemoryReader(dimension, attention_temperature=8.0)
    state = torch.zeros(1, 4, dimension)
    state[0, SLOT_P, 0] = 1.0
    keys = torch.zeros(1, 2, dimension)
    keys[0, 0, 0] = 8.0
    keys[0, 1, 1] = 8.0
    values = keys.clone()
    types = torch.full((1, 2), ROW_REL, dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.bool)
    opcode = torch.tensor([OPCODE_IDS["READ_P"]])
    source = torch.tensor([SLOT_P])
    with_noop = reader(state, keys, values, types, mask, opcode, torch.tensor([511]), source)
    without_control = reader(state, keys, values, types, mask, opcode, torch.zeros(1, dimension), source)
    assert torch.equal(with_noop.attention, without_control.attention)
    assert torch.equal(with_noop.payload, without_control.payload)


def test_reader_concentrates_on_exact_key_and_returns_p2_fixture_payload() -> None:
    dimension = 8
    reader = SharedMemoryReader(dimension, attention_temperature=8.0)
    state = torch.zeros(1, 4, dimension)
    state[0, SLOT_P, 0] = 1.0
    keys = torch.zeros(1, 3, dimension)
    keys[0, 0, 0] = 8.0
    keys[0, 1, 1] = 8.0
    keys[0, 2, 2] = 8.0
    values = torch.zeros_like(keys)
    values[0, 0, 3] = 1.0
    result = reader(
        state,
        keys,
        values,
        torch.full((1, 3), ROW_REL, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([OPCODE_IDS["READ_P"]]),
        torch.tensor([511]),
        torch.tensor([SLOT_P]),
    )
    assert int(result.attention.argmax(dim=-1).item()) == 0
    assert float(result.attention[0, 0].detach()) > 0.99
    assert torch.cosine_similarity(result.payload, values[0, 0]).item() > 0.99


def test_reader_select_returns_hard_payload_and_preserves_soft_diagnostics() -> None:
    dimension = 8
    reader = SharedMemoryReader(dimension, attention_temperature=8.0)
    state = torch.zeros(1, 4, dimension)
    state[0, SLOT_P, 0] = 1.0
    keys = torch.zeros(1, 2, dimension)
    keys[0, 0, 0] = 8.0
    keys[0, 1, 1] = 8.0
    values = torch.zeros_like(keys)
    values[0, 0, 3] = 1.0
    values[0, 1, 4] = 1.0
    result = reader(
        state,
        keys,
        values,
        torch.full((1, 2), ROW_VEC, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([OPCODE_IDS["ACCUM_W"]]),
        torch.tensor([511]),
        torch.tensor([SLOT_P]),
        read_mode="SELECT",
    )
    assert result.read_mode == "SELECT"
    assert result.selected_index.tolist() == [0]
    assert torch.equal(result.payload, values[:, 0])
    assert torch.equal(result.attention, result.attention_soft)
    assert torch.equal(result.margin, result.selection_margin)


def test_reader_select_rejects_non_accum_w_opcode() -> None:
    reader = SharedMemoryReader(8)
    state = torch.zeros(1, 4, 8)
    keys = torch.zeros(1, 1, 8)
    types = torch.full((1, 1), ROW_REL, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="only for ACCUM_W"):
        reader(state, keys, keys, types, mask, torch.tensor([OPCODE_IDS["READ_P"]]), torch.tensor([511]), torch.tensor([SLOT_P]), read_mode="SELECT")


def test_select_ste_forward_matches_eval_gather() -> None:
    torch.manual_seed(41)
    reader = SharedMemoryReader(8, attention_temperature=4.0)
    state = torch.randn(1, 4, 8)
    keys = torch.randn(1, 3, 8)
    values = torch.randn(1, 3, 8)
    types = torch.full((1, 3), ROW_VEC, dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)
    args = (state, keys, values, types, mask, torch.tensor([OPCODE_IDS["ACCUM_W"]]), torch.tensor([511]), torch.tensor([SLOT_P]))
    reader.train()
    ste = reader(*args, read_mode="SELECT")
    reader.eval()
    hard = reader(*args, read_mode="SELECT")
    assert torch.equal(ste.selected_index, hard.selected_index)
    torch.testing.assert_close(ste.payload, hard.payload, rtol=0.0, atol=0.0)


def test_select_ste_score_gradient_matches_soft_substitute() -> None:
    scores = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
    values = torch.tensor([[[1.0, 2.0], [-2.0, 0.5], [0.25, -1.0]]])
    legal = torch.ones_like(scores, dtype=torch.bool)
    attention = torch.softmax(scores.masked_fill(~legal, torch.finfo(scores.dtype).min), dim=-1)
    selected = scores.argmax(dim=-1)
    valid = torch.ones(1, dtype=torch.bool)
    upstream = torch.tensor([[0.75, -1.25]])
    payload = _select_payload(attention, values, selected, valid, training=True)
    actual = torch.autograd.grad((payload * upstream).sum(), scores, retain_graph=True)[0]
    expected = torch.autograd.grad((attention.unsqueeze(-1) * values * upstream.unsqueeze(1)).sum(), scores)[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


def test_select_ste_value_gradient_hits_selected_row_only() -> None:
    scores = torch.tensor([[0.25, 3.0, -0.5]], requires_grad=True)
    values = torch.randn(1, 3, 4, requires_grad=True)
    attention = torch.softmax(scores, dim=-1)
    selected = scores.argmax(dim=-1)
    payload = _select_payload(attention, values, selected, torch.ones(1, dtype=torch.bool), training=True)
    gradient = torch.autograd.grad(payload.sum(), values)[0]
    expected = torch.zeros_like(values)
    expected[0, 1] = 1.0
    torch.testing.assert_close(gradient, expected, rtol=0.0, atol=0.0)


def test_mixed_select_blend_ste_gradient_matches_separate_mean_loss() -> None:
    scores = torch.tensor([[2.0, 0.5], [0.25, 1.5]], requires_grad=True)
    values = torch.randn(2, 2, 3, requires_grad=True)
    attention = torch.softmax(scores, dim=-1)
    selected = scores.argmax(dim=-1)
    valid = torch.ones(2, dtype=torch.bool)
    select_payload = _select_payload(attention, values, selected, valid, training=True)
    blend_payload = torch.einsum("bm,bmd->bd", attention, values)
    mixed = torch.where(torch.tensor([[True], [False]]), select_payload, blend_payload)
    mixed_grads = torch.autograd.grad(mixed.square().mean(), (scores, values))

    scores_ref = scores.detach().clone().requires_grad_()
    values_ref = values.detach().clone().requires_grad_()
    attention_ref = torch.softmax(scores_ref, dim=-1)
    selected_ref = scores_ref.argmax(dim=-1)
    select_ref = _select_payload(attention_ref[:1], values_ref[:1], selected_ref[:1], valid[:1], training=True)
    blend_ref = torch.einsum("bm,bmd->bd", attention_ref[1:], values_ref[1:])
    separate_loss = (select_ref.square().mean() + blend_ref.square().mean()) / 2.0
    separate_grads = torch.autograd.grad(separate_loss, (scores_ref, values_ref))
    torch.testing.assert_close(mixed_grads[0], separate_grads[0], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(mixed_grads[1], separate_grads[1], rtol=0.0, atol=1e-7)


def test_select_ste_masks_illegal_rows_and_nonread_padding() -> None:
    reader = SharedMemoryReader(4).train()
    state = torch.zeros(3, 4, 4)
    keys = torch.randn(3, 2, 4)
    values = torch.randn(3, 2, 4, requires_grad=True)
    types = torch.tensor([[ROW_VEC, ROW_REL], [ROW_REL, ROW_REL], [ROW_VEC, ROW_REL]])
    mask = torch.ones(3, 2, dtype=torch.bool)
    result = reader(
        state,
        keys,
        values,
        types,
        mask,
        torch.tensor([OPCODE_IDS["ACCUM_W"], OPCODE_IDS["ACCUM_W"], OPCODE_IDS["EMIT"]]),
        torch.tensor([511, 511, 511]),
        torch.tensor([SLOT_P, SLOT_P, SLOT_P]),
        read_mode=torch.tensor([READ_MODE_SELECT, READ_MODE_SELECT, READ_MODE_SELECT]),
    )
    assert result.valid.tolist() == [True, False, False]
    assert result.selected_index.tolist()[1:] == [-1, -1]
    assert torch.isfinite(result.payload).all()
    assert torch.equal(result.payload[1:], torch.zeros_like(result.payload[1:]))
    result.payload.sum().backward()
    assert values.grad is not None
    assert torch.count_nonzero(values.grad[1:]) == 0


def test_one_reader_call_per_read_round_and_none_for_alu_emit() -> None:
    model = UnifiedT1U0(dimension=8)
    state = torch.zeros(1, 4, 8)
    memory_keys = torch.eye(8)[:1]
    memory_values = torch.eye(8)[:1]
    memory_types = torch.tensor([[ROW_VEC]])
    row_mask = torch.ones(1, 1, dtype=torch.bool)
    presence = torch.ones(1, 4, dtype=torch.bool)
    for name, expected_calls in (("READ_P", 1), ("ALU_ADD", 0), ("READ_E", 1), ("ACCUM_W", 1), ("EMIT", 0)):
        model.memory_reader.reset_call_count()
        model.step(
            state,
            memory_keys,
            memory_values,
            memory_types,
            row_mask,
            torch.tensor([OPCODE_IDS[name]]),
            torch.tensor([0]),
            torch.tensor([SLOT_P]),
            torch.tensor([SLOT_R]),
            presence,
        )
        assert model.memory_reader.call_count == expected_calls, name


def test_emit_is_identity_after_each_task_round() -> None:
    """Padding EMIT rounds must not mutate state for any U0 task."""
    torch.manual_seed(29)
    model = UnifiedT1U0(dimension=8).eval()
    task_rounds = {
        "pointer_chasing": ("READ_P", ROW_REL),
        "multi_hop": ("READ_P", ROW_REL),
        "associative_recall": ("READ_E", ROW_PAIR),
        "variable_binding": ("READ_E", ROW_ATTR),
        "sequential_update": ("ALU_ADD", ROW_REL),
        "workspace_accumulation": ("ACCUM_W", ROW_VEC),
    }
    memory_keys = torch.eye(8)[:2]
    memory_values = torch.roll(memory_keys, shifts=1, dims=-1)
    row_mask = torch.ones(1, 2, dtype=torch.bool)
    presence = torch.ones(1, SLOT_COUNT, dtype=torch.bool)
    for task, (opcode_name, row_type) in task_rounds.items():
        memory_types = torch.full((1, 2), row_type, dtype=torch.long)
        state = torch.randn(1, SLOT_COUNT, 8)
        active_state, _, _ = model.step(
            state,
            memory_keys.unsqueeze(0),
            memory_values.unsqueeze(0),
            memory_types,
            row_mask,
            torch.tensor([OPCODE_IDS[opcode_name]]),
            torch.zeros(1, 8),
            torch.tensor([SLOT_P]),
            torch.tensor([SLOT_R if task == "sequential_update" else SLOT_W]),
            presence,
        )
        emitted_state, _, _ = model.step(
            active_state,
            memory_keys.unsqueeze(0),
            memory_values.unsqueeze(0),
            memory_types,
            row_mask,
            torch.tensor([OPCODE_IDS["EMIT"]]),
            torch.zeros(1, 8),
            torch.tensor([SLOT_P]),
            torch.tensor([SLOT_W]),
            presence,
        )
        torch.testing.assert_close(emitted_state, active_state, msg=f"EMIT mutated {task} state")
        for _ in range(2):
            emitted_state, _, emit_result = model.step(
                emitted_state,
                memory_keys.unsqueeze(0),
                memory_values.unsqueeze(0),
                memory_types,
                row_mask,
                torch.tensor([OPCODE_IDS["EMIT"]]),
                torch.zeros(1, 8),
                torch.tensor([SLOT_P]),
                torch.tensor([SLOT_W]),
                presence,
            )
            torch.testing.assert_close(emitted_state, active_state, msg=f"EMIT padding mutated {task} state")
            assert not emit_result.valid.any()
            assert torch.count_nonzero(emit_result.payload) == 0


def test_alu_adapter_is_inactive_for_retrieval_and_workspace_opcodes() -> None:
    torch.manual_seed(17)
    model = UnifiedT1U0(dimension=8).eval()
    model.core.mlp.network[-1].weight.data.normal_()
    model.core.mlp.network[-1].bias.data.normal_()
    state = torch.randn(2, 4, 8)
    immediate = torch.randn(2, 8)
    read_payload = torch.randn(2, 8)
    presence = torch.ones(2, 4, dtype=torch.bool)

    def run(opcode_name: str) -> torch.Tensor:
        opcode = torch.full((2,), OPCODE_IDS[opcode_name], dtype=torch.long)
        return model.core(
            model.normalize_state(state, presence),
            model.opcode_embedding(opcode),
            immediate,
            read_payload,
            model.slot_type_embeddings,
            presence,
            opcode=opcode,
        ).values

    before = {name: run(name) for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL", "READ_P", "READ_E", "ACCUM_W")}
    model.core.alu_adapters["ALU_ADD"]["up"].weight.data.normal_()
    after = {name: run(name) for name in before}
    for name in ("ALU_SUB", "ALU_MUL", "READ_P", "READ_E", "ACCUM_W"):
        torch.testing.assert_close(before[name], after[name])
    assert not torch.equal(before["ALU_ADD"], after["ALU_ADD"])


def test_inactive_typed_modules_have_none_grad_and_do_not_move() -> None:
    torch.manual_seed(23)
    model = UnifiedT1U0(dimension=8).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    state = torch.randn(1, 4, 8)
    memory = torch.zeros(1, 1, 8)
    memory_types = torch.tensor([[ROW_REL]])
    row_mask = torch.ones(1, 1, dtype=torch.bool)
    presence = torch.ones(1, 4, dtype=torch.bool)
    optimizer.zero_grad(set_to_none=True)
    next_state, _, _ = model.step(
        state,
        memory,
        memory,
        memory_types,
        row_mask,
        torch.tensor([OPCODE_IDS["READ_P"]]),
        torch.zeros(1, 8),
        torch.tensor([SLOT_P]),
        torch.tensor([SLOT_P]),
        presence,
    )
    typed = [(name, parameter) for name, parameter in model.named_parameters() if "alu_" in name.lower() or "operation_heads" in name.lower()]
    before = {name: parameter.detach().clone() for name, parameter in typed}
    next_state.sum().backward()
    assert all(parameter.grad is None for _, parameter in typed)
    optimizer.step()
    for name, parameter in typed:
        torch.testing.assert_close(parameter, before[name])


def run_pointer_chain(model: UnifiedT1U0, keys: torch.Tensor, values: torch.Tensor, rows: torch.Tensor, rounds: int) -> torch.Tensor:
    dimension = keys.shape[-1]
    state = torch.zeros(1, 4, dimension)
    state[0, SLOT_P] = keys[0]
    mask = torch.ones(1, rows.shape[0], dtype=torch.bool)
    types = torch.full((1, rows.shape[0]), ROW_REL, dtype=torch.long)
    candidates = CandidateState(torch.zeros_like(state))
    for _ in range(rounds):
        result = model.memory_reader(
            state,
            rows[:, 0].unsqueeze(0),
            rows[:, 1].unsqueeze(0),
            types,
            mask,
            torch.tensor([OPCODE_IDS["READ_P"]]),
            torch.tensor([0]),
            torch.tensor([SLOT_P]),
        )
        state = model.commit(
            state,
            candidates,
            result,
            torch.tensor([OPCODE_IDS["READ_P"]]),
            torch.tensor([SLOT_P]),
            torch.tensor([[True, False, False, False]]),
        )
    return state[0, SLOT_P]


def test_causal_frontier_one_read_cannot_follow_multiple_hops() -> None:
    dimension = 8
    model = UnifiedT1U0(dimension=dimension)
    model.memory_reader.attention_temperature = 40.0
    keys = torch.eye(dimension)[:5]
    rows = torch.stack((keys[:4], keys[1:5]), dim=1)
    for rounds in (1, 2, 3):
        prediction = run_pointer_chain(model, keys, keys, rows, rounds)
        assert torch.cosine_similarity(prediction.unsqueeze(0), keys[4].unsqueeze(0)).item() < 0.99
    prediction = run_pointer_chain(model, keys, keys, rows, 4)
    assert torch.cosine_similarity(prediction.unsqueeze(0), keys[4].unsqueeze(0)).item() > 0.99


def test_reader_invariant_to_row_permutation_and_relabelled_disjoint_mapping() -> None:
    dimension = 16
    model = UnifiedT1U0(dimension=dimension)
    model.memory_reader.attention_temperature = 40.0
    key_vectors = torch.eye(dimension)

    def make_rows(ids: list[int], permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.stack((key_vectors[ids[:-1]], key_vectors[ids[1:]]), dim=1)
        return rows[permutation], key_vectors[ids[-1]]

    ids_a = [0, 1, 2, 3, 4]
    ids_b = [8, 9, 10, 11, 12]
    for ids, permutation in ((ids_a, torch.tensor([2, 0, 3, 1])), (ids_b, torch.tensor([1, 3, 0, 2]))):
        rows, target = make_rows(ids, permutation)
        prediction = run_pointer_chain(model, key_vectors[ids], key_vectors[ids], rows, 4)
        assert torch.cosine_similarity(prediction.unsqueeze(0), target.unsqueeze(0)).item() > 0.99


def test_all_tasks_share_reader_core_and_typed_decoders() -> None:
    model = UnifiedT1U0()
    components = {task: model.components_for_task(task) for task in model.TASKS}
    assert {id(value["memory_reader"]) for value in components.values()} == {id(model.memory_reader)}
    assert {id(value["core"]) for value in components.values()} == {id(model.core)}
    assert {id(value["commit"]) for value in components.values()} == {id(model.commit)}
    assert id(components["pointer_chasing"]["decoder"]) == id(components["multi_hop"]["decoder"])
    assert id(components["associative_recall"]["decoder"]) == id(components["variable_binding"]["decoder"])
    assert id(components["pointer_chasing"]["decoder"]) != id(components["associative_recall"]["decoder"])
