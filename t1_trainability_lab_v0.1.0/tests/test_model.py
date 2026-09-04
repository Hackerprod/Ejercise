import pytest
import torch

from t1_trainability import RecurrentCore


@pytest.mark.parametrize(
    ("variant", "slots", "rounds"),
    (
        ("single", 4, 1),
        ("shared", 4, 4),
        ("untied", 4, 4),
        ("vector-state", 1, 4),
    ),
)
def test_baselines_forward_backward_without_nan(variant: str, slots: int, rounds: int) -> None:
    torch.manual_seed(0)
    model = RecurrentCore(dimension=64, slots=slots, rounds=rounds, variant=variant)  # type: ignore[arg-type]
    state = torch.randn(2, slots, 64, requires_grad=True)

    output, states = model(state, return_states=True)
    assert output.shape == state.shape
    assert len(states) == rounds + 1
    assert all(item.shape == state.shape for item in states)

    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(state.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_shared_and_untied_core_parameterization() -> None:
    shared = RecurrentCore(dimension=64, slots=4, rounds=4, variant="shared")
    untied = RecurrentCore(dimension=64, slots=4, rounds=4, variant="untied")

    assert len(shared.cores) == 1
    assert len(untied.cores) == 4


def test_unbatched_state_shape() -> None:
    model = RecurrentCore(dimension=128, slots=8, rounds=2, variant="shared")
    state = torch.randn(8, 128)

    output = model(state)

    assert output.shape == state.shape
