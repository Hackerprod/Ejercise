import pytest
import torch

from t1_trainability import InputAdapter, OutputReader, TokenVocabulary
from t1_trainability.data import (
    OUTPUT_CARDINALITIES,
    TASK_NAMES,
    encode_batch,
    generate_examples,
)


@pytest.mark.parametrize("task", TASK_NAMES)
def test_task_examples_encode_with_exact_targets(task: str) -> None:
    examples = generate_examples(task, "train", 8, seed=7)  # type: ignore[arg-type]
    vocabulary = TokenVocabulary()
    input_ids, mask, query_ids, targets = encode_batch(examples, vocabulary)

    assert input_ids.ndim == 2
    assert mask.shape == input_ids.shape
    assert query_ids.shape == (len(examples),)
    assert targets.shape == (len(examples),)
    assert int(targets.min()) >= 0
    assert int(targets.max()) < OUTPUT_CARDINALITIES[task]


def test_length_generalization_split_ranges() -> None:
    train = generate_examples("length_generalization", "train", 100, seed=9)
    test = generate_examples("length_generalization", "test", 100, seed=9)

    assert {int(row.metadata["hop_count"]) for row in train} <= {1, 2, 3}
    assert {int(row.metadata["hop_count"]) for row in test} <= {4, 5, 6}


def test_adapters_produce_task_logits() -> None:
    examples = generate_examples("associative_recall", "train", 2, seed=11)
    vocabulary = TokenVocabulary()
    input_ids, mask, query_ids, _ = encode_batch(examples, vocabulary)
    input_adapter = InputAdapter(len(vocabulary), dimension=64, slots=4, max_length=64)
    output_reader = OutputReader(len(vocabulary), dimension=64)

    state = input_adapter(input_ids, mask)
    logits = output_reader(state, query_ids, "associative_recall")

    assert state.shape == (2, 4, 64)
    assert logits.shape == (2, 32)
    assert torch.isfinite(logits).all()
