import json
import time
from pathlib import Path

from benchmarks.phase_e_stage import (
    compute_controller_budget,
    count_train_transitions,
    load_stage_stories,
    split_stage_stories,
)
from mrdl.language import EmbeddingTable, MemoryLevel, MRDLLanguagePipeline, SparseController


records = load_stage_stories("/root/mrdl/data/TinyStories-valid.txt", 5000, 8)
train, test, _ = split_stage_stories(records, 1729)
subset = train[:300]
transitions = count_train_transitions(subset)
budget = compute_controller_budget(transitions)
vocab = sorted({token for record in records for token in record["tokens"]})
original = SparseController.sparse_backprop


def run(tag: str, lr: float, clip: float | None = None) -> dict:
    def patched(self, active_records, error, learning_rate=0.001, signal_vector=None):
        if not active_records or any(record.level is not MemoryLevel.M2 for record in active_records):
            return 0
        signals = tuple(signal_vector or (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0))
        for index, signal in enumerate(signals):
            self.weights[index] -= lr * error * signal
            if clip is not None:
                self.weights[index] = max(-clip, min(clip, self.weights[index]))
        self.last_signal = signals
        self.updates += 1
        return 1

    SparseController.sparse_backprop = patched
    started = time.perf_counter()
    pipeline = MRDLLanguagePipeline(
        EmbeddingTable.random_frozen(vocab, 50, 1729),
        beam_width=4,
        propagation_rounds=4,
        vsa_dimension=8,
    )
    pipeline.observe_training(subset)
    pipeline.promote_supported_edges(
        minimum_support=2,
        limit=budget["resolved"]["promotion_limit"],
    )
    pipeline.train_controller(
        subset,
        max_steps=budget["resolved"]["controller_steps"],
        promotion_interval=budget["resolved"]["promotion_interval"],
        promotion_limit=budget["resolved"]["promotion_limit"],
    )
    metrics = pipeline.evaluate(test, "FULL")
    weights = list(pipeline.controller.weights)
    return {
        "tag": tag,
        "learning_rate": lr,
        "clip": clip,
        "updates": pipeline.controller_updates,
        "promoted": pipeline.promoted_edges,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "weights": weights,
        "max_abs_weight": max(abs(value) for value in weights),
        "elapsed_seconds": time.perf_counter() - started,
    }


try:
    results = [
        run("lr_1e-3", 1e-3),
        run("lr_1e-4", 1e-4),
        run("lr_1e-5", 1e-5),
        run("lr_1e-3_clip_1", 1e-3, 1.0),
    ]
finally:
    SparseController.sparse_backprop = original

Path("/root/mrdl/results/controller_stability_subset.json").write_text(
    json.dumps(
        {
            "subset_stories": len(subset),
            "subset_transitions": transitions,
            "test_stories": len(test),
            "budget": budget,
            "results": results,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
