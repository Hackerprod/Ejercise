from __future__ import annotations

import run_omega_native_runtime_p2r_opt2_diag as diagnostic


def test_diagnostic_masks_cover_requested_residual_blocks() -> None:
    assert diagnostic.VARIANTS["baseline"] == 0
    assert set(diagnostic.VARIANTS) == {
        "baseline",
        "attention_scores_softmax_mixing",
        "depth_embedding_pairwise",
        "rmsnorm_gates",
        "state_prelude",
        "history_buffer_bookkeeping",
        "depth_embedding_carry",
    }
    assert len(set(diagnostic.VARIANTS.values())) == len(diagnostic.VARIANTS)


def test_snapshot_metrics_exposes_worker_and_reduction_boundaries() -> None:
    snapshot = {
        "worker_start_ns": [100, 120, 110, 105],
        "worker_end_ns": [500, 520, 410, 405],
        "dispatch_start_ns": 50,
        "all_workers_done_ns": 550,
        "final_gradient_reduction_start_ns": 560,
        "final_gradient_reduction_end_ns": 610,
        "return_ns": 620,
    }
    metrics = diagnostic._snapshot_metrics(snapshot, "backward")
    assert metrics["worker_compute_max_ns"] == 400
    assert metrics["worker_compute_min_ns"] == 300
    assert metrics["worker_compute_mean_ns"] == 350.0
    assert metrics["serial_final_reduction_ns"] == 50
    assert metrics["dispatch_join_overhead_ns"] == 100
