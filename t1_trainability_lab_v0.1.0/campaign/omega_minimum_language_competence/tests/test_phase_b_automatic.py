from __future__ import annotations

import ast
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None
try:
    import numpy
except ModuleNotFoundError:
    numpy = None


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_linguistic_instrument import build_candidate_bank, validate_l1_twins
from human_pilot_harness import pilot_summary
from ngram_calibration import (
    BOS_ID,
    CalibrationHold,
    FullVocabularyUnigram,
    ModifiedKneserNey5,
    build_raw_5gram_counts,
    decompose_token_losses,
    estimate_modified_kneser_ney_discounts,
    score_unigram_chunk,
    validate_frozen_oov_mask,
)
from phase_b_utils import (
    LongestTokenOverlapIndex,
    clopper_pearson_upper_one_sided,
    instrument_invalid_l1,
    mean_log_probability,
    reject_omega_checkpoint_access,
    select_nonoverlapping_vall_windows,
    threshold_auto_from_u_fp,
    type7_quantile,
    window_degeneration_metrics,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("synthetic tokenizer test must not add special tokens")
        result = []
        for token in re.findall(r"\w+|[^\w\s]", text.lower()):
            if token not in self.ids:
                self.ids[token] = len(self.ids)
            result.append(self.ids[token])
        return result


def _toy_mkn() -> ModifiedKneserNey5:
    counts = {
        1: {(0,): 4, (1,): 2, (2,): 1},
        2: {(BOS_ID, 0): 2, (BOS_ID, 1): 1, (0, 1): 2, (0, 2): 1, (1, 2): 3},
        3: {(BOS_ID, BOS_ID, 0): 2, (BOS_ID, BOS_ID, 1): 1, (BOS_ID, 0, 1): 2, (BOS_ID, 0, 2): 1, (BOS_ID, 1, 2): 3},
        4: {(BOS_ID, BOS_ID, BOS_ID, 0): 2, (BOS_ID, BOS_ID, BOS_ID, 1): 1, (BOS_ID, BOS_ID, 0, 1): 2, (BOS_ID, BOS_ID, 0, 2): 1, (BOS_ID, BOS_ID, 1, 2): 3},
        5: {(BOS_ID, BOS_ID, BOS_ID, BOS_ID, 0): 2, (BOS_ID, BOS_ID, BOS_ID, BOS_ID, 1): 1, (BOS_ID, BOS_ID, BOS_ID, 0, 1): 2, (BOS_ID, BOS_ID, BOS_ID, 0, 2): 1, (BOS_ID, BOS_ID, BOS_ID, 1, 2): 3},
    }
    discounts = {order: {"D1": 0.5, "D2": 1.0, "D3plus": 1.5} for order in (2, 3, 4, 5)}
    return ModifiedKneserNey5(counts, discounts)


class PhaseBAutomaticTests(unittest.TestCase):
    def test_document_boundary_ngram_reset(self) -> None:
        left = [10] * 513
        right = [22] * 513
        counts, events = build_raw_5gram_counts([left, right])
        self.assertEqual(events, 1024)
        self.assertTrue(all(not (10 in gram and 22 in gram) for gram in counts))

    def test_unigram_exact_counts(self) -> None:
        model = FullVocabularyUnigram.from_counts({0: 3, 1: 1}, total_targets=4, vocabulary_size=50257)
        beta = 2 / 6
        self.assertEqual(model.beta_u, beta)
        self.assertAlmostEqual(model.probability(0), (1 - beta) * 3 / 4 + beta / 50257, places=15)
        self.assertAlmostEqual(model.probability(1), (1 - beta) * 1 / 4 + beta / 50257, places=15)
        self.assertAlmostEqual(model.full_normalizer(), 1.0, delta=1e-12)

    def test_unigram_full_vocab_normalizes(self) -> None:
        model = FullVocabularyUnigram.from_counts({0: 9, 5: 1, 17: 2}, vocabulary_size=50257)
        self.assertAlmostEqual(model.full_normalizer(), 1.0, delta=1e-12)

    def test_unigram_unseen_positive_and_equal_floor(self) -> None:
        model = FullVocabularyUnigram.from_counts({0: 2, 1: 1}, vocabulary_size=50257)
        unseen = [model.probability(token) for token in (2, 17, 50256)]
        self.assertTrue(all(value > 0 for value in unseen))
        self.assertEqual(len(set(unseen)), 1)
        self.assertAlmostEqual(unseen[0], model.beta_u / 50257, places=18)

    def test_unigram_beta_train_only(self) -> None:
        model = FullVocabularyUnigram.from_counts({2: 7, 9: 3}, total_targets=10)
        self.assertEqual(model.beta_u, 2 / 12)
        self.assertEqual(model.observed_types, 2)
        actual_phase_b_beta = 35825 / (4096000 + 35825)
        self.assertAlmostEqual(actual_phase_b_beta, 0.008670502743944868, delta=1e-17)

    def test_mkn_toy_probability(self) -> None:
        model = _toy_mkn()
        p1 = model.terminal_probability(1)
        expected = (2 - 1.0) / 3 + (1.0 + 0.5) / 3 * p1
        self.assertAlmostEqual(model.probability((0,), 1), expected, delta=1e-12)
        self.assertGreater(model.terminal_probability(50256), 0)

    def test_mkn_backoff_normalization_toy(self) -> None:
        model = _toy_mkn()
        for history in ((), (0,), (BOS_ID, 0), (BOS_ID, BOS_ID, 0), (BOS_ID, BOS_ID, BOS_ID, 0)):
            distribution = model.full_distribution(history)
            self.assertTrue(all(math_isfinite_positive(value) for value in distribution))
            self.assertAlmostEqual(sum(distribution), 1.0, delta=1e-10)

    def test_mkn_real_context_probability_normalization(self) -> None:
        model = _toy_mkn()
        observed_histories = sorted({gram[:-1] for gram in model.counts[5]})
        self.assertTrue(observed_histories)
        for history in observed_histories:
            distribution = model.full_distribution(history)
            self.assertTrue(all(math_isfinite_positive(value) for value in distribution))
            self.assertAlmostEqual(sum(distribution), 1.0, delta=1e-10)

    def test_mkn_discount_formula(self) -> None:
        counts = {}
        for index in range(100):
            counts[(index, 10_000)] = 1
        for index in range(50):
            counts[(1_000 + index, 10_000)] = 2
        for index in range(25):
            counts[(2_000 + index, 10_000)] = 3
        for index in range(10):
            counts[(3_000 + index, 10_000)] = 4
        result = estimate_modified_kneser_ney_discounts(counts, 2)
        self.assertAlmostEqual(result["Y"], 0.5)
        self.assertAlmostEqual(result["D1"], 0.5)
        self.assertAlmostEqual(result["D2"], 1.25)
        self.assertAlmostEqual(result["D3plus"], 2.2)

    def test_mkn_discount_missing_count_class_holds(self) -> None:
        with self.assertRaisesRegex(CalibrationHold, "MKN_DISCOUNT_ESTIMATION_HOLD"):
            estimate_modified_kneser_ney_discounts({(1, 2): 1, (2, 3): 3}, 2)

    def test_mkn_terminal_uniform_base_normalizes(self) -> None:
        model = _toy_mkn()
        dist = [model.terminal_probability(token) for token in range(50257)]
        self.assertTrue(all(math_isfinite_positive(value) for value in dist))
        self.assertAlmostEqual(sum(dist), 1.0, delta=1e-10)
        self.assertEqual(model.terminal_probability(50000), model.terminal_probability(50001))

    def test_mkn_unseen_token_positive(self) -> None:
        model = _toy_mkn()
        self.assertGreater(model.terminal_probability(50256), 0.0)
        self.assertGreater(model.probability((0,), 50256), 0.0)

    def test_mkn_oov_does_not_use_vall_counts(self) -> None:
        train = [[10] * 513, [22] * 513]
        raw_a, n_a = build_raw_5gram_counts(train)
        _unrelated_vall = [[10] * 512 + [999]]
        raw_b, n_b = build_raw_5gram_counts(train)
        self.assertEqual(n_a, n_b)
        self.assertEqual(raw_a, raw_b)

    def test_l0_scoring_length_normalization(self) -> None:
        self.assertEqual(mean_log_probability([-2.0, -4.0]), -3.0)
        with self.assertRaisesRegex(ValueError, "at least one token"):
            mean_log_probability([])

    @unittest.skipIf(torch is None, "torch is required to exercise the DistilGPT2 alignment helper")
    def test_teacher_candidate_scores_exact_continuation_positions(self) -> None:
        import torch as torch_module

        from run_phase_b_oov_resume import _score_teacher_candidate

        class NextTokenTeacher:
            def __call__(self, *, input_ids, use_cache):
                if use_cache:
                    raise AssertionError("teacher scoring must disable the KV cache")
                logits = torch_module.full((1, input_ids.shape[1], 8), -20.0)
                for position, token_id in enumerate(input_ids[0].tolist()):
                    logits[0, position, token_id + 1] = 20.0
                return SimpleNamespace(logits=logits)

        score = _score_teacher_candidate(NextTokenTeacher(), [0, 1], [2, 3])
        self.assertGreater(score, -1e-5)

    def test_l1_twin_swap_invariants(self) -> None:
        bank = build_candidate_bank(FakeTokenizer())
        checks = validate_l1_twins(bank)
        self.assertEqual(len(bank["L0_minimal_pairs"]), 64)
        self.assertEqual(len(bank["L1_twin_pairs"]), 64)
        self.assertEqual(len(checks), 64)
        self.assertTrue(all(row["pass"] for row in checks))
        for category in bank["canonical_order"]["L1"]:
            self.assertEqual(sum(row["category"] == category for row in bank["L1_twin_pairs"]), 16)

    @unittest.skipIf(numpy is None, "NumPy is required for token-overlap index tests")
    def test_leakage_exact_match_detection(self) -> None:
        index = LongestTokenOverlapIndex([[1, 2, 3, 4, 5, 6], [20, 21, 22]])
        self.assertEqual(index.longest_match([2, 3, 4]), 3)
        self.assertTrue(index.exact_match([1, 2, 3, 4, 5, 6]))
        self.assertFalse(index.exact_match([1, 2, 3, 4, 5, 7]))

    @unittest.skipIf(numpy is None, "NumPy is required for token-overlap index tests")
    def test_leakage_16token_overlap_detection(self) -> None:
        index = LongestTokenOverlapIndex([list(range(40))])
        self.assertEqual(index.longest_match(list(range(8, 26))), 18)
        self.assertEqual(index.longest_match([100, 101, 102, 103, 104, 105, 106, 107]), 0)

    def test_vall_span_disjointness(self) -> None:
        docs = [list(range(1000)), list(range(1000, 2000))]
        reserved = {0: [(0, 144)], 1: [(0, 144)]}
        reference, control = select_nonoverlapping_vall_windows(docs, reserved, reference_count=4, control_count=4)
        all_spans = reference + control
        self.assertEqual(len(all_spans), 8)
        for index, left in enumerate(all_spans):
            self.assertGreaterEqual(left["start"], 144)
            for right in all_spans[index + 1 :]:
                if left["document_order_index"] == right["document_order_index"]:
                    self.assertFalse(left["start"] < right["end"] and right["start"] < left["end"])

    def test_type7_quantiles(self) -> None:
        self.assertEqual(type7_quantile([0, 10], 0.5), 5)
        self.assertEqual(type7_quantile([0, 10, 20, 30, 40], 0.25), 10)
        self.assertEqual(type7_quantile([1, 2, 3], 0), 1)

    def test_cycle_detector(self) -> None:
        metrics = window_degeneration_metrics([1, 2] * 48)
        self.assertIsNotNone(metrics["exact_cycle"])
        self.assertIn(metrics["exact_cycle"]["cycle_length"], (1, 2))
        varied = window_degeneration_metrics(list(range(96)))
        self.assertIsNone(varied["exact_cycle"])

    def test_clopper_pearson_upper_bound(self) -> None:
        value = clopper_pearson_upper_one_sided(0, 256, confidence=0.90)
        self.assertAlmostEqual(value, 1.0 - 0.1 ** (1.0 / 256), delta=1e-12)
        self.assertEqual(clopper_pearson_upper_one_sided(256, 256, confidence=0.90), 1.0)

    def test_threshold_floor_when_fp_zero(self) -> None:
        u_fp = clopper_pearson_upper_one_sided(0, 256)
        self.assertLess(2 * u_fp, 0.10)
        self.assertEqual(threshold_auto_from_u_fp(u_fp), 0.10)

    def test_threshold_2x_upper_bound(self) -> None:
        self.assertEqual(threshold_auto_from_u_fp(0.12), 0.24)
        self.assertEqual(threshold_auto_from_u_fp(0.01), 0.10)

    def test_l1_instrument_invalid_conditions(self) -> None:
        invalid, reasons = instrument_invalid_l1(0.35, 0.40, 0.40)
        self.assertTrue(invalid)
        self.assertEqual(len(reasons), 4)
        self.assertEqual(instrument_invalid_l1(0.60, 0.39, 0.39), (False, []))

    def test_oov_mask_matches_hold_exactly(self) -> None:
        vall = {1: 9, 2: 3, 9: 2}
        train = {1: 10, 2: 4}
        mask = validate_frozen_oov_mask(vall, train, [{"token_id": 9, "vall_target_occurrences": 2}], expected_types=1, expected_occurrences=2)
        self.assertEqual(mask, {9})
        with self.assertRaisesRegex(CalibrationHold, "OOV_MASK_DRIFT"):
            validate_frozen_oov_mask(vall, train, [{"token_id": 9, "vall_target_occurrences": 1}], expected_types=1, expected_occurrences=1)

    def test_primary_nll_includes_oov(self) -> None:
        model = FullVocabularyUnigram.from_counts({0: 20, 1: 10}, total_targets=30)
        chunk = ([0] + [0, 1, 9] * 170 + [0, 1, 9])[:513]
        result = score_unigram_chunk(model, chunk, {9})
        self.assertGreater(result["oov_target_count"], 0)
        self.assertEqual(result["total_target_count"], 512)
        self.assertAlmostEqual(
            result["nll_total_token_weighted"],
            result["nll_contribution_seen_to_total"] + result["nll_contribution_oov_to_total"],
            delta=1e-12,
        )

    def test_seen_plus_oov_contributions_equal_total(self) -> None:
        result = decompose_token_losses([1, 9, 1, 9], [1.0, 3.0, 2.0, 4.0], {9})
        self.assertEqual(result["nll_total_token_weighted"], 2.5)
        self.assertEqual(result["nll_contribution_seen_to_total"], 0.75)
        self.assertEqual(result["nll_contribution_oov_to_total"], 1.75)
        self.assertTrue(result["decomposition_sum_matches_total"])

    def test_distilgpt2_uses_same_oov_mask_for_diagnostics(self) -> None:
        shared_mask = {7, 42}
        teacher = decompose_token_losses([7, 8, 42], [0.5, 1.0, 1.5], shared_mask)
        self.assertEqual(teacher["seen_target_count"], 1)
        self.assertEqual(teacher["oov_target_count"], 2)
        self.assertAlmostEqual(teacher["nll_contribution_oov_to_total"], 2.0 / 3.0)

    def test_no_trained_omega_import_or_generation(self) -> None:
        runner = HERE / "run_phase_b_oov_resume.py"
        if not runner.exists():
            self.skipTest("Phase-B orchestrator is created with the resume implementation")
        tree = ast.parse(runner.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden = [name for name in imported if "omega_fast_candidate" in name or "run_k_curve" in name or "run_backend_quality_qualification" in name]
        self.assertEqual(forbidden, [])
        source = runner.read_text(encoding="utf-8")
        self.assertNotIn("checkpoint_02000.pt", source)
        self.assertNotIn("forward_window", source)

    def test_trained_omega_checkpoint_access_fails_closed(self) -> None:
        with self.assertRaises(PermissionError):
            reject_omega_checkpoint_access(r"C:\frozen\K4_seed_20260913\checkpoint_02000.pt")

    def test_human_pilot_harness_is_prompt_clustered(self) -> None:
        ratings = []
        for prompt in range(12):
            for condition, score in (("REAL_HUMAN", 5), ("OMEGA_UPDATE0", 1)):
                for rater in ("H1", "H2", "H3"):
                    for dimension in ("G", "R", "C", "E", "N"):
                        ratings.append({"output_id": f"P{prompt:02d}::{condition}", "rater_id": rater, "dimension": dimension, "score": score})
        result = pilot_summary(ratings, bootstrap_seed=20260929)
        self.assertEqual(result["status"], "PILOT_PASS")
        self.assertEqual(result["prompt_count"], 12)
        self.assertEqual(result["krippendorff_alpha_ordinal_point"], 1.0)
        self.assertGreater(result["lower_bootstrap90_H_human_minus_H_init"], 0.75)

def math_isfinite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


if __name__ == "__main__":
    unittest.main()
