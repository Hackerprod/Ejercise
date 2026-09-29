"""Deterministic full-vocabulary unigram and modified KN-5 baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


GPT2_VOCAB_SIZE = 50257
TARGETS_PER_CHUNK = 512
BOS_ID = -1  # context-only sentinel, never an output token


class CalibrationHold(RuntimeError):
    """A required probability/discount is undefined; callers must stop."""


@dataclass(frozen=True)
class FullVocabularyUnigram:
    counts: Mapping[int, int]
    total_targets: int
    vocabulary_size: int
    observed_types: int
    beta_u: float

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[int, int],
        *,
        total_targets: int | None = None,
        vocabulary_size: int = GPT2_VOCAB_SIZE,
    ) -> "FullVocabularyUnigram":
        clean = {int(token): int(count) for token, count in counts.items() if int(count) > 0}
        if vocabulary_size != GPT2_VOCAB_SIZE:
            raise ValueError("UNIGRAM_FULLVOCAB_UNIFORM_BASE_V1 is frozen to GPT-2 V=50257")
        if any(token < 0 or token >= vocabulary_size for token in clean):
            raise ValueError("unigram training counts contain token ids outside GPT-2 vocabulary")
        observed_total = sum(clean.values())
        n = observed_total if total_targets is None else int(total_targets)
        if n <= 0 or observed_total != n:
            raise ValueError(f"unigram count total mismatch: counts={observed_total}, N={n}")
        t = len(clean)
        if t <= 0:
            raise CalibrationHold("UNIGRAM_FULLVOCAB_UNIFORM_BASE_V1 has no observed token types")
        beta = t / (n + t)
        return cls(clean, n, vocabulary_size, t, beta)

    def probability(self, token_id: int) -> float:
        token_id = int(token_id)
        if not 0 <= token_id < self.vocabulary_size:
            raise ValueError(f"token id outside fixed vocabulary: {token_id}")
        return (1.0 - self.beta_u) * self.counts.get(token_id, 0) / self.total_targets + self.beta_u / self.vocabulary_size

    def full_normalizer(self) -> float:
        seen_mass = math.fsum(self.probability(token_id) for token_id in self.counts)
        unseen_count = self.vocabulary_size - self.observed_types
        return seen_mass + unseen_count * self.beta_u / self.vocabulary_size


def estimate_modified_kneser_ney_discounts(
    counts: Mapping[tuple[int, ...], int], order: int
) -> dict[str, float | int]:
    """Chen-Goodman modified discounts from this order's count-of-counts only."""
    if order not in (2, 3, 4, 5):
        raise ValueError("MKN discounts are estimated for orders 2 through 5")
    n1 = sum(int(value) == 1 for value in counts.values())
    n2 = sum(int(value) == 2 for value in counts.values())
    n3 = sum(int(value) == 3 for value in counts.values())
    n4 = sum(int(value) == 4 for value in counts.values())
    if n1 == 0 or n2 == 0 or n3 == 0:
        raise CalibrationHold(
            f"MKN_DISCOUNT_ESTIMATION_HOLD: order={order}; undefined count-of-counts n1={n1}, n2={n2}, n3={n3}, n4={n4}"
        )
    y = n1 / (n1 + 2.0 * n2)
    d1 = 1.0 - 2.0 * y * n2 / n1
    d2 = 2.0 - 3.0 * y * n3 / n2
    d3 = 3.0 - 4.0 * y * n4 / n3
    if not (0.0 < d1 < 1.0 and 0.0 < d2 < 2.0 and 0.0 <= d3 <= 3.0):
        raise CalibrationHold(
            f"MKN_DISCOUNT_ESTIMATION_HOLD: order={order}; D1={d1}, D2={d2}, D3+={d3} outside count-class bounds"
        )
    return {"order": order, "n1": n1, "n2": n2, "n3": n3, "n4": n4, "Y": y, "D1": d1, "D2": d2, "D3plus": d3}


def build_raw_5gram_counts(presentations: Iterable[Sequence[int]]) -> tuple[Counter[tuple[int, ...]], int]:
    """Count target-ending 5-grams, resetting histories at every chunk presentation."""
    counts: Counter[tuple[int, ...]] = Counter()
    target_events = 0
    for presentation_index, source_seq in enumerate(presentations):
        source = [int(value) for value in source_seq]
        if len(source) != 513:
            raise ValueError(f"training presentation {presentation_index} is not a complete 513-token chunk")
        for target_position in range(1, 513):
            history = source[max(0, target_position - 4) : target_position]
            padded = [BOS_ID] * (4 - len(history)) + history
            counts[tuple(padded + [source[target_position]])] += 1
            target_events += 1
    return counts, target_events


def _continuation_counts(raw_5grams: Mapping[tuple[int, ...], int]) -> dict[int, dict[tuple[int, ...], int]]:
    counts: dict[int, Mapping[tuple[int, ...], int]] = {5: raw_5grams}
    upper_types: Iterable[tuple[int, ...]] = counts[5].keys()
    for order in (4, 3, 2, 1):
        lower: Counter[tuple[int, ...]] = Counter()
        # Each upper-order type contributes one distinct left continuation to
        # its suffix. Repeated presentations do not inflate continuation counts.
        for gram in upper_types:
            if len(gram) != order + 1:
                raise AssertionError("MKN continuation recursion received an unexpected n-gram order")
            lower[tuple(gram[1:])] += 1
        counts[order] = dict(lower)
        upper_types = counts[order].keys()
    return counts


class ModifiedKneserNey5:
    """Interpolated MKN-5 with a full-vocabulary uniform terminal base."""

    def __init__(
        self,
        counts_by_order: Mapping[int, Mapping[tuple[int, ...], int]],
        discounts_by_order: Mapping[int, Mapping[str, float | int]],
        *,
        vocabulary_size: int = GPT2_VOCAB_SIZE,
    ) -> None:
        if vocabulary_size != GPT2_VOCAB_SIZE:
            raise ValueError("MKN5_FULLVOCAB_UNIFORM_BASE_V1 is frozen to GPT-2 V=50257")
        self.vocabulary_size = vocabulary_size
        # Keep the large order-5 counter by reference; this model never mutates counts.
        self.counts = {order: counts_by_order[order] for order in range(1, 6)}
        self.discounts = {order: dict(discounts_by_order[order]) for order in range(2, 6)}
        self.history_totals: dict[int, dict[tuple[int, ...], int]] = {}
        self.history_lambdas: dict[int, dict[tuple[int, ...], float]] = {}
        for order in range(2, 6):
            stats: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
            for gram, raw_count in self.counts[order].items():
                count = int(raw_count)
                if len(gram) != order or count <= 0:
                    raise ValueError(f"invalid effective order-{order} n-gram count")
                row = stats[tuple(gram[:-1])]
                row[0] += count
                if count == 1:
                    row[1] += 1
                elif count == 2:
                    row[2] += 1
                elif count >= 3:
                    row[3] += 1
            self.history_totals[order] = {history: values[0] for history, values in stats.items()}
            d1 = float(self.discounts[order]["D1"])
            d2 = float(self.discounts[order]["D2"])
            d3 = float(self.discounts[order]["D3plus"])
            self.history_lambdas[order] = {
                history: (d1 * values[1] + d2 * values[2] + d3 * values[3]) / values[0]
                for history, values in stats.items()
            }
        terminal = self.counts[1]
        self.c_total = sum(int(value) for value in terminal.values())
        self.t_c = len(terminal)
        if self.c_total <= 0 or self.t_c <= 0:
            raise CalibrationHold("MKN_DISCOUNT_ESTIMATION_HOLD: terminal continuation counts are empty")
        self.beta_kn = self.t_c / (self.c_total + self.t_c)

    @classmethod
    def from_raw_5grams(cls, raw_5grams: Mapping[tuple[int, ...], int]) -> "ModifiedKneserNey5":
        counts = _continuation_counts(raw_5grams)
        discounts = {order: estimate_modified_kneser_ney_discounts(counts[order], order) for order in (2, 3, 4, 5)}
        return cls(counts, discounts)

    @classmethod
    def from_presentations(cls, presentations: Iterable[Sequence[int]]) -> tuple["ModifiedKneserNey5", dict[str, Any]]:
        raw_5grams, events = build_raw_5gram_counts(presentations)
        model = cls.from_raw_5grams(raw_5grams)
        metadata = {
            "target_events": events,
            "raw_5gram_types": len(raw_5grams),
            "effective_ngram_types_by_order": {str(order): len(model.counts[order]) for order in range(1, 6)},
            "count_of_counts_and_discounts_by_order": {str(order): model.discounts[order] for order in (2, 3, 4, 5)},
            "C_total": model.c_total,
            "T_C": model.t_c,
            "beta_KN": model.beta_kn,
            "terminal_base": "(1-beta_KN)*C(w)/C_total + beta_KN/V, V=50257",
        }
        return model, metadata

    def terminal_probability(self, token_id: int) -> float:
        token_id = int(token_id)
        if not 0 <= token_id < self.vocabulary_size:
            raise ValueError(f"token id outside GPT-2 vocabulary: {token_id}")
        c = self.counts[1].get((token_id,), 0)
        return (1.0 - self.beta_kn) * c / self.c_total + self.beta_kn / self.vocabulary_size

    def probability(self, history: Sequence[int], token_id: int) -> float:
        token_id = int(token_id)
        if not 0 <= token_id < self.vocabulary_size:
            raise ValueError(f"token id outside GPT-2 vocabulary: {token_id}")
        history_tuple = tuple(int(value) for value in history)
        order = min(5, len(history_tuple) + 1)
        return self._probability(order, history_tuple, token_id)

    def _probability(self, order: int, history: tuple[int, ...], token_id: int) -> float:
        if order <= 1:
            return self.terminal_probability(token_id)
        history = tuple(history[-(order - 1) :])
        total = self.history_totals[order].get(history, 0)
        if total <= 0:
            return self._probability(order - 1, history[1:] if history else (), token_id)
        count = self.counts[order].get(history + (token_id,), 0)
        if count == 1:
            discount = float(self.discounts[order]["D1"])
        elif count == 2:
            discount = float(self.discounts[order]["D2"])
        elif count >= 3:
            discount = float(self.discounts[order]["D3plus"])
        else:
            discount = 0.0
        direct = max(float(count) - discount, 0.0) / total
        backoff = self.history_lambdas[order].get(history, 0.0)
        return direct + backoff * self._probability(order - 1, history[1:] if history else (), token_id)

    def full_distribution(self, history: Sequence[int]) -> list[float]:
        return [self.probability(history, token_id) for token_id in range(self.vocabulary_size)]

    def score_chunk(self, tokens: Sequence[int], oov_ids: set[int]) -> dict[str, Any]:
        source = [int(value) for value in tokens]
        if len(source) != 513:
            raise ValueError("MKN5 VALL scoring expects a complete 513-token chunk")
        total_sum = 0.0
        seen_sum = 0.0
        oov_sum = 0.0
        seen_count = 0
        oov_count = 0
        for target_position in range(1, 513):
            history = source[max(0, target_position - 4) : target_position]
            history = [BOS_ID] * (4 - len(history)) + history
            target = source[target_position]
            probability = self.probability(history, target)
            if probability <= 0.0 or not math.isfinite(probability):
                raise CalibrationHold(f"MKN5_NONFINITE_OR_ZERO_PROBABILITY token={target}")
            loss = -math.log(probability)
            total_sum += loss
            if target in oov_ids:
                oov_sum += loss
                oov_count += 1
            else:
                seen_sum += loss
                seen_count += 1
        total_count = TARGETS_PER_CHUNK
        total_nll = total_sum / total_count
        return {
            "nll_total_token_weighted": total_nll,
            "ppl_total": math.exp(total_nll),
            "nll_seen_targets": seen_sum / seen_count if seen_count else None,
            "nll_oov_targets": oov_sum / oov_count if oov_count else None,
            "nll_contribution_seen_to_total": seen_sum / total_count,
            "nll_contribution_oov_to_total": oov_sum / total_count,
            "seen_target_count": seen_count,
            "oov_target_count": oov_count,
            "total_target_count": total_count,
            "finite": math.isfinite(total_nll) and math.isfinite(math.exp(total_nll)),
        }

    def score_continuation(self, prompt_ids: Sequence[int], continuation_ids: Sequence[int]) -> float:
        if not continuation_ids:
            raise ValueError("KN5 continuation scoring requires at least one token")
        context = [int(value) for value in prompt_ids]
        log_probability_sum = 0.0
        for target in continuation_ids:
            history = context[-4:]
            history = [BOS_ID] * (4 - len(history)) + history
            probability = self.probability(history, int(target))
            if probability <= 0.0 or not math.isfinite(probability):
                raise CalibrationHold("KN5 candidate probability is non-finite or zero")
            log_probability_sum += math.log(probability)
            context.append(int(target))
        return log_probability_sum / len(continuation_ids)


def score_unigram_chunk(model: FullVocabularyUnigram, tokens: Sequence[int], oov_ids: set[int]) -> dict[str, Any]:
    source = [int(value) for value in tokens]
    if len(source) != 513:
        raise ValueError("unigram VALL scoring expects a complete 513-token chunk")
    total_sum = 0.0
    seen_sum = 0.0
    oov_sum = 0.0
    seen_count = 0
    oov_count = 0
    for target in source[1:]:
        probability = model.probability(target)
        if probability <= 0.0 or not math.isfinite(probability):
            raise CalibrationHold(f"unigram non-finite/zero probability for token {target}")
        loss = -math.log(probability)
        total_sum += loss
        if target in oov_ids:
            oov_sum += loss
            oov_count += 1
        else:
            seen_sum += loss
            seen_count += 1
    total = TARGETS_PER_CHUNK
    nll = total_sum / total
    return {
        "nll_total_token_weighted": nll,
        "ppl_total": math.exp(nll),
        "nll_seen_targets": seen_sum / seen_count if seen_count else None,
        "nll_oov_targets": oov_sum / oov_count if oov_count else None,
        "nll_contribution_seen_to_total": seen_sum / total,
        "nll_contribution_oov_to_total": oov_sum / total,
        "seen_target_count": seen_count,
        "oov_target_count": oov_count,
        "total_target_count": total,
        "finite": math.isfinite(nll) and math.isfinite(math.exp(nll)),
    }


def validate_frozen_oov_mask(
    vall_target_counts: Mapping[int, int],
    train_target_counts: Mapping[int, int],
    frozen_oov_rows: Sequence[Mapping[str, int]],
    *,
    expected_types: int = 560,
    expected_occurrences: int = 1198,
) -> set[int]:
    observed = {
        int(token_id): int(count)
        for token_id, count in vall_target_counts.items()
        if int(token_id) not in train_target_counts
    }
    frozen = {int(row["token_id"]): int(row["vall_target_occurrences"]) for row in frozen_oov_rows}
    if observed != frozen:
        raise CalibrationHold("ABSOLUTE_CALIBRATION_OOV_MASK_DRIFT: recomputed OOV mask differs from the frozen HOLD artifact")
    if len(frozen) != expected_types or sum(frozen.values()) != expected_occurrences:
        raise CalibrationHold(
            f"ABSOLUTE_CALIBRATION_OOV_MASK_DRIFT: types={len(frozen)} occurrences={sum(frozen.values())}, "
            f"expected={expected_types}/{expected_occurrences}"
        )
    return set(frozen)


def decompose_token_losses(
    token_ids: Sequence[int],
    per_token_nll: Sequence[float],
    oov_ids: set[int],
) -> dict[str, Any]:
    if len(token_ids) != len(per_token_nll) or not token_ids:
        raise ValueError("token/loss vectors must have equal nonzero lengths")
    seen_sum = 0.0
    oov_sum = 0.0
    seen_count = 0
    oov_count = 0
    for token_id, nll in zip(token_ids, per_token_nll):
        value = float(nll)
        if not math.isfinite(value):
            raise FloatingPointError("per-token NLL must be finite")
        if int(token_id) in oov_ids:
            oov_sum += value
            oov_count += 1
        else:
            seen_sum += value
            seen_count += 1
    total_count = seen_count + oov_count
    total_nll = (seen_sum + oov_sum) / total_count
    return {
        "nll_total_token_weighted": total_nll,
        "nll_seen_targets": seen_sum / seen_count if seen_count else None,
        "nll_oov_targets": oov_sum / oov_count if oov_count else None,
        "nll_contribution_seen_to_total": seen_sum / total_count,
        "nll_contribution_oov_to_total": oov_sum / total_count,
        "seen_target_count": seen_count,
        "oov_target_count": oov_count,
        "total_target_count": total_count,
        "decomposition_sum_matches_total": math.isclose(
            total_nll,
            seen_sum / total_count + oov_sum / total_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
