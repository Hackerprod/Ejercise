"""Deterministic helpers for the non-OMEGA Phase-B calibration instruments."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - production installs provide NumPy for large overlap indexes
    np = None  # type: ignore[assignment]


WINDOW_TOKENS = 96
REPEAT_CYCLE_MAX_LENGTH = 8


def reject_omega_checkpoint_access(path: str | Path) -> None:
    """Fail closed if Phase B attempts to open an OMEGA checkpoint artifact."""
    candidate = Path(path)
    if re.fullmatch(r"checkpoint_\d+\.pt", candidate.name, flags=re.IGNORECASE) or "k_curve_primary" in str(candidate).lower():
        raise PermissionError(f"Phase B is forbidden from loading OMEGA checkpoint artifacts: {candidate}")


def mean_log_probability(token_log_probabilities: Sequence[float]) -> float:
    if not token_log_probabilities:
        raise ValueError("candidate continuation must have at least one token")
    if any(not math.isfinite(float(value)) for value in token_log_probabilities):
        raise ValueError("candidate continuation has a non-finite token log probability")
    return sum(float(value) for value in token_log_probabilities) / len(token_log_probabilities)


def type7_quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("Type-7 quantile requires nonempty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("Type-7 quantile input must be finite")
    h = (len(ordered) - 1) * quantile
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    if lower == upper:
        return ordered[lower]
    fraction = h - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _longest_cycle(tokens: Sequence[int]) -> dict[str, int] | None:
    values = [int(token) for token in tokens]
    for length in range(1, REPEAT_CYCLE_MAX_LENGTH + 1):
        needed = 4 * length
        for start in range(0, len(values) - needed + 1):
            pattern = values[start : start + length]
            repetitions = 1
            offset = start + length
            while offset + length <= len(values) and values[offset : offset + length] == pattern:
                repetitions += 1
                offset += length
            if repetitions >= 4:
                return {"cycle_length": length, "repetitions": repetitions, "start_token": start}
    return None


def window_degeneration_metrics(tokens: Sequence[int]) -> dict[str, Any]:
    values = [int(token) for token in tokens]
    if len(values) != WINDOW_TOKENS:
        raise ValueError(f"degeneration windows must be exactly {WINDOW_TOKENS} tokens")
    diversity: dict[str, float] = {}
    for order in (1, 2, 4):
        grams = [tuple(values[index : index + order]) for index in range(len(values) - order + 1)]
        diversity[f"distinct_{order}"] = len(set(grams)) / len(grams)
    fourgrams = [tuple(values[index : index + 4]) for index in range(len(values) - 3)]
    recurrence = Counter(fourgrams)
    return {
        **diversity,
        "max_4gram_recurrence": max(recurrence.values(), default=0),
        "exact_cycle": _longest_cycle(values),
        "finite": all(math.isfinite(value) for value in diversity.values()),
    }


def corpus_extreme_degenerate(metrics: dict[str, Any], reference_intervals: dict[str, Sequence[float]]) -> bool:
    violations = 0
    for metric in ("distinct_1", "distinct_2", "distinct_4", "max_4gram_recurrence"):
        lower, upper = reference_intervals[metric]
        value = float(metrics[metric])
        if value < float(lower) or value > float(upper):
            violations += 1
    cycle = metrics.get("exact_cycle")
    return violations >= 2 or (
        cycle is not None
        and int(cycle["cycle_length"]) <= REPEAT_CYCLE_MAX_LENGTH
        and int(cycle["repetitions"]) >= 4
    )


def clopper_pearson_upper_one_sided(successes: int, trials: int, confidence: float = 0.90) -> float:
    """Exact one-sided Clopper-Pearson upper confidence limit for a binomial rate."""
    k, n = int(successes), int(trials)
    if n <= 0 or not 0 <= k <= n or not 0.0 < confidence < 1.0:
        raise ValueError("invalid Clopper-Pearson arguments")
    if k == n:
        return 1.0
    alpha = 1.0 - confidence
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)

    def cdf_at(p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        term = (1.0 - p) ** n
        total = term
        ratio = p / (1.0 - p)
        for index in range(1, k + 1):
            term *= ((n - index + 1) / index) * ratio
            total += term
        return total

    lower = k / n
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if cdf_at(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def threshold_auto_from_u_fp(u_fp: float) -> float:
    return max(0.10, 2.0 * float(u_fp))


def instrument_invalid_l1(
    distilgpt2_full: float,
    distilgpt2_trunc5: float,
    kn5_full: float,
) -> tuple[bool, list[str]]:
    reasons = []
    if float(distilgpt2_full) < 0.60:
        reasons.append("DISTILGPT2_PAIR_SUCCESS_FULL_LT_0_60")
    if float(distilgpt2_trunc5) >= 0.40:
        reasons.append("DISTILGPT2_PAIR_SUCCESS_TRUNC5_GE_0_40")
    if float(distilgpt2_full) - float(distilgpt2_trunc5) <= 0.0:
        reasons.append("DISTILGPT2_FULL_MINUS_TRUNC5_LE_0")
    if float(kn5_full) >= 0.40:
        reasons.append("KN5_PAIR_SUCCESS_FULL_GE_0_40")
    return bool(reasons), reasons


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(int(start_a), int(start_b)) < min(int(end_a), int(end_b))


def select_nonoverlapping_vall_windows(
    documents: Sequence[Sequence[int]],
    reserved_spans: dict[int, Sequence[tuple[int, int]]],
    *,
    reference_count: int = 256,
    control_count: int = 256,
    window_tokens: int = WINDOW_TOKENS,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    candidates: list[dict[str, int]] = []
    for document_index, sequence in enumerate(documents):
        limit = len(sequence) - window_tokens + 1
        for start in range(0, max(0, limit), window_tokens):
            end = start + window_tokens
            if any(spans_overlap(start, end, left, right) for left, right in reserved_spans.get(document_index, ())):
                continue
            candidates.append({"document_order_index": document_index, "start": start, "end": end})
    if len(candidates) < reference_count + control_count:
        raise ValueError(
            f"VALL has only {len(candidates)} non-overlapping unreserved windows; "
            f"requires {reference_count + control_count}"
        )
    chosen = candidates[: reference_count + control_count]
    reference = chosen[:reference_count]
    control = chosen[reference_count:]
    all_rows = reference + control
    for index, left in enumerate(all_rows):
        for right in all_rows[index + 1 :]:
            if left["document_order_index"] == right["document_order_index"] and spans_overlap(left["start"], left["end"], right["start"], right["end"]):
                raise AssertionError("VALL degeneration windows overlap")
    return reference, control


class LongestTokenOverlapIndex:
    """Exact longest common contiguous token match against document-bounded sequences.

    A 64-bit rolling hash indexes 16/8/4/2/1-grams; every hash hit is verified
    against token IDs before extending, so collisions cannot cause false matches.
    Separating sequences prevents matches from crossing document boundaries.
    """

    _MASK = (1 << 64) - 1
    _BASE = 1_000_003

    def __init__(self, documents: Sequence[Sequence[int]]) -> None:
        self.documents = [tuple(int(value) for value in row) for row in documents]
        self._indexes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def _window_hash(cls, values: Sequence[int], start: int, length: int) -> int:
        result = 0
        for value in values[start : start + length]:
            result = (result * cls._BASE + int(value) + 1) & cls._MASK
        return result

    def _build_index(self, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np is None:
            raise RuntimeError("NumPy is required for deterministic token-overlap indexing")
        if length in self._indexes:
            return self._indexes[length]
        windows = sum(max(0, len(row) - length + 1) for row in self.documents)
        hashes = np.empty(windows, dtype=np.uint64)
        doc_ids = np.empty(windows, dtype=np.int32)
        starts = np.empty(windows, dtype=np.int32)
        write = 0
        power = pow(self._BASE, length - 1, 1 << 64)
        for document_id, row in enumerate(self.documents):
            if len(row) < length:
                continue
            rolling = self._window_hash(row, 0, length)
            for start in range(len(row) - length + 1):
                hashes[write] = rolling
                doc_ids[write] = document_id
                starts[write] = start
                write += 1
                if start + length < len(row):
                    old = (row[start] + 1) * power
                    rolling = ((rolling - old) * self._BASE + row[start + length] + 1) & self._MASK
        if write != windows:
            raise AssertionError("rolling-hash window count mismatch")
        order = np.argsort(hashes, kind="mergesort")
        index = (hashes[order], doc_ids[order], starts[order])
        self._indexes[length] = index
        return index

    def longest_match(self, query: Sequence[int]) -> int:
        pattern = tuple(int(value) for value in query)
        if not pattern:
            return 0
        max_length = len(pattern)
        levels = [value for value in (16, 8, 4, 2, 1) if value <= max_length]
        if not levels or levels[0] != max_length and max_length < 16:
            levels.insert(0, max_length)
        best = 0
        for length in levels:
            sorted_hashes, doc_ids, starts = self._build_index(length)
            for query_start in range(max_length - length + 1):
                query_hash = self._window_hash(pattern, query_start, length)
                left = int(np.searchsorted(sorted_hashes, query_hash, side="left"))
                right = int(np.searchsorted(sorted_hashes, query_hash, side="right"))
                for index in range(left, right):
                    document_id = int(doc_ids[index])
                    corpus_start = int(starts[index])
                    document = self.documents[document_id]
                    if document[corpus_start : corpus_start + length] != pattern[query_start : query_start + length]:
                        continue
                    query_left = query_start
                    corpus_left = corpus_start
                    while query_left > 0 and corpus_left > 0 and pattern[query_left - 1] == document[corpus_left - 1]:
                        query_left -= 1
                        corpus_left -= 1
                    query_right = query_start + length
                    corpus_right = corpus_start + length
                    while query_right < max_length and corpus_right < len(document) and pattern[query_right] == document[corpus_right]:
                        query_right += 1
                        corpus_right += 1
                    best = max(best, query_right - query_left)
            if best >= length:
                return best
        return best

    def exact_match(self, query: Sequence[int]) -> bool:
        return self.longest_match(query) == len(query) and bool(query)
