"""Rating-analysis utilities only; this module does not generate any model text."""

from __future__ import annotations

import math
import random
from statistics import median
from typing import Any, Mapping, Sequence

from phase_b_utils import type7_quantile


DIMENSIONS = ("G", "R", "C", "E", "N")


def krippendorff_alpha_ordinal(ratings: Sequence[Mapping[str, Any]]) -> float:
    """Krippendorff alpha with ordinal distances from pooled marginal midranks."""
    units: dict[tuple[str, str], list[int]] = {}
    frequencies = {value: 0 for value in range(1, 6)}
    for row in ratings:
        value = int(row["score"])
        if value not in frequencies:
            raise ValueError("human rating must be an integer from 1 through 5")
        unit = (str(row["output_id"]), str(row["dimension"]))
        units.setdefault(unit, []).append(value)
        frequencies[value] += 1
    if not units or any(len(values) < 2 for values in units.values()):
        raise ValueError("each (output, dimension) unit needs at least two independent ratings")
    total = sum(frequencies.values())
    if total < 2:
        raise ValueError("alpha requires at least two ratings")
    cumulative = 0
    midrank: dict[int, float] = {}
    for category in range(1, 6):
        midrank[category] = (cumulative + frequencies[category] / 2.0) / total
        cumulative += frequencies[category]

    def distance(left: int, right: int) -> float:
        return (midrank[left] - midrank[right]) ** 2

    observed_sum = 0.0
    observed_weight = 0
    for values in units.values():
        pair_sum = 0.0
        for i, left in enumerate(values):
            for j, right in enumerate(values):
                if i != j:
                    pair_sum += distance(left, right)
        observed_sum += pair_sum / (len(values) - 1)
        observed_weight += len(values)
    observed = observed_sum / observed_weight
    expected_numerator = 0.0
    for left, left_count in frequencies.items():
        for right, right_count in frequencies.items():
            if left != right:
                expected_numerator += left_count * right_count * distance(left, right)
    expected = expected_numerator / (total * (total - 1))
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def _rater_composites(ratings: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, dict[str, list[int]]]] = {}
    for row in ratings:
        output_id = str(row["output_id"])
        rater_id = str(row["rater_id"])
        dimension = str(row["dimension"])
        if dimension not in DIMENSIONS:
            raise ValueError(f"unknown rating dimension: {dimension}")
        scores.setdefault(output_id, {}).setdefault(rater_id, {}).setdefault(dimension, []).append(int(row["score"]))
    result: dict[str, dict[str, float]] = {}
    for output_id, raters in scores.items():
        result[output_id] = {}
        for rater_id, dimensions in raters.items():
            if set(dimensions) != set(DIMENSIONS) or any(len(dimensions[name]) != 1 for name in DIMENSIONS):
                raise ValueError(f"rater {rater_id} must score every dimension exactly once for {output_id}")
            result[output_id][rater_id] = sum(dimensions[name][0] for name in DIMENSIONS) / len(DIMENSIONS)
    return result


def _cluster_bootstrap_lower(differences: Sequence[float], *, seed: int, replicates: int = 10_000) -> float:
    if not differences:
        raise ValueError("prompt-cluster bootstrap requires at least one paired prompt")
    rng = random.Random(seed)
    n = len(differences)
    means = [sum(float(differences[rng.randrange(n)]) for _ in range(n)) / n for _ in range(replicates)]
    return type7_quantile(means, 0.10)


def pilot_summary(ratings: Sequence[Mapping[str, Any]], *, bootstrap_seed: int = 20260929) -> dict[str, Any]:
    alpha = krippendorff_alpha_ordinal(ratings)
    composites = _rater_composites(ratings)
    by_prompt: dict[str, dict[str, float]] = {}
    for output_id, raters in composites.items():
        prompt_id, condition = output_id.split("::", 1)
        by_prompt.setdefault(prompt_id, {})[condition] = median(raters.values())
    if not by_prompt or any(set(conditions) != {"REAL_HUMAN", "OMEGA_UPDATE0"} for conditions in by_prompt.values()):
        raise ValueError("pilot requires paired REAL_HUMAN and OMEGA_UPDATE0 outputs for each prompt")
    differences = [by_prompt[prompt]["REAL_HUMAN"] - by_prompt[prompt]["OMEGA_UPDATE0"] for prompt in sorted(by_prompt)]
    lower = _cluster_bootstrap_lower(differences, seed=bootstrap_seed)
    passed = alpha >= 0.40 and lower > 0.75
    return {
        "status": "PILOT_PASS" if passed else "HUMAN_INSTRUMENT_INVALID",
        "rater_count": len({str(row["rater_id"]) for row in ratings}),
        "prompt_count": len(by_prompt),
        "krippendorff_alpha_ordinal_point": alpha,
        "lower_bootstrap90_H_human_minus_H_init": lower,
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": bootstrap_seed,
        "pilot_valid": passed,
    }


def main_human_summary(ratings: Sequence[Mapping[str, Any]], *, bootstrap_seed: int = 20260929) -> dict[str, Any]:
    alpha = krippendorff_alpha_ordinal(ratings)
    composites = _rater_composites(ratings)
    output_medians = {output_id: median(rater_values.values()) for output_id, rater_values in composites.items()}
    by_condition: dict[str, list[tuple[str, float]]] = {}
    for output_id, value in output_medians.items():
        pieces = output_id.split("::")
        if len(pieces) < 2:
            raise ValueError(f"main output id must encode a condition and optional seed: {output_id}")
        by_condition.setdefault(pieces[1], []).append((output_id, value))
    if alpha < 0.40:
        return {
            "status": "HUMAN_INCONCLUSIVE_ONE_RESCUE_ALLOWED",
            "krippendorff_alpha_ordinal_point": alpha,
            "human_result_can_pass": False,
            "human_result_can_fail": False,
        }
    for condition in ("REAL_HUMAN", "OMEGA_UPDATE0", "OMEGA_TRAINED"):
        if condition not in by_condition:
            raise ValueError(f"missing human condition {condition}")
    human_by_prompt = {output.split("::")[0]: value for output, value in by_condition["REAL_HUMAN"]}
    init_by_prompt = {output.split("::")[0]: value for output, value in by_condition["OMEGA_UPDATE0"]}
    omega_by_prompt = {output.split("::")[0]: value for output, value in by_condition["OMEGA_TRAINED"]}
    shared_prompts = sorted(set(human_by_prompt) & set(init_by_prompt) & set(omega_by_prompt))
    if not shared_prompts:
        raise ValueError("human comparison has no common prompt clusters")
    delta_init = [omega_by_prompt[prompt] - init_by_prompt[prompt] for prompt in shared_prompts]
    human_anchor = [human_by_prompt[prompt] - init_by_prompt[prompt] for prompt in shared_prompts]
    lower_delta = _cluster_bootstrap_lower(delta_init, seed=bootstrap_seed + 1)
    lower_anchor = _cluster_bootstrap_lower(human_anchor, seed=bootstrap_seed + 2)
    omega_rows = by_condition["OMEGA_TRAINED"]
    seed_values: dict[str, list[float]] = {}
    for output_id, value in omega_rows:
        seed = output_id.split("::")[-1]
        seed_values.setdefault(seed, []).append(value)
    seed_means = {seed: sum(values) / len(values) for seed, values in seed_values.items()}
    catastrophic = 0
    omega_output_ids = {output_id for output_id, _ in omega_rows}
    for output_id in omega_output_ids:
        catastrophic_dims = 0
        for dimension in DIMENSIONS:
            dimension_ratings = [
                int(row["score"])
                for row in ratings
                if str(row["output_id"]) == output_id and str(row["dimension"]) == dimension
            ]
            if dimension_ratings and median(dimension_ratings) == 1:
                catastrophic_dims += 1
        catastrophic += catastrophic_dims >= 2
    catastrophic_rate = catastrophic / len(omega_output_ids)
    omega_h_values = [value for _, value in omega_rows]
    lower_omega = _cluster_bootstrap_lower(omega_h_values, seed=bootstrap_seed)
    human_mean = sum(value for _, value in by_condition["REAL_HUMAN"]) / len(by_condition["REAL_HUMAN"])
    init_mean = sum(value for _, value in by_condition["OMEGA_UPDATE0"]) / len(by_condition["OMEGA_UPDATE0"])
    passed = (
        lower_omega >= 3.0
        and all(value >= 2.75 for value in seed_means.values())
        and catastrophic_rate <= 0.10
        and lower_delta >= 0.50
        and lower_anchor > 0.75
    )
    return {
        "status": "HUMAN_PASS" if passed else "HUMAN_FAIL",
        "krippendorff_alpha_ordinal_point": alpha,
        "H_human_mean": human_mean,
        "H_init_mean": init_mean,
        "lower_bootstrap90_H_omega": lower_omega,
        "lower_bootstrap90_H_omega_minus_H_init": lower_delta,
        "lower_bootstrap90_H_human_minus_H_init": lower_anchor,
        "omega_seed_mean_H": seed_means,
        "human_catastrophic_count": catastrophic,
        "human_catastrophic_rate": catastrophic_rate,
        "prompt_clusters": len(shared_prompts),
        "pilot_rescue_used": len({str(row["rater_id"]) for row in ratings}) == 5,
    }
