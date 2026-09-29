"""Synthetic-only paired-TOST sensitivity; never reads candidate outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "preparation" / "statistical_sensitivity.json"
SEED = 20260924
PAIRED_SEEDS = 5
T_CRITICAL_90_DF4 = 2.131846786326383
MARGINS = {"delta_K": 0.02, "eta_depth": 0.01}
DISPERSIONS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
MONTE_CARLO_REPLICATES = 20000


def tost_equivalent(values: list[float], margin: float) -> tuple[bool, float, float, float, float]:
    if len(values) != PAIRED_SEEDS:
        raise ValueError("synthetic TOST requires five paired values")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = T_CRITICAL_90_DF4 * sd / math.sqrt(PAIRED_SEEDS)
    lower, upper = mean - half_width, mean + half_width
    return lower > -margin and upper < margin, mean, sd, lower, upper


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"synthetic sensitivity report is immutable: {OUTPUT}")
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    for endpoint, margin in MARGINS.items():
        for sigma in DISPERSIONS:
            mean_scenarios = {
                "zero": 0.0,
                "half_margin": 0.5 * margin,
                "at_margin": margin,
                "one_and_half_margin": 1.5 * margin,
            }
            counts = {name: 0 for name in mean_scenarios}
            for _ in range(MONTE_CARLO_REPLICATES):
                for scenario, true_mean in mean_scenarios.items():
                    sample = [rng.gauss(true_mean, sigma) for _ in range(PAIRED_SEEDS)]
                    passed, _mean, _sd, _lower, _upper = tost_equivalent(sample, margin)
                    counts[scenario] += int(passed)
            rows.append({
                "endpoint": endpoint,
                "margin_nats_per_token": margin,
                "synthetic_seed_sd": sigma,
                "expected_90pct_ci_half_width_at_that_sd": T_CRITICAL_90_DF4 * sigma / math.sqrt(PAIRED_SEEDS),
                "max_sd_for_zero_mean_ci_to_fit_margin": margin * math.sqrt(PAIRED_SEEDS) / T_CRITICAL_90_DF4,
                "monte_carlo_replicates": MONTE_CARLO_REPLICATES,
                "equivalence_pass_fraction_by_true_mean_scenario": {
                    name: count / MONTE_CARLO_REPLICATES for name, count in counts.items()
                },
            })
    report = {
        "schema": "omega-backend-quality-synthetic-tost-sensitivity-v1",
        "synthetic_only": True,
        "real_candidate_results_read": False,
        "real_training": False,
        "paired_seed_count": PAIRED_SEEDS,
        "paired_t_ci": {"coverage": 0.90, "df": 4, "critical_value": T_CRITICAL_90_DF4},
        "synthetic_rng_seed": SEED,
        "dispersion_grid_nats_per_token": list(DISPERSIONS),
        "scenarios": rows,
        "interpretation": "Sensitivity study only; margins unchanged regardless of simulated pass fraction.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS_SYNTHETIC_ONLY",
        "output": str(OUTPUT),
        "cases": len(rows),
        "eta_sd_0.02_half_width": next(row["expected_90pct_ci_half_width_at_that_sd"] for row in rows if row["endpoint"] == "eta_depth" and row["synthetic_seed_sd"] == 0.02),
        "eta_max_sd_for_zero_mean_pass": MARGINS["eta_depth"] * math.sqrt(PAIRED_SEEDS) / T_CRITICAL_90_DF4,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
