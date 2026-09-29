import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paired_ci import T_CRITICAL_DF4_975, depth_gate_passes, paired_deltas, paired_student_t_ci


def test_five_paired_deltas_and_fixed_student_t_interval() -> None:
    k1 = [3.00, 3.00, 3.00, 3.25, 3.00]
    k4 = [2.90, 2.80, 2.95, 3.10, 3.00]
    result = paired_student_t_ci(k1, k4)

    assert paired_deltas(k1, k4).tolist() == pytest.approx([0.10, 0.20, 0.05, 0.15, 0.00])
    assert result["n"] == 5
    assert result["mean_delta"] == pytest.approx(0.10)
    expected_std = math.sqrt(0.025 / 4.0)
    expected_se = expected_std / math.sqrt(5.0)
    expected_margin = T_CRITICAL_DF4_975 * expected_se
    assert result["sample_std"] == pytest.approx(expected_std)
    assert result["standard_error"] == pytest.approx(expected_se)
    assert result["lower"] == pytest.approx(0.10 - expected_margin)
    assert result["upper"] == pytest.approx(0.10 + expected_margin)
    assert result["t_critical"] == 2.7764451051977987
    assert "gate" not in result


def test_depth_gate_is_separate_from_confidence_interval() -> None:
    assert depth_gate_passes(0.05)
    assert not depth_gate_passes(0.049999)


def test_fixed_df_four_ci_rejects_non_pilot_sample_size() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        paired_student_t_ci([1.0, 1.0], [0.9, 0.9])
