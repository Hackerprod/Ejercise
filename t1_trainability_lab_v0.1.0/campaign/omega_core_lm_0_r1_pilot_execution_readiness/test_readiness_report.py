import json
from pathlib import Path


REPORT = Path(__file__).with_name("readiness_report.json")


def test_report_requires_unexecuted_image_protocol() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    protocol = report["container_verification"]
    assert report["schema"] == "omega-core-lm-0-r1-pilot-execution-readiness-v2"
    assert report["source_commit"] == "2901e58834bc0d8d8225ccaf51ca136901468698"
    assert report["R1_identity"] == "269a4d79b5a1e6df8c230962e2b8c9e237095f18"
    assert protocol["required_before_publish"] is True
    assert protocol["required_after_pull_by_digest_and_clean_boot"] is True
    assert protocol["executed"] is False
    assert protocol["remote_execution_claim"] is False
    assert protocol["status"] == "PENDING_NOT_EXECUTED"
    assert protocol["required_checks"] == [
        "source_commit",
        "R1_identity",
        "runner",
        "requirements.lock",
        "model.py",
        "Dockerfile snapshot",
    ]


def test_report_preserves_cost_and_storage_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    plan = report["execution_plan"]
    assert plan["campaign_cost_formula"] == "C_campaign = t_update × p_GPU × N_updates + evaluation + I/O + overhead"
    assert plan["seed_total_compute"] == "118.9h/4.95d"
    assert plan["serial_total_compute"] == "24.8d"
    assert plan["gpu_count_by_seed"] == 5
    assert plan["same_total_compute_cost_as_serial"] is True
    assert plan["fifteen_gpu_policy"] == "none; five identical GPUs by seed only"
    assert plan["T_total_formula"] == "5*[20000*t_shared_K1 + 20000*t_shared_K4 + 20000*t_untied_K4] seconds"
    assert report["budget"]["validation_evaluations"] == "41*15=615"
    assert report["budget"]["q4t3_vol_read_only_sizing"] == "50GB target; pending remote measurement"
