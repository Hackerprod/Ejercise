"""Step 1 report hook kept outside test modules so pytest registers it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from test_equivalence import _write_report


def _is_equivalence_call(report: Any) -> bool:
    path = getattr(report, "path", None)
    if path is None:
        path = getattr(report, "fspath", None)
    return report.when == "call" and path is not None and Path(str(path)).name == "test_equivalence.py"


class _Step1ExecutionTracker:
    def __init__(self, config: Any) -> None:
        self.config = config

    def pytest_runtest_logreport(self, report: Any) -> None:
        if _is_equivalence_call(report):
            self.config._step1_equivalence_executed = True


def pytest_configure(config: Any) -> None:
    config._step1_equivalence_executed = False
    config.pluginmanager.register(_Step1ExecutionTracker(config), "step1_execution_tracker")


def pytest_sessionfinish(session, exitstatus: int) -> None:
    if getattr(session.config, "_step1_equivalence_executed", False):
        _write_report(session, exitstatus)
