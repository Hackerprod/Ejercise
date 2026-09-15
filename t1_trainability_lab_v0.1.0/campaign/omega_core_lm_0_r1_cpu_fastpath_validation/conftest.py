"""Step 1 report hook kept outside test modules so pytest registers it."""

from __future__ import annotations

from test_equivalence import _write_report


def pytest_sessionfinish(session, exitstatus: int) -> None:
    _write_report(session, exitstatus)
