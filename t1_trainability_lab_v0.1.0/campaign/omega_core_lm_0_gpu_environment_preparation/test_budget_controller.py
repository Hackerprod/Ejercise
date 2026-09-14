#!/usr/bin/env python3
"""Synthetic tests for simulation-only budget controller."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from budget_controller import (
    BudgetPolicy,
    Intent,
    PodObservation,
    PodState,
    SimulatedBudgetController,
)


def test_budget_and_confirmed_lifecycle() -> None:
    launched = datetime(2026, 9, 14, tzinfo=timezone.utc)
    controller = SimulatedBudgetController(BudgetPolicy(max_hours=2.0, hourly_rate=0.17))
    controller.observe(PodObservation(
        "pod-1", "RTX A4000 16GB", PodState.RUNNING, launched,
        "q4t3-vol", "6yrppoqpkz", "US-MO-2",
    ))
    decision = controller.evaluate(launched + timedelta(hours=1.5))
    assert decision.intent is Intent.CONTINUE
    assert decision.actions_authorized is False
    assert controller.evaluate(launched + timedelta(hours=2)).intent is Intent.REQUEST_STOP
    assert controller.request_stop().intent is Intent.REQUEST_STOP
    assert controller.evaluate(launched + timedelta(hours=2, minutes=1)).intent is Intent.AWAIT_STOP_CONFIRMATION
    assert controller.confirm_stop(PodState.STOPPED).intent is Intent.REQUEST_TERMINATE
    assert controller.request_terminate().intent is Intent.REQUEST_TERMINATE
    assert controller.evaluate(launched + timedelta(hours=2, minutes=2)).intent is Intent.AWAIT_TERMINATE_CONFIRMATION
    assert controller.confirm_terminate(PodState.TERMINATED).intent is Intent.CONTINUE


def test_replacement_and_binding_changes_are_rejected() -> None:
    launched = datetime(2026, 9, 14, tzinfo=timezone.utc)
    controller = SimulatedBudgetController()
    observation = PodObservation(
        "pod-1", "RTX A4000 16GB", PodState.RUNNING, launched,
        "q4t3-vol", "6yrppoqpkz", "US-MO-2",
    )
    controller.observe(observation)
    try:
        controller.observe(PodObservation(
            "pod-2", "RTX A4000 16GB", PodState.RUNNING, launched,
            "q4t3-vol", "6yrppoqpkz", "US-MO-2",
        ))
    except ValueError as exc:
        assert "replacement" in str(exc)
    else:
        raise AssertionError("replacement pod accepted")
    try:
        controller.observe(PodObservation(
            "pod-1", "RTX A4000 16GB", PodState.RUNNING, launched,
            "other-volume", "other-id", "US-MO-2",
        ))
    except ValueError as exc:
        assert "volume" in str(exc)
    else:
        raise AssertionError("binding change accepted")


if __name__ == "__main__":
    test_budget_and_confirmed_lifecycle()
    test_replacement_and_binding_changes_are_rejected()
    print("budget controller synthetic tests: PASS")
