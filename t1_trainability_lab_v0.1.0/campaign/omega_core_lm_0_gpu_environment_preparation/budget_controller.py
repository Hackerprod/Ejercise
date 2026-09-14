#!/usr/bin/env python3
"""Simulation-only budget guard for one future OMEGA GPU pod.

This module creates no pod, calls no provider API, changes no volume, and
cannot authorize spending. It emits lifecycle intents that a human/operator
may reconcile against an independently observed provider state. Before any
real rental, an external operator or authorized controller must execute the
RunPod action and confirm the provider state; this simulation never releases
billable resources by itself.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


DEFAULT_GPU_TYPE = "NVIDIA L4"
DEFAULT_MAX_HOURS = 2.0


class PodState(str, Enum):
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    TERMINATE_REQUESTED = "terminate_requested"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class Intent(str, Enum):
    CONTINUE = "continue"
    REQUEST_STOP = "request_stop"
    AWAIT_STOP_CONFIRMATION = "await_stop_confirmation"
    REQUEST_TERMINATE = "request_terminate"
    AWAIT_TERMINATE_CONFIRMATION = "await_terminate_confirmation"
    BLOCK = "block"


@dataclass(frozen=True)
class BudgetPolicy:
    hourly_rate: float
    gpu_type: str = DEFAULT_GPU_TYPE
    max_hours: float = DEFAULT_MAX_HOURS
    volume_name: str = "q4t3-vol"
    volume_id: str = "6yrppoqpkz"
    data_center: str = "US-MO-2"

    def __post_init__(self) -> None:
        if self.max_hours <= 0:
            raise ValueError("max_hours must be positive")
        if self.hourly_rate < 0:
            raise ValueError("hourly_rate cannot be negative")


@dataclass(frozen=True)
class PodObservation:
    pod_id: str
    gpu_type: str
    state: PodState
    launched_at: datetime
    volume_name: str
    volume_id: str
    data_center: str


@dataclass(frozen=True)
class BudgetDecision:
    intent: Intent
    reason: str
    elapsed_hours: float
    estimated_cost: float
    actions_authorized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "reason": self.reason,
            "elapsed_hours": round(self.elapsed_hours, 6),
            "estimated_cost": round(self.estimated_cost, 6),
            "actions_authorized": self.actions_authorized,
            "authority": "simulation-only; human confirmation required",
        }


class SimulatedBudgetController:
    """Single-pod state machine; all provider mutations remain external."""

    def __init__(self, policy: BudgetPolicy) -> None:
        self.policy = policy
        self._observation: PodObservation | None = None
        self._stop_confirmed = False
        self._terminate_confirmed = False

    @property
    def observation(self) -> PodObservation | None:
        return self._observation

    def observe(self, observation: PodObservation) -> None:
        if self._observation is not None and observation.pod_id != self._observation.pod_id:
            raise ValueError("replacement pods are forbidden by this controller")
        if observation.gpu_type != self.policy.gpu_type:
            raise ValueError("observed GPU differs from policy candidate")
        if (
            observation.volume_name != self.policy.volume_name
            or observation.volume_id != self.policy.volume_id
            or observation.data_center != self.policy.data_center
        ):
            raise ValueError("volume or region binding differs from prepared policy")
        self._observation = observation

    def request_stop(self) -> BudgetDecision:
        self._require_observation()
        if self._observation.state != PodState.RUNNING:
            raise ValueError("stop request requires independently observed running state")
        self._observation = _replace_state(self._observation, PodState.STOP_REQUESTED)
        return self._decision(Intent.REQUEST_STOP, "budget limit reached; operator may stop exactly this pod")

    def confirm_stop(self, observed_state: PodState) -> BudgetDecision:
        self._require_observation()
        if self._observation.state != PodState.STOP_REQUESTED:
            raise ValueError("stop confirmation requires an outstanding stop request")
        if observed_state != PodState.STOPPED:
            raise ValueError("stop confirmation requires observed stopped state")
        self._observation = _replace_state(self._observation, observed_state)
        self._stop_confirmed = True
        return self._decision(Intent.REQUEST_TERMINATE, "stopped state confirmed; operator may terminate exactly this pod")

    def request_terminate(self) -> BudgetDecision:
        self._require_observation()
        if not self._stop_confirmed or self._observation.state != PodState.STOPPED:
            raise ValueError("termination request requires confirmed stopped state")
        self._observation = _replace_state(self._observation, PodState.TERMINATE_REQUESTED)
        return self._decision(Intent.REQUEST_TERMINATE, "termination intent issued for exactly this pod")

    def confirm_terminate(self, observed_state: PodState) -> BudgetDecision:
        self._require_observation()
        if not self._stop_confirmed or self._observation.state != PodState.TERMINATE_REQUESTED:
            raise ValueError("termination confirmation requires outstanding termination request")
        if observed_state != PodState.TERMINATED:
            raise ValueError("termination confirmation requires observed terminated state")
        self._observation = _replace_state(self._observation, observed_state)
        self._terminate_confirmed = True
        return self._decision(Intent.CONTINUE, "termination confirmed; no replacement pod may be created")

    def evaluate(self, now: datetime) -> BudgetDecision:
        self._require_observation()
        if now < self._observation.launched_at:
            raise ValueError("now cannot precede launch time")
        if self._observation.state == PodState.RUNNING:
            decision = self._decision(Intent.CONTINUE, "within budget", now)
            if decision.elapsed_hours >= self.policy.max_hours:
                return self._decision(Intent.REQUEST_STOP, "budget limit reached; stop confirmation required", now)
            return decision
        if self._observation.state == PodState.STOP_REQUESTED:
            return self._decision(Intent.AWAIT_STOP_CONFIRMATION, "stop intent issued; inspect provider state", now)
        if self._observation.state == PodState.STOPPED:
            return self._decision(Intent.REQUEST_TERMINATE, "stopped state observed; termination confirmation required", now)
        if self._observation.state == PodState.TERMINATE_REQUESTED:
            return self._decision(Intent.AWAIT_TERMINATE_CONFIRMATION, "termination intent issued; inspect provider state", now)
        if self._observation.state == PodState.TERMINATED:
            return self._decision(Intent.CONTINUE, "pod terminated; replacement creation forbidden", now)
        return self._decision(Intent.BLOCK, "unknown provider state; do not mutate or relaunch", now)

    def _require_observation(self) -> None:
        if self._observation is None:
            raise ValueError("no independently observed pod registered")

    def _decision(self, intent: Intent, reason: str, now: datetime | None = None) -> BudgetDecision:
        self._require_observation()
        observed_now = now or datetime.now(timezone.utc)
        elapsed = max(0.0, (observed_now - self._observation.launched_at).total_seconds() / 3600.0)
        return BudgetDecision(intent, reason, elapsed, elapsed * self.policy.hourly_rate)


def _replace_state(observation: PodObservation, state: PodState) -> PodObservation:
    return PodObservation(
        observation.pod_id,
        observation.gpu_type,
        state,
        observation.launched_at,
        observation.volume_name,
        observation.volume_id,
        observation.data_center,
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", default="future-a4000-1")
    parser.add_argument("--state", choices=[state.value for state in PodState], default=PodState.RUNNING.value)
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--hourly-rate", type=float, required=True, help="required rate from current regional offer")
    parser.add_argument("--launched-at", default="2026-09-14T00:00:00+00:00")
    parser.add_argument("--now", default=None)
    args = parser.parse_args()
    controller = SimulatedBudgetController(BudgetPolicy(hourly_rate=args.hourly_rate, gpu_type=args.gpu_type))
    controller.observe(PodObservation(
        args.pod_id, args.gpu_type, PodState(args.state), _parse_time(args.launched_at),
        "q4t3-vol", "6yrppoqpkz", "US-MO-2",
    ))
    now = _parse_time(args.now) if args.now else datetime.now(timezone.utc)
    print(json.dumps(controller.evaluate(now).as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
