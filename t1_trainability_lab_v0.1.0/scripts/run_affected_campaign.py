"""Rerun only corrected T1-B tasks while preserving clean task artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from run_campaign import SEEDS, TASKS, pilot_grid, run_one, save_json  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
AFFECTED = ("multi_hop", "length_generalization", "variable_binding")


def affected_grid() -> list[tuple[str, str, int, int, int]]:
    pilot_configs = [(variant, slots, rounds) for _task, variant, slots, rounds in pilot_grid() if _task == TASKS[0]]
    pilot = [(task, variant, slots, rounds, 101) for task in AFFECTED for variant, slots, rounds in pilot_configs]
    selected = list(pilot)

    for seed in SEEDS[1:]:
        selected.append(("multi_hop", "shared", 4, 4, seed))
        selected.append(("multi_hop", "shared", 4, 1, seed))
        selected.append(("multi_hop", "untied", 4, 4, seed))
        selected.append(("multi_hop", "shared", 1, 4, seed))
        selected.append(("multi_hop", "shared", 8, 4, seed))
        selected.append(("length_generalization", "shared", 4, 4, seed))
        selected.append(("length_generalization", "shared", 4, 8, seed))
        selected.append(("variable_binding", "shared", 4, 4, seed))
        selected.append(("variable_binding", "untied", 4, 4, seed))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = args.output_dir / "runs"
    log = args.output_dir / "affected_campaign.log"
    runs = affected_grid()
    if len(runs) != 60:
        raise RuntimeError(f"expected 60 affected runs, got {len(runs)}")

    # Replace only corrected task artifacts. Clean associative/sequential runs
    # remain untouched under the same campaign root.
    for task in AFFECTED:
        task_dir = runs_root / task
        if task_dir.exists():
            shutil.rmtree(task_dir)

    started = time.perf_counter()
    completed = 0
    save_json(args.output_dir / "affected_plan.json", {"total_runs": len(runs), "runs": runs, "tasks": AFFECTED, "dimension": 64})
    for index, run in enumerate(runs, start=1):
        if time.perf_counter() - started > 10 * 60 * 60:
            save_json(args.output_dir / "affected_state.json", {"status": "over_10_hours", "completed": completed, "total": len(runs), "elapsed_seconds": time.perf_counter() - started})
            raise RuntimeError("affected campaign exceeded 10 hours")
        run_one(run, runs_root, log, index, len(runs))
        completed = index
        save_json(args.output_dir / "affected_state.json", {"status": "running", "completed": completed, "total": len(runs), "elapsed_seconds": time.perf_counter() - started, "pilot_complete": completed >= 24})
        if completed == 24:
            save_json(args.output_dir / "affected_pilot_complete.json", {"status": "pilot_passed", "runs": 24, "elapsed_seconds": time.perf_counter() - started})
    save_json(args.output_dir / "affected_state.json", {"status": "completed", "completed": completed, "total": len(runs), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({"status": "completed", "runs": completed, "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
