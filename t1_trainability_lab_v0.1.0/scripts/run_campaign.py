"""Run T1 pilot and gate-critical seed expansion sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_run.py"
TASKS = (
    "associative_recall",
    "multi_hop",
    "variable_binding",
    "sequential_update",
    "length_generalization",
)
SEEDS = (101, 202, 303, 404, 505)


def pilot_grid() -> list[tuple[str, str, int, int]]:
    configs = (
        ("single", 4, 1),
        ("shared", 4, 1),
        ("shared", 4, 4),
        ("shared", 4, 8),
        ("shared", 1, 4),
        ("shared", 8, 4),
        ("untied", 4, 4),
        ("vector-state", 1, 4),
    )
    return [(task, variant, slots, rounds) for task in TASKS for variant, slots, rounds in configs]


def full_grid() -> list[tuple[str, str, int, int, int]]:
    pilot = pilot_grid()
    selected: set[tuple[str, str, int, int, int]] = {
        (*config, 101) for config in pilot
    }
    # Gate 1: canonical shared model, five seeds, every task.
    for task in TASKS:
        for seed in SEEDS:
            selected.add((task, "shared", 4, 4, seed))
    # Gate 2: depth utility on multi-hop, five seeds.
    for seed in SEEDS:
        selected.add(("multi_hop", "shared", 4, 1, seed))
        selected.add(("multi_hop", "shared", 4, 4, seed))
    # Gate 3: sharing ceiling on four principal tasks, five seeds.
    for task in TASKS[:4]:
        for seed in SEEDS:
            selected.add((task, "shared", 4, 4, seed))
            selected.add((task, "untied", 4, 4, seed))
    # Gate 4: slot ablation on associative recall and multi-hop, five seeds.
    for task in TASKS[:2]:
        for seed in SEEDS:
            selected.add((task, "shared", 1, 4, seed))
            selected.add((task, "shared", 4, 4, seed))
            selected.add((task, "shared", 8, 4, seed))
    # Gate 5: OOD length generalization, five seeds.
    for seed in SEEDS:
        selected.add(("length_generalization", "shared", 4, 8, seed))
    return sorted(selected)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_one(run: tuple[str, str, int, int, int], runs_root: Path, log: Path, index: int, total: int) -> dict[str, object]:
    task, variant, slots, rounds, seed = run
    run_dir = runs_root / task / variant / f"D64_S{slots}_R{rounds}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAIN),
        "--task", task,
        "--variant", variant,
        "--dimension", "64",
        "--slots", str(slots),
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--output-dir", str(run_dir),
    ]
    started = time.perf_counter()
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n[{index}/{total}] START {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    final_path = run_dir / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
    final["process_returncode"] = result.returncode
    final["run_elapsed_seconds"] = elapsed
    save_json(final_path, final)
    if result.returncode != 0 or final.get("status") != "completed" or final.get("finite") is not True:
        raise RuntimeError(f"run failed or became non-finite: {run_dir}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = args.output_dir / "runs"
    log = args.output_dir / "campaign.log"
    runs = full_grid()
    state_path = args.output_dir / "state.json"
    started = time.perf_counter()
    completed: list[dict[str, object]] = []
    save_json(args.output_dir / "plan.json", {"total_runs": len(runs), "runs": runs, "dimension": 64})
    for index, run in enumerate(runs, start=1):
        elapsed = time.perf_counter() - started
        if elapsed > 10 * 60 * 60:
            save_json(state_path, {"status": "over_10_hours", "completed": len(completed), "total": len(runs), "elapsed_seconds": elapsed})
            raise RuntimeError("campaign exceeded 10 hours")
        result = run_one(run, runs_root, log, index, len(runs))
        completed.append(result)
        save_json(
            state_path,
            {
                "status": "running",
                "completed": index,
                "total": len(runs),
                "elapsed_seconds": time.perf_counter() - started,
                "last_run": result,
                "pilot_complete": index >= 40,
            },
        )
        if index == 40:
            save_json(args.output_dir / "pilot_complete.json", {"status": "pilot_passed", "runs": 40, "elapsed_seconds": time.perf_counter() - started})
    save_json(
        state_path,
        {"status": "completed", "completed": len(completed), "total": len(runs), "elapsed_seconds": time.perf_counter() - started},
    )
    print(json.dumps({"status": "completed", "runs": len(completed), "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
