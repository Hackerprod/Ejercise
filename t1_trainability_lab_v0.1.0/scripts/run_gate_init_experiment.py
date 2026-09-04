"""Run the 10-run gate-init diagnostic experiment without touching campaign/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_run.py"
SEEDS = (101, 202, 303, 404, 505)
REASON = "Gate2/Gate6 diagnosis: compare neutral sigmoid gate init 0.5 against original 0.1"


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def experiment_grid() -> list[tuple[str, int, int]]:
    return [("multi_hop", rounds, seed) for rounds in (1, 4) for seed in SEEDS]


def run_one(task: str, rounds: int, seed: int, runs_root: Path, log: Path) -> dict[str, object]:
    run_dir = runs_root / task / "shared" / f"D64_S4_R{rounds}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAIN),
        "--task", task,
        "--variant", "shared",
        "--dimension", "64",
        "--slots", "4",
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--init-gate-probability", "0.5",
        "--experiment-reason", REASON,
        "--output-dir", str(run_dir),
    ]
    started = time.perf_counter()
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"START {' '.join(command)}\n")
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "gate_init_p05")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = args.output_dir / "runs"
    log = args.output_dir / "experiment.log"
    runs = experiment_grid()
    save_json(
        args.output_dir / "plan.json",
        {
            "total_runs": len(runs),
            "task": "multi_hop",
            "variant": "shared",
            "dimension": 64,
            "slots": 4,
            "rounds": [1, 4],
            "seeds": list(SEEDS),
            "init_gate_probability": 0.5,
            "experiment_reason": REASON,
        },
    )
    completed = []
    started = time.perf_counter()
    for index, (task, rounds, seed) in enumerate(runs, start=1):
        result = run_one(task, rounds, seed, runs_root, log)
        completed.append(result)
        save_json(
            args.output_dir / "state.json",
            {"status": "running", "completed": index, "total": len(runs), "elapsed_seconds": time.perf_counter() - started, "last_run": result},
        )
    save_json(args.output_dir / "state.json", {"status": "completed", "completed": len(completed), "total": len(runs), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({"status": "completed", "runs": len(completed), "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
