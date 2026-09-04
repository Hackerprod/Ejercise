"""Run remaining SU-4 training and frozen SU-5 evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_sequential_update_su4.py"
EVALUATE = ROOT / "scripts" / "evaluate_sequential_update_su5.py"
SEEDS = (202, 303, 404, 505)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], output_dir: Path, log_name: str) -> int:
    with (output_dir / log_name).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(" ".join(command) + "\n")
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "campaign" / "sequential_update_su4_su5_replication")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--samples-per-h", type=int, default=2000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        seed_root = args.output_root / f"seed_{seed}"
        train_dir = seed_root / "su4"
        eval_dir = seed_root / "su5"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        train_command = [sys.executable, str(TRAIN), "--seed", str(seed), "--max-steps", str(args.max_steps), "--output-dir", str(train_dir)]
        train_returncode = run(train_command, train_dir, "train.log")
        train_final_path = train_dir / "final.json"
        train_final = json.loads(train_final_path.read_text(encoding="utf-8")) if train_final_path.exists() else {}
        train_final["process_returncode"] = train_returncode
        save_json(train_final_path, train_final)
        if train_returncode != 0 or train_final.get("status") != "completed" or train_final.get("finite") is not True:
            raise RuntimeError(f"SU-4 training failed for seed {seed}")
        eval_command = [sys.executable, str(EVALUATE), "--seed", str(seed), "--checkpoint", str(train_dir / "final.pt"), "--output-dir", str(eval_dir), "--samples-per-h", str(args.samples_per_h)]
        eval_returncode = run(eval_command, eval_dir, "evaluate.log")
        eval_final_path = eval_dir / "final.json"
        eval_final = json.loads(eval_final_path.read_text(encoding="utf-8")) if eval_final_path.exists() else {}
        eval_final["process_returncode"] = eval_returncode
        save_json(eval_final_path, eval_final)
        if eval_returncode != 0 or eval_final.get("status") != "completed" or eval_final.get("finite") is not True:
            raise RuntimeError(f"SU-5 evaluation failed for seed {seed}")
        results[str(seed)] = {"su4": train_final, "su5": eval_final}
    summary = {"status": "completed", "seeds": list(SEEDS), "results": results, "elapsed_seconds": time.perf_counter() - started}
    save_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
