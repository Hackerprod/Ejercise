"""T2-I2 frozen validity controls for complete-sequence slotting."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT, SOURCE_ROOT
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256
from t2_i2_semantic_writer import SemanticWriter, tensorize
from train_t2_i2 import output_for_seed, checkpoint_for_seed

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
SCORER = CAMPAIGN / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST = CAMPAIGN / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None:
        super().__init__(); self.core = core; self.condition = condition
    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(writer: SemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad(): return writer(token_ids, lengths)[0]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args()
    output = output_for_seed(args.seed) / "controls" if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for_seed(args.seed)
    writer = SemanticWriter(); writer.load_state_dict(torch.load(checkpoint, weights_only=False)["writer"], strict=True); writer.eval()
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")): raise RuntimeError("dispatcher received forbidden inputs")
    core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); core.eval(); manifest = load_base_manifests()["test"]; fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA); executor = load_executor(); ctrl1 = load_ctrl1(); payload = torch.load(SCORER, weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(payload["controller"], strict=True); scorer.eval(); episodes = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}
    aliases = (("AT_LEAST", (1, 0)), ("MINIMUM", (1, 0)), ("AVOID", (0, 1)), ("EXCLUDE", (0, 1)))
    position = {"samples": 0, "success": 0}; memory = {"samples": 0, "success": 0}; noop = {name: {"samples": 0, "success": 0} for name, _ in (("NONE", (0, 0)), *aliases)}
    cache: dict[str, Adapter] = {}
    def run(text: str, constraints: tuple[int, int], value: int) -> dict:
        if text not in cache: cache[text] = Adapter(core, encode(writer, text).sum(0))
        return run_learned(executor, ctrl1, scorer, cache[text], manifest, episodes[10], value if constraints == (1, 0) else 0, value if constraints == (0, 1) else 0, constraints)
    for value in range(32):
        for alias, constraints in aliases:
            operator = f"{alias} VALUE_{value}"; forms = (operator, f"{operator} AND NOOP VALUE_0", f"NOOP VALUE_0 AND {operator}")
            results = [run(text, constraints, value) for text in forms]; position["samples"] += 1; position["success"] += int(all(item["success"] for item in results) and len({tuple(event["action_name"] for event in item["events"]) for item in results}) == 1); memory["samples"] += 1; memory["success"] += int(results[1]["success"] and results[2]["success"] and tuple(event["action_name"] for event in results[1]["events"]) == tuple(event["action_name"] for event in results[2]["events"]))
        for name, constraints in (("NONE", (0, 0)),) + aliases:
            for variant in ((0, 1) if name == "NONE" else (1, 2)):
                for dummy in range(32):
                    text = f"NOOP VALUE_{dummy}" if name == "NONE" and variant == 0 else f"NOOP VALUE_{dummy} AND NOOP VALUE_{(dummy + 11) % 32}" if name == "NONE" else f"{name} VALUE_{value} AND NOOP VALUE_{dummy}" if variant == 1 else f"NOOP VALUE_{dummy} AND {name} VALUE_{value}"
                    result = run(text, constraints, value); noop[name]["samples"] += 1; noop[name]["success"] += int(result["success"])
    summary = {"position_invariance": position, "early_memory": memory, "noop_exhaustive": noop}
    passed = position["success"] == position["samples"] and memory["success"] == memory["samples"] and all(item["success"] == item["samples"] for item in noop.values())
    result = {"status": "passed" if passed else "failed", "task": "T2-I2", "phase": "controls", "training": False, "checkpoint": sha256(checkpoint), "summary": summary, "dispatcher_guard": "passed"}
    path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(path), "sha256": file_sha(path), **result}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
