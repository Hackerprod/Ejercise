"""Gate 2: frozen full-panel behavioral cardinality audit for R2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, tensorize
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i2_r2 import checkpoint_for_seed, output_for_seed
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT, SOURCE_ROOT

ROOT = Path(__file__).resolve().parents[1]


def encode(writer: CompetitiveSemanticWriter, text: str) -> torch.Tensor:
    token_ids, lengths = tensorize([text])
    with torch.no_grad(): return writer(token_ids, lengths)[0]


def load_panels(seed: int) -> tuple[dict, str]:
    path = output_for_seed(seed) / "panel_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = output_for_seed(args.seed) / "cardinality" if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha = load_panels(args.seed); writer = CompetitiveSemanticWriter(); writer.load_state_dict(torch.load(checkpoint_for_seed(args.seed), weights_only=False)["writer"], strict=True); writer.eval(); supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); supervisor.eval(); labels = torch.load(SOURCE_ROOT / "train" / "labels.pt", weights_only=False); observations = torch.load(SOURCE_ROOT / "train" / "observations.pt", weights_only=False)
    def evaluate(kind: str, value: int, text: str) -> tuple[bool, dict[str, int]]:
        slots = encode(writer, text); panel = manifest["panels"][f"{kind}:{value}"]; row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], dtype=torch.long); features = observations["features"][row_ids]; targets = labels["action"][row_ids]; empty = supervisor(features, torch.zeros((len(row_ids), 32))).argmax(dim=-1); outcomes = {}
        for permutation_name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
            active = supervisor(features, slots[permutation[0]].expand(len(row_ids), -1)).argmax(dim=-1); inactive = supervisor(features, slots[permutation[1]].expand(len(row_ids), -1)).argmax(dim=-1); outcomes[permutation_name] = int(torch.equal(active, targets) and torch.equal(inactive, empty))
        return bool(outcomes["identity"] or outcomes["swap"]), outcomes
    results = []; total = passed = 0
    for kind, aliases in (("FLOOR", ("AT_LEAST", "MINIMUM")), ("AVOID", ("AVOID", "EXCLUDE"))):
        for alias in aliases:
            for value in range(32):
                for form in (1, 2, 3):
                    text = f"{alias} VALUE_{value}" if form == 1 else f"{alias} VALUE_{value} AND NOOP VALUE_0" if form == 2 else f"NOOP VALUE_0 AND {alias} VALUE_{value}"; ok, outcomes = evaluate(kind, value, text); results.append({"kind": kind, "alias": alias, "value": value, "form": form, "J": manifest["panels"][f"{kind}:{value}"]["J"], "passed": ok, "permutations": outcomes}); total += 1; passed += int(ok)
    for form, text in ((1, "NOOP VALUE_0"), (2, "NOOP VALUE_0 AND NOOP VALUE_11")):
        ok, outcomes = evaluate("NONE", 0, text); results.append({"kind": "NONE", "value": 0, "form": form, "J": manifest["panels"]["NONE:0"]["J"], "passed": ok, "permutations": outcomes}); total += 1; passed += int(ok)
    result = {"status": "passed" if passed == total else "failed", "task": "T2-I2-R2", "phase": "cardinality_behavioral_audit", "training": False, "panel_manifest_sha256": manifest_sha, "matching": {"samples": total, "success": passed}, "results": results, "criterion": "one permutation reproduces every panel action and other slot matches zero-conditioned behavior"}; path = output / "results.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
