"""Generate legacy NB5 manifests or an explicit deterministic fresh battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "campaign" / "nb5_manifests"
LEGACY_SEEDS = (6801, 6802, 6803, 6804, 6805)
FRESH_SEEDS = (6901, 6902, 6903, 6904, 6905)
FRESH_BLOCKS = {
    6901: (7000, 7499),
    6902: (8000, 8499),
    6903: (9000, 9499),
    6904: (10000, 10499),
    6905: (11000, 11499),
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def planned_fresh_swaps() -> tuple[bool, tuple[bool, ...]]:
    r = random.Random(6900)
    major = bool(r.getrandbits(1))
    swap_plan = [major] * 3 + [not major] * 2
    r.shuffle(swap_plan)
    if tuple(swap_plan) != (False, True, False, True, False):
        raise AssertionError(f"unexpected fresh swap plan: {swap_plan}")
    return major, tuple(swap_plan)


def build(seed: int, *, swap: bool | None = None, id_block: tuple[int, int] | None = None) -> dict[str, object]:
    r = random.Random(seed)
    perm = list(range(32))
    r.shuffle(perm)
    if swap is None:
        swap = r.choice([False, True])
    floor = "OP_Y" if swap else "OP_X"
    avoid = "OP_X" if swap else "OP_Y"
    if id_block is None:
        base = 1000 + (seed - 6800) * 1000
        id_block = (base, base + 499)
    ids = r.sample(range(id_block[0], id_block[1] + 1), 35)
    names = [*(f"ARG_{i:02d}" for i in range(32)), "OP_X", "OP_Y", "LINK"]
    token_ids = dict(zip(names, ids))
    train: list[dict[str, object]] = []
    for i in range(32):
        argument = f"ARG_{i:02d}"
        train += [
            {"kind": "atomic", "operator": floor, "argument": argument, "value": perm[i], "constraints": "floor"},
            {"kind": "atomic", "operator": avoid, "argument": argument, "value": perm[i], "constraints": "avoid"},
            {"kind": "noop", "operator": "NOOP", "argument": argument, "value": perm[i], "constraints": "none"},
        ]
    inverse = {value: index for index, value in enumerate(perm)}
    test: list[dict[str, object]] = []
    for reverse in (False, True):
        for lower in range(32):
            for forbidden in range(32):
                if lower == forbidden:
                    continue
                lower_arg = f"ARG_{inverse[lower]:02d}"
                forbidden_arg = f"ARG_{inverse[forbidden]:02d}"
                if reverse:
                    tokens = [avoid, forbidden_arg, "LINK", floor, lower_arg]
                else:
                    tokens = [floor, lower_arg, "LINK", avoid, forbidden_arg]
                test.append(
                    {
                        "order": "reverse" if reverse else "natural",
                        "tokens": tokens,
                        "token_ids": [token_ids[token] for token in tokens],
                        "lower": lower,
                        "forbidden": forbidden,
                    }
                )
    return {
        "schema": "NB5-fresh-lexical-cipher-v1",
        "domain_seed": seed,
        "value_count": 32,
        "permutation": perm,
        "operator_floor": floor,
        "operator_avoid": avoid,
        "token_ids": token_ids,
        "train": train,
        "test": test,
        "test_count": len(test),
        "train_count": len(train),
        "weights_fresh_init_required": True,
        "forbidden_in_training": "joint",
    }


def write(seed: int, output_dir: Path = OUT, *, swap: bool | None = None, id_block: tuple[int, int] | None = None) -> tuple[Path, str]:
    manifest = build(seed, swap=swap, id_block=id_block)
    path = output_dir / f"manifest_{seed}_v2.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path, sha_bytes(path.read_bytes())


def legacy_generate() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in LEGACY_SEEDS:
        path, digest = write(seed)
        _, regenerated_digest = write(seed)
        rows.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": digest,
                "regenerated_sha256": regenerated_digest,
                "reproducible": digest == regenerated_digest,
            }
        )
    return {"manifests": rows, "all_reproducible": all(row["reproducible"] for row in rows)}


def fresh_generate(output_dir: Path) -> dict[str, object]:
    seeds = FRESH_SEEDS
    major, swap_plan = planned_fresh_swaps()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, swap in zip(seeds, swap_plan):
        path, digest = write(seed, output_dir, swap=swap, id_block=FRESH_BLOCKS[seed])
        _, regenerated_digest = write(seed, output_dir, swap=swap, id_block=FRESH_BLOCKS[seed])
        rows.append(
            {
                "seed": seed,
                "swap": swap,
                "id_block": list(FRESH_BLOCKS[seed]),
                "path": str(path),
                "sha256": digest,
                "regenerated_sha256": regenerated_digest,
                "reproducible": digest == regenerated_digest,
            }
        )
    return {
        "mode": "fresh",
        "major": major,
        "swap_plan": list(swap_plan),
        "seeds": list(seeds),
        "manifests": rows,
        "all_reproducible": all(row["reproducible"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="generate only the authorized 6901-6905 battery")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    result = fresh_generate(args.output_dir) if args.fresh else legacy_generate()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
