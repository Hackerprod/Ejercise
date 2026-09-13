"""Check legacy NB5 manifests or an explicit strict fresh battery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = ROOT / "campaign" / "nb5_manifests"
DEFAULT_OUTPUT = DEFAULT_MANIFEST_DIR / "nb5_v2_semantic_check.json"
LEGACY_SEEDS = (6801, 6802, 6803, 6804, 6805)
FRESH_SEEDS = (6901, 6902, 6903, 6904, 6905)
FRESH_BLOCKS = {
    6901: (7000, 7499),
    6902: (8000, 8499),
    6903: (9000, 9499),
    6904: (10000, 10499),
    6905: (11000, 11499),
}
LEGACY_DEVELOPMENT_RANGE = range(2000, 6500)
TOKEN_ORDER = [*(f"ARG_{i:02d}" for i in range(32)), "LINK", "OP_X", "OP_Y"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hash(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fresh_swap_plan() -> tuple[bool, ...]:
    import random

    r = random.Random(6900)
    major = bool(r.getrandbits(1))
    swap_plan = [major] * 3 + [not major] * 2
    r.shuffle(swap_plan)
    if tuple(swap_plan) != (False, True, False, True, False):
        raise AssertionError(f"unexpected fresh swap plan: {swap_plan}")
    return tuple(swap_plan)


def expected_operators(seed: int, fresh: bool) -> tuple[str, str]:
    if fresh:
        swap = fresh_swap_plan()[FRESH_SEEDS.index(seed)]
        return ("OP_Y", "OP_X") if swap else ("OP_X", "OP_Y")
    return ("OP_X", "OP_Y")


def check_manifest(path: Path, seed: int, *, fresh: bool) -> dict[str, Any]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"seed": seed, "path": str(path), "rows": 0, "errors": [f"load: {error}"]}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(manifest.get("schema") == "NB5-fresh-lexical-cipher-v1", "schema mismatch")
    require(int(manifest.get("domain_seed", -1)) == seed, "domain_seed mismatch")
    token_ids = manifest.get("token_ids")
    require(isinstance(token_ids, dict), "token_ids is not an object")
    if isinstance(token_ids, dict):
        require(list(token_ids) == TOKEN_ORDER, "token_ids key order mismatch")
        values = list(token_ids.values())
        require(len(values) == 35 and all(isinstance(value, int) for value in values), "token_ids must contain 35 integer IDs")
        require(len(set(values)) == len(values), "token IDs are not unique within manifest")

    expected_floor, expected_avoid = expected_operators(seed, fresh)
    require(manifest.get("operator_floor") == expected_floor, "operator_floor mismatch")
    require(manifest.get("operator_avoid") == expected_avoid, "operator_avoid mismatch")
    permutation = manifest.get("permutation")
    require(isinstance(permutation, list) and len(permutation) == 32 and set(permutation) == set(range(32)), "permutation is not a bijection over 0..31")

    train = manifest.get("train", [])
    require(isinstance(train, list) and len(train) == 96, "train count is not 96")
    if isinstance(train, list):
        counts = {kind: 0 for kind in ("floor", "avoid", "noop")}
        for index, row in enumerate(train):
            if not isinstance(row, dict):
                errors.append(f"train[{index}] is not an object")
                continue
            constraints = row.get("constraints")
            kind = constraints if constraints != "none" else "noop"
            if kind in counts:
                counts[kind] += 1
            require(kind in counts, f"train[{index}] has invalid kind")
            require(row.get("kind") != "joint" and constraints != "joint", f"train[{index}] contains forbidden joint data")
            expected_arg = f"ARG_{index // 3:02d}"
            require(row.get("argument") == expected_arg, f"train[{index}] argument/order mismatch")
            require(row.get("value") == permutation[index // 3], f"train[{index}] value/permutation mismatch")
            if kind == "floor":
                require(row.get("operator") == expected_floor, f"train[{index}] floor operator mismatch")
            elif kind == "avoid":
                require(row.get("operator") == expected_avoid, f"train[{index}] avoid operator mismatch")
            else:
                require(row.get("operator") == "NOOP", f"train[{index}] noop operator mismatch")
        require(counts == {"floor": 32, "avoid": 32, "noop": 32}, f"train kind counts mismatch: {counts}")

    inverse = {value: index for index, value in enumerate(permutation)} if isinstance(permutation, list) and set(permutation) == set(range(32)) else {}
    test = manifest.get("test", [])
    require(isinstance(test, list) and len(test) == 1984, "test count is not 1984")
    natural = [row for row in test if isinstance(row, dict) and row.get("order") == "natural"] if isinstance(test, list) else []
    reverse = [row for row in test if isinstance(row, dict) and row.get("order") == "reverse"] if isinstance(test, list) else []
    require(len(natural) == 992, "natural count is not 992")
    require(len(reverse) == 992, "reverse count is not 992")
    natural_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    order_pairs: dict[str, set[tuple[int, int]]] = {"natural": set(), "reverse": set()}
    for index, row in enumerate(test if isinstance(test, list) else []):
        if not isinstance(row, dict):
            errors.append(f"test[{index}] is not an object")
            continue
        order = row.get("order")
        require(order in order_pairs, f"test[{index}] has invalid order")
        try:
            lower = int(row["lower"])
            forbidden = int(row["forbidden"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"test[{index}] has invalid semantic values")
            continue
        require(0 <= lower < 32 and 0 <= forbidden < 32, f"test[{index}] values outside 0..31")
        require(lower != forbidden, f"test[{index}] is diagonal")
        if order not in order_pairs:
            continue
        pair = (lower, forbidden)
        require(pair not in order_pairs[order], f"test[{index}] duplicate {order} pair")
        order_pairs[order].add(pair)
        lower_arg = f"ARG_{inverse[lower]:02d}" if lower in inverse else "<invalid>"
        forbidden_arg = f"ARG_{inverse[forbidden]:02d}" if forbidden in inverse else "<invalid>"
        expected_tokens = (
            [expected_floor, lower_arg, "LINK", expected_avoid, forbidden_arg]
            if order == "natural"
            else [expected_avoid, forbidden_arg, "LINK", expected_floor, lower_arg]
        )
        require(row.get("tokens") == expected_tokens, f"test[{index}] token order/operator/LINK mismatch")
        if isinstance(token_ids, dict):
            require(row.get("token_ids") == [token_ids.get(token) for token in expected_tokens], f"test[{index}] token_ids inconsistency")
        if order == "natural":
            natural_by_pair[pair] = row
    require(order_pairs["natural"] == order_pairs["reverse"], "natural/reverse pair catalogs differ")
    for index, row in enumerate(reverse):
        pair = (int(row["lower"]), int(row["forbidden"]))
        natural_row = natural_by_pair.get(pair)
        expected = (
            [expected_avoid, natural_row["tokens"][4], "LINK", expected_floor, natural_row["tokens"][1]]
            if natural_row is not None
            else None
        )
        require(natural_row is not None and row.get("tokens") == expected, f"reverse[{index}] incomplete clause swap")

    if fresh and seed in FRESH_BLOCKS and isinstance(token_ids, dict):
        start, end = FRESH_BLOCKS[seed]
        require(all(start <= value <= end for value in token_ids.values()), "fresh token ID outside assigned physical block")
    return {
        "seed": seed,
        "path": str(path),
        "sha256": sha(path),
        "rows": len(test) if isinstance(test, list) else 0,
        "natural_rows": len(natural),
        "reverse_rows": len(reverse),
        "semantic_errors": len(errors),
        "errors": errors[:20],
    }


def existing_manifest_ids(manifest_dir: Path, selected: set[Path]) -> tuple[set[int], list[str]]:
    ids: set[int] = set()
    files: list[str] = []
    for path in sorted(ROOT.rglob("*manifest*.json")):
        resolved = path.resolve()
        if resolved in selected:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        token_ids = payload.get("token_ids")
        if not isinstance(token_ids, dict) or not all(isinstance(value, int) for value in token_ids.values()):
            continue
        ids.update(token_ids.values())
        files.append(str(path))
    return ids, files


def fresh_check(manifest_dir: Path, output: Path, seeds: tuple[int, ...]) -> int:
    if seeds != FRESH_SEEDS:
        raise ValueError(f"fresh checker requires seeds {FRESH_SEEDS}, got {seeds}")
    paths = [manifest_dir / f"manifest_{seed}_v2.json" for seed in seeds]
    per_manifest = [check_manifest(path, seed, fresh=True) if path.exists() else {"seed": seed, "path": str(path), "rows": 0, "semantic_errors": 1, "errors": ["manifest missing"]} for path, seed in zip(paths, seeds)]
    selected_ids: set[Path] = {path.resolve() for path in paths}
    old_ids, old_files = existing_manifest_ids(manifest_dir, selected_ids)
    id_errors: list[str] = []
    fresh_ids: set[int] = set()
    for seed, path in zip(seeds, paths):
        try:
            token_ids = json.loads(path.read_text(encoding="utf-8"))["token_ids"]
            values = set(token_ids.values())
        except Exception:
            values = set()
        overlap = fresh_ids & values
        if overlap:
            id_errors.append(f"seed {seed}: overlap with another fresh seed: {sorted(overlap)}")
        fresh_ids.update(values)
        start, end = FRESH_BLOCKS[seed]
        outside = values - set(range(start, end + 1))
        if outside:
            id_errors.append(f"seed {seed}: IDs outside block {start}-{end}: {sorted(outside)}")
        legacy_overlap = values & set(LEGACY_DEVELOPMENT_RANGE)
        if legacy_overlap:
            id_errors.append(f"seed {seed}: overlap with legacy development range: {sorted(legacy_overlap)}")
        existing_overlap = values & old_ids
        if existing_overlap:
            id_errors.append(f"seed {seed}: overlap with existing manifest IDs: {sorted(existing_overlap)}")

    checked_rows = sum(item["rows"] for item in per_manifest)
    errors = sum(item["semantic_errors"] for item in per_manifest) + len(id_errors)
    artifact: dict[str, Any] = {
        "status": "passed" if errors == 0 and checked_rows == 9920 else "failed",
        "mode": "fresh",
        "task": "T2-NOBYPASS-2-RCSEP-NOGATE-FREEZE",
        "swap_plan": list(fresh_swap_plan()),
        "seeds": list(seeds),
        "manifests": per_manifest,
        "id_disjointness": {
            "fresh_ids": len(fresh_ids),
            "existing_manifest_files_checked": old_files,
            "existing_manifest_ids_checked": len(old_ids),
            "legacy_development_range": [2000, 6499],
            "errors": id_errors,
        },
        "checked_rows": checked_rows,
        "checked_expected": 9920,
        "semantic_errors": errors,
        "validation": [
            "schema/domain_seed",
            "token order and permutation",
            "train counts/kinds/no joint",
            "test count and natural/reverse counts",
            "pair uniqueness and non-diagonal values",
            "semantic lower/forbidden values",
            "operator/LINK positions",
            "token_ids consistency",
            "corrected complete reverse clause mapping",
            "fresh physical ID disjointness",
        ],
    }
    artifact["artifact_self_hash"] = artifact_hash(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


def legacy_main() -> int:
    out = []
    for seed in LEGACY_SEEDS:
        path = DEFAULT_MANIFEST_DIR / f"manifest_{seed}_v2.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = manifest["test"]
        natural = {(x["lower"], x["forbidden"]): x for x in rows if x["order"] == "natural"}
        errors = []
        if len(rows) != 1984 or len(natural) != 992:
            errors.append(f"counts rows={len(rows)} natural={len(natural)}")
        for row in rows:
            tokens = row["tokens"]
            floor = manifest["operator_floor"]
            avoid = manifest["operator_avoid"]
            if tokens[0] == floor:
                lower_arg, forbidden_arg = tokens[1], tokens[4]
            elif tokens[0] == avoid:
                forbidden_arg, lower_arg = tokens[1], tokens[4]
            else:
                errors.append("bad operators")
                continue
            if (manifest["permutation"][int(lower_arg[4:])] != row["lower"] or manifest["permutation"][int(forbidden_arg[4:])] != row["forbidden"]):
                errors.append(f"semantic {row['order']} {row['lower']},{row['forbidden']}")
            if row["order"] == "reverse":
                natural_row = natural.get((row["lower"], row["forbidden"]))
                expected = [avoid, natural_row["tokens"][4], "LINK", floor, natural_row["tokens"][1]] if natural_row else None
                if expected is None or tokens != expected:
                    errors.append(f"reverse pair {row['lower']},{row['forbidden']}")
        out.append({"seed": seed, "path": str(path), "sha256": sha(path), "rows": len(rows), "semantic_errors": len(errors), "errors": errors[:20]})
    result = {"status": "passed" if all(not item["semantic_errors"] for item in out) else "failed", "manifests": out, "checked_rows": sum(item["rows"] for item in out), "checked_expected": 9920}
    print(json.dumps(result, indent=2))
    DEFAULT_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="strictly check only the authorized 6901-6905 battery")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.fresh and args.seeds is None and args.output == DEFAULT_OUTPUT and args.manifest_dir == DEFAULT_MANIFEST_DIR:
        return legacy_main()
    seeds = tuple(args.seeds) if args.seeds is not None else FRESH_SEEDS
    return fresh_check(args.manifest_dir, args.output, seeds)


if __name__ == "__main__":
    raise SystemExit(main())
