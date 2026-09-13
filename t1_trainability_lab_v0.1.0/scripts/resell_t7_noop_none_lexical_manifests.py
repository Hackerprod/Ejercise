"""Add missing fresh-domain fields to existing T7 preparation manifests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_t7_noop_none_lexical_development import (  # noqa: E402
    ID_BLOCKS,
    SEEDS,
    base_token_ids,
    domain_permutation,
)


MANIFEST_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation" / "manifests"


def resell(seed: int) -> dict[str, object]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    token_order = list(manifest["vocabulary"]["stage_a_token_order"])
    role_assignment = manifest["operator_role_assignment"]
    active_mapping = {operator: role for operator, role in role_assignment.items() if role != "NOOP"}
    manifest.update(
        {
            "permutation": domain_permutation(seed),
            "token_order": token_order,
            "token_ids": base_token_ids(seed, token_order),
            "id_block": list(ID_BLOCKS[seed]),
            "id_semantics": "physical IDs fresh and disjoint; model uses manifest-local vocabulary indices",
            "operator_for_role": {role: operator for operator, role in active_mapping.items()},
        }
    )
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return {"seed": seed, "path": path.relative_to(ROOT).as_posix(), "bytes": len(payload)}


def main() -> None:
    print(json.dumps({"status": "resold", "manifests": [resell(seed) for seed in SEEDS]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
