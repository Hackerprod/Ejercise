"""Consolidate frozen U0-A provenance and baseline results for U0-B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.unified import UnifiedT1U0  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_hash(revision: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT.parent,
        text=True,
    ).strip()


def parameter_counts() -> dict[str, object]:
    model = UnifiedT1U0(64)
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if name.startswith("memory_reader."):
            group = "P_reader"
        elif name.startswith("core.workspace_correction."):
            group = "P_correction"
        elif name.startswith("core.alu_"):
            group = "P_typed_adapters"
        elif name.startswith("core."):
            group = "P_core"
        elif name.startswith("commit.operation_heads."):
            group = "P_ALU_heads"
        elif name.startswith(("token_embedding.", "opcode_embedding.", "immediate_embedding.")) or name == "slot_type_embeddings":
            group = "P_codebooks"
        elif name.startswith(("pointer_decoder.", "evidence_decoder.", "register_decoder.", "workspace_decoder.")):
            group = "P_decoders"
        else:
            raise RuntimeError(f"unclassified parameter: {name}")
        counts[group] += parameter.numel()

    for key in ("P_core", "P_reader", "P_codebooks", "P_decoders", "P_ALU_heads", "P_typed_adapters", "P_correction"):
        counts.setdefault(key, 0)
    total = sum(counts.values())
    typed = counts["P_ALU_heads"] + counts["P_typed_adapters"]
    nonshared = counts["P_codebooks"] + counts["P_decoders"] + counts["P_ALU_heads"] + counts["P_typed_adapters"] + counts["P_correction"]
    return {
        "counts": dict(counts),
        "P_total": total,
        "P_typed": typed,
        "P_total_no_compartido": nonshared,
        "ratios": {
            "P_typed/P_core": typed / counts["P_core"],
            "P_total_no_compartido/P_total": nonshared / total,
        },
        "definitions": {
            "P_core": "core trunk excluding ALU typed projections/adapters and frozen workspace correction",
            "P_reader": "all SharedMemoryReader parameters, including reader control embeddings",
            "P_codebooks": "root token/opcode/immediate embeddings plus slot type embeddings",
            "P_decoders": "all decoder parameters; tied cosine decoders have zero parameters",
            "P_ALU_heads": "TypedCommit operation_heads for ALU_ADD/SUB/MUL",
            "P_typed_adapters": "core ALU left/right/opcode projections plus per-opcode low-rank adapters",
            "P_correction": "frozen core.workspace_correction parameters reserved for U0-C",
            "P_total_no_compartido": "codebooks + decoders + ALU heads + typed adapters + correction; excludes shared core and reader",
        },
    }


def main() -> int:
    campaign_root = ROOT / "campaign"
    runs: dict[str, dict[str, object]] = {}
    source_manifest: dict[str, str] | None = None
    source_manifest_hash: str | None = None
    for seed in SEEDS:
        run_dir = campaign_root / f"u0a_iso_clean_seed{seed}_12000"
        final_path = run_dir / "final.json"
        config_path = run_dir / "config.json"
        manifest_path = run_dir / "source_manifest.json"
        checkpoint_path = run_dir / "best.pt"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if source_manifest is None:
            source_manifest = manifest
            source_manifest_hash = sha256(manifest_path)
        elif manifest != source_manifest:
            raise RuntimeError(f"source manifest differs for seed {seed}")
        runs[str(seed)] = {
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "config": config,
            "baseline_no_ablation": json.loads(final_path.read_text(encoding="utf-8")),
        }

    assert source_manifest is not None
    assert source_manifest_hash is not None
    test_hashes = {path: digest for path, digest in source_manifest.items() if "\\test." in path}
    archive = {
        "schema": "T1-U0-A frozen provenance archive for U0-B",
        "spec": "T1.4_Spec_U0B_U0C_Ablaciones.md",
        "source_decision": "MD/Respuesta_8.md",
        "code_commits": {
            "u0a_implementation_and_results": commit_hash("852a4cb"),
            "u0b_u0c_authorization": commit_hash("32502c7"),
        },
        "curriculum": {
            "protocol": "ISO-UPDATE",
            "supersteps": 12000,
            "optimizer_steps": 12000,
            "batches_per_task": 12000,
            "tasks_per_superstep": 6,
            "batch_size": 128,
            "sequential_schedule": "five H1 complete-table batches then one existing H3-H6 composition batch",
            "sequential_h1_batches": 10000,
            "sequential_composition_batches": 2000,
            "routing": "oracle opcode/immediate/source/destination/row mask",
            "optimizer": "AdamW lr=3e-4; trunk weight_decay=1e-4; typed embeddings/heads/decoders/norms/biases weight_decay=0",
            "workspace_correction": "frozen",
            "retraining_after_ablation": False,
        },
        "datasets": {
            "source_manifest_sha256": source_manifest_hash,
            "source_hashes": source_manifest,
            "sealed_test_hashes": test_hashes,
            "canonicalization": "existing source datasets adapted to canonical REL/PAIR/ASSIGN/ATTR/VEC; H1 replay table generated in-memory",
        },
        "parameters": parameter_counts(),
        "runs": runs,
        "baseline_aggregate": {
            "status": "PASS_STRONG",
            "functional_coexistence": "PASS_STRONG",
            "training_efficiency": "OPEN / not evaluated",
            "workspace_ledger": "PASS, no systematic degradation vs post-refactor isolated baseline; H6 error by seed 0.00029-0.00144",
            "note": "Workspace is not exactly zero or bit-exact; U0-A criterion is preservation within joint checkpoint relative to isolated post-refactor baseline.",
        },
    }
    output = campaign_root / "u0a_u0b_provenance_archive.json"
    output.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "seeds": list(SEEDS), "checkpoint_count": len(runs), "parameter_counts": archive["parameters"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
