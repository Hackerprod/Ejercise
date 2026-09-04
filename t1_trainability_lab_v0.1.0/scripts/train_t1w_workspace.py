"""Train and evaluate T1-W continuous workspace microgate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.workspace import WorkspaceCore  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
DATASET_SEEDS = {"train": 10101, "val": 20202, "test": 30303}
H_VALUES = (2, 4, 6)
ROUNDS = (1, 2, 4, 6)
SAMPLES_PER_H = 1024
DIMENSION = 64


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_split(seed: int) -> tuple[Tensor, Tensor, Tensor]:
    vectors_by_h = []
    lengths_by_h = []
    targets_by_h = []
    for index, hops in enumerate(H_VALUES):
        generator = torch.Generator().manual_seed(seed + 1009 * index)
        vectors = torch.randn((SAMPLES_PER_H, hops, DIMENSION), generator=generator)
        padded = torch.zeros((SAMPLES_PER_H, max(H_VALUES), DIMENSION))
        padded[:, :hops, :] = vectors
        vectors_by_h.append(padded)
        lengths_by_h.append(torch.full((SAMPLES_PER_H,), hops, dtype=torch.long))
        targets_by_h.append(vectors.sum(dim=1))
    return torch.cat(vectors_by_h), torch.cat(lengths_by_h), torch.cat(targets_by_h)


def ensure_datasets(dataset_dir: Path) -> dict[str, dict[str, object]]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for split, seed in DATASET_SEEDS.items():
        path = dataset_dir / f"{split}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            vectors, lengths, targets = payload["vectors"], payload["lengths"], payload["targets"]
        else:
            vectors, lengths, targets = make_split(seed)
            torch.save({"vectors": vectors, "lengths": lengths, "targets": targets, "seed": seed}, path)
        manifest[split] = {"seed": seed, "samples": int(len(lengths)), "h_counts": {str(hops): int((lengths == hops).sum()) for hops in H_VALUES}, "vectors_sha256": tensor_sha256(vectors), "targets_sha256": tensor_sha256(targets)}
    save_json(dataset_dir / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def evaluate(core: WorkspaceCore, dataset: TensorDataset, *, mode: str) -> dict[str, dict[str, dict[str, float]]]:
    vectors, lengths, targets = dataset.tensors
    results: dict[str, dict[str, dict[str, float]]] = {str(hops): {str(rounds): {} for rounds in ROUNDS} for hops in H_VALUES}
    for rounds in ROUNDS:
        output = core(vectors, lengths, rounds=rounds, mode=mode)
        error = torch.linalg.vector_norm(output - targets, dim=-1) / torch.linalg.vector_norm(targets, dim=-1)
        cosine = torch.nn.functional.cosine_similarity(output, targets, dim=-1)
        for hops in H_VALUES:
            selected = lengths == hops
            results[str(hops)][str(rounds)] = {"cosine": float(cosine[selected].mean()), "normalized_error": float(error[selected].mean())}
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets" / "t1w_workspace")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "t1w_workspace_seed101")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--identity-bypass", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = ensure_datasets(args.dataset_dir)
    train_payload = torch.load(args.dataset_dir / "train.pt", map_location="cpu", weights_only=False)
    val_payload = torch.load(args.dataset_dir / "val.pt", map_location="cpu", weights_only=False)
    test_payload = torch.load(args.dataset_dir / "test.pt", map_location="cpu", weights_only=False)
    train_data = TensorDataset(train_payload["vectors"], train_payload["lengths"], train_payload["targets"])
    val_data = TensorDataset(val_payload["vectors"], val_payload["lengths"], val_payload["targets"])
    test_data = TensorDataset(test_payload["vectors"], test_payload["lengths"], test_payload["targets"])
    core = WorkspaceCore(DIMENSION, identity_bypass=args.identity_bypass)
    optimizer = torch.optim.AdamW(core.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    config = {"phase": "T1-W workspace microgate", "seed": args.seed, "dataset_seeds": DATASET_SEEDS, "dimension": DIMENSION, "h_values": H_VALUES, "rounds": ROUNDS, "samples_per_h": SAMPLES_PER_H, "vectors": "fixed dense Gaussian N(0,1), persisted per split", "transition": "W_next=W+F(e,Norm(W))", "identity_bypass": args.identity_bypass, "operator": "e + correction_mlp([e,Norm(W)])" if args.identity_bypass else "Linear(128,256)->SiLU->Linear(256,64)", "head": "Identity(W_H)", "controls": "same checkpoint inference ablations: frozen/replaced/R<H", "optimizer": "AdamW(lr=3e-4, weight_decay=1e-4)", "max_steps": args.max_steps}
    save_json(args.output_dir / "config.json", config)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    iterator = iter(loader)
    for step_index in range(1, args.max_steps + 1):
        try:
            vectors, lengths, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            vectors, lengths, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        output = core(vectors, lengths, rounds=6, mode="residual")
        loss = criterion(output, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step_index}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0))
        optimizer.step()
        if step_index % 100 == 0 or step_index == args.max_steps:
            val_output = core(val_payload["vectors"], val_payload["lengths"], rounds=6, mode="residual")
            val_error = torch.linalg.vector_norm(val_output - val_payload["targets"], dim=-1) / torch.linalg.vector_norm(val_payload["targets"], dim=-1)
            val_cosine = torch.nn.functional.cosine_similarity(val_output, val_payload["targets"], dim=-1)
            metric = {"step": step_index, "loss": float(loss.detach()), "val_cosine": float(val_cosine.mean()), "val_normalized_error": float(val_error.mean()), "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
    torch.save({"config": config, "core": core.state_dict()}, args.output_dir / "final.pt")
    results = {mode: evaluate(core, test_data, mode=mode) for mode in ("residual", "frozen", "replaced")}
    final = {"status": "completed", "finite": True, "seed": args.seed, "dataset_manifest": manifest, "results": results, "elapsed_seconds": time.perf_counter() - started}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
