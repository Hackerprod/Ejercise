"""Bounded Phase E stage-1 benchmark and checkpoint tooling.

This module deliberately orchestrates the existing MRDL language pipeline. It
does not alter transport, lanes, promotion, VSA, composition-gate, or
generation implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from mrdl.evidence import MemoryLevel
from mrdl.promotion import PromotionState, RecordState
from mrdl.language import (
    EmbeddingTable,
    EvalMetrics,
    MRDLLanguagePipeline,
    TrigramBaseline,
    tokenize,
)


CHECKPOINT_VERSION = 1
DEFAULT_STORY_COUNT = 5000
DEFAULT_STORY_TOKEN_LIMIT = 8
DEFAULT_SPLIT_SEED = 1729
DEFAULT_CONTROLLER_STEPS = 64
DEFAULT_VSA_DIMENSION = 8
DEFAULT_PROMOTION_LIMIT = 128


def _story_id(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_stage_stories(
    path: str | Path,
    story_count: int = DEFAULT_STORY_COUNT,
    story_token_limit: int = DEFAULT_STORY_TOKEN_LIMIT,
) -> list[dict[str, Any]]:
    """Load a bounded, ordered source slice with stable IDs.

    Story boundaries are corpus delimiters. Truncation is per story and keeps
    EOS, so no story is split into train and test examples.
    """
    if story_count <= 0 or story_token_limit < 4:
        raise ValueError("story_count must be positive and story_token_limit must be >= 4")
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    delimiter = "<|endoftext|>"
    blocks = [block.strip() for block in raw.split(delimiter) if block.strip()]
    selected: list[dict[str, Any]] = []
    for source_index, text in enumerate(blocks[:story_count]):
        tokens = ["<bos>", *tokenize(text), "<eos>"]
        if len(tokens) > story_token_limit:
            tokens = tokens[:story_token_limit]
            tokens[-1] = "<eos>"
        if len(tokens) < 4:
            continue
        selected.append({
            "story_id": _story_id(text),
            "source_index": source_index,
            "tokens": tokens,
        })
    return selected


def split_stage_stories(
    records: Sequence[dict[str, Any]],
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[list[str]], list[list[str]], dict[str, Any]]:
    """Deterministically split whole stories by content hash.

    Hash bucketing prevents adjacent-source artifacts and guarantees that a
    story ID appears in exactly one partition.
    """
    train: list[list[str]] = []
    test: list[list[str]] = []
    train_ids: list[str] = []
    test_ids: list[str] = []
    for record in records:
        digest = hashlib.sha256(f"{split_seed}:{record['story_id']}".encode("ascii")).digest()
        target = train if int.from_bytes(digest[:8], "big") % 5 else test
        target.append(list(record["tokens"]))
        (train_ids if target is train else test_ids).append(str(record["story_id"]))
    if not train or not test:
        raise ValueError("split produced empty train or test partition")
    if set(train_ids) & set(test_ids):
        raise AssertionError("story overlap detected between train and test")
    manifest = {
        "algorithm": "sha256(split_seed:story_id) mod 5; bucket 0=test",
        "split_seed": split_seed,
        "train_story_ids": train_ids,
        "test_story_ids": test_ids,
    }
    return train, test, manifest


def _proc_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return 0


def _host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    except (FileNotFoundError, ValueError):
        return values
    return values


def _swap_usage() -> dict[str, int]:
    memory = _host_memory()
    return {
        "total_bytes": memory.get("SwapTotal", 0),
        "free_bytes": memory.get("SwapFree", 0),
        "used_bytes": memory.get("SwapTotal", 0) - memory.get("SwapFree", 0),
    }


def _resource_snapshot(root: str | Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    memory = _host_memory()
    return {
        "timestamp": time.time(),
        "process_rss_bytes": _proc_rss_bytes(),
        "host_total_bytes": memory.get("MemTotal", 0),
        "host_available_bytes": memory.get("MemAvailable", 0),
        "host_used_bytes": memory.get("MemTotal", 0) - memory.get("MemAvailable", 0),
        "swap": _swap_usage(),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "disk_used_bytes": usage.used,
    }


class ResourceMonitor:
    def __init__(self, root: str | Path, interval_seconds: float = 0.25) -> None:
        self.root = str(root)
        self.interval_seconds = interval_seconds
        self.before: dict[str, Any] = {}
        self.after: dict[str, Any] = {}
        self.peak: dict[str, Any] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.before = _resource_snapshot(self.root)
        self.peak = dict(self.before)
        self._thread = threading.Thread(target=self._sample, name="phase-e-resource-monitor", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = _resource_snapshot(self.root)
            self.samples += 1
            self.peak["process_rss_bytes"] = max(self.peak.get("process_rss_bytes", 0), current["process_rss_bytes"])
            self.peak["host_used_bytes"] = max(self.peak.get("host_used_bytes", 0), current["host_used_bytes"])
            self.peak["host_available_bytes"] = min(self.peak.get("host_available_bytes", current["host_available_bytes"]), current["host_available_bytes"])
            self.peak["disk_used_bytes"] = max(self.peak.get("disk_used_bytes", 0), current["disk_used_bytes"])
            self.peak["disk_free_bytes"] = min(self.peak.get("disk_free_bytes", current["disk_free_bytes"]), current["disk_free_bytes"])
            self.peak["swap"] = current["swap"]

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.after = _resource_snapshot(self.root)

    def to_dict(self) -> dict[str, Any]:
        return {"before": self.before, "peak": self.peak, "after": self.after, "samples": self.samples}


def _metric_dict(metrics: EvalMetrics) -> dict[str, Any]:
    return asdict(metrics)


def evaluate_trigram(train: Sequence[Sequence[str]], test: Sequence[Sequence[str]]) -> dict[str, Any]:
    baseline = TrigramBaseline()
    baseline.train(train)
    loss = 0.0
    correct = 0
    tokens = 0
    for story in test:
        context = ["<bos>"]
        for target in story[1:]:
            prediction = baseline.predict(context)
            loss -= math.log(max(1e-9, prediction.probabilities.get(target, 1e-9)))
            correct += prediction.token == target
            tokens += 1
            context.append(target)
    return {
        "stories": len(test),
        "tokens": tokens,
        "loss": loss / max(1, tokens),
        "accuracy": correct / max(1, tokens),
        "memory_bytes": sum(len(key[0]) + len(key[1]) + len(value) * 16 for key, value in baseline.counts.items()),
    }


def reachability_metrics(
    pipeline: MRDLLanguagePipeline,
    test: Sequence[Sequence[str]],
    max_depth: int = 4,
) -> dict[str, Any]:
    """Measure first target appearance in FULL propagation frontiers.

    This is graph reachability, not decoder argmax. Each target is checked
    against exact d1..d4 frontiers from the final stage state.
    """
    if max_depth != 4:
        raise ValueError("Phase E stage 1 requires exact d1..d4 reachability")
    full, _ = pipeline._ensure_engines()
    first = Counter()
    first_by_seed: dict[str, dict[str, int | None]] = {}
    total = 0
    for story in test:
        for index in range(1, len(story)):
            context_token = story[index - 1]
            target = story[index]
            if context_token not in first_by_seed:
                full.propagate(context_token, 1.0, max_depth)
                seed_depths: dict[str, int | None] = {}
                for depth, frontier in enumerate(full.round_frontiers, start=1):
                    for candidate in frontier:
                        seed_depths.setdefault(candidate.node_id, depth)
                first_by_seed[context_token] = seed_depths
            first_depth = first_by_seed[context_token].get(target)
            first["never" if first_depth is None else f"d{first_depth}"] += 1
            total += 1
    d1 = first["d1"]
    d2 = first["d2"]
    d3 = first["d3"]
    d4 = first["d4"]
    never = first["never"]
    return {
        "total_targets": total,
        "never_reachable": never,
        "first_reachable_d1": d1,
        "first_reachable_d2": d2,
        "first_reachable_d3": d3,
        "first_reachable_d4": d4,
        "first_reachable_d_le_2": d1 + d2,
        "first_reachable_d_ge_3": d3 + d4,
        "fractions": {
            "never_reachable": never / max(1, total),
            "first_reachable_d_le_2": (d1 + d2) / max(1, total),
            "first_reachable_d_ge_3": (d3 + d4) / max(1, total),
        },
        "depth_definition": "first appearance of gold target node in FULL exact propagation frontier; decoder score/argmax excluded",
    }


def _json_edge(edge: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in edge.items():
        if key == "causal_certificate":
            result[key] = asdict(value)
        elif isinstance(value, MemoryLevel):
            result[key] = value.value
        else:
            result[key] = value
    return result


def checkpoint_pipeline(
    pipeline: MRDLLanguagePipeline,
    path: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata,
        "pipeline_config": {
            "embedding_mode": pipeline.embeddings.mode,
            "embedding_dimension": pipeline.embeddings.dimension,
            "embedding_source": pipeline.embeddings.source,
            "external_pretrained": pipeline.embeddings.external_pretrained,
            "beam_width": pipeline.beam_width,
            "context_window": pipeline.context_window,
            "propagation_rounds": pipeline.propagation_rounds,
            "composition_depth": pipeline.composition_depth,
            "transport_coefficient_mode": pipeline.edge_memory.transport_coefficient_mode,
            "vsa_dimension": pipeline.vsa_memory.dimension,
            "vsa_seed": pipeline.vsa_memory.seed,
        },
        "embeddings": {token: list(vector) for token, vector in sorted(pipeline.embeddings.vectors.items())},
        "edges": [_json_edge(edge) | {"source": source, "target": target} for (source, target), edge in sorted(pipeline.edge_memory.edges.items())],
        "controller": {"weights": list(pipeline.controller.weights), "updates": pipeline.controller.updates},
        "counters": {
            "delta_updates": pipeline.edge_memory.delta_updates,
            "promoted_edges": pipeline.promoted_edges,
            "causal_metrics": pipeline.causal_metrics,
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
    encoded = destination.read_bytes()
    return {
        "path": str(destination),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "contents": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "embedding_vectors": len(checkpoint["embeddings"]),
            "edges": len(checkpoint["edges"]),
            "m2_edges": sum(edge["level"] == MemoryLevel.M2.value for edge in checkpoint["edges"]),
            "controller_weights": len(checkpoint["controller"]["weights"]),
            "completed_story_count": metadata["completed_story_count"],
            "resume_next_source_index": metadata["resume_next_source_index"],
        },
    }


def load_pipeline_checkpoint(path: str | Path) -> MRDLLanguagePipeline:
    """Restore stage state without running any follow-on corpus stage."""
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    config = checkpoint["pipeline_config"]
    vectors = {token: tuple(values) for token, values in checkpoint["embeddings"].items()}
    embeddings = EmbeddingTable(
        config["embedding_mode"],
        config["embedding_dimension"],
        vectors,
        config["embedding_source"],
        config["external_pretrained"],
    )
    pipeline = MRDLLanguagePipeline(
        embeddings,
        beam_width=config["beam_width"],
        context_window=config["context_window"],
        propagation_rounds=config["propagation_rounds"],
        composition_depth=config["composition_depth"],
        vsa_dimension=config["vsa_dimension"],
        vsa_seed=config["vsa_seed"],
        transport_coefficient_mode=config["transport_coefficient_mode"],
    )
    for serialized in checkpoint["edges"]:
        source = serialized.pop("source")
        target = serialized.pop("target")
        serialized["level"] = MemoryLevel(serialized["level"])
        pipeline.edge_memory.edges[(source, target)] = serialized
        record_id = str(serialized["record_id"])
        pipeline.promotion_store.add_dependency(record_id)
        pipeline.promotion_store.create_m1(record_id, float(serialized["relation"]), 0.0, 1_000_000_000.0)
        if serialized["level"] is MemoryLevel.M2:
            record = pipeline.promotion_store._records[record_id]
            record.state = RecordState.PROMOTED
            record.promotion_state = PromotionState.PROMOTED
    pipeline.controller.weights = list(checkpoint["controller"]["weights"])
    pipeline.controller.updates = int(checkpoint["controller"]["updates"])
    pipeline.edge_memory.delta_updates = int(checkpoint["counters"]["delta_updates"])
    pipeline._invalidate_engines()
    return pipeline


def run_mode(
    records: Sequence[dict[str, Any]],
    train: Sequence[Sequence[str]],
    test: Sequence[Sequence[str]],
    mode: str,
    pretrained_path: Path,
    dimension: int,
    seed: int,
    beam: int,
    controller_steps: int,
    vsa_dimension: int,
    promotion_limit: int,
    checkpoint_path: Path,
    split_manifest: dict[str, Any],
    corpus_path: Path,
    story_token_limit: int,
) -> dict[str, Any]:
    vocabulary = sorted({token for record in records for token in record["tokens"]})
    if mode == "pretrained_frozen":
        embeddings = EmbeddingTable.from_frozen_file(vocabulary, pretrained_path, dimension, external_pretrained=str(pretrained_path).endswith(".gz"))
    else:
        embeddings = EmbeddingTable.random_frozen(vocabulary, dimension, seed)
    pipeline = MRDLLanguagePipeline(embeddings, beam, propagation_rounds=4, vsa_dimension=vsa_dimension)
    pipeline.observe_training(train)
    promoted = pipeline.promote_supported_edges(minimum_support=2, limit=promotion_limit)
    controller_updates = pipeline.train_controller(train, max_steps=controller_steps, promotion_limit=promotion_limit)
    graph = pipeline.evaluate(test, "FULL")
    reachability = reachability_metrics(pipeline, test)
    checkpoint_metadata = {
        "phase": "E",
        "stage": 1,
        "completed_story_count": len(records),
        "resume_next_source_index": max(int(record["source_index"]) for record in records) + 1,
        "corpus_path": str(corpus_path),
        "story_token_limit": story_token_limit,
        "vsa_dimension": vsa_dimension,
        "train_stories": len(train),
        "test_stories": len(test),
        "split_manifest": split_manifest,
        "not_run": ["phase_e_stage2", "10000_story_resume"],
    }
    checkpoint = checkpoint_pipeline(pipeline, checkpoint_path, checkpoint_metadata)
    restored = load_pipeline_checkpoint(checkpoint_path)
    if restored.promoted_edges != pipeline.promoted_edges or len(restored.edge_memory.edges) != len(pipeline.edge_memory.edges):
        raise AssertionError("checkpoint round-trip changed model state counts")
    return {
        "embedding_mode": mode,
        "embedding_source": embeddings.source,
        "transport_coefficient_mode": pipeline.edge_memory.transport_coefficient_mode,
        "train_stories": len(train),
        "test_stories": len(test),
        "vocab_size": len(vocabulary),
        "promoted_edges_during_stage": promoted,
        "controller_updates_during_stage": controller_updates,
        "graph": _metric_dict(graph),
        "reachability": reachability,
        "checkpoint": checkpoint,
    }


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.stories > DEFAULT_STORY_COUNT:
        raise ValueError("stage 1 is bounded at 5000 stories; 10000 resume is not run by this tool")
    corpus_path = Path(args.corpus)
    records = load_stage_stories(corpus_path, args.stories, args.story_token_limit)
    train, test, split_manifest = split_stage_stories(records, args.split_seed)
    monitor = ResourceMonitor(args.resource_root)
    started = time.perf_counter()
    monitor.start()
    try:
        modes = []
        for mode in ("random_frozen", "pretrained_frozen"):
            modes.append(run_mode(
                records, train, test, mode, Path(args.pretrained_path), args.dimension,
                args.seed, args.beam, args.controller_steps, args.vsa_dimension, args.promotion_limit,
                Path(args.output).with_name(f"phase_e_stage1_{mode}.checkpoint.json"),
                split_manifest, corpus_path, args.story_token_limit,
            ))
        baseline_started = time.perf_counter()
        baseline = evaluate_trigram(train, test)
        baseline_elapsed = time.perf_counter() - baseline_started
    finally:
        monitor.stop()
    elapsed = time.perf_counter() - started
    result = {
        "phase": "E",
        "stage": 1,
        "status": "completed",
        "corpus": str(corpus_path),
        "story_policy": {
            "requested_stories": args.stories,
            "loaded_stories": len(records),
            "max_tokens_per_story_including_bos_eos": args.story_token_limit,
            "truncation": "per-story tail replaced with <eos>; no story crosses partition",
        },
        "vsa_policy": {
            "dimension": args.vsa_dimension,
            "seed": args.seed,
            "reason": "stage-only bounded benchmark setting; existing VSA implementation unchanged",
        },
        "split": {
            "train_stories": len(train),
            "test_stories": len(test),
            "overlap_story_ids": sorted(set(split_manifest["train_story_ids"]) & set(split_manifest["test_story_ids"])),
            "algorithm": split_manifest["algorithm"],
            "split_seed": args.split_seed,
        },
        "runtime": {
            "wall_seconds_pipeline_modes": elapsed,
            "wall_seconds_trigram": baseline_elapsed,
            "wall_seconds_total": elapsed + baseline_elapsed,
        },
        "resources": monitor.to_dict(),
        "baseline_trigram": baseline,
        "modes": modes,
        "safety": {
            "ram_budget_bytes": 7_800_000_000,
            "swap_required_bytes": 0,
            "ten_k_run": False,
            "recommendation_rule": "safe only if peak host used remains materially below RAM budget, swap stays zero, disk margin remains substantial, and checkpoint round-trip passes",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/TinyStories-valid.txt"))
    parser.add_argument("--pretrained-path", type=Path, default=Path("data/glove-wiki-gigaword-50.gz"))
    parser.add_argument("--stories", type=int, default=DEFAULT_STORY_COUNT)
    parser.add_argument("--story-token-limit", type=int, default=DEFAULT_STORY_TOKEN_LIMIT)
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--controller-steps", type=int, default=DEFAULT_CONTROLLER_STEPS)
    parser.add_argument("--vsa-dimension", type=int, default=DEFAULT_VSA_DIMENSION)
    parser.add_argument("--promotion-limit", type=int, default=DEFAULT_PROMOTION_LIMIT)
    parser.add_argument("--resource-root", type=Path, default=Path("/"))
    parser.add_argument("--output", type=Path, default=Path("results/phase_e_stage1.json"))
    args = parser.parse_args()
    print(json.dumps(run_stage(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
