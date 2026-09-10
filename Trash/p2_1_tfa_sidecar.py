"""P2.1 TFA sidecar; read-only checkpoint evaluation.

This file is uploaded to /tmp on the benchmark host and is not part of MRDL.
It imports the frozen pipeline, learns scalar-per-channel complex phase
operators from existing controller/VSA signals, and evaluates O0-O3.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/root/mrdl")
sys.path.insert(0, str(ROOT))
from benchmarks.phase_e_stage import load_pipeline_checkpoint, load_stage_stories, split_stage_stories
from mrdl.evidence import MemoryLevel
from mrdl.language import MRDLLanguagePipeline, TrigramBaseline

CHECKPOINT = ROOT / "results/phase_e_stage1_random_frozen.checkpoint.json"
CORPUS = ROOT / "data/TinyStories-valid.txt"
OUT = Path("/tmp/p2_1_tfa_results.json")


def pack(values):
    vals = list(values)
    if len(vals) % 2:
        vals.append(0.0)
    z = [complex(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    n = math.sqrt(sum(abs(v) ** 2 for v in z))
    return tuple(v / n for v in z) if n > 0 else tuple(z)


def phase_align(left, right):
    nl = math.sqrt(sum(abs(v) ** 2 for v in left))
    nr = math.sqrt(sum(abs(v) ** 2 for v in right))
    if nl <= 1e-15 or nr <= 1e-15:
        return 0.0
    return sum((a.conjugate() * b).real for a, b in zip(left, right)) / (nl * nr)


def add_vec(a, b):
    return tuple(x + y for x, y in zip(a, b))


def mul_phase(phase, x):
    return tuple(p * v for p, v in zip(phase, x))


def state_signal(pipeline, edge, source, target, cache):
    """Pack same ten VSA/controller channels from frozen edge-local state."""
    continuation = cache.setdefault(("continuation", source), pipeline.vsa_memory.continuation_scores((source,)))
    cleanup = cache.setdefault(("cleanup", source), pipeline.vsa_memory.recover(source))
    edge_count = max(1, len(pipeline.edge_memory.edges))
    return pack((
        1.0 / max(1, pipeline.composition_depth),
        max(0.0, float(continuation.get(target, 0.0))),
        max(0.0, 1.0 - float(cleanup.noise)),
        max(0.0, min(1.0, float(edge.get("role_score", 1.0)))),
        min(1.0, pipeline.vsa_memory.binding_count / edge_count),
        1.0 / max(1, pipeline.composition_depth),
        1.0 if target in continuation else 0.0,
        min(1.0, abs(float(edge.get("relation", 0.0)))),
        min(1.0, max(0.0, float(edge.get("confidence", 0.0)))),
        1.0,
    ))


def baseline_report(train, test):
    baseline = TrigramBaseline()
    baseline.train(train)
    total = correct = 0
    for story in test:
        context = ["<bos>"]
        for target in story[1:]:
            prediction = baseline.predict(context)
            correct += prediction.token == target
            total += 1
            context.append(target)
    return {"tokens": total, "correct": correct, "accuracy": correct / max(1, total)}


def categorize(pipeline, prediction, source, target):
    route_map = dict(prediction.candidate_record_ids)
    route = route_map.get(prediction.token, ())
    fallback = prediction.fallback != "none"
    if fallback:
        bucket = "fallback"
    elif len(route_map.get(target, ())) > 1:
        bucket = "composed"
    else:
        bucket = "direct"
    target_route = route_map.get(target, ())
    return bucket, bool(prediction.token == target), (1 if bucket == "direct" else len(target_route)), prediction.actual_rounds


def evaluate(pipeline, test):
    counts = Counter()
    correct = Counter()
    depths = defaultdict(lambda: [0, 0])
    total = 0
    for story in test:
        context = ["<bos>"]
        for target in story[1:]:
            prediction = pipeline.predict(context)
            bucket, hit, route_len, rounds = categorize(pipeline, prediction, context[-1], target)
            counts[bucket] += 1
            correct[bucket] += int(hit)
            depths[str(min(4, max(1, route_len if route_len else rounds)))][0] += 1
            depths[str(min(4, max(1, route_len if route_len else rounds)))][1] += int(hit)
            total += 1
            context.append(target)
    return {
        "tokens": total,
        "counts": dict(counts),
        "correct": dict(correct),
        "accuracy": sum(correct.values()) / max(1, total),
        "accuracy_by_bucket": {k: correct[k] / max(1, counts[k]) for k in counts},
        "fraction_by_bucket": {k: counts[k] / max(1, total) for k in counts},
        "depth_counts": {k: v[0] for k, v in sorted(depths.items())},
        "depth_correct": {k: v[1] for k, v in sorted(depths.items())},
        "depth_accuracy": {k: v[1] / max(1, v[0]) for k, v in sorted(depths.items())},
    }


class TFA:
    def __init__(self, pipeline: MRDLLanguagePipeline, models, variant):
        self.pipeline = pipeline
        self.models = models
        self.variant = variant
        self._cleanup_cache = {}
        self._continuation_cache = {}
        self._role_cache = {str(e["record_id"]): float(e.get("role_score", 1.0)) for e in pipeline.edge_memory.edges.values()}

    def signal(self, depth, candidate, seeds):
        key = tuple(seeds)
        continuation_scores = self._continuation_cache.setdefault(key, self.pipeline.vsa_memory.continuation_scores(seeds))
        role_scores = []
        for record_id in candidate.route_record_ids:
            if record_id in self._role_cache:
                role_scores.append(self._role_cache[record_id])
        role = sum(role_scores) / max(1, len(role_scores))
        cleanup = self._cleanup_cache.setdefault(candidate.node_id, self.pipeline.vsa_memory.recover(candidate.node_id))
        binding = min(1.0, self.pipeline.vsa_memory.binding_count / max(1, len(self.pipeline.edge_memory.edges)))
        route = min(1.0, len(candidate.route_record_ids) / max(1, self.pipeline.composition_depth))
        coverage = 1.0 if candidate.node_id in continuation_scores else 0.0
        operator = min(1.0, abs(candidate.operator.apply(1.0)))
        score = min(1.0, max(0.0, candidate.score))
        return (depth / max(1, self.pipeline.composition_depth), max(0.0, continuation_scores.get(candidate.node_id, 0.0)), max(0.0, 1.0 - cleanup.noise), role, binding, route, coverage, operator, score, 1.0)

    def q(self, candidate, depth, seeds, continuation_scores, cleanup_cache):
        signal = self.signal(depth, candidate, seeds)
        z = pack(signal)
        qs = []
        for record_id in candidate.route_record_ids:
            model = self.models.get(record_id)
            if model is None:
                qs.append(0.0)
                continue
            if self.variant in ("O2", "O3"):
                z = mul_phase(model["phase"], z)
            if self.variant in ("O1", "O3"):
                z = add_vec(z, model["nu"])
            if self.variant == "O3":
                qs.append(max(phase_align(z, proto) for proto in model["prototypes"]))
            else:
                qs.append(phase_align(z, model["target"]))
        return min(qs, default=0.0)

    def score(self, engine, seeds=(), required_rounds=0):
        scores = defaultdict(float)
        active_ids = set()
        continuation_scores = self.pipeline.vsa_memory.continuation_scores(seeds)
        cleanup_cache = self._cleanup_cache
        frontiers = engine.round_frontiers or (engine.frontier,)
        seen = defaultdict(set)
        candidate_record_ids = defaultdict(set)
        for depth, frontier in enumerate(frontiers, 1):
            if depth <= required_rounds:
                continue
            depth_weight = 1.0 / depth
            for candidate in frontier:
                roots = candidate.provenance_roots or (f"node:{candidate.node_id}",)
                if all(root in seen[candidate.node_id] for root in roots):
                    continue
                seen[candidate.node_id].update(roots)
                candidate_record_ids[candidate.node_id].update(candidate.route_record_ids)
                q = self.q(candidate, depth, seeds, continuation_scores, cleanup_cache)
                factor = max(math.ulp(1.0), (q + 1.0) / 2.0)
                signal = self.signal(depth, candidate, seeds)
                controller_factor = max(math.ulp(1.0), self.pipeline.controller.score(signal))
                scores[candidate.node_id] += depth_weight * controller_factor * max(math.ulp(1.0), candidate.score) * factor
                active_ids.update(candidate.route_record_ids)
        for token, similarity in continuation_scores.items():
            if token in scores:
                scores[token] += 0.05 * max(0.0, similarity)
        self.pipeline._last_candidate_record_ids = tuple((t, tuple(sorted(v))) for t, v in sorted(candidate_record_ids.items()))
        return scores, tuple(sorted(active_ids))


def make_tfa_pipeline(path, models, variant):
    base = load_pipeline_checkpoint(path)
    tfa = TFA(base, models, variant)
    original = base._scores
    base._scores = lambda engine, seeds=(), required_rounds=0: tfa.score(engine, seeds, required_rounds)
    return base


def hydrate_frozen_vsa(pipeline, train):
    """Rebuild only non-serialized VSA/role state in scratch memory.

    Checkpoint intentionally serializes edges/controller, not VSA bundles.
    Hydration copies scratch VSA state back and never re-observes checkpoint
    edges, so frozen edge support/confidence/controller remain unchanged.
    """
    scratch = load_pipeline_checkpoint(CHECKPOINT)
    scratch.observe_training(train)
    pipeline.vsa_memory = scratch.vsa_memory
    pipeline.role_discovery = scratch.role_discovery
    pipeline.edge_memory.vsa_memory = pipeline.vsa_memory
    pipeline.edge_memory.role_discovery = pipeline.role_discovery
    pipeline._training_stories = tuple(tuple(story) for story in train)


def collect_models(pipeline, train):
    hydrate_frozen_vsa(pipeline, train)
    records = {(s, t): e for (s, t), e in pipeline.edge_memory.edges.items() if MemoryLevel(e["level"]) is MemoryLevel.M2}
    pipeline._ensure_engines()
    observations = defaultdict(list)
    contexts = 0
    signal_cache = {}
    for story in train:
        context = ["<bos>"]
        for index, target in enumerate(story[1:-1], 1):
            nxt = story[index + 1]
            edge = records.get((context[-1], target))
            next_edge = records.get((target, nxt))
            after_context = context + [target]
            if edge is not None and next_edge is not None:
                x = state_signal(pipeline, edge, context[-1], target, signal_cache)
                y = state_signal(pipeline, next_edge, target, nxt, signal_cache)
                observations[str(edge["record_id"])].append((x, y))
            context = after_context
            contexts += 1
    models = {}
    for rid, obs in observations.items():
        dim = len(obs[0][0])
        phase = []
        for j in range(dim):
            corr = sum(y[j] * x[j].conjugate() for x, y in obs)
            phase.append(corr / abs(corr) if abs(corr) > 1e-15 else 1 + 0j)
        targets = [y for _, y in obs]
        nu = tuple(sum(y[j] - phase[j] * x[j] for x, y in obs) / len(obs) for j in range(dim))
        # Deterministic farthest-point two-prototype split; k<=2, never random.
        prototypes = [targets[0]]
        if len(targets) > 1:
            second = max(targets[1:], key=lambda v: 1.0 - phase_align(v, targets[0]))
            prototypes.append(second)
        models[rid] = {"phase": tuple(phase), "nu": nu, "target": tuple(sum(y[j] for y in targets) / len(targets) for j in range(dim)), "prototypes": tuple(prototypes), "observations": len(obs)}
    return models, contexts, sum(len(v) for v in observations.values())


def evaluate_all(checkpoint_path, models, train, test):
    """Run one propagation per context, score O0/O1/O2/O3 on same frontiers."""
    base = load_pipeline_checkpoint(checkpoint_path)
    hydrate_frozen_vsa(base, train)
    tfa = {variant: TFA(base, models, variant) for variant in ("O1", "O2", "O3")}
    states = {v: {"counts": Counter(), "correct": Counter(), "depths": defaultdict(lambda: [0, 0])} for v in tfa}

    def consume(name, prediction):
        bucket, hit, route_len, rounds = categorize(base, prediction, context[-1], target)
        state = states[name]
        state["counts"][bucket] += 1
        state["correct"][bucket] += int(hit)
        depth = str(min(4, max(1, route_len if route_len else rounds)))
        state["depths"][depth][0] += 1
        state["depths"][depth][1] += int(hit)

    for story in test:
        context = ["<bos>"]
        for target in story[1:]:
            open_expectations = base._open_expectations(context)
            start, required = base._propagation_context(context)
            full, clean, seeds = base._run_context(context)
            propagation = dict(base._last_propagation)
            required_count = len(required)
            for variant, scorer in tfa.items():
                vscores, vactive = scorer.score(full, seeds, required_count)
                vcids = base._last_candidate_record_ids
                if vscores:
                    vprediction = base._prediction_from_scores(vscores, clean, base.composition_depth, "none", vactive, open_expectations, propagation, vcids)
                else:
                    probabilities = base.edge_memory.continuation_distribution(seeds)
                    if probabilities:
                        token = max(probabilities, key=lambda item: (probabilities[item], item))
                        vprediction = base._prediction_from_scores(probabilities, clean, base.composition_depth, "seed_unigram", vactive, open_expectations, propagation, ())
                        vprediction = vprediction.__class__(token, vprediction.probabilities, vprediction.candidates, vprediction.clean_health_ratio, True, vprediction.propagation_rounds, "seed_unigram", vprediction.active_record_ids, vprediction.open_expectations, vprediction.eos_candidate, vprediction.requested_depth, vprediction.actual_rounds, vprediction.stop_reason, vprediction.propagation_energy, vprediction.energy_drop_ratio, vprediction.distribution_stability, vprediction.novel_candidates, ())
                    else:
                        vprediction = base._prediction_from_scores({}, clean, base.composition_depth, "unknown", vactive, open_expectations, propagation, ())
                consume(variant, vprediction)
            context.append(target)

    rows = {}
    for name, state in states.items():
        counts, correct, depths = state["counts"], state["correct"], state["depths"]
        total = sum(counts.values())
        rows[name] = {"tokens": total, "counts": dict(counts), "correct": dict(correct), "accuracy": sum(correct.values()) / max(1, total), "accuracy_by_bucket": {k: correct[k] / max(1, counts[k]) for k in counts}, "fraction_by_bucket": {k: counts[k] / max(1, total) for k in counts}, "depth_counts": {k: v[0] for k, v in sorted(depths.items())}, "depth_correct": {k: v[1] for k, v in sorted(depths.items())}, "depth_accuracy": {k: v[1] / max(1, v[0]) for k, v in sorted(depths.items())}}
    return rows


def main():
    started = time.time()
    cp_bytes = CHECKPOINT.read_bytes()
    cp_sha = hashlib.sha256(cp_bytes).hexdigest()
    records = load_stage_stories(CORPUS, story_count=10000, story_token_limit=8)
    train, test, manifest = split_stage_stories(records, 1729)
    checkpoint = json.loads(cp_bytes)
    assert cp_sha == "47e3f089e1717cbbede3097fe4aafbd98c6c4d2155e06a1122c40d8420ad3940"
    assert len(records) == 10000 and len(train) == 8007 and len(test) == 1993
    assert sum(len(s) - 1 for s in test) == 13951
    assert manifest["train_story_ids"] == checkpoint["metadata"]["split_manifest"]["train_story_ids"]
    assert manifest["test_story_ids"] == checkpoint["metadata"]["split_manifest"]["test_story_ids"]
    baseline = baseline_report(train, test)
    reference_graph = json.loads((ROOT / "results/phase_e_stage1_10k.json").read_text(encoding="utf-8"))["modes"][0]["graph"]
    model_source = load_pipeline_checkpoint(CHECKPOINT)
    models, contexts, observations = collect_models(model_source, train)
    rows = evaluate_all(CHECKPOINT, models, train, test)
    result = {
        "diagnostic": "P2.1_TFA_sidecar",
        "scope": "read-only; no production source, checkpoint, Fold_B, controller, promotion, Step2, AOC/RCR/REN, or rotations changed",
        "protocol": {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": cp_sha, "corpus": str(CORPUS), "split_seed": 1729, "train_stories": len(train), "test_stories": len(test), "train_transitions": sum(len(s)-1 for s in train), "test_transitions": sum(len(s)-1 for s in test), "test_tokens": baseline["tokens"], "embedding_anchor_dimension": checkpoint["pipeline_config"]["embedding_dimension"], "context_signal_dimension": 10, "complex_residual_channels": 5, "complex_normalization": "unit-norm packed controller/VSA signal pairs"},
        "baseline_trigram": baseline,
        "baseline_o0_checkpoint": {"accuracy": reference_graph["accuracy"], "correct": int(round(reference_graph["accuracy"] * reference_graph["tokens"])), "tokens": reference_graph["tokens"], "fallback_counts": reference_graph["fallback_counts"], "fallback_fraction": reference_graph["fallback_fraction"], "source": "phase_e_stage1_10k.json random_frozen graph; same checkpoint SHA"},
        "variants": rows,
        "training": {"operator": "diagonal scalar complex phase from per-channel sum(y*conj(x)); closed-form residual nu mean(y-Ux)", "O1": "U=I, learned nu", "O2": "learned U, nu=0", "O3": "learned U + learned nu + deterministic k<=2 output prototypes", "contexts_scanned": contexts, "edge_observations": observations, "m2_edges_with_models": len(models), "unobserved_edges_identity_zero_innovation": sum(1 for e in checkpoint["edges"] if e["level"] == "M2" and str(e["record_id"]) not in models)},
        "q_tfa": {"definition": "route Q=min edge phase alignment; score replacement=(Q+1)/2, preserving nonnegative score semantics", "alignment": "Re(sum(conj(z)*prototype))/(||z|| ||prototype||)", "fallback_for_missing_model": "Q=0 => neutral factor 0.5"},
        "elapsed_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
