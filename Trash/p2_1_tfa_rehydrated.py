"""P2.1 TFA sidecar with ordered VSA rehydration.

Read-only benchmark. It deliberately does not import or modify production
source. Train replay restores VSABindingMemory using the same bind() sequence
used by the validated Fold_B accuracy diagnostic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/root/mrdl")
sys.path.insert(0, str(ROOT))

from benchmarks.phase_e_stage import load_pipeline_checkpoint, load_stage_stories, split_stage_stories
from mrdl.evidence import MemoryLevel
from mrdl.language import MRDLLanguagePipeline

CHECKPOINT = ROOT / "results/phase_e_stage1_random_frozen.checkpoint.json"
CORPUS = ROOT / "data/TinyStories-valid.txt"
OUTPUT = ROOT / "results/p2_1_tfa_rehydrated_results.json"
CHECKPOINT_SHA = "47e3f089e1717cbbede3097fe4aafbd98c6c4d2155e06a1122c40d8420ad3940"
CONTEXT_WINDOW = 2


def rehydrate_vsa(pipeline, train):
    """Replay train binds in exact story/token order; never replay test."""
    observations = defaultdict(list)
    bind_count = 0
    for story in train:
        for index in range(1, len(story)):
            source, target = story[index - 1], story[index]
            before = vsa_state(pipeline.vsa_memory, source)
            pipeline.vsa_memory.bind(source, target)
            after = vsa_state(pipeline.vsa_memory, source)
            observations[f"edge:{source}->{target}"].append((before, after))
            bind_count += 1
    return observations, bind_count


def vsa_state(vsa, key):
    """Actual bipolar binding bundle as unit-norm complex channels."""
    bundle = vsa._bundles.get(key)
    if bundle is None:
        return tuple()
    values = bundle.values
    if len(values) % 2:
        values = (*values, 1)
    z = tuple(complex(values[i], values[i + 1]) for i in range(0, len(values), 2))
    norm = math.sqrt(sum(abs(value) ** 2 for value in z))
    return tuple(value / norm for value in z) if norm else z


def align(left, right):
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    nl = math.sqrt(sum(abs(value) ** 2 for value in left))
    nr = math.sqrt(sum(abs(value) ** 2 for value in right))
    if nl <= 1e-15 or nr <= 1e-15:
        return 0.0
    return sum((a.conjugate() * b).real for a, b in zip(left, right)) / (nl * nr)


def centered(values):
    if not values:
        return values
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def phase_operator(observations):
    """Closed-form circular phase direction from centered cross-correlation."""
    cross = 0j
    for x, y in observations:
        xc, yc = centered(x), centered(y)
        n = min(len(xc), len(yc))
        cross += sum(yc[index] * xc[index].conjugate() for index in range(n))
    return cross / abs(cross) if abs(cross) > 1e-15 else 1 + 0j


def apply_phase(operator, vector):
    return tuple(operator * value for value in vector)


def add(left, right):
    n = min(len(left), len(right))
    return tuple(left[index] + right[index] for index in range(n))


def prototype_pair(values):
    """Deterministic k_in/k_out, k<=2, via farthest phase separation."""
    if not values:
        return (tuple(),)
    first = values[0]
    if len(values) == 1:
        return (first,)
    second = max(values[1:], key=lambda candidate: 1.0 - align(first, candidate))
    return (first, second)


def learn_models(observations):
    """Fit TFA models from replay snapshots without mutating VSA state."""
    models = {}
    for record_id, pairs in observations.items():
        operator = phase_operator(pairs)
        innovations = [tuple(y[index] - operator * x[index] for index in range(min(len(x), len(y)))) for x, y in pairs]
        dimension = min((len(value) for value in innovations), default=0)
        innovation = tuple(sum(value[index] for value in innovations) / len(innovations) for index in range(dimension)) if innovations else tuple()
        inputs = prototype_pair([x for x, _ in pairs])
        outputs = prototype_pair([y for _, y in pairs])
        target = tuple(sum(y[index] for _, y in pairs) / len(pairs) for index in range(min(map(len, (y for _, y in pairs)))))
        models[record_id] = {"operator": operator, "innovation": innovation, "k_in": inputs, "k_out": outputs, "target": target, "observations": len(pairs)}
    return models, sum(len(values) for values in observations.values()), len(observations)


class TFAPipeline(MRDLLanguagePipeline):
    def __init__(self, *args, models, variant, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models
        self.variant = variant
        self.context_vsa = None
        self._context_state_cache = {}
        self._cleanup_cache = {}
        self._role_map = {}

    def _state_for_context(self, seeds):
        if self.context_vsa is None:
            return tuple()
        key = tuple(seeds)
        if key not in self._context_state_cache:
            self._context_state_cache[key] = vsa_state(self.context_vsa, seeds[-1] if seeds else "<bos>")
        return self._context_state_cache[key]

    def _q_tfa(self, candidate, seeds, state=None):
        if self.variant == "O0":
            return 1.0
        use_phase = self.variant in {"O2", "O3"}
        use_innovation = self.variant in {"O1", "O3"}
        use_prototypes = self.variant == "O3"
        z = self._state_for_context(seeds) if state is None else state
        alignments = []
        for record_id in candidate.route_record_ids:
            model = self.models.get(record_id)
            if model is None:
                alignments.append(0.0)
                continue
            input_state = z
            if use_phase:
                z = apply_phase(model["operator"], z)
            if use_innovation:
                z = add(z, model["innovation"])
            targets = model["k_out"] if use_prototypes else (model["target"],)
            output_alignment = max((align(z, target) for target in targets), default=0.0)
            if use_prototypes:
                input_alignment = max((align(input_state, source) for source in model["k_in"]), default=0.0)
                alignments.append(0.5 * (input_alignment + output_alignment))
            else:
                alignments.append(output_alignment)
        return min(alignments, default=0.0)

    def _scores(self, engine, seeds=(), required_rounds=0):
        scores = defaultdict(float)
        active_ids = set()
        continuation_scores = self.vsa_memory.continuation_scores(seeds)
        if not self._role_map:
            self._role_map = {str(edge["record_id"]): float(edge.get("role_score", 1.0)) for edge in self.edge_memory.edges.values()}
        cleanup_cache = self._cleanup_cache
        frontiers = engine.round_frontiers or (engine.frontier,)
        seen_provenance_by_node = defaultdict(set)
        candidate_record_ids = defaultdict(set)
        for depth, frontier in enumerate(frontiers, start=1):
            if depth <= required_rounds:
                continue
            depth_weight = 1.0 / depth
            for candidate in frontier:
                roots = candidate.provenance_roots or (f"node:{candidate.node_id}",)
                if all(root in seen_provenance_by_node[candidate.node_id] for root in roots):
                    continue
                seen_provenance_by_node[candidate.node_id].update(roots)
                candidate_record_ids[candidate.node_id].update(candidate.route_record_ids)
                role_scores = []
                for record_id in candidate.route_record_ids:
                    if record_id in self._role_map:
                        role_scores.append(self._role_map[record_id])
                role = sum(role_scores) / max(1, len(role_scores))
                cleanup = cleanup_cache.setdefault(candidate.node_id, self.vsa_memory.recover(candidate.node_id))
                binding = min(1.0, self.vsa_memory.binding_count / max(1, len(self.edge_memory.edges)))
                route = min(1.0, len(candidate.route_record_ids) / max(1, self.composition_depth))
                signal = (depth / max(1, self.composition_depth), max(0.0, continuation_scores.get(candidate.node_id, 0.0)), max(0.0, 1.0 - cleanup.noise), role, binding, route, 1.0 if candidate.node_id in continuation_scores else 0.0, min(1.0, abs(candidate.operator.apply(1.0))), min(1.0, max(0.0, candidate.score)), 1.0)
                controller_factor = max(math.ulp(1.0), self.controller.score(signal))
                q = self._q_tfa(candidate, seeds)
                q_factor = max(math.ulp(1.0), (q + 1.0) / 2.0)
                operator_magnitude = max(math.ulp(1.0), abs(candidate.operator.apply(1.0)))
                scores[candidate.node_id] += depth_weight * controller_factor * max(math.ulp(1.0), candidate.score) * operator_magnitude * q_factor
                active_ids.update(candidate.route_record_ids)
        for token, similarity in continuation_scores.items():
            if token in scores:
                scores[token] += 0.05 * max(0.0, similarity)
        self._last_candidate_record_ids = tuple((token, tuple(sorted(record_ids))) for token, record_ids in sorted(candidate_record_ids.items()))
        return scores, tuple(sorted(active_ids))


def group_name(prediction, context):
    if prediction.fallback != "none":
        return "fallback"
    route_ids = dict(prediction.candidate_record_ids).get(prediction.token, ())
    edge_hops = sum(record_id.startswith("edge:") for record_id in route_ids)
    required_count = max(0, min(len(context), CONTEXT_WINDOW) - 1)
    true_hops = edge_hops - required_count
    if edge_hops == 0:
        return "continuation_only"
    return "direct" if true_hops <= 1 else "composed"


def evaluate_baseline(test, pipeline):
    groups = {name: [0, 0] for name in ("direct", "composed", "continuation_only", "fallback")}
    total = correct = 0
    for story in test:
        context = ["<bos>"]
        for target in story[1:]:
            prediction = pipeline.predict(context)
            hit = int(prediction.token == target)
            name = group_name(prediction, context)
            groups[name][0] += 1
            groups[name][1] += hit
            total += 1
            correct += hit
            context.append(target)
    return {"total": total, "correct": correct, "accuracy": correct / total, "groups": {name: {"count": count, "correct": hit, "accuracy": hit / count if count else None} for name, (count, hit) in groups.items()}}


def evaluate_variant(test, pipeline, train_vsa):
    groups = {name: [0, 0] for name in ("direct", "composed", "continuation_only", "fallback")}
    total = correct = 0
    for story in test:
        pipeline.context_vsa = copy.deepcopy(train_vsa)
        pipeline._context_state_cache.clear()
        context = ["<bos>"]
        for target in story[1:]:
            pipeline.context_vsa  # explicit before-state read occurs in _q_tfa
            prediction = pipeline.predict(context)
            hit = int(prediction.token == target)
            name = group_name(prediction, context)
            groups[name][0] += 1
            groups[name][1] += hit
            total += 1
            correct += hit
            pipeline.context_vsa.bind(context[-1], target)
            context.append(target)
    return {"total": total, "correct": correct, "accuracy": correct / total, "groups": {name: {"count": count, "correct": hit, "accuracy": hit / count if count else None} for name, (count, hit) in groups.items()}}


def make_tfa_pipeline(source, models, variant, vsa_memory):
    pipeline = TFAPipeline(
        source.embeddings,
        beam_width=source.beam_width,
        context_window=source.context_window,
        propagation_rounds=source.propagation_rounds,
        composition_depth=source.composition_depth,
        vsa_dimension=source.vsa_memory.dimension,
        vsa_seed=source.vsa_memory.seed,
        transport_coefficient_mode=source.edge_memory.transport_coefficient_mode,
        models=models,
        variant=variant,
    )
    pipeline.edge_memory = source.edge_memory
    pipeline.controller.weights = list(source.controller.weights)
    pipeline.controller.updates = source.controller.updates
    pipeline.vsa_memory = copy.deepcopy(vsa_memory)
    pipeline.role_discovery = source.role_discovery
    return pipeline


def smoke_q_tfa(source, models, train_vsa):
    """Exercise distinct O1/O2/O3 formulas on eight deterministic train edges."""
    control_map = {
        "O1": {"phase": False, "innovation": True, "prototypes": False},
        "O2": {"phase": True, "innovation": False, "prototypes": False},
        "O3": {"phase": True, "innovation": True, "prototypes": True},
    }
    probe_pipelines = {variant: make_tfa_pipeline(source, models, variant, train_vsa) for variant in control_map}
    candidates = []
    for record_id in sorted(models):
        source_token = record_id.split("->", 1)[0].removeprefix("edge:")
        state = vsa_state(train_vsa, source_token)
        if not state:
            continue
        values = {}
        candidate = SimpleNamespace(route_record_ids=(record_id,))
        for variant, pipeline in probe_pipelines.items():
            values[variant] = pipeline._q_tfa(candidate, (), state=state)
        vector = tuple(values[variant] for variant in control_map)
        prototype_bonus = 2.0 if len(models[record_id]["k_out"]) > 1 else 0.0
        phase_bonus = min(1.0, abs(models[record_id]["operator"] - 1.0))
        variation = len({round(value, 12) for value in vector}) + sum(abs(vector[i] - vector[j]) for i in range(3) for j in range(i + 1, 3)) + prototype_bonus + phase_bonus
        candidates.append((variation, record_id))
    edge_ids = tuple(record_id for _, record_id in sorted(candidates, key=lambda item: (-item[0], item[1]))[:8])
    assert len(edge_ids) == 8, f"Q_TFA smoke requires 8 fixed edges, got {len(edge_ids)}"
    variants = {}
    for variant in control_map:
        pipeline = make_tfa_pipeline(source, models, variant, train_vsa)
        values = {}
        for record_id in edge_ids:
            candidate = SimpleNamespace(route_record_ids=(record_id,))
            source_token = record_id.split("->", 1)[0].removeprefix("edge:")
            values[record_id] = pipeline._q_tfa(candidate, (), state=vsa_state(train_vsa, source_token))
        variants[variant] = values
    vectors = {variant: tuple(values[record_id] for record_id in edge_ids) for variant, values in variants.items()}
    assert len(set(vectors.values())) == 3, f"Q_TFA variants not distinct: {vectors}"
    assert len({tuple(sorted(control.items())) for control in control_map.values()}) == 3
    smoke = {"edges": edge_ids, "controls": control_map, "values": variants}
    print(json.dumps({"q_tfa_smoke": smoke}, sort_keys=True))
    return smoke


def main():
    started = time.time()
    checkpoint_bytes = CHECKPOINT.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    assert checkpoint_sha == CHECKPOINT_SHA
    records = load_stage_stories(CORPUS, 10000, 8)
    train, test, manifest = split_stage_stories(records, 1729)
    assert len(train) == 8007 and len(test) == 1993 and sum(len(story) - 1 for story in test) == 13951

    pipeline = load_pipeline_checkpoint(CHECKPOINT)
    observations_by_edge, bind_count = rehydrate_vsa(pipeline, train)
    expected_train_transitions = sum(len(story) - 1 for story in train)
    assert expected_train_transitions == 56049
    assert bind_count == expected_train_transitions == 56049
    models, observations, modeled_edges = learn_models(observations_by_edge)
    train_vsa = copy.deepcopy(pipeline.vsa_memory)

    if "--smoke-only" in sys.argv:
        smoke = smoke_q_tfa(pipeline, models, train_vsa)
        print(json.dumps({"status": "q_tfa_smoke_passed", "train_bind_count": bind_count, "q_tfa_smoke": smoke}, indent=2, sort_keys=True))
        return

    baseline_pipeline = make_tfa_pipeline(pipeline, {}, "O0", train_vsa)
    baseline = evaluate_baseline(test, baseline_pipeline)
    assert baseline["total"] == 13951
    assert baseline["correct"] == 4237
    assert baseline["groups"]["direct"]["count"] == 13232
    assert baseline["groups"]["direct"]["correct"] == 4120
    assert baseline["groups"]["composed"]["count"] == 278
    assert baseline["groups"]["composed"]["correct"] == 4
    assert baseline["groups"]["fallback"]["count"] == 441
    assert baseline["groups"]["fallback"]["correct"] == 113

    smoke = smoke_q_tfa(pipeline, models, train_vsa)
    gate_summary = {
        "o0_exact": baseline,
        "single_ordered_train_replay": {"bind_count": bind_count, "expected": expected_train_transitions},
        "q_tfa_smoke": smoke,
    }
    if "--gates-only" in sys.argv:
        print(json.dumps({"status": "gates_passed", "gates": gate_summary}, indent=2, sort_keys=True))
        return
    assert gate_summary["o0_exact"]["total"] == 13951
    assert gate_summary["o0_exact"]["correct"] == 4237
    assert gate_summary["single_ordered_train_replay"]["bind_count"] == 56049

    variants = {}
    for variant in ("O1", "O2", "O3"):
        tfa = make_tfa_pipeline(pipeline, models, variant, train_vsa)
        variants[variant] = evaluate_variant(test, tfa, train_vsa)

    result = {
        "diagnostic": "P2.1_TFA_rehydrated_sidecar",
        "scope": "read-only; no production source, checkpoint, lanes, controller, promotion, Step2, or rotations modified",
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha},
        "protocol": {"corpus": str(CORPUS), "stories": 10000, "train_stories": len(train), "test_stories": len(test), "split_seed": 1729, "test_transitions": 13951, "context_window": CONTEXT_WINDOW, "vsa_rehydration": "ordered pipeline.vsa_memory.bind(story[i-1], story[i]) over train only", "residual_state": "raw train-rehydrated bipolar VSA binding bundle, paired into unit-norm complex channels; test uses per-story deepcopy and binds actual target after scoring"},
        "baseline_o0": baseline,
        "training": {"observations": observations, "modeled_edges": modeled_edges, "train_bind_count": bind_count, "expected_train_transitions": expected_train_transitions, "operator": "scalar circular phase from centered complex cross-correlation", "innovation": "mean(y-Ux)", "prototypes": "deterministic k_in/k_out farthest phase pair, k<=2"},
        "gates": {"o0_exact": True, "single_ordered_train_replay": True, "q_tfa_smoke": smoke},
        "variants": variants,
        "elapsed_seconds": time.time() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result_bytes = OUTPUT.read_bytes()
    print(json.dumps({"output": str(OUTPUT), "bytes": len(result_bytes), "sha256": hashlib.sha256(result_bytes).hexdigest(), "baseline": baseline, "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
