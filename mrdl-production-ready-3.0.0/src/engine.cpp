#include "mrdl/engine.hpp"

#include <set>

namespace mrdl {

class LaneWorker final {
public:
    LaneWorker() : thread_([this](std::stop_token stop) { run(stop); }) {}
    ~LaneWorker() {
        thread_.request_stop();
        condition_.notify_all();
    }

    std::future<LanePrediction> submit(std::function<LanePrediction()> function) {
        Work work;
        work.function = std::move(function);
        auto future = work.promise.get_future();
        {
            std::lock_guard lock(mutex_);
            queue_.push_back(std::move(work));
        }
        condition_.notify_one();
        return future;
    }

private:
    struct Work {
        std::function<LanePrediction()> function;
        std::promise<LanePrediction> promise;
    };

    void run(std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::optional<Work> work;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, stop, [this] { return !queue_.empty(); });
                if (stop.stop_requested()) break;
                work.emplace(std::move(queue_.front()));
                queue_.pop_front();
            }
            try {
                work->promise.set_value(work->function());
            } catch (...) {
                work->promise.set_exception(std::current_exception());
            }
        }
        std::lock_guard lock(mutex_);
        for (auto& work : queue_) {
            try { throw Error("lane worker stopped"); }
            catch (...) { work.promise.set_exception(std::current_exception()); }
        }
        queue_.clear();
    }

    std::jthread thread_;
    std::mutex mutex_;
    std::condition_variable_any condition_;
    std::deque<Work> queue_;
};

namespace {

constexpr std::size_t feature_index(ScoreFeature feature) noexcept {
    return static_cast<std::size_t>(feature);
}

float log_add_exp(float lhs, float rhs) noexcept {
    if (!std::isfinite(lhs)) return rhs;
    if (!std::isfinite(rhs)) return lhs;
    const float maximum = std::max(lhs, rhs);
    return maximum + std::log(std::exp(lhs - maximum) + std::exp(rhs - maximum));
}

std::uint64_t branch_seed(std::uint64_t operation, Lane lane, std::uint64_t ordinal) noexcept {
    std::uint64_t hash = hash_combine(operation, static_cast<std::uint8_t>(lane));
    return hash_combine(hash, ordinal);
}

float channel_cosine(std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.empty() || rhs.empty() || lhs.size() != rhs.size()) return 0.0F;
    return cosine(lhs, rhs);
}

std::vector<OpenExpectation> update_expectations(const RouteCapsule& parent,
                                                 const RelationRecord& relation,
                                                 float& coverage) {
    auto expectations = parent.open_expectations;
    coverage = 0.0F;
    if (relation.relation.closure_signal() > 0.15F && !expectations.empty()) {
        const auto strongest = std::max_element(expectations.begin(), expectations.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.strength < rhs.strength; });
        expectations.erase(strongest);
        coverage = 1.0F;
    }
    if (relation.relation.continuation_signal() > 0.15F) {
        const auto key = hash_combine(relation.id, relation.destination);
        const auto existing = std::find_if(expectations.begin(), expectations.end(),
            [&](const OpenExpectation& expectation) { return expectation.key == key; });
        if (existing == expectations.end() && expectations.size() < 8U) {
            expectations.push_back(OpenExpectation{key, std::abs(relation.relation.continuation_signal())});
        }
    }
    return expectations;
}

std::uint64_t candidate_hash(std::span<const Branch> branches) noexcept {
    std::uint64_t hash = 0x43414e44ULL;
    for (const auto& branch : branches) {
        hash = hash_combine(hash, branch.candidate);
        hash = hash_combine(hash, branch.relation_id);
    }
    return hash;
}

}  // namespace

float LanePrediction::score_of(TokenId token, float missing) const noexcept {
    const auto it = std::find_if(candidates.begin(), candidates.end(),
                                 [&](const CandidateScore& candidate) { return candidate.token == token; });
    return it == candidates.end() ? missing : it->score;
}

std::optional<std::vector<float>> PureComputeCache::get(const PureComputeKey& key) const {
    std::lock_guard lock(mutex_);
    const auto it = values_.find(key);
    return it == values_.end() ? std::optional<std::vector<float>>{} : std::optional<std::vector<float>>{it->second};
}

void PureComputeCache::put(PureComputeKey key, std::vector<float> value) {
    std::lock_guard lock(mutex_);
    if (values_.size() > 4096U) values_.clear();
    values_.emplace(std::move(key), std::move(value));
}

void PureComputeCache::clear() {
    std::lock_guard lock(mutex_);
    values_.clear();
}

LaneEngine::LaneEngine(Lane lane,
                       const IRelationStore& graph,
                       const IEmbeddingStore& embeddings,
                       const Controller& controller,
                       const RoleInducer& roles,
                       EngineConfig config,
                       std::shared_ptr<IFoldPolicy> fold_policy)
    : lane_(lane), graph_(graph), embeddings_(embeddings), controller_(controller), roles_(roles),
      config_(config), fold_policy_(std::move(fold_policy)) {
    if (!fold_policy_) fold_policy_ = std::make_shared<DiverseBeamFold>();
}

std::vector<RouteCapsule> LaneEngine::initialize_capsules(std::span<const TokenId> context) const {
    const std::size_t dimension = embeddings_.dimension();
    std::vector<RouteCapsule> capsules;
    if (context.empty()) {
        std::vector<TokenId> bos{kBosToken};
        return initialize_capsules(bos);
    }
    const std::size_t begin = context.size() > 64U ? context.size() - 64U : 0U;
    capsules.reserve(context.size() - begin);
    std::vector<float> compressed(dimension, 0.0F);
    std::uint64_t route_hash = 0x524f5554ULL;
    for (std::size_t position = begin; position < context.size(); ++position) {
        const TokenId token = context[position] < embeddings_.token_count() ? context[position] : kUnkToken;
        std::vector<float> embedding(dimension);
        embeddings_.dequantize(token, embedding);
        RouteCapsule capsule;
        capsule.id = branch_seed(route_hash, lane_, position + 1U);
        capsule.current_node = token;
        capsule.contextual_state.resize(dimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            capsule.contextual_state[i] = 0.80F * embedding[i] + 0.20F * compressed[i];
            compressed[i] = 0.85F * compressed[i] + 0.15F * embedding[i];
        }
        normalize_in_place(capsule.contextual_state);
        route_hash = hash_combine(route_hash, token);
        capsule.route_signature = route_hash;
        const std::size_t distance = context.size() - 1U - position;
        capsule.energy = std::exp(-0.18F * static_cast<float>(distance));
        capsule.composed_transform = MonomialOperator::identity(dimension);

        const auto slot = structural_slot_key(context, position, route_hash);
        if (const auto role = roles_.role_for(slot)) {
            RoleBinding binding;
            binding.role = *role;
            binding.entity = token;
            binding.bound_vector = MonomialOperator::seeded(dimension, *role).apply(embedding);
            for (std::size_t i = 0; i < dimension; ++i) {
                capsule.contextual_state[i] += 0.10F * binding.bound_vector[i];
            }
            normalize_in_place(capsule.contextual_state);
            capsule.role_bindings.push_back(std::move(binding));
        }
        capsules.push_back(std::move(capsule));
    }
    return capsules;
}

std::vector<float> LaneEngine::apply_relation(const RelationRecord& relation,
                                              std::span<const float> input,
                                              PureComputeCache* cache,
                                              std::uint64_t controller_version) const {
    const PureComputeKey key{relation.id, relation.version, controller_version,
                             hash_floats(input), relation.transform.full_hash()};
    if (cache) {
        if (auto value = cache->get(key)) return *value;
    }
    auto result = relation.transform.apply(input);
    if (cache) cache->put(key, result);
    return result;
}

LanePrediction LaneEngine::predict(std::span<const TokenId> context,
                                   const RelationRecord* temporary_relation,
                                   PureComputeCache* pure_cache,
                                   std::uint64_t operation_id,
                                   std::uint64_t deterministic_seed) const {
    const auto started = std::chrono::steady_clock::now();
    LanePrediction prediction;
    prediction.lane = lane_;
    prediction.execution.operation_id = operation_id;
    if (operation_id == 0U) operation_id = mix64(deterministic_seed ^ static_cast<std::uint64_t>(unix_millis()));
    if (deterministic_seed == 0U) deterministic_seed = hash_combine(operation_id, static_cast<std::uint8_t>(lane_));

    const std::size_t top_k = lane_ == Lane::Full ? config_.top_k_full : config_.top_k_clean;
    const std::size_t beam = lane_ == Lane::Full ? config_.beam_full : config_.beam_clean;
    auto active = initialize_capsules(context);
    // The active state is beam-bounded from the first round. Older context is already
    // compressed into each recent capsule, so retaining the highest-energy tail keeps
    // the O(G*B) memory contract without discarding all long-range signal.
    if (active.size() > beam) active.erase(active.begin(), active.end() - static_cast<std::ptrdiff_t>(beam));
    ContextHistory history(64U);
    for (const TokenId token : context) history.observe_token(token);
    OnePassPortRouter router(config_);  // Lane-local mutable state; never shared.
    const auto controller_snapshot = controller_.snapshot();

    std::unordered_map<TokenId, CandidateScore> accumulated;
    std::uint64_t branch_ordinal = 1U;
    std::optional<TokenId> prior_top;
    std::uint32_t stable_rounds = 0U;
    std::uint64_t potential_ops = 0U;
    std::size_t final_active = active.size();

    for (std::uint32_t depth = 0; depth < config_.max_rounds; ++depth) {
        if (active.empty()) break;
        prediction.metrics.active_state_peak = std::max<std::uint64_t>(prediction.metrics.active_state_peak, active.size());
        potential_ops += static_cast<std::uint64_t>(std::min(active.size(), beam)) * top_k;
        router.begin_round();
        std::vector<Branch> expanded;
        expanded.reserve(std::min<std::size_t>(active.size() * top_k, beam * top_k));
        LaneRoundReplay round;
        round.depth = depth;
        round.fold_budget = static_cast<std::uint32_t>(beam);
        round.deterministic_seed = hash_combine(deterministic_seed, depth);

        for (const auto& capsule : active) {
            const std::size_t retrieval_limit = std::max<std::size_t>(top_k * 4U, top_k);
            auto owned = graph_.outgoing(lane_, capsule.current_node, retrieval_limit);
            prediction.metrics.candidate_retrievals += owned.size();

            std::vector<const RelationRecord*> edges;
            edges.reserve(owned.size() + 1U);
            for (const auto& edge : owned) edges.push_back(edge.get());
            if (temporary_relation && temporary_relation->source == capsule.current_node &&
                std::none_of(edges.begin(), edges.end(), [&](const RelationRecord* edge) { return edge->id == temporary_relation->id; })) {
                edges.push_back(temporary_relation);
            }

            struct RankedEdge { const RelationRecord* edge; float score; };
            std::vector<RankedEdge> ranked;
            ranked.reserve(edges.size());
            for (const auto* edge : edges) {
                const float compatibility = edge->relation.context_compatibility(capsule.contextual_state,
                    capsule.route_signature, capsule.depth + 1U, capsule.route_signature);
                ranked.push_back(RankedEdge{edge, compatibility + 0.20F * safe_logit(edge->confidence)});
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.score != rhs.score) return lhs.score > rhs.score;
                return lhs.edge->id < rhs.edge->id;
            });
            if (ranked.size() > top_k) ranked.resize(top_k);  // Filtering occurs before gate/ports/beam.

            std::shared_ptr<const RelationRecord> previous;
            if (!capsule.local_contributions.empty()) previous = graph_.get(capsule.local_contributions.back());
            for (const auto& ranked_edge : ranked) {
                const RelationRecord& edge = *ranked_edge.edge;
                if (temporary_relation != &edge) {
                    require(edge.eligible(lane_), "ineligible edge reached lane hot path");
                }
                float key_compatibility = 1.0F;
                float binding_compatibility = 0.0F;
                float conflict = 0.0F;
                if (previous) {
                    key_compatibility = channel_cosine(previous->relation.composition(), edge.relation.composition());
                    binding_compatibility = channel_cosine(previous->relation.role(), edge.relation.role());
                    conflict = std::clamp(history.cycle_penalty(hash_combine(capsule.route_signature, edge.id)) * 0.25F, 0.0F, 1.0F);
                }
                const GateDecision gate = capsule.depth == 0U ? GateDecision::Compose :
                    controller_.composition_gate(key_compatibility, binding_compatibility, conflict);
                ++prediction.metrics.gate_evaluations;
                round.gate_decisions.push_back(gate);
                if (gate == GateDecision::Reject) continue;

                auto transformed = apply_relation(edge, capsule.contextual_state, pure_cache, controller_snapshot.version);
                ++prediction.metrics.operator_evaluations;
                const float operator_fit = embeddings_.cosine_row(edge.destination, transformed);
                const float relation_context = edge.relation.context_compatibility(capsule.contextual_state,
                    capsule.route_signature, capsule.depth + 1U, hash_combine(capsule.route_signature, edge.id));

                RouteCapsule next = capsule;
                next.id = branch_seed(operation_id, lane_, branch_ordinal++);
                next.current_node = edge.destination;
                next.contextual_state = std::move(transformed);
                normalize_in_place(next.contextual_state);
                next.parent_references = {capsule.id};
                next.local_contributions.push_back(edge.id);
                next.edge_confidences.push_back(edge.confidence);
                next.depth = capsule.depth + 1U;
                next.route_signature = hash_combine(capsule.route_signature, edge.id);
                next.composed_transform = MonomialOperator::compose(edge.transform, capsule.composed_transform);

                float coverage = 0.0F;
                next.open_expectations = update_expectations(capsule, edge, coverage);
                const float fit01 = std::clamp((operator_fit + 1.0F) * 0.5F, 0.0F, 1.0F);
                next.energy = capsule.energy * (0.50F + 0.50F * fit01) * (0.70F + 0.30F * edge.confidence);
                if (next.energy < config_.branch_energy_floor) continue;

                const auto port_key = make_port_key(next);
                const auto assignment = router.route(next.current_node, port_key, next.energy);
                prediction.metrics.port_assignments += assignment.count;
                if (assignment.count == 0U) continue;

                const float path_q = path_confidence(next.edge_confidences,
                                                     config_.confidence_epsilon,
                                                     config_.length_log_penalty);
                const float repetition = history.repetition_penalty(edge.destination) +
                                         (history.short_loop_detected(edge.destination) ? 1.0F : 0.0F);
                const float cycle = history.cycle_penalty(next.route_signature);
                const float saturation = history.saturation(edge.id);

                ScoreFeatures features{};
                features[feature_index(ScoreFeature::Bias)] = 1.0F;
                features[feature_index(ScoreFeature::OperatorFit)] = operator_fit;
                features[feature_index(ScoreFeature::RelationContext)] = relation_context;
                features[feature_index(ScoreFeature::Confidence)] = safe_logit(edge.confidence) / 8.0F;
                features[feature_index(ScoreFeature::Support)] = std::log1p(static_cast<float>(edge.support));
                features[feature_index(ScoreFeature::Energy)] = next.energy;
                features[feature_index(ScoreFeature::Coverage)] = coverage;
                features[feature_index(ScoreFeature::Continuation)] = edge.relation.continuation_signal();
                features[feature_index(ScoreFeature::Closure)] = edge.relation.closure_signal();
                features[feature_index(ScoreFeature::Repetition)] = config_.repetition_penalty * repetition +
                                                                    config_.saturation_penalty * saturation;
                features[feature_index(ScoreFeature::Cycle)] = config_.cycle_penalty * cycle;
                features[feature_index(ScoreFeature::PathConfidence)] = path_q;
                features[feature_index(ScoreFeature::Length)] = std::log1p(static_cast<float>(next.depth));

                const float local_score = controller_.score(features);
                next.accumulated_score = capsule.accumulated_score + local_score;
                for (std::size_t assignment_index = 0; assignment_index < assignment.count; ++assignment_index) {
                    Branch branch;
                    branch.capsule = next;
                    branch.capsule.id = branch_seed(operation_id, lane_, branch_ordinal++);
                    branch.capsule.port_id = assignment.ports[assignment_index];
                    branch.capsule.energy = assignment.energies[assignment_index];
                    branch.candidate = edge.destination;
                    branch.features = features;
                    branch.gate_decision = gate;
                    branch.relation_id = edge.id;
                    branch.score = next.accumulated_score + std::log(std::max(assignment.energies[assignment_index], 1.0e-8F));
                    expanded.push_back(std::move(branch));
                    ++prediction.metrics.branches_created;
                }
                round.relation_versions.emplace_back(edge.id, edge.version);
                round.parent_branch_ids.push_back(capsule.id);
            }
        }
        router.end_round();
        if (expanded.empty()) break;

        round.candidate_set_hash = candidate_hash(expanded);
        const auto survivors = fold_policy_->fold(expanded, beam);
        std::unordered_set<BranchId> survivor_ids;
        survivor_ids.reserve(survivors.size());
        for (const auto& branch : survivors) survivor_ids.insert(branch.capsule.id);
        for (const auto& branch : expanded) {
            if (survivor_ids.contains(branch.capsule.id)) continue;
            const float upper = std::exp(std::min(0.0F, branch.score - survivors.front().score));
            prediction.shadow.discarded.push_back(ShadowBranch{branch.capsule.id, branch.relation_id,
                                                               branch.candidate, branch.score, upper});
            prediction.shadow.maximum_influence = std::max(prediction.shadow.maximum_influence, upper);
        }
        round.shadow_upper_bound = prediction.shadow.maximum_influence;

        active.clear();
        active.reserve(survivors.size());
        for (const auto& branch : survivors) {
            active.push_back(branch.capsule);
            round.survivor_ids.push_back(branch.capsule.id);
            prediction.execution.entries.push_back(ExecutionTraceEntry{branch.capsule.id, branch.relation_id,
                                                                        branch.candidate, branch.score, depth});
            auto it = accumulated.find(branch.candidate);
            if (it == accumulated.end()) {
                accumulated.emplace(branch.candidate, CandidateScore{branch.candidate, branch.score,
                                                                      branch.relation_id, branch.features});
            } else {
                const float previous_score = it->second.score;
                it->second.score = log_add_exp(previous_score, branch.score);
                if (branch.score > previous_score) {
                    it->second.relation_id = branch.relation_id;
                    it->second.features = branch.features;
                }
            }
            history.observe_route(branch.capsule.route_signature);
            history.observe_edge(branch.relation_id);
        }
        prediction.metrics.branches_surviving += survivors.size();
        final_active = active.size();
        prediction.rounds.push_back(std::move(round));

        if (!accumulated.empty()) {
            const auto top = std::max_element(accumulated.begin(), accumulated.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.second.score < rhs.second.score; });
            if (prior_top && *prior_top == top->first) ++stable_rounds;
            else stable_rounds = 0U;
            prior_top = top->first;
        }
        float energy = 0.0F;
        for (const auto& capsule : active) energy += capsule.energy;
        if (stable_rounds >= 1U || energy < config_.branch_energy_floor) break;
    }

    prediction.final_capsules = active;
    prediction.candidates.reserve(accumulated.size());
    for (auto& [_, candidate] : accumulated) prediction.candidates.push_back(std::move(candidate));
    std::sort(prediction.candidates.begin(), prediction.candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        return lhs.token < rhs.token;
    });
    if (prediction.candidates.empty()) {
        prediction.selected = kEosToken;
        prediction.margin = 0.0F;
        prediction.metrics.empty = true;
    } else {
        prediction.selected = prediction.candidates.front().token;
        prediction.margin = prediction.candidates.size() > 1U ?
            prediction.candidates[0].score - prediction.candidates[1].score :
            std::numeric_limits<float>::infinity();
    }

    const float active_ratio = beam == 0U ? 0.0F :
        std::min(1.0F, static_cast<float>(final_active) / static_cast<float>(beam));
    const float operation_ratio = potential_ops == 0U ? 0.0F :
        std::min(1.0F, static_cast<float>(prediction.metrics.operator_evaluations) / static_cast<float>(potential_ops));
    prediction.metrics.clean_health_ratio = lane_ == Lane::Clean ? std::min(active_ratio, operation_ratio) : 1.0F;
    prediction.metrics.replay_steps = prediction.rounds.size();
    const auto finished = std::chrono::steady_clock::now();
    prediction.metrics.runtime_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
    return prediction;
}

DualLaneEngine::DualLaneEngine(const IRelationStore& graph,
                               const IEmbeddingStore& embeddings,
                               const Controller& controller,
                               const RoleInducer& roles,
                               EngineConfig config,
                               std::shared_ptr<ReplayRecorder> replay,
                               std::shared_ptr<MetricsRegistry> metrics)
    : full_(Lane::Full, graph, embeddings, controller, roles, config),
      clean_(Lane::Clean, graph, embeddings, controller, roles, config),
      config_(config), controller_(controller), replay_(std::move(replay)), metrics_(std::move(metrics)) {
    if (config_.parallel_lanes) worker_ = std::make_unique<LaneWorker>();
}

DualLaneEngine::~DualLaneEngine() = default;

DualPrediction DualLaneEngine::predict(std::span<const TokenId> context,
                                       bool persist_replay,
                                       std::uint64_t deterministic_seed) const {
    DualPrediction result;
    result.operation_id = next_operation_.fetch_add(1U, std::memory_order_relaxed);
    if (deterministic_seed == 0U) deterministic_seed = hash_combine(0x4455414cULL, result.operation_id);
    result.base_seed = deterministic_seed;
    PureComputeCache cache;
    PureComputeCache* shared_pure = config_.exact_pure_reuse ? &cache : nullptr;
    const auto started = std::chrono::steady_clock::now();
    if (worker_) {
        auto clean_future = worker_->submit([this, context, shared_pure, operation = result.operation_id, deterministic_seed] {
            return clean_.predict(context, nullptr, shared_pure, operation, deterministic_seed);
        });
        result.full = full_.predict(context, nullptr, shared_pure, result.operation_id, deterministic_seed);
        result.clean = clean_future.get();
    } else {
        result.full = full_.predict(context, nullptr, shared_pure, result.operation_id, deterministic_seed);
        result.clean = clean_.predict(context, nullptr, shared_pure, result.operation_id, deterministic_seed);
    }
    const auto finished = std::chrono::steady_clock::now();

    if (result.clean.metrics.empty) result.certification = Certification::Empty;
    else if (result.full.selected != result.clean.selected) result.certification = Certification::Provisional;
    else if (result.clean.margin >= config_.clean_margin) result.certification = Certification::Clean;
    else result.certification = Certification::Fragile;

    result.metrics.full = result.full.metrics;
    result.metrics.clean = result.clean.metrics;
    result.metrics.total_runtime_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
    result.metrics.runtime_ratio = result.full.metrics.runtime_ns == 0U ? 0.0F :
        static_cast<float>(result.metrics.total_runtime_ns) / static_cast<float>(result.full.metrics.runtime_ns);
    result.metrics.operator_ratio = result.full.metrics.operator_evaluations == 0U ? 0.0F :
        static_cast<float>(result.full.metrics.operator_evaluations + result.clean.metrics.operator_evaluations) /
        static_cast<float>(result.full.metrics.operator_evaluations);

    if (persist_replay && replay_) {
        const std::size_t rounds = std::max(result.full.rounds.size(), result.clean.rounds.size());
        result.replay_ids.reserve(rounds);
        for (std::size_t depth = 0; depth < rounds; ++depth) {
            ReplayStep step;
            step.operation_id = result.operation_id;
            step.controller_version = controller_.snapshot().version;
            step.depth = static_cast<std::uint32_t>(depth);
            step.deterministic_seed = hash_combine(deterministic_seed, depth);
            std::set<std::pair<RelationId, std::uint64_t>> versions;
            std::set<BranchId> parents;
            auto copy_lane = [&](const LaneRoundReplay& source, LaneReplayData& destination) {
                destination.gate_decisions = source.gate_decisions;
                destination.fold_budget = source.fold_budget;
                destination.survivor_ids = source.survivor_ids;
                destination.shadow_upper_bound = source.shadow_upper_bound;
                destination.candidate_set_hash = source.candidate_set_hash;
                versions.insert(source.relation_versions.begin(), source.relation_versions.end());
                parents.insert(source.parent_branch_ids.begin(), source.parent_branch_ids.end());
            };
            if (depth < result.full.rounds.size()) copy_lane(result.full.rounds[depth], step.lanes[0]);
            if (depth < result.clean.rounds.size()) copy_lane(result.clean.rounds[depth], step.lanes[1]);
            step.relation_versions.assign(versions.begin(), versions.end());
            step.parent_branch_ids.assign(parents.begin(), parents.end());
            result.replay_ids.push_back(replay_->record(std::move(step)));
        }
        result.full.execution.replay_steps.clear();
        result.clean.execution.replay_steps.clear();
    }

    if (metrics_) {
        metrics_->observe_prediction(Lane::Full, result.full.metrics);
        metrics_->observe_prediction(Lane::Clean, result.clean.metrics);
        if (result.clean.metrics.clean_health_ratio < config_.clean_health_threshold) {
            metrics_->increment("mrdl_clean_degenerate_total");
        }
    }
    return result;
}

LanePrediction DualLaneEngine::clean_only(std::span<const TokenId> context,
                                              std::uint64_t deterministic_seed) const {
    const auto operation = next_operation_.fetch_add(1U, std::memory_order_relaxed);
    return clean_.predict(context, nullptr, nullptr, operation, deterministic_seed);
}

LanePrediction DualLaneEngine::clean_with_temporary(std::span<const TokenId> context,
                                                    const RelationRecord& temporary,
                                                    std::uint64_t deterministic_seed) const {
    const auto operation = next_operation_.fetch_add(1U, std::memory_order_relaxed);
    return clean_.predict(context, &temporary, nullptr, operation, deterministic_seed);
}

LanePrediction DualLaneEngine::clean_replay(std::span<const TokenId> context,
                                            const RelationRecord* temporary,
                                            std::uint64_t operation_id,
                                            std::uint64_t deterministic_seed) const {
    require(operation_id != 0U, "replay operation id cannot be zero");
    require(deterministic_seed != 0U, "replay seed cannot be zero");
    return clean_.predict(context, temporary, nullptr, operation_id, deterministic_seed);
}

}  // namespace mrdl
