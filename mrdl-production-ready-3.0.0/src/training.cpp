#include "mrdl/training.hpp"
#include "mrdl/version.hpp"

#include <set>

namespace mrdl {
namespace {

class SnapshotRelationStore final : public IRelationStore {
public:
    explicit SnapshotRelationStore(const ReplayClosure& closure) {
        for (const auto& snapshot : closure.relation_snapshots) {
            auto relation = std::make_shared<const RelationRecord>(RelationRecord::deserialize(snapshot.payload));
            // A closure can contain multiple versions of one edge. The version requested by
            // relation_versions wins; the root's audited version is supplied separately as a temporary edge.
            const auto requested = std::find(closure.relation_versions.begin(), closure.relation_versions.end(),
                std::pair<RelationId, std::uint64_t>{snapshot.relation, snapshot.version});
            if (requested == closure.relation_versions.end()) continue;
            const auto existing = relations_.find(relation->id);
            if (existing != relations_.end() && existing->second->version >= relation->version) continue;
            relations_[relation->id] = relation;
        }
        for (const auto& [id, relation] : relations_) {
            full_[relation->source].push_back(id);
            if (relation->level == MemoryLevel::M2) clean_[relation->source].push_back(id);
            between_[RelationKey{relation->source, relation->destination, relation->prototype}] = id;
        }
    }

    std::vector<std::shared_ptr<const RelationRecord>> outgoing(Lane lane, NodeId source, std::size_t limit) const override {
        const auto& index = lane == Lane::Full ? full_ : clean_;
        const auto it = index.find(source);
        if (it == index.end()) return {};
        std::vector<std::shared_ptr<const RelationRecord>> result;
        result.reserve(it->second.size());
        for (const auto id : it->second) result.push_back(relations_.at(id));
        std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs->confidence != rhs->confidence) return lhs->confidence > rhs->confidence;
            return lhs->id < rhs->id;
        });
        if (limit != 0U && result.size() > limit) result.resize(limit);
        return result;
    }

    std::shared_ptr<const RelationRecord> get(RelationId id) const override {
        const auto it = relations_.find(id);
        return it == relations_.end() ? std::shared_ptr<const RelationRecord>{} : it->second;
    }

    std::vector<std::shared_ptr<const RelationRecord>> between(NodeId source, NodeId destination) const override {
        std::vector<std::shared_ptr<const RelationRecord>> result;
        for (std::uint16_t prototype = 0; prototype < 256U; ++prototype) {
            const auto it = between_.find(RelationKey{source, destination, static_cast<std::uint8_t>(prototype)});
            if (it != between_.end()) result.push_back(relations_.at(it->second));
        }
        return result;
    }

    GraphStats stats() const override {
        GraphStats stats;
        stats.relations_total = relations_.size();
        for (const auto& [_, relation] : relations_) {
            if (relation->level == MemoryLevel::M2) ++stats.relations_m2;
            else if (relation->level == MemoryLevel::M1) ++stats.relations_m1;
        }
        for (const auto& [_, ids] : full_) stats.full_index_entries += ids.size();
        for (const auto& [_, ids] : clean_) stats.clean_index_entries += ids.size();
        stats.nodes_with_full_edges = full_.size();
        stats.nodes_with_clean_edges = clean_.size();
        return stats;
    }

private:
    std::unordered_map<RelationId, std::shared_ptr<const RelationRecord>> relations_;
    std::unordered_map<NodeId, std::vector<RelationId>> full_;
    std::unordered_map<NodeId, std::vector<RelationId>> clean_;
    std::unordered_map<RelationKey, RelationId, RelationKeyHash> between_;
};

double prediction_nll(const LanePrediction& prediction, TokenId target) {
    constexpr float missing_score = -20.0F;
    float maximum = missing_score;
    for (const auto& candidate : prediction.candidates) maximum = std::max(maximum, candidate.score);
    const float target_score = prediction.score_of(target, missing_score);
    maximum = std::max(maximum, target_score);
    double denominator = std::exp(static_cast<double>(missing_score - maximum));  // unseen mass bucket
    bool target_seen = false;
    for (const auto& candidate : prediction.candidates) {
        denominator += std::exp(static_cast<double>(candidate.score - maximum));
        if (candidate.token == target) target_seen = true;
    }
    if (!target_seen) denominator += std::exp(static_cast<double>(target_score - maximum));
    return std::log(denominator) + static_cast<double>(maximum - target_score);
}

std::optional<CandidateScore> find_candidate(const LanePrediction& prediction, TokenId target) {
    const auto it = std::find_if(prediction.candidates.begin(), prediction.candidates.end(),
        [&](const CandidateScore& candidate) { return candidate.token == target; });
    return it == prediction.candidates.end() ? std::optional<CandidateScore>{} : std::optional<CandidateScore>{*it};
}

bool close_enough(float lhs, float rhs, float tolerance = 1.0e-5F) noexcept {
    const float scale = std::max({1.0F, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= tolerance * scale;
}

bool audit_frontier_certified(const LanePrediction& prediction, float epsilon) noexcept {
    if (!prediction.metrics.empty) return prediction.shadow.certified(prediction.margin, epsilon);

    // Cold-start CLEAN can be a genuinely empty counterfactual control. That case has
    // no eligible edge, no gate/operator call and no discarded branch, so there is no
    // hidden frontier capable of changing the ablation result. Treating its numeric
    // margin as zero creates a permanent M2 bootstrap deadlock. A collapsed CLEAN lane
    // that retrieved or evaluated anything is deliberately NOT granted this exemption.
    return prediction.candidates.empty() && prediction.rounds.empty() &&
           prediction.metrics.candidate_retrievals == 0U &&
           prediction.metrics.gate_evaluations == 0U &&
           prediction.metrics.operator_evaluations == 0U &&
           prediction.shadow.discarded.empty() &&
           prediction.shadow.maximum_influence == 0.0F;
}

bool replay_matches_clean(const LanePrediction& prediction,
                          const ReplayClosure& closure,
                          const ReplayRecorder& replay) {
    if (prediction.execution.operation_id != closure.operation_id) return false;
    std::unordered_map<std::uint32_t, ReplayStep> stored;
    for (const ReplayId replay_id : closure.replay_steps) {
        const auto step = replay.get(replay_id);
        if (!step || step->operation_id != closure.operation_id ||
            step->controller_version != closure.controller_version) return false;
        stored[step->depth] = *step;
    }
    for (const auto& round : prediction.rounds) {
        const auto it = stored.find(round.depth);
        if (it == stored.end()) return false;
        const auto& lane = it->second.lanes[static_cast<std::size_t>(Lane::Clean)];
        if (lane.fold_budget != round.fold_budget || lane.gate_decisions != round.gate_decisions ||
            lane.survivor_ids != round.survivor_ids || lane.candidate_set_hash != round.candidate_set_hash ||
            !close_enough(lane.shadow_upper_bound, round.shadow_upper_bound)) return false;
        stored.erase(it);
    }
    // A ReplayStep can exist for a later FULL-only round. Its CLEAN payload must be empty.
    for (const auto& [_, step] : stored) {
        const auto& lane = step.lanes[static_cast<std::size_t>(Lane::Clean)];
        if (!lane.gate_decisions.empty() || lane.fold_budget != 0U || !lane.survivor_ids.empty() ||
            lane.candidate_set_hash != 0U || !close_enough(lane.shadow_upper_bound, 0.0F)) return false;
    }
    return true;
}

}  // namespace

double TrainStats::average_loss() const noexcept { return tokens == 0U ? 0.0 : negative_log_likelihood / static_cast<double>(tokens); }
double TrainStats::perplexity() const noexcept { return std::exp(std::min(average_loss(), 80.0)); }
double TrainStats::accuracy() const noexcept { return tokens == 0U ? 0.0 : static_cast<double>(correct) / static_cast<double>(tokens); }
double TrainStats::tokens_per_second() const noexcept { return elapsed_seconds <= 0.0 ? 0.0 : static_cast<double>(tokens) / elapsed_seconds; }

void ModelRuntime::prepare(const AppConfig& config,
                           const std::filesystem::path& corpus,
                           EmbeddingInit embedding_mode,
                           const std::optional<std::filesystem::path>& external_embeddings) {
    config.validate();
    const bool already_exists = std::filesystem::exists(config.persistence.tokenizer) ||
                                std::filesystem::exists(config.persistence.embeddings) ||
                                std::filesystem::exists(config.persistence.database);
    require(!already_exists, "prepared model files already exist; back up and remove them explicitly before reinitializing");
    std::filesystem::create_directories(config.persistence.model_dir);
    ScopeExit rollback_partial([&] {
        std::error_code ignored;
        for (const auto& path : std::array<std::filesystem::path, 7>{
                 config.persistence.tokenizer,
                 config.persistence.embeddings,
                 config.persistence.database,
                 std::filesystem::path(config.persistence.database.string() + "-wal"),
                 std::filesystem::path(config.persistence.database.string() + "-shm"),
                 config.persistence.model_dir / "config.effective.ini",
                 config.persistence.model_dir / "manifest.json"}) {
            std::filesystem::remove(path, ignored);
            ignored.clear();
        }
    });
    const auto tokenizer = HybridTokenizer::build_from_corpus(corpus, config.tokenizer);
    tokenizer.save(config.persistence.tokenizer);
    EmbeddingBuildOptions options;
    options.mode = embedding_mode;
    options.dimension = config.model.embedding_dim;
    options.seed = config.model.seed;
    options.external_f32 = external_embeddings;
    FrozenEmbeddingStore::build(config.persistence.embeddings, tokenizer, corpus, options);
    auto database = std::make_shared<SqliteModelStore>(config.persistence.database,
                                                       config.persistence.sqlite_busy_timeout_ms,
                                                       config.persistence.synchronous_full);
    static constexpr std::string_view family = "MRDL-3-production-core";
    static constexpr std::string_view build_version = MRDL_VERSION_STRING;
    database->set_meta("model_family", std::as_bytes(std::span(family.data(), family.size())));
    database->set_meta("created_by_version", std::as_bytes(std::span(build_version.data(), build_version.size())));
    BinaryWriter metadata;
    metadata.pod(config.model.embedding_dim);
    metadata.pod(config.model.relation_dim);
    metadata.pod(config.model.max_relation_prototypes);
    metadata.pod(config.model.seed);
    const auto data = metadata.take();
    database->set_meta("model_config", data);
    database->checkpoint_wal();
    config.save(config.persistence.model_dir / "config.effective.ini");
    rollback_partial.dismiss();
}

ModelRuntime::ModelRuntime(AppConfig config)
    : config_(std::move(config)),
      tokenizer_(HybridTokenizer::load(config_.persistence.tokenizer)),
      embeddings_(FrozenEmbeddingStore::load(config_.persistence.embeddings)),
      persistence_(std::make_shared<SqliteModelStore>(config_.persistence.database,
                                                      config_.persistence.sqlite_busy_timeout_ms,
                                                      config_.persistence.synchronous_full)),
      graph_(persistence_),
      replay_(std::make_shared<ReplayRecorder>(persistence_)),
      metrics_(std::make_shared<MetricsRegistry>()) {
    if (config_.runtime.threads < 2U) config_.engine.parallel_lanes = false;
    require(tokenizer_.size() == embeddings_.token_count(), "tokenizer/embedding vocabulary mismatch");
    require(embeddings_.dimension() == config_.model.embedding_dim, "embedding dimension does not match config");
    static constexpr std::string_view expected_family = "MRDL-3-production-core";
    const auto family = persistence_->get_meta("model_family");
    require(family.has_value() && family->size() == expected_family.size() &&
            std::equal(family->begin(), family->end(),
                       std::as_bytes(std::span(expected_family.data(), expected_family.size())).begin()),
            "model database belongs to an incompatible runtime family");
    const auto model_metadata = persistence_->get_meta("model_config");
    require(model_metadata.has_value(), "model database is missing model_config metadata");
    BinaryReader metadata(*model_metadata);
    require(metadata.pod<std::uint32_t>() == config_.model.embedding_dim,
            "configured embedding_dim differs from prepared model");
    require(metadata.pod<std::uint32_t>() == config_.model.relation_dim,
            "configured relation_dim differs from prepared model");
    require(metadata.pod<std::uint32_t>() == config_.model.max_relation_prototypes,
            "configured max_relation_prototypes differs from prepared model");
    require(metadata.pod<std::uint64_t>() == config_.model.seed,
            "configured seed differs from prepared model");
    require(metadata.empty(), "trailing model_config metadata");
    for (auto& relation : persistence_->load_relations()) graph_.load_relation(std::move(relation));
    if (const auto controller = persistence_->load_controller()) controller_.restore(*controller);
    if (const auto roles = persistence_->load_role_inducer()) roles_.restore(*roles);
    promotion_ = std::make_unique<PromotionManager>(graph_, *replay_, controller_, roles_, persistence_);
    promotion_->load();
    engine_ = std::make_unique<DualLaneEngine>(graph_, embeddings_, controller_, roles_, config_.engine, replay_, metrics_);
}

std::unique_ptr<ModelRuntime> ModelRuntime::open(AppConfig config) {
    config.validate();
    return std::unique_ptr<ModelRuntime>(new ModelRuntime(std::move(config)));
}

double ModelRuntime::sparse_nll(const LanePrediction& prediction, TokenId target) const {
    return prediction_nll(prediction, target);
}

std::vector<float> ModelRuntime::contextual_source_vector(std::span<const TokenId> context,
                                                          std::size_t source_position) const {
    require(source_position < context.size(), "source position out of range");
    std::vector<float> result(embeddings_.dimension());
    embeddings_.dequantize(context[source_position], result);
    std::vector<float> neighbor(embeddings_.dimension());
    const std::size_t left = source_position > 3U ? source_position - 3U : 0U;
    float weight_sum = 1.0F;
    for (std::size_t i = left; i < context.size(); ++i) {
        if (i == source_position) continue;
        embeddings_.dequantize(context[i], neighbor);
        const float distance = static_cast<float>(i > source_position ? i - source_position : source_position - i);
        const float weight = 0.15F / std::max(distance, 1.0F);
        for (std::size_t d = 0; d < result.size(); ++d) result[d] += weight * neighbor[d];
        weight_sum += weight;
    }
    for (float& value : result) value /= weight_sum;
    normalize_in_place(result);
    return result;
}

ReplayClosure ModelRuntime::build_closure(const RelationRecord& root,
                                          const DualPrediction& prediction,
                                          std::span<const RelationSnapshot> prediction_snapshots) {
    ReplayClosure closure;
    closure.root_relation = root.id;
    closure.root_version = root.version;
    closure.operation_id = prediction.operation_id;
    closure.base_seed = prediction.base_seed;
    closure.replay_steps = prediction.replay_ids;
    closure.controller_version = controller_.snapshot().version;
    closure.controller_snapshot = controller_.serialize();
    closure.role_snapshot = roles_.serialize();

    if (closure.replay_steps.empty()) {
        ReplayStep step;
        step.operation_id = prediction.operation_id;
        step.controller_version = closure.controller_version;
        step.deterministic_seed = prediction.base_seed;
        step.relation_versions.emplace_back(root.id, root.version);
        closure.replay_steps.push_back(replay_->record(std::move(step)));
    }

    std::set<std::pair<RelationId, std::uint64_t>> versions;
    for (std::size_t i = 0; i < closure.replay_steps.size(); ++i) {
        const auto step = replay_->get(closure.replay_steps[i]);
        require(step.has_value(), "replay step disappeared while creating closure");
        versions.insert(step->relation_versions.begin(), step->relation_versions.end());
        closure.deterministic_seeds.push_back(step->deterministic_seed);
    }
    versions.emplace(root.id, root.version);
    closure.relation_versions.assign(versions.begin(), versions.end());

    for (const auto& [relation_id, version] : closure.relation_versions) {
        std::vector<std::byte> payload;
        if (relation_id == root.id && root.version == version) {
            payload = root.serialize();
        } else {
            const auto captured = std::find_if(prediction_snapshots.begin(), prediction_snapshots.end(),
                [&](const RelationSnapshot& snapshot) {
                    return snapshot.relation == relation_id && snapshot.version == version;
                });
            if (captured != prediction_snapshots.end()) {
                payload = captured->payload;
            } else {
                const auto relation = graph_.get(relation_id);
                if (!relation || relation->version != version) {
                    throw Error("cannot snapshot exact relation version for replay closure");
                }
                payload = relation->serialize();
            }
        }
        closure.relation_snapshots.push_back(RelationSnapshot{relation_id, version, payload});
        closure.snapshot_hashes.push_back(hash_bytes(payload));
    }
    closure.binding_hashes.push_back(hash_combine(root.source, root.destination));
    require(closure.complete(), "constructed replay closure is incomplete");
    return closure;
}

std::optional<RelationId> ModelRuntime::learn_mode_b(std::span<const TokenId> context,
                                                     TokenId target,
                                                     const DualPrediction& prediction,
                                                     std::string_view source_name) {
    if (context.empty()) return std::nullopt;
    const std::size_t source_count = std::min<std::size_t>(config_.training.max_source_capsules, context.size());
    std::optional<RelationId> last;
    std::vector<float> target_vector(embeddings_.dimension());
    embeddings_.dequantize(target, target_vector);

    // Capture every exact relation version used by the prediction before the first
    // fast-memory write. A single token can update multiple source relations; without
    // this snapshot, the first update can erase a historical version needed by the
    // replay closure of a later update from the same prediction.
    std::vector<RelationSnapshot> prediction_snapshots;
    std::set<std::pair<RelationId, std::uint64_t>> captured_versions;
    for (const ReplayId replay_id : prediction.replay_ids) {
        const auto step = replay_->get(replay_id);
        require(step.has_value(), "replay step disappeared before snapshot capture");
        for (const auto& relation_version : step->relation_versions) {
            if (!captured_versions.insert(relation_version).second) continue;
            const auto relation = graph_.get(relation_version.first);
            require(relation && relation->version == relation_version.second,
                    "prediction relation changed before replay snapshot capture");
            prediction_snapshots.push_back(RelationSnapshot{
                relation->id, relation->version, relation->serialize()
            });
        }
    }

    for (std::size_t offset = 0; offset < source_count; ++offset) {
        const std::size_t position = context.size() - 1U - offset;
        const TokenId source = context[position];
        const auto source_vector = contextual_source_vector(context, position);
        auto existing = graph_.between(source, target);

        std::shared_ptr<const RelationRecord> best_m1;
        float best_m1_error = std::numeric_limits<float>::infinity();
        float best_any_error = std::numeric_limits<float>::infinity();
        std::unordered_set<std::uint8_t> used_prototypes;
        for (const auto& relation : existing) {
            used_prototypes.insert(relation->prototype);
            const float error = relation->transform.normalized_error(source_vector, target_vector);
            best_any_error = std::min(best_any_error, error);
            if (relation->level == MemoryLevel::M1 && relation->escrow_state == EscrowState::Active && error < best_m1_error) {
                best_m1_error = error;
                best_m1 = relation;
            }
        }

        RelationRecord updated;
        if (best_m1) {
            updated = *best_m1;
        } else {
            std::optional<std::uint8_t> prototype;
            for (std::uint32_t candidate = 0; candidate < config_.model.max_relation_prototypes; ++candidate) {
                if (!used_prototypes.contains(static_cast<std::uint8_t>(candidate))) {
                    prototype = static_cast<std::uint8_t>(candidate);
                    break;
                }
            }
            if (!prototype) {
                // All K prototypes are consolidated and already explain the event: no unsafe M2 mutation.
                if (best_any_error < 0.40F) continue;
                continue;
            }
            updated.id = graph_.allocate_id();
            updated.source = source;
            updated.destination = target;
            updated.prototype = *prototype;
            updated.level = MemoryLevel::M1;
            updated.lanes = LaneMask::from_level(MemoryLevel::M1);
            updated.support = 0U;
            updated.confidence = 0.08F;
            updated.version = 1U;
            updated.created_at_ms = unix_millis();
            updated.transform = MonomialOperator::seeded(embeddings_.dimension(),
                hash_combine(hash_combine(config_.model.seed, source), hash_combine(target, *prototype)));
            updated.relation = RelationVector(config_.model.relation_dim);
        }

        updated.transform.update_delta(source_vector, target_vector,
                                       config_.training.fast_learning_rate,
                                       config_.training.relation_weight_decay);
        const auto slot = structural_slot_key(context, position, hash_floats(source_vector));
        updated.relation.update_observation(source_vector, target_vector, slot,
                                            static_cast<std::uint32_t>(offset + 1U),
                                            hash_combine(slot, target), target == kEosToken,
                                            updated.confidence, config_.training.fast_learning_rate);
        ++updated.support;
        updated.confidence = std::min(config_.memory.m1_confidence_cap,
                                      0.08F + 0.10F * std::log1p(static_cast<float>(updated.support)));
        updated.expires_at_ms = unix_millis() + config_.memory.m1_ttl_seconds * 1000;
        updated.escrow_state = EscrowState::Active;
        ++updated.version;
        updated.updated_at_ms = unix_millis();
        // Build the closure while the graph still contains every exact pre-update version
        // referenced by the prediction. The new root version is embedded from `updated`.
        auto closure = build_closure(updated, prediction, prediction_snapshots);
        graph_.upsert(updated);
        EscrowObservation observation;
        observation.contextual_key = hash_bytes(std::as_bytes(context));
        observation.observed_content = target;
        observation.context_tokens.assign(context.begin(), context.end());
        observation.bound_frame = target_vector;
        observation.active_trace = closure.replay_steps;
        observation.source = std::string(source_name);
        observation.timestamp_ms = unix_millis();
        observation.support = 1.0F;
        observation.source_position = static_cast<std::uint32_t>(position);
        promotion_->remember(updated.id, std::move(observation), std::move(closure),
                             config_.memory.m1_confidence_cap,
                             config_.memory.m1_ttl_seconds);
        last = updated.id;
    }
    return last;
}

TrainStats ModelRuntime::train(const std::filesystem::path& corpus,
                               TrainProgressCallback progress) {
    require(config_.training.mode == "B", "this production build enables Mode B; Mode A is isolated from the M1/M2 production path");
    const auto started = std::chrono::steady_clock::now();
    TrainStats stats;
    for (std::uint32_t epoch = 0; epoch < config_.training.epochs; ++epoch) {
        std::ifstream stream(corpus);
        if (!stream) throw Error("cannot open training corpus: " + corpus.string());
        std::string line;
        while (std::getline(stream, line)) {
            line.push_back('\n');
            const auto tokens = tokenizer_.encode(line, true, true);
            std::deque<TokenId> context;
            context.push_back(kBosToken);
            for (std::size_t position = 1U; position < tokens.size(); ++position) {
                const TokenId target = tokens[position];
                std::vector<TokenId> context_vector(context.begin(), context.end());

                // Replay steps, relation mutation and escrow closure for one target are one
                // durability unit. This collapses many synchronous SQLite autocommits into
                // one fsync while preserving exact replay. A failed commit is fatal; the
                // process is reopened from the rolled-back database rather than continuing
                // with potentially divergent in-memory state.
                auto token_transaction = persistence_->begin_write_transaction();
                const auto prediction = engine_->predict(context_vector, true,
                    hash_combine(config_.model.seed, stats.tokens + 1U));
                const auto learned = learn_mode_b(context_vector, target, prediction,
                                                  corpus.filename().string());
                if (learned) {
                    token_transaction.commit();
                    ++stats.m1_writes;
                } else {
                    token_transaction.rollback();
                    for (const ReplayId replay_id : prediction.replay_ids) replay_->erase(replay_id);
                }

                stats.negative_log_likelihood += sparse_nll(prediction.full, target);
                ++stats.tokens;
                if (prediction.full.selected == target) ++stats.correct;
                if (prediction.clean.metrics.empty) ++stats.clean_empty;
                context.push_back(target);
                while (context.size() > config_.training.context_tokens) context.pop_front();

                if (config_.training.auto_audit && stats.tokens % config_.training.batch_tokens == 0U) {
                    audit_pending(64U, &stats);
                    promotion_->expire_due();
                    if (progress) {
                        TrainStats snapshot = stats;
                        snapshot.elapsed_seconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started).count();
                        progress(snapshot);
                    }
                }
                if (config_.training.checkpoint_every_tokens != 0U &&
                    stats.tokens % config_.training.checkpoint_every_tokens == 0U) checkpoint();
            }
        }
    }
    if (config_.training.auto_audit) audit_pending(0U, &stats);
    checkpoint();
    const auto finished = std::chrono::steady_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(finished - started).count();
    if (progress) progress(stats);
    return stats;
}

AuditOutcome ModelRuntime::audit_record(const EscrowRecord& record) const {
    AuditOutcome outcome;
    if (record.observations.empty()) {
        outcome.reason = "no observations";
        return outcome;
    }
    std::size_t positive = 0U;
    std::size_t certified = 0U;
    float influence_sum = 0.0F;
    std::optional<ScoreFeatures> positive_features;
    std::vector<ScoreFeatures> negatives;

    const std::size_t sample_count = std::min<std::size_t>(record.observations.size(), config_.memory.audit_top_m);
    std::size_t processed_samples = 0U;
    for (std::size_t index = 0; index < sample_count; ++index) {
        const auto& observation = record.observations[index];
        const auto& closure = observation.replay_closure;
        SnapshotRelationStore snapshot_graph(closure);
        Controller historical_controller;
        historical_controller.restore(closure.controller_snapshot);
        RoleInducer historical_roles;
        historical_roles.restore(closure.role_snapshot);
        DualLaneEngine historical_engine(snapshot_graph, embeddings_, historical_controller,
                                          historical_roles, config_.engine);

        const auto root_snapshot = std::find_if(closure.relation_snapshots.begin(), closure.relation_snapshots.end(),
            [&](const RelationSnapshot& snapshot) {
                return snapshot.relation == closure.root_relation && snapshot.version == closure.root_version;
            });
        if (root_snapshot == closure.relation_snapshots.end()) {
            outcome.replay_exact = false;
            outcome.reason = "root relation snapshot is missing from replay closure";
            return outcome;
        }
        const RelationRecord temporary = RelationRecord::deserialize(root_snapshot->payload);
        const auto baseline = historical_engine.clean_replay(observation.context_tokens, nullptr,
                                                              closure.operation_id, closure.base_seed);
        if (!replay_matches_clean(baseline, closure, *replay_)) {
            outcome.replay_exact = false;
            outcome.reason = "stored CLEAN execution cannot be reproduced exactly";
            return outcome;
        }
        const auto modified = historical_engine.clean_replay(observation.context_tokens, &temporary,
                                                              closure.operation_id, closure.base_seed);
        const float influence = modified.score_of(observation.observed_content) -
                                baseline.score_of(observation.observed_content);
        influence_sum += influence;
        ++processed_samples;
        if (influence > 0.0F) ++positive;
        if (audit_frontier_certified(baseline, 0.02F) &&
            audit_frontier_certified(modified, 0.02F)) ++certified;
        if (!positive_features) {
            if (const auto candidate = find_candidate(modified, observation.observed_content)) {
                positive_features = candidate->features;
            }
        }
        for (const auto& candidate : baseline.candidates) {
            if (candidate.token != observation.observed_content && negatives.size() < config_.training.negative_samples) {
                negatives.push_back(candidate.features);
            }
        }

        if (!observation.context_tokens.empty()) {
            if (observation.source_position >= observation.context_tokens.size()) {
                outcome.replay_exact = false;
                outcome.reason = "escrow source position is outside the historical context";
                return outcome;
            }
            const auto slot = structural_slot_key(observation.context_tokens,
                                                  observation.source_position,
                                                  observation.contextual_key);
            outcome.role_observations.push_back(RoleObservation{slot, observation.observed_content,
                                                                 hash_combine(slot, observation.contextual_key)});
        }
    }

    if (processed_samples == 0U) {
        outcome.reason = "no replay sample could be audited";
        return outcome;
    }
    const float count = static_cast<float>(processed_samples);
    outcome.causal_influence = influence_sum / count;
    outcome.stability_ratio = count > 0.0F ? static_cast<float>(positive) / count : 0.0F;
    const float certification_ratio = count > 0.0F ? static_cast<float>(certified) / count : 0.0F;
    outcome.stable = outcome.stability_ratio >= config_.memory.promotion_stability_ratio && certification_ratio >= 0.50F;
    // A zero-effect edge is never promoted, even when the configured minimum is zero.
    // Zero is useful as a bootstrap threshold, not as permission to consolidate a no-op.
    outcome.accepted = outcome.stable &&
                       outcome.causal_influence > std::max(0.0F, config_.memory.promotion_min_influence);
    outcome.reason = outcome.accepted ? "counterfactual audit passed" :
                     (outcome.causal_influence < 0.0F ? "negative causal influence" : "insufficient stable influence");
    if (positive_features) outcome.positive_features = *positive_features;
    outcome.negative_features = std::move(negatives);
    return outcome;
}

std::size_t ModelRuntime::audit_pending(std::size_t max_relations, TrainStats* aggregate) {
    auto candidates = promotion_->promotion_candidates(config_.memory.promotion_min_support,
                                                        config_.memory.promotion_min_contexts);
    if (max_relations != 0U && candidates.size() > max_relations) candidates.resize(max_relations);
    std::size_t promoted = 0U;
    for (const RelationId relation : candidates) {
        if (!promotion_->reserve(relation) || !promotion_->begin_audit(relation)) continue;
        const auto record = promotion_->get(relation);
        if (!record) continue;
        const auto outcome = audit_record(*record);
        if (!outcome.replay_exact) {
            promotion_->mark_unreplayable(relation, outcome.reason);
            if (aggregate) ++aggregate->audits_rejected;
        } else if (outcome.accepted) {
            if (promotion_->complete(relation, outcome, config_.training.controller_learning_rate)) {
                ++promoted;
                if (aggregate) ++aggregate->promotions;
            }
        } else if (outcome.causal_influence < -config_.memory.promotion_min_influence) {
            promotion_->reject(relation, outcome.reason);
            if (aggregate) ++aggregate->audits_rejected;
        } else {
            promotion_->release_to_active(relation, outcome.reason);
            if (aggregate) ++aggregate->audits_deferred;
        }
    }
    return promoted;
}

EvalStats ModelRuntime::evaluate(const std::filesystem::path& corpus, std::uint64_t max_tokens) {
    const auto started = std::chrono::steady_clock::now();
    EvalStats stats;
    std::ifstream stream(corpus);
    if (!stream) throw Error("cannot open evaluation corpus: " + corpus.string());
    std::string line;
    bool stop = false;
    while (!stop && std::getline(stream, line)) {
        line.push_back('\n');
        const auto tokens = tokenizer_.encode(line, true, true);
        std::deque<TokenId> context;
        context.push_back(kBosToken);
        for (std::size_t position = 1U; position < tokens.size(); ++position) {
            const TokenId target = tokens[position];
            const std::vector<TokenId> context_vector(context.begin(), context.end());
            const auto prediction = engine_->predict(context_vector, false,
                hash_combine(config_.model.seed ^ 0x4556414cULL, stats.tokens + 1U));
            stats.full_nll += sparse_nll(prediction.full, target);
            stats.clean_nll += sparse_nll(prediction.clean, target);
            ++stats.tokens;
            if (prediction.full.selected == target) ++stats.correct_full;
            if (prediction.clean.selected == target) ++stats.correct_clean;
            if (prediction.clean.metrics.empty) ++stats.clean_empty;
            context.push_back(target);
            while (context.size() > config_.training.context_tokens) context.pop_front();
            if (max_tokens != 0U && stats.tokens >= max_tokens) { stop = true; break; }
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(finished - started).count();
    return stats;
}

GenerationResult ModelRuntime::generate(std::string_view prompt,
                                        std::uint32_t max_tokens,
                                        float temperature,
                                        std::uint64_t seed) {
    if (max_tokens == 0U) max_tokens = config_.runtime.max_generation_tokens;
    if (temperature < 0.0F) temperature = config_.runtime.temperature;
    if (seed == 0U) seed = config_.model.seed ^ static_cast<std::uint64_t>(unix_millis());
    std::mt19937_64 rng(seed);
    auto context = tokenizer_.encode(prompt, true, false);
    GenerationResult result;
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t step = 0; step < max_tokens; ++step) {
        const std::size_t begin = context.size() > config_.training.context_tokens ?
                                  context.size() - config_.training.context_tokens : 0U;
        const auto prediction = engine_->predict(std::span<const TokenId>(context).subspan(begin), false,
                                                  hash_combine(seed, step));
        TokenId selected = prediction.full.selected;
        if (temperature > 0.0F && !prediction.full.candidates.empty()) {
            const std::size_t count = std::min<std::size_t>(prediction.full.candidates.size(),
                                                            config_.runtime.top_p_candidates);
            const float maximum = prediction.full.candidates.front().score;
            std::vector<double> weights(count);
            for (std::size_t i = 0; i < count; ++i) {
                weights[i] = std::exp(static_cast<double>((prediction.full.candidates[i].score - maximum) / temperature));
            }
            std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
            selected = prediction.full.candidates[distribution(rng)].token;
        }
        result.certifications.push_back(prediction.certification);
        if (selected == kEosToken) break;
        result.generated.push_back(selected);
        context.push_back(selected);
    }
    const auto finished = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finished - started).count();
    result.tokens_per_second = seconds > 0.0 ? static_cast<double>(result.generated.size()) / seconds : 0.0;
    result.text = tokenizer_.decode(result.generated);
    return result;
}

std::size_t ModelRuntime::garbage_collect() {
    const auto expired = promotion_->expire_due();
    const auto collected = promotion_->collect_rejected_and_unreplayable(unix_millis() - 24LL * 60LL * 60LL * 1000LL);
    return expired + collected;
}

void ModelRuntime::checkpoint() {
    const auto controller = controller_.serialize();
    persistence_->save_controller(controller, controller_.snapshot().version);
    const auto roles = roles_.serialize();
    persistence_->save_role_inducer(roles);
    persistence_->checkpoint_wal();
}

void ModelRuntime::backup(const std::filesystem::path& destination_directory) {
    checkpoint();
    std::filesystem::create_directories(destination_directory);
    persistence_->backup_to(destination_directory / "mrdl.db");
    std::filesystem::copy_file(config_.persistence.tokenizer,
                               destination_directory / config_.persistence.tokenizer.filename(),
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(config_.persistence.embeddings,
                               destination_directory / config_.persistence.embeddings.filename(),
                               std::filesystem::copy_options::overwrite_existing);
}

bool ModelRuntime::integrity_check(std::string* diagnostic) const {
    return persistence_->integrity_check(diagnostic);
}

}  // namespace mrdl
