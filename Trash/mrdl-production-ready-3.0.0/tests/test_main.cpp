#include "mrdl/baselines.hpp"
#include "mrdl/config.hpp"
#include "mrdl/controller.hpp"
#include "mrdl/embeddings.hpp"
#include "mrdl/engine.hpp"
#include "mrdl/graph.hpp"
#include "mrdl/persistence.hpp"
#include "mrdl/process_lock.hpp"
#include "mrdl/promotion.hpp"
#include "mrdl/relation.hpp"
#include "mrdl/replay.hpp"
#include "mrdl/routing.hpp"
#include "mrdl/tokenizer.hpp"
#include "mrdl/training.hpp"

#include <atomic>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>

namespace {

using namespace mrdl;

struct TestFailure final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CHECK(condition) do { \
    if (!(condition)) throw TestFailure(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": CHECK failed: " #condition); \
} while (false)

#define CHECK_EQ(lhs, rhs) do { \
    const auto _lhs = (lhs); const auto _rhs = (rhs); \
    if (!(_lhs == _rhs)) { \
        std::ostringstream _message; \
        _message << __FILE__ << ':' << __LINE__ << ": CHECK_EQ failed: " #lhs " != " #rhs; \
        throw TestFailure(_message.str()); \
    } \
} while (false)

#define CHECK_NEAR(lhs, rhs, tolerance) do { \
    const auto _lhs = static_cast<double>(lhs); const auto _rhs = static_cast<double>(rhs); \
    const auto _tol = static_cast<double>(tolerance); \
    if (!(std::abs(_lhs - _rhs) <= _tol * std::max({1.0, std::abs(_lhs), std::abs(_rhs)}))) { \
        std::ostringstream _message; \
        _message << std::setprecision(12) << __FILE__ << ':' << __LINE__ \
                 << ": CHECK_NEAR failed: " << _lhs << " vs " << _rhs << " tol=" << _tol; \
        throw TestFailure(_message.str()); \
    } \
} while (false)

#define CHECK_THROWS(statement) do { \
    bool _thrown = false; \
    try { (void)(statement); } catch (const std::exception&) { _thrown = true; } \
    if (!_thrown) throw TestFailure(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": expected exception: " #statement); \
} while (false)

class TempDirectory final {
public:
    TempDirectory() {
        const auto base = std::filesystem::temp_directory_path();
        for (std::uint64_t attempt = 0; attempt < 1000U; ++attempt) {
            path_ = base / ("mrdl-test-" + std::to_string(unix_millis()) + "-" +
                            std::to_string(mix64(attempt ^ reinterpret_cast<std::uintptr_t>(this))));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) return;
        }
        throw TestFailure("cannot create test temporary directory");
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
private:
    std::filesystem::path path_;
};

void write_text(const std::filesystem::path& path, std::string_view value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw TestFailure("cannot write test file");
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!stream) throw TestFailure("cannot finish writing test file");
}

class DenseEmbeddingStore final : public IEmbeddingStore {
public:
    DenseEmbeddingStore(std::size_t token_count = 1024U, std::size_t dimension = 16U)
        : token_count_(token_count), dimension_(dimension), values_(token_count * dimension) {
        for (std::size_t token = 0; token < token_count_; ++token) {
            auto row = std::span<float>(values_).subspan(token * dimension_, dimension_);
            for (std::size_t column = 0; column < dimension_; ++column) {
                const float a = static_cast<float>((token + 1U) * (column + 3U));
                row[column] = std::sin(a * 0.173F) + 0.6F * std::cos(a * 0.071F + static_cast<float>(column));
            }
            normalize_in_place(row);
        }
    }

    [[nodiscard]] std::size_t token_count() const noexcept override { return token_count_; }
    [[nodiscard]] std::size_t dimension() const noexcept override { return dimension_; }
    void dequantize(TokenId id, std::span<float> output) const override {
        CHECK(id < token_count_);
        CHECK_EQ(output.size(), dimension_);
        const auto source = std::span<const float>(values_).subspan(static_cast<std::size_t>(id) * dimension_, dimension_);
        std::copy(source.begin(), source.end(), output.begin());
    }
    [[nodiscard]] float dot_row(TokenId id, std::span<const float> vector) const override {
        CHECK(id < token_count_);
        return dot(std::span<const float>(values_).subspan(static_cast<std::size_t>(id) * dimension_, dimension_), vector);
    }
    [[nodiscard]] float cosine_row(TokenId id, std::span<const float> vector) const override {
        CHECK(id < token_count_);
        return cosine(std::span<const float>(values_).subspan(static_cast<std::size_t>(id) * dimension_, dimension_), vector);
    }
    [[nodiscard]] std::vector<float> row(TokenId id) const {
        std::vector<float> result(dimension_);
        dequantize(id, result);
        return result;
    }
private:
    std::size_t token_count_;
    std::size_t dimension_;
    std::vector<float> values_;
};

RelationRecord make_relation(const DenseEmbeddingStore& embeddings,
                             RelationId id,
                             TokenId source,
                             TokenId destination,
                             MemoryLevel level,
                             std::uint8_t prototype = 0,
                             float confidence = 0.8F,
                             std::uint64_t support = 16U,
                             std::uint64_t seed = 1U) {
    RelationRecord record;
    record.id = id;
    record.source = source;
    record.destination = destination;
    record.prototype = prototype;
    record.level = level;
    record.lanes = LaneMask::from_level(level);
    record.support = support;
    record.confidence = confidence;
    record.version = 1U;
    record.created_at_ms = unix_millis();
    record.updated_at_ms = record.created_at_ms;
    record.expires_at_ms = level == MemoryLevel::M1 ? record.created_at_ms + 60000 : 0;
    record.escrow_state = level == MemoryLevel::M2 ? EscrowState::Promoted : EscrowState::Active;
    record.transform = MonomialOperator::seeded(embeddings.dimension(), hash_combine(seed, id));
    const auto source_vector = embeddings.row(source);
    const auto target_vector = embeddings.row(destination);
    for (int iteration = 0; iteration < 32; ++iteration) {
        record.transform.update_delta(source_vector, target_vector, 0.15F, 0.0F);
    }
    record.relation = RelationVector(16U);
    for (int iteration = 0; iteration < 8; ++iteration) {
        record.relation.update_observation(source_vector, target_vector,
                                           hash_combine(source, prototype), 1U,
                                           hash_combine(destination, prototype),
                                           destination == kEosToken, confidence, 0.25F);
    }
    return record;
}

float retrieval_priority_reference(const RelationRecord& relation) {
    const float support = std::log1p(static_cast<float>(relation.support));
    const float confidence = safe_logit(relation.confidence);
    const float penalty = relation.escrow_state == EscrowState::Rejected ||
                          relation.escrow_state == EscrowState::Expired ||
                          relation.escrow_state == EscrowState::Unreplayable ? -100.0F : 0.0F;
    return confidence + 0.15F * support + penalty;
}

class ReferenceRelationStore final : public IRelationStore {
public:
    explicit ReferenceRelationStore(std::vector<RelationRecord> records) {
        for (auto& record : records) {
            record.lanes = LaneMask::from_level(record.level);
            auto value = std::make_shared<const RelationRecord>(std::move(record));
            records_[value->id] = value;
        }
    }
    std::vector<std::shared_ptr<const RelationRecord>> outgoing(Lane lane, NodeId source, std::size_t limit) const override {
        std::vector<std::shared_ptr<const RelationRecord>> result;
        for (const auto& [_, relation] : records_) {
            if (relation->source != source || !relation->eligible(lane)) continue;
            if (relation->escrow_state == EscrowState::Rejected || relation->escrow_state == EscrowState::Expired ||
                relation->escrow_state == EscrowState::Unreplayable) continue;
            result.push_back(relation);
        }
        std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            const float left = retrieval_priority_reference(*lhs);
            const float right = retrieval_priority_reference(*rhs);
            if (left != right) return left > right;
            return lhs->id < rhs->id;
        });
        if (limit != 0U && result.size() > limit) result.resize(limit);
        return result;
    }
    std::shared_ptr<const RelationRecord> get(RelationId id) const override {
        const auto it = records_.find(id);
        return it == records_.end() ? std::shared_ptr<const RelationRecord>{} : it->second;
    }
    std::vector<std::shared_ptr<const RelationRecord>> between(NodeId source, NodeId destination) const override {
        std::vector<std::shared_ptr<const RelationRecord>> result;
        for (const auto& [_, relation] : records_) {
            if (relation->source == source && relation->destination == destination) result.push_back(relation);
        }
        std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) { return lhs->prototype < rhs->prototype; });
        return result;
    }
    GraphStats stats() const override {
        GraphStats result;
        result.relations_total = records_.size();
        std::unordered_set<NodeId> full_nodes;
        std::unordered_set<NodeId> clean_nodes;
        for (const auto& [_, relation] : records_) {
            ++result.full_index_entries;
            full_nodes.insert(relation->source);
            if (relation->level == MemoryLevel::M2) {
                ++result.relations_m2;
                ++result.clean_index_entries;
                clean_nodes.insert(relation->source);
            } else if (relation->level == MemoryLevel::M1) {
                ++result.relations_m1;
            }
        }
        result.nodes_with_full_edges = full_nodes.size();
        result.nodes_with_clean_edges = clean_nodes.size();
        return result;
    }
private:
    std::unordered_map<RelationId, std::shared_ptr<const RelationRecord>> records_;
};

void compare_lane_predictions(const LanePrediction& lhs, const LanePrediction& rhs, float tolerance = 1.0e-6F) {
    CHECK_EQ(lhs.lane, rhs.lane);
    CHECK_EQ(lhs.selected, rhs.selected);
    CHECK_EQ(lhs.candidates.size(), rhs.candidates.size());
    CHECK_EQ(lhs.rounds.size(), rhs.rounds.size());
    CHECK_EQ(lhs.metrics.candidate_retrievals, rhs.metrics.candidate_retrievals);
    CHECK_EQ(lhs.metrics.operator_evaluations, rhs.metrics.operator_evaluations);
    CHECK_EQ(lhs.metrics.gate_evaluations, rhs.metrics.gate_evaluations);
    CHECK_EQ(lhs.metrics.branches_created, rhs.metrics.branches_created);
    CHECK_EQ(lhs.metrics.branches_surviving, rhs.metrics.branches_surviving);
    CHECK_EQ(lhs.metrics.active_state_peak, rhs.metrics.active_state_peak);
    for (std::size_t i = 0; i < lhs.candidates.size(); ++i) {
        CHECK_EQ(lhs.candidates[i].token, rhs.candidates[i].token);
        CHECK_EQ(lhs.candidates[i].relation_id, rhs.candidates[i].relation_id);
        CHECK_NEAR(lhs.candidates[i].score, rhs.candidates[i].score, tolerance);
        for (std::size_t feature = 0; feature < kScoreFeatureCount; ++feature) {
            CHECK_NEAR(lhs.candidates[i].features[feature], rhs.candidates[i].features[feature], tolerance);
        }
    }
    for (std::size_t i = 0; i < lhs.rounds.size(); ++i) {
        CHECK_EQ(lhs.rounds[i].depth, rhs.rounds[i].depth);
        CHECK_EQ(lhs.rounds[i].fold_budget, rhs.rounds[i].fold_budget);
        CHECK_EQ(lhs.rounds[i].gate_decisions, rhs.rounds[i].gate_decisions);
        CHECK_EQ(lhs.rounds[i].parent_branch_ids, rhs.rounds[i].parent_branch_ids);
        CHECK_EQ(lhs.rounds[i].survivor_ids, rhs.rounds[i].survivor_ids);
        CHECK_EQ(lhs.rounds[i].relation_versions, rhs.rounds[i].relation_versions);
        CHECK_EQ(lhs.rounds[i].candidate_set_hash, rhs.rounds[i].candidate_set_hash);
        CHECK_NEAR(lhs.rounds[i].shadow_upper_bound, rhs.rounds[i].shadow_upper_bound, tolerance);
    }
}

EngineConfig small_engine_config() {
    EngineConfig config;
    config.top_k_full = 8U;
    config.top_k_clean = 8U;
    config.beam_full = 8U;
    config.beam_clean = 8U;
    config.max_rounds = 4U;
    config.max_ports_per_node = 8U;
    config.port_capacity = 8U;
    config.port_similarity_threshold = 0.50F;
    config.port_pressure_threshold = 0.0F;
    config.branch_energy_floor = 1.0e-8F;
    config.parallel_lanes = false;
    config.exact_pure_reuse = false;
    return config;
}

ReplayClosure make_valid_closure(RelationRecord root,
                                 ReplayRecorder& recorder,
                                 const Controller& controller,
                                 const RoleInducer& roles,
                                 std::uint64_t operation_id = 91U,
                                 std::uint64_t seed = 1234U) {
    ReplayStep step;
    step.operation_id = operation_id;
    step.controller_version = controller.snapshot().version;
    step.relation_versions.emplace_back(root.id, root.version);
    step.deterministic_seed = seed;
    step.depth = 0U;
    const ReplayId replay_id = recorder.record(step);

    ReplayClosure closure;
    closure.root_relation = root.id;
    closure.root_version = root.version;
    closure.operation_id = operation_id;
    closure.base_seed = seed;
    closure.replay_steps = {replay_id};
    closure.relation_versions = {{root.id, root.version}};
    auto payload = root.serialize();
    closure.relation_snapshots.push_back(RelationSnapshot{root.id, root.version, payload});
    closure.controller_version = controller.snapshot().version;
    closure.controller_snapshot = controller.serialize();
    closure.role_snapshot = roles.serialize();
    closure.deterministic_seeds = {seed};
    closure.snapshot_hashes = {hash_bytes(payload)};
    closure.binding_hashes = {hash_combine(root.source, root.destination)};
    CHECK(closure.complete());
    return closure;
}

EscrowObservation make_observation(const ReplayClosure& closure, TokenId source, TokenId target, std::uint64_t context_key) {
    EscrowObservation observation;
    observation.contextual_key = context_key;
    observation.observed_content = target;
    observation.context_tokens = {source};
    observation.bound_frame = {0.1F, 0.2F, 0.3F};
    observation.active_trace = closure.replay_steps;
    observation.replay_closure = closure;
    observation.source = "unit-test";
    observation.timestamp_ms = unix_millis();
    observation.support = 1.0F;
    observation.source_position = 0U;
    return observation;
}


void test_escrow_observation_source_position_roundtrip() {
    EscrowObservation observation;
    observation.contextual_key = 99U;
    observation.observed_content = 7U;
    observation.context_tokens = {1U, 2U, 3U, 4U};
    observation.bound_frame = {0.1F, -0.2F};
    observation.source = "roundtrip";
    observation.timestamp_ms = 12345;
    observation.source_position = 2U;
    const auto decoded = EscrowObservation::deserialize(observation.serialize());
    CHECK_EQ(decoded.contextual_key, observation.contextual_key);
    CHECK_EQ(decoded.context_tokens, observation.context_tokens);
    CHECK_EQ(decoded.source_position, 2U);
}

void test_prepare_failure_rolls_back_partial_artifacts() {
    TempDirectory temp;
    const auto corpus = temp.path() / "corpus.txt";
    write_text(corpus, "one two three.\n");
    const auto invalid_embeddings = temp.path() / "bad.f32";
    write_text(invalid_embeddings, "wrong-size");
    AppConfig config;
    config.model.embedding_dim = 16U;
    config.model.relation_dim = 8U;
    config.tokenizer.vocab_size = 320U;
    config.persistence.model_dir = temp.path() / "model";
    config.persistence.database = config.persistence.model_dir / "mrdl.db";
    config.persistence.tokenizer = config.persistence.model_dir / "tokenizer.mrdltok";
    config.persistence.embeddings = config.persistence.model_dir / "embeddings.mrdlemb";
    config.validate();
    CHECK_THROWS(ModelRuntime::prepare(config, corpus, EmbeddingInit::ExternalFloat32, invalid_embeddings));
    CHECK(!std::filesystem::exists(config.persistence.database));
    CHECK(!std::filesystem::exists(config.persistence.tokenizer));
    CHECK(!std::filesystem::exists(config.persistence.embeddings));
    CHECK(!std::filesystem::exists(config.persistence.model_dir / "config.effective.ini"));
}

void test_tokenizer_roundtrip_and_persistence() {
    TempDirectory temp;
    const auto corpus = temp.path() / "corpus.txt";
    write_text(corpus, "Hola mundo.\nEl banco está abierto.\nUTF-8: árbol, niño, 中文.\n");
    TokenizerConfig config;
    config.vocab_size = 320U;
    config.heavy_hitter_multiplier = 3U;
    config.lowercase = false;
    auto tokenizer = HybridTokenizer::build_from_corpus(corpus, config);
    CHECK(tokenizer.size() >= kFirstLearnedToken);
    const std::string sample = "Hola, árbol y 中文; bytes exactos.\n";
    const auto ids = tokenizer.encode(sample, true, true);
    CHECK_EQ(ids.front(), kBosToken);
    CHECK_EQ(ids.back(), kEosToken);
    CHECK_EQ(tokenizer.decode(ids), sample);
    const auto path = temp.path() / "tokenizer.mrdltok";
    tokenizer.save(path);
    const auto loaded = HybridTokenizer::load(path);
    CHECK_EQ(loaded.size(), tokenizer.size());
    CHECK_EQ(loaded.encode(sample, true, true), ids);
    CHECK_EQ(loaded.decode(ids), sample);
}

void test_frozen_embeddings_checksum_and_repeatability() {
    TempDirectory temp;
    const auto corpus = temp.path() / "corpus.txt";
    write_text(corpus, "uno dos tres\ndos tres cuatro\n");
    TokenizerConfig tokenizer_config;
    tokenizer_config.vocab_size = 300U;
    const auto tokenizer = HybridTokenizer::build_from_corpus(corpus, tokenizer_config);
    EmbeddingBuildOptions options;
    options.mode = EmbeddingInit::Random;
    options.dimension = 16U;
    options.seed = 77U;
    const auto first = temp.path() / "first.mrdlemb";
    const auto second = temp.path() / "second.mrdlemb";
    FrozenEmbeddingStore::build(first, tokenizer, corpus, options);
    FrozenEmbeddingStore::build(second, tokenizer, corpus, options);
    auto one = FrozenEmbeddingStore::load(first);
    auto two = FrozenEmbeddingStore::load(second);
    CHECK_EQ(one.token_count(), tokenizer.size());
    CHECK_EQ(one.dimension(), 16U);
    CHECK_EQ(one.content_hash(), two.content_hash());
    std::vector<float> row1(16U), row2(16U);
    one.dequantize(kByteTokenBase + static_cast<TokenId>('a'), row1);
    two.dequantize(kByteTokenBase + static_cast<TokenId>('a'), row2);
    CHECK_EQ(row1, row2);

    const auto corrupt = temp.path() / "corrupt.mrdlemb";
    std::filesystem::copy_file(first, corrupt);
    {
        std::fstream stream(corrupt, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekg(-1, std::ios::end);
        char byte = 0;
        stream.read(&byte, 1);
        byte ^= 0x5a;
        stream.seekp(-1, std::ios::end);
        stream.write(&byte, 1);
    }
    CHECK_THROWS(FrozenEmbeddingStore::load(corrupt));
}

void test_monomial_composition_and_serialization() {
    constexpr std::size_t dimension = 32U;
    const auto first = MonomialOperator::seeded(dimension, 11U);
    const auto second = MonomialOperator::seeded(dimension, 22U);
    const auto composed = MonomialOperator::compose(second, first);
    std::vector<float> input(dimension);
    for (std::size_t i = 0; i < dimension; ++i) input[i] = std::sin(static_cast<float>(i) * 0.3F);
    const auto nested = second.apply(first.apply(input));
    const auto direct = composed.apply(input);
    for (std::size_t i = 0; i < dimension; ++i) CHECK_NEAR(nested[i], direct[i], 1.0e-6F);
    const auto restored = MonomialOperator::deserialize(composed.serialize());
    CHECK_EQ(restored.full_hash(), composed.full_hash());
    CHECK_EQ(restored.apply(input), direct);
}

void test_relation_vector_and_record_roundtrip() {
    DenseEmbeddingStore embeddings;
    auto record = make_relation(embeddings, 7U, 10U, 20U, MemoryLevel::M1, 2U, 0.42F, 31U, 9U);
    record.derived = true;
    record.derived_from = {1U, 2U};
    const auto restored = RelationRecord::deserialize(record.serialize());
    CHECK_EQ(restored.id, record.id);
    CHECK_EQ(restored.source, record.source);
    CHECK_EQ(restored.destination, record.destination);
    CHECK_EQ(restored.prototype, record.prototype);
    CHECK_EQ(restored.level, record.level);
    CHECK_EQ(restored.support, record.support);
    CHECK_EQ(restored.transform.full_hash(), record.transform.full_hash());
    CHECK_EQ(restored.derived_from, record.derived_from);
    CHECK_EQ(restored.relation.dimension(), record.relation.dimension());
    for (std::size_t i = 0; i < record.relation.dimension(); ++i) {
        CHECK_NEAR(restored.relation.values()[i], record.relation.values()[i], 0.015F);
    }
    const std::array<float, 4> confidence{0.9F, 0.8F, 0.7F, 0.6F};
    const float q4 = path_confidence(confidence, 0.05F, 0.08F);
    const std::array<float, 2> prefix{0.9F, 0.8F};
    const float q2 = path_confidence(prefix, 0.05F, 0.08F);
    CHECK(q4 < q2);
    CHECK(q4 > 0.0F);
}

void test_graph_physical_lane_indexes_and_duplicate_guard() {
    DenseEmbeddingStore embeddings;
    GraphStore graph;
    graph.upsert(make_relation(embeddings, 1U, 10U, 20U, MemoryLevel::M2, 0U, 0.8F));
    graph.upsert(make_relation(embeddings, 2U, 10U, 21U, MemoryLevel::M1, 0U, 0.99F));
    const auto full = graph.outgoing(Lane::Full, 10U);
    const auto clean = graph.outgoing(Lane::Clean, 10U);
    CHECK_EQ(full.size(), 2U);
    CHECK_EQ(clean.size(), 1U);
    CHECK_EQ(clean.front()->id, 1U);
    const auto stats = graph.stats();
    CHECK_EQ(stats.full_index_entries, 2U);
    CHECK_EQ(stats.clean_index_entries, 1U);
    CHECK_THROWS(graph.upsert(make_relation(embeddings, 3U, 10U, 20U, MemoryLevel::M1, 0U, 0.7F)));
    CHECK(graph.promote(2U, 1U));
    CHECK_EQ(graph.outgoing(Lane::Clean, 10U).size(), 2U);
}

void test_A_equivalence_against_filtering_reference() {
    DenseEmbeddingStore embeddings;
    Controller controller;
    RoleInducer roles;
    auto config = small_engine_config();
    std::vector<RelationRecord> records;
    records.push_back(make_relation(embeddings, 1U, 10U, 20U, MemoryLevel::M2, 0U, 0.91F));
    records.push_back(make_relation(embeddings, 2U, 10U, 21U, MemoryLevel::M2, 0U, 0.83F));
    records.push_back(make_relation(embeddings, 3U, 10U, 30U, MemoryLevel::M1, 0U, 0.98F));
    records.push_back(make_relation(embeddings, 4U, 20U, 22U, MemoryLevel::M2, 0U, 0.88F));
    records.push_back(make_relation(embeddings, 5U, 21U, 23U, MemoryLevel::M2, 0U, 0.77F));
    records.push_back(make_relation(embeddings, 6U, 30U, 31U, MemoryLevel::M1, 0U, 0.97F));

    GraphStore indexed;
    for (const auto& record : records) indexed.load_relation(record);
    ReferenceRelationStore reference(records);
    const std::array<TokenId, 1> context{10U};

    LaneEngine indexed_full(Lane::Full, indexed, embeddings, controller, roles, config);
    LaneEngine reference_full(Lane::Full, reference, embeddings, controller, roles, config);
    compare_lane_predictions(indexed_full.predict(context, nullptr, nullptr, 55U, 777U),
                             reference_full.predict(context, nullptr, nullptr, 55U, 777U));

    LaneEngine indexed_clean(Lane::Clean, indexed, embeddings, controller, roles, config);
    LaneEngine reference_clean(Lane::Clean, reference, embeddings, controller, roles, config);
    compare_lane_predictions(indexed_clean.predict(context, nullptr, nullptr, 56U, 778U),
                             reference_clean.predict(context, nullptr, nullptr, 56U, 778U));
}

void test_B_C_clean_non_interference_and_control_isolation() {
    static_assert(!std::is_constructible_v<PromotionPermit, RelationId, std::uint64_t>);
    DenseEmbeddingStore embeddings;
    Controller controller;
    RoleInducer roles;
    auto config = small_engine_config();
    GraphStore graph;
    graph.upsert(make_relation(embeddings, 1U, 40U, 41U, MemoryLevel::M2, 0U, 0.82F));
    graph.upsert(make_relation(embeddings, 2U, 40U, 42U, MemoryLevel::M2, 0U, 0.78F));
    graph.upsert(make_relation(embeddings, 3U, 41U, 43U, MemoryLevel::M2, 0U, 0.81F));
    const auto controller_before = controller.serialize();
    LaneEngine clean(Lane::Clean, graph, embeddings, controller, roles, config);
    const std::array<TokenId, 1> context{40U};
    const auto before = clean.predict(context, nullptr, nullptr, 100U, 200U);

    for (std::uint8_t index = 0; index < 4U; ++index) {
        auto hostile = make_relation(embeddings, 100U + index, 40U, 100U + index,
                                     MemoryLevel::M1, index, 0.999999F,
                                     1000000000ULL, 0xdeadbeefULL + index);
        for (float& value : hostile.relation.values()) value = 1000000.0F;
        graph.upsert(std::move(hostile));
    }
    const auto after = clean.predict(context, nullptr, nullptr, 100U, 200U);
    compare_lane_predictions(before, after, 0.0F);
    CHECK_EQ(controller.serialize(), controller_before);
    CHECK_EQ(graph.outgoing(Lane::Clean, 40U).size(), 2U);
    CHECK_EQ(graph.outgoing(Lane::Full, 40U).size(), 6U);

    LaneEngine full(Lane::Full, graph, embeddings, controller, roles, config);
    const auto full_prediction = full.predict(context, nullptr, nullptr, 100U, 200U);
    CHECK(std::any_of(full_prediction.candidates.begin(), full_prediction.candidates.end(),
                      [](const CandidateScore& candidate) { return candidate.token >= 100U && candidate.token <= 103U; }));
}

void test_ports_are_one_pass_bounded_and_energy_conserving() {
    auto config = small_engine_config();
    config.max_ports_per_node = 2U;
    config.port_capacity = 4U;
    config.port_similarity_threshold = -1.0F;
    OnePassPortRouter router(config);
    router.begin_round();
    std::vector<float> key1(16U, 0.0F), key2(16U, 0.0F);
    key1[0] = 1.0F;
    key2[1] = 1.0F;
    const auto first = router.route(9U, key1, 0.75F);
    const auto second = router.route(9U, key2, 0.5F);
    CHECK(first.count >= 1U && first.count <= 2U);
    CHECK(second.count >= 1U && second.count <= 2U);
    const float first_energy = first.energies[0] + first.energies[1];
    const float second_energy = second.energies[0] + second.energies[1];
    CHECK_NEAR(first_energy, 0.75F, 1.0e-6F);
    CHECK_NEAR(second_energy, 0.5F, 1.0e-6F);
    router.end_round();
    CHECK(router.port_count(9U) <= 2U);
}

void test_D_beam_and_replay_storage_are_bounded() {
    DenseEmbeddingStore embeddings(2048U, 16U);
    GraphStore graph;
    Controller controller;
    RoleInducer roles;
    auto config = small_engine_config();
    config.top_k_full = 4U;
    config.top_k_clean = 4U;
    config.beam_full = 4U;
    config.beam_clean = 4U;
    config.max_rounds = 12U;
    config.parallel_lanes = false;

    RelationId id = 1U;
    constexpr std::uint32_t width = 8U;
    constexpr std::uint32_t branching = 4U;
    constexpr std::uint32_t depth = 12U;
    const TokenId root = 500U;
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        const std::uint32_t source_count = layer == 0U ? 1U : width;
        for (std::uint32_t source_index = 0; source_index < source_count; ++source_index) {
            const TokenId source = layer == 0U ? root : 500U + layer * width + source_index;
            for (std::uint32_t edge = 0; edge < branching; ++edge) {
                const TokenId destination = 500U + (layer + 1U) * width + ((source_index + edge) % width);
                graph.upsert(make_relation(embeddings, id++, source, destination,
                                           MemoryLevel::M2, static_cast<std::uint8_t>(edge),
                                           0.75F + 0.01F * static_cast<float>(edge), 20U + edge, layer));
            }
        }
    }
    auto recorder = std::make_shared<ReplayRecorder>();
    DualLaneEngine engine(graph, embeddings, controller, roles, config, recorder);
    const std::array<TokenId, 1> context{root};
    const auto prediction = engine.predict(context, true, 9988U);
    CHECK(prediction.full.metrics.active_state_peak <= config.beam_full);
    CHECK(prediction.clean.metrics.active_state_peak <= config.beam_clean);
    CHECK(prediction.full.rounds.size() <= config.max_rounds);
    CHECK(prediction.clean.rounds.size() <= config.max_rounds);
    CHECK(prediction.replay_ids.size() <= config.max_rounds);
    CHECK(prediction.full.metrics.branches_surviving <=
          static_cast<std::uint64_t>(config.max_rounds) * config.beam_full);
    CHECK(prediction.clean.metrics.branches_surviving <=
          static_cast<std::uint64_t>(config.max_rounds) * config.beam_clean);
    for (const ReplayId replay_id : prediction.replay_ids) {
        const auto step = recorder->get(replay_id);
        CHECK(step.has_value());
        CHECK(step->relation_versions.size() <=
              static_cast<std::size_t>(2U * config.beam_full * config.top_k_full));
    }
}

void test_replay_serialization_and_persistent_high_watermark() {
    TempDirectory temp;
    auto store = std::make_shared<SqliteModelStore>(temp.path() / "model.db", 1000U, false);
    ReplayStep step;
    step.operation_id = 44U;
    step.controller_version = 7U;
    step.relation_versions = {{1U, 2U}, {3U, 4U}};
    step.parent_branch_ids = {9U, 10U};
    step.lanes[0].gate_decisions = {GateDecision::Compose, GateDecision::Defer};
    step.lanes[0].fold_budget = 8U;
    step.lanes[0].survivor_ids = {11U};
    step.lanes[0].candidate_set_hash = 123U;
    step.deterministic_seed = 99U;
    step.depth = 3U;
    const auto decoded = ReplayStep::deserialize(step.serialize());
    CHECK_EQ(decoded.operation_id, step.operation_id);
    CHECK_EQ(decoded.relation_versions, step.relation_versions);
    CHECK_EQ(decoded.lanes[0].gate_decisions, step.lanes[0].gate_decisions);

    ReplayId first_id = 0U;
    {
        ReplayRecorder first(store);
        first_id = first.record(step);
        CHECK_EQ(first_id, 1U);
    }
    ReplayRecorder second(store);
    ReplayStep later = step;
    later.id = 0U;
    later.operation_id = 45U;
    const ReplayId second_id = second.record(later);
    CHECK(second_id > first_id);
    CHECK_EQ(store->max_step_id(), second_id);
    CHECK(second.get(first_id).has_value());
    std::string diagnostic;
    CHECK(store->integrity_check(&diagnostic));
    CHECK_EQ(diagnostic, "ok");
}

void test_E_promotion_TTL_replay_and_automatic_clean_integration() {
    TempDirectory temp;
    DenseEmbeddingStore embeddings;
    auto persistence = std::make_shared<SqliteModelStore>(temp.path() / "model.db", 1000U, true);
    GraphStore graph(persistence);
    Controller controller;
    RoleInducer::Config role_config;
    role_config.min_support = 1U;
    role_config.variable_score_threshold = 0.0F;
    RoleInducer roles(role_config);
    ReplayRecorder recorder(persistence);
    PromotionManager promotion(graph, recorder, controller, roles, persistence);

    auto relation = make_relation(embeddings, 1U, 60U, 61U, MemoryLevel::M1, 0U, 0.40F, 4U);
    graph.upsert(relation);
    const auto closure = make_valid_closure(relation, recorder, controller, roles, 500U, 600U);
    promotion.remember(relation.id, make_observation(closure, relation.source, relation.destination, 1U),
                       closure, 0.45F, 1);
    CHECK_EQ(graph.outgoing(Lane::Clean, relation.source).size(), 0U);
    CHECK(promotion.reserve(relation.id));
    CHECK(promotion.begin_audit(relation.id));
    CHECK_EQ(promotion.expire_due(unix_millis() + 5000), 0U);
    const auto pinned = promotion.get(relation.id);
    CHECK(pinned.has_value());
    CHECK(pinned->expiry_pending);
    CHECK_EQ(pinned->state, EscrowState::Auditing);
    CHECK(graph.get(relation.id) != nullptr);

    AuditOutcome outcome;
    outcome.accepted = true;
    outcome.stable = true;
    outcome.replay_exact = true;
    outcome.causal_influence = 0.9F;
    outcome.stability_ratio = 1.0F;
    outcome.reason = "test promotion";
    outcome.positive_features.fill(0.5F);
    ScoreFeatures negative{};
    negative.fill(-0.25F);
    outcome.negative_features.push_back(negative);
    const auto slot = structural_slot_key(std::array<TokenId, 1>{relation.source}, 0U, 1U);
    outcome.role_observations.push_back(RoleObservation{slot, relation.destination, 123U});
    const auto controller_version = controller.snapshot().version;
    const auto permit = promotion.complete(relation.id, outcome, 0.01F);
    CHECK(permit.has_value());
    CHECK_EQ(permit->relation_id(), relation.id);
    const auto promoted = graph.get(relation.id);
    CHECK(promoted != nullptr);
    CHECK_EQ(promoted->level, MemoryLevel::M2);
    CHECK_EQ(promoted->escrow_state, EscrowState::Promoted);
    CHECK_EQ(graph.outgoing(Lane::Clean, relation.source).size(), 1U);
    CHECK(controller.snapshot().version > controller_version);
    CHECK(roles.role_for(slot).has_value());
    const auto escrow = promotion.get(relation.id);
    CHECK(escrow.has_value());
    CHECK_EQ(escrow->state, EscrowState::Promoted);
    CHECK(!escrow->expiry_pending);
    CHECK_EQ(promotion.expire_due(unix_millis() + 100000), 0U);

    PromotionManager reloaded(graph, recorder, controller, roles, persistence);
    reloaded.load();
    const auto after_restart = reloaded.get(relation.id);
    CHECK(after_restart.has_value());
    CHECK_EQ(after_restart->state, EscrowState::Promoted);
}

void test_E_missing_replay_marks_unreplayable_without_promotion() {
    TempDirectory temp;
    DenseEmbeddingStore embeddings;
    auto persistence = std::make_shared<SqliteModelStore>(temp.path() / "model.db", 1000U, false);
    GraphStore graph(persistence);
    Controller controller;
    RoleInducer roles;
    ReplayRecorder recorder(persistence);
    PromotionManager promotion(graph, recorder, controller, roles, persistence);
    auto relation = make_relation(embeddings, 10U, 70U, 71U, MemoryLevel::M1, 0U, 0.4F, 4U);
    graph.upsert(relation);
    const auto closure = make_valid_closure(relation, recorder, controller, roles, 700U, 800U);
    promotion.remember(relation.id, make_observation(closure, relation.source, relation.destination, 2U),
                       closure, 0.45F, 1000);
    CHECK(recorder.erase(closure.replay_steps.front()));
    CHECK(!promotion.reserve(relation.id));
    const auto record = promotion.get(relation.id);
    CHECK(record.has_value());
    CHECK_EQ(record->state, EscrowState::Unreplayable);
    const auto graph_record = graph.get(relation.id);
    CHECK(graph_record != nullptr);
    CHECK_EQ(graph_record->level, MemoryLevel::M1);
    CHECK_EQ(graph_record->escrow_state, EscrowState::Unreplayable);
    CHECK_EQ(graph.outgoing(Lane::Clean, relation.source).size(), 0U);
}

void test_E_atomic_reservation_race_has_one_winner() {
    TempDirectory temp;
    DenseEmbeddingStore embeddings;
    auto persistence = std::make_shared<SqliteModelStore>(temp.path() / "model.db", 1000U, false);
    GraphStore graph(persistence);
    Controller controller;
    RoleInducer roles;
    ReplayRecorder recorder(persistence);
    PromotionManager promotion(graph, recorder, controller, roles, persistence);
    auto relation = make_relation(embeddings, 20U, 80U, 81U, MemoryLevel::M1, 0U, 0.4F, 4U);
    graph.upsert(relation);
    const auto closure = make_valid_closure(relation, recorder, controller, roles, 900U, 901U);
    promotion.remember(relation.id, make_observation(closure, relation.source, relation.destination, 3U),
                       closure, 0.45F, 1000);

    std::atomic<int> winners{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 64; ++i) {
        threads.emplace_back([&] { if (promotion.reserve(relation.id)) ++winners; });
    }
    for (auto& thread : threads) thread.join();
    CHECK_EQ(winners.load(), 1);
    const auto record = promotion.get(relation.id);
    CHECK(record.has_value());
    CHECK_EQ(record->state, EscrowState::AuditReserved);
    CHECK_EQ(record->pin_count, 1);
    CHECK(promotion.release_to_active(relation.id, "race complete"));
}

void test_ngram_baseline_roundtrip() {
    NGramBaseline baseline(3U, 0.1);
    const std::array<TokenId, 3> context{1U, 2U, 3U};
    for (int i = 0; i < 10; ++i) baseline.observe(context, 4U);
    for (int i = 0; i < 2; ++i) baseline.observe(context, 5U);
    CHECK_EQ(baseline.predict(context), 4U);
    CHECK(baseline.probability(context, 4U) > baseline.probability(context, 5U));
    TempDirectory temp;
    const auto path = temp.path() / "baseline.ngram";
    baseline.save(path);
    const auto loaded = NGramBaseline::load(path);
    CHECK_EQ(loaded.order(), 3U);
    CHECK_EQ(loaded.observations(), baseline.observations());
    CHECK_EQ(loaded.predict(context), 4U);
    CHECK_NEAR(loaded.probability(context, 4U), baseline.probability(context, 4U), 1.0e-12);
}

void test_sqlite_write_transaction_commit_and_rollback() {
    TempDirectory temp;
    DenseEmbeddingStore embeddings(32U, 8U);
    auto store = std::make_shared<SqliteModelStore>(temp.path() / "mrdl.db", 1000U, false);
    const auto relation = make_relation(embeddings, 1U, 4U, 5U, MemoryLevel::M1);

    {
        auto transaction = store->begin_write_transaction();
        CHECK(transaction.active());
        store->persist_relation(relation);
        transaction.rollback();
        CHECK(!transaction.active());
    }
    CHECK(store->load_relations().empty());

    {
        auto transaction = store->begin_write_transaction();
        store->persist_relation(relation);
        // Destructor must roll back an uncommitted durability unit.
    }
    CHECK(store->load_relations().empty());

    {
        auto transaction = store->begin_write_transaction();
        store->persist_relation(relation);
        transaction.commit();
        CHECK(!transaction.active());
    }
    const auto loaded = store->load_relations();
    CHECK_EQ(loaded.size(), 1U);
    CHECK_EQ(loaded.front().id, relation.id);
}

void test_process_lock_excludes_second_writer() {
    TempDirectory temp;
    ProcessLock first(temp.path(), LockMode::Exclusive, false);
    CHECK(first.owns_lock());
    CHECK_THROWS(ProcessLock(temp.path(), LockMode::Exclusive, false));
}

void test_config_save_load_and_validation() {
    TempDirectory temp;
    AppConfig config;
    config.model.embedding_dim = 32U;
    config.model.relation_dim = 16U;
    config.persistence.model_dir = temp.path() / "model";
    config.persistence.database = config.persistence.model_dir / "mrdl.db";
    config.persistence.tokenizer = config.persistence.model_dir / "tokenizer.mrdltok";
    config.persistence.embeddings = config.persistence.model_dir / "embeddings.mrdlemb";
    config.runtime.threads = 4U;
    config.validate();
    const auto path = temp.path() / "config.ini";
    config.save(path);
    const auto loaded = AppConfig::load(path);
    CHECK_EQ(loaded.model.embedding_dim, config.model.embedding_dim);
    CHECK_EQ(loaded.model.relation_dim, config.model.relation_dim);
    CHECK_EQ(loaded.persistence.database, config.persistence.database);
    loaded.validate();
    auto invalid = loaded;
    invalid.engine.beam_clean = 0U;
    CHECK_THROWS(invalid.validate());
    invalid = loaded;
    invalid.training.mode = "A";
    CHECK_THROWS(invalid.validate());

    const auto malformed_integer = temp.path() / "bad-int.ini";
    write_text(malformed_integer, "[model]\nembedding_dim = not-a-number\n");
    CHECK_THROWS(AppConfig::load(malformed_integer));
    const auto malformed_boolean = temp.path() / "bad-bool.ini";
    write_text(malformed_boolean, "[tokenizer]\nlowercase = perhaps\n");
    CHECK_THROWS(AppConfig::load(malformed_boolean));
    const auto unknown_key = temp.path() / "unknown.ini";
    write_text(unknown_key, "[engine]\nbeam_celan = 12\n");
    CHECK_THROWS(AppConfig::load(unknown_key));
    const auto duplicate_key = temp.path() / "duplicate.ini";
    write_text(duplicate_key, "[engine]\nbeam_clean = 12\nbeam_clean = 16\n");
    CHECK_THROWS(AppConfig::load(duplicate_key));
}

void test_cold_start_audit_promotes_from_vacuous_clean_control() {
    TempDirectory temp;
    const auto corpus = temp.path() / "bootstrap.txt";
    write_text(corpus,
               "alpha beta.\n"
               "alpha beta.\n"
               "alpha beta.\n"
               "alpha beta.\n"
               "alpha beta.\n"
               "alpha beta.\n");

    AppConfig config;
    config.model.embedding_dim = 16U;
    config.model.relation_dim = 8U;
    config.model.max_relation_prototypes = 2U;
    config.tokenizer.vocab_size = 320U;
    config.engine.top_k_full = 4U;
    config.engine.top_k_clean = 4U;
    config.engine.beam_full = 8U;
    config.engine.beam_clean = 8U;
    config.engine.max_rounds = 2U;
    config.engine.max_ports_per_node = 4U;
    config.engine.port_capacity = 4U;
    config.engine.port_pressure_threshold = 0.0F;
    config.engine.branch_energy_floor = 1.0e-8F;
    config.engine.parallel_lanes = false;
    config.training.context_tokens = 4U;
    config.training.max_source_capsules = 1U;
    config.training.epochs = 1U;
    config.training.batch_tokens = 4U;
    config.training.auto_audit = true;
    config.training.checkpoint_every_tokens = 100000U;
    config.memory.promotion_min_support = 2U;
    config.memory.promotion_min_contexts = 1U;
    config.memory.promotion_min_influence = 0.0F;
    config.memory.promotion_stability_ratio = 0.0F;
    config.memory.audit_top_m = 4U;
    config.persistence.model_dir = temp.path() / "model";
    config.persistence.database = config.persistence.model_dir / "mrdl.db";
    config.persistence.tokenizer = config.persistence.model_dir / "tokenizer.mrdltok";
    config.persistence.embeddings = config.persistence.model_dir / "embeddings.mrdlemb";
    config.persistence.synchronous_full = false;
    config.runtime.threads = 1U;
    config.validate();

    ModelRuntime::prepare(config, corpus, EmbeddingInit::RandomIndexing);
    auto runtime = ModelRuntime::open(config);
    const auto stats = runtime->train(corpus);
    const auto graph = runtime->graph().stats();
    CHECK(stats.promotions > 0U);
    CHECK(graph.relations_m2 > 0U);
    CHECK(graph.clean_index_entries > 0U);
}

void test_end_to_end_prepare_train_open_eval_backup() {
    TempDirectory temp;
    const auto corpus = temp.path() / "tiny.txt";
    write_text(corpus,
               "el gato corre.\n"
               "el perro corre.\n"
               "el gato duerme.\n"
               "el perro duerme.\n");
    AppConfig config;
    config.model.embedding_dim = 16U;
    config.model.relation_dim = 8U;
    config.model.max_relation_prototypes = 2U;
    config.tokenizer.vocab_size = 320U;
    config.engine.top_k_full = 4U;
    config.engine.top_k_clean = 4U;
    config.engine.beam_full = 4U;
    config.engine.beam_clean = 4U;
    config.engine.max_rounds = 2U;
    config.engine.max_ports_per_node = 4U;
    config.engine.port_capacity = 4U;
    config.engine.port_pressure_threshold = 0.0F;
    config.engine.branch_energy_floor = 1.0e-8F;
    config.engine.parallel_lanes = false;
    config.training.context_tokens = 4U;
    config.training.max_source_capsules = 2U;
    config.training.epochs = 1U;
    config.training.batch_tokens = 64U;
    config.training.auto_audit = false;
    config.training.checkpoint_every_tokens = 100000U;
    config.memory.promotion_min_support = 100U;
    config.memory.promotion_min_contexts = 100U;
    config.persistence.model_dir = temp.path() / "model";
    config.persistence.database = config.persistence.model_dir / "mrdl.db";
    config.persistence.tokenizer = config.persistence.model_dir / "tokenizer.mrdltok";
    config.persistence.embeddings = config.persistence.model_dir / "embeddings.mrdlemb";
    config.persistence.synchronous_full = false;
    config.runtime.threads = 1U;
    config.runtime.max_generation_tokens = 4U;
    config.validate();

    ModelRuntime::prepare(config, corpus, EmbeddingInit::Random);
    CHECK(std::filesystem::exists(config.persistence.tokenizer));
    CHECK(std::filesystem::exists(config.persistence.embeddings));
    CHECK(std::filesystem::exists(config.persistence.database));
    auto runtime = ModelRuntime::open(config);
    const auto train = runtime->train(corpus);
    CHECK(train.tokens > 0U);
    CHECK(train.m1_writes > 0U);
    CHECK(runtime->graph().stats().relations_m1 > 0U);
    const auto eval = runtime->evaluate(corpus, 32U);
    CHECK(eval.tokens > 0U);
    const auto generated = runtime->generate("el gato", 2U, 0.0F, 5U);
    CHECK(generated.certifications.size() <= 2U);
    std::string integrity;
    CHECK(runtime->integrity_check(&integrity));
    runtime->checkpoint();
    const auto backup = temp.path() / "backup";
    runtime->backup(backup);
    CHECK(std::filesystem::exists(backup / "mrdl.db"));
    CHECK(std::filesystem::exists(backup / config.persistence.tokenizer.filename()));
    CHECK(std::filesystem::exists(backup / config.persistence.embeddings.filename()));
    runtime.reset();
    auto reopened = ModelRuntime::open(config);
    CHECK(reopened->graph().stats().relations_total > 0U);
    CHECK(reopened->integrity_check());
}

using TestFunction = void (*)();
struct TestCase { const char* name; TestFunction function; };

const std::vector<TestCase> tests{
    {"escrow_observation_source_position_roundtrip", test_escrow_observation_source_position_roundtrip},
    {"prepare_failure_rolls_back_partial_artifacts", test_prepare_failure_rolls_back_partial_artifacts},
    {"tokenizer_roundtrip_and_persistence", test_tokenizer_roundtrip_and_persistence},
    {"frozen_embeddings_checksum_and_repeatability", test_frozen_embeddings_checksum_and_repeatability},
    {"monomial_composition_and_serialization", test_monomial_composition_and_serialization},
    {"relation_vector_and_record_roundtrip", test_relation_vector_and_record_roundtrip},
    {"graph_physical_lane_indexes_and_duplicate_guard", test_graph_physical_lane_indexes_and_duplicate_guard},
    {"A_equivalence_against_filtering_reference", test_A_equivalence_against_filtering_reference},
    {"B_C_clean_non_interference_and_control_isolation", test_B_C_clean_non_interference_and_control_isolation},
    {"ports_are_one_pass_bounded_and_energy_conserving", test_ports_are_one_pass_bounded_and_energy_conserving},
    {"D_beam_and_replay_storage_are_bounded", test_D_beam_and_replay_storage_are_bounded},
    {"replay_serialization_and_persistent_high_watermark", test_replay_serialization_and_persistent_high_watermark},
    {"E_promotion_TTL_replay_and_automatic_clean_integration", test_E_promotion_TTL_replay_and_automatic_clean_integration},
    {"E_missing_replay_marks_unreplayable_without_promotion", test_E_missing_replay_marks_unreplayable_without_promotion},
    {"E_atomic_reservation_race_has_one_winner", test_E_atomic_reservation_race_has_one_winner},
    {"ngram_baseline_roundtrip", test_ngram_baseline_roundtrip},
    {"sqlite_write_transaction_commit_and_rollback", test_sqlite_write_transaction_commit_and_rollback},
    {"process_lock_excludes_second_writer", test_process_lock_excludes_second_writer},
    {"config_save_load_and_validation", test_config_save_load_and_validation},
    {"cold_start_audit_promotes_from_vacuous_clean_control", test_cold_start_audit_promotes_from_vacuous_clean_control},
    {"end_to_end_prepare_train_open_eval_backup", test_end_to_end_prepare_train_open_eval_backup},
};

}  // namespace

int main(int argc, char** argv) {
    std::optional<std::string> filter;
    bool list_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--list") list_only = true;
        else if (argument == "--filter" && i + 1 < argc) filter = argv[++i];
        else if (argument.starts_with("--filter=")) filter = std::string(argument.substr(9));
        else {
            std::cerr << "unknown test option: " << argument << '\n';
            return 2;
        }
    }
    if (list_only) {
        for (const auto& test : tests) std::cout << test.name << '\n';
        return 0;
    }

    std::size_t passed = 0U;
    std::size_t failed = 0U;
    const auto started = std::chrono::steady_clock::now();
    for (const auto& test : tests) {
        if (filter && std::string_view(test.name).find(*filter) == std::string_view::npos) continue;
        try {
            test.function();
            ++passed;
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "tests_passed=" << passed << " tests_failed=" << failed
              << " elapsed_seconds=" << std::fixed << std::setprecision(3) << seconds << '\n';
    return failed == 0U ? 0 : 1;
}
