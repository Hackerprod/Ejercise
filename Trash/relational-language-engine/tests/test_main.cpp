#include "rlm/audit.hpp"
#include "rlm/binary.hpp"
#include "rlm/config.hpp"
#include "rlm/embedding_store.hpp"
#include "rlm/engine.hpp"
#include "rlm/promotion.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/replay.hpp"
#include "rlm/search.hpp"
#include "rlm/wal.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace rlm;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(std::string("CHECK failed: ") + #condition + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (false)
#define CHECK_OK(expression) do { const Status _status = (expression); if (!_status.ok()) throw std::runtime_error(_status.to_string()); } while (false)

template <typename T>
T require(Result<T> result) {
  if (!result) throw std::runtime_error(result.status().to_string());
  return std::move(result).value();
}

class TempDirectory final {
 public:
  explicit TempDirectory(std::string_view name) {
    static std::atomic<std::uint64_t> counter{1};
    path_ = std::filesystem::temp_directory_path() /
            ("rlm-" + std::string(name) + "-" + std::to_string(unix_time_ms()) + "-" +
             std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }
  ~TempDirectory() { std::error_code ec; std::filesystem::remove_all(path_, ec); }
  const std::filesystem::path& path() const noexcept { return path_; }
 private:
  std::filesystem::path path_;
};

std::vector<std::pair<std::string, std::vector<float>>> rows() {
  return {
      {"a", {1.0F, 0.0F, 0.0F, 0.0F}},
      {"b", {0.0F, 1.0F, 0.0F, 0.0F}},
      {"c", {0.0F, 0.0F, 1.0F, 0.0F}},
      {"d", {0.0F, 0.0F, 0.0F, 1.0F}},
      {"e", {0.5F, 0.5F, 0.0F, 0.0F}},
  };
}

StorageConfig storage(const std::filesystem::path& root, const std::filesystem::path& embeddings) {
  StorageConfig config;
  config.state_dir = root / "state";
  config.embedding_file = embeddings;
  config.shards = 4;
  config.durability = Durability::data;
  return config;
}

EngineConfig engine_config(const std::filesystem::path& root, const std::filesystem::path& embeddings) {
  EngineConfig config;
  config.storage = storage(root, embeddings);
  config.search.beam_width = 4;
  config.search.candidate_k = 8;
  config.search.max_depth = 2;
  config.search.replay_reopen_limit = 128;
  config.runtime.parallel_lanes = true;
  config.runtime.worker_threads = 4;
  config.runtime.queue_capacity = 16;
  config.training.batch_tokens = 4;
  config.training.context_radius = 2;
  config.training.evidence_cases_per_edge = 2;
  config.training.auto_promote_per_batch = 4;
  config.training.min_support_for_promotion = 1000000;
  config.training.max_m1_edges = 10000;
  config.training.checkpoint_every_batches = 1;
  config.audit.min_exact_cases = 1;
  config.audit.max_cases = 2;
  config.audit.max_unknown_cases = 1;
  config.replay.max_records = 1000;
  CHECK_OK(config.validate());
  return config;
}

RelationObservation observation(const IEmbeddingStore& embeddings, TokenId source, TokenId target,
                                float confidence, std::uint64_t count = 1) {
  RelationObservation result;
  result.source = source;
  result.target = target;
  result.relation = require(embeddings.relation_vector(source, target));
  result.confidence = confidence;
  result.count = count;
  result.observed_at_ms = unix_time_ms();
  return result;
}

void test_embedding_roundtrip() {
  TempDirectory temp("embeddings");
  const auto file = temp.path() / "embeddings.rle";
  CHECK_OK(EmbeddingFileBuilder::from_rows(rows(), file));
  FrozenEmbeddingStore store;
  CHECK_OK(store.open(file));
  CHECK(store.dimension() == 4);
  CHECK(store.token_count() == 5);
  CHECK(store.checksum() != 0);
  CHECK(require(store.token_id("c")) == 2);
  std::vector<float> vector(4);
  CHECK_OK(store.copy_embedding(0, vector));
  CHECK(vector[0] > 0.99F);
  const auto relation = require(store.relation_vector(0, 1)).dequantize();
  CHECK(relation.size() == 4);
  CHECK(relation[0] < 0.0F && relation[1] > 0.0F);
  // The public API has no mutation path; reopening produces the identical frozen checksum.
  FrozenEmbeddingStore second;
  CHECK_OK(second.open(file));
  CHECK(second.checksum() == store.checksum());
}

void test_clean_physical_isolation_and_no_ghost_branch() {
  TempDirectory temp("clean");
  const auto embedding_file = temp.path() / "embeddings.rle";
  CHECK_OK(EmbeddingFileBuilder::from_rows(rows(), embedding_file));
  FrozenEmbeddingStore embeddings;
  CHECK_OK(embeddings.open(embedding_file));
  RelationRepository relations;
  CHECK_OK(relations.open(storage(temp.path(), embedding_file), embeddings.dimension()));

  RelationEdge m2;
  m2.source = 0; m2.target = 1; m2.id = deterministic_edge_id(0, 1); m2.tier = EvidenceTier::m2;
  m2.relation = require(embeddings.relation_vector(0, 1)); m2.confidence = 0.20F; m2.support = 2;
  m2.created_at_ms = m2.last_seen_ms = unix_time_ms();
  CHECK_OK(relations.upsert_m2(m2));
  const RelationObservation high_m1 = observation(embeddings, 0, 2, 0.95F, 10);
  CHECK(require(relations.apply_observation_batch(101, std::span<const RelationObservation>(&high_m1, 1))));
  const RelationObservation only_m1 = observation(embeddings, 3, 2, 0.99F, 10);
  CHECK(require(relations.apply_observation_batch(102, std::span<const RelationObservation>(&only_m1, 1))));

  ScoringConfig scoring;
  scoring.confidence_weight = 10.0F; scoring.support_weight = 0.0F; scoring.relation_weight = 0.0F;
  scoring.target_weight = 0.0F; scoring.context_weight = 0.0F; scoring.repetition_penalty = 0.0F;
  SearchConfig search_config;
  search_config.beam_width = 2; search_config.candidate_k = 4; search_config.max_depth = 1;
  FrozenLinearController controller(embeddings, scoring);
  VectorRelationComposer composer(embeddings, search_config);
  BeamSearch search(embeddings, relations, controller, composer, search_config);

  SearchRequest clean; clean.context = {0}; clean.lane = Lane::clean; clean.capture_trace = true;
  SearchRequest full = clean; full.lane = Lane::full;
  const SearchResult clean_result = require(search.run(clean));
  const SearchResult full_result = require(search.run(full));
  CHECK(clean_result.has_prediction && clean_result.best_token == 1);
  CHECK(full_result.has_prediction && full_result.best_token == 2);
  CHECK(clean_result.trace.has_value());
  for (const auto& step : clean_result.trace->steps)
    for (const auto& parent : step.parents)
      for (const auto& candidate : parent.candidates) CHECK(candidate.tier == EvidenceTier::m2);

  SearchRequest absent; absent.context = {3}; absent.lane = Lane::clean; absent.capture_trace = true;
  const SearchResult absent_result = require(search.run(absent));
  CHECK(!absent_result.has_prediction);
  CHECK(absent_result.path_edges.empty());
  // CLEAN absence is an absent branch, never a fabricated zero monomial/operator.
  CHECK(absent_result.trace && absent_result.trace->winning_edges.empty());
}

void test_wal_recovery_batch_idempotence_and_tail_repair() {
  TempDirectory temp("wal");
  const auto embedding_file = temp.path() / "embeddings.rle";
  CHECK_OK(EmbeddingFileBuilder::from_rows(rows(), embedding_file));
  FrozenEmbeddingStore embeddings; CHECK_OK(embeddings.open(embedding_file));
  const StorageConfig config = storage(temp.path(), embedding_file);
  const EdgeId id = deterministic_edge_id(0, 1);
  {
    RelationRepository relations; CHECK_OK(relations.open(config, embeddings.dimension()));
    const RelationObservation item = observation(embeddings, 0, 1, 0.8F, 2);
    CHECK(require(relations.apply_observation_batch(777, std::span<const RelationObservation>(&item, 1))));
    CHECK(!require(relations.apply_observation_batch(777, std::span<const RelationObservation>(&item, 1))));
    CHECK(require(relations.get(id, EvidenceTier::m1)).support == 2);
    CHECK_OK(relations.flush());
  }
  {
    std::ofstream tail(config.state_dir / "m1" / "changes.wal", std::ios::binary | std::ios::app);
    tail.write("BAD", 3);
  }
  {
    RelationRepository recovered; CHECK_OK(recovered.open(config, embeddings.dimension()));
    CHECK(require(recovered.get(id, EvidenceTier::m1)).support == 2);
    const RelationObservation duplicate = observation(embeddings, 0, 1, 0.8F, 2);
    CHECK(!require(recovered.apply_observation_batch(777,
        std::span<const RelationObservation>(&duplicate, 1))));
  }
}

void test_replay_bound_and_roundtrip() {
  ReplayTrace trace;
  trace.created_at_ms = unix_time_ms(); trace.repository_epoch = 7; trace.embedding_checksum = 9;
  trace.lane = Lane::full; trace.input_tokens = {0, 1}; trace.beam_width = 2; trace.candidate_k = 3;
  trace.max_depth = 2; trace.winning_score = 1.0F;
  for (std::size_t depth = 0; depth < 2; ++depth) {
    TraceStep step; step.depth = depth;
    for (std::size_t parent_index = 0; parent_index < 2; ++parent_index) {
      TraceParent parent; parent.token = static_cast<TokenId>(parent_index); parent.parent_score = 0.1F;
      parent.certificate = PruningCertificate{false, true, 3, -1.0e30F, -1.0e30F};
      for (std::size_t candidate = 0; candidate < 3; ++candidate) {
        parent.candidates.push_back(TraceCandidate{static_cast<EdgeId>(1 + depth * 10 + parent_index * 3 + candidate),
                                                   static_cast<TokenId>(candidate), EvidenceTier::m2, 0.5F});
      }
      step.parents.push_back(std::move(parent));
    }
    trace.steps.push_back(std::move(step));
  }
  CHECK_OK(trace.validate());
  CHECK(trace.candidate_record_count() == 12);
  const ReplayTrace restored = require(deserialize_replay_trace(serialize_replay_trace(trace)));
  CHECK(restored.candidate_record_count() == 12);
  restored; // retain compile-time coverage of a const roundtrip value
  trace.steps[0].parents[0].candidates.push_back(TraceCandidate{999, 1, EvidenceTier::m2, 0.2F});
  CHECK(!trace.validate().ok());
}

class AlwaysPromoteAuditor final : public ICounterfactualAuditor {
 public:
  Result<AuditReport> audit(const RelationEdge&, std::span<const TraceId>) const override {
    AuditReport report; report.verdict = AuditVerdict::promote; report.summary = "test"; return report;
  }
};

void test_promotion_journal_recovery() {
  TempDirectory temp("promotion");
  const auto embedding_file = temp.path() / "embeddings.rle";
  CHECK_OK(EmbeddingFileBuilder::from_rows(rows(), embedding_file));
  FrozenEmbeddingStore embeddings; CHECK_OK(embeddings.open(embedding_file));
  RelationRepository relations; CHECK_OK(relations.open(storage(temp.path(), embedding_file), embeddings.dimension()));
  const RelationObservation item = observation(embeddings, 0, 1, 0.9F, 5);
  CHECK(require(relations.apply_observation_batch(900, std::span<const RelationObservation>(&item, 1))));
  const RelationEdge edge = require(relations.get(deterministic_edge_id(0, 1), EvidenceTier::m1));
  const auto promotion_root = temp.path() / "state" / "promotion";
  CHECK_OK(ensure_directory(promotion_root));
  {
    WriteAheadLog journal; CHECK_OK(journal.open(promotion_root / "promotion.wal", Durability::data));
    ByteWriter prepare; prepare.u64(42);
    const auto serialized = serialize_relation_edge(edge);
    prepare.u32(static_cast<std::uint32_t>(serialized.size())); prepare.bytes(serialized);
    CHECK(require(journal.append(1, prepare.data())) > 0);
  }
  AlwaysPromoteAuditor auditor;
  PromotionManager manager(relations, auditor);
  CHECK_OK(manager.open(promotion_root, Durability::data));
  CHECK(require(relations.get(edge.id, EvidenceTier::m2)).tier == EvidenceTier::m2);
  const auto old = relations.get(edge.id, EvidenceTier::m1);
  CHECK(!old && old.status().code() == ErrorCode::not_found);
}

void test_ttl_respects_promotion_pin() {
  TempDirectory temp("ttl");
  const auto embedding_file = temp.path() / "embeddings.rle";
  CHECK_OK(EmbeddingFileBuilder::from_rows(rows(), embedding_file));
  FrozenEmbeddingStore embeddings; CHECK_OK(embeddings.open(embedding_file));
  RelationRepository relations; CHECK_OK(relations.open(storage(temp.path(), embedding_file), embeddings.dimension()));
  const RelationObservation item = observation(embeddings, 0, 1, 0.9F, 5);
  CHECK(require(relations.apply_observation_batch(901, std::span<const RelationObservation>(&item, 1))));
  const EdgeId id = deterministic_edge_id(0, 1);
  EdgePin pin = require(relations.pin_m1(id));
  CHECK(require(relations.expire_m1(unix_time_ms() + 1000)) == 0);
  CHECK(require(relations.get(id, EvidenceTier::m1)).id == id);
  pin.release();
  CHECK(require(relations.expire_m1(unix_time_ms() + 1000)) == 1);
}

void test_trainer_resume_and_parallel_lane_stress() {
  TempDirectory temp("engine");
  const auto embedding_file = temp.path() / "embeddings.rle";
  const auto corpus = temp.path() / "corpus.txt";
  CHECK_OK(EmbeddingFileBuilder::from_rows(rows(), embedding_file));
  {
    std::ofstream output(corpus);
    for (int i = 0; i < 100; ++i) output << "a b c a b d ";
  }
  const EngineConfig config = engine_config(temp.path(), embedding_file);
  EngineHealth first_health;
  TrainingStats first_stats;
  {
    RelationalLanguageEngine engine; CHECK_OK(engine.open(config));
    TrainerOptions options; options.auto_promote = false; options.resume = true;
    first_stats = require(engine.train(corpus, options));
    first_health = engine.health();
    CHECK(first_health.m1_edges > 0);
    std::vector<std::future<void>> clients;
    for (int client = 0; client < 4; ++client) {
      clients.push_back(std::async(std::launch::async, [&engine]() {
        for (int i = 0; i < 100; ++i) {
          const auto result = require(engine.infer_text("a b", "both", 1));
          CHECK(result.full.has_value()); CHECK(result.clean.has_value());
        }
      }));
    }
    for (auto& client : clients) client.get();
  }
  {
    RelationalLanguageEngine engine; CHECK_OK(engine.open(config));
    TrainerOptions options; options.auto_promote = false; options.resume = true;
    const TrainingStats resumed = require(engine.train(corpus, options));
    const EngineHealth second_health = engine.health();
    CHECK(second_health.m1_edges == first_health.m1_edges);
    CHECK(resumed.raw_tokens_seen == first_stats.raw_tokens_seen);
    CHECK(resumed.known_tokens_seen == first_stats.known_tokens_seen);
    CHECK(resumed.batches_committed == first_stats.batches_committed);
  }
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"embedding_roundtrip", test_embedding_roundtrip},
      {"clean_physical_isolation_and_no_ghost", test_clean_physical_isolation_and_no_ghost_branch},
      {"wal_recovery_batch_idempotence_tail_repair", test_wal_recovery_batch_idempotence_and_tail_repair},
      {"replay_bound_and_roundtrip", test_replay_bound_and_roundtrip},
      {"promotion_journal_recovery", test_promotion_journal_recovery},
      {"ttl_respects_promotion_pin", test_ttl_respects_promotion_pin},
      {"trainer_resume_and_parallel_lane_stress", test_trainer_resume_and_parallel_lane_stress},
  };
  std::size_t failed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  std::cout << "tests=" << tests.size() << " failed=" << failed << '\n';
  return failed == 0 ? 0 : 1;
}
