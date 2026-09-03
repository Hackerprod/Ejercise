#include "rlm/engine.hpp"

#include <chrono>
#include <future>
#include <limits>

namespace rlm {

Status RelationalLanguageEngine::open(EngineConfig config) {
  if (opened_) return Status(ErrorCode::failed_precondition, "engine already open");
  RLM_RETURN_IF_ERROR(config.validate());
  config_ = std::move(config);
  embeddings_ = std::make_unique<FrozenEmbeddingStore>();
  RLM_RETURN_IF_ERROR(embeddings_->open(config_.storage.embedding_file));
  relations_ = std::make_unique<RelationRepository>();
  RLM_RETURN_IF_ERROR(relations_->open(config_.storage, embeddings_->dimension()));
  replay_store_ = std::make_unique<ReplayStore>();
  RLM_RETURN_IF_ERROR(replay_store_->open(config_.storage.state_dir / "replay", config_.replay,
                                          config_.storage.durability));
  controller_ = std::make_unique<FrozenLinearController>(*embeddings_, config_.scoring);
  composer_ = std::make_unique<VectorRelationComposer>(*embeddings_, config_.search);
  search_ = std::make_unique<BeamSearch>(*embeddings_, *relations_, *controller_, *composer_, config_.search);
  auditor_ = std::make_unique<CounterfactualAuditor>(*search_, *replay_store_, config_.audit);
  promotions_ = std::make_unique<PromotionManager>(*relations_, *auditor_);
  RLM_RETURN_IF_ERROR(promotions_->open(config_.storage.state_dir / "promotion", config_.storage.durability));
  try {
    workers_ = std::make_unique<BoundedThreadPool>(config_.runtime.worker_threads,
                                                   config_.runtime.queue_capacity);
  } catch (const std::exception& error) {
    return Status(ErrorCode::resource_exhausted, std::string("cannot create worker pool: ") + error.what());
  }
  opened_ = true;
  return Status::Ok();
}

Result<SearchResult> RelationalLanguageEngine::run_lane(std::span<const TokenId> context,
                                                        Lane lane,
                                                        std::size_t depth) const {
  SearchRequest request;
  request.context.assign(context.begin(), context.end());
  request.lane = lane;
  request.max_depth_override = depth;
  request.capture_trace = false;
  return search_->run(request);
}

Result<DualLaneResult> RelationalLanguageEngine::infer_text(std::string_view text,
                                                            std::string_view lane,
                                                            std::size_t depth) const {
  if (!opened_) return Status(ErrorCode::failed_precondition, "engine is not open");
  auto tokens = tokenize_whitespace(text, *embeddings_, config_.training.reject_unknown_tokens);
  if (!tokens) return tokens.status();
  auto result = infer_tokens(tokens.value().tokens, lane, depth);
  if (!result) return result.status();
  result.value().unknown_tokens = tokens.value().unknown_tokens;
  return result;
}

Result<DualLaneResult> RelationalLanguageEngine::infer_tokens(std::span<const TokenId> context,
                                                              std::string_view lane,
                                                              std::size_t depth) const {
  if (!opened_) return Status(ErrorCode::failed_precondition, "engine is not open");
  if (context.empty()) return Status(ErrorCode::invalid_argument, "inference context is empty");
  const bool want_full = lane == "both" || lane == "full";
  const bool want_clean = lane == "both" || lane == "clean";
  if (!want_full && !want_clean) return Status(ErrorCode::invalid_argument, "lane must be full, clean, or both");
  const auto started = std::chrono::steady_clock::now();
  DualLaneResult output;
  bool success = false;
  auto finish_metrics = [&]() {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    metrics_.record_inference(success, want_full && want_clean,
                              static_cast<std::uint64_t>(std::max<std::int64_t>(0, elapsed)));
  };
  if (want_full && want_clean && config_.runtime.parallel_lanes && workers_->thread_count() >= 2) {
    auto full_future = workers_->submit([this, values = std::vector<TokenId>(context.begin(), context.end()), depth]() {
      return run_lane(values, Lane::full, depth);
    });
    auto clean_future = workers_->submit([this, values = std::vector<TokenId>(context.begin(), context.end()), depth]() {
      return run_lane(values, Lane::clean, depth);
    });
    auto full = full_future.get();
    auto clean = clean_future.get();
    if (!full) { finish_metrics(); return full.status(); }
    if (!clean) { finish_metrics(); return clean.status(); }
    output.full = std::move(full).value();
    output.clean = std::move(clean).value();
  } else {
    if (want_full) {
      auto full = run_lane(context, Lane::full, depth);
      if (!full) { finish_metrics(); return full.status(); }
      output.full = std::move(full).value();
    }
    if (want_clean) {
      auto clean = run_lane(context, Lane::clean, depth);
      if (!clean) { finish_metrics(); return clean.status(); }
      output.clean = std::move(clean).value();
    }
  }
  success = true;
  finish_metrics();
  return output;
}

Result<TrainingStats> RelationalLanguageEngine::train(const std::filesystem::path& corpus,
                                                      TrainerOptions options) {
  if (!opened_) return Status(ErrorCode::failed_precondition, "engine is not open");
  Trainer trainer(*embeddings_, *relations_, *search_, *replay_store_, *promotions_, config_);
  return trainer.train_file(corpus, options);
}

Status RelationalLanguageEngine::checkpoint() {
  if (!opened_) return Status(ErrorCode::failed_precondition, "engine is not open");
  RLM_RETURN_IF_ERROR(replay_store_->flush());
  RLM_RETURN_IF_ERROR(promotions_->flush());
  return relations_->checkpoint();
}

Result<std::size_t> RelationalLanguageEngine::expire_now() {
  if (!opened_) return Status(ErrorCode::failed_precondition, "engine is not open");
  if (config_.training.m1_ttl_seconds == 0) return std::size_t{0};
  const std::uint64_t ttl_ms = config_.training.m1_ttl_seconds > std::numeric_limits<std::uint64_t>::max() / 1000ULL
      ? std::numeric_limits<std::uint64_t>::max()
      : config_.training.m1_ttl_seconds * 1000ULL;
  const std::uint64_t now = unix_time_ms();
  return relations_->expire_m1(now > ttl_ms ? now - ttl_ms : 0);
}

EngineHealth RelationalLanguageEngine::health() const {
  EngineHealth output;
  output.ready = opened_;
  if (!opened_) return output;
  output.embedding_checksum = embeddings_->checksum();
  output.embedding_dimension = embeddings_->dimension();
  output.vocabulary_size = embeddings_->token_count();
  output.m1_edges = relations_->m1_count();
  output.m2_edges = relations_->m2_count();
  output.replay_records = replay_store_->size();
  output.repository_epoch = relations_->epoch();
  return output;
}

std::string RelationalLanguageEngine::metrics_text() const {
  const EngineHealth state = health();
  return metrics_.prometheus(state.m1_edges, state.m2_edges, state.replay_records);
}

}  // namespace rlm
