#pragma once

#include "rlm/audit.hpp"
#include "rlm/config.hpp"
#include "rlm/embedding_store.hpp"
#include "rlm/metrics.hpp"
#include "rlm/promotion.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/replay.hpp"
#include "rlm/search.hpp"
#include "rlm/thread_pool.hpp"
#include "rlm/trainer.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rlm {

struct DualLaneResult final {
  std::optional<SearchResult> full;
  std::optional<SearchResult> clean;
  std::size_t unknown_tokens{0};
};

struct EngineHealth final {
  bool ready{false};
  std::uint64_t embedding_checksum{0};
  std::size_t embedding_dimension{0};
  std::size_t vocabulary_size{0};
  std::size_t m1_edges{0};
  std::size_t m2_edges{0};
  std::size_t replay_records{0};
  Epoch repository_epoch{0};
};

class RelationalLanguageEngine final {
 public:
  RelationalLanguageEngine() = default;
  ~RelationalLanguageEngine() = default;
  RelationalLanguageEngine(const RelationalLanguageEngine&) = delete;
  RelationalLanguageEngine& operator=(const RelationalLanguageEngine&) = delete;

  [[nodiscard]] Status open(EngineConfig config);
  [[nodiscard]] Result<DualLaneResult> infer_text(std::string_view text,
                                                  std::string_view lane = "both",
                                                  std::size_t depth = 0) const;
  [[nodiscard]] Result<DualLaneResult> infer_tokens(std::span<const TokenId> context,
                                                    std::string_view lane = "both",
                                                    std::size_t depth = 0) const;
  [[nodiscard]] Result<TrainingStats> train(const std::filesystem::path& corpus,
                                            TrainerOptions options = {});
  [[nodiscard]] Status checkpoint();
  [[nodiscard]] Result<std::size_t> expire_now();
  [[nodiscard]] EngineHealth health() const;
  [[nodiscard]] std::string metrics_text() const;
  [[nodiscard]] const IEmbeddingStore& embeddings() const { return *embeddings_; }
  [[nodiscard]] const EngineConfig& config() const noexcept { return config_; }

 private:
  [[nodiscard]] Result<SearchResult> run_lane(std::span<const TokenId> context,
                                              Lane lane,
                                              std::size_t depth) const;

  EngineConfig config_;
  std::unique_ptr<FrozenEmbeddingStore> embeddings_;
  std::unique_ptr<RelationRepository> relations_;
  std::unique_ptr<ReplayStore> replay_store_;
  std::unique_ptr<FrozenLinearController> controller_;
  std::unique_ptr<VectorRelationComposer> composer_;
  std::unique_ptr<BeamSearch> search_;
  std::unique_ptr<CounterfactualAuditor> auditor_;
  std::unique_ptr<PromotionManager> promotions_;
  std::unique_ptr<BoundedThreadPool> workers_;
  mutable EngineMetrics metrics_;
  bool opened_{false};
};

}  // namespace rlm
