#pragma once

#include "rlm/config.hpp"
#include "rlm/embedding_store.hpp"
#include "rlm/promotion.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/replay.hpp"
#include "rlm/search.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace rlm {

struct TrainingStats final {
  std::uint64_t raw_tokens_seen{0};
  std::uint64_t known_tokens_seen{0};
  std::uint64_t unknown_tokens_seen{0};
  std::uint64_t batches_committed{0};
  std::uint64_t unique_relations_observed{0};
  std::uint64_t replay_traces_written{0};
  std::uint64_t promotions_committed{0};
  std::uint64_t promotions_rejected{0};
  std::uint64_t promotions_unknown{0};
  std::uint64_t m1_expired{0};
};

struct TrainerOptions final {
  bool resume{true};
  bool auto_promote{true};
  bool checkpoint_at_end{true};
};

class Trainer final {
 public:
  Trainer(const IEmbeddingStore& embeddings,
          RelationRepository& relations,
          ISearchStrategy& search,
          IReplayStore& replay_store,
          PromotionManager& promotions,
          EngineConfig config)
      : embeddings_(embeddings), relations_(relations), search_(search),
        replay_store_(replay_store), promotions_(promotions), config_(std::move(config)) {}

  [[nodiscard]] Result<TrainingStats> train_file(const std::filesystem::path& corpus,
                                                  TrainerOptions options = {});

 private:
  struct Checkpoint;
  struct Aggregate;
  [[nodiscard]] Result<std::uint64_t> corpus_fingerprint(const std::filesystem::path& corpus) const;
  [[nodiscard]] Result<Checkpoint> load_checkpoint(const std::filesystem::path& corpus,
                                                    std::uint64_t corpus_hash) const;
  [[nodiscard]] Status save_checkpoint(const Checkpoint& checkpoint) const;
  [[nodiscard]] Result<std::vector<EdgeId>> process_batch(
      BatchId batch_id,
      std::span<const TokenId> prefix,
      std::span<const TokenId> batch,
      TrainingStats& stats,
      std::unordered_map<EdgeId, std::vector<std::vector<TokenId>>>& samples);
  [[nodiscard]] Status collect_evidence_and_promote(
      std::span<const EdgeId> edge_ids,
      const std::unordered_map<EdgeId, std::vector<std::vector<TokenId>>>& samples,
      bool auto_promote,
      TrainingStats& stats);

  const IEmbeddingStore& embeddings_;
  RelationRepository& relations_;
  ISearchStrategy& search_;
  IReplayStore& replay_store_;
  PromotionManager& promotions_;
  const EngineConfig config_;
};

}  // namespace rlm
