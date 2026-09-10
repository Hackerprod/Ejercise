#pragma once

#include "rlm/config.hpp"
#include "rlm/embedding_store.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/replay.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rlm {

struct SpectralBranch final {
  TokenId token{kInvalidToken};
  std::vector<float> state;
  float score{0.0F};
  std::vector<EdgeId> path_edges;
  std::vector<TokenId> path_tokens;
};

class IScoringController {
 public:
  virtual ~IScoringController() = default;
  [[nodiscard]] virtual Result<float> transition_score(const SpectralBranch& parent,
                                                        const RelationEdge& edge,
                                                        std::span<const TokenId> original_context) const = 0;
  [[nodiscard]] virtual float omitted_upper_bound(float parent_score,
                                                   float next_confidence) const noexcept = 0;
};

class IBranchComposer {
 public:
  virtual ~IBranchComposer() = default;
  [[nodiscard]] virtual Result<SpectralBranch> compose(const SpectralBranch& parent,
                                                        const RelationEdge& edge,
                                                        float transition_score) const = 0;
};

class FrozenLinearController final : public IScoringController {
 public:
  FrozenLinearController(const IEmbeddingStore& embeddings, ScoringConfig config)
      : embeddings_(embeddings), config_(config) {}

  [[nodiscard]] Result<float> transition_score(const SpectralBranch& parent,
                                                const RelationEdge& edge,
                                                std::span<const TokenId> original_context) const override;
  [[nodiscard]] float omitted_upper_bound(float parent_score,
                                           float next_confidence) const noexcept override;

 private:
  const IEmbeddingStore& embeddings_;
  const ScoringConfig config_;
};

class VectorRelationComposer final : public IBranchComposer {
 public:
  VectorRelationComposer(const IEmbeddingStore& embeddings, SearchConfig config)
      : embeddings_(embeddings), config_(config) {}

  [[nodiscard]] Result<SpectralBranch> compose(const SpectralBranch& parent,
                                                const RelationEdge& edge,
                                                float transition_score) const override;

 private:
  const IEmbeddingStore& embeddings_;
  const SearchConfig config_;
};

struct SearchRequest final {
  std::vector<TokenId> context;
  Lane lane{Lane::full};
  std::optional<EdgeId> exclude_edge;
  EdgeId edge_under_test{0};
  TokenId expected_target{kInvalidToken};
  std::size_t max_depth_override{0};
  std::size_t candidate_k_override{0};
  bool capture_trace{true};
};

struct SearchResult final {
  bool has_prediction{false};
  TokenId best_token{kInvalidToken};
  float score{-1.0e30F};
  std::vector<EdgeId> path_edges;
  std::vector<TokenId> path_tokens;
  std::vector<float> final_state;
  bool exact_within_beam{false};
  bool stable_epoch{false};
  Epoch repository_epoch{0};
  std::optional<ReplayTrace> trace;
};

enum class ReplayExactness : std::uint8_t { exact = 0, unknown = 1 };

struct ExactReplayResult final {
  ReplayExactness exactness{ReplayExactness::unknown};
  SearchResult result;
  std::size_t candidate_k_used{0};
  std::string reason;
};

class ISearchStrategy {
 public:
  virtual ~ISearchStrategy() = default;
  [[nodiscard]] virtual Result<SearchResult> run(const SearchRequest& request) const = 0;
  [[nodiscard]] virtual Result<ExactReplayResult> replay_exact(const ReplayTrace& trace,
                                                               std::optional<EdgeId> exclude_edge) const = 0;
};

class BeamSearch final : public ISearchStrategy {
 public:
  BeamSearch(const IEmbeddingStore& embeddings,
             const IRelationRepository& relations,
             const IScoringController& controller,
             const IBranchComposer& composer,
             SearchConfig config)
      : embeddings_(embeddings), relations_(relations), controller_(controller),
        composer_(composer), config_(config) {}

  [[nodiscard]] Result<SearchResult> run(const SearchRequest& request) const override;
  [[nodiscard]] Result<ExactReplayResult> replay_exact(const ReplayTrace& trace,
                                                       std::optional<EdgeId> exclude_edge) const override;

 private:
  [[nodiscard]] Result<SearchResult> run_once(const SearchRequest& request) const;
  [[nodiscard]] Result<std::vector<float>> initial_state(std::span<const TokenId> context) const;
  [[nodiscard]] static std::uint64_t state_fingerprint(std::span<const float> state) noexcept;

  const IEmbeddingStore& embeddings_;
  const IRelationRepository& relations_;
  const IScoringController& controller_;
  const IBranchComposer& composer_;
  const SearchConfig config_;
};

}  // namespace rlm
