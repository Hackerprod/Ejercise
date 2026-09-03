#pragma once

#include "rlm/config.hpp"
#include "rlm/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rlm {

struct RelationEdge final {
  EdgeId id{0};
  TokenId source{kInvalidToken};
  TokenId target{kInvalidToken};
  EvidenceTier tier{EvidenceTier::m1};
  QuantizedVector relation;
  float confidence{0.0F};
  std::uint64_t support{0};
  std::uint64_t created_at_ms{0};
  std::uint64_t last_seen_ms{0};
  std::vector<TraceId> evidence;

  [[nodiscard]] Status validate(std::size_t dimension) const;
};

struct RelationObservation final {
  TokenId source{kInvalidToken};
  TokenId target{kInvalidToken};
  QuantizedVector relation;
  float confidence{0.0F};
  std::uint64_t count{1};
  std::uint64_t observed_at_ms{0};
};

struct CandidatePage final {
  std::vector<RelationEdge> edges;
  bool has_more{false};
  float next_confidence{0.0F};
};

class EdgePin final {
 public:
  EdgePin() = default;
  explicit EdgePin(std::function<void()> release) : release_(std::move(release)) {}
  ~EdgePin() { release(); }
  EdgePin(const EdgePin&) = delete;
  EdgePin& operator=(const EdgePin&) = delete;
  EdgePin(EdgePin&& other) noexcept : release_(std::move(other.release_)) { other.release_ = {}; }
  EdgePin& operator=(EdgePin&& other) noexcept {
    if (this != &other) {
      release();
      release_ = std::move(other.release_);
      other.release_ = {};
    }
    return *this;
  }
  void release() noexcept {
    if (release_) {
      auto callback = std::move(release_);
      release_ = {};
      callback();
    }
  }
  [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(release_); }

 private:
  std::function<void()> release_;
};

class IRelationRepository {
 public:
  virtual ~IRelationRepository() = default;
  [[nodiscard]] virtual Result<CandidatePage> outgoing(TokenId source,
                                                       Lane lane,
                                                       std::size_t limit,
                                                       std::optional<EdgeId> exclude = std::nullopt) const = 0;
  [[nodiscard]] virtual Result<RelationEdge> get(EdgeId id, EvidenceTier tier) const = 0;
  [[nodiscard]] virtual Epoch epoch() const noexcept = 0;
};

class RelationRepository final : public IRelationRepository {
 public:
  RelationRepository();
  ~RelationRepository() override;
  RelationRepository(const RelationRepository&) = delete;
  RelationRepository& operator=(const RelationRepository&) = delete;

  [[nodiscard]] Status open(const StorageConfig& config, std::size_t embedding_dimension);
  [[nodiscard]] Result<CandidatePage> outgoing(TokenId source,
                                               Lane lane,
                                               std::size_t limit,
                                               std::optional<EdgeId> exclude = std::nullopt) const override;
  [[nodiscard]] Result<RelationEdge> get(EdgeId id, EvidenceTier tier) const override;
  [[nodiscard]] Epoch epoch() const noexcept override;

  [[nodiscard]] Result<bool> apply_observation_batch(BatchId batch_id,
                                                      std::span<const RelationObservation> observations);
  [[nodiscard]] Status attach_evidence(EdgeId edge_id,
                                       std::span<const TraceId> trace_ids,
                                       std::size_t max_evidence);
  [[nodiscard]] Status upsert_m2(const RelationEdge& edge);
  [[nodiscard]] Status erase_m1(EdgeId edge_id);
  [[nodiscard]] Result<EdgePin> pin_m1(EdgeId edge_id);
  [[nodiscard]] Result<std::size_t> expire_m1(std::uint64_t cutoff_ms,
                                              std::size_t max_to_remove = 100000);

  [[nodiscard]] std::size_t m1_count() const;
  [[nodiscard]] std::size_t m2_count() const;
  [[nodiscard]] Status flush();
  [[nodiscard]] Status checkpoint();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::vector<std::byte> serialize_relation_edge(const RelationEdge& edge);
[[nodiscard]] Result<RelationEdge> deserialize_relation_edge(std::span<const std::byte> payload,
                                                             std::size_t expected_dimension);

}  // namespace rlm
