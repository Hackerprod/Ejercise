#pragma once

#include "rlm/config.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/wal.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace rlm {

struct TraceCandidate final {
  EdgeId edge_id{0};
  TokenId target{kInvalidToken};
  EvidenceTier tier{EvidenceTier::m1};
  float transition_score{0.0F};
};

struct PruningCertificate final {
  bool truncated{false};
  bool certified_safe{true};
  std::size_t enumerated{0};
  float beam_cutoff{-1.0e30F};
  float omitted_upper_bound{-1.0e30F};
};

struct TraceParent final {
  TokenId token{kInvalidToken};
  std::uint64_t state_fingerprint{0};
  float parent_score{0.0F};
  std::vector<TraceCandidate> candidates;
  PruningCertificate certificate;
};

struct TraceStep final {
  std::size_t depth{0};
  std::vector<TraceParent> parents;
};

struct ReplayTrace final {
  TraceId id{0};
  std::uint64_t created_at_ms{0};
  Epoch repository_epoch{0};
  std::uint64_t embedding_checksum{0};
  Lane lane{Lane::full};
  std::vector<TokenId> input_tokens;
  std::size_t beam_width{0};
  std::size_t candidate_k{0};
  std::size_t max_depth{0};
  EdgeId edge_under_test{0};
  TokenId expected_target{kInvalidToken};
  std::vector<TraceStep> steps;
  std::vector<EdgeId> winning_edges;
  std::vector<TokenId> winning_tokens;
  float winning_score{-1.0e30F};
  bool exhaustive{false};

  [[nodiscard]] Status validate() const;
  [[nodiscard]] std::size_t candidate_record_count() const noexcept;
};

[[nodiscard]] std::vector<std::byte> serialize_replay_trace(const ReplayTrace& trace);
[[nodiscard]] Result<ReplayTrace> deserialize_replay_trace(std::span<const std::byte> payload);

class IReplayStore {
 public:
  virtual ~IReplayStore() = default;
  [[nodiscard]] virtual Result<TraceId> put(ReplayTrace trace) = 0;
  [[nodiscard]] virtual Result<ReplayTrace> get(TraceId id) const = 0;
  [[nodiscard]] virtual Status flush() = 0;
};

class ReplayStore final : public IReplayStore {
 public:
  ReplayStore() = default;
  ~ReplayStore() override = default;
  ReplayStore(const ReplayStore&) = delete;
  ReplayStore& operator=(const ReplayStore&) = delete;

  [[nodiscard]] Status open(const std::filesystem::path& root,
                            const ReplayConfig& config,
                            Durability durability);
  [[nodiscard]] Result<TraceId> put(ReplayTrace trace) override;
  [[nodiscard]] Result<ReplayTrace> get(TraceId id) const override;
  [[nodiscard]] Status flush() override;
  [[nodiscard]] std::size_t size() const;

 private:
  [[nodiscard]] Status apply_record(const WalRecord& record);
  [[nodiscard]] Status erase_locked(TraceId id, bool persist);
  void index_locked(ReplayTrace trace);

  mutable std::mutex mutex_;
  std::size_t max_records_{0};
  WriteAheadLog wal_;
  std::unordered_map<TraceId, ReplayTrace> traces_;
  std::deque<TraceId> insertion_order_;
  std::atomic<std::uint64_t> id_counter_{1};
};

}  // namespace rlm
