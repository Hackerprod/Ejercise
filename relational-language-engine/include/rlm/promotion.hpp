#pragma once

#include "rlm/audit.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/wal.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <unordered_map>

namespace rlm {

struct PromotionResult final {
  AuditVerdict verdict{AuditVerdict::unknown};
  bool committed{false};
  AuditReport audit;
  std::string message;
};

class PromotionManager final {
 public:
  PromotionManager(RelationRepository& relations,
                   const ICounterfactualAuditor& auditor)
      : relations_(relations), auditor_(auditor) {}
  PromotionManager(const PromotionManager&) = delete;
  PromotionManager& operator=(const PromotionManager&) = delete;

  [[nodiscard]] Status open(const std::filesystem::path& root, Durability durability);
  [[nodiscard]] Result<PromotionResult> promote(EdgeId edge_id);
  [[nodiscard]] Status flush();

 private:
  struct Pending final { std::uint64_t transaction{0}; RelationEdge edge; };
  [[nodiscard]] Status apply_journal(const WalRecord& record);
  [[nodiscard]] Status recover_pending();
  [[nodiscard]] Status commit_locked(std::uint64_t transaction, const RelationEdge& edge);

  RelationRepository& relations_;
  const ICounterfactualAuditor& auditor_;
  WriteAheadLog journal_;
  std::mutex commit_mutex_;
  std::unordered_map<std::uint64_t, Pending> pending_;
  std::atomic<std::uint64_t> next_transaction_{1};
  bool opened_{false};
};

}  // namespace rlm
