#pragma once

#include "rlm/config.hpp"
#include "rlm/relation_store.hpp"
#include "rlm/replay.hpp"
#include "rlm/search.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace rlm {

struct AuditCaseResult final {
  TraceId trace_id{0};
  bool exact{false};
  bool causal{false};
  bool clean_control_valid{false};
  float causal_delta{0.0F};
  std::string reason;
};

struct AuditReport final {
  AuditVerdict verdict{AuditVerdict::unknown};
  std::size_t exact_cases{0};
  std::size_t passed_cases{0};
  std::size_t unknown_cases{0};
  std::vector<AuditCaseResult> cases;
  std::string summary;
};

class ICounterfactualAuditor {
 public:
  virtual ~ICounterfactualAuditor() = default;
  [[nodiscard]] virtual Result<AuditReport> audit(const RelationEdge& edge,
                                                  std::span<const TraceId> trace_ids) const = 0;
};

class CounterfactualAuditor final : public ICounterfactualAuditor {
 public:
  CounterfactualAuditor(const ISearchStrategy& search,
                        const IReplayStore& replay_store,
                        AuditConfig config)
      : search_(search), replay_store_(replay_store), config_(config) {}

  [[nodiscard]] Result<AuditReport> audit(const RelationEdge& edge,
                                          std::span<const TraceId> trace_ids) const override;

 private:
  [[nodiscard]] static bool same_result(const SearchResult& a, const SearchResult& b);

  const ISearchStrategy& search_;
  const IReplayStore& replay_store_;
  const AuditConfig config_;
};

}  // namespace rlm
