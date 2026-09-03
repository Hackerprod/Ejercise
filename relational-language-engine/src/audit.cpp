#include "rlm/audit.hpp"

#include <algorithm>
#include <cmath>

namespace rlm {
namespace {
ReplayTrace clean_request_from(const ReplayTrace& source) {
  ReplayTrace clean;
  clean.created_at_ms = source.created_at_ms;
  clean.repository_epoch = source.repository_epoch;
  clean.embedding_checksum = source.embedding_checksum;
  clean.lane = Lane::clean;
  clean.input_tokens = source.input_tokens;
  clean.beam_width = source.beam_width;
  clean.candidate_k = source.candidate_k;
  clean.max_depth = source.max_depth;
  clean.edge_under_test = source.edge_under_test;
  clean.expected_target = source.expected_target;
  clean.winning_score = -1.0e30F;
  return clean;
}
}  // namespace

bool CounterfactualAuditor::same_result(const SearchResult& a, const SearchResult& b) {
  if (a.has_prediction != b.has_prediction) return false;
  if (!a.has_prediction) return true;
  return a.best_token == b.best_token && a.path_edges == b.path_edges &&
         std::abs(a.score - b.score) <= 1.0e-5F &&
         l2_distance(a.final_state, b.final_state) <= 1.0e-4F;
}

Result<AuditReport> CounterfactualAuditor::audit(const RelationEdge& edge,
                                                 std::span<const TraceId> trace_ids) const {
  if (edge.tier != EvidenceTier::m1) return Status(ErrorCode::invalid_argument, "only M1 edges can be audited for promotion");
  if (trace_ids.empty()) {
    AuditReport report;
    report.summary = "edge has no replay evidence";
    return report;
  }
  AuditReport report;
  const std::size_t case_limit = std::min(config_.max_cases, trace_ids.size());
  report.cases.reserve(case_limit);
  for (std::size_t i = 0; i < case_limit; ++i) {
    AuditCaseResult case_result;
    case_result.trace_id = trace_ids[i];
    auto trace_result = replay_store_.get(trace_ids[i]);
    if (!trace_result) {
      ++report.unknown_cases;
      case_result.reason = trace_result.status().to_string();
      report.cases.push_back(std::move(case_result));
      continue;
    }
    const ReplayTrace& trace = trace_result.value();
    if (trace.edge_under_test != edge.id ||
        std::find(trace.winning_edges.begin(), trace.winning_edges.end(), edge.id) == trace.winning_edges.end()) {
      case_result.reason = "trace does not establish use of the candidate edge";
      ++report.unknown_cases;
      report.cases.push_back(std::move(case_result));
      continue;
    }
    auto original = search_.replay_exact(trace, std::nullopt);
    if (!original) return original.status();
    if (original.value().exactness != ReplayExactness::exact || !original.value().result.has_prediction ||
        std::find(original.value().result.path_edges.begin(), original.value().result.path_edges.end(), edge.id) ==
            original.value().result.path_edges.end()) {
      case_result.reason = original.value().reason.empty() ? "original replay is not exact" : original.value().reason;
      ++report.unknown_cases;
      report.cases.push_back(std::move(case_result));
      continue;
    }
    auto removed = search_.replay_exact(trace, edge.id);
    if (!removed) return removed.status();
    if (removed.value().exactness != ReplayExactness::exact) {
      case_result.reason = removed.value().reason;
      ++report.unknown_cases;
      report.cases.push_back(std::move(case_result));
      continue;
    }
    ReplayTrace clean_trace = clean_request_from(trace);
    auto clean_a = search_.replay_exact(clean_trace, std::nullopt);
    auto clean_b = search_.replay_exact(clean_trace, edge.id);
    if (!clean_a) return clean_a.status();
    if (!clean_b) return clean_b.status();
    if (clean_a.value().exactness != ReplayExactness::exact || clean_b.value().exactness != ReplayExactness::exact) {
      case_result.reason = "CLEAN control could not be reproduced exactly";
      ++report.unknown_cases;
      report.cases.push_back(std::move(case_result));
      continue;
    }
    case_result.clean_control_valid = same_result(clean_a.value().result, clean_b.value().result);
    if (!case_result.clean_control_valid) {
      case_result.reason = "excluding an M1 edge changed CLEAN execution";
      case_result.exact = true;
      ++report.exact_cases;
      report.cases.push_back(std::move(case_result));
      continue;
    }
    const SearchResult& with_edge = original.value().result;
    const SearchResult& without_edge = removed.value().result;
    if (!without_edge.has_prediction || with_edge.best_token != without_edge.best_token) {
      case_result.causal = true;
      case_result.causal_delta = without_edge.has_prediction ? with_edge.score - without_edge.score : with_edge.score;
    } else {
      case_result.causal_delta = with_edge.score - without_edge.score;
      case_result.causal = case_result.causal_delta >= config_.causal_margin ||
                           with_edge.path_edges != without_edge.path_edges;
    }
    case_result.exact = true;
    ++report.exact_cases;
    if (case_result.causal && case_result.clean_control_valid) {
      ++report.passed_cases;
      case_result.reason = "exact counterfactual changed the winning derivation while CLEAN remained invariant";
    } else {
      case_result.reason = "candidate edge did not meet the configured causal margin";
    }
    report.cases.push_back(std::move(case_result));
  }

  const float pass_fraction = report.exact_cases == 0
      ? 0.0F
      : static_cast<float>(report.passed_cases) / static_cast<float>(report.exact_cases);
  if (report.exact_cases >= config_.min_exact_cases &&
      report.unknown_cases <= config_.max_unknown_cases &&
      pass_fraction >= config_.min_pass_fraction) {
    report.verdict = AuditVerdict::promote;
    report.summary = "promotion criteria satisfied with exact counterfactual evidence";
  } else if (report.exact_cases >= config_.min_exact_cases && pass_fraction < config_.min_pass_fraction) {
    report.verdict = AuditVerdict::reject;
    report.summary = "exact evidence is sufficient but causal pass fraction is below threshold";
  } else {
    report.verdict = AuditVerdict::unknown;
    report.summary = "insufficient exact evidence; edge remains in M1";
  }
  return report;
}

}  // namespace rlm
