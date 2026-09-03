#include "rlm/search.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace rlm {
namespace {
constexpr float kNegativeFinite = -1.0e30F;
constexpr float kCertificateEpsilon = 1.0e-6F;

bool branch_order(const SpectralBranch& a, const SpectralBranch& b) {
  if (a.score != b.score) return a.score > b.score;
  if (a.token != b.token) return a.token < b.token;
  return a.path_edges < b.path_edges;
}

bool contains_token(std::span<const TokenId> values, TokenId token) {
  return std::find(values.begin(), values.end(), token) != values.end();
}
}  // namespace

Result<float> FrozenLinearController::transition_score(
    const SpectralBranch& parent,
    const RelationEdge& edge,
    std::span<const TokenId> original_context) const {
  if (parent.state.size() != embeddings_.dimension()) {
    return Status(ErrorCode::invalid_argument, "branch state dimension mismatch");
  }
  if (edge.relation.size() != embeddings_.dimension()) {
    return Status(ErrorCode::data_loss, "edge relation dimension mismatch");
  }
  std::vector<float> target(embeddings_.dimension());
  Status status = embeddings_.copy_embedding(edge.target, target);
  if (!status) return status;
  const std::vector<float> relation = edge.relation.dequantize();
  std::vector<float> desired(relation.size());
  std::vector<float> predicted(relation.size());
  for (std::size_t i = 0; i < relation.size(); ++i) {
    desired[i] = target[i] - parent.state[i];
    predicted[i] = parent.state[i] + relation[i];
  }
  const float relation_alignment = cosine(relation, desired);
  const float target_alignment = cosine(predicted, target);
  const float context_alignment = cosine(parent.state, target);
  constexpr double kSupportScale = 13.815511557963774;  // log(1 + 1e6)
  const float support_signal = std::clamp(static_cast<float>(std::log1p(static_cast<double>(edge.support)) / kSupportScale), 0.0F, 1.0F);
  const bool repeated = contains_token(original_context, edge.target) || contains_token(parent.path_tokens, edge.target);
  const float score = parent.score +
                      config_.confidence_weight * edge.confidence +
                      config_.support_weight * support_signal +
                      config_.relation_weight * relation_alignment +
                      config_.target_weight * target_alignment +
                      config_.context_weight * context_alignment -
                      (repeated ? config_.repetition_penalty : 0.0F);
  if (!std::isfinite(score)) return Status(ErrorCode::internal, "controller produced a non-finite score");
  return score;
}

float FrozenLinearController::omitted_upper_bound(float parent_score,
                                                  float next_confidence) const noexcept {
  const float bounded_confidence = std::clamp(next_confidence, 0.0F, 1.0F);
  return parent_score + config_.confidence_weight * bounded_confidence +
         config_.support_weight + config_.relation_weight +
         config_.target_weight + config_.context_weight;
}

Result<SpectralBranch> VectorRelationComposer::compose(const SpectralBranch& parent,
                                                       const RelationEdge& edge,
                                                       float transition_score_value) const {
  if (parent.state.size() != embeddings_.dimension() || edge.relation.size() != embeddings_.dimension()) {
    return Status(ErrorCode::invalid_argument, "cannot compose vectors with different dimensions");
  }
  if (!std::isfinite(transition_score_value)) return Status(ErrorCode::invalid_argument, "transition score is non-finite");
  if (edge.tier != EvidenceTier::m1 && edge.tier != EvidenceTier::m2) {
    return Status(ErrorCode::data_loss, "edge tier is invalid");
  }
  std::vector<float> target(embeddings_.dimension());
  Status status = embeddings_.copy_embedding(edge.target, target);
  if (!status) return status;
  const std::vector<float> relation = edge.relation.dequantize();
  std::vector<float> next(parent.state.size());
  for (std::size_t i = 0; i < next.size(); ++i) {
    next[i] = config_.state_decay * parent.state[i] +
              config_.relation_mix * relation[i] +
              config_.target_mix * target[i];
  }
  next = normalized(next);
  SpectralBranch branch;
  branch.token = edge.target;
  branch.state = std::move(next);
  branch.score = transition_score_value;
  branch.path_edges = parent.path_edges;
  branch.path_edges.push_back(edge.id);
  branch.path_tokens = parent.path_tokens;
  branch.path_tokens.push_back(edge.target);
  return branch;
}

Result<std::vector<float>> BeamSearch::initial_state(std::span<const TokenId> context) const {
  if (context.empty()) return Status(ErrorCode::invalid_argument, "search context is empty");
  std::vector<float> state(embeddings_.dimension(), 0.0F);
  std::vector<float> temporary(embeddings_.dimension());
  double total_weight = 0.0;
  const std::size_t start = context.size() > 64 ? context.size() - 64 : 0;
  for (std::size_t i = start; i < context.size(); ++i) {
    Status status = embeddings_.copy_embedding(context[i], temporary);
    if (!status) return status;
    const double weight = static_cast<double>(i - start + 1);
    total_weight += weight;
    for (std::size_t d = 0; d < state.size(); ++d) state[d] += static_cast<float>(temporary[d] * weight);
  }
  if (total_weight <= 0.0) return Status(ErrorCode::internal, "invalid context weight");
  for (float& value : state) value = static_cast<float>(value / total_weight);
  return normalized(state);
}

std::uint64_t BeamSearch::state_fingerprint(std::span<const float> state) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (float value : state) {
    const long rounded = std::lround(std::clamp(value, -1.0F, 1.0F) * 32767.0F);
    const std::int16_t quantized = static_cast<std::int16_t>(std::clamp(rounded, -32767L, 32767L));
    hash = stable_hash64(std::as_bytes(std::span{&quantized, std::size_t{1}}), hash);
  }
  return hash;
}

Result<SearchResult> BeamSearch::run(const SearchRequest& request) const {
  Result<SearchResult> last = Status(ErrorCode::unknown, "search did not run");
  for (int attempt = 0; attempt < 3; ++attempt) {
    last = run_once(request);
    if (!last) return last.status();
    if (last.value().stable_epoch) return last;
  }
  return last;
}

Result<SearchResult> BeamSearch::run_once(const SearchRequest& request) const {
  if (request.context.empty()) return Status(ErrorCode::invalid_argument, "search context is empty");
  if (request.lane != Lane::full && request.lane != Lane::clean) return Status(ErrorCode::invalid_argument, "search lane is invalid");
  const std::size_t depth_limit = request.max_depth_override == 0 ? config_.max_depth : request.max_depth_override;
  const std::size_t candidate_limit = request.candidate_k_override == 0 ? config_.candidate_k : request.candidate_k_override;
  if (depth_limit == 0 || depth_limit > 4096 || candidate_limit == 0 || candidate_limit > config_.replay_reopen_limit) {
    return Status(ErrorCode::invalid_argument, "search override exceeds configured safety bounds");
  }
  const Epoch start_epoch = relations_.epoch();
  auto initial = initial_state(request.context);
  if (!initial) return initial.status();
  SpectralBranch root;
  root.token = request.context.back();
  root.state = std::move(initial).value();
  root.score = 0.0F;
  std::vector<SpectralBranch> beam{root};
  ReplayTrace trace;
  trace.created_at_ms = unix_time_ms();
  trace.repository_epoch = start_epoch;
  trace.embedding_checksum = embeddings_.checksum();
  trace.lane = request.lane;
  trace.input_tokens = request.context;
  trace.beam_width = config_.beam_width;
  trace.candidate_k = candidate_limit;
  trace.max_depth = depth_limit;
  trace.edge_under_test = request.edge_under_test;
  trace.expected_target = request.expected_target;
  bool all_certified = true;

  for (std::size_t depth = 0; depth < depth_limit; ++depth) {
    TraceStep trace_step;
    trace_step.depth = depth;
    trace_step.parents.reserve(beam.size());
    std::vector<SpectralBranch> expanded;
    expanded.reserve(beam.size() * std::min(candidate_limit, std::size_t{1024}));

    struct PendingCertificate final { std::size_t trace_parent_index; float upper_bound; bool truncated; };
    std::vector<PendingCertificate> pending;
    pending.reserve(beam.size());

    for (const SpectralBranch& parent : beam) {
      auto page = relations_.outgoing(parent.token, request.lane, candidate_limit, request.exclude_edge);
      if (!page) return page.status();
      TraceParent trace_parent;
      trace_parent.token = parent.token;
      trace_parent.state_fingerprint = state_fingerprint(parent.state);
      trace_parent.parent_score = parent.score;
      trace_parent.certificate.truncated = page.value().has_more;
      trace_parent.certificate.enumerated = page.value().edges.size();
      trace_parent.candidates.reserve(page.value().edges.size());
      for (const RelationEdge& edge : page.value().edges) {
        if (request.lane == Lane::clean && edge.tier == EvidenceTier::m1) {
          return Status(ErrorCode::internal, "CLEAN repository leaked an M1 edge");
        }
        auto score = controller_.transition_score(parent, edge, request.context);
        if (!score) return score.status();
        auto child = composer_.compose(parent, edge, score.value());
        if (!child) return child.status();
        expanded.push_back(std::move(child).value());
        trace_parent.candidates.push_back(TraceCandidate{edge.id, edge.target, edge.tier, score.value()});
      }
      const float upper = page.value().has_more
          ? controller_.omitted_upper_bound(parent.score, page.value().next_confidence)
          : kNegativeFinite;
      pending.push_back(PendingCertificate{trace_step.parents.size(), upper, page.value().has_more});
      trace_step.parents.push_back(std::move(trace_parent));
    }

    if (expanded.empty()) {
      for (PendingCertificate& certificate : pending) {
        TraceParent& parent = trace_step.parents[certificate.trace_parent_index];
        parent.certificate.beam_cutoff = kNegativeFinite;
        parent.certificate.omitted_upper_bound = certificate.upper_bound;
        parent.certificate.certified_safe = !certificate.truncated;
        all_certified = all_certified && parent.certificate.certified_safe;
      }
      if (request.capture_trace) trace.steps.push_back(std::move(trace_step));
      break;
    }
    std::sort(expanded.begin(), expanded.end(), branch_order);
    const float cutoff = expanded.size() >= config_.beam_width ? expanded[config_.beam_width - 1].score : kNegativeFinite;
    if (expanded.size() > config_.beam_width) expanded.resize(config_.beam_width);
    for (PendingCertificate& certificate : pending) {
      TraceParent& parent = trace_step.parents[certificate.trace_parent_index];
      parent.certificate.beam_cutoff = cutoff;
      parent.certificate.omitted_upper_bound = certificate.upper_bound;
      parent.certificate.certified_safe = !certificate.truncated || certificate.upper_bound <= cutoff + kCertificateEpsilon;
      all_certified = all_certified && parent.certificate.certified_safe;
    }
    if (request.capture_trace) trace.steps.push_back(std::move(trace_step));
    beam = std::move(expanded);
  }

  SearchResult result;
  result.repository_epoch = start_epoch;
  result.stable_epoch = relations_.epoch() == start_epoch;
  result.exact_within_beam = all_certified && result.stable_epoch;
  if (!beam.empty() && !beam.front().path_edges.empty()) {
    const SpectralBranch& best = beam.front();
    result.has_prediction = true;
    result.best_token = best.token;
    result.score = best.score;
    result.path_edges = best.path_edges;
    result.path_tokens = best.path_tokens;
    result.final_state = best.state;
  } else {
    result.final_state = root.state;
  }
  if (request.capture_trace) {
    trace.winning_edges = result.path_edges;
    trace.winning_tokens = result.path_tokens;
    trace.winning_score = result.score;
    trace.exhaustive = result.exact_within_beam;
    result.trace = std::move(trace);
  }
  return result;
}

Result<ExactReplayResult> BeamSearch::replay_exact(const ReplayTrace& trace,
                                                  std::optional<EdgeId> exclude_edge) const {
  const Status validation = trace.validate();
  if (!validation) return validation;
  ExactReplayResult output;
  if (trace.embedding_checksum != embeddings_.checksum()) {
    output.reason = "frozen embedding checksum changed";
    return output;
  }
  if (trace.repository_epoch != relations_.epoch()) {
    output.reason = "relation repository epoch changed since evidence capture";
    return output;
  }
  std::size_t candidate_k = std::max<std::size_t>(1, trace.candidate_k);
  while (candidate_k <= config_.replay_reopen_limit) {
    SearchRequest request;
    request.context = trace.input_tokens;
    request.lane = trace.lane;
    request.exclude_edge = exclude_edge;
    request.edge_under_test = trace.edge_under_test;
    request.expected_target = trace.expected_target;
    request.max_depth_override = trace.max_depth;
    request.candidate_k_override = candidate_k;
    request.capture_trace = true;
    auto replay = run(request);
    if (!replay) return replay.status();
    output.result = std::move(replay).value();
    output.candidate_k_used = candidate_k;
    if (output.result.stable_epoch && output.result.exact_within_beam) {
      output.exactness = ReplayExactness::exact;
      output.reason = "all pruning certificates are exact within configured beam semantics";
      return output;
    }
    if (!output.result.stable_epoch) {
      output.reason = "repository mutated during replay";
      return output;
    }
    if (candidate_k == config_.replay_reopen_limit) break;
    const std::size_t doubled = candidate_k > config_.replay_reopen_limit / 2
        ? config_.replay_reopen_limit
        : candidate_k * 2;
    if (doubled == candidate_k) break;
    candidate_k = doubled;
  }
  output.reason = "reopen limit reached before every pruning certificate became safe";
  return output;
}

}  // namespace rlm
