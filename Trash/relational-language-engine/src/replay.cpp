#include "rlm/replay.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rlm {
namespace {
constexpr std::uint16_t kPut = 1;
constexpr std::uint16_t kDelete = 2;
constexpr std::size_t kMaxTraceTokens = 1'000'000;
constexpr std::size_t kMaxTraceSteps = 4096;
constexpr std::size_t kMaxParentsPerStep = 65'536;
constexpr std::size_t kMaxCandidatesPerParent = 1'000'000;

bool valid_tier(std::uint8_t value) {
  return value == static_cast<std::uint8_t>(EvidenceTier::m1) ||
         value == static_cast<std::uint8_t>(EvidenceTier::m2);
}
}  // namespace

std::size_t ReplayTrace::candidate_record_count() const noexcept {
  std::size_t count = 0;
  for (const TraceStep& step : steps) {
    for (const TraceParent& parent : step.parents) count += parent.candidates.size();
  }
  return count;
}

Status ReplayTrace::validate() const {
  if (created_at_ms == 0 || embedding_checksum == 0 || input_tokens.empty() ||
      input_tokens.size() > kMaxTraceTokens || beam_width == 0 || candidate_k == 0 || max_depth == 0 ||
      beam_width > kMaxParentsPerStep || candidate_k > kMaxCandidatesPerParent || max_depth > kMaxTraceSteps ||
      steps.size() > max_depth || steps.size() > kMaxTraceSteps ||
      winning_tokens.size() != winning_edges.size() || !std::isfinite(winning_score)) {
    return Status(ErrorCode::invalid_argument, "replay trace header is invalid");
  }
  if (lane != Lane::full && lane != Lane::clean) return Status(ErrorCode::invalid_argument, "replay trace lane is invalid");
  const std::size_t theoretical_max = max_depth * beam_width * candidate_k;
  if (candidate_record_count() > theoretical_max) {
    return Status(ErrorCode::invalid_argument, "replay trace exceeds O(D*B*k) bound");
  }
  for (std::size_t depth = 0; depth < steps.size(); ++depth) {
    const TraceStep& step = steps[depth];
    if (step.depth != depth || step.parents.size() > beam_width) {
      return Status(ErrorCode::invalid_argument, "replay trace step bound is invalid");
    }
    for (const TraceParent& parent : step.parents) {
      if (parent.token == kInvalidToken || !std::isfinite(parent.parent_score) ||
          parent.candidates.size() > candidate_k || !std::isfinite(parent.certificate.beam_cutoff) ||
          !std::isfinite(parent.certificate.omitted_upper_bound)) {
        return Status(ErrorCode::invalid_argument, "replay trace parent is invalid");
      }
      for (const TraceCandidate& candidate : parent.candidates) {
        if (candidate.edge_id == 0 || candidate.target == kInvalidToken ||
            !std::isfinite(candidate.transition_score) ||
            (candidate.tier != EvidenceTier::m1 && candidate.tier != EvidenceTier::m2)) {
          return Status(ErrorCode::invalid_argument, "replay trace candidate is invalid");
        }
        if (lane == Lane::clean && candidate.tier == EvidenceTier::m1) {
          return Status(ErrorCode::invalid_argument, "CLEAN replay trace contains M1 evidence");
        }
      }
    }
  }
  return Status::Ok();
}

std::vector<std::byte> serialize_replay_trace(const ReplayTrace& trace) {
  ByteWriter writer;
  writer.u64(trace.id);
  writer.u64(trace.created_at_ms);
  writer.u64(trace.repository_epoch);
  writer.u64(trace.embedding_checksum);
  writer.u8(static_cast<std::uint8_t>(trace.lane));
  writer.u8(trace.exhaustive ? 1U : 0U); writer.u16(0);
  writer.u64(trace.edge_under_test);
  writer.u32(trace.expected_target);
  writer.u32(static_cast<std::uint32_t>(trace.beam_width));
  writer.u32(static_cast<std::uint32_t>(trace.candidate_k));
  writer.u32(static_cast<std::uint32_t>(trace.max_depth));
  writer.f32(trace.winning_score);
  writer.u32(static_cast<std::uint32_t>(trace.input_tokens.size()));
  for (TokenId token : trace.input_tokens) writer.u32(token);
  writer.u32(static_cast<std::uint32_t>(trace.steps.size()));
  for (const TraceStep& step : trace.steps) {
    writer.u32(static_cast<std::uint32_t>(step.depth));
    writer.u32(static_cast<std::uint32_t>(step.parents.size()));
    for (const TraceParent& parent : step.parents) {
      writer.u32(parent.token);
      writer.u64(parent.state_fingerprint);
      writer.f32(parent.parent_score);
      writer.u8(parent.certificate.truncated ? 1U : 0U);
      writer.u8(parent.certificate.certified_safe ? 1U : 0U); writer.u16(0);
      writer.u32(static_cast<std::uint32_t>(parent.certificate.enumerated));
      writer.f32(parent.certificate.beam_cutoff);
      writer.f32(parent.certificate.omitted_upper_bound);
      writer.u32(static_cast<std::uint32_t>(parent.candidates.size()));
      for (const TraceCandidate& candidate : parent.candidates) {
        writer.u64(candidate.edge_id);
        writer.u32(candidate.target);
        writer.u8(static_cast<std::uint8_t>(candidate.tier));
        writer.u8(0); writer.u16(0);
        writer.f32(candidate.transition_score);
      }
    }
  }
  writer.u32(static_cast<std::uint32_t>(trace.winning_edges.size()));
  for (std::size_t i = 0; i < trace.winning_edges.size(); ++i) {
    writer.u64(trace.winning_edges[i]);
    writer.u32(trace.winning_tokens[i]);
  }
  return writer.take();
}

Result<ReplayTrace> deserialize_replay_trace(std::span<const std::byte> payload) {
  ByteReader reader(payload);
  ReplayTrace trace;
  auto id = reader.u64(); auto created = reader.u64(); auto epoch = reader.u64(); auto checksum = reader.u64();
  auto lane = reader.u8(); auto exhaustive = reader.u8(); auto reserved = reader.u16();
  auto edge = reader.u64(); auto expected = reader.u32();
  auto beam = reader.u32(); auto k = reader.u32(); auto depth_limit = reader.u32(); auto winning_score = reader.f32();
  auto input_count = reader.u32();
  (void)reserved;
  if (!id || !created || !epoch || !checksum || !lane || !exhaustive || !edge || !expected || !beam || !k ||
      !depth_limit || !winning_score || !input_count || lane.value() > 1 || input_count.value() > kMaxTraceTokens) {
    return Status(ErrorCode::data_loss, "truncated or invalid replay trace header");
  }
  trace.id = id.value(); trace.created_at_ms = created.value(); trace.repository_epoch = epoch.value();
  trace.embedding_checksum = checksum.value(); trace.lane = static_cast<Lane>(lane.value());
  trace.exhaustive = exhaustive.value() != 0; trace.edge_under_test = edge.value();
  trace.expected_target = expected.value(); trace.beam_width = beam.value(); trace.candidate_k = k.value();
  trace.max_depth = depth_limit.value(); trace.winning_score = winning_score.value();
  trace.input_tokens.reserve(input_count.value());
  for (std::uint32_t i = 0; i < input_count.value(); ++i) {
    auto token = reader.u32(); if (!token) return token.status(); trace.input_tokens.push_back(token.value());
  }
  auto step_count = reader.u32();
  if (!step_count || step_count.value() > kMaxTraceSteps) return Status(ErrorCode::data_loss, "invalid replay step count");
  trace.steps.reserve(step_count.value());
  for (std::uint32_t step_index = 0; step_index < step_count.value(); ++step_index) {
    auto depth = reader.u32(); auto parent_count = reader.u32();
    if (!depth || !parent_count || parent_count.value() > kMaxParentsPerStep) return Status(ErrorCode::data_loss, "invalid replay parent count");
    TraceStep step; step.depth = depth.value(); step.parents.reserve(parent_count.value());
    for (std::uint32_t parent_index = 0; parent_index < parent_count.value(); ++parent_index) {
      TraceParent parent;
      auto token = reader.u32(); auto fingerprint = reader.u64(); auto parent_score = reader.f32();
      auto truncated = reader.u8(); auto certified = reader.u8(); auto reserved_parent = reader.u16();
      auto enumerated = reader.u32(); auto cutoff = reader.f32(); auto upper = reader.f32(); auto candidate_count = reader.u32();
      (void)reserved_parent;
      if (!token || !fingerprint || !parent_score || !truncated || !certified || !enumerated || !cutoff || !upper ||
          !candidate_count || candidate_count.value() > kMaxCandidatesPerParent) {
        return Status(ErrorCode::data_loss, "invalid replay parent");
      }
      parent.token = token.value(); parent.state_fingerprint = fingerprint.value(); parent.parent_score = parent_score.value();
      parent.certificate = PruningCertificate{truncated.value() != 0, certified.value() != 0,
                                              enumerated.value(), cutoff.value(), upper.value()};
      parent.candidates.reserve(candidate_count.value());
      for (std::uint32_t candidate_index = 0; candidate_index < candidate_count.value(); ++candidate_index) {
        auto edge_id = reader.u64(); auto target = reader.u32(); auto tier = reader.u8();
        auto reserved_candidate8 = reader.u8(); auto reserved_candidate16 = reader.u16(); auto score = reader.f32();
        (void)reserved_candidate8; (void)reserved_candidate16;
        if (!edge_id || !target || !tier || !score || !valid_tier(tier.value())) {
          return Status(ErrorCode::data_loss, "invalid replay candidate");
        }
        parent.candidates.push_back(TraceCandidate{edge_id.value(), target.value(),
                                                    static_cast<EvidenceTier>(tier.value()), score.value()});
      }
      step.parents.push_back(std::move(parent));
    }
    trace.steps.push_back(std::move(step));
  }
  auto winning_count = reader.u32();
  if (!winning_count || winning_count.value() > trace.max_depth) return Status(ErrorCode::data_loss, "invalid replay winning path count");
  trace.winning_edges.reserve(winning_count.value()); trace.winning_tokens.reserve(winning_count.value());
  for (std::uint32_t i = 0; i < winning_count.value(); ++i) {
    auto winning_edge = reader.u64(); auto winning_token = reader.u32();
    if (!winning_edge || !winning_token) return Status(ErrorCode::data_loss, "truncated replay winning path");
    trace.winning_edges.push_back(winning_edge.value()); trace.winning_tokens.push_back(winning_token.value());
  }
  if (reader.remaining() != 0) return Status(ErrorCode::data_loss, "trailing bytes in replay trace");
  const Status status = trace.validate();
  if (!status) return Status(ErrorCode::data_loss, status.message());
  return trace;
}

Status ReplayStore::open(const std::filesystem::path& root,
                         const ReplayConfig& config,
                         Durability durability) {
  if (max_records_ != 0) return Status(ErrorCode::failed_precondition, "replay store already open");
  if (config.max_records == 0) return Status(ErrorCode::invalid_argument, "replay max_records cannot be zero");
  RLM_RETURN_IF_ERROR(ensure_directory(root));
  max_records_ = config.max_records;
  RLM_RETURN_IF_ERROR(wal_.open(root / "replay.wal", durability));
  RLM_RETURN_IF_ERROR(wal_.replay([this](const WalRecord& record) { return apply_record(record); }, true));
  while (traces_.size() > max_records_) {
    const TraceId id = insertion_order_.front(); insertion_order_.pop_front(); traces_.erase(id);
  }
  return Status::Ok();
}

Result<TraceId> ReplayStore::put(ReplayTrace trace) {
  if (max_records_ == 0) return Status(ErrorCode::failed_precondition, "replay store is not open");
  if (trace.id == 0) {
    const std::uint64_t counter = id_counter_.fetch_add(1, std::memory_order_relaxed);
    trace.id = hash_combine64(hash_combine64(trace.created_at_ms, trace.repository_epoch), counter);
    if (trace.id == 0) trace.id = counter;
  }
  const Status validation = trace.validate();
  if (!validation) return validation;
  const std::vector<std::byte> payload = serialize_replay_trace(trace);
  auto appended = wal_.append(kPut, payload);
  if (!appended) return appended.status();
  std::lock_guard lock(mutex_);
  const TraceId id = trace.id;
  index_locked(std::move(trace));
  while (traces_.size() > max_records_) {
    const TraceId oldest = insertion_order_.front();
    const Status status = erase_locked(oldest, true);
    if (!status) return status;
  }
  return id;
}

Result<ReplayTrace> ReplayStore::get(TraceId id) const {
  std::lock_guard lock(mutex_);
  const auto found = traces_.find(id);
  if (found == traces_.end()) return Status(ErrorCode::not_found, "replay trace not found");
  return found->second;
}

Status ReplayStore::flush() { return wal_.flush(); }
std::size_t ReplayStore::size() const { std::lock_guard lock(mutex_); return traces_.size(); }

Status ReplayStore::apply_record(const WalRecord& record) {
  std::lock_guard lock(mutex_);
  if (record.kind == kPut) {
    auto trace = deserialize_replay_trace(record.payload);
    if (!trace) return trace.status();
    id_counter_.store(std::max(id_counter_.load(std::memory_order_relaxed), trace.value().id + 1), std::memory_order_relaxed);
    index_locked(std::move(trace).value());
    return Status::Ok();
  }
  if (record.kind == kDelete) {
    ByteReader reader(record.payload);
    auto id = reader.u64();
    if (!id || reader.remaining() != 0) return Status(ErrorCode::data_loss, "invalid replay delete record");
    traces_.erase(id.value());
    insertion_order_.erase(std::remove(insertion_order_.begin(), insertion_order_.end(), id.value()), insertion_order_.end());
    return Status::Ok();
  }
  return Status(ErrorCode::data_loss, "unknown replay WAL record kind");
}

Status ReplayStore::erase_locked(TraceId id, bool persist) {
  if (persist) {
    ByteWriter writer; writer.u64(id);
    auto appended = wal_.append(kDelete, writer.data());
    if (!appended) return appended.status();
  }
  traces_.erase(id);
  insertion_order_.erase(std::remove(insertion_order_.begin(), insertion_order_.end(), id), insertion_order_.end());
  return Status::Ok();
}

void ReplayStore::index_locked(ReplayTrace trace) {
  insertion_order_.erase(std::remove(insertion_order_.begin(), insertion_order_.end(), trace.id), insertion_order_.end());
  insertion_order_.push_back(trace.id);
  traces_.insert_or_assign(trace.id, std::move(trace));
}

}  // namespace rlm
