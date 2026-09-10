#include "rlm/relation_store.hpp"

#include "rlm/binary.hpp"
#include "rlm/wal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace rlm {
namespace {
constexpr std::uint16_t kWalUpsert = 1;
constexpr std::uint16_t kWalErase = 2;
constexpr std::uint16_t kWalBatch = 3;
constexpr std::array<std::byte, 8> kSnapshotMagic{
    std::byte{'R'}, std::byte{'L'}, std::byte{'S'}, std::byte{'N'},
    std::byte{'A'}, std::byte{'P'}, std::byte{'1'}, std::byte{0}};
constexpr std::uint32_t kSnapshotVersion = 1;
constexpr std::size_t kMaxEvidence = 256;
constexpr std::size_t kMaxBatchObservations = 2'000'000;

bool edge_order(const RelationEdge& a, const RelationEdge& b) {
  if (a.confidence != b.confidence) return a.confidence > b.confidence;
  if (a.support != b.support) return a.support > b.support;
  return a.id < b.id;
}

bool equivalent(const RelationEdge& a, const RelationEdge& b) {
  return a.id == b.id && a.source == b.source && a.target == b.target && a.tier == b.tier &&
         a.relation.scale == b.relation.scale && a.relation.values == b.relation.values &&
         a.confidence == b.confidence && a.support == b.support &&
         a.created_at_ms == b.created_at_ms && a.last_seen_ms == b.last_seen_ms &&
         a.evidence == b.evidence;
}

std::vector<std::byte> serialize_batch(BatchId batch_id,
                                       std::span<const RelationObservation> observations) {
  ByteWriter writer;
  writer.u64(batch_id);
  writer.u32(static_cast<std::uint32_t>(observations.size()));
  for (const RelationObservation& observation : observations) {
    writer.u32(observation.source);
    writer.u32(observation.target);
    writer.qvector(observation.relation);
    writer.f32(observation.confidence);
    writer.u64(observation.count);
    writer.u64(observation.observed_at_ms);
  }
  return writer.take();
}

Result<std::pair<BatchId, std::vector<RelationObservation>>> deserialize_batch(
    std::span<const std::byte> payload,
    std::size_t dimension) {
  ByteReader reader(payload);
  auto batch_id = reader.u64();
  auto count = reader.u32();
  if (!batch_id || !count) return Status(ErrorCode::data_loss, "truncated observation batch");
  if (count.value() > kMaxBatchObservations) return Status(ErrorCode::resource_exhausted, "observation batch exceeds safety limit");
  std::vector<RelationObservation> observations;
  observations.reserve(count.value());
  for (std::uint32_t i = 0; i < count.value(); ++i) {
    auto source = reader.u32();
    auto target = reader.u32();
    auto relation = reader.qvector(dimension, dimension);
    auto confidence = reader.f32();
    auto support = reader.u64();
    auto observed = reader.u64();
    if (!source || !target || !relation || !confidence || !support || !observed) {
      return Status(ErrorCode::data_loss, "truncated observation in batch");
    }
    if (source.value() == kInvalidToken || target.value() == kInvalidToken ||
        confidence.value() < 0.0F || confidence.value() > 1.0F || support.value() == 0) {
      return Status(ErrorCode::data_loss, "invalid observation in batch");
    }
    observations.push_back(RelationObservation{source.value(), target.value(), std::move(relation).value(),
                                                confidence.value(), support.value(), observed.value()});
  }
  if (reader.remaining() != 0) return Status(ErrorCode::data_loss, "trailing bytes in observation batch");
  return std::make_pair(batch_id.value(), std::move(observations));
}

class TierStore final {
 public:
  TierStore(EvidenceTier tier, std::size_t shards, std::size_t dimension, Durability durability)
      : tier_(tier), dimension_(dimension), durability_(durability) {
    shards_.reserve(shards);
    for (std::size_t i = 0; i < shards; ++i) shards_.push_back(std::make_unique<Shard>());
  }

  Status open(const std::filesystem::path& root) {
    root_ = root;
    RLM_RETURN_IF_ERROR(ensure_directory(root_));
    RLM_RETURN_IF_ERROR(load_snapshot());
    RLM_RETURN_IF_ERROR(wal_.open(root_ / "changes.wal", durability_));
    RLM_RETURN_IF_ERROR(wal_.replay([this](const WalRecord& record) { return apply_wal(record); }, true));
    return Status::Ok();
  }

  Result<CandidatePage> outgoing(TokenId source, std::size_t limit,
                                 std::optional<EdgeId> exclude) const {
    if (limit == 0) return Status(ErrorCode::invalid_argument, "candidate limit must be positive");
    const Shard& shard = *shards_[shard_index(source)];
    std::vector<RelationEdge> all;
    {
      std::shared_lock lock(shard.mutex);
      const auto found = shard.outgoing.find(source);
      if (found != shard.outgoing.end()) {
        all.reserve(found->second.size());
        for (const auto& [id, edge] : found->second) {
          if (!exclude.has_value() || id != *exclude) all.push_back(edge);
        }
      }
    }
    std::sort(all.begin(), all.end(), edge_order);
    CandidatePage page;
    if (all.size() > limit) {
      page.has_more = true;
      page.next_confidence = all[limit].confidence;
      all.resize(limit);
    }
    page.edges = std::move(all);
    return page;
  }

  Result<RelationEdge> get(EdgeId id) const {
    TokenId source = kInvalidToken;
    {
      std::shared_lock lock(index_mutex_);
      const auto found = source_by_id_.find(id);
      if (found == source_by_id_.end()) return Status(ErrorCode::not_found, "relation edge not found");
      source = found->second;
    }
    const Shard& shard = *shards_[shard_index(source)];
    std::shared_lock lock(shard.mutex);
    const auto by_source = shard.outgoing.find(source);
    if (by_source == shard.outgoing.end()) return Status(ErrorCode::not_found, "relation edge index is stale");
    const auto found = by_source->second.find(id);
    if (found == by_source->second.end()) return Status(ErrorCode::not_found, "relation edge index is stale");
    return found->second;
  }

  bool contains(EdgeId id) const {
    std::shared_lock lock(index_mutex_);
    return source_by_id_.contains(id);
  }

  Result<bool> apply_batch(BatchId batch_id, std::span<const RelationObservation> observations) {
    if (tier_ != EvidenceTier::m1) return Status(ErrorCode::failed_precondition, "observation batches are M1-only");
    if (batch_id == 0) return Status(ErrorCode::invalid_argument, "batch id cannot be zero");
    if (observations.size() > kMaxBatchObservations) return Status(ErrorCode::resource_exhausted, "observation batch is too large");
    for (const auto& observation : observations) {
      const Status status = validate_observation(observation);
      if (!status) return status;
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    {
      std::shared_lock batch_lock(batch_mutex_);
      if (applied_batches_.contains(batch_id)) return false;
    }
    const std::vector<std::byte> payload = serialize_batch(batch_id, observations);
    auto appended = wal_.append(kWalBatch, payload);
    if (!appended) return appended.status();
    bool changed = false;
    for (const RelationObservation& observation : observations) changed = apply_observation_memory(observation) || changed;
    {
      std::unique_lock batch_lock(batch_mutex_);
      applied_batches_.insert(batch_id);
    }
    if (changed || !observations.empty()) epoch_.fetch_add(1, std::memory_order_release);
    return true;
  }

  Status upsert(RelationEdge edge) {
    edge.tier = tier_;
    const Status validation = edge.validate(dimension_);
    if (!validation) return validation;
    std::lock_guard mutation_lock(mutation_mutex_);
    const std::vector<std::byte> payload = serialize_relation_edge(edge);
    auto appended = wal_.append(kWalUpsert, payload);
    if (!appended) return appended.status();
    if (apply_upsert_memory(std::move(edge))) epoch_.fetch_add(1, std::memory_order_release);
    return Status::Ok();
  }

  Status erase(EdgeId id) {
    std::lock_guard mutation_lock(mutation_mutex_);
    ByteWriter writer;
    writer.u64(id);
    auto appended = wal_.append(kWalErase, writer.data());
    if (!appended) return appended.status();
    if (apply_erase_memory(id)) epoch_.fetch_add(1, std::memory_order_release);
    return Status::Ok();
  }

  std::vector<RelationEdge> expired_before(std::uint64_t cutoff_ms, std::size_t limit) const {
    std::vector<RelationEdge> output;
    output.reserve(std::min(limit, count()));
    for (const auto& shard_ptr : shards_) {
      const Shard& shard = *shard_ptr;
      std::shared_lock lock(shard.mutex);
      for (const auto& [source, edges] : shard.outgoing) {
        (void)source;
        for (const auto& [id, edge] : edges) {
          (void)id;
          if (edge.last_seen_ms < cutoff_ms) {
            output.push_back(edge);
            if (output.size() >= limit) return output;
          }
        }
      }
    }
    return output;
  }

  std::size_t count() const noexcept { return count_.load(std::memory_order_acquire); }
  Epoch epoch() const noexcept { return epoch_.load(std::memory_order_acquire); }
  Status flush() { return wal_.flush(); }

  Status checkpoint() {
    std::lock_guard mutation_lock(mutation_mutex_);
    std::vector<RelationEdge> edges = all_edges();
    std::sort(edges.begin(), edges.end(), [](const RelationEdge& a, const RelationEdge& b) { return a.id < b.id; });
    std::vector<BatchId> batches;
    {
      std::shared_lock lock(batch_mutex_);
      batches.assign(applied_batches_.begin(), applied_batches_.end());
    }
    std::sort(batches.begin(), batches.end());
    ByteWriter writer;
    writer.bytes(kSnapshotMagic);
    writer.u32(kSnapshotVersion);
    writer.u8(static_cast<std::uint8_t>(tier_));
    writer.u8(0); writer.u16(0);
    writer.u32(static_cast<std::uint32_t>(dimension_));
    writer.u64(epoch());
    writer.u64(static_cast<std::uint64_t>(batches.size()));
    writer.u64(static_cast<std::uint64_t>(edges.size()));
    for (BatchId id : batches) writer.u64(id);
    for (const RelationEdge& edge : edges) {
      const auto payload = serialize_relation_edge(edge);
      writer.u32(static_cast<std::uint32_t>(payload.size()));
      writer.bytes(payload);
    }
    writer.u32(crc32(writer.data()));
    RLM_RETURN_IF_ERROR(write_file_atomic(root_ / "snapshot.bin", writer.data(), durability_));
    return wal_.reset();
  }

 private:
  struct Shard final {
    mutable std::shared_mutex mutex;
    std::unordered_map<TokenId, std::unordered_map<EdgeId, RelationEdge>> outgoing;
  };

  std::size_t shard_index(TokenId source) const noexcept {
    return static_cast<std::size_t>(source) % shards_.size();
  }

  Status validate_observation(const RelationObservation& observation) const {
    if (observation.source == kInvalidToken || observation.target == kInvalidToken ||
        observation.count == 0 ||
        !std::isfinite(observation.confidence) || observation.confidence < 0.0F || observation.confidence > 1.0F) {
      return Status(ErrorCode::invalid_argument, "invalid relation observation");
    }
    return observation.relation.validate(dimension_);
  }

  bool apply_observation_memory(const RelationObservation& observation) {
    const EdgeId id = deterministic_edge_id(observation.source, observation.target);
    Shard& shard = *shards_[shard_index(observation.source)];
    std::unique_lock shard_lock(shard.mutex);
    auto& by_id = shard.outgoing[observation.source];
    auto found = by_id.find(id);
    if (found == by_id.end()) {
      RelationEdge edge;
      edge.id = id;
      edge.source = observation.source;
      edge.target = observation.target;
      edge.tier = tier_;
      edge.relation = observation.relation;
      edge.confidence = observation.confidence;
      edge.support = observation.count;
      edge.created_at_ms = observation.observed_at_ms;
      edge.last_seen_ms = observation.observed_at_ms;
      by_id.emplace(id, edge);
      {
        std::unique_lock index_lock(index_mutex_);
        source_by_id_[id] = observation.source;
      }
      count_.fetch_add(1, std::memory_order_release);
      return true;
    }
    RelationEdge& edge = found->second;
    const std::uint64_t old_support = edge.support;
    const std::uint64_t max_add = std::numeric_limits<std::uint64_t>::max() - old_support;
    const std::uint64_t add = std::min(observation.count, max_add);
    const std::uint64_t new_support = old_support + add;
    const double denominator = static_cast<double>(new_support);
    if (denominator > 0.0) {
      edge.confidence = static_cast<float>((static_cast<double>(edge.confidence) * old_support +
                                            static_cast<double>(observation.confidence) * add) / denominator);
      const auto old_vector = edge.relation.dequantize();
      const auto incoming_vector = observation.relation.dequantize();
      std::vector<float> merged(dimension_);
      for (std::size_t i = 0; i < dimension_; ++i) {
        merged[i] = static_cast<float>((static_cast<double>(old_vector[i]) * old_support +
                                        static_cast<double>(incoming_vector[i]) * add) / denominator);
      }
      auto quantized = QuantizedVector::quantize(normalized(merged));
      if (quantized) edge.relation = std::move(quantized).value();
    }
    edge.support = new_support;
    edge.last_seen_ms = std::max(edge.last_seen_ms, observation.observed_at_ms);
    return add != 0;
  }

  bool apply_upsert_memory(RelationEdge edge) {
    Shard& shard = *shards_[shard_index(edge.source)];
    std::unique_lock shard_lock(shard.mutex);
    auto& by_id = shard.outgoing[edge.source];
    const auto found = by_id.find(edge.id);
    if (found != by_id.end() && equivalent(found->second, edge)) return false;
    const bool inserted = found == by_id.end();
    by_id.insert_or_assign(edge.id, edge);
    {
      std::unique_lock index_lock(index_mutex_);
      source_by_id_[edge.id] = edge.source;
    }
    if (inserted) count_.fetch_add(1, std::memory_order_release);
    return true;
  }

  bool apply_erase_memory(EdgeId id) {
    TokenId source = kInvalidToken;
    {
      std::shared_lock index_lock(index_mutex_);
      const auto found = source_by_id_.find(id);
      if (found == source_by_id_.end()) return false;
      source = found->second;
    }
    Shard& shard = *shards_[shard_index(source)];
    std::unique_lock shard_lock(shard.mutex);
    auto by_source = shard.outgoing.find(source);
    if (by_source == shard.outgoing.end() || by_source->second.erase(id) == 0) return false;
    if (by_source->second.empty()) shard.outgoing.erase(by_source);
    {
      std::unique_lock index_lock(index_mutex_);
      source_by_id_.erase(id);
    }
    count_.fetch_sub(1, std::memory_order_release);
    return true;
  }

  std::vector<RelationEdge> all_edges() const {
    std::vector<RelationEdge> output;
    output.reserve(count());
    for (const auto& shard_ptr : shards_) {
      const Shard& shard = *shard_ptr;
      std::shared_lock lock(shard.mutex);
      for (const auto& [source, edges] : shard.outgoing) {
        (void)source;
        for (const auto& [id, edge] : edges) { (void)id; output.push_back(edge); }
      }
    }
    return output;
  }

  Status apply_wal(const WalRecord& record) {
    if (record.kind == kWalUpsert) {
      auto edge = deserialize_relation_edge(record.payload, dimension_);
      if (!edge) return edge.status();
      if (edge.value().tier != tier_) return Status(ErrorCode::data_loss, "WAL edge tier mismatch");
      if (apply_upsert_memory(std::move(edge).value())) epoch_.fetch_add(1, std::memory_order_release);
      return Status::Ok();
    }
    if (record.kind == kWalErase) {
      ByteReader reader(record.payload);
      auto id = reader.u64();
      if (!id || reader.remaining() != 0) return Status(ErrorCode::data_loss, "invalid erase WAL record");
      if (apply_erase_memory(id.value())) epoch_.fetch_add(1, std::memory_order_release);
      return Status::Ok();
    }
    if (record.kind == kWalBatch) {
      if (tier_ != EvidenceTier::m1) return Status(ErrorCode::data_loss, "batch record in M2 WAL");
      auto batch = deserialize_batch(record.payload, dimension_);
      if (!batch) return batch.status();
      {
        std::shared_lock lock(batch_mutex_);
        if (applied_batches_.contains(batch.value().first)) return Status::Ok();
      }
      bool changed = false;
      for (const auto& observation : batch.value().second) changed = apply_observation_memory(observation) || changed;
      {
        std::unique_lock lock(batch_mutex_);
        applied_batches_.insert(batch.value().first);
      }
      if (changed || !batch.value().second.empty()) epoch_.fetch_add(1, std::memory_order_release);
      return Status::Ok();
    }
    return Status(ErrorCode::data_loss, "unknown relation WAL record kind");
  }

  Status load_snapshot() {
    const std::filesystem::path path = root_ / "snapshot.bin";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return Status::Ok();
    if (ec) return Status(ErrorCode::io_error, "cannot stat relation snapshot: " + ec.message());
    auto file = read_file(path);
    if (!file) return file.status();
    if (file.value().size() < kSnapshotMagic.size() + 4) return Status(ErrorCode::data_loss, "relation snapshot is too small");
    const auto bytes = std::span<const std::byte>(file.value());
    ByteReader footer(bytes.last(4));
    auto expected_crc = footer.u32();
    if (!expected_crc || crc32(bytes.first(bytes.size() - 4)) != expected_crc.value()) {
      return Status(ErrorCode::data_loss, "relation snapshot CRC mismatch");
    }
    ByteReader reader(bytes.first(bytes.size() - 4));
    auto magic = reader.bytes(kSnapshotMagic.size());
    auto version = reader.u32();
    auto tier = reader.u8();
    auto reserved8 = reader.u8();
    auto reserved16 = reader.u16();
    auto dimension = reader.u32();
    auto snapshot_epoch = reader.u64();
    auto batch_count = reader.u64();
    auto edge_count = reader.u64();
    (void)reserved8; (void)reserved16;
    if (!magic || !version || !tier || !dimension || !snapshot_epoch || !batch_count || !edge_count ||
        !std::equal(kSnapshotMagic.begin(), kSnapshotMagic.end(), magic.value().begin()) ||
        version.value() != kSnapshotVersion || tier.value() != static_cast<std::uint8_t>(tier_) ||
        dimension.value() != dimension_) {
      return Status(ErrorCode::data_loss, "relation snapshot header mismatch");
    }
    if (batch_count.value() > 100'000'000ULL || edge_count.value() > 1'000'000'000ULL) {
      return Status(ErrorCode::resource_exhausted, "relation snapshot count exceeds safety limit");
    }
    {
      std::unique_lock lock(batch_mutex_);
      for (std::uint64_t i = 0; i < batch_count.value(); ++i) {
        auto id = reader.u64();
        if (!id) return Status(ErrorCode::data_loss, "truncated snapshot batch index");
        applied_batches_.insert(id.value());
      }
    }
    for (std::uint64_t i = 0; i < edge_count.value(); ++i) {
      auto size = reader.u32();
      if (!size || size.value() > 256U * 1024U * 1024U) return Status(ErrorCode::data_loss, "invalid edge size in snapshot");
      auto payload = reader.bytes(size.value());
      if (!payload) return payload.status();
      auto edge = deserialize_relation_edge(payload.value(), dimension_);
      if (!edge || edge.value().tier != tier_) return edge ? Status(ErrorCode::data_loss, "snapshot edge tier mismatch") : edge.status();
      apply_upsert_memory(std::move(edge).value());
    }
    if (reader.remaining() != 0) return Status(ErrorCode::data_loss, "trailing bytes in relation snapshot");
    epoch_.store(snapshot_epoch.value(), std::memory_order_release);
    return Status::Ok();
  }

  EvidenceTier tier_;
  std::size_t dimension_;
  Durability durability_;
  std::filesystem::path root_;
  std::vector<std::unique_ptr<Shard>> shards_;
  mutable std::shared_mutex index_mutex_;
  std::unordered_map<EdgeId, TokenId> source_by_id_;
  mutable std::shared_mutex batch_mutex_;
  std::unordered_set<BatchId> applied_batches_;
  mutable std::mutex mutation_mutex_;
  WriteAheadLog wal_;
  std::atomic<std::size_t> count_{0};
  std::atomic<Epoch> epoch_{0};
};

class PinRegistry final {
 public:
  bool pin(EdgeId id) {
    std::lock_guard lock(mutex_);
    State& state = states_[id];
    if (state.expiring) return false;
    ++state.pins;
    return true;
  }
  void unpin(EdgeId id) noexcept {
    std::lock_guard lock(mutex_);
    const auto found = states_.find(id);
    if (found == states_.end()) return;
    if (found->second.pins > 0) --found->second.pins;
    if (found->second.pins == 0 && !found->second.expiring) states_.erase(found);
  }
  bool mark_expiring(EdgeId id) {
    std::lock_guard lock(mutex_);
    State& state = states_[id];
    if (state.pins != 0 || state.expiring) return false;
    state.expiring = true;
    return true;
  }
  void finish_expiring(EdgeId id) noexcept {
    std::lock_guard lock(mutex_);
    states_.erase(id);
  }
 private:
  struct State { std::size_t pins{0}; bool expiring{false}; };
  std::mutex mutex_;
  std::unordered_map<EdgeId, State> states_;
};

}  // namespace

Status RelationEdge::validate(std::size_t dimension) const {
  if (id == 0 || source == kInvalidToken || target == kInvalidToken ||
      id != deterministic_edge_id(source, target)) {
    return Status(ErrorCode::invalid_argument, "relation edge identity is invalid");
  }
  if (tier != EvidenceTier::m1 && tier != EvidenceTier::m2) return Status(ErrorCode::invalid_argument, "relation edge tier is invalid");
  if (!std::isfinite(confidence) || confidence < 0.0F || confidence > 1.0F || support == 0 ||
      created_at_ms == 0 || last_seen_ms < created_at_ms || evidence.size() > kMaxEvidence) {
    return Status(ErrorCode::invalid_argument, "relation edge metadata is invalid");
  }
  return relation.validate(dimension);
}

std::vector<std::byte> serialize_relation_edge(const RelationEdge& edge) {
  ByteWriter writer;
  writer.u64(edge.id);
  writer.u32(edge.source);
  writer.u32(edge.target);
  writer.u8(static_cast<std::uint8_t>(edge.tier));
  writer.u8(0); writer.u16(0);
  writer.qvector(edge.relation);
  writer.f32(edge.confidence);
  writer.u64(edge.support);
  writer.u64(edge.created_at_ms);
  writer.u64(edge.last_seen_ms);
  writer.u32(static_cast<std::uint32_t>(edge.evidence.size()));
  for (TraceId id : edge.evidence) writer.u64(id);
  return writer.take();
}

Result<RelationEdge> deserialize_relation_edge(std::span<const std::byte> payload,
                                               std::size_t expected_dimension) {
  ByteReader reader(payload);
  auto id = reader.u64();
  auto source = reader.u32();
  auto target = reader.u32();
  auto tier = reader.u8();
  auto reserved8 = reader.u8();
  auto reserved16 = reader.u16();
  auto relation = expected_dimension == 0
      ? reader.qvector(0, 1U << 20U)
      : reader.qvector(expected_dimension, expected_dimension);
  auto confidence = reader.f32();
  auto support = reader.u64();
  auto created = reader.u64();
  auto seen = reader.u64();
  auto evidence_count = reader.u32();
  (void)reserved8; (void)reserved16;
  if (!id || !source || !target || !tier || !relation || !confidence || !support || !created || !seen || !evidence_count) {
    return Status(ErrorCode::data_loss, "truncated relation edge");
  }
  if (evidence_count.value() > kMaxEvidence) return Status(ErrorCode::data_loss, "relation evidence count exceeds limit");
  RelationEdge edge;
  edge.id = id.value(); edge.source = source.value(); edge.target = target.value();
  edge.tier = static_cast<EvidenceTier>(tier.value()); edge.relation = std::move(relation).value();
  edge.confidence = confidence.value(); edge.support = support.value();
  edge.created_at_ms = created.value(); edge.last_seen_ms = seen.value();
  edge.evidence.reserve(evidence_count.value());
  for (std::uint32_t i = 0; i < evidence_count.value(); ++i) {
    auto trace = reader.u64();
    if (!trace) return trace.status();
    edge.evidence.push_back(trace.value());
  }
  if (reader.remaining() != 0) return Status(ErrorCode::data_loss, "trailing bytes in relation edge");
  const Status status = edge.validate(expected_dimension == 0 ? edge.relation.size() : expected_dimension);
  if (!status) return Status(ErrorCode::data_loss, status.message());
  return edge;
}

struct RelationRepository::Impl final {
  std::unique_ptr<TierStore> m1;
  std::unique_ptr<TierStore> m2;
  PinRegistry pins;
  ProcessFileLock process_lock;
  std::size_t dimension{0};
  std::atomic<Epoch> mutation_epoch{0};
};

RelationRepository::RelationRepository() : impl_(std::make_unique<Impl>()) {}
RelationRepository::~RelationRepository() = default;

Status RelationRepository::open(const StorageConfig& config, std::size_t embedding_dimension) {
  if (impl_->m1 || impl_->m2) return Status(ErrorCode::failed_precondition, "relation repository already open");
  if (embedding_dimension == 0) return Status(ErrorCode::invalid_argument, "embedding dimension is zero");
  RLM_RETURN_IF_ERROR(ensure_directory(config.state_dir));
  RLM_RETURN_IF_ERROR(impl_->process_lock.acquire(config.state_dir / "ENGINE.lock"));
  impl_->dimension = embedding_dimension;
  impl_->m1 = std::make_unique<TierStore>(EvidenceTier::m1, config.shards, embedding_dimension, config.durability);
  impl_->m2 = std::make_unique<TierStore>(EvidenceTier::m2, config.shards, embedding_dimension, config.durability);
  Status status = impl_->m2->open(config.state_dir / "m2");
  if (status) status = impl_->m1->open(config.state_dir / "m1");
  if (!status) {
    impl_->m1.reset(); impl_->m2.reset(); impl_->process_lock.release();
    return status;
  }
  impl_->mutation_epoch.store(std::max(impl_->m1->epoch(), impl_->m2->epoch()), std::memory_order_release);
  return Status::Ok();
}

Result<CandidatePage> RelationRepository::outgoing(TokenId source, Lane lane, std::size_t limit,
                                                   std::optional<EdgeId> exclude) const {
  if (!impl_->m1 || !impl_->m2) return Status(ErrorCode::failed_precondition, "relation repository is not open");
  if (lane == Lane::clean) return impl_->m2->outgoing(source, limit, exclude);
  auto m2 = impl_->m2->outgoing(source, limit + 1, exclude);
  auto m1 = impl_->m1->outgoing(source, limit + 1, exclude);
  if (!m2) return m2.status();
  if (!m1) return m1.status();
  std::vector<RelationEdge> merged;
  merged.reserve(m2.value().edges.size() + m1.value().edges.size());
  merged.insert(merged.end(), m2.value().edges.begin(), m2.value().edges.end());
  merged.insert(merged.end(), m1.value().edges.begin(), m1.value().edges.end());
  std::sort(merged.begin(), merged.end(), edge_order);
  CandidatePage output;
  float next = 0.0F;
  if (merged.size() > limit) {
    next = std::max(next, merged[limit].confidence);
    merged.resize(limit);
    output.has_more = true;
  }
  if (m1.value().has_more) { output.has_more = true; next = std::max(next, m1.value().next_confidence); }
  if (m2.value().has_more) { output.has_more = true; next = std::max(next, m2.value().next_confidence); }
  output.next_confidence = next;
  output.edges = std::move(merged);
  return output;
}

Result<RelationEdge> RelationRepository::get(EdgeId id, EvidenceTier tier) const {
  if (!impl_->m1 || !impl_->m2) return Status(ErrorCode::failed_precondition, "relation repository is not open");
  return tier == EvidenceTier::m1 ? impl_->m1->get(id) : impl_->m2->get(id);
}

Epoch RelationRepository::epoch() const noexcept {
  return impl_->mutation_epoch.load(std::memory_order_acquire);
}

Result<bool> RelationRepository::apply_observation_batch(BatchId batch_id,
                                                         std::span<const RelationObservation> observations) {
  if (!impl_->m1 || !impl_->m2) return Status(ErrorCode::failed_precondition, "relation repository is not open");
  std::vector<RelationObservation> filtered;
  filtered.reserve(observations.size());
  for (const RelationObservation& observation : observations) {
    if (!impl_->m2->contains(deterministic_edge_id(observation.source, observation.target))) filtered.push_back(observation);
  }
  auto result = impl_->m1->apply_batch(batch_id, filtered);
  if (!result) return result.status();
  if (result.value()) impl_->mutation_epoch.fetch_add(1, std::memory_order_release);
  return result.value();
}

Status RelationRepository::attach_evidence(EdgeId edge_id,
                                           std::span<const TraceId> trace_ids,
                                           std::size_t max_evidence) {
  if (max_evidence == 0 || max_evidence > kMaxEvidence) return Status(ErrorCode::invalid_argument, "max evidence is invalid");
  auto edge = impl_->m1->get(edge_id);
  if (!edge) return edge.status();
  bool changed = false;
  for (TraceId trace : trace_ids) {
    if (trace == 0) continue;
    if (std::find(edge.value().evidence.begin(), edge.value().evidence.end(), trace) == edge.value().evidence.end()) {
      edge.value().evidence.push_back(trace);
      changed = true;
    }
  }
  if (!changed) return Status::Ok();
  if (edge.value().evidence.size() > max_evidence) {
    edge.value().evidence.erase(edge.value().evidence.begin(),
                                edge.value().evidence.begin() + static_cast<std::ptrdiff_t>(edge.value().evidence.size() - max_evidence));
  }
  // Evidence IDs are replay metadata, not search semantics. Do not advance the semantic epoch.
  RLM_RETURN_IF_ERROR(impl_->m1->upsert(edge.value()));
  return Status::Ok();
}

Status RelationRepository::upsert_m2(const RelationEdge& source_edge) {
  RelationEdge edge = source_edge;
  edge.tier = EvidenceTier::m2;
  RLM_RETURN_IF_ERROR(impl_->m2->upsert(std::move(edge)));
  impl_->mutation_epoch.fetch_add(1, std::memory_order_release);
  return Status::Ok();
}

Status RelationRepository::erase_m1(EdgeId edge_id) {
  RLM_RETURN_IF_ERROR(impl_->m1->erase(edge_id));
  impl_->mutation_epoch.fetch_add(1, std::memory_order_release);
  return Status::Ok();
}

Result<EdgePin> RelationRepository::pin_m1(EdgeId edge_id) {
  if (!impl_->m1->contains(edge_id)) return Status(ErrorCode::not_found, "cannot pin missing M1 edge");
  if (!impl_->pins.pin(edge_id)) return Status(ErrorCode::unavailable, "M1 edge is being expired");
  return EdgePin([registry = &impl_->pins, edge_id]() { registry->unpin(edge_id); });
}

Result<std::size_t> RelationRepository::expire_m1(std::uint64_t cutoff_ms, std::size_t max_to_remove) {
  if (max_to_remove == 0) return std::size_t{0};
  const std::vector<RelationEdge> candidates = impl_->m1->expired_before(cutoff_ms, max_to_remove);
  std::size_t removed = 0;
  for (const RelationEdge& edge : candidates) {
    if (!impl_->pins.mark_expiring(edge.id)) continue;
    const Status status = impl_->m1->erase(edge.id);
    impl_->pins.finish_expiring(edge.id);
    if (!status) return status;
    ++removed;
    impl_->mutation_epoch.fetch_add(1, std::memory_order_release);
  }
  return removed;
}

std::size_t RelationRepository::m1_count() const { return impl_->m1 ? impl_->m1->count() : 0; }
std::size_t RelationRepository::m2_count() const { return impl_->m2 ? impl_->m2->count() : 0; }

Status RelationRepository::flush() {
  RLM_RETURN_IF_ERROR(impl_->m1->flush());
  return impl_->m2->flush();
}

Status RelationRepository::checkpoint() {
  RLM_RETURN_IF_ERROR(impl_->m2->checkpoint());
  return impl_->m1->checkpoint();
}

}  // namespace rlm
