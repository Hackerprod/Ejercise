#include "rlm/trainer.hpp"

#include "rlm/binary.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace rlm {
namespace {
constexpr std::array<std::byte, 8> kCheckpointMagic{
    std::byte{'R'}, std::byte{'L'}, std::byte{'T'}, std::byte{'R'},
    std::byte{'A'}, std::byte{'I'}, std::byte{'N'}, std::byte{'1'}};
constexpr std::uint32_t kCheckpointVersion = 1;

std::string normalize_token(std::string_view raw) {
  std::size_t begin = 0;
  std::size_t end = raw.size();
  while (begin < end && std::ispunct(static_cast<unsigned char>(raw[begin])) != 0 && raw[begin] != '_' && raw[begin] != '-') ++begin;
  while (end > begin && std::ispunct(static_cast<unsigned char>(raw[end - 1])) != 0 && raw[end - 1] != '_' && raw[end - 1] != '-') --end;
  std::string output(raw.substr(begin, end - begin));
  for (char& ch : output) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return output;
}

BatchId make_batch_id(std::uint64_t corpus_hash, std::uint64_t raw_begin, std::uint64_t raw_end) {
  return hash_combine64(hash_combine64(corpus_hash, raw_begin), raw_end);
}

void write_stats(ByteWriter& writer, const TrainingStats& stats) {
  writer.u64(stats.raw_tokens_seen); writer.u64(stats.known_tokens_seen); writer.u64(stats.unknown_tokens_seen);
  writer.u64(stats.batches_committed); writer.u64(stats.unique_relations_observed);
  writer.u64(stats.replay_traces_written); writer.u64(stats.promotions_committed);
  writer.u64(stats.promotions_rejected); writer.u64(stats.promotions_unknown); writer.u64(stats.m1_expired);
}

Result<TrainingStats> read_stats(ByteReader& reader) {
  TrainingStats stats;
  auto raw = reader.u64(); auto known = reader.u64(); auto unknown = reader.u64(); auto batches = reader.u64();
  auto relations = reader.u64(); auto traces = reader.u64(); auto promoted = reader.u64();
  auto rejected = reader.u64(); auto indeterminate = reader.u64(); auto expired = reader.u64();
  if (!raw || !known || !unknown || !batches || !relations || !traces || !promoted || !rejected || !indeterminate || !expired) {
    return Status(ErrorCode::data_loss, "truncated training statistics");
  }
  stats.raw_tokens_seen = raw.value(); stats.known_tokens_seen = known.value(); stats.unknown_tokens_seen = unknown.value();
  stats.batches_committed = batches.value(); stats.unique_relations_observed = relations.value();
  stats.replay_traces_written = traces.value(); stats.promotions_committed = promoted.value();
  stats.promotions_rejected = rejected.value(); stats.promotions_unknown = indeterminate.value(); stats.m1_expired = expired.value();
  return stats;
}
}  // namespace

struct Trainer::Checkpoint final {
  std::uint64_t corpus_hash{0};
  std::uint64_t embedding_checksum{0};
  TrainingStats stats;
  std::vector<TokenId> tail;
};

struct Trainer::Aggregate final {
  RelationObservation observation;
  double confidence_weighted_sum{0.0};
};

Result<std::uint64_t> Trainer::corpus_fingerprint(const std::filesystem::path& corpus) const {
  std::ifstream input(corpus, std::ios::binary);
  if (!input) return Status(ErrorCode::io_error, "cannot open corpus: '" + corpus.string() + "'");
  std::array<char, 1U << 20U> buffer{};
  std::uint64_t hash = 1469598103934665603ULL;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash = stable_hash64(std::as_bytes(std::span{buffer.data(), static_cast<std::size_t>(count)}), hash);
    }
  }
  if (!input.eof()) return Status(ErrorCode::io_error, "failed while hashing corpus: '" + corpus.string() + "'");
  return hash_combine64(hash, embeddings_.checksum());
}

Result<Trainer::Checkpoint> Trainer::load_checkpoint(const std::filesystem::path& corpus,
                                                       std::uint64_t corpus_hash) const {
  const std::filesystem::path path = config_.storage.state_dir / "training" / "checkpoint.bin";
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    if (ec) return Status(ErrorCode::io_error, "cannot stat training checkpoint: " + ec.message());
    Checkpoint empty; empty.corpus_hash = corpus_hash; empty.embedding_checksum = embeddings_.checksum(); return empty;
  }
  auto file = read_file(path, 64U * 1024U * 1024U);
  if (!file) return file.status();
  const auto bytes = std::span<const std::byte>(file.value());
  if (bytes.size() < kCheckpointMagic.size() + 4) return Status(ErrorCode::data_loss, "training checkpoint is too small");
  ByteReader footer(bytes.last(4));
  auto expected_crc = footer.u32();
  if (!expected_crc || crc32(bytes.first(bytes.size() - 4)) != expected_crc.value()) {
    return Status(ErrorCode::data_loss, "training checkpoint CRC mismatch");
  }
  ByteReader reader(bytes.first(bytes.size() - 4));
  auto magic = reader.bytes(kCheckpointMagic.size()); auto version = reader.u32();
  auto stored_corpus = reader.u64(); auto stored_embeddings = reader.u64();
  if (!magic || !version || !stored_corpus || !stored_embeddings ||
      !std::equal(kCheckpointMagic.begin(), kCheckpointMagic.end(), magic.value().begin()) ||
      version.value() != kCheckpointVersion) {
    return Status(ErrorCode::data_loss, "training checkpoint header mismatch");
  }
  if (stored_corpus.value() != corpus_hash) {
    return Status(ErrorCode::failed_precondition, "checkpoint belongs to a different corpus: '" + corpus.string() + "'");
  }
  if (stored_embeddings.value() != embeddings_.checksum()) {
    return Status(ErrorCode::failed_precondition, "checkpoint embedding checksum does not match frozen store");
  }
  auto stats = read_stats(reader);
  auto tail_count = reader.u32();
  if (!stats || !tail_count || tail_count.value() > config_.training.context_radius) {
    return Status(ErrorCode::data_loss, "invalid training checkpoint tail");
  }
  Checkpoint checkpoint;
  checkpoint.corpus_hash = stored_corpus.value(); checkpoint.embedding_checksum = stored_embeddings.value();
  checkpoint.stats = stats.value(); checkpoint.tail.reserve(tail_count.value());
  for (std::uint32_t i = 0; i < tail_count.value(); ++i) {
    auto token = reader.u32(); if (!token) return token.status(); checkpoint.tail.push_back(token.value());
  }
  if (reader.remaining() != 0) return Status(ErrorCode::data_loss, "trailing bytes in training checkpoint");
  return checkpoint;
}

Status Trainer::save_checkpoint(const Checkpoint& checkpoint) const {
  ByteWriter writer;
  writer.bytes(kCheckpointMagic); writer.u32(kCheckpointVersion);
  writer.u64(checkpoint.corpus_hash); writer.u64(checkpoint.embedding_checksum);
  write_stats(writer, checkpoint.stats);
  writer.u32(static_cast<std::uint32_t>(checkpoint.tail.size()));
  for (TokenId token : checkpoint.tail) writer.u32(token);
  writer.u32(crc32(writer.data()));
  return write_file_atomic(config_.storage.state_dir / "training" / "checkpoint.bin",
                           writer.data(), config_.storage.durability);
}

Result<std::vector<EdgeId>> Trainer::process_batch(
    BatchId batch_id,
    std::span<const TokenId> prefix,
    std::span<const TokenId> batch,
    TrainingStats& stats,
    std::unordered_map<EdgeId, std::vector<std::vector<TokenId>>>& samples) {
  std::vector<TokenId> sequence;
  sequence.reserve(prefix.size() + batch.size());
  sequence.insert(sequence.end(), prefix.begin(), prefix.end());
  sequence.insert(sequence.end(), batch.begin(), batch.end());
  std::unordered_map<EdgeId, Aggregate> aggregates;
  aggregates.reserve(batch.size() * std::min<std::size_t>(config_.training.context_radius, 8));
  const std::size_t prefix_size = prefix.size();
  const std::uint64_t now = unix_time_ms();
  for (std::size_t target_index = prefix_size; target_index < sequence.size(); ++target_index) {
    const std::size_t available = std::min(config_.training.context_radius, target_index);
    for (std::size_t distance = 1; distance <= available; ++distance) {
      const std::size_t source_index = target_index - distance;
      const TokenId source = sequence[source_index];
      const TokenId target = sequence[target_index];
      const EdgeId id = deterministic_edge_id(source, target);
      auto found = aggregates.find(id);
      if (found == aggregates.end()) {
        auto relation = embeddings_.relation_vector(source, target);
        if (!relation) return relation.status();
        Aggregate aggregate;
        aggregate.observation.source = source;
        aggregate.observation.target = target;
        aggregate.observation.relation = std::move(relation).value();
        aggregate.observation.observed_at_ms = now;
        aggregate.observation.count = 0;
        found = aggregates.emplace(id, std::move(aggregate)).first;
      }
      const float confidence = 1.0F / static_cast<float>(distance);
      ++found->second.observation.count;
      found->second.confidence_weighted_sum += confidence;
      auto& edge_samples = samples[id];
      if (edge_samples.size() < config_.training.evidence_cases_per_edge) {
        const std::size_t context_begin = source_index > 31 ? source_index - 31 : 0;
        edge_samples.emplace_back(sequence.begin() + static_cast<std::ptrdiff_t>(context_begin),
                                  sequence.begin() + static_cast<std::ptrdiff_t>(source_index + 1));
      }
    }
  }
  std::vector<RelationObservation> observations;
  observations.reserve(aggregates.size());
  std::vector<EdgeId> edge_ids;
  edge_ids.reserve(aggregates.size());
  for (auto& [id, aggregate] : aggregates) {
    // count starts at zero in a default observation; each occurrence increments it.
    if (aggregate.observation.count == 0) continue;
    aggregate.observation.confidence = static_cast<float>(
        aggregate.confidence_weighted_sum / static_cast<double>(aggregate.observation.count));
    observations.push_back(std::move(aggregate.observation));
    edge_ids.push_back(id);
  }
  auto applied = relations_.apply_observation_batch(batch_id, observations);
  if (!applied) return applied.status();
  // This batch is after the trainer checkpoint offset. Count it once in trainer progress
  // even when the relation WAL already contains it after a crash-before-checkpoint recovery.
  ++stats.batches_committed;
  stats.unique_relations_observed += observations.size();
  return edge_ids;
}

Status Trainer::collect_evidence_and_promote(
    std::span<const EdgeId> edge_ids,
    const std::unordered_map<EdgeId, std::vector<std::vector<TokenId>>>& samples,
    bool auto_promote,
    TrainingStats& stats) {
  std::vector<EdgeId> sorted(edge_ids.begin(), edge_ids.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  std::size_t promotion_attempts = 0;
  for (EdgeId id : sorted) {
    auto edge = relations_.get(id, EvidenceTier::m1);
    if (!edge) {
      if (edge.status().code() == ErrorCode::not_found) continue;
      return edge.status();
    }
    if (edge.value().support < config_.training.min_support_for_promotion ||
        edge.value().confidence < config_.training.min_confidence_for_promotion) {
      continue;
    }
    const auto sample_it = samples.find(id);
    if (sample_it == samples.end() || sample_it->second.empty()) continue;
    std::vector<TraceId> fresh_traces;
    const std::size_t needed = std::min(config_.training.evidence_cases_per_edge, sample_it->second.size());
    fresh_traces.reserve(needed);
    for (std::size_t sample_index = 0; sample_index < needed; ++sample_index) {
      SearchRequest request;
      request.context = sample_it->second[sample_index];
      request.lane = Lane::full;
      request.edge_under_test = id;
      request.expected_target = edge.value().target;
      request.max_depth_override = 1;
      request.capture_trace = true;
      auto search = search_.run(request);
      if (!search) return search.status();
      if (!search.value().has_prediction || !search.value().trace.has_value() ||
          std::find(search.value().path_edges.begin(), search.value().path_edges.end(), id) == search.value().path_edges.end()) {
        continue;
      }
      auto trace_id = replay_store_.put(std::move(*search.value().trace));
      if (!trace_id) return trace_id.status();
      fresh_traces.push_back(trace_id.value());
      ++stats.replay_traces_written;
    }
    if (!fresh_traces.empty()) {
      RLM_RETURN_IF_ERROR(relations_.attach_evidence(id, fresh_traces,
                                                     config_.training.evidence_cases_per_edge));
    }
    if (!auto_promote || fresh_traces.size() < config_.audit.min_exact_cases ||
        promotion_attempts >= config_.training.auto_promote_per_batch) {
      continue;
    }
    ++promotion_attempts;
    auto promotion = promotions_.promote(id);
    if (!promotion) return promotion.status();
    if (promotion.value().committed) ++stats.promotions_committed;
    else if (promotion.value().verdict == AuditVerdict::reject) ++stats.promotions_rejected;
    else ++stats.promotions_unknown;
  }
  return Status::Ok();
}

Result<TrainingStats> Trainer::train_file(const std::filesystem::path& corpus,
                                          TrainerOptions options) {
  auto hash = corpus_fingerprint(corpus);
  if (!hash) return hash.status();
  Checkpoint checkpoint;
  checkpoint.corpus_hash = hash.value();
  checkpoint.embedding_checksum = embeddings_.checksum();
  if (options.resume) {
    auto loaded = load_checkpoint(corpus, hash.value());
    if (!loaded) return loaded.status();
    checkpoint = std::move(loaded).value();
  }
  std::ifstream input(corpus);
  if (!input) return Status(ErrorCode::io_error, "cannot open corpus: '" + corpus.string() + "'");
  std::uint64_t raw_position = 0;
  std::string raw;
  while (raw_position < checkpoint.stats.raw_tokens_seen && (input >> raw)) ++raw_position;
  if (raw_position != checkpoint.stats.raw_tokens_seen) {
    return Status(ErrorCode::data_loss, "corpus ended before the saved training offset");
  }
  std::vector<TokenId> prefix = checkpoint.tail;
  std::vector<TokenId> batch;
  batch.reserve(config_.training.batch_tokens);
  std::uint64_t batch_raw_begin = raw_position;
  std::size_t batches_since_checkpoint = 0;

  auto commit_batch = [&](std::uint64_t raw_end) -> Status {
    if (batch.empty()) return Status::Ok();
    std::unordered_map<EdgeId, std::vector<std::vector<TokenId>>> samples;
    const BatchId batch_id = make_batch_id(hash.value(), batch_raw_begin, raw_end);
    auto edge_ids = process_batch(batch_id, prefix, batch, checkpoint.stats, samples);
    if (!edge_ids) return edge_ids.status();
    RLM_RETURN_IF_ERROR(collect_evidence_and_promote(edge_ids.value(), samples, options.auto_promote,
                                                     checkpoint.stats));
    for (TokenId token : batch) {
      prefix.push_back(token);
      if (prefix.size() > config_.training.context_radius) prefix.erase(prefix.begin());
    }
    checkpoint.tail = prefix;
    ++batches_since_checkpoint;
    batch.clear();
    batch_raw_begin = raw_end;
    if (config_.training.m1_ttl_seconds > 0) {
      const std::uint64_t ttl_ms = config_.training.m1_ttl_seconds > std::numeric_limits<std::uint64_t>::max() / 1000ULL
          ? std::numeric_limits<std::uint64_t>::max()
          : config_.training.m1_ttl_seconds * 1000ULL;
      const std::uint64_t now = unix_time_ms();
      const std::uint64_t cutoff = now > ttl_ms ? now - ttl_ms : 0;
      auto expired = relations_.expire_m1(cutoff, 100000);
      if (!expired) return expired.status();
      checkpoint.stats.m1_expired += expired.value();
    }
    if (relations_.m1_count() > config_.training.max_m1_edges) {
      return Status(ErrorCode::resource_exhausted,
                    "M1 edge limit reached after TTL sweep; increase storage or tighten promotion/TTL policy");
    }
    if (batches_since_checkpoint >= config_.training.checkpoint_every_batches) {
      RLM_RETURN_IF_ERROR(relations_.flush());
      RLM_RETURN_IF_ERROR(replay_store_.flush());
      RLM_RETURN_IF_ERROR(promotions_.flush());
      RLM_RETURN_IF_ERROR(save_checkpoint(checkpoint));
      batches_since_checkpoint = 0;
    }
    return Status::Ok();
  };

  while (input >> raw) {
    ++raw_position;
    ++checkpoint.stats.raw_tokens_seen;
    const std::string token_value = normalize_token(raw);
    if (token_value.empty()) continue;
    auto token = embeddings_.token_id(token_value);
    if (!token) {
      ++checkpoint.stats.unknown_tokens_seen;
      if (config_.training.reject_unknown_tokens) return token.status();
      continue;
    }
    batch.push_back(token.value());
    ++checkpoint.stats.known_tokens_seen;
    if (batch.size() >= config_.training.batch_tokens) {
      const Status status = commit_batch(raw_position);
      if (!status) return status;
    }
  }
  if (!input.eof()) return Status(ErrorCode::io_error, "failed while reading corpus");
  const Status final_status = commit_batch(raw_position);
  if (!final_status) return final_status;
  RLM_RETURN_IF_ERROR(relations_.flush());
  RLM_RETURN_IF_ERROR(replay_store_.flush());
  RLM_RETURN_IF_ERROR(promotions_.flush());
  if (options.checkpoint_at_end) RLM_RETURN_IF_ERROR(save_checkpoint(checkpoint));
  return checkpoint.stats;
}

}  // namespace rlm
