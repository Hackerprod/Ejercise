#include "rlm/embedding_store.hpp"

#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace rlm {
namespace {
constexpr std::array<std::byte, 8> kMagic{
    std::byte{'R'}, std::byte{'L'}, std::byte{'E'}, std::byte{'M'},
    std::byte{'B'}, std::byte{'D'}, std::byte{'1'}, std::byte{0}};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxTokenLength = 1U << 20U;
constexpr std::size_t kMaxDimension = 1U << 20U;
constexpr std::size_t kMaxTokens = 1ULL << 32U;

Status build_file(const std::vector<std::pair<std::string, std::vector<float>>>& rows,
                  const std::filesystem::path& output,
                  Durability durability) {
  if (rows.empty()) return Status(ErrorCode::invalid_argument, "embedding input is empty");
  const std::size_t dimension = rows.front().second.size();
  if (dimension == 0 || dimension > kMaxDimension) {
    return Status(ErrorCode::invalid_argument, "embedding dimension is invalid");
  }
  if (rows.size() > kMaxTokens) return Status(ErrorCode::resource_exhausted, "too many embedding rows");
  std::unordered_set<std::string> tokens;
  tokens.reserve(rows.size());
  std::uint64_t vocabulary_hash = 1469598103934665603ULL;
  ByteWriter writer;
  writer.bytes(kMagic);
  writer.u32(kVersion);
  writer.u32(static_cast<std::uint32_t>(dimension));
  writer.u64(static_cast<std::uint64_t>(rows.size()));
  for (const auto& [token, vector] : rows) {
    if (token.empty() || token.size() > kMaxTokenLength) {
      return Status(ErrorCode::invalid_argument, "embedding token length is invalid");
    }
    if (vector.size() != dimension) {
      return Status(ErrorCode::invalid_argument, "embedding rows have inconsistent dimensions");
    }
    if (!tokens.insert(token).second) {
      return Status(ErrorCode::invalid_argument, "duplicate embedding token: " + token);
    }
    vocabulary_hash = stable_hash64(token, vocabulary_hash);
  }
  writer.u64(vocabulary_hash);
  for (const auto& [token, vector] : rows) {
    auto quantized = QuantizedVector::quantize(vector);
    if (!quantized) return quantized.status();
    writer.string(token);
    writer.f32(quantized.value().scale);
    writer.bytes(std::as_bytes(std::span{quantized.value().values.data(), quantized.value().values.size()}));
  }
  writer.u32(crc32(writer.data()));
  return write_file_atomic(output, writer.data(), durability);
}

std::string normalize_token(std::string_view raw) {
  std::size_t begin = 0;
  std::size_t end = raw.size();
  while (begin < end && std::ispunct(static_cast<unsigned char>(raw[begin])) != 0 && raw[begin] != '_' && raw[begin] != '-') ++begin;
  while (end > begin && std::ispunct(static_cast<unsigned char>(raw[end - 1])) != 0 && raw[end - 1] != '_' && raw[end - 1] != '-') --end;
  std::string output(raw.substr(begin, end - begin));
  for (char& ch : output) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return output;
}
}  // namespace

Status FrozenEmbeddingStore::open(const std::filesystem::path& path) {
  if (mapping_.open()) return Status(ErrorCode::failed_precondition, "embedding store already open");
  RLM_RETURN_IF_ERROR(mapping_.open_read_only(path));
  const auto all = mapping_.bytes();
  if (all.size() < 8 + 4 + 4 + 8 + 8 + 4) {
    mapping_.close();
    return Status(ErrorCode::data_loss, "embedding file is too small");
  }
  const std::uint32_t expected_crc = [&]() {
    const std::size_t offset = all.size() - 4;
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      value |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(all[offset + i])) << (i * 8U);
    }
    return value;
  }();
  if (crc32(all.first(all.size() - 4)) != expected_crc) {
    mapping_.close();
    return Status(ErrorCode::data_loss, "embedding file CRC mismatch");
  }
  ByteReader reader(all.first(all.size() - 4));
  auto magic = reader.bytes(kMagic.size());
  if (!magic || !std::equal(kMagic.begin(), kMagic.end(), magic.value().begin())) {
    mapping_.close();
    return Status(ErrorCode::data_loss, "embedding file magic mismatch");
  }
  auto version = reader.u32();
  auto dimension = reader.u32();
  auto count = reader.u64();
  auto vocabulary_hash = reader.u64();
  if (!version || !dimension || !count || !vocabulary_hash) {
    mapping_.close();
    return Status(ErrorCode::data_loss, "truncated embedding header");
  }
  if (version.value() != kVersion || dimension.value() == 0 || dimension.value() > kMaxDimension ||
      count.value() == 0 || count.value() > kMaxTokens) {
    mapping_.close();
    return Status(ErrorCode::data_loss, "unsupported or invalid embedding header");
  }
  dimension_ = dimension.value();
  entries_.reserve(static_cast<std::size_t>(count.value()));
  token_to_id_.reserve(static_cast<std::size_t>(count.value()));
  std::uint64_t computed_vocabulary_hash = 1469598103934665603ULL;
  for (std::uint64_t index = 0; index < count.value(); ++index) {
    auto token_result = reader.string(kMaxTokenLength);
    auto scale_result = reader.f32();
    auto quantized_result = reader.bytes(dimension_);
    if (!token_result || !scale_result || !quantized_result) {
      entries_.clear(); token_to_id_.clear(); mapping_.close();
      return Status(ErrorCode::data_loss, "truncated embedding record " + std::to_string(index));
    }
    if (scale_result.value() <= 0.0F) {
      entries_.clear(); token_to_id_.clear(); mapping_.close();
      return Status(ErrorCode::data_loss, "embedding record has invalid scale");
    }
    const TokenId id = static_cast<TokenId>(index);
    std::string token_value = std::move(token_result).value();
    computed_vocabulary_hash = stable_hash64(token_value, computed_vocabulary_hash);
    if (!token_to_id_.emplace(token_value, id).second) {
      entries_.clear(); token_to_id_.clear(); mapping_.close();
      return Status(ErrorCode::data_loss, "duplicate token in embedding file");
    }
    entries_.push_back(Entry{std::move(token_value), scale_result.value(), quantized_result.value()});
  }
  if (reader.remaining() != 0) {
    entries_.clear(); token_to_id_.clear(); mapping_.close();
    return Status(ErrorCode::data_loss, "trailing bytes in embedding file");
  }
  if (computed_vocabulary_hash != vocabulary_hash.value()) {
    entries_.clear(); token_to_id_.clear(); mapping_.close();
    return Status(ErrorCode::data_loss, "embedding vocabulary checksum mismatch");
  }
  checksum_ = stable_hash64(all.first(all.size() - 4));
  return Status::Ok();
}

Result<TokenId> FrozenEmbeddingStore::token_id(std::string_view token_value) const {
  const auto found = token_to_id_.find(std::string(token_value));
  if (found == token_to_id_.end()) return Status(ErrorCode::not_found, "token is not in frozen vocabulary: " + std::string(token_value));
  return found->second;
}

Result<std::string_view> FrozenEmbeddingStore::token(TokenId id) const {
  if (static_cast<std::size_t>(id) >= entries_.size()) return Status(ErrorCode::not_found, "token id is out of range");
  return std::string_view(entries_[id].token);
}

Status FrozenEmbeddingStore::copy_embedding(TokenId id, std::span<float> output) const {
  if (static_cast<std::size_t>(id) >= entries_.size()) return Status(ErrorCode::not_found, "token id is out of range");
  if (output.size() != dimension_) return Status(ErrorCode::invalid_argument, "embedding output dimension mismatch");
  const Entry& entry = entries_[id];
  for (std::size_t i = 0; i < dimension_; ++i) {
    output[i] = static_cast<float>(static_cast<std::int8_t>(std::to_integer<unsigned char>(entry.quantized[i]))) * entry.scale;
  }
  return Status::Ok();
}

Result<QuantizedVector> FrozenEmbeddingStore::relation_vector(TokenId source, TokenId target) const {
  std::vector<float> source_vector(dimension_);
  std::vector<float> target_vector(dimension_);
  Status status = copy_embedding(source, source_vector);
  if (!status) return status;
  status = copy_embedding(target, target_vector);
  if (!status) return status;
  std::vector<float> relation(dimension_);
  for (std::size_t i = 0; i < dimension_; ++i) relation[i] = target_vector[i] - source_vector[i];
  relation = normalized(relation);
  return QuantizedVector::quantize(relation);
}

Status EmbeddingFileBuilder::from_text(const std::filesystem::path& input,
                                       const std::filesystem::path& output,
                                       Durability durability) {
  std::ifstream stream(input);
  if (!stream) return Status(ErrorCode::io_error, "cannot open embedding text file: '" + input.string() + "'");
  std::vector<std::pair<std::string, std::vector<float>>> rows;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    if (line.empty() || line.front() == '#') continue;
    std::istringstream parser(line);
    std::string token_value;
    if (!(parser >> token_value)) continue;
    std::vector<float> vector;
    float value = 0.0F;
    while (parser >> value) vector.push_back(value);
    if (!parser.eof()) return Status(ErrorCode::invalid_argument, "invalid float at embedding line " + std::to_string(line_number));
    if (vector.empty()) return Status(ErrorCode::invalid_argument, "missing vector at embedding line " + std::to_string(line_number));
    rows.emplace_back(std::move(token_value), std::move(vector));
  }
  return build_file(rows, output, durability);
}

Status EmbeddingFileBuilder::from_rows(
    const std::vector<std::pair<std::string, std::vector<float>>>& rows,
    const std::filesystem::path& output,
    Durability durability) {
  return build_file(rows, output, durability);
}

Result<TokenizationResult> tokenize_whitespace(std::string_view text,
                                               const IEmbeddingStore& embeddings,
                                               bool reject_unknown) {
  TokenizationResult output;
  std::size_t begin = 0;
  while (begin < text.size()) {
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) ++begin;
    if (begin >= text.size()) break;
    std::size_t end = begin;
    while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end])) == 0) ++end;
    const std::string token_value = normalize_token(text.substr(begin, end - begin));
    if (!token_value.empty()) {
      auto id = embeddings.token_id(token_value);
      if (id) {
        output.tokens.push_back(id.value());
      } else {
        ++output.unknown_tokens;
        if (reject_unknown) return Status(ErrorCode::not_found, "unknown token: " + token_value);
      }
    }
    begin = end;
  }
  if (output.tokens.empty()) return Status(ErrorCode::invalid_argument, "input contains no known tokens");
  return output;
}

}  // namespace rlm
