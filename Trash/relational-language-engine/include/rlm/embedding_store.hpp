#pragma once

#include "rlm/binary.hpp"
#include "rlm/types.hpp"

#include <filesystem>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rlm {

class IEmbeddingStore {
 public:
  virtual ~IEmbeddingStore() = default;
  [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
  [[nodiscard]] virtual std::size_t token_count() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t checksum() const noexcept = 0;
  [[nodiscard]] virtual Result<TokenId> token_id(std::string_view token) const = 0;
  [[nodiscard]] virtual Result<std::string_view> token(TokenId id) const = 0;
  [[nodiscard]] virtual Status copy_embedding(TokenId id, std::span<float> output) const = 0;
  [[nodiscard]] virtual Result<QuantizedVector> relation_vector(TokenId source, TokenId target) const = 0;
};

class FrozenEmbeddingStore final : public IEmbeddingStore {
 public:
  FrozenEmbeddingStore() = default;
  ~FrozenEmbeddingStore() override = default;
  FrozenEmbeddingStore(const FrozenEmbeddingStore&) = delete;
  FrozenEmbeddingStore& operator=(const FrozenEmbeddingStore&) = delete;

  [[nodiscard]] Status open(const std::filesystem::path& path);
  [[nodiscard]] std::size_t dimension() const noexcept override { return dimension_; }
  [[nodiscard]] std::size_t token_count() const noexcept override { return entries_.size(); }
  [[nodiscard]] std::uint64_t checksum() const noexcept override { return checksum_; }
  [[nodiscard]] Result<TokenId> token_id(std::string_view token) const override;
  [[nodiscard]] Result<std::string_view> token(TokenId id) const override;
  [[nodiscard]] Status copy_embedding(TokenId id, std::span<float> output) const override;
  [[nodiscard]] Result<QuantizedVector> relation_vector(TokenId source, TokenId target) const override;

 private:
  struct Entry final {
    std::string token;
    float scale{1.0F};
    std::span<const std::byte> quantized;
  };

  MappedFile mapping_;
  std::size_t dimension_{0};
  std::uint64_t checksum_{0};
  std::vector<Entry> entries_;
  std::unordered_map<std::string, TokenId> token_to_id_;
};

class EmbeddingFileBuilder final {
 public:
  [[nodiscard]] static Status from_text(const std::filesystem::path& input,
                                        const std::filesystem::path& output,
                                        Durability durability = Durability::full);
  [[nodiscard]] static Status from_rows(
      const std::vector<std::pair<std::string, std::vector<float>>>& rows,
      const std::filesystem::path& output,
      Durability durability = Durability::full);
};

struct TokenizationResult final {
  std::vector<TokenId> tokens;
  std::size_t unknown_tokens{0};
};

[[nodiscard]] Result<TokenizationResult> tokenize_whitespace(std::string_view text,
                                                              const IEmbeddingStore& embeddings,
                                                              bool reject_unknown);

}  // namespace rlm
