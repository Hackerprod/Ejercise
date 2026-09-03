#pragma once

#include "rlm/status.hpp"
#include "rlm/types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace rlm {

class TrigramBaseline final {
 public:
  void train(std::span<const TokenId> tokens);
  [[nodiscard]] std::optional<TokenId> predict(std::span<const TokenId> context) const;
  [[nodiscard]] double next_token_accuracy(std::span<const TokenId> tokens) const;
  [[nodiscard]] std::size_t state_count() const noexcept;

 private:
  using Counts = std::unordered_map<TokenId, std::uint64_t>;
  static std::uint64_t pair_key(TokenId a, TokenId b) noexcept;
  static std::optional<TokenId> best(const Counts& counts);

  std::unordered_map<std::uint64_t, Counts> trigram_;
  std::unordered_map<TokenId, Counts> bigram_;
  Counts unigram_;
};

}  // namespace rlm
