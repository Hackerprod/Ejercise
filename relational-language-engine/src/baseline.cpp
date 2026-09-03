#include "rlm/baseline.hpp"

#include <algorithm>
#include <iterator>

namespace rlm {

std::uint64_t TrigramBaseline::pair_key(TokenId a, TokenId b) noexcept {
  return (static_cast<std::uint64_t>(a) << 32U) | static_cast<std::uint64_t>(b);
}

std::optional<TokenId> TrigramBaseline::best(const Counts& counts) {
  if (counts.empty()) return std::nullopt;
  auto selected = counts.begin();
  for (auto it = std::next(counts.begin()); it != counts.end(); ++it) {
    if (it->second > selected->second || (it->second == selected->second && it->first < selected->first)) selected = it;
  }
  return selected->first;
}

void TrigramBaseline::train(std::span<const TokenId> tokens) {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    ++unigram_[tokens[i]];
    if (i >= 1) ++bigram_[tokens[i - 1]][tokens[i]];
    if (i >= 2) ++trigram_[pair_key(tokens[i - 2], tokens[i - 1])][tokens[i]];
  }
}

std::optional<TokenId> TrigramBaseline::predict(std::span<const TokenId> context) const {
  if (context.size() >= 2) {
    const auto found = trigram_.find(pair_key(context[context.size() - 2], context.back()));
    if (found != trigram_.end()) {
      const auto result = best(found->second);
      if (result) return result;
    }
  }
  if (!context.empty()) {
    const auto found = bigram_.find(context.back());
    if (found != bigram_.end()) {
      const auto result = best(found->second);
      if (result) return result;
    }
  }
  return best(unigram_);
}

double TrigramBaseline::next_token_accuracy(std::span<const TokenId> tokens) const {
  if (tokens.size() < 3) return 0.0;
  std::size_t correct = 0;
  std::size_t total = 0;
  for (std::size_t i = 2; i < tokens.size(); ++i) {
    const auto prediction = predict(tokens.first(i));
    if (prediction && *prediction == tokens[i]) ++correct;
    ++total;
  }
  return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

std::size_t TrigramBaseline::state_count() const noexcept {
  return trigram_.size() + bigram_.size() + unigram_.size();
}

}  // namespace rlm
