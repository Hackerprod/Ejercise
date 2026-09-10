#pragma once

#include "mrdl/common.hpp"

namespace mrdl {

struct BaselineMetrics {
    std::uint64_t tokens{0};
    double negative_log_likelihood{0.0};
    std::uint64_t correct{0};
    double elapsed_seconds{0.0};

    [[nodiscard]] double average_loss() const noexcept {
        return tokens == 0U ? 0.0 : negative_log_likelihood / static_cast<double>(tokens);
    }
    [[nodiscard]] double perplexity() const noexcept {
        return std::exp(std::min(average_loss(), 80.0));
    }
    [[nodiscard]] double accuracy() const noexcept {
        return tokens == 0U ? 0.0 : static_cast<double>(correct) / static_cast<double>(tokens);
    }
    [[nodiscard]] double tokens_per_second() const noexcept {
        return elapsed_seconds <= 0.0 ? 0.0 : static_cast<double>(tokens) / elapsed_seconds;
    }
};

struct NGramKey {
    std::array<TokenId, 4> tokens{};
    std::uint8_t length{0};

    bool operator==(const NGramKey&) const = default;
};

struct NGramKeyHash {
    std::size_t operator()(const NGramKey& key) const noexcept {
        std::uint64_t hash = hash_combine(0x4e4752414dULL, key.length);
        for (std::size_t i = 0; i < key.length; ++i) hash = hash_combine(hash, key.tokens[i]);
        return static_cast<std::size_t>(hash);
    }
};

class NGramBaseline final {
public:
    explicit NGramBaseline(std::uint32_t order = 3U, double alpha = 0.1);

    void observe(std::span<const TokenId> context, TokenId target);
    [[nodiscard]] TokenId predict(std::span<const TokenId> context) const;
    [[nodiscard]] double probability(std::span<const TokenId> context, TokenId target) const;
    [[nodiscard]] double negative_log_likelihood(std::span<const TokenId> context, TokenId target) const;

    void save(const std::filesystem::path& path) const;
    static NGramBaseline load(const std::filesystem::path& path);

    [[nodiscard]] std::uint32_t order() const noexcept { return order_; }
    [[nodiscard]] std::uint64_t observations() const noexcept { return total_observations_; }
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return unigram_.size(); }
    [[nodiscard]] std::size_t context_count() const noexcept { return contexts_.size(); }

private:
    struct NextCounts {
        std::uint64_t total{0};
        std::unordered_map<TokenId, std::uint64_t> counts;
    };

    std::uint32_t order_{3U};
    double alpha_{0.1};
    std::uint64_t total_observations_{0};
    std::unordered_map<TokenId, std::uint64_t> unigram_;
    std::unordered_map<NGramKey, NextCounts, NGramKeyHash> contexts_;
    mutable std::vector<TokenId> top_unigrams_;
    mutable bool top_cache_dirty_{true};

    [[nodiscard]] NGramKey key_for(std::span<const TokenId> context, std::size_t length) const;
    void rebuild_top_cache() const;
};

}  // namespace mrdl
