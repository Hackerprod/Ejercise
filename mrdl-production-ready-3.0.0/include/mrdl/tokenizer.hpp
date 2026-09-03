#pragma once

#include "mrdl/common.hpp"
#include "mrdl/config.hpp"

#include <queue>

namespace mrdl {

class HybridTokenizer final {
public:
    HybridTokenizer() = default;

    static HybridTokenizer load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const;

    static HybridTokenizer build_from_corpus(const std::filesystem::path& corpus,
                                             const TokenizerConfig& config);

    [[nodiscard]] std::vector<TokenId> encode(std::string_view text,
                                              bool add_bos = false,
                                              bool add_eos = false) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> ids,
                                     bool skip_special = true) const;

    [[nodiscard]] std::string_view token(TokenId id) const;
    [[nodiscard]] std::size_t size() const noexcept { return id_to_piece_.size(); }
    [[nodiscard]] bool lowercase() const noexcept { return lowercase_; }

    static std::vector<std::string> split_pieces(std::string_view text, bool lowercase);

private:
    bool lowercase_{false};
    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, TokenId> piece_to_id_;

    void rebuild_index();
};

class CorpusTokenStream final {
public:
    CorpusTokenStream(const std::filesystem::path& corpus,
                      const HybridTokenizer& tokenizer,
                      bool add_boundaries = true);

    bool next(TokenId& token);
    [[nodiscard]] std::uint64_t line_number() const noexcept { return line_number_; }

private:
    std::ifstream stream_;
    const HybridTokenizer& tokenizer_;
    bool add_boundaries_{true};
    std::vector<TokenId> buffer_;
    std::size_t offset_{0};
    std::uint64_t line_number_{0};
};

}  // namespace mrdl
