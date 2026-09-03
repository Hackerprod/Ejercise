#pragma once

#include "mrdl/common.hpp"
#include "mrdl/config.hpp"
#include "mrdl/tokenizer.hpp"

namespace mrdl {

class IEmbeddingStore {
public:
    virtual ~IEmbeddingStore() = default;
    [[nodiscard]] virtual std::size_t token_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
    virtual void dequantize(TokenId id, std::span<float> output) const = 0;
    [[nodiscard]] virtual float dot_row(TokenId id, std::span<const float> vector) const = 0;
    [[nodiscard]] virtual float cosine_row(TokenId id, std::span<const float> vector) const = 0;
};

enum class EmbeddingInit : std::uint8_t { Random = 0, RandomIndexing = 1, ExternalFloat32 = 2 };

struct EmbeddingBuildOptions {
    EmbeddingInit mode{EmbeddingInit::RandomIndexing};
    std::uint32_t dimension{96};
    std::uint32_t context_window{4};
    std::uint32_t sparse_context_nonzero{8};
    std::uint64_t seed{0x4d52444cULL};
    std::optional<std::filesystem::path> external_f32;
};

class FrozenEmbeddingStore final : public IEmbeddingStore {
public:
    FrozenEmbeddingStore() = default;
    ~FrozenEmbeddingStore() override;
    FrozenEmbeddingStore(const FrozenEmbeddingStore&) = delete;
    FrozenEmbeddingStore& operator=(const FrozenEmbeddingStore&) = delete;
    FrozenEmbeddingStore(FrozenEmbeddingStore&& other) noexcept;
    FrozenEmbeddingStore& operator=(FrozenEmbeddingStore&& other) noexcept;

    static FrozenEmbeddingStore load(const std::filesystem::path& path);
    static void build(const std::filesystem::path& output,
                      const HybridTokenizer& tokenizer,
                      const std::optional<std::filesystem::path>& corpus,
                      const EmbeddingBuildOptions& options);

    [[nodiscard]] std::size_t token_count() const noexcept override { return token_count_; }
    [[nodiscard]] std::size_t dimension() const noexcept override { return dimension_; }
    void dequantize(TokenId id, std::span<float> output) const override;
    [[nodiscard]] float dot_row(TokenId id, std::span<const float> vector) const override;
    [[nodiscard]] float cosine_row(TokenId id, std::span<const float> vector) const override;

    [[nodiscard]] std::uint64_t content_hash() const noexcept { return content_hash_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    int fd_{-1};
    void* mapping_{nullptr};
    std::size_t mapping_size_{0};
    std::size_t token_count_{0};
    std::size_t dimension_{0};
    std::size_t row_stride_{0};
    const std::byte* rows_{nullptr};
    std::uint64_t content_hash_{0};

    void close() noexcept;
    [[nodiscard]] const std::byte* row(TokenId id) const;
};

}  // namespace mrdl
