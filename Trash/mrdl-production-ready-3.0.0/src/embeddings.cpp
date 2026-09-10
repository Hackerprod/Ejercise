#include "mrdl/embeddings.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mrdl {
namespace {

struct EmbeddingHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t storage_kind;  // 1 = row-wise symmetric int8 + float scale
    std::uint64_t token_count;
    std::uint32_t dimension;
    std::uint32_t row_stride;
    std::uint64_t payload_hash;
    std::uint64_t seed;
    std::array<std::uint8_t, 16> reserved{};
};
static_assert(sizeof(EmbeddingHeader) == 64U);

void atomic_write(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp." + std::to_string(unix_millis());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw Error("cannot create embedding file: " + temporary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        stream.flush();
        if (!stream) throw Error("failed writing embedding file: " + temporary);
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temporary, path, ec);
    }
    if (ec) {
        std::filesystem::remove(temporary);
        throw Error("cannot atomically install embeddings: " + ec.message());
    }
}

std::vector<float> random_embeddings(std::size_t count, std::size_t dim, std::uint64_t seed) {
    require(count <= std::numeric_limits<std::size_t>::max() / dim, "embedding allocation overflow");
    std::vector<float> values(count * dim);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    for (std::size_t row = 0; row < count; ++row) {
        auto vector = std::span<float>(values).subspan(row * dim, dim);
        for (float& value : vector) value = distribution(rng);
        normalize_in_place(vector);
    }
    return values;
}

void add_sparse_signature(std::span<float> target,
                          TokenId context,
                          float weight,
                          std::uint32_t nonzero,
                          std::uint64_t seed) {
    std::uint64_t state = hash_combine(seed, context);
    const auto count = std::min<std::size_t>(nonzero, target.size());
    for (std::size_t n = 0; n < count; ++n) {
        state = mix64(state + n);
        const std::size_t index = static_cast<std::size_t>(state % target.size());
        const float sign = ((state >> 63U) != 0U) ? 1.0F : -1.0F;
        target[index] += sign * weight;
    }
}

std::vector<float> random_indexing_embeddings(const HybridTokenizer& tokenizer,
                                               const std::filesystem::path& corpus,
                                               const EmbeddingBuildOptions& options) {
    const std::size_t count = tokenizer.size();
    const std::size_t dim = options.dimension;
    require(count <= std::numeric_limits<std::size_t>::max() / dim, "embedding allocation overflow");
    std::vector<float> values(count * dim, 0.0F);
    std::vector<std::uint64_t> frequencies(count, 0U);

    std::ifstream stream(corpus);
    if (!stream) throw Error("cannot open corpus for random indexing: " + corpus.string());
    std::string line;
    while (std::getline(stream, line)) {
        line.push_back('\n');
        const auto tokens = tokenizer.encode(line, true, true);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const TokenId center = tokens[i];
            if (center >= count) continue;
            ++frequencies[center];
            auto target = std::span<float>(values).subspan(static_cast<std::size_t>(center) * dim, dim);
            const std::size_t left = i > options.context_window ? i - options.context_window : 0U;
            const std::size_t right = std::min(tokens.size(), i + static_cast<std::size_t>(options.context_window) + 1U);
            for (std::size_t j = left; j < right; ++j) {
                if (j == i) continue;
                const auto distance = static_cast<float>(i > j ? i - j : j - i);
                const float weight = 1.0F / distance;
                add_sparse_signature(target, tokens[j], weight, options.sparse_context_nonzero, options.seed);
            }
        }
    }

    std::mt19937_64 rng(options.seed ^ 0x454d424544ULL);
    std::normal_distribution<float> noise(0.0F, 0.02F);
    for (std::size_t row = 0; row < count; ++row) {
        auto vector = std::span<float>(values).subspan(row * dim, dim);
        for (float& value : vector) value += noise(rng);
        normalize_in_place(vector);
    }
    return values;
}

std::vector<float> external_embeddings(const std::filesystem::path& input,
                                       std::size_t token_count,
                                       std::size_t dimension) {
    std::ifstream stream(input, std::ios::binary | std::ios::ate);
    if (!stream) throw Error("cannot open external embedding matrix: " + input.string());
    const auto size = stream.tellg();
    const auto expected = static_cast<std::uint64_t>(token_count) * dimension * sizeof(float);
    require(size >= 0 && static_cast<std::uint64_t>(size) == expected,
            "external embedding matrix has wrong byte size");
    std::vector<float> values(token_count * dimension);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected));
    require(static_cast<bool>(stream), "failed reading external embeddings");
    for (std::size_t row = 0; row < token_count; ++row) {
        normalize_in_place(std::span<float>(values).subspan(row * dimension, dimension));
    }
    return values;
}

std::vector<std::byte> quantize_file(std::span<const float> values,
                                     std::size_t token_count,
                                     std::size_t dimension,
                                     std::uint64_t seed) {
    const std::size_t row_stride = sizeof(float) + dimension;
    require(token_count <= (std::numeric_limits<std::size_t>::max() - sizeof(EmbeddingHeader)) / row_stride,
            "embedding file size overflow");
    std::vector<std::byte> result(sizeof(EmbeddingHeader) + token_count * row_stride);
    auto* payload = result.data() + sizeof(EmbeddingHeader);

    for (std::size_t row = 0; row < token_count; ++row) {
        const auto vector = values.subspan(row * dimension, dimension);
        float maximum = 0.0F;
        for (const float value : vector) maximum = std::max(maximum, std::abs(value));
        const float scale = maximum > 1.0e-12F ? maximum / 127.0F : 1.0F;
        std::memcpy(payload + row * row_stride, &scale, sizeof(scale));
        auto* quantized = reinterpret_cast<std::int8_t*>(payload + row * row_stride + sizeof(scale));
        for (std::size_t column = 0; column < dimension; ++column) {
            const float normalized = vector[column] / scale;
            const auto rounded = static_cast<int>(std::nearbyint(normalized));
            quantized[column] = static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
        }
    }

    EmbeddingHeader header{};
    std::memcpy(header.magic, "MRDLEMB", 7U);
    header.version = 1U;
    header.storage_kind = 1U;
    header.token_count = token_count;
    header.dimension = static_cast<std::uint32_t>(dimension);
    header.row_stride = static_cast<std::uint32_t>(row_stride);
    header.payload_hash = hash_bytes(std::span<const std::byte>(payload, token_count * row_stride));
    header.seed = seed;
    std::memcpy(result.data(), &header, sizeof(header));
    return result;
}

}  // namespace

FrozenEmbeddingStore::~FrozenEmbeddingStore() { close(); }

FrozenEmbeddingStore::FrozenEmbeddingStore(FrozenEmbeddingStore&& other) noexcept {
    *this = std::move(other);
}

FrozenEmbeddingStore& FrozenEmbeddingStore::operator=(FrozenEmbeddingStore&& other) noexcept {
    if (this == &other) return *this;
    close();
    path_ = std::move(other.path_);
    fd_ = std::exchange(other.fd_, -1);
    mapping_ = std::exchange(other.mapping_, nullptr);
    mapping_size_ = std::exchange(other.mapping_size_, 0U);
    token_count_ = std::exchange(other.token_count_, 0U);
    dimension_ = std::exchange(other.dimension_, 0U);
    row_stride_ = std::exchange(other.row_stride_, 0U);
    rows_ = std::exchange(other.rows_, nullptr);
    content_hash_ = std::exchange(other.content_hash_, 0U);
    return *this;
}

void FrozenEmbeddingStore::close() noexcept {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, mapping_size_);
        mapping_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    rows_ = nullptr;
    mapping_size_ = 0U;
    token_count_ = 0U;
    dimension_ = 0U;
    row_stride_ = 0U;
}

FrozenEmbeddingStore FrozenEmbeddingStore::load(const std::filesystem::path& path) {
    FrozenEmbeddingStore store;
    store.path_ = path;
    store.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (store.fd_ < 0) throw Error("cannot open embeddings: " + path.string() + ": " + std::strerror(errno));

    struct stat status {};
    if (::fstat(store.fd_, &status) != 0) throw Error("cannot stat embeddings: " + std::string(std::strerror(errno)));
    require(status.st_size >= static_cast<off_t>(sizeof(EmbeddingHeader)), "embedding file too small");
    store.mapping_size_ = static_cast<std::size_t>(status.st_size);
    store.mapping_ = ::mmap(nullptr, store.mapping_size_, PROT_READ, MAP_SHARED, store.fd_, 0);
    if (store.mapping_ == MAP_FAILED) {
        store.mapping_ = nullptr;
        throw Error("cannot mmap embeddings: " + std::string(std::strerror(errno)));
    }

    const auto* header = static_cast<const EmbeddingHeader*>(store.mapping_);
    require(std::memcmp(header->magic, "MRDLEMB", 7U) == 0, "bad embedding magic");
    require(header->version == 1U && header->storage_kind == 1U, "unsupported embedding format");
    store.token_count_ = static_cast<std::size_t>(header->token_count);
    store.dimension_ = header->dimension;
    store.row_stride_ = header->row_stride;
    require(store.row_stride_ == sizeof(float) + store.dimension_, "invalid embedding row stride");
    require(sizeof(EmbeddingHeader) + store.token_count_ * store.row_stride_ == store.mapping_size_,
            "embedding file length mismatch");
    store.rows_ = static_cast<const std::byte*>(store.mapping_) + sizeof(EmbeddingHeader);
    const auto payload = std::span<const std::byte>(store.rows_, store.token_count_ * store.row_stride_);
    require(hash_bytes(payload) == header->payload_hash, "embedding checksum mismatch");
    store.content_hash_ = header->payload_hash;
    return store;
}

void FrozenEmbeddingStore::build(const std::filesystem::path& output,
                                 const HybridTokenizer& tokenizer,
                                 const std::optional<std::filesystem::path>& corpus,
                                 const EmbeddingBuildOptions& options) {
    require(options.dimension >= 16U, "embedding dimension too small");
    std::vector<float> values;
    switch (options.mode) {
        case EmbeddingInit::Random:
            values = random_embeddings(tokenizer.size(), options.dimension, options.seed);
            break;
        case EmbeddingInit::RandomIndexing:
            require(corpus.has_value(), "random-indexing embeddings require a corpus");
            values = random_indexing_embeddings(tokenizer, *corpus, options);
            break;
        case EmbeddingInit::ExternalFloat32:
            require(options.external_f32.has_value(), "external embedding mode requires a file");
            values = external_embeddings(*options.external_f32, tokenizer.size(), options.dimension);
            break;
    }
    const auto file = quantize_file(values, tokenizer.size(), options.dimension, options.seed);
    atomic_write(output, file);
}

const std::byte* FrozenEmbeddingStore::row(TokenId id) const {
    require(id < token_count_, "embedding token id out of range");
    return rows_ + static_cast<std::size_t>(id) * row_stride_;
}

void FrozenEmbeddingStore::dequantize(TokenId id, std::span<float> output) const {
    require(output.size() == dimension_, "embedding output dimension mismatch");
    const auto* base = row(id);
    float scale = 1.0F;
    std::memcpy(&scale, base, sizeof(scale));
    const auto* values = reinterpret_cast<const std::int8_t*>(base + sizeof(scale));
    for (std::size_t i = 0; i < dimension_; ++i) output[i] = scale * static_cast<float>(values[i]);
}

float FrozenEmbeddingStore::dot_row(TokenId id, std::span<const float> vector) const {
    require(vector.size() == dimension_, "embedding dot dimension mismatch");
    const auto* base = row(id);
    float scale = 1.0F;
    std::memcpy(&scale, base, sizeof(scale));
    const auto* values = reinterpret_cast<const std::int8_t*>(base + sizeof(scale));
    float sum = 0.0F;
    for (std::size_t i = 0; i < dimension_; ++i) {
        sum = std::fma(static_cast<float>(values[i]), vector[i], sum);
    }
    return scale * sum;
}

float FrozenEmbeddingStore::cosine_row(TokenId id, std::span<const float> vector) const {
    const auto* base = row(id);
    float scale = 1.0F;
    std::memcpy(&scale, base, sizeof(scale));
    const auto* values = reinterpret_cast<const std::int8_t*>(base + sizeof(scale));
    float q_norm = 0.0F;
    float v_norm = 0.0F;
    float product = 0.0F;
    for (std::size_t i = 0; i < dimension_; ++i) {
        const float q = static_cast<float>(values[i]);
        product = std::fma(q, vector[i], product);
        q_norm = std::fma(q, q, q_norm);
        v_norm = std::fma(vector[i], vector[i], v_norm);
    }
    const float denominator = std::sqrt(q_norm * v_norm);
    (void)scale;  // Symmetric row scale cancels in cosine.
    return denominator > 1.0e-12F ? product / denominator : 0.0F;
}

}  // namespace mrdl
