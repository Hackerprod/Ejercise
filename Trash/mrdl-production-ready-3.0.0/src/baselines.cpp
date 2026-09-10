#include "mrdl/baselines.hpp"

namespace mrdl {
namespace {

struct NGramHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t payload_size;
    std::uint64_t payload_hash;
};
static_assert(std::is_trivially_copyable_v<NGramHeader>);

void atomic_write(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp." + std::to_string(unix_millis());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw Error("cannot create baseline file: " + temporary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        stream.flush();
        if (!stream) throw Error("failed writing baseline file: " + temporary);
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) {
        std::filesystem::remove(temporary);
        throw Error("cannot install baseline file: " + error.message());
    }
}

}  // namespace

NGramBaseline::NGramBaseline(std::uint32_t order, double alpha)
    : order_(order), alpha_(alpha) {
    require(order_ >= 1U && order_ <= 5U, "n-gram order must be 1..5");
    require(std::isfinite(alpha_) && alpha_ > 0.0 && alpha_ <= 10.0, "invalid n-gram smoothing alpha");
}

NGramKey NGramBaseline::key_for(std::span<const TokenId> context, std::size_t length) const {
    require(length <= 4U && length <= context.size(), "invalid n-gram context length");
    NGramKey key;
    key.length = static_cast<std::uint8_t>(length);
    const std::size_t begin = context.size() - length;
    for (std::size_t i = 0; i < length; ++i) key.tokens[i] = context[begin + i];
    return key;
}

void NGramBaseline::observe(std::span<const TokenId> context, TokenId target) {
    ++unigram_[target];
    ++total_observations_;
    const std::size_t maximum = std::min<std::size_t>(order_ - 1U, context.size());
    for (std::size_t length = 1U; length <= maximum; ++length) {
        auto& next = contexts_[key_for(context, length)];
        ++next.total;
        ++next.counts[target];
    }
    top_cache_dirty_ = true;
}

void NGramBaseline::rebuild_top_cache() const {
    if (!top_cache_dirty_) return;
    std::vector<std::pair<TokenId, std::uint64_t>> ranked;
    ranked.reserve(unigram_.size());
    for (const auto& [token, count] : unigram_) ranked.emplace_back(token, count);
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) return lhs.second > rhs.second;
        return lhs.first < rhs.first;
    });
    const std::size_t keep = std::min<std::size_t>(ranked.size(), 256U);
    top_unigrams_.clear();
    top_unigrams_.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) top_unigrams_.push_back(ranked[i].first);
    top_cache_dirty_ = false;
}

double NGramBaseline::probability(std::span<const TokenId> context, TokenId target) const {
    const double vocabulary = static_cast<double>(std::max<std::size_t>(unigram_.size(), 1U));
    const auto unigram = unigram_.find(target);
    const double count = unigram == unigram_.end() ? 0.0 : static_cast<double>(unigram->second);
    double estimate = (count + alpha_) /
        (static_cast<double>(total_observations_) + alpha_ * vocabulary);

    const std::size_t maximum = std::min<std::size_t>(order_ - 1U, context.size());
    for (std::size_t length = 1U; length <= maximum; ++length) {
        const auto row = contexts_.find(key_for(context, length));
        if (row == contexts_.end() || row->second.total == 0U) continue;
        const auto target_count = row->second.counts.find(target);
        const double observed = target_count == row->second.counts.end() ? 0.0 :
            static_cast<double>(target_count->second);
        const double total = static_cast<double>(row->second.total);
        const double distinct = static_cast<double>(row->second.counts.size());
        const double lambda = total / (total + distinct);  // Witten-Bell interpolation.
        const double maximum_likelihood = observed / total;
        estimate = lambda * maximum_likelihood + (1.0 - lambda) * estimate;
    }
    return std::clamp(estimate, 1.0e-12, 1.0);
}

double NGramBaseline::negative_log_likelihood(std::span<const TokenId> context, TokenId target) const {
    return -std::log(probability(context, target));
}

TokenId NGramBaseline::predict(std::span<const TokenId> context) const {
    if (unigram_.empty()) return kEosToken;
    rebuild_top_cache();
    std::unordered_set<TokenId> candidates(top_unigrams_.begin(), top_unigrams_.end());
    const std::size_t maximum = std::min<std::size_t>(order_ - 1U, context.size());
    for (std::size_t length = 1U; length <= maximum; ++length) {
        const auto row = contexts_.find(key_for(context, length));
        if (row == contexts_.end()) continue;
        for (const auto& [token, _] : row->second.counts) candidates.insert(token);
    }
    TokenId best = kEosToken;
    double best_probability = -1.0;
    for (const TokenId token : candidates) {
        const double value = probability(context, token);
        if (value > best_probability || (value == best_probability && token < best)) {
            best = token;
            best_probability = value;
        }
    }
    return best;
}

void NGramBaseline::save(const std::filesystem::path& path) const {
    BinaryWriter writer;
    writer.pod(order_);
    writer.pod(alpha_);
    writer.pod(total_observations_);

    std::vector<std::pair<TokenId, std::uint64_t>> unigrams(unigram_.begin(), unigram_.end());
    std::sort(unigrams.begin(), unigrams.end());
    writer.pod<std::uint64_t>(unigrams.size());
    for (const auto& [token, count] : unigrams) { writer.pod(token); writer.pod(count); }

    std::vector<std::pair<NGramKey, const NextCounts*>> rows;
    rows.reserve(contexts_.size());
    for (const auto& [key, counts] : contexts_) rows.emplace_back(key, &counts);
    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first.length != rhs.first.length) return lhs.first.length < rhs.first.length;
        return lhs.first.tokens < rhs.first.tokens;
    });
    writer.pod<std::uint64_t>(rows.size());
    for (const auto& [key, counts] : rows) {
        writer.pod(key.length);
        for (std::size_t i = 0; i < key.length; ++i) writer.pod(key.tokens[i]);
        writer.pod(counts->total);
        std::vector<std::pair<TokenId, std::uint64_t>> next(counts->counts.begin(), counts->counts.end());
        std::sort(next.begin(), next.end());
        writer.pod<std::uint64_t>(next.size());
        for (const auto& [token, count] : next) { writer.pod(token); writer.pod(count); }
    }
    const auto payload = writer.take();
    NGramHeader header{};
    std::memcpy(header.magic, "MRDLNGR", 7U);
    header.version = 1U;
    header.payload_size = payload.size();
    header.payload_hash = hash_bytes(payload);
    std::vector<std::byte> file(sizeof(header) + payload.size());
    std::memcpy(file.data(), &header, sizeof(header));
    std::memcpy(file.data() + sizeof(header), payload.data(), payload.size());
    atomic_write(path, file);
}

NGramBaseline NGramBaseline::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw Error("cannot open baseline file: " + path.string());
    const auto length = stream.tellg();
    require(length >= static_cast<std::streamoff>(sizeof(NGramHeader)), "baseline file too small");
    std::vector<std::byte> file(static_cast<std::size_t>(length));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(file.data()), length);
    require(static_cast<bool>(stream), "failed reading baseline file");
    NGramHeader header{};
    std::memcpy(&header, file.data(), sizeof(header));
    require(std::memcmp(header.magic, "MRDLNGR", 7U) == 0 && header.version == 1U,
            "unsupported baseline format");
    require(header.payload_size == file.size() - sizeof(header), "baseline length mismatch");
    const auto payload = std::span<const std::byte>(file).subspan(sizeof(header));
    require(hash_bytes(payload) == header.payload_hash, "baseline checksum mismatch");

    BinaryReader reader(payload);
    const auto order = reader.pod<std::uint32_t>();
    const auto alpha = reader.pod<double>();
    NGramBaseline baseline(order, alpha);
    baseline.total_observations_ = reader.pod<std::uint64_t>();
    const auto unigram_count = reader.pod<std::uint64_t>();
    for (std::uint64_t i = 0; i < unigram_count; ++i) {
        const auto token = reader.pod<TokenId>();
        const auto count = reader.pod<std::uint64_t>();
        baseline.unigram_[token] = count;
    }
    const auto row_count = reader.pod<std::uint64_t>();
    for (std::uint64_t row_index = 0; row_index < row_count; ++row_index) {
        NGramKey key;
        key.length = reader.pod<std::uint8_t>();
        require(key.length <= 4U, "corrupt baseline context length");
        for (std::size_t i = 0; i < key.length; ++i) key.tokens[i] = reader.pod<TokenId>();
        NextCounts counts;
        counts.total = reader.pod<std::uint64_t>();
        const auto next_count = reader.pod<std::uint64_t>();
        std::uint64_t verified_total = 0U;
        for (std::uint64_t i = 0; i < next_count; ++i) {
            const TokenId token = reader.pod<TokenId>();
            const auto count_value = reader.pod<std::uint64_t>();
            counts.counts[token] = count_value;
            verified_total += count_value;
        }
        require(verified_total == counts.total, "corrupt baseline row total");
        baseline.contexts_.emplace(key, std::move(counts));
    }
    require(reader.empty(), "trailing baseline payload");
    baseline.top_cache_dirty_ = true;
    return baseline;
}

}  // namespace mrdl
