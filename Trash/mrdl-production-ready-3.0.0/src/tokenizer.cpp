#include "mrdl/tokenizer.hpp"

#include <cctype>
#include <queue>

namespace mrdl {
namespace {

struct TokenizerHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t flags;
    std::uint64_t token_count;
    std::uint64_t payload_hash;
};
static_assert(std::is_trivially_copyable_v<TokenizerHeader>);

bool is_space(unsigned char c) noexcept {
    return std::isspace(c) != 0;
}

int piece_class(unsigned char c) noexcept {
    if (is_space(c)) return 0;
    if (std::isalnum(c) != 0 || c == '_' || c >= 0x80U) return 1;
    return 2;
}

std::string ascii_lower(std::string_view text) {
    std::string result(text);
    for (char& c : result) {
        const auto uc = static_cast<unsigned char>(c);
        if (uc < 0x80U) c = static_cast<char>(std::tolower(uc));
    }
    return result;
}

class SpaceSaving final {
public:
    explicit SpaceSaving(std::size_t capacity) : capacity_(std::max<std::size_t>(capacity, 1U)) {}

    void observe(const std::string& token) {
        auto it = entries_.find(token);
        if (it != entries_.end()) {
            ++it->second.count;
            heap_.push(HeapItem{it->second.count, token});
            maybe_rebuild_heap();
            return;
        }
        if (entries_.size() < capacity_) {
            entries_.emplace(token, Entry{1U, 0U});
            heap_.push(HeapItem{1U, token});
            return;
        }
        clean_heap();
        require(!heap_.empty(), "space-saving heap unexpectedly empty");
        const auto minimum = heap_.top();
        heap_.pop();
        const auto old = entries_.find(minimum.token);
        require(old != entries_.end(), "space-saving index corruption");
        const std::uint64_t replaced_count = old->second.count;
        entries_.erase(old);
        entries_.emplace(token, Entry{replaced_count + 1U, replaced_count});
        heap_.push(HeapItem{replaced_count + 1U, token});
        maybe_rebuild_heap();
    }

    [[nodiscard]] std::unordered_set<std::string> candidates() const {
        std::unordered_set<std::string> result;
        result.reserve(entries_.size());
        for (const auto& [token, _] : entries_) result.insert(token);
        return result;
    }

private:
    struct Entry { std::uint64_t count; std::uint64_t error; };
    struct HeapItem { std::uint64_t count; std::string token; };
    struct Greater {
        bool operator()(const HeapItem& lhs, const HeapItem& rhs) const noexcept {
            return lhs.count > rhs.count;
        }
    };

    void clean_heap() {
        while (!heap_.empty()) {
            const auto& top = heap_.top();
            const auto it = entries_.find(top.token);
            if (it != entries_.end() && it->second.count == top.count) break;
            heap_.pop();
        }
    }

    void maybe_rebuild_heap() {
        if (heap_.size() <= capacity_ * 8U) return;
        std::priority_queue<HeapItem, std::vector<HeapItem>, Greater> rebuilt;
        for (const auto& [token, entry] : entries_) rebuilt.push(HeapItem{entry.count, token});
        heap_.swap(rebuilt);
    }

    std::size_t capacity_;
    std::unordered_map<std::string, Entry> entries_;
    std::priority_queue<HeapItem, std::vector<HeapItem>, Greater> heap_;
};

void atomic_write(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp." + std::to_string(unix_millis());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw Error("cannot create temporary file: " + temporary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw Error("failed writing temporary file: " + temporary);
        stream.flush();
        if (!stream) throw Error("failed flushing temporary file: " + temporary);
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
        throw Error("cannot atomically replace " + path.string() + ": " + ec.message());
    }
}

}  // namespace

std::vector<std::string> HybridTokenizer::split_pieces(std::string_view input, bool lowercase) {
    const std::string owned = lowercase ? ascii_lower(input) : std::string(input);
    std::vector<std::string> pieces;
    std::size_t start = 0;
    while (start < owned.size()) {
        const int cls = piece_class(static_cast<unsigned char>(owned[start]));
        std::size_t end = start + 1U;
        while (end < owned.size() && piece_class(static_cast<unsigned char>(owned[end])) == cls) {
            // Punctuation stays granular enough to preserve composition/closure signals.
            if (cls == 2 && owned[end] != owned[start]) break;
            ++end;
        }
        pieces.emplace_back(owned.substr(start, end - start));
        start = end;
    }
    return pieces;
}

HybridTokenizer HybridTokenizer::build_from_corpus(const std::filesystem::path& corpus,
                                                    const TokenizerConfig& config) {
    require(config.vocab_size > kFirstLearnedToken, "vocab too small");
    const std::size_t learned_budget = static_cast<std::size_t>(config.vocab_size - kFirstLearnedToken);
    const std::size_t candidate_capacity = learned_budget * std::max<std::uint32_t>(config.heavy_hitter_multiplier, 1U);
    SpaceSaving sketch(candidate_capacity);

    {
        std::ifstream stream(corpus);
        if (!stream) throw Error("cannot open corpus: " + corpus.string());
        std::string line;
        while (std::getline(stream, line)) {
            for (const auto& piece : split_pieces(line, config.lowercase)) sketch.observe(piece);
            sketch.observe("\n");
        }
    }

    const auto candidates = sketch.candidates();
    std::unordered_map<std::string, std::uint64_t> exact;
    exact.reserve(candidates.size());
    {
        std::ifstream stream(corpus);
        if (!stream) throw Error("cannot reopen corpus: " + corpus.string());
        std::string line;
        while (std::getline(stream, line)) {
            for (const auto& piece : split_pieces(line, config.lowercase)) {
                if (candidates.contains(piece)) ++exact[piece];
            }
            if (candidates.contains("\n")) ++exact["\n"];
        }
    }

    std::vector<std::pair<std::string, std::uint64_t>> ranked;
    ranked.reserve(exact.size());
    for (auto& [piece, count] : exact) ranked.emplace_back(std::move(piece), count);
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) return lhs.second > rhs.second;
        return lhs.first < rhs.first;
    });
    if (ranked.size() > learned_budget) ranked.resize(learned_budget);

    HybridTokenizer tokenizer;
    tokenizer.lowercase_ = config.lowercase;
    tokenizer.id_to_piece_.reserve(static_cast<std::size_t>(kFirstLearnedToken) + ranked.size());
    tokenizer.id_to_piece_.push_back("<PAD>");
    tokenizer.id_to_piece_.push_back("<BOS>");
    tokenizer.id_to_piece_.push_back("<EOS>");
    tokenizer.id_to_piece_.push_back("<UNK>");
    for (std::size_t value = 0; value < kByteTokenCount; ++value) {
        tokenizer.id_to_piece_.emplace_back(1, static_cast<char>(value));
    }
    for (auto& [piece, _] : ranked) tokenizer.id_to_piece_.push_back(std::move(piece));
    tokenizer.rebuild_index();
    return tokenizer;
}

void HybridTokenizer::rebuild_index() {
    piece_to_id_.clear();
    piece_to_id_.reserve(id_to_piece_.size());
    for (std::size_t i = kFirstLearnedToken; i < id_to_piece_.size(); ++i) {
        piece_to_id_.emplace(id_to_piece_[i], static_cast<TokenId>(i));
    }
}

std::vector<TokenId> HybridTokenizer::encode(std::string_view text, bool add_bos, bool add_eos) const {
    require(!id_to_piece_.empty(), "tokenizer is not initialized");
    std::vector<TokenId> result;
    result.reserve(text.size() / 3U + 4U);
    if (add_bos) result.push_back(kBosToken);
    for (const auto& piece : split_pieces(text, lowercase_)) {
        const auto it = piece_to_id_.find(piece);
        if (it != piece_to_id_.end()) {
            result.push_back(it->second);
            continue;
        }
        for (const char value : piece) {
            const auto byte = static_cast<unsigned char>(value);
            result.push_back(kByteTokenBase + static_cast<TokenId>(byte));
        }
    }
    if (add_eos) result.push_back(kEosToken);
    return result;
}

std::string HybridTokenizer::decode(std::span<const TokenId> ids, bool skip_special) const {
    std::string result;
    for (const TokenId id : ids) {
        if (id >= id_to_piece_.size()) {
            if (!skip_special) result.append("<BAD_TOKEN>");
            continue;
        }
        if (id <= kUnkToken) {
            if (!skip_special) result.append(id_to_piece_[id]);
            continue;
        }
        result.append(id_to_piece_[id]);
    }
    return result;
}

std::string_view HybridTokenizer::token(TokenId id) const {
    require(id < id_to_piece_.size(), "token id out of range");
    return id_to_piece_[id];
}

void HybridTokenizer::save(const std::filesystem::path& path) const {
    require(!id_to_piece_.empty(), "cannot save empty tokenizer");
    BinaryWriter payload;
    for (const auto& piece : id_to_piece_) payload.string(piece);
    const auto& body = payload.data();

    TokenizerHeader header{};
    std::memcpy(header.magic, "MRDLTOK", 7U);
    header.version = 1U;
    header.flags = lowercase_ ? 1U : 0U;
    header.token_count = id_to_piece_.size();
    header.payload_hash = hash_bytes(body);

    std::vector<std::byte> file(sizeof(header) + body.size());
    std::memcpy(file.data(), &header, sizeof(header));
    if (!body.empty()) std::memcpy(file.data() + sizeof(header), body.data(), body.size());
    atomic_write(path, file);
}

HybridTokenizer HybridTokenizer::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw Error("cannot open tokenizer: " + path.string());
    const auto end = stream.tellg();
    require(end >= static_cast<std::streamoff>(sizeof(TokenizerHeader)), "tokenizer file too small");
    std::vector<std::byte> file(static_cast<std::size_t>(end));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file.size()));
    require(static_cast<bool>(stream), "failed reading tokenizer");

    TokenizerHeader header{};
    std::memcpy(&header, file.data(), sizeof(header));
    require(std::memcmp(header.magic, "MRDLTOK", 7U) == 0, "bad tokenizer magic");
    require(header.version == 1U, "unsupported tokenizer version");
    const std::span<const std::byte> payload(file.data() + sizeof(header), file.size() - sizeof(header));
    require(hash_bytes(payload) == header.payload_hash, "tokenizer checksum mismatch");

    BinaryReader reader(payload);
    HybridTokenizer tokenizer;
    tokenizer.lowercase_ = (header.flags & 1U) != 0U;
    tokenizer.id_to_piece_.reserve(static_cast<std::size_t>(header.token_count));
    for (std::uint64_t i = 0; i < header.token_count; ++i) tokenizer.id_to_piece_.push_back(reader.string());
    require(reader.empty(), "trailing tokenizer payload");
    require(tokenizer.id_to_piece_.size() >= kFirstLearnedToken, "tokenizer missing byte fallback tokens");
    tokenizer.rebuild_index();
    return tokenizer;
}

CorpusTokenStream::CorpusTokenStream(const std::filesystem::path& corpus,
                                     const HybridTokenizer& tokenizer,
                                     bool add_boundaries)
    : stream_(corpus), tokenizer_(tokenizer), add_boundaries_(add_boundaries) {
    if (!stream_) throw Error("cannot open corpus: " + corpus.string());
}

bool CorpusTokenStream::next(TokenId& token) {
    while (offset_ >= buffer_.size()) {
        std::string line;
        if (!std::getline(stream_, line)) return false;
        ++line_number_;
        line.push_back('\n');
        buffer_ = tokenizer_.encode(line, add_boundaries_, add_boundaries_);
        offset_ = 0;
    }
    token = buffer_[offset_++];
    return true;
}

}  // namespace mrdl
