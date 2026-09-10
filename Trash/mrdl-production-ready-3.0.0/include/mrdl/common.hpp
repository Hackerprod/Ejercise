#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace mrdl {

static_assert(std::endian::native == std::endian::little,
              "MRDL persisted formats require a little-endian platform");
static_assert(sizeof(std::size_t) == 8U, "MRDL requires a 64-bit userspace");
static_assert(sizeof(float) == 4U && std::numeric_limits<float>::is_iec559,
              "MRDL requires IEEE-754 binary32 float");
static_assert(sizeof(bool) == 1U, "MRDL persisted booleans require one-byte bool");

using TokenId = std::uint32_t;
using NodeId = std::uint32_t;
using RelationId = std::uint64_t;
using BranchId = std::uint64_t;
using ReplayId = std::uint64_t;
using RoleId = std::uint32_t;
using Clock = std::chrono::system_clock;
using TimePoint = Clock::time_point;

constexpr TokenId kPadToken = 0;
constexpr TokenId kBosToken = 1;
constexpr TokenId kEosToken = 2;
constexpr TokenId kUnkToken = 3;
constexpr TokenId kByteTokenBase = 4;
constexpr std::size_t kByteTokenCount = 256;
constexpr TokenId kFirstLearnedToken = kByteTokenBase + static_cast<TokenId>(kByteTokenCount);

class Error final : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw Error(std::string(message));
    }
}

inline std::int64_t unix_millis(TimePoint point = Clock::now()) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(point.time_since_epoch()).count();
}

inline TimePoint from_unix_millis(std::int64_t value) {
    return TimePoint{std::chrono::milliseconds(value)};
}

enum class Lane : std::uint8_t { Full = 0, Clean = 1 };
enum class MemoryLevel : std::uint8_t { M0 = 0, M1 = 1, M2 = 2 };
enum class GateDecision : std::uint8_t { Compose = 0, Reject = 1, Defer = 2 };
enum class Certification : std::uint8_t { Clean = 0, Provisional = 1, Fragile = 2, Empty = 3 };
enum class EscrowState : std::uint8_t {
    Active = 0,
    AuditReserved = 1,
    Auditing = 2,
    Promoted = 3,
    Rejected = 4,
    Expired = 5,
    Unreplayable = 6
};

inline std::string_view to_string(Lane lane) {
    return lane == Lane::Full ? "FULL" : "CLEAN";
}
inline std::string_view to_string(MemoryLevel level) {
    switch (level) {
        case MemoryLevel::M0: return "M0";
        case MemoryLevel::M1: return "M1";
        case MemoryLevel::M2: return "M2";
    }
    return "UNKNOWN";
}
inline std::string_view to_string(Certification value) {
    switch (value) {
        case Certification::Clean: return "clean";
        case Certification::Provisional: return "provisional";
        case Certification::Fragile: return "fragile";
        case Certification::Empty: return "empty";
    }
    return "unknown";
}
inline std::string_view to_string(EscrowState value) {
    switch (value) {
        case EscrowState::Active: return "ACTIVE";
        case EscrowState::AuditReserved: return "AUDIT_RESERVED";
        case EscrowState::Auditing: return "AUDITING";
        case EscrowState::Promoted: return "PROMOTED";
        case EscrowState::Rejected: return "REJECTED";
        case EscrowState::Expired: return "EXPIRED";
        case EscrowState::Unreplayable: return "UNREPLAYABLE";
    }
    return "UNKNOWN";
}

struct LaneMask {
    bool participates_in_full{true};
    bool participates_in_clean{false};

    [[nodiscard]] bool participates(Lane lane) const noexcept {
        return lane == Lane::Full ? participates_in_full : participates_in_clean;
    }

    static LaneMask from_level(MemoryLevel level) noexcept {
        return LaneMask{true, level == MemoryLevel::M2};
    }
};

inline std::uint64_t mix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

inline std::uint64_t hash_combine(std::uint64_t seed, std::uint64_t value) noexcept {
    return mix64(seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U)));
}

inline std::uint64_t hash_bytes(std::span<const std::byte> bytes, std::uint64_t seed = 1469598103934665603ULL) noexcept {
    std::uint64_t h = seed;
    for (const auto byte : bytes) {
        h ^= static_cast<std::uint8_t>(byte);
        h *= 1099511628211ULL;
    }
    return mix64(h);
}

inline std::uint64_t hash_floats(std::span<const float> values, std::uint64_t seed = 0) noexcept {
    return hash_bytes(std::as_bytes(values), seed);
}

inline float clamp_finite(float value, float low, float high, float fallback = 0.0F) noexcept {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, low, high);
}

inline float sigmoid(float x) noexcept {
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(x);
    return z / (1.0F + z);
}

inline float safe_logit(float probability) noexcept {
    const float p = std::clamp(probability, 1.0e-6F, 1.0F - 1.0e-6F);
    return std::log(p / (1.0F - p));
}

inline float dot(std::span<const float> a, std::span<const float> b) {
    require(a.size() == b.size(), "dot: dimension mismatch");
    float sum = 0.0F;
    std::size_t i = 0;
#if defined(__AVX2__)
    // Kept scalar intentionally here; quantized hot paths use specialized kernels.
#endif
    for (; i < a.size(); ++i) sum = std::fma(a[i], b[i], sum);
    return sum;
}

inline float l2_norm(std::span<const float> a) noexcept {
    float sum = 0.0F;
    for (const float v : a) sum = std::fma(v, v, sum);
    return std::sqrt(std::max(sum, 0.0F));
}

inline float cosine(std::span<const float> a, std::span<const float> b) {
    const float denom = l2_norm(a) * l2_norm(b);
    return denom > 1.0e-12F ? dot(a, b) / denom : 0.0F;
}

inline void normalize_in_place(std::span<float> values) noexcept {
    const float norm = l2_norm(values);
    if (norm <= 1.0e-12F) return;
    const float inv = 1.0F / norm;
    for (float& value : values) value *= inv;
}

class BinaryWriter {
public:
    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* begin = reinterpret_cast<const std::byte*>(&value);
        data_.insert(data_.end(), begin, begin + sizeof(T));
    }

    template <typename T>
    void vector(std::span<const T> values) {
        const std::uint64_t size = values.size();
        pod(size);
        if (!values.empty()) {
            const auto bytes = std::as_bytes(values);
            data_.insert(data_.end(), bytes.begin(), bytes.end());
        }
    }

    void string(std::string_view value) {
        const std::uint64_t size = value.size();
        pod(size);
        const auto* begin = reinterpret_cast<const std::byte*>(value.data());
        data_.insert(data_.end(), begin, begin + value.size());
    }

    [[nodiscard]] const std::vector<std::byte>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<std::byte> take() noexcept { return std::move(data_); }

private:
    std::vector<std::byte> data_;
};

class BinaryReader {
public:
    explicit BinaryReader(std::span<const std::byte> data) : data_(data) {}

    template <typename T>
    T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(offset_ + sizeof(T) <= data_.size(), "binary decode overflow");
        T value{};
        std::memcpy(&value, data_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    template <typename T>
    std::vector<T> vector() {
        const auto size = pod<std::uint64_t>();
        require(size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T)), "binary vector too large");
        const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(T);
        require(offset_ + bytes <= data_.size(), "binary vector overflow");
        std::vector<T> result(static_cast<std::size_t>(size));
        if (bytes != 0) std::memcpy(result.data(), data_.data() + offset_, bytes);
        offset_ += bytes;
        return result;
    }

    std::string string() {
        const auto size = pod<std::uint64_t>();
        require(size <= data_.size() - offset_, "binary string overflow");
        std::string result(reinterpret_cast<const char*>(data_.data() + offset_), static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
        return result;
    }

    [[nodiscard]] bool empty() const noexcept { return offset_ == data_.size(); }

private:
    std::span<const std::byte> data_;
    std::size_t offset_{0};
};

class ScopeExit final {
public:
    explicit ScopeExit(std::function<void()> fn) : fn_(std::move(fn)) {}
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&& other) noexcept : fn_(std::move(other.fn_)) { other.fn_ = {}; }
    ~ScopeExit() { if (fn_) fn_(); }
    void dismiss() noexcept { fn_ = {}; }
private:
    std::function<void()> fn_;
};

}  // namespace mrdl
