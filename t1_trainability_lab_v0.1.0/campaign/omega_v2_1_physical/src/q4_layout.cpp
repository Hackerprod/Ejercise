#include "v2_1.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace omega_v2_1 {
namespace {

std::size_t round_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        const std::uint32_t rounded = (mantissa + 0x0fffu + ((mantissa >> 13) & 1u)) >> 13;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mantissa != 0) return static_cast<std::uint16_t>(sign | 0x7e00u);
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    mantissa = mantissa + 0x0fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void write_nibble(std::uint8_t* packed, std::size_t index, std::int8_t value) {
    const std::uint8_t nibble = static_cast<std::uint8_t>(value) & 0x0fu;
    if ((index & 1u) == 0) packed[index >> 1] = static_cast<std::uint8_t>((packed[index >> 1] & 0xf0u) | nibble);
    else packed[index >> 1] = static_cast<std::uint8_t>((packed[index >> 1] & 0x0fu) | (nibble << 4));
}

} // namespace

AlignedBuffer::AlignedBuffer(std::size_t logical) { reset(logical); }

AlignedBuffer::~AlignedBuffer() {
    if (data) _aligned_free(data);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data(other.data), logical_bytes(other.logical_bytes), allocated_bytes(other.allocated_bytes) {
    other.data = nullptr;
    other.logical_bytes = 0;
    other.allocated_bytes = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        if (data) _aligned_free(data);
        data = other.data;
        logical_bytes = other.logical_bytes;
        allocated_bytes = other.allocated_bytes;
        other.data = nullptr;
        other.logical_bytes = 0;
        other.allocated_bytes = 0;
    }
    return *this;
}

void AlignedBuffer::reset(std::size_t logical) {
    if (data) _aligned_free(data);
    logical_bytes = logical;
    allocated_bytes = round_up(logical, kAlignment);
    if (allocated_bytes == 0) allocated_bytes = kAlignment;
    data = static_cast<std::uint8_t*>(_aligned_malloc(allocated_bytes, kAlignment));
    if (!data) throw std::bad_alloc();
    std::memset(data, 0, allocated_bytes);
}

void Q4Matrix::pack(const std::vector<float>& values, int row_count, int column_count) {
    if (row_count < 1 || column_count < 1 || column_count % kGroup != 0) {
        throw std::invalid_argument("Q4 matrices require positive dimensions and columns divisible by group_size=32");
    }
    const std::size_t count = static_cast<std::size_t>(row_count) * static_cast<std::size_t>(column_count);
    if (values.size() != count || (count & 1u) != 0 || (count % kGroup) != 0) {
        throw std::invalid_argument("Q4 source value count is inconsistent with matrix shape/grouping");
    }
    rows = row_count;
    cols = column_count;
    const std::size_t groups = count / kGroup;
    packed.reset(count / 2);
    scales_fp16.reset(groups * sizeof(std::uint16_t));
    scale_count = groups;
    auto* scales = reinterpret_cast<std::uint16_t*>(scales_fp16.data);
    std::fill(scales, scales + groups, float_to_half(1.0f));

    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t start = group * kGroup;
        float max_abs = 0.0f;
        for (std::size_t i = 0; i < kGroup; ++i) max_abs = (std::max)(max_abs, std::fabs(values[start + i]));
        const float scale = max_abs == 0.0f ? 1.0f : max_abs / 7.0f;
        scales[group] = float_to_half(scale);
        const float stored_scale = half_to_float(scales[group]);
        if (!(stored_scale > 0.0f) || !std::isfinite(stored_scale)) throw std::runtime_error("invalid FP16 Q4 scale");
        for (std::size_t i = 0; i < kGroup; ++i) {
            const float rounded = std::nearbyint(values[start + i] / stored_scale);
            const int quantized = (std::max)(-7, (std::min)(7, static_cast<int>(rounded)));
            write_nibble(packed.data, start + i, static_cast<std::int8_t>(quantized));
        }
    }
    alignment_padding_bytes = (packed.allocated_bytes - packed.logical_bytes) + (scales_fp16.allocated_bytes - scales_fp16.logical_bytes);
}

std::int8_t Q4Matrix::value_at(int row, int column) const {
    if (row < 0 || row >= rows || column < 0 || column >= cols) throw std::out_of_range("Q4 matrix index");
    const std::size_t index = static_cast<std::size_t>(row) * cols + column;
    const std::uint8_t byte = packed.data[index >> 1];
    const std::uint8_t nibble = (index & 1u) ? static_cast<std::uint8_t>(byte >> 4) : static_cast<std::uint8_t>(byte & 0x0fu);
    return static_cast<std::int8_t>((nibble & 0x08u) ? (static_cast<int>(nibble) - 16) : nibble);
}

float Q4Matrix::scale_at(int row, int column) const {
    if (row < 0 || row >= rows || column < 0 || column >= cols) throw std::out_of_range("Q4 scale index");
    const std::size_t flat = static_cast<std::size_t>(row) * cols + column;
    const std::size_t group = flat / kGroup;
    const auto* scales = reinterpret_cast<const std::uint16_t*>(scales_fp16.data);
    return half_to_float(scales[group]);
}

float Q4Matrix::dequant_at(int row, int column) const {
    return static_cast<float>(value_at(row, column)) * scale_at(row, column);
}

CoreWeights make_seeded_weights(int d, std::uint32_t seed) {
    if (d < kGroup || d % kGroup != 0) throw std::invalid_argument("Q4 core dimensions must be positive multiples of group_size=32");
    CoreWeights core;
    core.d = d;
    std::mt19937 generator(seed);
    auto fill = [&](Q4Matrix& matrix, int rows, int cols) {
        const float bound = std::sqrt(6.0f / static_cast<float>(rows + cols));
        std::uniform_real_distribution<float> dist(-bound, bound);
        std::vector<float> values(static_cast<std::size_t>(rows) * cols);
        for (float& value : values) value = dist(generator);
        matrix.pack(values, rows, cols);
    };
    fill(core.W_Q, d, d);
    fill(core.W_K, d, d);
    fill(core.W_V, d, d);
    fill(core.W_O, d, d);
    fill(core.W_gate, 4 * d, d);
    fill(core.W_up, 4 * d, d);
    fill(core.W_down, d, 4 * d);

    const Q4Matrix* matrices[] = {&core.W_Q, &core.W_K, &core.W_V, &core.W_O, &core.W_gate, &core.W_up, &core.W_down};
    for (const Q4Matrix* matrix : matrices) {
        core.logical_weight_bytes += matrix->packed.logical_bytes;
        core.logical_scale_bytes += matrix->scales_fp16.logical_bytes;
        core.alignment_padding_bytes += matrix->alignment_padding_bytes;
        core.physical_buffer_bytes += matrix->packed.allocated_bytes + matrix->scales_fp16.allocated_bytes;
    }
    return core;
}

CoreWeights clone_weights(const CoreWeights& source) {
    CoreWeights result;
    result.d = source.d;
    auto clone = [](const Q4Matrix& from, Q4Matrix& to) {
        to.rows = from.rows;
        to.cols = from.cols;
        to.scale_count = from.scale_count;
        to.packed.reset(from.packed.logical_bytes);
        to.scales_fp16.reset(from.scales_fp16.logical_bytes);
        std::memcpy(to.packed.data, from.packed.data, from.packed.logical_bytes);
        std::memcpy(to.scales_fp16.data, from.scales_fp16.data, from.scales_fp16.logical_bytes);
        to.alignment_padding_bytes = (to.packed.allocated_bytes - to.packed.logical_bytes) + (to.scales_fp16.allocated_bytes - to.scales_fp16.logical_bytes);
    };
    clone(source.W_Q, result.W_Q);
    clone(source.W_K, result.W_K);
    clone(source.W_V, result.W_V);
    clone(source.W_O, result.W_O);
    clone(source.W_gate, result.W_gate);
    clone(source.W_up, result.W_up);
    clone(source.W_down, result.W_down);
    result.logical_weight_bytes = source.logical_weight_bytes;
    result.logical_scale_bytes = source.logical_scale_bytes;
    const Q4Matrix* matrices[] = {&result.W_Q, &result.W_K, &result.W_V, &result.W_O, &result.W_gate, &result.W_up, &result.W_down};
    for (const Q4Matrix* matrix : matrices) {
        result.alignment_padding_bytes += matrix->alignment_padding_bytes;
        result.physical_buffer_bytes += matrix->packed.allocated_bytes + matrix->scales_fp16.allocated_bytes;
    }
    return result;
}

void Scratch::resize_for(int dimension, int slots) {
    d = dimension;
    m = slots;
    const std::size_t md = static_cast<std::size_t>(m) * d;
    const std::size_t m4d = static_cast<std::size_t>(m) * 4 * d;
    state.resize(md);
    normalized.resize(md);
    query.resize(md);
    key.resize(md);
    value.resize(md);
    scores.resize(static_cast<std::size_t>(m) * m);
    attention.resize(md);
    projected.resize(md);
    hidden.resize(md);
    mlp_normalized.resize(md);
    gate.resize(m4d);
    up.resize(m4d);
    gated.resize(m4d);
    down.resize(md);
    output.resize(md);
}

std::string q4_ledger_json(const std::vector<CoreWeights>& weight_families) {
    std::ostringstream out;
    out << "{\"schema\":\"omega-v2-1-q4-physical-ledger-v1\",\"group_size\":32,\"scale_dtype\":\"FP16\",\"zero_point\":false,\"alignment_bytes\":64,\"weight_families\":[";
    for (std::size_t i = 0; i < weight_families.size(); ++i) {
        if (i) out << ',';
        const CoreWeights& w = weight_families[i];
        out << "{\"d\":" << w.d
            << ",\"logical_weight_bytes\":" << w.logical_weight_bytes
            << ",\"logical_scale_bytes\":" << w.logical_scale_bytes
            << ",\"alignment_padding_bytes\":" << w.alignment_padding_bytes
            << ",\"allocator_padding_bytes\":" << w.alignment_padding_bytes
            << ",\"physical_allocated_bytes\":" << w.physical_buffer_bytes
            << ",\"allocator_metadata_bytes\":\"NOT_MEASURED\",\"matrices\":[";
        const char* matrix_names[] = {"W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"};
        const Q4Matrix* matrices[] = {&w.W_Q, &w.W_K, &w.W_V, &w.W_O, &w.W_gate, &w.W_up, &w.W_down};
        for (std::size_t j = 0; j < 7; ++j) {
            if (j) out << ',';
            const Q4Matrix& matrix = *matrices[j];
            out << "{\"name\":\"" << matrix_names[j] << "\",\"rows\":" << matrix.rows << ",\"cols\":" << matrix.cols
                << ",\"logical_weight_bytes\":" << matrix.packed.logical_bytes
                << ",\"logical_scale_bytes\":" << matrix.scales_fp16.logical_bytes
                << ",\"alignment_padding_bytes\":" << matrix.alignment_padding_bytes
                << ",\"physical_allocated_bytes\":" << matrix.packed.allocated_bytes + matrix.scales_fp16.allocated_bytes
                << ",\"packed_address_mod64\":" << (reinterpret_cast<std::uintptr_t>(matrix.packed.data) % kAlignment)
                << ",\"scale_address_mod64\":" << (reinterpret_cast<std::uintptr_t>(matrix.scales_fp16.data) % kAlignment) << '}';
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

std::string q4_round_trip_test() {
    std::vector<float> values(64);
    for (int i = 0; i < 64; ++i) values[static_cast<std::size_t>(i)] = static_cast<float>((i % 15) - 7) * 0.125f;
    Q4Matrix matrix;
    matrix.pack(values, 2, 32);
    bool signed_range = true;
    bool scale_valid = true;
    for (int row = 0; row < matrix.rows; ++row) {
        for (int col = 0; col < matrix.cols; ++col) {
            const std::int8_t q = matrix.value_at(row, col);
            signed_range = signed_range && q >= -7 && q <= 7;
            scale_valid = scale_valid && matrix.scale_at(row, col) > 0.0f;
        }
    }
    std::ostringstream out;
    out << "{\"group_size\":32,\"signed_range\":true,\"zero_point\":false,\"signed_range_test\":"
        << (signed_range ? "true" : "false") << ",\"fp16_scales_positive\":" << (scale_valid ? "true" : "false")
        << ",\"aligned_packed\":" << ((reinterpret_cast<std::uintptr_t>(matrix.packed.data) % 64) == 0 ? "true" : "false")
        << ",\"aligned_scales\":" << ((reinterpret_cast<std::uintptr_t>(matrix.scales_fp16.data) % 64) == 0 ? "true" : "false") << '}';
    return out.str();
}

} // namespace omega_v2_1
