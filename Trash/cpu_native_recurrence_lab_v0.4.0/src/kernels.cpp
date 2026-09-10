#include "cnrl/kernels.hpp"
#include <limits>
#include <stdexcept>
#include <immintrin.h>

#if defined(_MSC_VER)
#define CNRL_NOINLINE __declspec(noinline)
#else
#define CNRL_NOINLINE __attribute__((noinline))
#endif

namespace cnrl {
namespace {
constexpr std::uint32_t kVec = 16;
constexpr std::uint32_t kMaxD = 65'536;

inline std::int32_t hsum(__m256i value) noexcept {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

inline __m256i load_i8x16_as_i16(const std::int8_t* ptr) noexcept {
  return _mm256_cvtepi8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

CNRL_NOINLINE void fused1(const KernelCall& c, std::uint32_t sb) noexcept {
  for (std::uint32_t row = 0; row < c.rows; ++row) {
    const auto* w = c.weights + static_cast<std::size_t>(row) * c.dimension;
    const auto* x0 = c.state + static_cast<std::size_t>(sb) * c.dimension;
    __m256i a0 = _mm256_setzero_si256();
    std::int32_t t0 = 0;
    std::uint32_t d = 0;
    for (; d + kVec <= c.dimension; d += kVec) {
      const __m256i vw = load_i8x16_as_i16(w + d);
      a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(vw, load_i8x16_as_i16(x0 + d)));
    }
    for (; d < c.dimension; ++d) t0 += static_cast<std::int32_t>(w[d]) * x0[d];
    c.output[static_cast<std::size_t>(sb) * c.output_stride + c.row_offset + row] = hsum(a0) + t0;
  }
}

CNRL_NOINLINE void fused2(const KernelCall& c, std::uint32_t sb) noexcept {
  for (std::uint32_t row = 0; row < c.rows; ++row) {
    const auto* w = c.weights + static_cast<std::size_t>(row) * c.dimension;
    const auto* x0 = c.state + static_cast<std::size_t>(sb + 0U) * c.dimension;
    const auto* x1 = c.state + static_cast<std::size_t>(sb + 1U) * c.dimension;
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    std::int32_t t0 = 0, t1 = 0;
    std::uint32_t d = 0;
    for (; d + kVec <= c.dimension; d += kVec) {
      const __m256i vw = load_i8x16_as_i16(w + d);
      a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(vw, load_i8x16_as_i16(x0 + d)));
      a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(vw, load_i8x16_as_i16(x1 + d)));
    }
    for (; d < c.dimension; ++d) {
      const auto ww = static_cast<std::int32_t>(w[d]);
      t0 += ww * x0[d]; t1 += ww * x1[d];
    }
    const auto out = c.row_offset + row;
    c.output[static_cast<std::size_t>(sb + 0U) * c.output_stride + out] = hsum(a0) + t0;
    c.output[static_cast<std::size_t>(sb + 1U) * c.output_stride + out] = hsum(a1) + t1;
  }
}

CNRL_NOINLINE void fused4(const KernelCall& c, std::uint32_t sb) noexcept {
  for (std::uint32_t row = 0; row < c.rows; ++row) {
    const auto* w = c.weights + static_cast<std::size_t>(row) * c.dimension;
    const auto* x0 = c.state + static_cast<std::size_t>(sb + 0U) * c.dimension;
    const auto* x1 = c.state + static_cast<std::size_t>(sb + 1U) * c.dimension;
    const auto* x2 = c.state + static_cast<std::size_t>(sb + 2U) * c.dimension;
    const auto* x3 = c.state + static_cast<std::size_t>(sb + 3U) * c.dimension;
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    std::int32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    std::uint32_t d = 0;
    for (; d + kVec <= c.dimension; d += kVec) {
      const __m256i vw = load_i8x16_as_i16(w + d);
      a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(vw, load_i8x16_as_i16(x0 + d)));
      a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(vw, load_i8x16_as_i16(x1 + d)));
      a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(vw, load_i8x16_as_i16(x2 + d)));
      a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(vw, load_i8x16_as_i16(x3 + d)));
    }
    for (; d < c.dimension; ++d) {
      const auto ww = static_cast<std::int32_t>(w[d]);
      t0 += ww*x0[d]; t1 += ww*x1[d]; t2 += ww*x2[d]; t3 += ww*x3[d];
    }
    const auto out = c.row_offset + row;
    c.output[static_cast<std::size_t>(sb+0U)*c.output_stride+out] = hsum(a0)+t0;
    c.output[static_cast<std::size_t>(sb+1U)*c.output_stride+out] = hsum(a1)+t1;
    c.output[static_cast<std::size_t>(sb+2U)*c.output_stride+out] = hsum(a2)+t2;
    c.output[static_cast<std::size_t>(sb+3U)*c.output_stride+out] = hsum(a3)+t3;
  }
}

CNRL_NOINLINE void fused8(const KernelCall& c, std::uint32_t sb) noexcept {
  for (std::uint32_t row = 0; row < c.rows; ++row) {
    const auto* w = c.weights + static_cast<std::size_t>(row) * c.dimension;
    const auto* x0 = c.state + static_cast<std::size_t>(sb+0U)*c.dimension;
    const auto* x1 = c.state + static_cast<std::size_t>(sb+1U)*c.dimension;
    const auto* x2 = c.state + static_cast<std::size_t>(sb+2U)*c.dimension;
    const auto* x3 = c.state + static_cast<std::size_t>(sb+3U)*c.dimension;
    const auto* x4 = c.state + static_cast<std::size_t>(sb+4U)*c.dimension;
    const auto* x5 = c.state + static_cast<std::size_t>(sb+5U)*c.dimension;
    const auto* x6 = c.state + static_cast<std::size_t>(sb+6U)*c.dimension;
    const auto* x7 = c.state + static_cast<std::size_t>(sb+7U)*c.dimension;
    __m256i a0=_mm256_setzero_si256(),a1=_mm256_setzero_si256();
    __m256i a2=_mm256_setzero_si256(),a3=_mm256_setzero_si256();
    __m256i a4=_mm256_setzero_si256(),a5=_mm256_setzero_si256();
    __m256i a6=_mm256_setzero_si256(),a7=_mm256_setzero_si256();
    std::int32_t t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0;
    std::uint32_t d=0;
    for (; d+kVec<=c.dimension; d+=kVec) {
      const __m256i vw=load_i8x16_as_i16(w+d);
      a0=_mm256_add_epi32(a0,_mm256_madd_epi16(vw,load_i8x16_as_i16(x0+d)));
      a1=_mm256_add_epi32(a1,_mm256_madd_epi16(vw,load_i8x16_as_i16(x1+d)));
      a2=_mm256_add_epi32(a2,_mm256_madd_epi16(vw,load_i8x16_as_i16(x2+d)));
      a3=_mm256_add_epi32(a3,_mm256_madd_epi16(vw,load_i8x16_as_i16(x3+d)));
      a4=_mm256_add_epi32(a4,_mm256_madd_epi16(vw,load_i8x16_as_i16(x4+d)));
      a5=_mm256_add_epi32(a5,_mm256_madd_epi16(vw,load_i8x16_as_i16(x5+d)));
      a6=_mm256_add_epi32(a6,_mm256_madd_epi16(vw,load_i8x16_as_i16(x6+d)));
      a7=_mm256_add_epi32(a7,_mm256_madd_epi16(vw,load_i8x16_as_i16(x7+d)));
    }
    for (; d<c.dimension; ++d) {
      const auto ww=static_cast<std::int32_t>(w[d]);
      t0+=ww*x0[d];t1+=ww*x1[d];t2+=ww*x2[d];t3+=ww*x3[d];
      t4+=ww*x4[d];t5+=ww*x5[d];t6+=ww*x6[d];t7+=ww*x7[d];
    }
    const auto out=c.row_offset+row;
    c.output[static_cast<std::size_t>(sb+0U)*c.output_stride+out]=hsum(a0)+t0;
    c.output[static_cast<std::size_t>(sb+1U)*c.output_stride+out]=hsum(a1)+t1;
    c.output[static_cast<std::size_t>(sb+2U)*c.output_stride+out]=hsum(a2)+t2;
    c.output[static_cast<std::size_t>(sb+3U)*c.output_stride+out]=hsum(a3)+t3;
    c.output[static_cast<std::size_t>(sb+4U)*c.output_stride+out]=hsum(a4)+t4;
    c.output[static_cast<std::size_t>(sb+5U)*c.output_stride+out]=hsum(a5)+t5;
    c.output[static_cast<std::size_t>(sb+6U)*c.output_stride+out]=hsum(a6)+t6;
    c.output[static_cast<std::size_t>(sb+7U)*c.output_stride+out]=hsum(a7)+t7;
  }
}
}  // namespace

void validate_kernel_call(const KernelCall& c) {
  if (!c.weights || !c.state || !c.output) throw std::invalid_argument("null kernel pointer");
  if (c.rows==0 || c.dimension==0 || c.slots==0) throw std::invalid_argument("zero kernel dimension");
  if (c.dimension>kMaxD) throw std::invalid_argument("D exceeds safe int32 accumulator range");
  if (c.output_stride<c.row_offset+c.rows) throw std::invalid_argument("output stride too small");
  if (c.slot_tile!=1 && c.slot_tile!=2 && c.slot_tile!=4 && c.slot_tile!=8) {
    throw std::invalid_argument("slot tile must be 1,2,4,8");
  }
}

namespace {
void scalar_unchecked(const KernelCall& c) {
  for (std::uint32_t s=0;s<c.slots;++s) {
    const auto* x=c.state+static_cast<std::size_t>(s)*c.dimension;
    for (std::uint32_t r=0;r<c.rows;++r) {
      const auto* w=c.weights+static_cast<std::size_t>(r)*c.dimension;
      std::int64_t sum=0;
      for (std::uint32_t d=0;d<c.dimension;++d) sum+=static_cast<std::int32_t>(w[d])*x[d];
      if (sum<INT32_MIN || sum>INT32_MAX) throw std::overflow_error("dot product overflow");
      c.output[static_cast<std::size_t>(s)*c.output_stride+c.row_offset+r]=static_cast<std::int32_t>(sum);
    }
  }
}

void repeat_unchecked(const KernelCall& c) noexcept {
  for (std::uint32_t s=0;s<c.slots;++s) {
    const auto* x=c.state+static_cast<std::size_t>(s)*c.dimension;
    for(std::uint32_t row=0;row<c.rows;++row) {
      const auto* w=c.weights+static_cast<std::size_t>(row)*c.dimension;
      __m256i acc=_mm256_setzero_si256();
      std::int32_t tail=0;
      std::uint32_t d=0;
      for(;d+kVec<=c.dimension;d+=kVec) {
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(load_i8x16_as_i16(w+d),
                                                    load_i8x16_as_i16(x+d)));
      }
      for(;d<c.dimension;++d) tail+=static_cast<std::int32_t>(w[d])*x[d];
      c.output[static_cast<std::size_t>(s)*c.output_stride+c.row_offset+row]=hsum(acc)+tail;
    }
  }
}

void fused_unchecked(const KernelCall& c) noexcept {
  std::uint32_t s=0;
  while (s<c.slots) {
    const auto left=c.slots-s;
    if (c.slot_tile>=8 && left>=8) { fused8(c,s); s+=8; }
    else if (c.slot_tile>=4 && left>=4) { fused4(c,s); s+=4; }
    else if (c.slot_tile>=2 && left>=2) { fused2(c,s); s+=2; }
    else { fused1(c,s); ++s; }
  }
}
}  // namespace

void matmul_scalar_reference(const KernelCall& c) {
  validate_kernel_call(c);
  scalar_unchecked(c);
}

void matmul_avx2_repeat(const KernelCall& c) {
  validate_kernel_call(c);
  repeat_unchecked(c);
}

void matmul_avx2_fused(const KernelCall& c) {
  validate_kernel_call(c);
  fused_unchecked(c);
}

void run_kernel_unchecked(KernelKind kind, const KernelCall& call) {
  switch (kind) {
    case KernelKind::scalar: scalar_unchecked(call); break;
    case KernelKind::avx2_repeat: repeat_unchecked(call); break;
    case KernelKind::avx2_fused: fused_unchecked(call); break;
  }
}

void run_kernel(KernelKind kind, const KernelCall& call) {
  validate_kernel_call(call);
  run_kernel_unchecked(kind, call);
}

std::uint64_t kernel_macs(const KernelCall& c) noexcept {
  return static_cast<std::uint64_t>(c.rows)*c.dimension*c.slots;
}
}  // namespace cnrl
