#pragma once
#include <cstdint>
#include "cnrl/types.hpp"
namespace cnrl {
struct KernelCall {
  const std::int8_t* weights = nullptr;
  const std::int8_t* state = nullptr;
  std::int32_t* output = nullptr;
  std::uint32_t rows = 0;
  std::uint32_t dimension = 0;
  std::uint32_t slots = 0;
  std::uint32_t row_offset = 0;
  std::uint32_t output_stride = 0;
  std::uint32_t slot_tile = 4;
};
void validate_kernel_call(const KernelCall& call);
void matmul_scalar_reference(const KernelCall& call);
void matmul_avx2_repeat(const KernelCall& call);
void matmul_avx2_fused(const KernelCall& call);
void run_kernel(KernelKind kind, const KernelCall& call);
// Benchmark-only path. The caller must validate once before entering the timed loop.
void run_kernel_unchecked(KernelKind kind, const KernelCall& call);
[[nodiscard]] std::uint64_t kernel_macs(const KernelCall& call) noexcept;
}  // namespace cnrl
