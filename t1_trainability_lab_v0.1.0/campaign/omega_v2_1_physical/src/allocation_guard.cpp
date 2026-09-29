#include "v2_1.hpp"

#include <atomic>
#include <cstdlib>
#include <malloc.h>
#include <new>

namespace {
std::atomic<bool> g_track_allocations{false};
std::atomic<std::uint64_t> g_timed_allocations{0};

void count_if_timed() noexcept {
    if (g_track_allocations.load(std::memory_order_relaxed)) g_timed_allocations.fetch_add(1, std::memory_order_relaxed);
}
}

void* operator new(std::size_t size) {
    count_if_timed();
    if (void* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    count_if_timed();
    if (void* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

void* operator new(std::size_t size, std::align_val_t alignment) {
    count_if_timed();
    if (void* memory = _aligned_malloc(size ? size : 1, static_cast<std::size_t>(alignment))) return memory;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    count_if_timed();
    if (void* memory = _aligned_malloc(size ? size : 1, static_cast<std::size_t>(alignment))) return memory;
    throw std::bad_alloc();
}

void operator delete(void* memory, std::align_val_t) noexcept { _aligned_free(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { _aligned_free(memory); }
void operator delete(void* memory, std::size_t, std::align_val_t) noexcept { _aligned_free(memory); }
void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept { _aligned_free(memory); }

namespace omega_v2_1 {

void begin_timed_allocation_count() {
    g_timed_allocations.store(0, std::memory_order_relaxed);
    g_track_allocations.store(true, std::memory_order_release);
}

std::uint64_t end_timed_allocation_count() {
    g_track_allocations.store(false, std::memory_order_release);
    return g_timed_allocations.load(std::memory_order_relaxed);
}

} // namespace omega_v2_1
