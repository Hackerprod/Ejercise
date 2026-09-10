#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace cnrl {

template <typename T, std::size_t Alignment = 64>
class AlignedBuffer {
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(Alignment >= alignof(T));
  static_assert((Alignment & (Alignment - 1U)) == 0U);

 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t count) { resize(count); }
  AlignedBuffer(const AlignedBuffer& other) { assign(other.data_, other.size_); }
  AlignedBuffer& operator=(const AlignedBuffer& other) {
    if (this != &other) assign(other.data_, other.size_);
    return *this;
  }
  AlignedBuffer(AlignedBuffer&& other) noexcept { swap(other); }
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }
  ~AlignedBuffer() { reset(); }

  void resize(std::size_t count) {
    if (count == size_) return;
    reset();
    if (count == 0) return;
    if (count > static_cast<std::size_t>(-1) / sizeof(T)) {
      throw std::overflow_error("AlignedBuffer size overflow");
    }
    const std::size_t bytes = count * sizeof(T);
#if defined(_WIN32)
    data_ = static_cast<T*>(_aligned_malloc(bytes, Alignment));
    if (data_ == nullptr) throw std::bad_alloc();
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, Alignment, bytes) != 0 || raw == nullptr) {
      throw std::bad_alloc();
    }
    data_ = static_cast<T*>(raw);
#endif
    size_ = count;
  }

  void assign(const T* source, std::size_t count) {
    resize(count);
    if (count != 0) std::memcpy(data_, source, count * sizeof(T));
  }
  void fill_zero() noexcept {
    if (size_ != 0) std::memset(data_, 0, size_ * sizeof(T));
  }
  void reset() noexcept {
    if (data_ != nullptr) {
#if defined(_WIN32)
      _aligned_free(data_);
#else
      std::free(data_);
#endif
    }
    data_ = nullptr;
    size_ = 0;
  }
  void swap(AlignedBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  T& operator[](std::size_t index) noexcept { return data_[index]; }
  const T& operator[](std::size_t index) const noexcept { return data_[index]; }
  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace cnrl
