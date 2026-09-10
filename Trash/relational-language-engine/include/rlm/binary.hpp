#pragma once

#include "rlm/status.hpp"
#include "rlm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rlm {

[[nodiscard]] std::uint32_t crc32(std::span<const std::byte> bytes) noexcept;

class ByteWriter final {
 public:
  void u8(std::uint8_t value);
  void u16(std::uint16_t value);
  void u32(std::uint32_t value);
  void u64(std::uint64_t value);
  void f32(float value);
  void bytes(std::span<const std::byte> value);
  void string(std::string_view value);
  void qvector(const QuantizedVector& value);

  [[nodiscard]] const std::vector<std::byte>& data() const noexcept { return data_; }
  [[nodiscard]] std::vector<std::byte>& data() noexcept { return data_; }
  [[nodiscard]] std::vector<std::byte> take() noexcept { return std::move(data_); }

 private:
  std::vector<std::byte> data_;
};

class ByteReader final {
 public:
  explicit ByteReader(std::span<const std::byte> data) : data_(data) {}

  [[nodiscard]] Result<std::uint8_t> u8();
  [[nodiscard]] Result<std::uint16_t> u16();
  [[nodiscard]] Result<std::uint32_t> u32();
  [[nodiscard]] Result<std::uint64_t> u64();
  [[nodiscard]] Result<float> f32();
  [[nodiscard]] Result<std::span<const std::byte>> bytes(std::size_t count);
  [[nodiscard]] Result<std::string> string(std::size_t max_length = 1U << 20U);
  [[nodiscard]] Result<QuantizedVector> qvector(std::size_t expected_dim = 0,
                                                std::size_t max_dim = 1U << 20U);

  [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - offset_; }
  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

 private:
  std::span<const std::byte> data_;
  std::size_t offset_{0};
};

class MappedFile final {
 public:
  MappedFile() = default;
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  [[nodiscard]] Status open_read_only(const std::filesystem::path& path);
  void close() noexcept;
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  [[nodiscard]] bool open() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  int fd_{-1};
  const std::byte* data_{nullptr};
  std::size_t size_{0};
};

class ProcessFileLock final {
 public:
  ProcessFileLock() = default;
  ~ProcessFileLock();
  ProcessFileLock(const ProcessFileLock&) = delete;
  ProcessFileLock& operator=(const ProcessFileLock&) = delete;

  [[nodiscard]] Status acquire(const std::filesystem::path& path);
  void release() noexcept;
  [[nodiscard]] bool held() const noexcept { return fd_ >= 0; }

 private:
  int fd_{-1};
};

[[nodiscard]] Status ensure_directory(const std::filesystem::path& path);
[[nodiscard]] Result<std::vector<std::byte>> read_file(const std::filesystem::path& path,
                                                       std::size_t max_bytes = 1ULL << 34U);
[[nodiscard]] Status write_file_atomic(const std::filesystem::path& path,
                                       std::span<const std::byte> bytes,
                                       Durability durability = Durability::full);
[[nodiscard]] Status fsync_path(const std::filesystem::path& path, bool data_only);
[[nodiscard]] Status truncate_file(const std::filesystem::path& path, std::uint64_t size);

}  // namespace rlm
