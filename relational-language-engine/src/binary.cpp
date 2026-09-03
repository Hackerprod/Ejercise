#include "rlm/binary.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rlm {
namespace {

Status errno_status(std::string_view operation, const std::filesystem::path& path) {
  return Status(ErrorCode::io_error,
                std::string(operation) + " '" + path.string() + "': " + std::strerror(errno));
}

Status write_all(int fd, std::span<const std::byte> bytes, const std::filesystem::path& path) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = ::write(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) continue;
      return errno_status("write", path);
    }
    if (count == 0) return Status(ErrorCode::io_error, "zero-byte write to '" + path.string() + "'");
    offset += static_cast<std::size_t>(count);
  }
  return Status::Ok();
}

template <typename T>
void append_le(std::vector<std::byte>& output, T value) {
  static_assert(std::is_unsigned_v<T>);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    output.push_back(static_cast<std::byte>((value >> (i * 8U)) & static_cast<T>(0xffU)));
  }
}

template <typename T>
Result<T> read_le(std::span<const std::byte> data, std::size_t& offset) {
  static_assert(std::is_unsigned_v<T>);
  if (data.size() - offset < sizeof(T)) return Status(ErrorCode::data_loss, "truncated binary value");
  T value = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    value |= static_cast<T>(std::to_integer<unsigned char>(data[offset + i])) << (i * 8U);
  }
  offset += sizeof(T);
  return value;
}

}  // namespace

std::uint32_t crc32(std::span<const std::byte> bytes) noexcept {
  std::uint32_t crc = 0xffffffffU;
  for (const std::byte byte : bytes) {
    crc ^= std::to_integer<std::uint8_t>(byte);
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xedb88320U & mask);
    }
  }
  return ~crc;
}

void ByteWriter::u8(std::uint8_t value) { data_.push_back(static_cast<std::byte>(value)); }
void ByteWriter::u16(std::uint16_t value) { append_le(data_, value); }
void ByteWriter::u32(std::uint32_t value) { append_le(data_, value); }
void ByteWriter::u64(std::uint64_t value) { append_le(data_, value); }
void ByteWriter::f32(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  u32(bits);
}
void ByteWriter::bytes(std::span<const std::byte> value) {
  data_.insert(data_.end(), value.begin(), value.end());
}
void ByteWriter::string(std::string_view value) {
  if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("string exceeds binary format limit");
  }
  u32(static_cast<std::uint32_t>(value.size()));
  bytes(std::as_bytes(std::span{value.data(), value.size()}));
}
void ByteWriter::qvector(const QuantizedVector& value) {
  if (value.values.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("vector exceeds binary format limit");
  }
  f32(value.scale);
  u32(static_cast<std::uint32_t>(value.values.size()));
  bytes(std::as_bytes(std::span{value.values.data(), value.values.size()}));
}

Result<std::uint8_t> ByteReader::u8() {
  if (remaining() < 1) return Status(ErrorCode::data_loss, "truncated u8");
  return std::to_integer<std::uint8_t>(data_[offset_++]);
}
Result<std::uint16_t> ByteReader::u16() { return read_le<std::uint16_t>(data_, offset_); }
Result<std::uint32_t> ByteReader::u32() { return read_le<std::uint32_t>(data_, offset_); }
Result<std::uint64_t> ByteReader::u64() { return read_le<std::uint64_t>(data_, offset_); }
Result<float> ByteReader::f32() {
  auto bits = u32();
  if (!bits) return bits.status();
  float value = 0.0F;
  const std::uint32_t raw = bits.value();
  std::memcpy(&value, &raw, sizeof(value));
  if (!std::isfinite(value)) return Status(ErrorCode::data_loss, "non-finite float in binary data");
  return value;
}
Result<std::span<const std::byte>> ByteReader::bytes(std::size_t count) {
  if (count > remaining()) return Status(ErrorCode::data_loss, "truncated byte range");
  const auto result = data_.subspan(offset_, count);
  offset_ += count;
  return result;
}
Result<std::string> ByteReader::string(std::size_t max_length) {
  auto size_result = u32();
  if (!size_result) return size_result.status();
  const std::size_t size = size_result.value();
  if (size > max_length) return Status(ErrorCode::resource_exhausted, "binary string exceeds configured limit");
  auto value = bytes(size);
  if (!value) return value.status();
  const auto chars = std::span(reinterpret_cast<const char*>(value.value().data()), value.value().size());
  return std::string(chars.begin(), chars.end());
}
Result<QuantizedVector> ByteReader::qvector(std::size_t expected_dim, std::size_t max_dim) {
  auto scale = f32();
  if (!scale) return scale.status();
  auto size_result = u32();
  if (!size_result) return size_result.status();
  const std::size_t size = size_result.value();
  if (size == 0 || size > max_dim) return Status(ErrorCode::data_loss, "invalid quantized vector dimension");
  if (expected_dim != 0 && size != expected_dim) return Status(ErrorCode::data_loss, "quantized vector dimension mismatch");
  auto raw = bytes(size);
  if (!raw) return raw.status();
  QuantizedVector output;
  output.scale = scale.value();
  output.values.resize(size);
  std::memcpy(output.values.data(), raw.value().data(), size);
  const Status status = output.validate(expected_dim);
  if (!status) return status;
  return output;
}

MappedFile::~MappedFile() { close(); }
MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), data_(other.data_), size_(other.size_) {
  other.fd_ = -1;
  other.data_ = nullptr;
  other.size_ = 0;
}
MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}
Status MappedFile::open_read_only(const std::filesystem::path& path) {
  close();
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) return errno_status("open", path);
  struct stat info {};
  if (::fstat(fd_, &info) != 0) {
    const Status status = errno_status("fstat", path);
    close();
    return status;
  }
  if (info.st_size <= 0) {
    close();
    return Status(ErrorCode::data_loss, "cannot mmap an empty file: '" + path.string() + "'");
  }
  size_ = static_cast<std::size_t>(info.st_size);
  void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (mapped == MAP_FAILED) {
    const Status status = errno_status("mmap", path);
    close();
    return status;
  }
  data_ = static_cast<const std::byte*>(mapped);
  return Status::Ok();
}
void MappedFile::close() noexcept {
  if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
  data_ = nullptr;
  size_ = 0;
}
std::span<const std::byte> MappedFile::bytes() const noexcept { return {data_, size_}; }

ProcessFileLock::~ProcessFileLock() { release(); }
Status ProcessFileLock::acquire(const std::filesystem::path& path) {
  if (held()) return Status(ErrorCode::failed_precondition, "process lock already held");
  RLM_RETURN_IF_ERROR(ensure_directory(path.parent_path()));
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0640);
  if (fd_ < 0) return errno_status("open lock", path);
  if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    const Status status = errno == EWOULDBLOCK
        ? Status(ErrorCode::unavailable, "another engine process owns '" + path.string() + "'")
        : errno_status("flock", path);
    release();
    return status;
  }
  return Status::Ok();
}
void ProcessFileLock::release() noexcept {
  if (fd_ >= 0) {
    ::flock(fd_, LOCK_UN);
    ::close(fd_);
  }
  fd_ = -1;
}

Status ensure_directory(const std::filesystem::path& path) {
  if (path.empty()) return Status::Ok();
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) return Status(ErrorCode::io_error, "create directory '" + path.string() + "': " + ec.message());
  if (!std::filesystem::is_directory(path, ec) || ec) {
    return Status(ErrorCode::failed_precondition, "path is not a directory: '" + path.string() + "'");
  }
  return Status::Ok();
}

Result<std::vector<std::byte>> read_file(const std::filesystem::path& path, std::size_t max_bytes) {
  std::error_code ec;
  const std::uintmax_t size_raw = std::filesystem::file_size(path, ec);
  if (ec) return Status(ErrorCode::io_error, "file_size '" + path.string() + "': " + ec.message());
  if (size_raw > max_bytes || size_raw > std::numeric_limits<std::size_t>::max()) {
    return Status(ErrorCode::resource_exhausted, "file exceeds configured size limit: '" + path.string() + "'");
  }
  const std::size_t size = static_cast<std::size_t>(size_raw);
  std::vector<std::byte> output(size);
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return errno_status("open", path);
  std::size_t offset = 0;
  Status status = Status::Ok();
  while (offset < size) {
    const ssize_t count = ::read(fd, output.data() + offset, size - offset);
    if (count < 0) {
      if (errno == EINTR) continue;
      status = errno_status("read", path);
      break;
    }
    if (count == 0) {
      status = Status(ErrorCode::data_loss, "unexpected EOF in '" + path.string() + "'");
      break;
    }
    offset += static_cast<std::size_t>(count);
  }
  ::close(fd);
  if (!status) return status;
  return output;
}

Status fsync_path(const std::filesystem::path& path, bool data_only) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return errno_status("open for sync", path);
  const int result = data_only ? ::fdatasync(fd) : ::fsync(fd);
  const Status status = result == 0 ? Status::Ok() : errno_status("sync", path);
  ::close(fd);
  return status;
}

Status write_file_atomic(const std::filesystem::path& path,
                         std::span<const std::byte> bytes,
                         Durability durability) {
  RLM_RETURN_IF_ERROR(ensure_directory(path.parent_path()));
  const std::filesystem::path temporary = path.string() + ".tmp." + std::to_string(::getpid()) + "." +
                                          std::to_string(unix_time_ms());
  const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0640);
  if (fd < 0) return errno_status("create", temporary);
  Status status = write_all(fd, bytes, temporary);
  if (status && durability != Durability::none) {
    const int result = durability == Durability::data ? ::fdatasync(fd) : ::fsync(fd);
    if (result != 0) status = errno_status("sync", temporary);
  }
  if (::close(fd) != 0 && status) status = errno_status("close", temporary);
  if (!status) {
    ::unlink(temporary.c_str());
    return status;
  }
  if (::rename(temporary.c_str(), path.c_str()) != 0) {
    status = errno_status("rename", path);
    ::unlink(temporary.c_str());
    return status;
  }
  if (durability == Durability::full) {
    const int directory_fd = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory_fd < 0) return errno_status("open directory for sync", path.parent_path());
    if (::fsync(directory_fd) != 0) status = errno_status("sync directory", path.parent_path());
    ::close(directory_fd);
  }
  return status;
}

Status truncate_file(const std::filesystem::path& path, std::uint64_t size) {
  if (::truncate(path.c_str(), static_cast<off_t>(size)) != 0) return errno_status("truncate", path);
  return Status::Ok();
}

}  // namespace rlm
