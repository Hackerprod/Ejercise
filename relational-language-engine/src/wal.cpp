#include "rlm/wal.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rlm {
namespace {
constexpr std::array<std::byte, 4> kMagic{std::byte{'R'}, std::byte{'L'}, std::byte{'W'}, std::byte{'1'}};
constexpr std::uint16_t kVersion = 1;
constexpr std::size_t kHeaderSize = 32;

Status io_error(std::string_view operation, const std::filesystem::path& path) {
  return Status(ErrorCode::io_error,
                std::string(operation) + " '" + path.string() + "': " + std::strerror(errno));
}

Status write_all(int fd, std::span<const std::byte> bytes, const std::filesystem::path& path) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = ::write(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) continue;
      return io_error("write WAL", path);
    }
    if (count == 0) return Status(ErrorCode::io_error, "zero-byte WAL write");
    offset += static_cast<std::size_t>(count);
  }
  return Status::Ok();
}

Result<std::vector<std::byte>> pread_exact_or_tail(int fd, std::uint64_t offset, std::size_t count,
                                                   const std::filesystem::path& path) {
  std::vector<std::byte> output(count);
  std::size_t read_count = 0;
  while (read_count < count) {
    const ssize_t result = ::pread(fd, output.data() + read_count, count - read_count,
                                   static_cast<off_t>(offset + read_count));
    if (result < 0) {
      if (errno == EINTR) continue;
      return io_error("read WAL", path);
    }
    if (result == 0) {
      output.resize(read_count);
      return output;
    }
    read_count += static_cast<std::size_t>(result);
  }
  return output;
}
}  // namespace

WriteAheadLog::~WriteAheadLog() {
  std::lock_guard lock(mutex_);
  if (fd_ >= 0) ::close(fd_);
}

Status WriteAheadLog::open(const std::filesystem::path& path, Durability durability) {
  std::lock_guard lock(mutex_);
  if (fd_ >= 0) return Status(ErrorCode::failed_precondition, "WAL already open");
  path_ = path;
  durability_ = durability;
  RLM_RETURN_IF_ERROR(ensure_directory(path.parent_path()));
  return open_fd_locked();
}

Status WriteAheadLog::open_fd_locked() {
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC, 0640);
  if (fd_ < 0) return io_error("open WAL", path_);
  return Status::Ok();
}

Status WriteAheadLog::replay(const std::function<Status(const WalRecord&)>& callback,
                             bool repair_truncated_tail) {
  std::lock_guard lock(mutex_);
  if (fd_ < 0) return Status(ErrorCode::failed_precondition, "WAL not open");
  struct stat info {};
  if (::fstat(fd_, &info) != 0) return io_error("stat WAL", path_);
  const std::uint64_t file_size = static_cast<std::uint64_t>(info.st_size);
  std::uint64_t offset = 0;
  std::uint64_t highest_sequence = 0;
  while (offset < file_size) {
    auto header_result = pread_exact_or_tail(fd_, offset, kHeaderSize, path_);
    if (!header_result) return header_result.status();
    const auto& header = header_result.value();
    if (header.size() < kHeaderSize) {
      if (!repair_truncated_tail) return Status(ErrorCode::data_loss, "truncated WAL header");
      if (::ftruncate(fd_, static_cast<off_t>(offset)) != 0) return io_error("repair WAL", path_);
      break;
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
      return Status(ErrorCode::data_loss, "WAL magic mismatch at offset " + std::to_string(offset));
    }
    ByteReader reader(header);
    auto magic_skip = reader.bytes(kMagic.size());
    if (!magic_skip) return magic_skip.status();
    auto version = reader.u16();
    auto kind = reader.u16();
    auto sequence = reader.u64();
    auto payload_size = reader.u32();
    auto payload_crc = reader.u32();
    auto header_crc = reader.u32();
    auto reserved = reader.u32();
    (void)reserved;
    if (!version || !kind || !sequence || !payload_size || !payload_crc || !header_crc) {
      return Status(ErrorCode::data_loss, "invalid WAL header");
    }
    if (version.value() != kVersion) return Status(ErrorCode::data_loss, "unsupported WAL version");
    if (crc32(std::span(header).first(24)) != header_crc.value()) {
      return Status(ErrorCode::data_loss, "WAL header CRC mismatch at offset " + std::to_string(offset));
    }
    constexpr std::uint32_t kMaxRecord = 256U * 1024U * 1024U;
    if (payload_size.value() > kMaxRecord) return Status(ErrorCode::data_loss, "WAL record exceeds safety limit");
    auto payload_result = pread_exact_or_tail(fd_, offset + kHeaderSize, payload_size.value(), path_);
    if (!payload_result) return payload_result.status();
    const auto& payload = payload_result.value();
    if (payload.size() < payload_size.value()) {
      if (!repair_truncated_tail) return Status(ErrorCode::data_loss, "truncated WAL payload");
      if (::ftruncate(fd_, static_cast<off_t>(offset)) != 0) return io_error("repair WAL", path_);
      break;
    }
    if (crc32(payload) != payload_crc.value()) {
      return Status(ErrorCode::data_loss, "WAL payload CRC mismatch at offset " + std::to_string(offset));
    }
    const Status callback_status = callback(WalRecord{kind.value(), sequence.value(), payload});
    if (!callback_status) return callback_status;
    highest_sequence = std::max(highest_sequence, sequence.value());
    offset += kHeaderSize + payload_size.value();
  }
  next_sequence_ = highest_sequence + 1;
  if (::lseek(fd_, 0, SEEK_END) < 0) return io_error("seek WAL", path_);
  return Status::Ok();
}

Result<std::uint64_t> WriteAheadLog::append(std::uint16_t kind,
                                            std::span<const std::byte> payload) {
  std::lock_guard lock(mutex_);
  if (fd_ < 0) return Status(ErrorCode::failed_precondition, "WAL not open");
  if (payload.size() > 256ULL * 1024ULL * 1024ULL) {
    return Status(ErrorCode::resource_exhausted, "WAL payload exceeds 256 MiB limit");
  }
  const std::uint64_t sequence = next_sequence_++;
  ByteWriter header;
  header.bytes(kMagic);
  header.u16(kVersion);
  header.u16(kind);
  header.u64(sequence);
  header.u32(static_cast<std::uint32_t>(payload.size()));
  header.u32(crc32(payload));
  header.u32(crc32(std::span(header.data()).first(24)));
  header.u32(0);
  if (header.data().size() != kHeaderSize) return Status(ErrorCode::internal, "WAL header size invariant failed");
  Status status = write_all(fd_, header.data(), path_);
  if (status) status = write_all(fd_, payload, path_);
  if (status && durability_ != Durability::none) {
    const int result = durability_ == Durability::data ? ::fdatasync(fd_) : ::fsync(fd_);
    if (result != 0) status = io_error("sync WAL", path_);
  }
  if (!status) return status;
  return sequence;
}

Status WriteAheadLog::flush() {
  std::lock_guard lock(mutex_);
  if (fd_ < 0) return Status(ErrorCode::failed_precondition, "WAL not open");
  if (durability_ == Durability::none) return Status::Ok();
  const int result = durability_ == Durability::data ? ::fdatasync(fd_) : ::fsync(fd_);
  return result == 0 ? Status::Ok() : io_error("sync WAL", path_);
}

Status WriteAheadLog::reset() {
  std::lock_guard lock(mutex_);
  if (fd_ < 0) return Status(ErrorCode::failed_precondition, "WAL not open");
  if (::ftruncate(fd_, 0) != 0) return io_error("truncate WAL", path_);
  if (::lseek(fd_, 0, SEEK_SET) < 0) return io_error("seek WAL", path_);
  next_sequence_ = 1;
  if (durability_ != Durability::none) {
    const int result = durability_ == Durability::data ? ::fdatasync(fd_) : ::fsync(fd_);
    if (result != 0) return io_error("sync WAL", path_);
  }
  return Status::Ok();
}

std::uint64_t WriteAheadLog::next_sequence() const noexcept {
  std::lock_guard lock(mutex_);
  return next_sequence_;
}

}  // namespace rlm
