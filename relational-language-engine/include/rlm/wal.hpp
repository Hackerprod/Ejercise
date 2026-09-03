#pragma once

#include "rlm/binary.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>

namespace rlm {

struct WalRecord final {
  std::uint16_t kind{0};
  std::uint64_t sequence{0};
  std::span<const std::byte> payload;
};

class WriteAheadLog final {
 public:
  WriteAheadLog() = default;
  ~WriteAheadLog();
  WriteAheadLog(const WriteAheadLog&) = delete;
  WriteAheadLog& operator=(const WriteAheadLog&) = delete;

  [[nodiscard]] Status open(const std::filesystem::path& path, Durability durability);
  [[nodiscard]] Status replay(const std::function<Status(const WalRecord&)>& callback,
                              bool repair_truncated_tail = true);
  [[nodiscard]] Result<std::uint64_t> append(std::uint16_t kind,
                                             std::span<const std::byte> payload);
  [[nodiscard]] Status flush();
  [[nodiscard]] Status reset();
  [[nodiscard]] std::uint64_t next_sequence() const noexcept;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  [[nodiscard]] Status open_fd_locked();

  mutable std::mutex mutex_;
  std::filesystem::path path_;
  int fd_{-1};
  Durability durability_{Durability::data};
  std::uint64_t next_sequence_{1};
};

}  // namespace rlm
