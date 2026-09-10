#pragma once

#include "mrdl/common.hpp"

namespace mrdl {

enum class LockMode : std::uint8_t { Shared = 0, Exclusive = 1 };

class ProcessLock final {
public:
    ProcessLock() = default;
    ProcessLock(const std::filesystem::path& model_directory,
                LockMode mode,
                bool wait = false);
    ~ProcessLock();
    ProcessLock(const ProcessLock&) = delete;
    ProcessLock& operator=(const ProcessLock&) = delete;
    ProcessLock(ProcessLock&& other) noexcept;
    ProcessLock& operator=(ProcessLock&& other) noexcept;

    [[nodiscard]] bool owns_lock() const noexcept { return fd_ >= 0; }

private:
    int fd_{-1};
    std::filesystem::path path_;
    void release() noexcept;
};

}  // namespace mrdl
