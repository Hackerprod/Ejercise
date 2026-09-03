#include "mrdl/process_lock.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace mrdl {

ProcessLock::ProcessLock(const std::filesystem::path& model_directory,
                         LockMode mode,
                         bool wait) {
    std::filesystem::create_directories(model_directory);
    path_ = model_directory / ".mrdl.lock";
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd_ < 0) throw Error("cannot open model lock: " + std::string(std::strerror(errno)));
    const int operation = (mode == LockMode::Exclusive ? LOCK_EX : LOCK_SH) | (wait ? 0 : LOCK_NB);
    if (::flock(fd_, operation) != 0) {
        const int code = errno;
        release();
        if (code == EWOULDBLOCK) throw Error("model is already in use by another MRDL process");
        throw Error("cannot acquire model lock: " + std::string(std::strerror(code)));
    }
    if (mode == LockMode::Exclusive) {
        const std::string owner = std::to_string(static_cast<long long>(::getpid())) + "\n";
        (void)::ftruncate(fd_, 0);
        (void)::pwrite(fd_, owner.data(), owner.size(), 0);
        (void)::fsync(fd_);
    }
}

ProcessLock::~ProcessLock() { release(); }

ProcessLock::ProcessLock(ProcessLock&& other) noexcept { *this = std::move(other); }

ProcessLock& ProcessLock::operator=(ProcessLock&& other) noexcept {
    if (this == &other) return *this;
    release();
    fd_ = std::exchange(other.fd_, -1);
    path_ = std::move(other.path_);
    return *this;
}

void ProcessLock::release() noexcept {
    if (fd_ >= 0) {
        (void)::flock(fd_, LOCK_UN);
        (void)::close(fd_);
        fd_ = -1;
    }
}

}  // namespace mrdl
