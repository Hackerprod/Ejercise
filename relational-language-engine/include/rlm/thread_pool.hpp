#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace rlm {

class BoundedThreadPool final {
 public:
  BoundedThreadPool(std::size_t thread_count, std::size_t queue_capacity)
      : queue_capacity_(queue_capacity) {
    if (thread_count == 0 || queue_capacity == 0) throw std::invalid_argument("thread pool limits must be positive");
    workers_.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      workers_.emplace_back([this](std::stop_token stop) { worker_loop(stop); });
    }
  }

  ~BoundedThreadPool() { shutdown(); }
  BoundedThreadPool(const BoundedThreadPool&) = delete;
  BoundedThreadPool& operator=(const BoundedThreadPool&) = delete;

  template <typename Function>
  auto submit(Function&& function) -> std::future<std::invoke_result_t<Function>> {
    using ResultType = std::invoke_result_t<Function>;
    auto task = std::make_shared<std::packaged_task<ResultType()>>(std::forward<Function>(function));
    std::future<ResultType> future = task->get_future();
    bool execute_inline = false;
    {
      std::lock_guard lock(mutex_);
      if (stopping_) throw std::runtime_error("submit on stopped thread pool");
      if (inside_worker_ || queue_.size() >= queue_capacity_) {
        execute_inline = true;
      } else {
        queue_.emplace_back([task]() { (*task)(); });
      }
    }
    if (execute_inline) {
      (*task)();
    } else {
      condition_.notify_one();
    }
    return future;
  }

  void shutdown() noexcept {
    {
      std::lock_guard lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    condition_.notify_all();
    for (std::jthread& worker : workers_) worker.request_stop();
    workers_.clear();
    std::deque<std::function<void()>> remaining;
    {
      std::lock_guard lock(mutex_);
      remaining.swap(queue_);
    }
    for (auto& task : remaining) task();
  }

  [[nodiscard]] std::size_t thread_count() const noexcept { return workers_.size(); }

 private:
  void worker_loop(std::stop_token stop) noexcept {
    inside_worker_ = true;
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&]() { return stopping_ || stop.stop_requested() || !queue_.empty(); });
        if ((stopping_ || stop.stop_requested()) && queue_.empty()) break;
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      try { task(); } catch (...) { /* packaged_task transports exceptions to its future */ }
    }
    inside_worker_ = false;
  }

  inline static thread_local bool inside_worker_{false};
  const std::size_t queue_capacity_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::function<void()>> queue_;
  std::vector<std::jthread> workers_;
  bool stopping_{false};
};

}  // namespace rlm
