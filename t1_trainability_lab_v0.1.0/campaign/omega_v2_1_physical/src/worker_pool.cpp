#include "v2_1.hpp"

#include <cmath>
#include <sstream>

namespace omega_v2_1 {

struct WorkerPool::ThreadState {
    CpuSetRecord cpu_set;
    std::thread thread;
};

WorkerPool::WorkerPool() = default;

WorkerPool::WorkerPool(const std::vector<CpuSetRecord>& workers) {
    std::string error;
    if (!start(workers, error)) throw std::runtime_error(error);
}

WorkerPool::~WorkerPool() { stop(); }

bool WorkerPool::start(const std::vector<CpuSetRecord>& workers, std::string& error) {
    if (!threads_.empty()) {
        error = "worker pool already started";
        return false;
    }
    if (workers.empty()) {
        error = "worker pool requires at least one CPU-set";
        return false;
    }
    shutdown_ = false;
    affinity_intact_.store(true);
    startup_completed_ = 0;
    startup_error_.clear();
    worker_weights_.clear();
    row_ranges_.assign(workers.size(), {0, 0});
    for (const CpuSetRecord& cpu : workers) {
        auto state = std::make_unique<ThreadState>();
        state->cpu_set = cpu;
        cpu_set_ids_.push_back(cpu.id);
        worker_weights_.push_back(cpu.shard_weight > 0.0 ? cpu.shard_weight : 1.0);
        threads_.push_back(std::move(state));
    }
    for (std::size_t i = 0; i < threads_.size(); ++i) {
        threads_[i]->thread = std::thread([this, i, cpu = workers[i]] { worker_main(i, cpu); });
    }
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_ready_.wait(lock, [&] { return startup_completed_ == threads_.size(); });
    }
    if (!startup_error_.empty()) {
        error = startup_error_;
        stop();
        return false;
    }
    ready_.store(true);
    return true;
}

void WorkerPool::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        ++generation_;
    }
    cv_work_.notify_all();
    for (auto& state : threads_) {
        if (state && state->thread.joinable()) state->thread.join();
    }
    threads_.clear();
    ready_.store(false);
}

void WorkerPool::parallel_for(std::size_t count, void* context, void (*function)(void*, std::size_t)) {
    if (!ready_.load() || !function) throw std::runtime_error("worker pool not ready or null job function");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_context_ = context;
        job_function_ = function;
        work_count_ = count;
        const double total_weight = std::accumulate(worker_weights_.begin(), worker_weights_.end(), 0.0);
        std::size_t begin = 0;
        double prefix = 0.0;
        for (std::size_t worker = 0; worker < worker_weights_.size(); ++worker) {
            prefix += worker_weights_[worker];
            const std::size_t end = (worker + 1 == worker_weights_.size())
                ? count
                : static_cast<std::size_t>(std::floor(static_cast<double>(count) * prefix / total_weight));
            row_ranges_[worker] = {begin, end};
            begin = end;
        }
        completed_ = 0;
        ++generation_;
    }
    cv_work_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return completed_ == threads_.size(); });
    job_context_ = nullptr;
    job_function_ = nullptr;
}

std::vector<std::pair<std::size_t, std::size_t>> WorkerPool::row_shards(std::size_t row_count) const {
    std::vector<std::pair<std::size_t, std::size_t>> result;
    result.reserve(worker_weights_.size());
    const double total_weight = std::accumulate(worker_weights_.begin(), worker_weights_.end(), 0.0);
    std::size_t begin = 0;
    double prefix = 0.0;
    for (std::size_t worker = 0; worker < worker_weights_.size(); ++worker) {
        prefix += worker_weights_[worker];
        const std::size_t end = (worker + 1 == worker_weights_.size())
            ? row_count
            : static_cast<std::size_t>(std::floor(static_cast<double>(row_count) * prefix / total_weight));
        result.emplace_back(begin, end);
        begin = end;
    }
    return result;
}

void WorkerPool::worker_main(std::size_t worker_index, CpuSetRecord cpu_set) {
    std::string affinity_error;
    const bool affinity_ok = set_current_cpu_set(cpu_set.id, cpu_set.group, cpu_set.logical_index, affinity_error);
    PROCESSOR_NUMBER current{};
    GetCurrentProcessorNumberEx(&current);
    const bool processor_matches = current.Group == cpu_set.group && current.Number == cpu_set.logical_index;
    const bool final_affinity_ok = affinity_ok && processor_matches;
    std::ostringstream affinity;
    affinity << "worker=" << worker_index << ",cpu_set_id=" << cpu_set.id
             << ",target_group=" << cpu_set.group << ",target_logical=" << static_cast<unsigned>(cpu_set.logical_index)
             << ",observed_group=" << current.Group << ",observed_logical=" << static_cast<unsigned>(current.Number)
             << ",affinity_ok=" << (final_affinity_ok ? "true" : "false");
    if (!final_affinity_ok) affinity << ",error=" << (affinity_ok ? "selected CPU-set did not match observed logical processor" : affinity_error);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        affinity_results_.push_back(affinity.str());
        if (!final_affinity_ok && startup_error_.empty()) startup_error_ = affinity.str();
        ++startup_completed_;
    }
    cv_ready_.notify_one();

    std::uint64_t observed_generation = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_work_.wait(lock, [&] { return shutdown_ || generation_ != observed_generation; });
        if (shutdown_) return;
        observed_generation = generation_;
        void* context = job_context_;
        void (*function)(void*, std::size_t) = job_function_;
        const auto range = row_ranges_[worker_index];
        lock.unlock();

        for (std::size_t index = range.first; index < range.second; ++index) function(context, index);

        lock.lock();
        PROCESSOR_NUMBER after{};
        GetCurrentProcessorNumberEx(&after);
        if (after.Group != cpu_set.group || after.Number != cpu_set.logical_index) affinity_intact_.store(false);
        ++completed_;
        if (completed_ == threads_.size()) cv_done_.notify_one();
    }
}

bool set_current_cpu_set(DWORD id, WORD group, BYTE logical_index, std::string& error) {
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), &id, 1)) {
        const DWORD code = GetLastError();
        error = "SetThreadSelectedCpuSets failed, win32=" + std::to_string(code);
        return false;
    }
    GROUP_AFFINITY affinity{};
    affinity.Group = group;
    affinity.Mask = static_cast<KAFFINITY>(1) << logical_index;
    if (!SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) {
        const DWORD code = GetLastError();
        error = "SetThreadGroupAffinity failed, win32=" + std::to_string(code);
        return false;
    }
    SwitchToThread();
    return true;
}

} // namespace omega_v2_1
