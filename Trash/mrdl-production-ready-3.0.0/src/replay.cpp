#include "mrdl/replay.hpp"

namespace mrdl {
namespace {

void write_lane(BinaryWriter& writer, const LaneReplayData& lane) {
    writer.vector<GateDecision>(lane.gate_decisions);
    writer.pod(lane.fold_budget);
    writer.vector<BranchId>(lane.survivor_ids);
    writer.pod(lane.shadow_upper_bound);
    writer.pod(lane.candidate_set_hash);
}

LaneReplayData read_lane(BinaryReader& reader) {
    LaneReplayData lane;
    lane.gate_decisions = reader.vector<GateDecision>();
    lane.fold_budget = reader.pod<std::uint32_t>();
    lane.survivor_ids = reader.vector<BranchId>();
    lane.shadow_upper_bound = reader.pod<float>();
    lane.candidate_set_hash = reader.pod<std::uint64_t>();
    return lane;
}

}  // namespace

std::vector<std::byte> ReplayStep::serialize() const {
    BinaryWriter writer;
    writer.pod(id);
    writer.pod(operation_id);
    writer.pod(controller_version);
    writer.pod<std::uint64_t>(relation_versions.size());
    for (const auto& [relation, version] : relation_versions) { writer.pod(relation); writer.pod(version); }
    writer.vector<BranchId>(parent_branch_ids);
    write_lane(writer, lanes[0]);
    write_lane(writer, lanes[1]);
    writer.pod(deterministic_seed);
    writer.pod(depth);
    return writer.take();
}

ReplayStep ReplayStep::deserialize(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    ReplayStep step;
    step.id = reader.pod<ReplayId>();
    step.operation_id = reader.pod<std::uint64_t>();
    step.controller_version = reader.pod<std::uint64_t>();
    const auto relation_count = reader.pod<std::uint64_t>();
    step.relation_versions.reserve(static_cast<std::size_t>(relation_count));
    for (std::uint64_t i = 0; i < relation_count; ++i) {
        const auto relation = reader.pod<RelationId>();
        const auto version = reader.pod<std::uint64_t>();
        step.relation_versions.emplace_back(relation, version);
    }
    step.parent_branch_ids = reader.vector<BranchId>();
    step.lanes[0] = read_lane(reader);
    step.lanes[1] = read_lane(reader);
    step.deterministic_seed = reader.pod<std::uint64_t>();
    step.depth = reader.pod<std::uint32_t>();
    require(reader.empty(), "trailing replay step payload");
    return step;
}

bool ReplayClosure::complete() const noexcept {
    if (root_relation == 0U || root_version == 0U || operation_id == 0U || base_seed == 0U ||
        controller_version == 0U || controller_snapshot.empty() || role_snapshot.empty() || replay_steps.empty() ||
        relation_versions.empty() || deterministic_seeds.size() != replay_steps.size() ||
        snapshot_hashes.size() != relation_snapshots.size()) return false;
    for (const auto& [relation, version] : relation_versions) {
        (void)version;
        const auto snapshot = std::find_if(relation_snapshots.begin(), relation_snapshots.end(),
            [&](const auto& item) { return item.relation == relation && item.version == version; });
        if (snapshot == relation_snapshots.end() || snapshot->payload.empty()) return false;
    }
    return true;
}

std::vector<std::byte> ReplayClosure::serialize() const {
    BinaryWriter writer;
    writer.pod(root_relation);
    writer.pod(root_version);
    writer.pod(operation_id);
    writer.pod(base_seed);
    writer.vector<ReplayId>(replay_steps);
    writer.pod<std::uint64_t>(relation_versions.size());
    for (const auto& [relation, version] : relation_versions) { writer.pod(relation); writer.pod(version); }
    writer.pod<std::uint64_t>(relation_snapshots.size());
    for (const auto& snapshot : relation_snapshots) {
        writer.pod(snapshot.relation);
        writer.pod(snapshot.version);
        writer.vector<std::byte>(snapshot.payload);
    }
    writer.pod(controller_version);
    writer.vector<std::byte>(controller_snapshot);
    writer.vector<std::byte>(role_snapshot);
    writer.vector<std::uint64_t>(deterministic_seeds);
    writer.vector<std::uint64_t>(snapshot_hashes);
    writer.vector<std::uint64_t>(binding_hashes);
    return writer.take();
}

ReplayClosure ReplayClosure::deserialize(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    ReplayClosure closure;
    closure.root_relation = reader.pod<RelationId>();
    closure.root_version = reader.pod<std::uint64_t>();
    closure.operation_id = reader.pod<std::uint64_t>();
    closure.base_seed = reader.pod<std::uint64_t>();
    closure.replay_steps = reader.vector<ReplayId>();
    const auto relation_count = reader.pod<std::uint64_t>();
    closure.relation_versions.reserve(static_cast<std::size_t>(relation_count));
    for (std::uint64_t i = 0; i < relation_count; ++i) {
        const auto relation = reader.pod<RelationId>();
        const auto version = reader.pod<std::uint64_t>();
        closure.relation_versions.emplace_back(relation, version);
    }
    const auto snapshot_count = reader.pod<std::uint64_t>();
    closure.relation_snapshots.reserve(static_cast<std::size_t>(snapshot_count));
    for (std::uint64_t i = 0; i < snapshot_count; ++i) {
        RelationSnapshot snapshot;
        snapshot.relation = reader.pod<RelationId>();
        snapshot.version = reader.pod<std::uint64_t>();
        snapshot.payload = reader.vector<std::byte>();
        closure.relation_snapshots.push_back(std::move(snapshot));
    }
    closure.controller_version = reader.pod<std::uint64_t>();
    closure.controller_snapshot = reader.vector<std::byte>();
    closure.role_snapshot = reader.vector<std::byte>();
    closure.deterministic_seeds = reader.vector<std::uint64_t>();
    closure.snapshot_hashes = reader.vector<std::uint64_t>();
    closure.binding_hashes = reader.vector<std::uint64_t>();
    require(reader.empty(), "trailing replay closure payload");
    return closure;
}

ReplayRecorder::ReplayRecorder(std::shared_ptr<IReplayRepository> repository)
    : repository_(std::move(repository)) {
    if (repository_) next_id_.store(repository_->max_step_id() + 1U, std::memory_order_relaxed);
}

ReplayId ReplayRecorder::record(ReplayStep step) {
    if (step.id == 0U) step.id = next_id_.fetch_add(1U, std::memory_order_relaxed);
    if (repository_) repository_->save_step(step);
    std::unique_lock lock(mutex_);
    next_id_.store(std::max(next_id_.load(std::memory_order_relaxed), step.id + 1U), std::memory_order_relaxed);
    steps_[step.id] = std::move(step);
    return step.id;
}

std::optional<ReplayStep> ReplayRecorder::get(ReplayId id) const {
    {
        std::shared_lock lock(mutex_);
        const auto it = steps_.find(id);
        if (it != steps_.end()) return it->second;
    }
    return repository_ ? repository_->load_step(id) : std::optional<ReplayStep>{};
}

bool ReplayRecorder::erase(ReplayId id) {
    if (repository_) repository_->delete_step(id);
    std::unique_lock lock(mutex_);
    return steps_.erase(id) != 0U;
}

bool ReplayRecorder::closure_available(const ReplayClosure& closure) const {
    if (!closure.complete()) return false;
    for (const ReplayId id : closure.replay_steps) {
        if (!get(id)) return false;
    }
    return true;
}

}  // namespace mrdl
