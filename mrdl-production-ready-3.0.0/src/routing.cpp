#include "mrdl/routing.hpp"

#include <numeric>

namespace mrdl {
namespace {

float entropy_from_histogram(const std::unordered_map<std::uint64_t, std::uint32_t>& histogram) {
    std::uint64_t total = 0;
    for (const auto& [_, count] : histogram) total += count;
    if (total == 0U) return 0.0F;
    float result = 0.0F;
    for (const auto& [_, count] : histogram) {
        const float probability = static_cast<float>(count) / static_cast<float>(total);
        result -= probability * std::log(std::max(probability, 1.0e-12F));
    }
    return result;
}

}  // namespace

OnePassPortRouter::OnePassPortRouter(const EngineConfig& config) : config_(config) {}

void OnePassPortRouter::begin_round() {
    for (auto& [_, node_ports] : ports_) {
        for (auto& port : node_ports) {
            port.current_load = 0U;
            port.accumulated_energy = 0.0F;
            std::fill(port.accumulated_key.begin(), port.accumulated_key.end(), 0.0F);
        }
    }
    key_histograms_.clear();
}

std::uint64_t OnePassPortRouter::quantized_key_hash(std::span<const float> key) {
    std::uint64_t hash = 0x504f5254ULL;
    for (const float value : key) {
        const auto quantized = static_cast<std::int16_t>(std::clamp(static_cast<int>(std::nearbyint(value * 16.0F)), -32767, 32767));
        hash = hash_combine(hash, static_cast<std::uint16_t>(quantized));
    }
    return hash;
}

float OnePassPortRouter::pressure(NodeId node) const {
    const auto histogram = key_histograms_.find(node);
    const float entropy = histogram == key_histograms_.end() ? 0.0F : entropy_from_histogram(histogram->second);
    std::uint64_t capsules = 0;
    if (histogram != key_histograms_.end()) {
        for (const auto& [_, count] : histogram->second) capsules += count;
    }
    const auto ports = ports_.find(node);
    const std::size_t count = ports == ports_.end() ? 0U : ports->second.size();
    return static_cast<float>(capsules) * (1.0F + 0.5F * entropy) / (1.0F + static_cast<float>(count));
}

ContextPort& OnePassPortRouter::create_port(NodeId node, std::span<const float> key) {
    auto& list = ports_[node];
    ContextPort port;
    port.id = next_port_id_++;
    port.frozen_key.assign(key.begin(), key.end());
    normalize_in_place(port.frozen_key);
    port.capacity = config_.port_capacity;
    port.accumulated_key.assign(key.size(), 0.0F);
    list.push_back(std::move(port));
    return list.back();
}

PortAssignment OnePassPortRouter::route(NodeId node, std::span<const float> key, float energy) {
    require(!key.empty(), "port key cannot be empty");
    ++assignment_count_;
    const auto key_hash = quantized_key_hash(key);
    ++key_histograms_[node][key_hash];
    auto& list = ports_[node];

    struct Match { float similarity; std::size_t index; };
    std::vector<Match> matches;
    matches.reserve(list.size());
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (list[i].current_load >= list[i].capacity) continue;
        matches.push_back(Match{cosine(list[i].frozen_key, key), i});
    }
    std::sort(matches.begin(), matches.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.similarity != rhs.similarity) return lhs.similarity > rhs.similarity;
        return lhs.index < rhs.index;
    });

    PortAssignment assignment;
    if (!matches.empty() && matches.front().similarity >= config_.port_similarity_threshold) {
        assignment.count = 1U;
        assignment.ports[0] = list[matches.front().index].id;
        assignment.energies[0] = energy;
        if (matches.size() > 1U && matches[1].similarity >= config_.port_similarity_threshold &&
            matches.front().similarity - matches[1].similarity <= 0.05F) {
            assignment.count = 2U;
            assignment.ports[1] = list[matches[1].index].id;
            assignment.energies[0] = energy * 0.5F;
            assignment.energies[1] = energy - assignment.energies[0];
        }
    } else if (list.size() < config_.max_ports_per_node &&
               (list.empty() || pressure(node) >= config_.port_pressure_threshold)) {
        auto& port = create_port(node, key);
        assignment.count = 1U;
        assignment.ports[0] = port.id;
        assignment.energies[0] = energy;
    } else if (!matches.empty()) {
        assignment.count = 1U;
        assignment.ports[0] = list[matches.front().index].id;
        assignment.energies[0] = energy;
        assignment.overflow = true;
    } else {
        assignment.overflow = true;
        return assignment;
    }

    for (std::size_t n = 0; n < assignment.count; ++n) {
        auto it = std::find_if(list.begin(), list.end(), [&](const ContextPort& port) { return port.id == assignment.ports[n]; });
        require(it != list.end(), "assigned port not found");
        ++it->current_load;
        it->utility += assignment.energies[n];
        it->accumulated_energy += assignment.energies[n];
        for (std::size_t i = 0; i < key.size(); ++i) {
            it->accumulated_key[i] = std::fma(assignment.energies[n], key[i], it->accumulated_key[i]);
        }
    }
    return assignment;
}

void OnePassPortRouter::end_round() {
    // Keys were frozen for the complete round. EMA occurs only now.
    for (auto& [_, node_ports] : ports_) {
        for (auto& port : node_ports) {
            if (port.accumulated_energy <= 1.0e-12F) continue;
            for (std::size_t i = 0; i < port.frozen_key.size(); ++i) {
                const float observation = port.accumulated_key[i] / port.accumulated_energy;
                port.frozen_key[i] = std::fma(0.10F, observation - port.frozen_key[i], port.frozen_key[i]);
            }
            normalize_in_place(port.frozen_key);
        }
    }
}

std::size_t OnePassPortRouter::port_count(NodeId node) const {
    const auto it = ports_.find(node);
    return it == ports_.end() ? 0U : it->second.size();
}

std::vector<Branch> DiverseBeamFold::fold(std::vector<Branch> branches, std::size_t budget) const {
    if (branches.empty() || budget == 0U) return {};

    struct MergeKey {
        TokenId candidate;
        std::uint64_t structure;
        bool operator==(const MergeKey&) const = default;
    };
    struct MergeKeyHash {
        std::size_t operator()(const MergeKey& key) const noexcept {
            return static_cast<std::size_t>(hash_combine(key.candidate, key.structure));
        }
    };

    std::unordered_map<MergeKey, std::vector<std::size_t>, MergeKeyHash> groups;
    groups.reserve(branches.size());
    for (std::size_t i = 0; i < branches.size(); ++i) {
        groups[MergeKey{branches[i].candidate, branches[i].capsule.composed_transform.structure_hash()}].push_back(i);
    }

    std::vector<Branch> merged;
    merged.reserve(groups.size());
    for (const auto& [_, indices] : groups) {
        if (indices.size() == 1U) {
            merged.push_back(std::move(branches[indices.front()]));
            continue;
        }
        std::size_t best = indices.front();
        std::vector<MonomialOperator> operators;
        std::vector<float> weights;
        operators.reserve(indices.size());
        weights.reserve(indices.size());
        float energy = 0.0F;
        float max_score = -std::numeric_limits<float>::infinity();
        for (const auto index : indices) {
            if (branches[index].score > branches[best].score) best = index;
            operators.push_back(branches[index].capsule.composed_transform);
            const float weight = std::max(branches[index].capsule.energy, 1.0e-6F);
            weights.push_back(weight);
            energy += weight;
            max_score = std::max(max_score, branches[index].score);
        }
        Branch branch = std::move(branches[best]);
        branch.capsule.composed_transform = MonomialOperator::blend_same_structure(operators, weights);
        branch.capsule.energy = energy;
        branch.score = max_score + std::log1p(static_cast<float>(indices.size())) * 0.05F;
        merged.push_back(std::move(branch));
    }

    std::sort(merged.begin(), merged.end(), [](const Branch& lhs, const Branch& rhs) {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.candidate != rhs.candidate) return lhs.candidate < rhs.candidate;
        return lhs.capsule.route_signature < rhs.capsule.route_signature;
    });

    std::unordered_map<TokenId, std::size_t> selected_by_token;
    std::vector<Branch> selected;
    selected.reserve(std::min(budget, merged.size()));
    // Budget is deliberately small. A bounded greedy pass avoids penalizing a token
    // merely because an earlier branch was inspected but never survived the beam.
    while (!merged.empty() && selected.size() < budget) {
        std::size_t best = 0U;
        float best_adjusted = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < merged.size(); ++i) {
            const float adjusted = merged[i].score - duplicate_penalty_ *
                static_cast<float>(selected_by_token[merged[i].candidate]);
            if (adjusted > best_adjusted ||
                (adjusted == best_adjusted && merged[i].candidate < merged[best].candidate)) {
                best = i;
                best_adjusted = adjusted;
            }
        }
        Branch chosen = std::move(merged[best]);
        chosen.score = best_adjusted;
        ++selected_by_token[chosen.candidate];
        selected.push_back(std::move(chosen));
        if (best + 1U != merged.size()) merged[best] = std::move(merged.back());
        merged.pop_back();
    }
    std::sort(selected.begin(), selected.end(), [](const Branch& lhs, const Branch& rhs) {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.candidate != rhs.candidate) return lhs.candidate < rhs.candidate;
        return lhs.capsule.route_signature < rhs.capsule.route_signature;
    });
    return selected;
}

void ContextHistory::observe_token(TokenId token) {
    tokens_.push_back(token);
    if (tokens_.size() > capacity_) tokens_.erase(tokens_.begin());
}

void ContextHistory::observe_route(std::uint64_t signature) {
    routes_.push_back(signature);
    if (routes_.size() > capacity_) routes_.erase(routes_.begin());
}

void ContextHistory::observe_edge(RelationId relation) {
    ++edge_use_[relation];
    if (edge_use_.size() > capacity_ * 4U) {
        for (auto it = edge_use_.begin(); it != edge_use_.end();) {
            if (it->second <= 1U) it = edge_use_.erase(it);
            else { it->second /= 2U; ++it; }
        }
    }
}

float ContextHistory::repetition_penalty(TokenId token) const {
    float penalty = 0.0F;
    for (std::size_t distance = 1U; distance <= tokens_.size(); ++distance) {
        if (tokens_[tokens_.size() - distance] == token) penalty += 1.0F / static_cast<float>(distance);
    }
    return penalty;
}

float ContextHistory::cycle_penalty(std::uint64_t next_signature) const {
    float penalty = 0.0F;
    for (std::size_t distance = 1U; distance <= routes_.size(); ++distance) {
        if (routes_[routes_.size() - distance] == next_signature) penalty += 1.0F / static_cast<float>(distance);
    }
    return penalty;
}

float ContextHistory::saturation(RelationId relation) const {
    const auto it = edge_use_.find(relation);
    return it == edge_use_.end() ? 0.0F : std::log1p(static_cast<float>(it->second));
}

bool ContextHistory::short_loop_detected(TokenId candidate) const {
    if (tokens_.size() < 3U) return false;
    const std::size_t n = tokens_.size();
    if (tokens_[n - 1U] == candidate && tokens_[n - 2U] == candidate) return true;
    if (n >= 4U && tokens_[n - 2U] == candidate && tokens_[n - 1U] == tokens_[n - 3U]) return true;
    return false;
}

std::uint64_t ContextHistory::compressed_hash() const noexcept {
    std::uint64_t hash = 0x48495354ULL;
    for (const TokenId token : tokens_) hash = hash_combine(hash, token);
    for (const auto route : routes_) hash = hash_combine(hash, route);
    return hash;
}

std::vector<float> make_port_key(const RouteCapsule& capsule, std::size_t dimension) {
    require(dimension >= 4U, "port key dimension too small");
    std::vector<float> key(dimension, 0.0F);
    std::uint64_t hash = capsule.route_signature;
    hash = hash_combine(hash, capsule.current_node);
    for (const auto& binding : capsule.role_bindings) {
        hash = hash_combine(hash, binding.role);
        hash = hash_combine(hash, binding.entity);
    }
    for (const auto& expectation : capsule.open_expectations) hash = hash_combine(hash, expectation.key);
    for (std::size_t i = 0; i < dimension; ++i) {
        hash = mix64(hash + i);
        key[i] = (hash >> 63U) != 0U ? 1.0F : -1.0F;
    }
    if (!capsule.contextual_state.empty()) {
        for (std::size_t i = 0; i < dimension; ++i) {
            key[i] += capsule.contextual_state[(i * capsule.contextual_state.size()) / dimension];
        }
    }
    normalize_in_place(key);
    return key;
}

std::uint64_t structural_slot_key(std::span<const TokenId> context,
                                  std::size_t position,
                                  std::uint64_t route_signature) {
    std::uint64_t hash = hash_combine(0x534c4f54ULL, route_signature);
    const TokenId left = position > 0U ? context[position - 1U] : kBosToken;
    const TokenId right = position + 1U < context.size() ? context[position + 1U] : kEosToken;
    hash = hash_combine(hash, left);
    hash = hash_combine(hash, right);
    hash = hash_combine(hash, position % 8U);
    return hash;
}

}  // namespace mrdl
