#pragma once

#include "kq_candidate2.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace omega_v2_1d {

std::vector<char> canonical_d640_q4_v1(const omega_v2_1::CoreWeights& weights);

void execute_stage_a_calibration(const std::filesystem::path& d512_fp32_stream,
                                const std::filesystem::path& state_inputs_root,
                                const std::filesystem::path& d640_canonical_stream,
                                const std::string& d640_canonical_sha256,
                                const std::filesystem::path& output_root);

} // namespace omega_v2_1d
