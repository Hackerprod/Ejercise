cmake_minimum_required(VERSION 3.24)
project(omega_v2_1c_residency_only LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "OMEGA-V2-1c requires native Windows x64 MSVC")
endif()

get_filename_component(V2_1_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omega_v2_1_physical/src" ABSOLUTE)
get_filename_component(KQ2_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omega_v2_1b_candidate_02/src" ABSOLUTE)

add_library(omega_v2_1c_candidate02_core STATIC
  "${V2_1_SOURCE_DIR}/hardware_topology.cpp"
  "${V2_1_SOURCE_DIR}/q4_layout.cpp"
  "${V2_1_SOURCE_DIR}/worker_pool.cpp"
  "${V2_1_SOURCE_DIR}/cache_controls.cpp"
  "${V2_1_SOURCE_DIR}/allocation_guard.cpp"
  "${KQ2_SOURCE_DIR}/q4_kernel_candidate2.cpp"
  "${KQ2_SOURCE_DIR}/full_block_candidate2.cpp"
  src/v2_1c_test_helpers.cpp
)
target_include_directories(omega_v2_1c_candidate02_core PUBLIC "${V2_1_SOURCE_DIR}" "${KQ2_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(omega_v2_1c_candidate02_core PUBLIC cxx_std_17)
target_compile_definitions(omega_v2_1c_candidate02_core PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2)
target_compile_options(omega_v2_1c_candidate02_core PRIVATE /O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc)
target_link_options(omega_v2_1c_candidate02_core PRIVATE /LTCG)

add_executable(omega_v2_1c_bench "${V2_1_SOURCE_DIR}/main.cpp" "${V2_1_SOURCE_DIR}/benchmark.cpp" src/select_attempt02_workers.cpp)
target_link_libraries(omega_v2_1c_bench PRIVATE omega_v2_1c_candidate02_core)
target_compile_features(omega_v2_1c_bench PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1c_bench PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2 OMEGA_V2_1C_RESIDENCY_ONLY=1 select_p_cores_by_h0=select_p_cores_by_h0_attempt02)
target_compile_options(omega_v2_1c_bench PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)

add_executable(omega_v2_1c_equivalence src/equivalence_candidate02.cpp)
target_link_libraries(omega_v2_1c_equivalence PRIVATE omega_v2_1c_candidate02_core)
target_compile_features(omega_v2_1c_equivalence PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1c_equivalence PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2 OMEGA_V2_1C_RESIDENCY_ONLY=1)
target_compile_options(omega_v2_1c_equivalence PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)
