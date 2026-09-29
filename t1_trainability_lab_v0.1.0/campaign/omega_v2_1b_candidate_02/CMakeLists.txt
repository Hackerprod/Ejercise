cmake_minimum_required(VERSION 3.24)
project(omega_v2_1b_candidate_02 LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "OMEGA-V2-1b candidate_02 KQ is Windows-native MSVC x64 only")
endif()

get_filename_component(V2_1_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omega_v2_1_physical/src" ABSOLUTE)

add_library(omega_v2_1b_candidate_02_core STATIC
  "${V2_1_SOURCE_DIR}/hardware_topology.cpp"
  "${V2_1_SOURCE_DIR}/q4_layout.cpp"
  "${V2_1_SOURCE_DIR}/worker_pool.cpp"
  "${V2_1_SOURCE_DIR}/allocation_guard.cpp"
  src/q4_kernel_candidate2.cpp
  src/full_block_candidate2.cpp
  src/platform_candidate2.cpp
)
target_include_directories(omega_v2_1b_candidate_02_core PUBLIC "${V2_1_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(omega_v2_1b_candidate_02_core PUBLIC cxx_std_17)
target_compile_definitions(omega_v2_1b_candidate_02_core PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2)
target_compile_options(omega_v2_1b_candidate_02_core PRIVATE /O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc)
target_link_options(omega_v2_1b_candidate_02_core PRIVATE /LTCG)

add_executable(omega_v2_1b_candidate_02 src/main_candidate2.cpp)
target_link_libraries(omega_v2_1b_candidate_02 PRIVATE omega_v2_1b_candidate_02_core)
target_compile_features(omega_v2_1b_candidate_02 PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1b_candidate_02 PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2)
target_compile_options(omega_v2_1b_candidate_02 PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)

add_executable(omega_v2_1b_candidate_02_correctness src/correctness_candidate2.cpp)
target_link_libraries(omega_v2_1b_candidate_02_correctness PRIVATE omega_v2_1b_candidate_02_core)
target_compile_features(omega_v2_1b_candidate_02_correctness PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1b_candidate_02_correctness PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2)
target_compile_options(omega_v2_1b_candidate_02_correctness PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)
