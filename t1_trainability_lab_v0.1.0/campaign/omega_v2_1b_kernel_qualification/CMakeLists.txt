cmake_minimum_required(VERSION 3.24)
project(omega_v2_1b_kernel_qualification LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "OMEGA-V2-1b qualification is Windows-native MSVC x64 only")
endif()

get_filename_component(V2_1_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omega_v2_1_physical/src" ABSOLUTE)

add_library(omega_v2_1b_kq_core STATIC
  "${V2_1_SOURCE_DIR}/hardware_topology.cpp"
  "${V2_1_SOURCE_DIR}/q4_layout.cpp"
  "${V2_1_SOURCE_DIR}/v2_full_block.cpp"
  "${V2_1_SOURCE_DIR}/worker_pool.cpp"
  "${V2_1_SOURCE_DIR}/cache_controls.cpp"
  "${V2_1_SOURCE_DIR}/allocation_guard.cpp"
  src/q4_kernel_candidate1.cpp
)
target_include_directories(omega_v2_1b_kq_core PUBLIC "${V2_1_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(omega_v2_1b_kq_core PUBLIC cxx_std_17)
target_compile_definitions(omega_v2_1b_kq_core PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=1)
target_compile_options(omega_v2_1b_kq_core PRIVATE /O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc)
target_link_options(omega_v2_1b_kq_core PRIVATE /LTCG)

add_executable(omega_v2_1b_kq src/main.cpp)
target_link_libraries(omega_v2_1b_kq PRIVATE omega_v2_1b_kq_core)
target_compile_features(omega_v2_1b_kq PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1b_kq PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=1)
target_compile_options(omega_v2_1b_kq PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)
