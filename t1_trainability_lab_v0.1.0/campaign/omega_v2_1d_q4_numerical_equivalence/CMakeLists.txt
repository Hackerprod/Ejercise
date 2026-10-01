cmake_minimum_required(VERSION 3.24)
project(omega_v2_1d_stage_a LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "OMEGA-V2-1d companion requires native Windows x64 MSVC")
endif()

get_filename_component(V2_1_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omega_v2_1_physical/src" ABSOLUTE)
get_filename_component(KQ2_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omega_v2_1b_candidate_02/src" ABSOLUTE)

add_executable(omega_v2_1d_stage_a
  "${V2_1_SOURCE_DIR}/q4_layout.cpp"
  "${V2_1_SOURCE_DIR}/worker_pool.cpp"
  "${V2_1_SOURCE_DIR}/allocation_guard.cpp"
  "${KQ2_SOURCE_DIR}/q4_kernel_candidate2.cpp"
  "${KQ2_SOURCE_DIR}/full_block_candidate2.cpp"
  src/stage_a_trace.cpp
  src/stage_a_metrics.cpp
  src/stage_a_execution.cpp
  src/stage_a_main.cpp
)
target_include_directories(omega_v2_1d_stage_a PRIVATE
  "${V2_1_SOURCE_DIR}"
  "${KQ2_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(omega_v2_1d_stage_a PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1d_stage_a PRIVATE
  WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00
  OMEGA_V2_1_AVX2=1 OMEGA_V2_1B_CANDIDATE_ID=2
  OMEGA_V2_1D_STAGE_A=1
)
target_compile_options(omega_v2_1d_stage_a PRIVATE /O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc)
target_link_options(omega_v2_1d_stage_a PRIVATE /LTCG)
