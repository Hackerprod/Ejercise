cmake_minimum_required(VERSION 3.24)
project(omega_v2_1b_candidate01_component_diagnostic LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "Candidate-01 component diagnostic is Windows-native MSVC x64 only")
endif()

get_filename_component(KQ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(V2_1_SOURCE_DIR "${KQ_ROOT}/../omega_v2_1_physical/src" ABSOLUTE)

add_library(candidate01_diag_core STATIC
  "${V2_1_SOURCE_DIR}/hardware_topology.cpp"
  "${V2_1_SOURCE_DIR}/q4_layout.cpp"
  "${V2_1_SOURCE_DIR}/worker_pool.cpp"
  "${V2_1_SOURCE_DIR}/cache_controls.cpp"
  "${V2_1_SOURCE_DIR}/allocation_guard.cpp"
  "${KQ_ROOT}/src/q4_kernel_candidate1.cpp"
)
target_include_directories(candidate01_diag_core PUBLIC "${V2_1_SOURCE_DIR}" "${KQ_ROOT}/src")
target_compile_features(candidate01_diag_core PUBLIC cxx_std_17)
target_compile_definitions(candidate01_diag_core PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1)
target_compile_options(candidate01_diag_core PRIVATE /O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc)
target_link_options(candidate01_diag_core PRIVATE /LTCG)

add_executable(omega_v2_1b_candidate01_diag component_diagnostic.cpp)
target_link_libraries(omega_v2_1b_candidate01_diag PRIVATE candidate01_diag_core)
target_compile_features(omega_v2_1b_candidate01_diag PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1b_candidate01_diag PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1)
target_compile_options(omega_v2_1b_candidate01_diag PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)
