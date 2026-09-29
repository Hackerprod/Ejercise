cmake_minimum_required(VERSION 3.24)
project(omega_v2_1_bench LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "OMEGA-V2-1 primary benchmark is Windows-native MSVC x64 only")
endif()

add_library(omega_v2_1_core STATIC
  src/hardware_topology.cpp
  src/q4_layout.cpp
  src/q4_kernel.cpp
  src/v2_full_block.cpp
  src/worker_pool.cpp
  src/cache_controls.cpp
  src/allocation_guard.cpp
)
target_include_directories(omega_v2_1_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(omega_v2_1_core PUBLIC cxx_std_17)
target_compile_definitions(omega_v2_1_core PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1)
target_compile_options(omega_v2_1_core PRIVATE /O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc)
target_link_options(omega_v2_1_core PRIVATE /LTCG)

add_executable(omega_v2_1_bench src/main.cpp src/benchmark.cpp)
target_link_libraries(omega_v2_1_bench PRIVATE omega_v2_1_core)
target_compile_features(omega_v2_1_bench PRIVATE cxx_std_17)
target_compile_definitions(omega_v2_1_bench PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 OMEGA_V2_1_AVX2=1)
target_compile_options(omega_v2_1_bench PRIVATE /O2 /arch:AVX2 /fp:precise /W4 /EHsc)
