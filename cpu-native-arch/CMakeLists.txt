cmake_minimum_required(VERSION 3.24)
project(cpu_native_arch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /O2 /EHsc)
endif()

add_executable(cpu_native_q4_probe
  q4_probe.cpp
)

add_executable(stream_bandwidth
  stream_bandwidth.cpp
)

add_executable(int8_probe
  int8_probe.cpp
)

add_executable(t0m_int8_probe
  t0m_int8_probe.cpp
)

if(MSVC)
  target_compile_options(t0m_int8_probe PRIVATE /arch:AVX2)
endif()

target_compile_features(cpu_native_q4_probe PRIVATE cxx_std_20)
target_compile_features(stream_bandwidth PRIVATE cxx_std_20)
target_compile_features(int8_probe PRIVATE cxx_std_20)
target_compile_features(t0m_int8_probe PRIVATE cxx_std_20)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME cpu_native_q4_probe_self_test
           COMMAND $<TARGET_FILE:cpu_native_q4_probe> --self-test)
  add_test(NAME cpu_native_q4_probe_variant_b_self_test
           COMMAND $<TARGET_FILE:cpu_native_q4_probe> --self-test --variant B)
  add_test(NAME cpu_native_q4_probe_variant_c_self_test
           COMMAND $<TARGET_FILE:cpu_native_q4_probe> --self-test --variant C)
  add_test(NAME stream_bandwidth_smoke
           COMMAND $<TARGET_FILE:stream_bandwidth> --mebibytes 8 --repetitions 1 --cpus 0,2,4,6)
  add_test(NAME int8_probe_self_test
           COMMAND $<TARGET_FILE:int8_probe> --kernel avx2 --m 8 --K 64 --depth 2 --iterations 1 --repetitions 1 --warmup 0)
  add_test(NAME t0m_int8_correction
           COMMAND $<TARGET_FILE:t0m_int8_probe> --self-test)
endif()
