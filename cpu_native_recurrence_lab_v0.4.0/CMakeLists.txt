cmake_minimum_required(VERSION 3.20)
project(cpu_native_recurrence_lab VERSION 0.4.0 LANGUAGES CXX)

option(CNRL_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(CNRL_ENABLE_SANITIZERS "Enable available runtime sanitizers" OFF)

find_package(Threads REQUIRED)

add_library(cnrl_core STATIC
  src/types.cpp
  src/platform.cpp
  src/sharding.cpp
  src/state.cpp
  src/weights.cpp
  src/kernels.cpp
  src/transitions.cpp
  src/benchmark.cpp
  src/csv.cpp
)
target_include_directories(cnrl_core PUBLIC include)
target_compile_features(cnrl_core PUBLIC cxx_std_20)
target_link_libraries(cnrl_core PUBLIC Threads::Threads)
target_compile_definitions(cnrl_core PUBLIC CNRL_VERSION_STRING="${PROJECT_VERSION}")

function(cnrl_apply_compile_policy target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      /arch:AVX2 /W4 /permissive- /EHsc /fp:precise
      $<$<CONFIG:Release>:/O2>
      $<$<CONFIG:RelWithDebInfo>:/O2>
    )
    if(CNRL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
    if(CNRL_ENABLE_SANITIZERS)
      target_compile_options(${target} PRIVATE /fsanitize=address)
      target_link_options(${target} PRIVATE /fsanitize=address /INCREMENTAL:NO)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:-O3>
      $<$<CONFIG:RelWithDebInfo>:-O3>
      -mavx2 -mfma -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    if(CNRL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(CNRL_ENABLE_SANITIZERS)
      target_compile_options(${target} PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endfunction()

cnrl_apply_compile_policy(cnrl_core)

function(cnrl_add_app name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE cnrl_core)
  target_compile_features(${name} PRIVATE cxx_std_20)
  cnrl_apply_compile_policy(${name})
endfunction()

cnrl_add_app(cnrl_gate apps/cnrl_gate.cpp)
cnrl_add_app(cnrl_topology apps/cnrl_topology.cpp)
cnrl_add_app(cnrl_calibrate apps/cnrl_calibrate.cpp)
cnrl_add_app(cnrl_bandwidth apps/cnrl_bandwidth.cpp)
cnrl_add_app(cnrl_transition_bench apps/cnrl_transition_bench.cpp)
cnrl_add_app(cnrl_tests tests/test_main.cpp)

enable_testing()
add_test(NAME cnrl_tests COMMAND cnrl_tests)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME cnrl_analyzer_tests
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_analyzer.py")
  add_test(NAME cnrl_cli_pipeline_tests
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tests/test_cli_pipeline.py"
                   --gate $<TARGET_FILE:cnrl_gate>
                   --analyzer "${CMAKE_SOURCE_DIR}/scripts/analyze_results.py")
endif()
