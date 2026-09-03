cmake_minimum_required(VERSION 3.20)
project(relational_language_engine VERSION 1.0.0 LANGUAGES CXX)

option(RLM_BUILD_TESTS "Build the test suite" ON)
option(RLM_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(RLM_ENABLE_LTO "Enable link-time optimization in Release builds" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

add_library(rlm_core
  src/status.cpp
  src/types.cpp
  src/binary.cpp
  src/wal.cpp
  src/config.cpp
  src/embedding_store.cpp
  src/relation_store.cpp
  src/replay.cpp
  src/search.cpp
  src/audit.cpp
  src/promotion.cpp
  src/trainer.cpp
  src/metrics.cpp
  src/baseline.cpp
  src/engine.cpp
)

target_include_directories(rlm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rlm_core PUBLIC Threads::Threads)
target_compile_features(rlm_core PUBLIC cxx_std_20)
target_compile_definitions(rlm_core PRIVATE _FILE_OFFSET_BITS=64)

if(MSVC)
  target_compile_options(rlm_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(rlm_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wnon-virtual-dtor)
endif()

if(RLM_ENABLE_ASAN AND NOT MSVC)
  target_compile_options(rlm_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(rlm_core PUBLIC -fsanitize=address,undefined)
endif()

if(RLM_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT RLM_IPO_SUPPORTED OUTPUT RLM_IPO_ERROR)
  if(RLM_IPO_SUPPORTED)
    set_property(TARGET rlm_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

add_executable(rlmctl apps/rlmctl.cpp)
target_link_libraries(rlmctl PRIVATE rlm_core)
install(TARGETS rlm_core rlmctl RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/rlm/config)

if(RLM_BUILD_TESTS)
  enable_testing()
  add_executable(rlm_tests tests/test_main.cpp)
  target_link_libraries(rlm_tests PRIVATE rlm_core)
  if(NOT MSVC)
    target_compile_options(rlm_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME rlm_all COMMAND rlm_tests)
  set_tests_properties(rlm_all PROPERTIES TIMEOUT 120)
endif()
