cmake_minimum_required(VERSION 3.20)

project(omega_recurrent_p2r0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OMEGA_PROFILE_INTERNAL "Compile temporary native profile instrumentation" OFF)
option(OMEGA_P2R_DIAGNOSTIC "Build isolated P2-R-OPT2 diagnostic targets" OFF)
option(OMEGA_FC2_REPLAY_CAPTURE "Build local FC2 M=8 replay capture library" OFF)
set(OMEGA_DIAGNOSTIC_ELIDE_MASK "0" CACHE STRING "Compile-time P2-R-OPT2 residual diagnostic mask")

if(OMEGA_PROFILE_INTERNAL AND OMEGA_P2R_DIAGNOSTIC)
  message(FATAL_ERROR "OMEGA_PROFILE_INTERNAL and OMEGA_P2R_DIAGNOSTIC are mutually exclusive")
endif()

if(OMEGA_FC2_REPLAY_CAPTURE)
  add_library(omega_recurrent_fc2_capture SHARED omega_recurrent.cpp)
  target_include_directories(omega_recurrent_fc2_capture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(omega_recurrent_fc2_capture PRIVATE
    OMEGA_RECURRENT_BUILD_SHARED=1
    OMEGA_FC2_REPLAY_CAPTURE=1
  )
  set_target_properties(omega_recurrent_fc2_capture PROPERTIES
    OUTPUT_NAME omega_recurrent_fc2_capture
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
  )
  foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(omega_recurrent_fc2_capture PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
      LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
      ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
    )
  endforeach()
  if(MSVC)
    target_compile_options(omega_recurrent_fc2_capture PRIVATE /W4 /permissive- /fp:precise /arch:AVX2)
  else()
    target_compile_options(omega_recurrent_fc2_capture PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -O2 -mavx2)
  endif()
endif()

add_library(omega_recurrent STATIC omega_recurrent.cpp)
set_target_properties(omega_recurrent PROPERTIES OUTPUT_NAME omega_recurrent_static)

add_library(omega_recurrent_shared SHARED omega_recurrent.cpp)
target_compile_definitions(omega_recurrent_shared PRIVATE OMEGA_RECURRENT_BUILD_SHARED=1)
set_target_properties(omega_recurrent_shared PROPERTIES
  OUTPUT_NAME omega_recurrent
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set_target_properties(omega_recurrent_shared PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
    LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
    ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
  )
endforeach()

foreach(target omega_recurrent omega_recurrent_shared)
  target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  if(OMEGA_PROFILE_INTERNAL)
    target_compile_definitions(${target} PRIVATE OMEGA_PROFILE_INTERNAL=1)
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /fp:precise)
    if(target STREQUAL "omega_recurrent_shared")
      target_compile_options(${target} PRIVATE /arch:AVX2)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endforeach()

add_executable(omega_recurrent_test omega_recurrent_test.cpp)
target_link_libraries(omega_recurrent_test PRIVATE omega_recurrent)

add_executable(omega_recurrent_shared_test omega_recurrent_test.cpp)
target_link_libraries(omega_recurrent_shared_test PRIVATE omega_recurrent_shared)
set_target_properties(omega_recurrent_shared_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set_target_properties(omega_recurrent_shared_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
  )
endforeach()

add_executable(omega_recurrent_instrumentation_equivalence_test omega_recurrent_instrumentation_equivalence_test.cpp)
target_link_libraries(omega_recurrent_instrumentation_equivalence_test PRIVATE omega_recurrent)

add_executable(omega_recurrent_matrix_view_test omega_recurrent_matrix_view_test.cpp)
target_link_libraries(omega_recurrent_matrix_view_test PRIVATE omega_recurrent)

add_executable(omega_recurrent_runtime_test omega_recurrent_runtime_test.cpp)
target_link_libraries(omega_recurrent_runtime_test PRIVATE omega_recurrent)

add_executable(omega_recurrent_shared_runtime_test omega_recurrent_runtime_test.cpp)
target_link_libraries(omega_recurrent_shared_runtime_test PRIVATE omega_recurrent_shared)
set_target_properties(omega_recurrent_shared_runtime_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set_target_properties(omega_recurrent_shared_runtime_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
  )
endforeach()

add_executable(omega_depth_reduction_local_test omega_depth_reduction_local_test.cpp)

add_executable(omega_fc1_microbenchmark fc1_microbenchmark.cpp)
add_executable(omega_fc2_microbenchmark fc2_microbenchmark.cpp)
add_executable(omega_qkv_microbenchmark qkv_microbenchmark.cpp)
add_executable(omega_out_projection_microbenchmark out_projection_microbenchmark.cpp)
add_executable(omega_fc2_dweight_local_cost_diagnostic fc2_dweight_local_cost_diagnostic.cpp)
target_include_directories(omega_fc2_dweight_local_cost_diagnostic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
  target_compile_options(omega_recurrent_test PRIVATE /W4 /permissive- /fp:precise)
  target_compile_options(omega_recurrent_shared_test PRIVATE /W4 /permissive- /fp:precise)
  target_compile_options(omega_recurrent_instrumentation_equivalence_test PRIVATE /W4 /permissive- /fp:precise)
  target_compile_options(omega_recurrent_matrix_view_test PRIVATE /W4 /permissive- /fp:precise)
  target_compile_options(omega_recurrent_runtime_test PRIVATE /W4 /permissive- /fp:precise)
  target_compile_options(omega_recurrent_shared_runtime_test PRIVATE /W4 /permissive- /fp:precise)
  target_compile_options(omega_depth_reduction_local_test PRIVATE /O2 /W4 /permissive- /fp:precise /arch:AVX2)
  target_compile_options(omega_fc1_microbenchmark PRIVATE /O2 /W4 /permissive- /fp:precise /arch:AVX2)
  target_compile_options(omega_fc2_microbenchmark PRIVATE /O2 /W4 /permissive- /fp:precise /arch:AVX2)
  target_compile_options(omega_qkv_microbenchmark PRIVATE /O2 /W4 /permissive- /fp:precise /arch:AVX2)
  target_compile_options(omega_out_projection_microbenchmark PRIVATE /O2 /W4 /permissive- /fp:precise /arch:AVX2)
  target_compile_options(omega_fc2_dweight_local_cost_diagnostic PRIVATE /O2 /W4 /permissive- /fp:precise /arch:AVX2)
else()
  target_compile_options(omega_recurrent_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_recurrent_shared_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_recurrent_instrumentation_equivalence_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_recurrent_matrix_view_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_recurrent_runtime_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_recurrent_shared_runtime_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_depth_reduction_local_test PRIVATE -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_fc1_microbenchmark PRIVATE -O2 -mavx2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_fc2_microbenchmark PRIVATE -O2 -mavx2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_qkv_microbenchmark PRIVATE -O2 -mavx2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_out_projection_microbenchmark PRIVATE -O2 -mavx2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  target_compile_options(omega_fc2_dweight_local_cost_diagnostic PRIVATE -O2 -mavx2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(OMEGA_PROFILE_INTERNAL)
  target_compile_definitions(omega_recurrent_runtime_test PRIVATE OMEGA_PROFILE_INTERNAL=1)
endif()

if(OMEGA_P2R_DIAGNOSTIC)
  add_library(omega_recurrent_diagnostic STATIC omega_recurrent.cpp)
  set_target_properties(omega_recurrent_diagnostic PROPERTIES OUTPUT_NAME omega_recurrent_diagnostic_static)
  add_library(omega_recurrent_diagnostic_shared SHARED omega_recurrent.cpp)
  target_compile_definitions(omega_recurrent_diagnostic_shared PRIVATE OMEGA_RECURRENT_BUILD_SHARED=1)
  set_target_properties(omega_recurrent_diagnostic_shared PROPERTIES
    OUTPUT_NAME omega_recurrent_diagnostic
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
  )
  foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(omega_recurrent_diagnostic_shared PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
      LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
      ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_CURRENT_BINARY_DIR}/python"
    )
  endforeach()
  foreach(target omega_recurrent_diagnostic omega_recurrent_diagnostic_shared)
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${target} PRIVATE
      OMEGA_P2R_DIAGNOSTIC=1
      OMEGA_DIAGNOSTIC_ELIDE_MASK=${OMEGA_DIAGNOSTIC_ELIDE_MASK}
    )
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive- /fp:precise)
      if(target STREQUAL "omega_recurrent_diagnostic_shared")
        target_compile_options(${target} PRIVATE /arch:AVX2)
      endif()
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
  endforeach()
  add_executable(omega_recurrent_diagnostic_test omega_recurrent_runtime_test.cpp)
  target_link_libraries(omega_recurrent_diagnostic_test PRIVATE omega_recurrent_diagnostic)
  target_compile_definitions(omega_recurrent_diagnostic_test PRIVATE
    OMEGA_P2R_DIAGNOSTIC=1
    OMEGA_DIAGNOSTIC_ELIDE_MASK=${OMEGA_DIAGNOSTIC_ELIDE_MASK}
  )
  if(MSVC)
    target_compile_options(omega_recurrent_diagnostic_test PRIVATE /W4 /permissive- /fp:precise)
  else()
    target_compile_options(omega_recurrent_diagnostic_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endif()

enable_testing()
add_test(
  NAME omega_recurrent_forward_goldens
  COMMAND omega_recurrent_test ${CMAKE_CURRENT_SOURCE_DIR}/../golden_cases
)
add_test(
  NAME omega_recurrent_shared_forward_goldens
  COMMAND omega_recurrent_shared_test ${CMAKE_CURRENT_SOURCE_DIR}/../golden_cases
)
add_test(
  NAME omega_recurrent_matrix_view
  COMMAND omega_recurrent_matrix_view_test
)
add_test(
  NAME omega_recurrent_runtime
  COMMAND omega_recurrent_runtime_test
)
add_test(
  NAME omega_recurrent_shared_runtime
  COMMAND omega_recurrent_shared_runtime_test
)
add_test(
  NAME omega_depth_reduction_local
  COMMAND omega_depth_reduction_local_test
)
