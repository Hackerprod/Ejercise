cmake_minimum_required(VERSION 3.20)

project(omega_recurrent_p2r0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /fp:precise)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endforeach()

add_executable(omega_recurrent_test omega_recurrent_test.cpp)
target_link_libraries(omega_recurrent_test PRIVATE omega_recurrent)

if(MSVC)
  target_compile_options(omega_recurrent_test PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(omega_recurrent_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

enable_testing()
add_test(
  NAME omega_recurrent_forward_goldens
  COMMAND omega_recurrent_test ${CMAKE_CURRENT_SOURCE_DIR}/../golden_cases
)
