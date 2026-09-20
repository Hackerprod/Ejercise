cmake_minimum_required(VERSION 3.20)

project(omega_recurrent_p2r0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(omega_recurrent STATIC omega_recurrent.cpp)
target_include_directories(omega_recurrent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
  target_compile_options(omega_recurrent PRIVATE /W4 /permissive- /fp:precise)
else()
  target_compile_options(omega_recurrent PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

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
