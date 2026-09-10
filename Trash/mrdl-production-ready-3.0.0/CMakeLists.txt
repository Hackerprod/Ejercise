cmake_minimum_required(VERSION 3.24)
project(mrdl VERSION 3.0.0 LANGUAGES CXX DESCRIPTION "Sparse relational language runtime")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(MRDL_BUILD_TESTS "Build MRDL tests" ON)
option(MRDL_BUILD_BENCHMARKS "Build MRDL benchmarks" ON)
option(MRDL_NATIVE_ARCH "Optimize release binaries for the build host CPU" ON)
option(MRDL_ENABLE_SANITIZERS "Enable ASan/UBSan" OFF)
option(MRDL_ENABLE_LTO "Enable link-time optimization in release builds" ON)
option(MRDL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

configure_file(
    cmake/version.hpp.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/mrdl/version.hpp"
    @ONLY
)

function(mrdl_apply_build_policy target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-Wnon-virtual-dtor>
        $<$<AND:$<BOOL:${MRDL_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:GNU,Clang>>:-Werror>
        $<$<AND:$<BOOL:${MRDL_NATIVE_ARCH}>,$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-march=native;-mtune=native>
    )
    if(MRDL_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

add_library(mrdl_core
    src/config.cpp
    src/tokenizer.cpp
    src/embeddings.cpp
    src/relation.cpp
    src/graph.cpp
    src/controller.cpp
    src/routing.cpp
    src/replay.cpp
    src/persistence.cpp
    src/promotion.cpp
    src/engine.cpp
    src/training.cpp
    src/baselines.cpp
    src/metrics.cpp
    src/process_lock.cpp
    src/diagnostics.cpp
)
add_library(mrdl::core ALIAS mrdl_core)
set_target_properties(mrdl_core PROPERTIES EXPORT_NAME core)

target_include_directories(mrdl_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mrdl_core PUBLIC SQLite::SQLite3 Threads::Threads)
target_compile_features(mrdl_core PUBLIC cxx_std_20)
target_compile_definitions(mrdl_core PUBLIC _FILE_OFFSET_BITS=64)
mrdl_apply_build_policy(mrdl_core)

set(MRDL_IPO_OK FALSE)
if(MRDL_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT MRDL_IPO_OK OUTPUT MRDL_IPO_ERROR)
    if(MRDL_IPO_OK)
        set_property(TARGET mrdl_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(WARNING "MRDL LTO unavailable: ${MRDL_IPO_ERROR}")
    endif()
endif()

add_executable(mrdl src/main.cpp)
target_link_libraries(mrdl PRIVATE mrdl_core)
mrdl_apply_build_policy(mrdl)
if(MRDL_ENABLE_LTO AND MRDL_IPO_OK)
    set_property(TARGET mrdl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(MRDL_BUILD_TESTS AND BUILD_TESTING)
    add_executable(mrdl_tests tests/test_main.cpp)
    target_link_libraries(mrdl_tests PRIVATE mrdl_core)
    mrdl_apply_build_policy(mrdl_tests)
    add_test(NAME mrdl_tests COMMAND mrdl_tests)
    set_tests_properties(mrdl_tests PROPERTIES TIMEOUT 180)
endif()

if(MRDL_BUILD_BENCHMARKS)
    add_executable(mrdl_bench benchmarks/bench_main.cpp)
    target_link_libraries(mrdl_bench PRIVATE mrdl_core)
    mrdl_apply_build_policy(mrdl_bench)
    if(MRDL_ENABLE_LTO AND MRDL_IPO_OK)
        set_property(TARGET mrdl_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

install(TARGETS mrdl mrdl_core
        EXPORT mrdlTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(MRDL_BUILD_BENCHMARKS)
    install(TARGETS mrdl_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/mrdl/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mrdl)
install(FILES config/vps.ini config/vps-system.ini config/docker.ini DESTINATION ${CMAKE_INSTALL_DATADIR}/mrdl/config)
install(FILES README.md CHANGELOG.md SECURITY.md DESTINATION ${CMAKE_INSTALL_DATADIR}/mrdl OPTIONAL)
install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/mrdl/docs
        FILES_MATCHING PATTERN "*.md")
install(EXPORT mrdlTargets
        FILE mrdlTargets.cmake
        NAMESPACE mrdl::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mrdl)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mrdlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    cmake/mrdlConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/mrdlConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mrdl
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mrdlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mrdlConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mrdl
)

set(CPACK_PACKAGE_NAME "mrdl")
set(CPACK_PACKAGE_VENDOR "Hackerprod")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Sparse relational language runtime")
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/model/;/.git/;~$;\\.swp$")
include(CPack)
