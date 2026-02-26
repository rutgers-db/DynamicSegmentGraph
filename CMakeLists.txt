cmake_minimum_required(VERSION 3.16)
project(DynamicSegmentGraph LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

# Allow opting into -O3/-march=native even for Debug builds (useful for
# profiling fast paths while keeping symbols).
option(DSG_DEBUG_OPTIMIZED "Enable -O3/-march=native for Debug configuration" ON)

function(dsg_apply_common_compile_options target)
    target_compile_options(${target} PRIVATE -Wall)

    if(OpenMP_CXX_FLAGS)
        target_compile_options(${target} PRIVATE ${OpenMP_CXX_FLAGS})
    endif()

    target_compile_options(${target} PRIVATE
    # These two options looks super important for performance, but I don't know exactly why.
        -msse4.2
        -ftree-vectorize
        $<$<CONFIG:Release>:-Ofast>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:MinSizeRel>:-march=native>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-march=native>
        $<$<CONFIG:RelWithDebInfo>:-g>
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:${DSG_DEBUG_OPTIMIZED}>>>:-O0>
        $<$<AND:$<CONFIG:Debug>,$<BOOL:${DSG_DEBUG_OPTIMIZED}>>:-O3>
        $<$<AND:$<CONFIG:Debug>,$<BOOL:${DSG_DEBUG_OPTIMIZED}>>:-march=native>
        $<$<CONFIG:Debug>:-g>
    )

    target_compile_definitions(${target} PRIVATE
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:DSG_DEBUG_BUILD>
    )
endfunction()

add_subdirectory(src/utils)

add_library(dsg_core STATIC src/dsg.cc)
target_include_directories(dsg_core PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/utils
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(dsg_core PRIVATE UTIL OpenMP::OpenMP_CXX)
dsg_apply_common_compile_options(dsg_core)

# -----------------------------------------------------------------------------
# Release vs Debug Configuration
# -----------------------------------------------------------------------------
# By default, dsg_core builds with high optimization (-O3) and native SIMD support (-march=native).
#
# To configure a Debug build (with symbols and no optimization) simply set:
#   cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
#
# To additionally force debug instrumentation while staying on a non-Debug
# configuration, enable the option below:
#   cmake -B build -DDSG_BUILD_INDEX_DEBUG=ON ...
#
# To enable Benchmark executables (e.g. test_hnsw_range), use:
#   cmake -B build -DBUILD_RANGE_BENCHMARKS=ON ...
#
# -----------------------------------------------------------------------------
option(DSG_BUILD_INDEX_DEBUG "Build a debug-instrumented build_index target" OFF)
if(DSG_BUILD_INDEX_DEBUG)
    target_compile_options(dsg_core PRIVATE -O0 -g)
    target_compile_definitions(dsg_core PRIVATE DSG_DEBUG_BUILD)
endif()
add_subdirectory(apps)

option(BUILD_RANGE_BENCHMARKS "Build range benchmark executables under tests/" OFF)
option(ENABLE_AVX512 "Enable AVX-512 instructions for the SIMD benchmark" OFF)


if(BUILD_RANGE_BENCHMARKS)
    add_executable(test_range_benchmark
        tests/test_range_benchmark.cpp
    )

    add_executable(test_range_benchmark_simd
        tests/test_range_benchmark_simd.cpp
    )
    target_compile_options(test_range_benchmark_simd PRIVATE -mavx2 -mfma)
    if(ENABLE_AVX512)
        target_compile_options(test_range_benchmark_simd PRIVATE -mavx512f)
    endif()

    add_executable(test_l2_benchmark
        tests/test_l2_benchmark.cpp
    )

endif()
