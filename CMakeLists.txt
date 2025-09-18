cmake_minimum_required(VERSION 3.10)
project(DynamicSegmentGraph)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 查找依赖包
find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)
find_package(OpenMP REQUIRED)

# 设置包含目录 - 优先使用原始项目的头文件
set(ORIGINAL_PROJECT_ROOT "/home/scavenger/DSG/DynamicSegmentGraph")
include_directories(${ORIGINAL_PROJECT_ROOT}/src)
include_directories(${ORIGINAL_PROJECT_ROOT}/include)
include_directories(${ORIGINAL_PROJECT_ROOT}/include/common)

# mydsg 项目的头文件（仅在需要时包含）
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/infrastructure)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/baselines)

# 添加子目录
add_subdirectory(src)
# 启用 benchmark 编译
add_subdirectory(benchmark)

# 创建简单的测试可执行文件
add_executable(test_build test_build.cpp)
target_link_libraries(test_build ${Boost_LIBRARIES})

# 创建测试可执行文件（如果需要的话）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../DynamicSegmentGraph/benchmark")
    message(STATUS "发现 benchmark 目录，可以添加更多测试...")
endif()
