# 创建 UTIL 库（兼容原始项目）
set(ORIGINAL_PROJECT_ROOT "/home/scavenger/DSG/DynamicSegmentGraph")
file(GLOB UTIL_SOURCES 
    "${ORIGINAL_PROJECT_ROOT}/src/common/*.cc"
    "${ORIGINAL_PROJECT_ROOT}/src/common/*.cpp"
)

# 调试：显示找到的源文件
message(STATUS "UTIL_SOURCES: ${UTIL_SOURCES}")

# 确保找到了源文件
if(NOT UTIL_SOURCES)
    message(FATAL_ERROR "未找到 UTIL 源文件，请检查路径: ${ORIGINAL_PROJECT_ROOT}/src/common/")
endif()

add_library(UTIL STATIC ${UTIL_SOURCES})
target_include_directories(UTIL PUBLIC
    ${ORIGINAL_PROJECT_ROOT}/include/common
    ${ORIGINAL_PROJECT_ROOT}/include
    ${ORIGINAL_PROJECT_ROOT}/src
)
target_compile_options(UTIL PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(UTIL ${OpenMP_CXX_FLAGS})
target_link_libraries(UTIL OpenMP::OpenMP_CXX)

# 创建 mydsg_core 库（暂时为空）
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "// 空的源文件用于创建库\n")
add_library(mydsg_core STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp
)

# 设置头文件包含路径
target_include_directories(mydsg_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces
)

# 链接依赖
target_link_libraries(mydsg_core ${Boost_LIBRARIES})
