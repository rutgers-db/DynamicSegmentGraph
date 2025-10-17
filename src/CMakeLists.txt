# 收集实际存在的源码，覆盖 core / infrastructure / baselines
file(GLOB_RECURSE MYDSG_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/**/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/**/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/baselines/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/baselines/**/*.cc"
)

# 排除示例/演示源码，避免被链接进库
list(FILTER MYDSG_CORE_SOURCES EXCLUDE REGEX ".*/memory_pool_example\\.cpp$")

# 确保找到源码
if(NOT MYDSG_CORE_SOURCES)
    message(FATAL_ERROR "未找到任何源码，请检查: ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# 创建核心静态库
add_library(mydsg_core STATIC ${MYDSG_CORE_SOURCES})

# 头文件包含路径
target_include_directories(mydsg_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces
    ${CMAKE_CURRENT_SOURCE_DIR}/baselines
)

# 编译与链接选项
target_compile_options(mydsg_core PRIVATE -Wall)
target_link_libraries(mydsg_core PUBLIC OpenMP::OpenMP_CXX ${Boost_LIBRARIES})
