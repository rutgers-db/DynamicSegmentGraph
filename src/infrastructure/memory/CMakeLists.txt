# 内存池模块CMakeLists.txt

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# 创建内存池库
add_library(memory_pool STATIC
    memory_pool.h
)

# 设置包含目录
target_include_directories(memory_pool PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 创建示例程序
add_executable(memory_pool_example
    memory_pool_example.cpp
)

# 链接库
target_link_libraries(memory_pool_example
    memory_pool
)

# 设置输出目录
set_target_properties(memory_pool PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(memory_pool_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则
install(TARGETS memory_pool
    ARCHIVE DESTINATION lib
)

install(FILES memory_pool.h
    DESTINATION include/memory
)
