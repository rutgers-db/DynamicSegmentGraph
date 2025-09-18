#!/bin/bash

# DynamicSegmentGraph 内存池测试脚本

echo "=========================================="
echo "DynamicSegmentGraph 内存池测试"
echo "=========================================="

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建构建目录
echo -e "${YELLOW}创建构建目录...${NC}"
mkdir -p build
cd build

# 运行CMake配置
echo -e "${YELLOW}配置CMake...${NC}"
if cmake ..; then
    echo -e "${GREEN}CMake配置成功${NC}"
else
    echo -e "${RED}CMake配置失败${NC}"
    exit 1
fi

# 编译项目
echo -e "${YELLOW}编译项目...${NC}"
if make -j$(nproc); then
    echo -e "${GREEN}编译成功${NC}"
else
    echo -e "${RED}编译失败${NC}"
    exit 1
fi

# 运行测试
echo -e "${YELLOW}运行性能测试...${NC}"
echo "=========================================="
if ./bin/memory_pool_example; then
    echo -e "${GREEN}测试完成${NC}"
else
    echo -e "${RED}测试失败${NC}"
    exit 1
fi

echo "=========================================="
echo -e "${GREEN}所有测试通过！${NC}"
echo "=========================================="

# 显示构建产物
echo -e "${YELLOW}构建产物:${NC}"
ls -la lib/
ls -la bin/

echo -e "${YELLOW}内存池头文件:${NC}"
ls -la ../memory_pool.h
