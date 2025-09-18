# mydsg 项目 Benchmark 运行指南

## 问题解决方案

由于 mydsg 重构项目中的 benchmark 存在兼容性问题，推荐使用原始项目中已经编译好的 benchmark 程序。

## 🎯 **正确的运行方式**

### 1. 使用原始项目的 benchmark

原始项目中已有完整编译的 benchmark 程序：

```bash
cd /home/scavenger/DSG/DynamicSegmentGraph/build/benchmark
ls -la
```

可用的程序：
- ✅ `build_index` - 构建索引
- ✅ `query_index` - 查询测试  
- ✅ `generate_groundtruth` - 生成真实值
- ✅ `compact_arbitrary` - 紧凑图任意范围测试
- ✅ `knn_first` - KNN 测试

### 2. 正确的运行命令

```bash
# 构建索引示例
cd /home/scavenger/DSG/DynamicSegmentGraph/build/benchmark
./build_index -N 1000 -k 16 -ef_max 500 -dataset deep10m -dataset_path ../../deep10M.fvecs -index_path ../../index -method compact

# 查询索引示例  
./query_index -N 1000 -k 16 -ef_max 500 -dataset deep10m -dataset_path ../../deep10M.fvecs -index_path ../../index -method compact -query_num 100 -query_k 10
```

### 3. 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `-N` | 数据集大小 | `1000` |
| `-k` | 索引参数 K | `16` |
| `-ef_max` | 最大 ef 值 | `500` |
| `-dataset` | 数据集名称 | `deep10m` |
| `-dataset_path` | 数据文件路径 | `../../deep10M.fvecs` |
| `-index_path` | 索引保存路径 | `../../index` |
| `-method` | 算法方法 | `compact` 或 `Seg2D` |

### 4. 数据文件检查

确保数据文件存在：
```bash
ls -la /home/scavenger/DSG/DynamicSegmentGraph/deep10M.fvecs
ls -la /home/scavenger/DSG/DynamicSegmentGraph/deep1B_queries.fvecs
```

## 🔧 **mydsg 项目状态**

### ✅ 当前可用功能
```bash
cd /home/scavenger/DSG/mydsg/build
./test_build  # 简单的构建测试程序
```

### 🚧 **开发中功能**
- 模块化架构已建立
- 基础构建系统正常
- 核心算法迁移进行中

## 📋 **推荐工作流程**

1. **性能测试**: 使用原始项目的 benchmark
   ```bash
   cd /home/scavenger/DSG/DynamicSegmentGraph/build/benchmark
   ./build_index [参数...]
   ```

2. **开发验证**: 使用 mydsg 的测试程序
   ```bash
   cd /home/scavenger/DSG/mydsg/build  
   ./test_build
   ```

3. **算法开发**: 在 mydsg 中进行模块化重构
   ```bash
   cd /home/scavenger/DSG/mydsg/src/core/algorithms
   # 编辑和测试算法代码
   ```

这样可以确保您既能进行性能测试，又能推进重构工作，两不耽误！





