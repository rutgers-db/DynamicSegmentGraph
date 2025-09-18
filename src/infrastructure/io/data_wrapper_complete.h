/**
 * @file data_wrapper_complete.h
 * @brief 完整的数据包装器定义
 * 
 * 从原始项目复制的完整 DataWrapper 类定义
 */
#pragma once

#include <string>
#include <vector>

// 使用标准库中的pair、string和vector类型
using std::pair;
using std::string;
using std::vector;

class DataWrapper {
public:
    /**
     * 构造函数初始化数据集名称、数据大小、查询数量和查询k值
     */
    DataWrapper(int num, int k_, string dataset_name, int data_size_)
        : dataset(dataset_name),   // 数据集名称
          data_size(data_size_),  // 数据大小
          query_num(num),         // 查询数量
          query_k(k_) {}          // 查询k值

    // 数据集名称（常量）
    const string dataset;
    
    // 版本号
    string version;
    
    // 数据大小（常量）
    const int data_size;
    
    // 查询数量（常量）
    const int query_num;
    
    // 查询k值（常量）
    const int query_k;
    
    // 数据维度
    size_t data_dim;
    
    // 节点数据
    vector<vector<float>> nodes;
    
    // 节点键值
    vector<int> node_keys;
    
    // 查询数据
    vector<vector<float>> querys;
    
    // 查询范围
    vector<pair<int, int>> query_ranges;
    
    // 查询键值
    vector<int> query_keys;
    
    // 查询ID
    vector<int> query_ids;
    
    // 真实邻居
    vector<vector<int>> groundtruth;

    // 方法声明（将在实现文件中定义）
    void readData(const string& dataset_path, const string& query_path);
    void generateRangeFilteringQueriesAndGroundtruth(bool save_to_file, string save_path);
    void generateHalfBoundedQueriesAndGroundtruth(bool save_to_file, string save_path);
    void generateRangeFilteringQueriesAndGroundtruthScalability(bool save_to_file, string save_path);
    void generateHalfBoundedQueriesAndGroundtruthScalability(bool save_to_file, string save_path);
    void generateHalfBoundedQueriesAndGroundtruthBenchmark(bool save_to_file, string save_path);
    void generateRangeFilteringQueriesAndGroundtruthBenchmark(bool save_to_file, string save_path);
};










