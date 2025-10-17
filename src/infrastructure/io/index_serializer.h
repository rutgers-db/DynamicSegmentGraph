#pragma once 

#include <algorithm>
#include <fstream>      
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>    
#include <string>       
#include <vector>

#include "core/algorithms/compact_types.h"
#include "interfaces/search_interface.h"
#include "infrastructure/io/data_loader.h"
#include "baselines/hnswlib.h"
#include "baselines/visited_list_pool.h"

// 前向声明 SeRF 命名空间的类型
namespace SeRF {
    struct OneSegmentNeighbors;
    struct DirectedSegNeighbors;
}

// 简单定义 SegmentNeighbors 类型别名（如果在其他地方没有定义）
using SegmentNeighbors = SeRF::DirectedSegNeighbors;

/**
 * @brief 紧凑图索引序列化实现
 * @details 负责 CompactGraph 索引的二进制序列化和反序列化
 */
class IndexCompactGraphSerializer {
public:
    /**
     * @brief 保存紧凑图索引到文件
     * @param file_path 保存路径
     */
    void save(const std::string &file_path) {
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for saving index.");
        }

        // Save directed_indexed_arr
        size_t arr_size = directed_indexed_arr.size();
        out.write((char *)&arr_size, sizeof(arr_size));
        for (auto &neighbors : directed_indexed_arr) {
            size_t nns_size = neighbors.nns.size();
            out.write((char *)&nns_size, sizeof(nns_size));
            out.write((char *)neighbors.nns.data(), nns_size * sizeof(Compact::CompressedPoint<float>));

            size_t rev_nns_size = neighbors.rev_nns.size();
            out.write((char *)&rev_nns_size, sizeof(rev_nns_size));
            out.write((char *)neighbors.rev_nns.data(), rev_nns_size * sizeof(Compact::CompressedPoint<float>));
        }

        out.close();
    }

    /**
     * @brief 从文件加载紧凑图索引
     * @param file_path 文件路径
     */
    void load(const std::string &file_path) {
        std::ifstream in(file_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open file for loading index.");
        }
        visited_list_pool_ = new hnswlib_incre::VisitedListPool(1, data_wrapper->data_size);
        // Load directed_indexed_arr
        size_t arr_size;
        in.read((char *)&arr_size, sizeof(arr_size));
        directed_indexed_arr.resize(arr_size);
        for (auto &neighbors : directed_indexed_arr) {
            size_t nns_size;
            in.read((char *)&nns_size, sizeof(nns_size));
            neighbors.nns.resize(nns_size);
            in.read((char *)neighbors.nns.data(), nns_size * sizeof(Compact::CompressedPoint<float>));

            size_t rev_nns_size;
            in.read((char *)&rev_nns_size, sizeof(rev_nns_size));
            neighbors.rev_nns.resize(rev_nns_size);
            in.read((char *)neighbors.rev_nns.data(), rev_nns_size * sizeof(Compact::CompressedPoint<float>));
        }

        in.close();

        // print out the basic neighbor amount of the loaded index
        countNeighbrs();
    }

private:
    // 成员变量需要根据实际的数据结构定义
    std::vector<Compact::DirectedPointNeighbors<float>> directed_indexed_arr;
    DataWrapper* data_wrapper;
    hnswlib_incre::VisitedListPool* visited_list_pool_;
    
    void countNeighbrs() {
        // 实现邻居计数逻辑
        std::cout << "Counting neighbors..." << std::endl;
    }
};

// Note: IndexSegmentGraph2DSerializer 被暂时注释，因为它依赖于需要在 segment_graph_2d.h 中定义的完整类型
// 如果需要使用该序列化器，请在包含此头文件之前先包含 core/algorithms/segment_graph_2d.h
/*
 * @brief 2D段图索引序列化实现
 * @details 负责 SegmentGraph2D 索引的二进制序列化和反序列化
 * 
 * 使用说明：此类已被注释，如需使用请：
 * 1. 在包含此头文件前先包含 segment_graph_2d.h
 * 2. 或将此类的实现移到 .cpp 文件中
 */