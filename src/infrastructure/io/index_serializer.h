#pragma once 

#include <algorithm>
#include <fstream>      
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>    
#include <string>       
#include <vector>

/**
 * @brief 索引序列化基类接口
 */
class BaseIndex {
public:
    virtual ~BaseIndex() = default;
    virtual void save(const std::string &file_path) = 0;
    virtual void load(const std::string &file_path) = 0;
};

/**
 * @brief 紧凑图索引序列化实现
 * @details 负责 CompactGraph 索引的二进制序列化和反序列化
 */
class IndexCompactGraph : public BaseIndex {
public:
    /**
     * @brief 保存紧凑图索引到文件
     * @param file_path 保存路径
     */
    void save(const std::string &file_path) override {
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
            out.write((char *)neighbors.nns.data(), nns_size * sizeof(CompressedPoint<float>));

            size_t rev_nns_size = neighbors.rev_nns.size();
            out.write((char *)&rev_nns_size, sizeof(rev_nns_size));
            out.write((char *)neighbors.rev_nns.data(), rev_nns_size * sizeof(CompressedPoint<float>));
        }

        out.close();
    }

    /**
     * @brief 从文件加载紧凑图索引
     * @param file_path 文件路径
     */
    void load(const std::string &file_path) override {
        std::ifstream in(file_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open file for loading index.");
        }
        visited_list_pool_ = new base_hnsw::VisitedListPool(1, data_wrapper->data_size);
        // Load directed_indexed_arr
        size_t arr_size;
        in.read((char *)&arr_size, sizeof(arr_size));
        directed_indexed_arr.resize(arr_size);
        for (auto &neighbors : directed_indexed_arr) {
            size_t nns_size;
            in.read((char *)&nns_size, sizeof(nns_size));
            neighbors.nns.resize(nns_size);
            in.read((char *)neighbors.nns.data(), nns_size * sizeof(CompressedPoint<float>));

            size_t rev_nns_size;
            in.read((char *)&rev_nns_size, sizeof(rev_nns_size));
            neighbors.rev_nns.resize(rev_nns_size);
            in.read((char *)neighbors.rev_nns.data(), rev_nns_size * sizeof(CompressedPoint<float>));
        }

        in.close();

        // print out the basic neighbor amount of the loaded index
        countNeighbrs();
    }

private:
    // 成员变量需要根据实际的数据结构定义
    std::vector<DirectedPointNeighbors<float>> directed_indexed_arr;
    DataWrapper* data_wrapper;
    base_hnsw::VisitedListPool* visited_list_pool_;
    
    void countNeighbrs() {
        // 实现邻居计数逻辑
        std::cout << "Counting neighbors..." << std::endl;
    }
};

/**
 * @brief 2D段图索引序列化实现
 * @details 负责 SegmentGraph2D 索引的二进制序列化和反序列化
 */
class IndexSegmentGraph2D : public BaseIndex {
public:
    /**
     * @brief 保存2D段图索引到文件
     * @param file_path 保存路径
     */
    void save(const std::string &file_path) override {
        std::ofstream output(file_path, std::ios::binary);
        unsigned counter = 0;
        base_hnsw::writeBinaryPOD(output, index_k);
        for (auto &segment : directed_indexed_arr) {
            base_hnsw::writeBinaryPOD(output, (int)segment.forward_nns.size());
            base_hnsw::writeBinaryPOD(output, (int)segment.reverse_nns.size());

            counter += 2;
            for (auto &nn : segment.forward_nns) {
                base_hnsw::writeBinaryPOD(output, nn.batch);
                base_hnsw::writeBinaryPOD(output, nn.start);
                base_hnsw::writeBinaryPOD(output, nn.end);
                base_hnsw::writeBinaryPOD(output, (int)nn.nns_id.size());
                for (auto &nn_id : nn.nns_id) {
                    base_hnsw::writeBinaryPOD(output, nn_id);
                    counter += 1;
                }
                counter += 4;
            }
            for (auto &nn_id : segment.reverse_nns) {
                base_hnsw::writeBinaryPOD(output, nn_id);
                counter += 1;
            }
        }
        std::cout << "Total write " << counter << " (int) to file " << file_path << std::endl;
    }

    /**
     * @brief 从文件加载2D段图索引
     * @param file_path 文件路径
     */
    void load(const std::string &file_path) override {
        std::ifstream input(file_path, std::ios::binary);
        if (!input.is_open()) throw std::runtime_error("Cannot open file");
        directed_indexed_arr.clear();
        directed_indexed_arr.resize(data_wrapper->data_size);
        base_hnsw::readBinaryPOD(input, index_k);
        std::cout << "Index K is " << index_k << std::endl;
        visited_list_pool_ = new base_hnsw::VisitedListPool(1, data_wrapper->data_size);
        int forward_num;
        int reverse_num;
        int batch_num;
        int start_pos;
        int end_pos;
        int nn_size;
        int one_nn;
        for (size_t i = 0; i < data_wrapper->data_size; i++) {
            base_hnsw::readBinaryPOD(input, forward_num);
            base_hnsw::readBinaryPOD(input, reverse_num);

            std::vector<OneSegmentNeighbors> neighbors;
            for (size_t j = 0; j < forward_num; j++) {
                base_hnsw::readBinaryPOD(input, batch_num);
                base_hnsw::readBinaryPOD(input, start_pos);
                base_hnsw::readBinaryPOD(input, end_pos);
                base_hnsw::readBinaryPOD(input, nn_size);
                std::vector<int> forward_nns;
                for (size_t k = 0; k < nn_size; k++) {
                    base_hnsw::readBinaryPOD(input, one_nn);
                    forward_nns.emplace_back(one_nn);
                }
                OneSegmentNeighbors one_forward_segment(batch_num, start_pos, end_pos);
                one_forward_segment.nns_id.swap(forward_nns);
                neighbors.emplace_back(one_forward_segment);
            }

            std::vector<int> reverse_nns;
            for (size_t j = 0; j < reverse_num; j++) {
                base_hnsw::readBinaryPOD(input, one_nn);
                reverse_nns.emplace_back(one_nn);
            }
            directed_indexed_arr[i].forward_nns.swap(neighbors);
            directed_indexed_arr[i].reverse_nns.swap(reverse_nns);
        }
        printOnebatch();
        countNeighbrs();
        std::cout << "Total # of neighbors: " << index_info->nodes_amount << std::endl;
    }

private:
    // 成员变量需要根据实际的数据结构定义
    std::vector<SegmentNeighbors> directed_indexed_arr;
    DataWrapper* data_wrapper;
    base_hnsw::VisitedListPool* visited_list_pool_;
    int index_k;
    IndexInfo* index_info;
    
    void printOnebatch() {
        // 实现批次打印逻辑
        std::cout << "Printing one batch info..." << std::endl;
    }
    
    void countNeighbrs() {
        // 实现邻居计数逻辑
        std::cout << "Counting neighbors..." << std::endl;
    }
};