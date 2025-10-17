/**
 * @file data_vecs.h
 * @brief 控制原始向量和查询数据结构
 * 
 * 此类用于封装数据集的基本信息以及存储节点和查询的相关数据，
 * 包括维度、权重属性、真实键值、节点及其键值、查询及其范围和键值，
 * 并提供读取数据、生成过滤查询及基准测试的方法。
 * 
 * @date 2023-06-19
 * @copyright Copyright (c) 2023
 */

 #pragma once

 #include <string>
 #include <vector>
 #include <random>
 #include <iostream>
 #include <fstream>
 #include <sys/time.h>
 #include <algorithm>
 
 #include "infrastructure/utils/utils.h"
 
 // 使用标准库中的pair、string和vector类型
 using std::pair;
 using std::string;
 using std::vector;
 using std::cout;
 using std::endl;

 // 前向声明数据读取函数
 inline void ReadDataWrapper(const string& dataset, const string& dataset_path, 
                      vector<vector<float>>& nodes, int data_size,
                      const string& query_path, vector<vector<float>>& querys,
                      int query_num, vector<int>& nodes_keys) {
     // TODO: 实现数据读取逻辑
     // 这是一个占位符实现
     nodes.clear();
     querys.clear();
     nodes_keys.clear();
 }

 inline void ReadGroundtruthQuery(vector<vector<int>>& groundtruth,
                           vector<pair<int, int>>& query_ranges,
                           vector<int>& query_ids,
                           const string& gt_path) {
     // TODO: 实现groundtruth读取逻辑
     // 这是一个占位符实现
     groundtruth.clear();
     query_ranges.clear();
     query_ids.clear();
 }
 
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
     
     // 是否为均匀权重
     bool is_even_weight;
     
     // 是否为真实键值
     bool real_keys;
     
     // 节点数据（待优化：从向量改为数组）???
     vector<vector<float>> nodes;
     
     // 节点键值
     vector<int> nodes_keys;
     
     // 原始查询数据
     vector<vector<float>> querys;
     
     // 查询键值
     vector<int> querys_keys;
     
     // 查询范围
     vector<pair<int, int>> query_ranges;
     
     // 地面实况数据
     vector<vector<int>> groundtruth;
     
     // 查询标识符
     vector<int> query_ids;
 
     /**
      * 读取数据文件
      */
     void readData(string &dataset_path, string &query_path);
     
     /**
      * 生成范围过滤查询和地面实况数据
      */
     void generateRangeFilteringQueriesAndGroundtruth(bool is_save = false, const string path = "");
     
     /**
      * 生成半边界查询和地面实况数据
      */
     void generateHalfBoundedQueriesAndGroundtruth(bool is_save = false, const string path = "");
     
     /**
      * 加载地面实况数据
      */
     void LoadGroundtruth(const string &gt_path);
 
     /**
      * 生成范围过滤查询和地面实况数据（可扩展性版本）
      */
     void generateRangeFilteringQueriesAndGroundtruthScalability(bool is_save = false, const string path = "");
     
     /**
      * 生成半边界查询和地面实况数据（可扩展性版本）
      */
     void generateHalfBoundedQueriesAndGroundtruthScalability(bool is_save = false, const string path = "");
     
     /**
      * 生成半边界查询和地面实况数据（基准测试版本）
      */
     void generateHalfBoundedQueriesAndGroundtruthBenchmark(bool is_save_to_file, const string save_path = "");
     
     /**
      * 生成范围过滤查询和地面实况数据（基准测试版本）
      */
     void generateRangeFilteringQueriesAndGroundtruthBenchmark(bool is_save_to_file, const string save_path = "");
 };
 
 /* ====================== 内联的辅助函数 ====================== */
 inline void SynthesizeQuerys(const vector<vector<float>> &nodes,
                              vector<vector<float>> &querys,
                              const int query_num) {
     int dim = nodes.front().size();
     std::default_random_engine e;
     std::uniform_int_distribution<int> u(0, nodes.size() - 1);
     querys.clear();
     querys.resize(query_num);
 
     for (unsigned n = 0; n < (unsigned)query_num; n++) {
         for (unsigned i = 0; i < (unsigned)dim; i++) {
             int select_idx = u(e);
             querys[n].emplace_back(nodes[select_idx][i]);
         }
     }
 }
 
 inline void SaveToCSVRow(const string &path, const int idx, const int l_bound, const int r_bound, const int pos_range, const int real_search_key_range, const int K_neighbor, const double &search_time, const vector<int> &gt, const vector<float> &pts) {
     std::ofstream file;
     file.open(path, std::ios_base::app);
     if (file) {
         file << idx << "," << l_bound << "," << r_bound << "," << pos_range << ","
              << real_search_key_range << "," << K_neighbor << "," << search_time
              << ",";
         for (auto ele : gt) {
             file << ele << " ";
         }
         // file << ",";
         // for (auto ele : pts) {
         //   file << ele << " ";
         // }
         file << "\n";
     }
     file.close();
 }
 
 /* ====================== 成员函数内联实现 ====================== */
 inline void DataWrapper::readData(string &dataset_path, string &query_path) {
     ReadDataWrapper(dataset, dataset_path, this->nodes, data_size, query_path,
                     this->querys, query_num, this->nodes_keys);
     cout << "Load vecs from: " << dataset_path << endl;
     cout << "# of vecs: " << nodes.size() << endl;
 
     // already sort data in sorted_data
     // if (dataset != "wiki-image" && dataset != "yt8m") {
     //   nodes_keys.resize(nodes.size());
     //   iota(nodes_keys.begin(), nodes_keys.end(), 0);
     // }
 
     if (querys.empty()) {
         cout << "Synthesizing querys..." << endl;
         SynthesizeQuerys(nodes, querys, query_num);
     }
 
     this->real_keys = false;
     vector<size_t> index_permutation; // already sort data ahead
 
     // if (dataset == "wiki-image" || dataset == "yt8m") {
     //   cout << "first search_key before sorting: " << nodes_keys.front() <<
     //   endl; cout << "sorting dataset: " << dataset << endl; index_permutation =
     //   sort_permutation(nodes_keys); apply_permutation_in_place(nodes,
     //   index_permutation); apply_permutation_in_place(nodes_keys,
     //   index_permutation); cout << "Dimension: " << nodes.front().size() <<
     //   endl; cout << "first search_key: " << nodes_keys.front() << endl;
     //   this->real_keys = true;
     // }
     this->data_dim = this->nodes.front().size();
 }
 
 inline void DataWrapper::generateRangeFilteringQueriesAndGroundtruth(
     bool is_save_to_file,
     const string save_path) {
     std::default_random_engine e;
     timeval t1, t2;
     double accu_time = 0.0;
 
     vector<int> query_range_list;
     float scale = 0.05;
     if (this->dataset == "local") {
         scale = 0.1;
     }
     int init_range = this->data_size * scale;
     while (init_range <= this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
 #ifdef LOG_DEBUG_MODE
     if (this->dataset == "local") {
         query_range_list.erase(
             query_range_list.begin(),
             query_range_list.begin() + 4 * query_range_list.size() / 9);
     }
 #endif
     if (0.01 * this->data_size > 100)
         query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);
 
     if (0.001 * this->data_size > 100)
         query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);
 
     if (0.0001 * this->data_size > 100)
         query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);
     if (this->data_size == 1000000) {
         query_range_list = {1000, 2000, 3000, 4000, 5000, 6000,
                             7000, 8000, 9000, 10000, 20000, 30000,
                             40000, 50000, 60000, 70000, 80000, 90000,
                             100000, 200000, 300000, 400000, 500000, 600000,
                             700000, 800000, 900000, 1000000};
     }
 
     cout << "Generating Groundtruth...\nRanges: ";
     print_set(query_range_list);
     // generate groundtruth
 
     for (auto range : query_range_list) {
         std::uniform_int_distribution<int> u_lbound(0,
                                                     this->data_size - range - 80);
         for (int i = 0; i < (int)this->querys.size(); i++) {
             int l_bound = u_lbound(e);
             int r_bound = l_bound + range - 1;
             if (range == this->data_size) {
                 l_bound = 0;
                 r_bound = this->data_size - 1;
             }
             int search_key_range = r_bound - l_bound + 1;
             if (this->real_keys)
                 search_key_range =
                     this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
             query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
             double greedy_time;
             gettimeofday(&t1, NULL);
             auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                     this->query_k);
             gettimeofday(&t2, NULL);
             CountTime(t1, t2, greedy_time);
             groundtruth.emplace_back(gt);
             query_ids.emplace_back(i);
             accu_time += greedy_time;
             if (is_save_to_file) {
                 SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                              this->query_k, greedy_time, gt, this->querys.at(i));
             }
         }
     }
 
     cout << " Done!" << endl
          << "Groundtruth Time: " << accu_time << endl;
     if (is_save_to_file) {
         cout << "Save GroundTruth to path: " << save_path << endl;
     }
 }
 
 inline void DataWrapper::generateHalfBoundedQueriesAndGroundtruth(
     bool is_save_to_file,
     const string save_path) {
     timeval t1, t2;
 
     vector<int> query_range_list;
     int init_range = this->data_size * 0.05;
     while (init_range <= this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * 0.05;
     }
     if (0.01 * this->data_size > 100)
         query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);
 
     if (0.001 * this->data_size > 100)
         query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);
 
     if (0.0001 * this->data_size > 100)
         query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);
 
     if (this->data_size == 1000000) {
         query_range_list = {1000, 2000, 3000, 4000, 5000, 6000,
                             7000, 8000, 9000, 10000, 20000, 30000,
                             40000, 50000, 60000, 70000, 80000, 90000,
                             100000, 200000, 300000, 400000, 500000, 600000,
                             700000, 800000, 900000, 1000000};
     }
 
     if (this->data_size == 100000) {
         query_range_list = {1000, 2000, 3000, 4000, 5000, 6000, 7000,
                             8000, 9000, 10000, 20000, 30000, 40000, 50000,
                             60000, 70000, 80000, 90000, 100000};
     }
 
     cout << "Generating Half Bounded Groundtruth...";
     cout << endl
          << "Ranges: " << endl;
     print_set(query_range_list);
     // generate groundtruth
     for (auto range : query_range_list) {
         for (int i = 0; i < (int)this->querys.size(); i++) {
             int l_bound = 0;
             int r_bound = range - 1;
 
             int search_key_range = r_bound - l_bound + 1;
             if (this->real_keys)
                 search_key_range =
                     this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
             query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
             double greedy_time;
             gettimeofday(&t1, NULL);
             auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                     this->query_k);
             gettimeofday(&t2, NULL);
             CountTime(t1, t2, greedy_time);
             groundtruth.emplace_back(gt);
             query_ids.emplace_back(i);
             if (is_save_to_file) {
                 SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                              this->query_k, greedy_time, gt, this->querys.at(i));
             }
         }
     }
     cout << "  Done!" << endl;
     if (is_save_to_file) {
         cout << "Save GroundTruth to path: " << save_path << endl;
     }
 }
 
 inline void DataWrapper::LoadGroundtruth(const string &gt_path) {
     this->groundtruth.clear();
     this->query_ranges.clear();
     this->query_ids.clear();
     cout << "Loading Groundtruth from" << gt_path << "...";
     ReadGroundtruthQuery(this->groundtruth, this->query_ranges, this->query_ids,
                          gt_path);
     cout << "    Done!" << endl;
 }
 
 inline void DataWrapper::generateRangeFilteringQueriesAndGroundtruthScalability(
     bool is_save_to_file,
     const string save_path) {
     std::default_random_engine e;
     timeval t1, t2;
     double accu_time = 0.0;
 
     vector<int> query_range_list;
     float scale = 0.001;
     int init_range = this->data_size * scale;
     while (init_range < 0.01 * this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
     scale = 0.01;
     init_range = this->data_size * scale;
     while (init_range < 0.1 * this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
 
     scale = 0.1;
     init_range = this->data_size * scale;
     while (init_range < 1 * this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
 
     query_range_list.emplace_back(this->data_size);
 
     cout << "Generating Groundtruth...\nRanges: ";
     print_set(query_range_list);
     cout << "sample size:" << this->nodes.size() << endl;
     // generate groundtruth
 
     for (auto range : query_range_list) {
         std::uniform_int_distribution<int> u_lbound(0,
                                                     this->data_size - range - 80);
         for (int i = 0; i < (int)this->querys.size(); i++) {
             int l_bound = u_lbound(e);
             int r_bound = l_bound + range - 1;
             if (range == this->data_size) {
                 l_bound = 0;
                 r_bound = this->data_size - 1;
             }
             int search_key_range = r_bound - l_bound + 1;
             // if (this->real_keys)
             //   search_key_range =
             //       this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
             query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
             double greedy_time;
             gettimeofday(&t1, NULL);
             auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                     this->query_k);
             gettimeofday(&t2, NULL);
             CountTime(t1, t2, greedy_time);
             groundtruth.emplace_back(gt);
             query_ids.emplace_back(i);
             accu_time += greedy_time;
             if (is_save_to_file) {
                 SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                              this->query_k, greedy_time, gt, this->querys.at(i));
             }
         }
     }
 
     cout << " Done!" << endl
          << "Groundtruth Time: " << accu_time << endl;
     if (is_save_to_file) {
         cout << "Save GroundTruth to path: " << save_path << endl;
     }
 }
 
 inline void DataWrapper::generateHalfBoundedQueriesAndGroundtruthScalability(
     bool is_save_to_file,
     const string save_path) {
     timeval t1, t2;
 
     vector<int> query_range_list;
     float scale = 0.001;
     int init_range = this->data_size * scale;
     while (init_range < 0.01 * this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
     scale = 0.01;
     init_range = this->data_size * scale;
     while (init_range < 0.1 * this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
 
     scale = 0.1;
     init_range = this->data_size * scale;
     while (init_range < 1 * this->data_size) {
         query_range_list.emplace_back(init_range);
         init_range += this->data_size * scale;
     }
 
     query_range_list.emplace_back(this->data_size);
 
     cout << "Generating Half Bounded Groundtruth...";
     cout << endl
          << "Ranges: " << endl;
     print_set(query_range_list);
     // generate groundtruth
     for (auto range : query_range_list) {
         for (int i = 0; i < (int)this->querys.size(); i++) {
             int l_bound = 0;
             int r_bound = range - 1;
 
             int search_key_range = r_bound - l_bound + 1;
             if (this->real_keys)
                 search_key_range =
                     this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
             query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
             double greedy_time;
             gettimeofday(&t1, NULL);
             auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                     this->query_k);
             gettimeofday(&t2, NULL);
             CountTime(t1, t2, greedy_time);
             groundtruth.emplace_back(gt);
             query_ids.emplace_back(i);
             if (is_save_to_file) {
                 SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                              this->query_k, greedy_time, gt, this->querys.at(i));
             }
         }
     }
     cout << "  Done!" << endl;
     if (is_save_to_file) {
         cout << "Save GroundTruth to path: " << save_path << endl;
     }
 }
 
 inline void DataWrapper::generateHalfBoundedQueriesAndGroundtruthBenchmark(
     bool is_save_to_file,
     const string save_path) {
     timeval t1, t2;
 
     vector<int> query_range_list;
     query_range_list.emplace_back(this->data_size * 0.001);
     query_range_list.emplace_back(this->data_size * 0.005);
     query_range_list.emplace_back(this->data_size * 0.01);
     query_range_list.emplace_back(this->data_size * 0.05);
     query_range_list.emplace_back(this->data_size * 0.1);
     query_range_list.emplace_back(this->data_size * 0.5);
     query_range_list.emplace_back(this->data_size);
 
     cout << "Generating Half Bounded Groundtruth...";
     cout << endl
          << "Ranges: " << endl;
     print_set(query_range_list);
     // generate groundtruth
     for (auto range : query_range_list) {
         for (int i = 0; i < (int)this->querys.size(); i++) {
             int l_bound = 0;
             int r_bound = range - 1;
 
             int search_key_range = r_bound - l_bound + 1;
             if (this->real_keys)
                 search_key_range =
                     this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
             query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
             double greedy_time;
             gettimeofday(&t1, NULL);
             auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                     this->query_k);
             gettimeofday(&t2, NULL);
             CountTime(t1, t2, greedy_time);
             groundtruth.emplace_back(gt);
             query_ids.emplace_back(i);
             if (is_save_to_file) {
                 SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                              this->query_k, greedy_time, gt, this->querys.at(i));
             }
         }
     }
     cout << "  Done!" << endl;
     if (is_save_to_file) {
         cout << "Save GroundTruth to path: " << save_path << endl;
     }
 }
 
 inline void DataWrapper::generateRangeFilteringQueriesAndGroundtruthBenchmark(
     bool is_save_to_file,
     const string save_path) {
     timeval t1, t2;
 
     vector<int> query_range_list;
     // query_range_list.emplace_back(this->data_size * 0.01);
     // query_range_list.emplace_back(this->data_size * 0.02);
     // query_range_list.emplace_back(this->data_size * 0.04);
     // query_range_list.emplace_back(this->data_size * 0.08);
     // query_range_list.emplace_back(this->data_size * 0.16);
     // query_range_list.emplace_back(this->data_size * 0.32);
     // query_range_list.emplace_back(this->data_size * 0.64);
     query_range_list.emplace_back(1000);
     query_range_list.emplace_back(2000);
     query_range_list.emplace_back(4000);
     query_range_list.emplace_back(8000);
     query_range_list.emplace_back(16000);
     query_range_list.emplace_back(32000);
     query_range_list.emplace_back(64000);
     // query_range_list.emplace_back(this->data_size);
 
     cout << "Generating Range Filtering Groundtruth...";
     cout << endl
          << "Ranges: " << endl;
     print_set(query_range_list);
     vector<double> bf_latency_aveInEachRange(query_range_list.size(), 0);
     std::default_random_engine e;
 
     for (int range_id = 0; range_id < (int)query_range_list.size(); range_id++) {
         auto &range = query_range_list[range_id];
         std::uniform_int_distribution<int> u_lbound(0,
                                                     std::max(this->data_size - range - 1, 0));
         for (int i = 0; i < (int)this->querys.size(); i++) {
             int l_bound = u_lbound(e);
             int r_bound = std::min(this->data_size - 1, l_bound + range - 1);
             int search_key_range = r_bound - l_bound + 1;
             // if (this->real_keys)
             //   search_key_range =
             //       this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
             query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
             double greedy_time;
             gettimeofday(&t1, NULL);
             auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                     this->query_k);
             gettimeofday(&t2, NULL);
             CountTime(t1, t2, greedy_time);
             bf_latency_aveInEachRange[range_id] += greedy_time;
 
             groundtruth.emplace_back(gt);
             query_ids.emplace_back(i);
             if (is_save_to_file) {
                 SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                              this->query_k, greedy_time, gt, this->querys.at(i));
             }
         }
         bf_latency_aveInEachRange[range_id] /= this->querys.size();
     }
     cout << "Here is bruteforce average time cost when generating groundtruth for each range:" << endl;
     print_set(bf_latency_aveInEachRange);
     if (is_save_to_file) {
         cout << "Save GroundTruth to path: " << save_path << endl;
     }
 }