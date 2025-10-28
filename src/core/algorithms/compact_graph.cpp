#include <algorithm>
#include <boost/functional/hash.hpp>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>
#include <random>
#include <unordered_map>
#include <fstream>
#include <utility>   // for std::pair

#include "baselines/hnswalg.h"
#include "baselines/hnswlib.h"
#include "infrastructure/io/data_loader.h"
#include "infrastructure/io/index_serializer.h"
#include "infrastructure/utils/utils.h"
#include "infrastructure/utils/distance.h"
#include "core/algorithms/compact_types.h"
#include "core/algorithms/compact_graph.h"  // Include the header file

using namespace base_hnsw;
using std::pair;

namespace Compact {

// CompactHNSW class is now defined in compact_graph.h
// IndexCompactGraph class is declared in compact_graph.h

// Member function implementations for IndexCompactGraph

IndexCompactGraph::IndexCompactGraph(base_hnsw::SpaceInterface<float> *s,
                                     const DataWrapper *data) :
    BaseIndex(data) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    index_info = new IndexInfo();
    index_info->index_version_type = "IndexCompactGraph";
}

void IndexCompactGraph::printOnebatch() {
    cout << "Print one batch" << endl;
    for (auto cp :
         directed_indexed_arr[data_wrapper->data_size / 2].nns) {
        cout << "[" << cp.external_id << "," << cp.ll << ","
             << cp.rr << "], ";
    }
    cout << endl;
}

void IndexCompactGraph::countNeighbrs() {
    size_t max_nns_len = 0;
    // 如果有向图索引不为空，则开始处理
    if (!directed_indexed_arr.empty()) {
        // 遍历所有节点的前向邻居列表
        for (unsigned j = 0; j < directed_indexed_arr.size(); j++) {
            index_info->nodes_amount += directed_indexed_arr[j].countNeighbors();
            max_nns_len = std::max(max_nns_len, directed_indexed_arr[j].nns.size());
        }
    }

    // 计算平均前向邻居数
    index_info->avg_forward_nns = index_info->nodes_amount / static_cast<float>(data_wrapper->data_size);

    // 打印日志（如果启用）
    if (isLog) {
        cout << "Max. nns length of one point" << max_nns_len << endl;
        cout << "Sum of forward nn #: " << index_info->nodes_amount << endl;
        cout << "Avg. forward nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
        cout << "Avg. delta nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
    }
}

void IndexCompactGraph::buildIndex(const IndexParams *index_params) {
    cout << "Building Index using " << index_info->index_version_type << endl;
    timeval tt1, tt2;
    visited_list_pool_ =
        new hnswlib_incre::VisitedListPool(1, data_wrapper->data_size);

    index_params_ = index_params;
    // build HNSW
    base_hnsw::L2Space space(data_wrapper->data_dim);
    hnsw = new CompactHNSW<float>(
        *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction, index_params->random_seed);

    directed_indexed_arr.clear();
    directed_indexed_arr.resize(data_wrapper->data_size);
    hnsw->compact_graph = &directed_indexed_arr;
    gettimeofday(&tt1, NULL);

    // random add points
    // Step 1: Generate a sequence 0, 1, ..., data_size - 1
    std::vector<size_t> permutation(data_wrapper->data_size);
    std::iota(permutation.begin(), permutation.end(), 0);

    // Step 2: Shuffle the sequence
    std::random_device rd;    // obtain a random number from hardware
    unsigned int seed = 2024; // fix the seed for debug
    // std::mt19937 g(rd());
    std::mt19937 g(seed); // seed the generator
    std::shuffle(permutation.begin(), permutation.end(), g);

    // Step 3: Traverse the shuffled sequence

    cout << "First point" << permutation[0] << endl;
    for (size_t i : permutation) {
        hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
    }

    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);

    cout << "All the forward batch nn #: " << hnsw->forward_batch_nn_amount << endl;
    cout << "Theoratical backward batch nn #: " << hnsw->backward_batch_theoratical_nn_amount << endl;
    // count neighbors number
    countNeighbrs();
}

void IndexCompactGraph::initForScabilityExp(const IndexParams *index_params, base_hnsw::L2Space *space) {
    if(visited_list_pool_ == nullptr)
        visited_list_pool_ =
            new hnswlib_incre::VisitedListPool(1, data_wrapper->data_size);
    index_params_ = index_params;
    // build HNSW
    hnsw = new CompactHNSW<float>(
        *index_params, space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction, index_params->random_seed);

    // directed_indexed_arr.clear();
    directed_indexed_arr.resize(data_wrapper->data_size);
    hnsw->compact_graph = &directed_indexed_arr;
}

void IndexCompactGraph::rebuild_batchInHNSW(vector<unsigned> &nodes_ids) {
    hnsw->if_rebuild_HNSW = true;
    timeval tt1, tt2;
    gettimeofday(&tt1, NULL);
    for (auto i : nodes_ids) {
        hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
    }

    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);
    cout << "Reinsert for rebuilding a  " << nodes_ids.size() << " batch need" << index_info->index_time << endl;
    // count neighbors number
    countNeighbrs();
    hnsw->if_rebuild_HNSW = false;
}

void IndexCompactGraph::insert_batch(vector<unsigned> &nodes_ids) {
    timeval tt1, tt2;
    gettimeofday(&tt1, NULL);
    for (auto i : nodes_ids) {
        hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
    }

    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);
    cout << "Insert a  " << nodes_ids.size() << " batch need" << index_info->index_time << endl;
    // count neighbors number
    countNeighbrs();
}

/**
 * 在指定范围内进行过滤搜索的核心函数
 * 
 * 该函数实现了基于压缩图结构的范围过滤最近邻搜索算法。算法采用贪心搜索策略，
 * 从多个入口点开始，通过遍历压缩邻居关系来寻找查询范围内的最近邻点。
 * 
 * @param search_params 搜索参数，包含ef值和K值等配置
 * @param search_info 搜索统计信息，用于记录性能指标
 * @param query 查询向量
 * @param query_bound 查询范围边界，pair<左边界, 右边界>
 * @return 返回查询范围内的K个最近邻点ID列表
 * 
 * 算法特点：
 * 1. 多入口点初始化：在查询范围内均匀选择3个起始点
 * 2. 压缩邻居遍历：利用压缩点的范围信息快速过滤无效邻居
 * 3. 双向邻居搜索：同时遍历前向和反向邻居关系
 * 4. 动态候选集管理：维护最优候选集和搜索边界
 */
vector<int> IndexCompactGraph::rangeFilteringSearchInRange(
    const SearchParams *search_params,
    SearchInfo *search_info,
    const vector<float> &query,
    const std::pair<int, int> query_bound) {
    // 预分配邻居缓存空间并清空
    fetched_nns.reserve(100);
    fetched_nns.clear();

    // 时间测量变量初始化
    timeval tt1, tt2, tt3, tt4;

    // 初始化访问标记系统，用于避免重复访问
    hnswlib_incre::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    hnswlib_incre::vl_type *visited_array = vl->mass;
    hnswlib_incre::vl_type visited_array_tag = vl->curV;
    
    // 搜索算法核心数据结构
    float lower_bound = std::numeric_limits<float>::max();  // 当前最差候选的距离下界
    std::priority_queue<pair<float, int>> top_candidates;   // 最终结果候选集（最大堆）
    std::priority_queue<pair<float, int>> candidate_set;    // 待探索候选集（最小堆，负距离）

    // 初始化搜索统计信息
    search_info->total_comparison = 0;
    search_info->internal_search_time = 0;
    search_info->pos_point_traverse_counter = 0;
    search_info->pos_point_used_counter = 0;
    search_info->neg_point_traverse_counter = 0;
    search_info->neg_point_used_counter = 0;
    search_info->cal_dist_time = 0;
    search_info->fetch_nns_time = 0;
    search_info->path_counter = 0;
    num_search_comparison = 0;

    // 多入口点初始化策略：在查询范围内均匀选择3个起始点
    // 这种策略可以避免局部最优，提高搜索的全局性
    {
        int lbound = query_bound.first;
        int interval = (query_bound.second - lbound) / 3;
        for (size_t i = 0; i < 3; i++) {
            int point = lbound + interval * i;
            float dist = EuclideanDistance(data_wrapper->nodes[point], query); 
            candidate_set.push(make_pair(-dist, point));  // 使用负距离实现最小堆                     
            visited_array[point] = visited_array_tag;     // 标记为已访问                        
        }
    }
    gettimeofday(&tt3, NULL);


    size_t hop_counter = 0;
    float total_traversed_nn_amount = 0;
    float pos_point_traverse_counter = 0;
    float pos_point_used_counter = 0;
    float neg_point_traverse_counter = 0;
    float neg_point_used_counter = 0;

    // 主搜索循环：贪心搜索最近邻
    while (!candidate_set.empty()) {
        std::pair<float, int> current_node_pair = candidate_set.top(); 
        int current_node_id = current_node_pair.second;

        // 剪枝条件：如果当前候选点距离大于已找到的最差结果，停止搜索
        if (-current_node_pair.first > lower_bound) 
        {
            break;
        }

#ifdef LOG_DEBUG_MODE
        cout << "current node: " << current_node_pair.second << "  -- "
             << -current_node_pair.first << endl;
#endif

        hop_counter++;
        candidate_set.pop();

        // 范围检查：确保当前节点在查询范围内
        if (current_node_id < query_bound.first || current_node_id > query_bound.second) {
            cout << "no satisfied range point" << endl;
            continue;
        }
        gettimeofday(&tt1, NULL);

        // 获取当前节点的前向和反向邻居列表
        auto const &pos_edges = directed_indexed_arr[current_node_id].nns;
        auto const &neg_edges = directed_indexed_arr[current_node_id].rev_nns;
        
        // 第一阶段：邻居过滤 - 利用压缩点的范围信息快速过滤有效邻居
        fetched_nns.clear();
        
        // 处理前向邻居（正向边）
        for (size_t i = 0; i < pos_edges.size(); i++) {
            const unsigned &candidate_id = pos_edges[i].external_id;
            
            // 基本范围检查
            if ((int)candidate_id < query_bound.first)
                continue;
            if ((int)candidate_id > query_bound.second) 
                break;  // 由于邻居已排序，可以提前退出
                
            // 压缩范围检查：利用压缩点的ll,lr,rl,rr边界信息
            const auto &cp = pos_edges[i];
            if (!cp.if_in_compressed_range(query_bound.first, query_bound.second)) {
                continue;
            }
            fetched_nns.emplace_back(candidate_id);
        }

        // 处理反向邻居（反向边）
        for (auto i = 0; i < neg_edges.size(); i++) {
            const unsigned &candidate_id = neg_edges[i].external_id;
            
            // 基本范围检查
            if (candidate_id < query_bound.first)
                continue;
            if (candidate_id > query_bound.second) 
                continue;
                
            // 压缩范围检查
            auto &cp = neg_edges[i];
            if (!cp.if_in_compressed_range(query_bound.first, query_bound.second)) {
                continue;
            }
            fetched_nns.emplace_back(candidate_id);
        }
        gettimeofday(&tt2, NULL);                              
        AccumulateTime(tt1, tt2, search_info->fetch_nns_time); 

        // 第二阶段：距离计算和候选集更新
        for (auto &candidate_id : fetched_nns) {
            // 避免重复访问同一个节点
            if (!(visited_array[candidate_id] == visited_array_tag)) 
            {
                visited_array[candidate_id] = visited_array_tag; 

                gettimeofday(&tt1, NULL); 
                // 计算查询点到候选点的实际距离
                float dist = fstdistfunc_(query.data(),
                                          data_wrapper->nodes[candidate_id].data(),
                                          dist_func_param_);

                num_search_comparison++; // 更新距离计算次数统计
                
                // 候选集更新策略：
                // 1. 如果候选集未满，直接添加
                // 2. 如果距离优于当前最差候选，替换并更新边界
                if (top_candidates.size() < search_params->search_ef || lower_bound > dist) {
                    candidate_set.push(make_pair(-dist, candidate_id));  // 添加到待探索集合
                    top_candidates.push(make_pair(dist, candidate_id));  // 添加到结果候选集
                    
                    // 维护候选集大小不超过ef
                    if (top_candidates.size() > search_params->search_ef) {
                        top_candidates.pop(); 
                    }
                    
                    // 更新搜索下界
                    if (!top_candidates.empty()) {
                        lower_bound = top_candidates.top().first; 
                    }
                }
                gettimeofday(&tt2, NULL);                            
                AccumulateTime(tt1, tt2, search_info->cal_dist_time);
            }
        }
        // 统计遍历的邻居总数
        total_traversed_nn_amount += float(pos_edges.size()) + float(neg_edges.size());
    }

    // 构建结果列表
    vector<int> res;
    while (top_candidates.size() > search_params->query_K) {
        top_candidates.pop(); 
    }

    while (!top_candidates.empty()) {
        res.emplace_back(top_candidates.top().second); 
        top_candidates.pop();
    }
    search_info->total_comparison += num_search_comparison;  
    search_info->path_counter += hop_counter;
    search_info->pos_point_traverse_counter = pos_point_traverse_counter;
    search_info->pos_point_used_counter = pos_point_used_counter;
    search_info->neg_point_traverse_counter = neg_point_traverse_counter;
    search_info->neg_point_used_counter = neg_point_used_counter;
    search_info->total_traversed_nn_amount = total_traversed_nn_amount;

    // 释放资源和更新时间统计
    visited_list_pool_->releaseVisitedList(vl);
    gettimeofday(&tt4, NULL);
    CountTime(tt3, tt4, search_info->internal_search_time);
    return res; // 返回结果列表
}

vector<int> IndexCompactGraph::rangeFilteringSearchOutBound(
    const SearchParams *search_params,
    SearchInfo *search_info,
    const vector<float> &query,
    const std::pair<int, int> query_bound) {
    return vector<int>();
}

// Save function to store the IndexCompactGraph to a file
void IndexCompactGraph::save(const std::string &file_path) {
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

// Load function to load the IndexCompactGraph from a file
void IndexCompactGraph::load(const std::string &file_path) {
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

IndexCompactGraph::~IndexCompactGraph() {
    delete hnsw;
    delete index_info;
    directed_indexed_arr.clear();
    delete visited_list_pool_;
}


SearchResult Compact::IndexCompactGraph::searchKnn(
    const SearchParams *search_params,
    SearchInfo *search_info,
    const vector<float> &query) {
    timeval t1, t2;
    gettimeofday(&t1, NULL);

    const int K = static_cast<int>(search_params ? search_params->query_K : 10);
    std::priority_queue<std::pair<float,int>> topk; // 最大堆，堆顶是当前最差（最大）距离

    for (int i = 0; i < data_wrapper->data_size; ++i) {
        float dist = EuclideanDistance(data_wrapper->nodes[i], query);
        if ((int)topk.size() < K) {
            topk.emplace(dist, i);
        } else if (dist < topk.top().first) {
            topk.pop();
            topk.emplace(dist, i);
        }
    }

    vector<int> neighbors;
    vector<float> distances;
    neighbors.reserve(topk.size());
    distances.reserve(topk.size());
    while (!topk.empty()) {
        neighbors.emplace_back(topk.top().second);
        distances.emplace_back(topk.top().first);
        topk.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
    std::reverse(distances.begin(), distances.end());

    gettimeofday(&t2, NULL);
    SearchResult res(std::move(neighbors), std::move(distances));
    res.comparisons_made = static_cast<size_t>(data_wrapper->data_size);
    res.search_time = CountTime(t1, t2);
    if (search_info) {
        search_info->internal_search_time = res.search_time;
        search_info->total_comparison = res.comparisons_made;
    }
    return res;
}

} // namespace Compact