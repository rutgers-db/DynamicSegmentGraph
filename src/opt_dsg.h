#pragma once

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

#include "searcher.hpp"
#include "memory.hpp"
#include "compact_graph.h"
#include <bitset>
#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "utils.h"
using namespace base_hnsw;
namespace DSG {

enum class Boundary : unsigned {
    CUR_PIVOT = 253,
    LEFT_MIN = 254,
    RIGHT_MAX = 255
};

class OPT_DSG {
public:
    size_t max_elements_{0};   // Maximum number of elements in the dataset
    size_t dim_{0};            // Dimensionality of each data point
    size_t M_out{0};           // Maximum out-degree for each node in the graph
    size_t ef_construction{0}; // Construction parameter for candidate pool size during search
    int Max_NNS_LEN{0};
    size_t size_data_per_element_{0};  // Size of each data element in memory
    size_t size_links_per_element_{0}; // Size of link data per element
    size_t data_size_{0};              // Total data size for all elements

    size_t size_links_per_layer_{0}; // Size of links per layer
    size_t offsetData_{0};           // Offset in memory to the data section for each element

    char *data_memory_{nullptr};     // Memory pool for storing data and links
    unsigned int *tmp_nns_link_list; // tempory link list that store the current node's neighbors and 4-tuples ranges
    L2Space *space;
    base_hnsw::DISTFUNC<float> fstdistfunc_; // Distance space (e.g., L2 norm)
    void *dist_func_param_{nullptr};         // Parameter for distance function

    size_t metric_distance_computations{0}; // Counts distance computations for performance metrics
    size_t metric_hops{0};                  // Counts hops taken during the search process

    int prefetch_lines{0}; // Cache prefetch size to optimize memory access
    const DataWrapper *data_wrapper;
    BaseIndex::IndexParams *params; // Pointer to data storage (e.g., data points)

    // Constructor to initialize the search structure with the given edge file
    OPT_DSG(std::string edgefilename, BaseIndex::IndexParams &index_params, const DataWrapper *data, base_hnsw::SpaceInterface<float> *space, int max_nns_len) :
        Max_NNS_LEN(max_nns_len) {
        params = &index_params;
        data_wrapper = data;

        // Open the binary file containing the edge information
        std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
        if (!edgefile.is_open())
            std::cerr << ("cannot open " + edgefilename);

        max_elements_ = data_wrapper->data_size; // Total number of data points
        dim_ = data_wrapper->data_dim;           // Dimension of each data point

        fstdistfunc_ = space->get_dist_func();           // Set distance function
        dist_func_param_ = space->get_dist_func_param(); // Set parameters for distance function
        M_out = params->K;                               // Set maximum out-degree for the graph nodes

        data_size_ = (dim_ + 7) / 8 * 8 * sizeof(float);                                                         // Calculate aligned data size per element
        size_links_per_element_ = (2 * Max_NNS_LEN * sizeof(tableint) + sizeof(linklistsizeint) + 31) / 32 * 32; // Link size per element with padding we multply 2 because it should also save the 4-tuple range
        size_data_per_element_ = size_links_per_element_ + data_size_;                                           // Total size per element
        offsetData_ = size_links_per_element_;                                                                   // Offset for data section in memory per element
        prefetch_lines = data_size_ >> 4;                                                                        // Cache line prefetching size

        // Allocate aligned memory for data and links for all elements
        data_memory_ = (char *)memory::align_mm<1 << 21>(max_elements_ * size_data_per_element_);
        tmp_nns_link_list = (unsigned int *)memory::align_mm<1 << 21>(size_links_per_element_);
        
        if (data_memory_ == nullptr || tmp_nns_link_list == nullptr)
            throw std::runtime_error("Not enough memory");

        // Load edges and data points from the binary edge file into memory
        size_t index_points_amount;
        edgefile.read((char *)&index_points_amount, sizeof(index_points_amount));
        assert(index_points_amount <= max_elements_);

        for (unsigned int pid = 0; pid < index_points_amount; pid++) {
            // read the current compressed points in the edge file
            // the current linklist would be stored in tmp_nns_link_list
            read_and_process_cps(edgefile, pid);

            linklistsizeint *linklist = get_linklist(pid);

            std::memcpy((char *)linklist, tmp_nns_link_list, size_links_per_element_);

            // Copy the actual data point from storage into the allocated memory
            // Remember here pid is internal Id not the external id.
            char *data = getDataByInternalId(pid);
            auto &point = data_wrapper->nodes[pid];
            std::size_t data_size = point.size() * sizeof(float);
            std::memcpy(data, point.data(), data_size);
        }
        edgefile.close();
        std::cout << "load index finished ..." << std::endl;
    }

    ~OPT_DSG() {
        free(data_memory_); // Release allocated memory
        free(tmp_nns_link_list);
        data_memory_ = nullptr;
    }

    /**
     * @brief Reduces the `tmp_nns` vector to a maximum specified length by selecting the most "useful" points.
     *
     * This function calculates a usefulness score for each point in `tmp_nns`, which is based on a coverage ratio
     * that represents how well each point's range covers the target intervals (ll-lr and rl-rr). The usefulness score
     * is calculated by:
     *
     * 1. Computing the left and right range spans (`lr - ll + 1` and `rr - rl + 1`).
     * 2. Calculating a total covered range by multiplying the left and right ranges.
     * 3. Calculating the total span from `ll` to `rr`.
     * 4. Using the ratio of covered range to total span as the usefulness score.
     *
     * Points are then sorted in descending order of usefulness score, and only the top `MAX_NNS_LEN` points are retained
     * in `tmp_nns`. This helps ensure that `tmp_nns` contains only the most effective points for further processing,
     * improving both storage efficiency and relevance.
     */
    vector<Compact::CompressedPoint<float>> tmp_nns;
    void drop_points_nns() {
        struct PointInfo {
            float usefulness_score;
            size_t index;
        };

        std::vector<PointInfo> point_scores;
        point_scores.reserve(tmp_nns.size());

        // Calculate usefulness score for each point in tmp_nns
        for (size_t i = 0; i < tmp_nns.size(); ++i) {
            auto &point = tmp_nns[i];

            // if the rr is larger then max_elements, temporarily fix it
            // this is likely based on the building index stage, sometime if rr = max_elements, I will double the rr
            if (point.rr > max_elements_) {
                point.rr = max_elements_;
            }

            // Calculate left and right ranges
            unsigned left_range = point.lr - point.ll + 1;
            unsigned right_range = point.rr - point.rl + 1;

            // Calculate total covered range
            unsigned covered_range = left_range * right_range;

            // Calculate total span from ll to rr
            unsigned total_span = (point.rr - point.external_id + 1) * (point.external_id - point.ll + 1);

            // Calculate coverage ratio
            float coverage_ratio = (total_span > 0) ? static_cast<float>(covered_range) / total_span : 0.0f;

            // Compute usefulness score
            float usefulness_score = coverage_ratio / log(point.rr - point.ll + 1);

            // Store the score along with index
            point_scores.push_back({usefulness_score, i});
        }

        // Sort points by usefulness score in descending order
        std::sort(point_scores.begin(), point_scores.end(),
                  [](const PointInfo &a, const PointInfo &b) {
                      return a.usefulness_score > b.usefulness_score;
                  });

        // Select top Max_NNS_LEN points based on usefulness score
        std::vector<Compact::CompressedPoint<float>> selected_points;
        selected_points.reserve(Max_NNS_LEN);
        for (size_t i = 0; i < std::min(Max_NNS_LEN, static_cast<int>(point_scores.size())); ++i) {
            selected_points.push_back(tmp_nns[point_scores[i].index]);
        }

        // Update tmp_nns with only the selected points
        tmp_nns = std::move(selected_points);
    }

    vector<unsigned int> tmp_external_ids;
    vector<pair<unsigned, unsigned>> tmp_ll;
    vector<pair<unsigned, unsigned>> tmp_lr;
    vector<pair<unsigned, unsigned>> tmp_rl;
    vector<pair<unsigned, unsigned>> tmp_rr;
    void read_and_process_cps(std::ifstream &edgefile, unsigned int cur_node_id) {
        size_t nns_size;
        edgefile.read((char *)&nns_size, sizeof(nns_size));
        tmp_nns.resize(nns_size);
        edgefile.read((char *)tmp_nns.data(), nns_size * sizeof(Compact::CompressedPoint<float>));

        // if the size of it is out limit
        if (static_cast<unsigned>(nns_size) > Max_NNS_LEN) {
            drop_points_nns();
        }

        // Extract external IDs into tmp_external_ids
        tmp_external_ids.clear();
        tmp_external_ids.push_back(0); // Ensure 0 is the first element
        for (const auto &point : tmp_nns) {
            tmp_external_ids.push_back(point.external_id);
        }
        tmp_external_ids.push_back(cur_node_id);                // Insert the current node ID
        sort(tmp_external_ids.begin(), tmp_external_ids.end()); // Sort external IDs

        // Ensure max_elements_ is the last element
        if (tmp_external_ids.back() != max_elements_) {
            tmp_external_ids.push_back(max_elements_);
        }

        // Prepare separate left and right parts relative to cur_node_id
        auto cur_node_it = std::find(tmp_external_ids.begin(), tmp_external_ids.end(), cur_node_id);
        if (cur_node_id == 0) { // corner case, cur_node_id == 0 means there will be 2 0s at the front of the tmp_external_ids
            cur_node_it++;
        }

        vector<unsigned> left_part(tmp_external_ids.begin() + 1, cur_node_it);    // left of cur_node_id
        vector<unsigned> right_part(cur_node_it + 1, tmp_external_ids.end() - 1); // right of cur_node_id

        // Prepare tmp_ll, tmp_lr, tmp_rl, tmp_rr
        tmp_ll.resize(nns_size);
        tmp_lr.resize(nns_size);
        tmp_rl.resize(nns_size);
        tmp_rr.resize(nns_size);

        // Extract ll, lr, rl, rr as pairs with indices
        // Because the original ll rr are [] inclusive range
        // Extract ll, lr, rl, rr as pairs with indices
        for (unsigned i = 0; i < tmp_nns.size(); i++) {
            tmp_ll[i] = (tmp_nns[i].ll != 0) ? make_pair(tmp_nns[i].ll - 1, i) : make_pair(0u, i);
            tmp_lr[i] = make_pair(tmp_nns[i].lr, i);
            tmp_rl[i] = make_pair(tmp_nns[i].rl, i);
            tmp_rr[i] = (tmp_nns[i].rr < max_elements_) ? make_pair(tmp_nns[i].rr + 1, i) : make_pair(static_cast<unsigned int>(max_elements_), i);
        }

        // Helper lambda to find adjusted index
        auto find_index = [&](unsigned value, bool flag) -> unsigned {
            auto it = lower_bound(tmp_external_ids.begin(), tmp_external_ids.end(), value);
            unsigned idx = distance(tmp_external_ids.begin(), it);

            if (flag && *it != value) { // If exact match not found, use the previous index to get the idx where tmp_external_ids[idx] < value
                if (it == tmp_external_ids.begin()) {
                    return static_cast<unsigned>(Boundary::LEFT_MIN);
                }
                --idx;
            }

            if (idx == 0) {
                return static_cast<unsigned>(Boundary::LEFT_MIN);
            } else if (idx == tmp_external_ids.size() - 1) {
                return static_cast<unsigned>(Boundary::RIGHT_MAX);
            } else if (tmp_external_ids[idx] == cur_node_id) {
                return static_cast<unsigned>(Boundary::CUR_PIVOT);
            }

            // Adjust index if in right part (needs -2 due to skipped boundaries)
            return (it > cur_node_it) ? idx - 2 : idx - 1;
        };

        // Store the size of external IDs
        tmp_nns_link_list[0] = (unsigned int)tmp_nns.size();

        // Store external IDs
        // we need to skip the first external id and the last one and also the current node id
        // Copy left and right parts into tmp_nns_link_list
        copy(left_part.begin(), left_part.end(), tmp_nns_link_list + 1);
        copy(right_part.begin(), right_part.end(), tmp_nns_link_list + 1 + left_part.size());

        // Store 4-tuples
        unsigned int *four_tuples_start = tmp_nns_link_list + 1 + tmp_nns.size();
        for (unsigned i = 0; i < tmp_nns.size(); i++) {
            unsigned ll_idx = find_index(tmp_ll[i].first, true);
            unsigned lr_idx = find_index(tmp_lr[i].first, false);
            unsigned rl_idx = find_index(tmp_rl[i].first, true);
            unsigned rr_idx = find_index(tmp_rr[i].first, false);

            // tweak ll and rl because maybe their external
            four_tuples_start[i] = (ll_idx << 24) | (lr_idx << 16) | (rl_idx << 8) | rr_idx;
        }

        // Skip the empty reverse neighbors
        edgefile.read((char *)&nns_size, sizeof(nns_size));
    }

    // Retrieves the data vector for a given internal ID
    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    // Retrieves the link list for a given node ID and layer
    linklistsizeint *get_linklist(tableint internal_id) const {
        return (linklistsizeint *)(data_memory_ + internal_id * size_data_per_element_);
    }

    // Gets the count of neighbors in the link list for a node
    int getListCount(linklistsizeint *ptr) const {
        return *((int *)ptr);
    }

    std::vector<unsigned int> selectEdges(unsigned int point_id, int ql, int qr, searcher::Bitset<uint64_t> &visited_set) {
        std::vector<unsigned int> res;
        res.reserve(32);

        // Get the link list for the given point_id
        linklistsizeint *linklist = get_linklist(point_id);
        unsigned int size = *linklist;                // First element of linklist gives the size
        unsigned int *eids_list = linklist + 1;       // Pointer to external IDs list
        unsigned int *tuples_list = eids_list + size; // Pointer to 4-tuple list

        // Iterate over each external_id in the list
        for (unsigned int i = 0; i < size; i++) {
            unsigned int external_id = *(eids_list + i); // Access the external_id

            // Check if ql <= external_id <= qr
            if (ql <= external_id && external_id <= qr) {
                // Skip neighbor if it has already been visited (checked in `visited_set`)
                if (visited_set.get(external_id))
                    continue;

                unsigned int range_tuple = *(tuples_list + i);

                // Extract ll_idx, lr_idx, rl_idx, rr_idx from the 4-tuple
                unsigned int ll_idx = (range_tuple >> 24);
                unsigned int lr_idx = (range_tuple >> 16) & 0xFF;
                unsigned int rl_idx = (range_tuple >> 8) & 0xFF;
                unsigned int rr_idx = range_tuple & 0xFF;

                // Decode ll, lr, rl, rr from their indices
                unsigned int ll, lr, rl, rr;

                if (ll_idx == static_cast<unsigned int>(Boundary::LEFT_MIN)) {
                    ll = 0;
                } else if (ll_idx == static_cast<unsigned int>(Boundary::RIGHT_MAX)) {
                    ll = max_elements_;
                } else if (ll_idx == static_cast<unsigned int>(Boundary::CUR_PIVOT)) {
                    ll = point_id;
                } else {
                    ll = *(eids_list + ll_idx);
                }

                if (lr_idx == static_cast<unsigned int>(Boundary::LEFT_MIN)) {
                    lr = 0;
                } else if (lr_idx == static_cast<unsigned int>(Boundary::RIGHT_MAX)) {
                    lr = max_elements_;
                } else if (lr_idx == static_cast<unsigned int>(Boundary::CUR_PIVOT)) {
                    lr = point_id;
                } else {
                    lr = *(eids_list + lr_idx);
                }

                if (rl_idx == static_cast<unsigned int>(Boundary::LEFT_MIN)) {
                    rl = 0;
                } else if (rl_idx == static_cast<unsigned int>(Boundary::RIGHT_MAX)) {
                    rl = max_elements_;
                } else if (rl_idx == static_cast<unsigned int>(Boundary::CUR_PIVOT)) {
                    rl = point_id;
                } else {
                    rl = *(eids_list + rl_idx);
                }

                if (rr_idx == static_cast<unsigned int>(Boundary::LEFT_MIN)) {
                    rr = 0;
                } else if (rr_idx == static_cast<unsigned int>(Boundary::RIGHT_MAX)) {
                    rr = max_elements_;
                } else if (rr_idx == static_cast<unsigned int>(Boundary::CUR_PIVOT)) {
                    rr = point_id;
                } else {
                    rr = *(eids_list + rr_idx);
                }

                // Check if ll <= ql <= lr and rl <= qr <= rr
                if (ll <= ql && ql <= lr && rl <= qr && qr <= rr) {
                    res.push_back(external_id);
                }
            }
        }

        return res;
    }

    vector<int> rangeFilteringSearchInRange(
        const BaseIndex::SearchParams *search_params,
        BaseIndex::SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) {
        // 时间测量变量初始化
        timeval tt1, tt2, tt3, tt4;

        // 初始化访问列表
        searcher::Bitset<uint64_t> visited_set(max_elements_);
        float lower_bound = std::numeric_limits<float>::max(); // 最低界限初始化为最大浮点数
        std::priority_queue<pair<float, int>> top_candidates;  // 优先队列存储候选结果
        std::priority_queue<pair<float, int>> candidate_set;   // 候选集优先队列

        search_info->reset();
        size_t num_search_comparison = 0;

        // 初始化三个entry points
        {
            int lbound = query_bound.first;
            int interval = (query_bound.second - lbound) / 3;
            for (size_t i = 0; i < 3; i++) {
                int point = lbound + interval * i;
                char *ep_data = getDataByInternalId(point);
                float dist = fstdistfunc_(ep_data, query.data(), dist_func_param_); // 计算距离
                candidate_set.emplace(make_pair(-dist, point));                        // 将负距离和点ID推入候选集
                visited_set.set(point);                                             // 标记已访问
            }
        }
        gettimeofday(&tt3, NULL);

        size_t hop_counter = 0;
        float total_traversed_nn_amount = 0;
        float pos_point_traverse_counter = 0;
        float pos_point_used_counter = 0;
        auto ef = search_params->search_ef;

        while (!candidate_set.empty()) {
            std::pair<float, int> current_node_pair = candidate_set.top(); // 获取当前节点
            int current_node_id = current_node_pair.second;

            if (-current_node_pair.first > lower_bound) // 如果当前节点的距离大于topk里最远的，则跳出循环
            {
                break;
            }

            hop_counter++;

            candidate_set.pop();

            auto const fetched_nns = selectEdges(current_node_id, query_bound.first, query_bound.second, visited_set);

            // gettimeofday(&tt2, NULL);                              // 结束时间记录
            // AccumulateTime(tt1, tt2, search_info->fetch_nns_time); // 累加邻居检索时间

            int num_edges = fetched_nns.size();
            for (int i = 0; i < std::min(num_edges, 3); ++i) {
                memory::mem_prefetch_L1(getDataByInternalId(fetched_nns[i]), this->prefetch_lines);
            }

            // now iterate fetched nn and calculate distance
            for (auto &candidate_id : fetched_nns) {
                // 标记为已访问
                visited_set.set(candidate_id);

                // 计算距离
                // gettimeofday(&tt1, NULL); // 开始时间记录
                char *cand_data = getDataByInternalId(candidate_id);
                float dist = fstdistfunc_(query.data(),
                                          cand_data,
                                          dist_func_param_);

                num_search_comparison++; // 更新比较次数

                if (top_candidates.size() < ef) {
                    candidate_set.emplace(-dist, candidate_id); // 推入候选集
                    top_candidates.emplace(dist, candidate_id); // 推入顶级候选集
                    lower_bound = top_candidates.top().first;
                } else if (dist < lower_bound) {
                    candidate_set.emplace(-dist, candidate_id); // 推入候选集
                    top_candidates.emplace(dist, candidate_id); // 推入顶级候选集
                    top_candidates.pop();
                    lower_bound = top_candidates.top().first;
                }
                // gettimeofday(&tt2, NULL);                             // 结束时间记录
                // AccumulateTime(tt1, tt2, search_info->cal_dist_time); // 累加距离计算时间
            }
            // total_traversed_nn_amount += float(fetched_nns.size());
        }
        // 构建结果列表
        
        while (top_candidates.size() > search_params->query_K) {
            top_candidates.pop(); // 减少候选集至所需K个
        }

        gettimeofday(&tt4, NULL);
        CountTime(tt3, tt4, search_info->internal_search_time);
        
        vector<int> res;
        while (!top_candidates.empty()) {
            res.emplace_back(top_candidates.top().second); // 提取节点ID构建结果
            top_candidates.pop();
        }
        search_info->total_comparison += num_search_comparison; // 更新总比较次数
        search_info->path_counter += hop_counter;
        search_info->total_traversed_nn_amount = total_traversed_nn_amount;

        // 更新时间统计
        return res; // 返回结果列表
        
    }
    

    // insert a point
    void insert(){
        // get neighbors

        // dfs

        // get positive edges

        // connect neg edges

    }


};
} // namespace DSG
