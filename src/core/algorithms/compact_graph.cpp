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

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "utils.h"
using namespace base_hnsw;

namespace Compact {
template <typename dist_t>
struct CompressedPoint {
    // TODO: here it should be a tuple not a pair, there should be ll, lr, rl, rr
    CompressedPoint(unsigned _external_id, unsigned _ll, unsigned _lr, unsigned _rl, unsigned _rr) :
        external_id(_external_id), ll(_ll), lr(_lr), rl(_rl), rr(_rr) {
    }

    CompressedPoint() {
    }

    unsigned external_id;
    unsigned ll, lr, rl, rr;

    inline bool const if_in_compressed_range(const unsigned &query_L, const unsigned &query_R) const {
        return ((ll <= query_L && query_L <= lr) && (rl <= query_R && query_R <= rr));
    }

    bool operator<(const CompressedPoint &other) const {
        // return this->dist < other.dist;
        return this->external_id < other.external_id;
    }
};

template <typename dist_t>
struct DirectedPointNeighbors {
    vector<CompressedPoint<dist_t>> nns;
    vector<CompressedPoint<dist_t>> rev_nns;

    size_t countNeighbors() {
        return nns.size() + rev_nns.size();
    }
};

template <typename dist_t>
class CompactHNSW : public HierarchicalNSW<float> {
public:
    /**
     * Construct a 2D segment graph hierarchical nearest neighbor search tree (Hierarchical Navigable Small World graph) instance.
     *
     * @param index_params Index parameter configuration object, containing key parameters for the index construction process.
     * @param s Distance calculation space interface, used for performing distance measurement operations.
     * @param max_elements Maximum number of elements, i.e., the maximum number of data points the index can accommodate.
     * @param M Default connectivity, each node is connected to M other nodes by default.
     * @param ef_construction Expansion factor, query efficiency parameter used during construction.
     * @param random_seed Random seed, used to initialize the random number generator.
     */
    CompactHNSW(const BaseIndex::IndexParams &index_params,
                SpaceInterface<float> *s,
                size_t max_elements,
                size_t M = 16,
                size_t ef_construction = 200,
                size_t random_seed = 100) :
        HierarchicalNSW(s, max_elements, M, index_params.ef_construction, random_seed) {
        // Assign the passed index parameter pointer to the member variable
        params = &index_params;

        // Set the maximum expansion factor to the ef_max value in the index parameters
        ef_max_ = index_params.ef_max;
    }

    unsigned max_external_id_ = 0;
    unsigned min_external_id_ = std::numeric_limits<unsigned>::max();

    // log
    size_t forward_batch_nn_amount = 0;
    size_t backward_batch_theoratical_nn_amount = 0;
    size_t drop_points_ = 0;

    size_t Mcurmax;

    // Pointer to a constant BaseIndex::IndexParams type, storing index parameters
    const BaseIndex::IndexParams *params;

    // Pointer to the segment graph neighbor list, representing edge information in the graph structure
    vector<DirectedPointNeighbors<dist_t>> *compact_graph;
    bool if_rebuild_HNSW = false;
    /**
     * Optimize the search process when building the HNSW graph, retaining more neighbor node information.
     * This is basically the original search, searching for the nearest in the current layer of the entire graph.
     * Perhaps it can be combined with (RNN-descent) to improve efficiency??? chaoji left it.
     *
     * @param ep_id Starting point ID
     * @param data_point Data point pointer
     * @param layer Current layer
     * @return Returns a priority queue containing distance and node ID pairs, sorted by distance.
     */
    virtual std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
    searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer) {
        // Get free visited list
        
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // Initialize candidate set and processing set
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidateSet;

        // Store the list of deleted adjacent nodes
        std::vector<pair<dist_t, tableint>> deleted_list;

        // Set the EF value during construction
        size_t ef_construction = ef_max_;

        // Calculate the lower bound of the starting point distance
        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                       dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        // Main loop: traverse the candidate set until it is empty
        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound) {
                break;
            }
            candidateSet.pop();

            // Process the current node
            tableint curNodeNum = curr_el_pair.second;
            std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

            // Get link list data based on the layer
            int *data;
            if (layer == 0) {
                data = (int *)get_linklist0(curNodeNum);
            } else {
                data = (int *)get_linklist(curNodeNum, layer);
            }
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

#ifdef USE_SSE
            // Prefetch instructions to improve performance
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            // Traverse each element in the link list
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                // Prefetch instructions to improve performance
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;

                // Calculate the distance from the candidate node to the target point
                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                // Update the candidate set and visited nodes
                if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    // Prefetch instructions to improve performance
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                                 _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    // Record and remove nodes that exceed the EF limit
                    if (top_candidates.size() > ef_construction) {
                        deleted_list.emplace_back(top_candidates.top());
                        top_candidates.pop();
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }

        // Release visited list resources
        visited_list_pool_->releaseVisitedList(vl);

        // Re-add previously recorded deleted nodes to the candidate set
        for (auto deleted_candidate : deleted_list) {
            top_candidates.emplace(deleted_candidate);
        }

        return top_candidates;
    }

    std::vector<tableint> selectedNeighbors;
    std::vector<tableint> return_list;
    unsigned iter_counter = 0;
    bool complete = false;
    tableint next_closest_entry_point;
    void init_selectedNeighbors() {
        selectedNeighbors.clear();
        return_list.clear();
        iter_counter = 0;
        next_closest_entry_point = 0;
        complete = false;
    }

    void get_selectedNeighbors(tableint passed_c, dist_t dist_to_query, const unsigned &Mcurmax) {
        if (complete)
            return;

        if (return_list.size() >= Mcurmax || iter_counter >= ef_basic_construction_) {
            // The first batch, also use for original HNSW constructing
            next_closest_entry_point =
                return_list.front(); // TODO: check whether the nearest neighbor
            for (auto point : return_list) {
                selectedNeighbors.push_back((tableint)point);
            }

            return_list.clear(); // Clear the return list here
            iter_counter = 0;
            complete = true;
            return;
        }

        iter_counter++;
        bool good = true;

        // Check if it will be pruned by the return list
        for (auto point : return_list) {
            dist_t curdist = fstdistfunc_(getDataByInternalId(point),
                                          getDataByInternalId(passed_c),
                                          dist_func_param_);

            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }

        if (good) {
            return_list.emplace_back(passed_c);
        }
    }

    // the internal ids of points sorted by distance from queue_closest
    std::vector<pair<unsigned, dist_t>> sorted_cands;
    
    /**
     * 支配关系缓存机制
     * 
     * 这是一个用于缓存点对之间支配关系计算结果的哈希表，避免在DFS过程中重复计算相同点对的距离。
     * 
     * 缓存机制设计：
     * - Key: encoded_pair = (pre_nb_idx << 16) + i 
     *   将两个索引编码为一个32位整数，高16位存储pre_nb_idx，低16位存储i
     * - Value: bool类型，表示支配关系结果
     *   true: sorted_cands[pre_nb_idx]支配sorted_cands[i] (距离更近)
     *   false: sorted_cands[pre_nb_idx]不支配sorted_cands[i]
     * 
     * 使用场景：
     * 在DFS递归搜索过程中，同一对点可能在不同的搜索路径中被多次比较，
     * 通过缓存避免重复的昂贵距离计算操作，显著提高算法效率。
     */
    std::unordered_map<unsigned, bool> calculated_pair;
    // TODO: we need to get the boundary of each nbr
    std::vector<bool> if_nbr;
    std::vector<unsigned> nbr_ll;
    std::vector<unsigned> nbr_lr;
    std::vector<unsigned> nbr_rl;
    std::vector<unsigned> nbr_rr;


    /**
     * 深度优先搜索算法，用于生成压缩邻居点的核心函数
     * 
     * 该函数实现了一个复杂的DFS算法，用于在给定的搜索空间中寻找最优的邻居点组合。
     * 算法的核心思想是通过递归搜索，找到一组互不支配的邻居点，并为每个邻居点
     * 计算其有效的查询范围边界。
     * 
     * @param prefix_idx 当前选中的邻居点索引前缀（在sorted_cands中的索引）
     * @param PIVOT_ID 中心点（枢轴点）的外部ID，作为范围划分的基准
     * @param L 当前搜索范围的左边界
     * @param R 当前搜索范围的右边界  
     * @param lr 左侧范围的右边界（用于范围压缩）
     * @param rl 右侧范围的左边界（用于范围压缩）
     * 
     * 算法工作流程：
     * 1. 检查是否达到最大邻居数限制
     * 2. 遍历候选点，检查是否在当前搜索范围[L,R]内
     * 3. 对每个候选点进行支配性检查（避免冗余邻居）
     * 4. 更新候选点的范围边界信息
     * 5. 递归搜索下一层
     * 6. 根据候选点位置动态调整搜索范围
     */
    void dfs(vector<unsigned> &prefix_idx, unsigned PIVOT_ID, unsigned L, unsigned R, unsigned lr, unsigned rl) {
        // 达到最大邻居数限制，停止搜索
        if (prefix_idx.size() == Mcurmax) {
            return;
        }

        // 确定搜索起始索引，避免重复选择
        unsigned st_idx = prefix_idx.empty() ? 0 : prefix_idx.back() + 1;
        
        // 遍历所有候选邻居点
        for (auto i = st_idx; i < sorted_cands.size(); ++i) {
            unsigned cur_external_id = getExternalLabel(sorted_cands[i].first);
            auto cur_dist = sorted_cands[i].second;

            // 检查当前候选点是否在搜索范围[L,R]内
            if (cur_external_id <= R && cur_external_id >= L) {
                bool dominated_flag = false;
                
                // 支配性检查：检查当前点是否被已选择的邻居点支配
                // 如果当前点到某个已选邻居的距离小于到查询点的距离，则被支配
                for (auto const &pre_nb_idx : prefix_idx) {
                    /**
                     * 缓存键编码策略：
                     * 将两个16位索引打包成一个32位整数作为哈希表的键
                     * 高16位: pre_nb_idx (已选择的邻居索引)
                     * 低16位: i (当前候选点索引)
                     * 
                     * 这种编码方式的优势：
                     * 1. 空间效率：用单个32位整数代替pair<int,int>
                     * 2. 哈希效率：整数哈希比pair哈希更快
                     * 3. 唯一性：确保不同点对有不同的键值
                     */
                    unsigned encoded_pair = (pre_nb_idx << 16) + i;

                    // 缓存查找：检查是否已经计算过这对点的支配关系
                    if (calculated_pair.find(encoded_pair) != calculated_pair.end()) {
                        // 缓存命中：直接使用之前计算的结果
                        const auto &domination_result = calculated_pair[encoded_pair];
                        if (domination_result == true) {
                            dominated_flag = true;
                            break;
                        }
                    } else {
                        // 缓存未命中：需要计算距离并缓存结果
                        
                        // 计算两点间的实际欧几里得距离
                        dist_t tmp_dist = fstdistfunc_(getDataByInternalId(sorted_cands[i].first), 
                                                      getDataByInternalId(sorted_cands[pre_nb_idx].first), 
                                                      dist_func_param_);
                        
                        // 支配关系判断：如果已选点到当前点的距离 < 当前点到查询点的距离，则被支配
                        auto domination_result = tmp_dist < cur_dist;
                        
                        // 将计算结果存入缓存，供后续使用
                        calculated_pair[encoded_pair] = domination_result;
                        
                        if (domination_result) {
                            dominated_flag = true;
                            break;
                        }
                    }
                }

                // 跳过被支配的点
                if (dominated_flag)
                    continue;

                // 根据当前点相对于枢轴点的位置，更新范围边界
                unsigned next_lr = (cur_external_id < PIVOT_ID) ? std::min(cur_external_id, lr) : lr;
                unsigned next_rl = (cur_external_id > PIVOT_ID) ? std::max(cur_external_id, rl) : rl;

                // 将当前点添加到选择列表
                prefix_idx.push_back(i);

                // 更新邻居点的范围边界信息
                if (if_nbr[i] == false) {
                    // 首次选择该点，初始化边界
                    if_nbr[i] = true;
                    nbr_ll[i] = L;
                    nbr_lr[i] = next_lr;
                    nbr_rl[i] = next_rl;
                    nbr_rr[i] = R;
                } else {
                    // 已选择过该点，扩展其有效范围
                    nbr_ll[i] = std::min(nbr_ll[i], L);
                    nbr_lr[i] = std::max(nbr_lr[i], next_lr);
                    nbr_rl[i] = std::min(nbr_rl[i], next_rl);
                    nbr_rr[i] = std::max(nbr_rr[i], R);
                }

                // 递归搜索下一层
                dfs(prefix_idx, PIVOT_ID, L, R, next_lr, next_rl);

                // 回溯，移除当前选择
                prefix_idx.pop_back();

                // 根据当前点位置动态调整搜索范围，实现范围分割优化
                if (cur_external_id < lr) {
                    L = cur_external_id + 1;  // 缩小左边界
                } else if (cur_external_id > rl) {
                    R = cur_external_id - 1;  // 缩小右边界
                } else {
                    break;  // 当前点在中间区域，停止搜索
                }
            }
        }
    }

    void generate_compressed_neighbors(
        std::priority_queue<std::pair<dist_t, tableint>> &queue_closest,
        unsigned center_external_id,
        const unsigned &index_k) {
        if (queue_closest.size() == 0) {
            return;
        }
        // unsigned tmp_left_bound = min_external_id_ == 0 ? 0 : min_external_id_ - 1;                                                              // consider we have 0 as min external id
        // unsigned tmp_right_bound = max_external_id_ == std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : max_external_id_ + 1; // consider we have too maximum value as max external id

        // // TODO: Make sure left bound as 0 is perfect? no any bugs? if -1 that will be fined but if 0 I am not sure
        unsigned tmp_left_bound = 0;
        unsigned tmp_right_bound = max_elements_;

        sorted_cands.clear();
        while (!queue_closest.empty()) {
            std::pair<dist_t, tableint> current_pair = queue_closest.top(); // 当前离我最近的点
            dist_t dist_to_query = -current_pair.first;
            sorted_cands.emplace_back(current_pair.second, dist_to_query); // 把queue_closest里的按照顺序塞进sorted_cands里面
            queue_closest.pop();
            get_selectedNeighbors(current_pair.second, dist_to_query, index_k);
        }

        // Not need to find compressed points, not need
        if (if_rebuild_HNSW == true) 
            return;

        // some initiliazation for some variables serving for dfs function
        vector<unsigned> prefix_idx;
        prefix_idx.reserve(Mcurmax);

        // TODO: we can shrink this memory that they do not need so much space we can integrate them into one data structure
        if_nbr.resize(sorted_cands.size());
        nbr_ll.resize(sorted_cands.size());
        nbr_lr.resize(sorted_cands.size());
        nbr_rl.resize(sorted_cands.size());
        nbr_rr.resize(sorted_cands.size());
        std::fill_n(if_nbr.begin(), if_nbr.size(), false);
        
        // 清空支配关系缓存，为新一轮的DFS搜索做准备
        // 每次generate_compressed_neighbors调用都会产生新的sorted_cands，
        // 因此之前的缓存结果不再适用，必须清空
        calculated_pair.clear();

        // always choosing the range [0, max_elements] as dfs input
        // If choose current [min_element_external_id, max_element_exteranl_id] that will make the recall a little bit lower
        dfs(prefix_idx, center_external_id, tmp_left_bound, tmp_right_bound, center_external_id, center_external_id);

        // generate the compressed point
        for (unsigned i = 0; i < if_nbr.size(); i++) {
            if (if_nbr[i]) {
                // TODO: each point is actually corresponding to a boundary we need to get the accurate positions
                unsigned tmp_external_id = getExternalLabel(sorted_cands[i].first);
                compact_graph->at(center_external_id).nns.emplace_back(tmp_external_id, nbr_ll[i], nbr_lr[i], nbr_rl[i], nbr_rr[i]);
            }
        }
        sort(compact_graph->at(center_external_id).nns.begin(), compact_graph->at(center_external_id).nns.end());
    }

    void gen_rev_neighbors(unsigned center_external_id) {
        auto &nns = compact_graph->at(center_external_id).nns;
        for (auto &point : nns) {
            auto rev_point_id = point.external_id;
            auto &rev_nns = compact_graph->at(rev_point_id).rev_nns;
            rev_nns.emplace_back(center_external_id, point.ll, point.lr, point.rl, point.rr);
        }
        return;
    }


    virtual tableint mutuallyConnectNewElementLevel0(
        const void *data_point, 
        tableint cur_c,        
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates, 
        int level,                                           
        bool isUpdate)                                       
    {
        Mcurmax = maxM0_; 


        unsigned external_id = getExternalLabel(cur_c);

        if (external_id > max_external_id_)
            max_external_id_ = external_id;
        if (external_id < min_external_id_)
            min_external_id_ = external_id;

        {
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            while (!top_candidates.empty()) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            init_selectedNeighbors();

            
            generate_compressed_neighbors(queue_closest, external_id, (unsigned)Mcurmax);

            if (if_rebuild_HNSW == false) {
                gen_rev_neighbors(external_id);
            }

            if (return_list.size()) // 这种情况是上面的while 跑完了 但是一个batch都没满 所以需要单独处理
            {
                // The first batch, also use for original HNSW constructing
                next_closest_entry_point = return_list.front();
                for (auto point : return_list) {
                    selectedNeighbors.push_back(point);
                }

                return_list.clear();
            }
        }

        {
            linklistsizeint *ll_cur;
            ll_cur = get_linklist0(cur_c);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error(
                    "The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *)(ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error(
                        "Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(
                link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            ll_other = get_linklist0(selectedNeighbors[idx]);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error(
                    "Trying to make a link on a non-existent level");

            tableint *data = (tableint *)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of
            // `selectedNeighbors[idx]` then no need to modify any connections or
            // run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(
                        getDataByInternalId(cur_c),
                        getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfunc_(getDataByInternalId(data[j]),
                                         getDataByInternalId(selectedNeighbors[idx]),
                                         dist_func_param_),
                            data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }
};

class IndexCompactGraph : public BaseIndex {
public:
    vector<DirectedPointNeighbors<float>> directed_indexed_arr;
    base_hnsw::DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;
    VisitedListPool *visited_list_pool_ = nullptr;
    IndexInfo *index_info = nullptr;
    const BaseIndex::IndexParams *index_params_;
    CompactHNSW<float> *hnsw;

    IndexCompactGraph(base_hnsw::SpaceInterface<float> *s,
                      const DataWrapper *data) :
        BaseIndex(data) {
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        index_info = new IndexInfo();
        index_info->index_version_type = "IndexCompactGraph";
    }

    void printOnebatch() {
        cout << "Print one batch" << endl;
        for (auto cp :
             directed_indexed_arr[data_wrapper->data_size / 2].nns) {
            cout << "[" << cp.external_id << "," << cp.ll << ","
                 << cp.rr << "], ";
        }
        cout << endl;
    }

    void countNeighbrs() {
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

    void buildIndex(const IndexParams *index_params) override {
        cout << "Building Index using " << index_info->index_version_type << endl;
        timeval tt1, tt2;
        visited_list_pool_ =
            new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

        index_params_ = index_params;
        // build HNSW
        L2Space space(data_wrapper->data_dim);
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
    };

    void initForScabilityExp(const IndexParams *index_params, L2Space *space) {
        if(visited_list_pool_ == nullptr)
            visited_list_pool_ =
                new base_hnsw::VisitedListPool(1, data_wrapper->data_size);
        index_params_ = index_params;
        // build HNSW
        hnsw = new CompactHNSW<float>(
            *index_params, space, 2 * data_wrapper->data_size, index_params->K,
            index_params->ef_construction, index_params->random_seed);

        // directed_indexed_arr.clear();
        directed_indexed_arr.resize(data_wrapper->data_size);
        hnsw->compact_graph = &directed_indexed_arr;
    }

    void rebuild_batchInHNSW(vector<unsigned> &nodes_ids) {
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

    void insert_batch(vector<unsigned> &nodes_ids) {
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

    vector<unsigned> fetched_nns;

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
    vector<int> rangeFilteringSearchInRange(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) override {
        // 预分配邻居缓存空间并清空
        fetched_nns.reserve(100);
        fetched_nns.clear();

        // 时间测量变量初始化
        timeval tt1, tt2, tt3, tt4;

        // 初始化访问标记系统，用于避免重复访问
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        
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
            for (auto i = 0; i < pos_edges.size(); i++) {
                const unsigned &candidate_id = pos_edges[i].external_id;
                
                // 基本范围检查
                if (candidate_id < query_bound.first)
                    continue;
                if (candidate_id > query_bound.second) 
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

    vector<int> rangeFilteringSearchOutBound(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) override {
        return vector<int>();
    }

    // Save function to store the IndexCompactGraph to a file
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

    // Load function to load the IndexCompactGraph from a file
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

    ~IndexCompactGraph() {
        delete hnsw;
        delete index_info;
        directed_indexed_arr.clear();
        delete visited_list_pool_;
    }
};
} // namespace Compact