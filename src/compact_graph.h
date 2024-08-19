/**
 * @file index_recursion_batch.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for arbitrary range filtering search
 * Compress N SegmentGraph
 * @date 2023-06-19; Revised 2024-01-10
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>
#include <random>

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "utils.h"
#include "sorted_array.h"
using namespace base_hnsw;

namespace Compact
{   
    template <typename dist_t>
    struct CompressedPoint
    {
        // 使用初始化列表构造函数，简化赋值操作并提高效率
        CompressedPoint(unsigned _external_id, unsigned l, unsigned r, dist_t _dist)
            : external_id(_external_id), left_bound(l), right_bound(r), dist(_dist) {}

        // 默认构造函数
        CompressedPoint() {}

        unsigned external_id, left_bound, right_bound;
        dist_t dist;
        size_t flag = 0;

        bool if_in_compressed_range(const unsigned center_external_id, const unsigned query_L, const unsigned query_R) const
        {
            auto left_inner = std::min(center_external_id, external_id);
            auto right_inner = std::max(center_external_id, external_id);
            return (query_L <= left_inner && query_R >= right_inner) && (left_bound <= query_L && right_bound >= query_R);
        }

        bool if_not_dominated(const size_t &cur_dom_relation)
        {   
            // 换成bool return
            return (flag & cur_dom_relation) == 0;
        }

        // 重载小于运算符 (<)，用于按照距离从小到大排序
        bool operator<(const CompressedPoint &other) const
        {
            return this->dist < other.dist;
        }
    };

    template <typename dist_t>
    struct DirectedPointNeighbors
    {
        vector<CompressedPoint<dist_t>> nns;

        size_t countNeighbors()
        {
            return nns.size();
        }
    };

    template <typename dist_t>
    class CompactHNSW : public HierarchicalNSW<float>
    {
    public:
        /**
         * 构造一个二维段图层次邻近搜索树（Hierarchical Navigable Small World graph）实例.
         *
         * @param index_params 索引参数配置对象，包含索引构建过程中的关键参数.
         * @param s 距离计算空间接口，用于执行距离度量操作.
         * @param max_elements 最大元素数量，即索引能容纳的最大数据点数.
         * @param M 默认连接度，每个节点默认与其他M个节点相连.
         * @param ef_construction 扩展因子，在构造过程中使用的查询效率参数.
         * @param random_seed 随机种子，用于初始化随机数生成器.
         */
        CompactHNSW(const BaseIndex::IndexParams &index_params,
                    SpaceInterface<float> *s, size_t max_elements,
                    size_t M = 16, size_t ef_construction = 200,
                    size_t random_seed = 100)
            : HierarchicalNSW(s, max_elements, M, index_params.ef_construction,
                              random_seed)
        {
            // 将传入的索引参数指针赋值给成员变量
            params = &index_params;

            // 设置最大扩展因子为索引参数中的ef_max值
            ef_max_ = index_params.ef_max;
        }

        unsigned max_external_id_ = 0;
        unsigned min_external_id_ = std::numeric_limits<unsigned>::max();

        // log
        size_t forward_batch_nn_amount = 0;
        size_t backward_batch_theoratical_nn_amount = 0;
        size_t drop_points_ = 0;

        // 指向BaseIndex::IndexParams类型的常量指针，存储索引参数
        const BaseIndex::IndexParams *params;

        // 存储指向段图邻居列表的指针，表示图结构中的边信息
        vector<DirectedPointNeighbors<dist_t>> *compact_graph;

        /**
         * 在构建HNSW图时优化搜索过程，保留更多邻居节点信息。
         * 这个是基本就是原本的search 就是在整个图里当前层搜最近的
         * 或许可以结合（RNN-descent）以提升效率??? chaoji left 的。
         *
         * @param ep_id 起始点ID
         * @param data_point 数据点指针
         * @param layer 当前层级
         * @return 返回一个优先队列，其中包含距离和节点ID对，按距离排序。
         */
        virtual std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
        searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer)
        {
            // 获取空闲访问列表
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // 初始化候选集和待处理集合
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                candidateSet;

            // 存储删除的邻接节点列表
            std::vector<pair<dist_t, tableint>> deleted_list;

            // 设置构造时的EF值
            size_t ef_construction = ef_max_;

            // 计算起始点的距离下界
            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                           dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            // 主循环：遍历候选集直到为空
            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound)
                {
                    break;
                }
                candidateSet.pop();

                // 处理当前节点
                tableint curNodeNum = curr_el_pair.second;
                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                // 根据层级获取链接列表数据
                int *data;
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);

#ifdef USE_SSE
                // 预取指令提高性能
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                // 遍历链接列表中的每个元素
                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    // 预取指令提高性能
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;

                    // 计算候选节点到目标点的距离
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    // 更新候选集和已访问节点
                    if (top_candidates.size() < ef_construction || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        // 预取指令提高性能
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                                     _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        // 记录并移除超出EF限制的节点
                        if (top_candidates.size() > ef_construction)
                        {
                            deleted_list.emplace_back(top_candidates.top());
                            top_candidates.pop();
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }

            // 释放访问列表资源
            visited_list_pool_->releaseVisitedList(vl);

            // 将之前记录的删除节点重新加入候选集
            for (auto deleted_candidate : deleted_list)
            {
                top_candidates.emplace(deleted_candidate);
            }

            return top_candidates;
        }

        std::vector<tableint> selectedNeighbors;
        std::vector<tableint> return_list;
        unsigned iter_counter = 0;
        bool complete = false;
        tableint next_closest_entry_point;
        void init_selectedNeighbors()
        {
            selectedNeighbors.clear();
            return_list.clear();
            iter_counter = 0;
            next_closest_entry_point = 0;
            complete = false;
        }

        void get_selectedNeighbors(tableint passed_c, dist_t dist_to_query, const unsigned &Mcurmax)
        {
            if (complete)
                return;

            if (return_list.size() >= Mcurmax || iter_counter >= ef_basic_construction_)
            {
                // The first batch, also use for original HNSW constructing
                next_closest_entry_point =
                    return_list.front(); // TODO: check whether the nearest neighbor
                for (auto point : return_list)
                {
                    selectedNeighbors.push_back((tableint)point);
                }

                return_list.clear(); // 这里会清空return list
                iter_counter = 0;
                complete = true;
                return;
            }

            iter_counter++;
            bool good = true;

            // 查看会不会被在return list的给prune掉
            for (auto point : return_list)
            {
                dist_t curdist = fstdistfunc_(getDataByInternalId(point),
                                              getDataByInternalId(passed_c),
                                              dist_func_param_);

                if (curdist < dist_to_query)
                {
                    good = false;
                    break;
                }
            }

            if (good)
            {
                return_list.emplace_back(passed_c);
            }
        }

        void generate_compressed_neighbors(
            std::priority_queue<std::pair<dist_t, tableint>> &queue_closest,
            unsigned center_external_id, const unsigned &index_k)
        {
            if (queue_closest.size() == 0)
            {
                return;
            }

            auto arr_size = index_k + 1; // maintain a arr of size k+1
            SortedArrayBase *right_sortedarr = new MinSortedArray(arr_size);
            SortedArrayBase *left_sortedarr = new MaxSortedArray(arr_size);

            unsigned cur_rank;
            // unsigned tmp_left_bound = min_external_id_ == 0 ? 0 : min_external_id_ - 1;                                                              // consider we have 0 as min external id
            // unsigned tmp_right_bound = max_external_id_ == std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : max_external_id_ + 1; // consider we have too maximum value as max external id
            unsigned tmp_left_bound = 0;
            unsigned tmp_right_bound = max_elements_;
            left_sortedarr->addPoint(tmp_left_bound, cur_rank);
            right_sortedarr->addPoint(tmp_right_bound, cur_rank);

            while (!queue_closest.empty())
            {
                std::pair<dist_t, tableint> curent_pair = queue_closest.top(); // 当前离我最近的点
                dist_t dist_to_query = -curent_pair.first;

                queue_closest.pop();
                /*这里调用回掉函数*/
                get_selectedNeighbors(curent_pair.second, dist_to_query, index_k);
                unsigned current_external_id = getExternalLabel(curent_pair.second);
                SortedArrayBase *cur_sortedarr;
                SortedArrayBase *other_sortedarr;

                // 根据当前外部ID与中心外部ID的关系选择当前排序数组和另一个排序数组
                if (current_external_id > center_external_id)
                {
                    cur_sortedarr = right_sortedarr;
                    other_sortedarr = left_sortedarr;
                }
                else
                {
                    cur_sortedarr = left_sortedarr;
                    other_sortedarr = right_sortedarr;
                }

                bool is_inserted = cur_sortedarr->addPoint(current_external_id, cur_rank);

                // only means current inserted point rank top k
                if (is_inserted && cur_rank <= index_k)
                {
                    unsigned rev_rank = arr_size - cur_rank + 1;
                    unsigned bound_1 = cur_sortedarr->getBackBound();                // we have consider the cur sortedarr is not full
                    unsigned bound_2 = other_sortedarr->getSpecifiedBound(rev_rank); // we have considered the size of other arr is lower than rev_rank
                    unsigned L = std::min(bound_1, bound_2);
                    unsigned R = std::max(bound_1, bound_2);

                    // generate the compressed point
                    // assert(L <= current_external_id);
                    // assert(R >= center_external_id);
                    compact_graph->at(center_external_id).nns.emplace_back(current_external_id, L, R, dist_to_query);
                }
            }

            delete right_sortedarr;
            delete left_sortedarr;
        }

        void gen_rev_neighbors(unsigned center_external_id)
        {
            auto &nns = compact_graph->at(center_external_id).nns;
            for (auto &point : nns)
            {
                auto rev_point_id = point.external_id;
                auto &rev_nns = compact_graph->at(rev_point_id).nns;
                rev_nns.emplace_back(center_external_id, point.left_bound, point.right_bound, point.dist);
            }
            return;
        }

        void gen_domination_relationship(unsigned center_external_id){
            auto & nns = compact_graph->at(center_external_id).nns;
            std::sort(nns.begin(), nns.end());
            for(unsigned i = 1; i < nns.size(); i++ ){
                size_t tmp_flag = 0;
                auto cur_dist = nns[i].dist;
                unsigned j_limit = std::min(i, static_cast<unsigned>(64));
                for(unsigned j = 0; j < j_limit; j++){
                    // calculate distance
                    auto tmp_dist = fstdistfunc_(getDataByLabel(nns[i].external_id),
                                             getDataByLabel(nns[j].external_id),
                                             dist_func_param_);
                    if(tmp_dist < cur_dist){
                        tmp_flag |= (1 << j);
                    }
                }
                nns[i].flag = tmp_flag;
            }
        }

        /**
         * @file src/compact_graph.h
         * @brief 互连新元素并递归地应用启发式剪枝算法并且对插入元素的attribute没有要求
         *
         * 此函数用于连接新的数据点到图中的现有节点，
         * 并通过优先队列处理候选邻居以优化连接过程。
         */

        virtual tableint mutuallyConnectNewElementLevel0(
            const void *data_point, /**< 当前数据点 */
            tableint cur_c,         /**< 当前节点的内部标识符 */
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> &top_candidates, /**< 候选邻居列表 */
            int level,                                           /**< layer level */
            bool isUpdate)                                       /**< 是否是更新已有的点 还是插入新的点 */
        {
            size_t Mcurmax = maxM0_; // 最大邻接数量

            // 获取当前节点的外部标签
            unsigned external_id = getExternalLabel(cur_c);

            // 更新目前所有节点的最大和最小外部标签值
            if (external_id > max_external_id_)
                max_external_id_ = external_id;
            if (external_id < min_external_id_)
                min_external_id_ = external_id;

            {
                // MAX_POS recursive pruning, combining into connect function.

                // 处理优先级队列：从最近到最远排序候选者
                std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
                while (!top_candidates.empty())
                {
                    queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                    top_candidates.pop();
                }

                init_selectedNeighbors();

                generate_compressed_neighbors(queue_closest, external_id, (unsigned)Mcurmax);

                gen_rev_neighbors(external_id);

                if (return_list.size()) // 这种情况是上面的while 跑完了 但是一个batch都没满 所以需要单独处理
                {
                    // The first batch, also use for original HNSW constructing
                    next_closest_entry_point = return_list.front();
                    for (auto point : return_list)
                    {
                        selectedNeighbors.push_back(point);
                    }

                    return_list.clear();
                }
            }

            // 把找到的selectedNeighbors放进cur_c的邻接列表
            {
                linklistsizeint *ll_cur;
                ll_cur = get_linklist0(cur_c);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error(
                        "The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error(
                            "Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {
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
                if (isUpdate)
                {
                    for (size_t j = 0; j < sz_link_list_other; j++)
                    {
                        if (data[j] == cur_c)
                        {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // 这里就正常跑就完事了
                // If cur_c is already present in the neighboring connections of
                // `selectedNeighbors[idx]` then no need to modify any connections or
                // run the heuristics.
                if (!is_cur_c_present)
                {
                    if (sz_link_list_other < Mcurmax)
                    {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else
                    {
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

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]),
                                             getDataByInternalId(selectedNeighbors[idx]),
                                             dist_func_param_),
                                data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0)
                        {
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

    class IndexCompactGraph : public BaseIndex
    {
    public:
        vector<DirectedPointNeighbors<float>> directed_indexed_arr;

        IndexCompactGraph(base_hnsw::SpaceInterface<float> *s,
                          const DataWrapper *data)
            : BaseIndex(data)
        {
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            index_info = new IndexInfo();
            index_info->index_version_type = "IndexCompactGraph";
        }
        base_hnsw::DISTFUNC<float> fstdistfunc_;
        void *dist_func_param_;

        VisitedListPool *visited_list_pool_;
        IndexInfo *index_info;
        const BaseIndex::IndexParams *index_params_;

        void printOnebatch()
        {
            cout << "Print one batch" << endl;
            for (auto cp :
                 directed_indexed_arr[data_wrapper->data_size / 2].nns)
            {
                cout << "[" << cp.external_id << "," << cp.left_bound << ","
                     << cp.right_bound << "], ";
            }
            cout << endl;
        }

        /**
         * @brief 计算图中的邻居节点数量统计信息
         *
         * 此方法遍历有向图索引数组以计算平均前向邻居数、最大前向批量邻居数，
         * 平均反向邻居数、最大反向邻居数以及相关批处理计数。
         */
        void countNeighbrs()
        {
            size_t max_nns_len = 0;
            // 如果有向图索引不为空，则开始处理
            if (!directed_indexed_arr.empty())
            {
                // 遍历所有节点的前向邻居列表
                for (unsigned j = 0; j < directed_indexed_arr.size(); j++)
                {
                    index_info->nodes_amount += directed_indexed_arr[j].countNeighbors();
                    max_nns_len = std::max(max_nns_len, directed_indexed_arr[j].nns.size());
                }
            }

            // 计算平均前向邻居数
            index_info->avg_forward_nns = index_info->nodes_amount / static_cast<float>(data_wrapper->data_size);

            // 打印日志（如果启用）
            if (isLog)
            {
                cout << "Max. nns length of one point" << max_nns_len << endl;
                cout << "Sum of forward nn #: " << index_info->nodes_amount << endl;
                cout << "Avg. forward nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
                cout << "Avg. delta nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
            }
        }

        void buildIndex(const IndexParams *index_params) override
        {
            cout << "Building Index using " << index_info->index_version_type << endl;
            timeval tt1, tt2;
            visited_list_pool_ =
                new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

            index_params_ = index_params;
            // build HNSW
            L2Space space(data_wrapper->data_dim);
            CompactHNSW<float> hnsw(
                *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
                index_params->ef_construction, index_params->random_seed);

            directed_indexed_arr.clear();
            directed_indexed_arr.resize(data_wrapper->data_size);
            hnsw.compact_graph = &directed_indexed_arr;
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
            for (size_t i : permutation)
            {
                hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
            }
            
            // generate online domination relationship
            // for (size_t i : permutation)
            // {
            //     hnsw.gen_domination_relationship(i);
            // }
            gettimeofday(&tt2, NULL);
            index_info->index_time = CountTime(tt1, tt2);

            cout << "All the forward batch nn #: " << hnsw.forward_batch_nn_amount << endl;
            cout << "Theoratical backward batch nn #: " << hnsw.backward_batch_theoratical_nn_amount << endl;

            // count neighbors number
            countNeighbrs();

            // if (index_params->print_one_batch)
            // {
            //     printOnebatch();
            // }
        };

        /**
         * @brief 范围过滤搜索，在范围内节点上计算距离。
         *
         * 此方法执行范围过滤搜索算法，仅在指定范围内的节点上计算距离，
         * 并返回最邻近点列表。
         *
         * @param search_params 搜索参数指针，包含控制批处理阈值和搜索ef值。
         * @param search_info 搜索信息结构体指针，用于记录搜索过程中的统计信息。
         * @param query 查询向量。
         * @param query_bound 查询边界对，定义查询范围。
         * @return vector<int> 返回最邻近点ID列表。
         */
        vector<int> rangeFilteringSearchInRange(
            const SearchParams *search_params, SearchInfo *search_info,
            const vector<float> &query,
            const std::pair<int, int> query_bound) override
        {
            // 时间测量变量初始化
            timeval tt1, tt2, tt3, tt4;

            // 初始化访问列表
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            float lower_bound = std::numeric_limits<float>::max(); // 最低界限初始化为最大浮点数
            std::priority_queue<pair<float, int>> top_candidates;  // 优先队列存储候选结果
            std::priority_queue<pair<float, int>> candidate_set;   // 候选集优先队列

            search_info->total_comparison = 0;
            search_info->internal_search_time = 0;
            search_info->cal_dist_time = 0;
            search_info->fetch_nns_time = 0;
            num_search_comparison = 0;

            // 初始化三个entry points
            {
                int lbound = query_bound.first;
                int interval = (query_bound.second - lbound) / 3;
                for (size_t i = 0; i < 3; i++)
                {
                    int point = lbound + interval * i;
                    float dist = EuclideanDistance(data_wrapper->nodes[point], query); // 计算距离
                    candidate_set.push(make_pair(-dist, point));                       // 将负距离和点ID推入候选集
                    visited_array[point] = visited_array_tag;                          // 标记已访问
                }
            }
            gettimeofday(&tt3, NULL);

            // only one center
            // float dist_enter = EuclideanDistance(data_nodes[l_bound], query);
            // candidate_set.push(make_pair(-dist_enter, l_bound));
            // TODO: How to find proper enters. // looks like useless

            size_t hop_counter = 0;
            while (!candidate_set.empty())
            {
                std::pair<float, int> current_node_pair = candidate_set.top(); // 获取当前节点
                int current_node_id = current_node_pair.second;

                if (-current_node_pair.first > lower_bound) // 如果当前节点的距离大于topk里最远的，则跳出循环
                {
                    break;
                }

#ifdef LOG_DEBUG_MODE
                cout << "current node: " << current_node_pair.second << "  -- "
                     << -current_node_pair.first << endl;
#endif

                // if (search_info->is_investigate) {
                //   search_info->SavePathInvestigate(current_node_pair.second,
                //                                    -current_node_pair.first,
                //                                    hop_counter, num_search_comparison);
                // }
                hop_counter++;

                candidate_set.pop();

                // // only search when candidate point is inside the range
                // this can be commented because no way to do this
                if (current_node_id < query_bound.first || current_node_id > query_bound.second)
                {
                    cout << "no satisfied range point" << endl;
                    continue;
                }

                // search cw on the fly
                vector<unsigned> neighbors_in_range;

                gettimeofday(&tt1, NULL);
                {
                    // size_t visited_flag = 0;
                    for(unsigned i = 0; i < directed_indexed_arr[current_node_id].nns.size(); i++){
                        auto & cp = directed_indexed_arr[current_node_id].nns[i];
                        if (cp.if_in_compressed_range(current_node_id, query_bound.first, query_bound.second))
                        {   
                            neighbors_in_range.emplace_back(cp.external_id);
                            // if(cp.if_not_dominated(visited_flag)){
                            //     neighbors_in_range.emplace_back(cp.external_id);
                            //     if(i < 64){
                            //         visited_flag |= 1 << i;
                            //     }
                            // }
                        }
                    }
                }
                gettimeofday(&tt2, NULL);                              // 结束时间记录
                AccumulateTime(tt1, tt2, search_info->fetch_nns_time); // 累加邻居检索时间

                // 处理邻居集合
                gettimeofday(&tt1, NULL); // 开始时间记录

                // TODO whether we need to limit the neighbors
                // unsigned cnt_positive_through_neighbors = 0;
                // const auto Mcurmax = 2 * index_params_->K; // 也需要看看效果 不过现在反向边和正向边放在一起，当然得扩大成2K // 也可以试试不加这个条件，理论上应该只会提升时间和recall 不会降低recall和time吧

                for (auto candidate_id : neighbors_in_range)
                {
                    // if (candidate_id < query_bound.first || candidate_id > query_bound.second) // 忽略越界节点
                    //     continue;

                    // if (cnt_positive_through_neighbors < Mcurmax)
                    //     cnt_positive_through_neighbors++;
                    // else
                    //     break;

                    if (!(visited_array[candidate_id] == visited_array_tag)) // 若未被访问过
                    {
                        visited_array[candidate_id] = visited_array_tag; // 标记为已访问

                        // 计算距离
                        float dist = fstdistfunc_(query.data(),
                                                  data_wrapper->nodes[candidate_id].data(),
                                                  dist_func_param_);

                        num_search_comparison++; // 更新比较次数
                        if (top_candidates.size() < search_params->search_ef || lower_bound > dist)
                        {
                            candidate_set.push(make_pair(-dist, candidate_id)); // 推入候选集
                            top_candidates.push(make_pair(dist, candidate_id)); // 推入顶级候选集
                            if (top_candidates.size() > search_params->search_ef)
                            {
                                top_candidates.pop(); // 维护候选集大小
                            }
                            if (!top_candidates.empty())
                            {
                                lower_bound = top_candidates.top().first; // 更新最低界限
                            }
                        }
                    }
                }

                gettimeofday(&tt2, NULL);                             // 结束时间记录
                AccumulateTime(tt1, tt2, search_info->cal_dist_time); // 累加距离计算时间
            }

            // 构建结果列表
            vector<int> res;
            while (top_candidates.size() > search_params->query_K)
            {
                top_candidates.pop(); // 减少候选集至所需K个
            }

            while (!top_candidates.empty())
            {
                res.emplace_back(top_candidates.top().second); // 提取节点ID构建结果
                top_candidates.pop();
            }
            search_info->total_comparison += num_search_comparison; // 更新总比较次数

#ifdef LOG_DEBUG_MODE
            print_set(res);
            cout << l_bound << "," << r_bound << endl;
            assert(false);
#endif

            // 释放资源和更新时间统计
            visited_list_pool_->releaseVisitedList(vl);
            gettimeofday(&tt4, NULL);
            CountTime(tt3, tt4, search_info->internal_search_time);
            return res; // 返回结果列表
        }

        vector<int> rangeFilteringSearchOutBound(
            const SearchParams *search_params, SearchInfo *search_info,
            const vector<float> &query,
            const std::pair<int, int> query_bound) override
        {
            return vector<int>();
        }
        ~IndexCompactGraph()
        {
            delete index_info;
            directed_indexed_arr.clear();
            delete visited_list_pool_;
        }
    };
} // namespace SeRF