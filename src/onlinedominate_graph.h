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

using namespace base_hnsw;

namespace OnlineDominate
{
    struct OnlineDominateRelation
    {
        vector<unsigned int> domination_flags;

        // 判断是否被 dominate
        // 如果domination_flags[nn_id]
        // 如果nn_id节点未被具有flag特性的节点支配则返回true；否则返回false
        bool if_dominated(unsigned nn_id, unsigned int flag)
        {
            return flag & domination_flags[nn_id] != 0;
        }

        OnlineDominateRelation() {}
        OnlineDominateRelation(unsigned int size)
        {
            domination_flags.resize(size);
        }
    };

    template <typename dist_t>
    class OnlineDominationGraph : public HierarchicalNSW<float>
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
        OnlineDominationGraph(const BaseIndex::IndexParams &index_params,
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

        OnlineDominationGraph(const BaseIndex::IndexParams &index_params,
                              SpaceInterface<float> *s, const std::string & location)
            : HierarchicalNSW(s, location)
        {
            // 将传入的索引参数指针赋值给成员变量
            params = &index_params;

            // 设置最大扩展因子为索引参数中的ef_max值
            ef_max_ = index_params.ef_max;
        }
        
        // 指向BaseIndex::IndexParams类型的常量指针，存储索引参数
        const BaseIndex::IndexParams *params;

        // 存储指向段图邻居列表的指针，表示图结构中的边信息
        vector<OnlineDominateRelation> *compact_graph;

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

        // only used for Level 0
        void getDominationRelationship(unsigned target_external_id)
        {
            tableint cur_c;
            auto search = label_lookup_.find(target_external_id);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            cur_c = search->second;
            auto &relation = compact_graph->at(target_external_id);
            int *data = (int *)get_linklist0(cur_c);
            char *vec_data = (getDataByInternalId(cur_c));
            size_t size = getListCount((linklistsizeint *)data);
            relation.domination_flags.resize(size);
            tableint *datal = (tableint *)(data + 1);
            auto pre_dist = -1.0;
            for (size_t i = 0; i < size; i++)
            {
                tableint nn_id = *(datal + i);
                unsigned domination_flag = 0;
                char *nn_vecdata = (getDataByInternalId(nn_id));
                auto dist = fstdistfunc_(vec_data, nn_vecdata, dist_func_param_);
                for (size_t j = 0; j < i; j++)
                {
                    tableint cur_id = *(datal + j);
                    char *currObj1 = (getDataByInternalId(cur_id));
                    auto cur_dist = fstdistfunc_(nn_vecdata, currObj1, dist_func_param_);
                    if (cur_dist < dist)
                    {
                        domination_flag |= 1 << j;
                    }
                }
                relation.domination_flags[i] = domination_flag;
                assert(dist >= pre_dist);
                pre_dist = dist;
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerWithOnlineDomination(tableint ep_id, const void *data_point, size_t ef, size_t & comparison_count) const
        {

            // 获取空闲访问列表
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // 初始化两个优先队列：顶部候选者和候选集
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates, candidate_set;

            // 计算下界距离值
            dist_t lowerBound;

            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                       dist_func_param_);
            comparison_count++;
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);

            // 标记起始节点为已访问
            visited_array[ep_id] = visited_array_tag;

            // 主循环处理候选节点
            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                if ((-current_node_pair.first) > lowerBound)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                unsigned visited_nn_flag = 0;
                auto cur_external_id = getExternalLabel(current_node_id);
                auto &relation = compact_graph->at(cur_external_id);
                // 遍历链接列表
                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);

                    // 如果当前候选节点未被访问过，则进一步处理
                    if (!(visited_array[candidate_id] == visited_array_tag) && relation.if_dominated(j - 1, visited_nn_flag) == false) // 
                    {
                        visited_array[candidate_id] = visited_array_tag;
                        visited_nn_flag |= 1 << (j - 1);
                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                        comparison_count++;
                        // 更新候选集和顶部候选者
                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
                            top_candidates.emplace(dist, candidate_id);
                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            // 释放访问列表资源并返回顶部候选者集合
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, labeltype>> searchKnnWithOnlineDomination(
            const void *query_data, size_t k, BaseIndex::SearchInfo *search_info) const
        {
            // 初始化结果优先队列
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            size_t comparison_count = 0;
            // 如果当前元素计数为零，则直接返回空结果
            if (cur_element_count == 0)
                return result;

            // 设置起始点为入口节点
            tableint currObj = enterpoint_node_;
            // 计算查询数据到入口节点的距离
            dist_t curdist = fstdistfunc_(
                query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            comparison_count++;
            // 遍历从最大层至第一层
            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;

                // 当前层级有变化时循环
                while (changed)
                {
                    changed = false;

                    // 获取链接列表数据
                    unsigned int *data = (unsigned int *)get_linklist(currObj, level);
                    // 获取列表大小
                    int size = getListCount(data);
                    // 更新度量统计信息
                    metric_hops++;
                    metric_distance_computions += size;

                    // 解析链接列表中的节点
                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        // 获取候选节点
                        tableint cand = datal[i];

                        // 检查候选节点的有效性
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");

                        // 计算查询数据到候选节点的距离
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                                dist_func_param_);
                        comparison_count++;
                        // 如果新距离小于当前最小距离，则更新之
                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            // 创建顶点候选优先队列
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates;

            // 根据是否已删除状态选择搜索基底层的方法
            if (has_deletions_)
            {
                top_candidates =
                    searchBaseLayerWithOnlineDomination(currObj, query_data, std::max(ef_, k), comparison_count);
            }
            else
            {
                top_candidates =
                    searchBaseLayerWithOnlineDomination(currObj, query_data, std::max(ef_, k), comparison_count);
            }

            // 调整候选队列大小以满足k值要求
            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }

            // 将候选队列转换为结果队列
            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first,
                                                         getExternalLabel(rez.second)));
                top_candidates.pop();
            }

            search_info->total_comparison += comparison_count;

            // 返回最终结果
            return result;
        }

        bool printOnePointNN(labeltype target_external_id)
        {
            tableint cur_c;
            auto search = label_lookup_.find(target_external_id);
            cur_c = search->second;
            auto &relation = compact_graph->at(target_external_id);

            bool if_dominated = false;
            for (auto domination_flag : relation.domination_flags)
            {
                if (domination_flag != 0)
                    if_dominated = true;
            }
            if (if_dominated == false)
                return false;

            cout << "Domination Flags: ";
            for (auto domination_flag : relation.domination_flags)
            {
                cout << domination_flag << " ";
            }
            cout << endl
                 << "Neighbor lists: ";

            int *data = (int *)get_linklist0(cur_c);
            size_t size = getListCount((linklistsizeint *)data);
            // 遍历链接列表
            for (size_t j = 1; j <= size; j++)
            {
                auto candidate_id = *(data + j);
                cout << getExternalLabel(candidate_id) << " ";
            }
            cout << endl;
            return if_dominated;
        }
    };

    class IndexOnlineDominationGraph : public BaseIndex
    {
    public:
        vector<OnlineDominateRelation> indexed_arr;

        IndexOnlineDominationGraph(base_hnsw::SpaceInterface<float> *s,
                                   const DataWrapper *data)
            : BaseIndex(data)
        {
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            index_info = new IndexInfo();
            index_info->index_version_type = "IndexOnlineDominationGraph";
        }
        base_hnsw::DISTFUNC<float> fstdistfunc_;
        void *dist_func_param_;

        VisitedListPool *visited_list_pool_;
        IndexInfo *index_info;
        const BaseIndex::IndexParams *index_params_;
        OnlineDominationGraph<float> *hnsw;

        void buildIndex(const IndexParams *index_params) override
        {
            cout << "Building Index using " << index_info->index_version_type << endl;
            timeval tt1, tt2;
            visited_list_pool_ =
                new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

            index_params_ = index_params;
            // build HNSW
            L2Space space(data_wrapper->data_dim);
            hnsw = new OnlineDominationGraph<float>(
                *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
                index_params->ef_construction, index_params->random_seed);

            indexed_arr.clear();
            indexed_arr.resize(data_wrapper->data_size);
            hnsw->compact_graph = &indexed_arr;
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
                hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
            }

            // Step 4: Get Compressed Edges
            // for (size_t i : permutation)
            // {
            //     hnsw->getDominationRelationship(i);
            // }

            gettimeofday(&tt2, NULL);
            index_info->index_time = CountTime(tt1, tt2);

            // auto st = data_wrapper->data_size / 2;
            // bool flag = false;
            // while(!flag && st < data_wrapper->data_size){
            //     auto flag = hnsw->printOnePointNN(st);
            //     st++;
            // }

            // // count neighbors number
            // countNeighbrs();
        };

        vector<int> searchInFullRange(const SearchParams *search_params, SearchInfo *search_info,
                                      const vector<float> &query)
        {
            timeval tt1, tt2;
            vector<int> res;

            search_info->total_comparison = 0;
            search_info->internal_search_time = 0;
            search_info->cal_dist_time = 0;
            search_info->fetch_nns_time = 0;

            
            gettimeofday(&tt1, NULL);
            hnsw->internalLevel_cmp = 0;
            hnsw->baseLevel_cmp = 0;
            auto top_candidates = hnsw->searchKnn(query.data(), search_params->search_ef);
            // auto top_candidates = hnsw->searchBaseLayerST<false, true>(500, query.data(), search_params->search_ef);
            // auto top_candidates = hnsw->searchKnnWithOnlineDomination(query.data(), search_params->search_ef, search_info);
            gettimeofday(&tt2, NULL);                                    // 结束时间记录
            AccumulateTime(tt1, tt2, search_info->internal_search_time); // 累加邻居检索时间
            search_info->total_comparison = hnsw->internalLevel_cmp + hnsw->baseLevel_cmp;
            while (top_candidates.size() > search_params->query_K)
            {
                top_candidates.pop(); // 减少候选集至所需K个
            }

            while (!top_candidates.empty())
            {
                res.emplace_back(top_candidates.top().second); // 提取节点ID构建结果
                top_candidates.pop();
            }

            return res;
        }

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
            //             // 时间测量变量初始化
            //             timeval tt1, tt2, tt3, tt4;

            //             // 初始化访问列表
            //             VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            //             vl_type *visited_array = vl->mass;
            //             vl_type visited_array_tag = vl->curV;
            //             float lower_bound = std::numeric_limits<float>::max(); // 最低界限初始化为最大浮点数
            //             std::priority_queue<pair<float, int>> top_candidates;  // 优先队列存储候选结果
            //             std::priority_queue<pair<float, int>> candidate_set;   // 候选集优先队列

            //             // 数据大小
            //             const int data_size = data_wrapper->data_size;
            //             search_info->total_comparison = 0;
            //             search_info->internal_search_time = 0;
            //             search_info->cal_dist_time = 0;
            //             search_info->fetch_nns_time = 0;
            //             num_search_comparison = 0;

            //             // 初始化三个entry points
            //             {
            //                 int lbound = query_bound.first;
            //                 int interval = (query_bound.second - lbound) / 3;
            //                 for (size_t i = 0; i < 3; i++)
            //                 {
            //                     int point = lbound + interval * i;
            //                     float dist = EuclideanDistance(data_wrapper->nodes[point], query); // 计算距离
            //                     candidate_set.push(make_pair(-dist, point));                       // 将负距离和点ID推入候选集
            //                     visited_array[point] = visited_array_tag;                          // 标记已访问
            //                 }
            //             }
            //             gettimeofday(&tt3, NULL);

            //             // only one center
            //             // float dist_enter = EuclideanDistance(data_nodes[l_bound], query);
            //             // candidate_set.push(make_pair(-dist_enter, l_bound));
            //             // TODO: How to find proper enters. // looks like useless

            //             size_t hop_counter = 0;
            //             while (!candidate_set.empty())
            //             {
            //                 std::pair<float, int> current_node_pair = candidate_set.top(); // 获取当前节点
            //                 int current_node_id = current_node_pair.second;

            //                 if (-current_node_pair.first > lower_bound) // 如果当前节点的距离大于topk里最远的，则跳出循环
            //                 {
            //                     break;
            //                 }

            // #ifdef LOG_DEBUG_MODE
            //                 cout << "current node: " << current_node_pair.second << "  -- "
            //                      << -current_node_pair.first << endl;
            // #endif

            //                 // if (search_info->is_investigate) {
            //                 //   search_info->SavePathInvestigate(current_node_pair.second,
            //                 //                                    -current_node_pair.first,
            //                 //                                    hop_counter, num_search_comparison);
            //                 // }
            //                 hop_counter++;

            //                 candidate_set.pop();

            //                 // // only search when candidate point is inside the range
            //                 // this can be commented because no way to do this
            //                 // if (current_node_id < query_bound.first || current_node_id > query_bound.second)
            //                 // {
            //                 //     cout << "no satisfied range point" << endl;
            //                 //     continue;
            //                 // }

            //                 // search cw on the fly
            //                 vector<const vector<int> *> neighbor_iterators;

            //                 gettimeofday(&tt1, NULL);
            //                 {
            //                     // TODO 这里可以改一下 就是不是有overlap的我就拿出来 而是看overlap的那些点 直到有k个我就不走了
            //                     decompressForwardPath(neighbor_iterators,
            //                                           indexed_arr[current_node_id].forward_nns,
            //                                           query_bound.first, query_bound.second);
            //                 }
            //                 gettimeofday(&tt2, NULL);                              // 结束时间记录
            //                 AccumulateTime(tt1, tt2, search_info->fetch_nns_time); // 累加邻居检索时间

            //                 // 处理邻居集合
            //                 gettimeofday(&tt1, NULL); // 开始时间记录
            //                 unsigned cnt_positive_through_neighbors = 0;
            //                 const auto Mcurmax = 2 * index_params_->K; // 也需要看看效果 不过现在反向边和正向边放在一起，当然得扩大成2K // 也可以试试不加这个条件，理论上应该只会提升时间和recall 不会降低recall和time吧
            //                 for (auto batch_it : neighbor_iterators)
            //                 {
            //                     for (auto candidate_id : *batch_it)
            //                     {
            //                         if (candidate_id < query_bound.first || candidate_id > query_bound.second) // 忽略越界节点
            //                             continue;

            //                         if (cnt_positive_through_neighbors < Mcurmax)
            //                             cnt_positive_through_neighbors++;
            //                         else
            //                             break;

            //                         if (!(visited_array[candidate_id] == visited_array_tag)) // 若未被访问过
            //                         {
            //                             visited_array[candidate_id] = visited_array_tag; // 标记为已访问

            //                             // 计算距离
            //                             float dist = fstdistfunc_(query.data(),
            //                                                       data_wrapper->nodes[candidate_id].data(),
            //                                                       dist_func_param_);

            // #ifdef LOG_DEBUG_MODE
            //                             // 输出调试信息
            // #endif

            //                             num_search_comparison++; // 更新比较次数
            //                             if (top_candidates.size() < search_params->search_ef || lower_bound > dist)
            //                             {
            //                                 candidate_set.push(make_pair(-dist, candidate_id)); // 推入候选集
            //                                 top_candidates.push(make_pair(dist, candidate_id)); // 推入顶级候选集
            //                                 if (top_candidates.size() > search_params->search_ef)
            //                                 {
            //                                     top_candidates.pop(); // 维护候选集大小
            //                                 }
            //                                 if (!top_candidates.empty())
            //                                 {
            //                                     lower_bound = top_candidates.top().first; // 更新最低界限
            //                                 }
            //                             }
            //                         }
            //                     }
            //                 }
            //                 gettimeofday(&tt2, NULL);                             // 结束时间记录
            //                 AccumulateTime(tt1, tt2, search_info->cal_dist_time); // 累加距离计算时间
            //             }

            //             // 构建结果列表
            //             vector<int> res;
            //             while (top_candidates.size() > search_params->query_K)
            //             {
            //                 top_candidates.pop(); // 减少候选集至所需K个
            //             }

            //             while (!top_candidates.empty())
            //             {
            //                 res.emplace_back(top_candidates.top().second); // 提取节点ID构建结果
            //                 top_candidates.pop();
            //             }
            //             search_info->total_comparison += num_search_comparison; // 更新总比较次数

            // #ifdef LOG_DEBUG_MODE
            //             print_set(res);
            //             cout << l_bound << "," << r_bound << endl;
            //             assert(false);
            // #endif

            //             // 释放资源和更新时间统计
            //             visited_list_pool_->releaseVisitedList(vl);
            //             gettimeofday(&tt4, NULL);
            //             CountTime(tt3, tt4, search_info->internal_search_time);
            // return res; // 返回结果列表
            return vector<int>();
        }

        vector<int> rangeFilteringSearchOutBound(
            const SearchParams *search_params, SearchInfo *search_info,
            const vector<float> &query,
            const std::pair<int, int> query_bound) override
        {
            return vector<int>();
        }
        ~IndexOnlineDominationGraph()
        {
            delete index_info;
            indexed_arr.clear();
            delete visited_list_pool_;
        }

        void saveIndex(const string& path){
            hnsw->saveIndex(path);
        }

        void loadIndex(const BaseIndex::IndexParams *index_params,
                              SpaceInterface<float> *s, const std::string & location){
            hnsw = new OnlineDominationGraph<float>(*index_params, s, location);
        }
    };
} // namespace SeRF