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

namespace Compact
{

    // 定义BatchNeighbors结构体，用于存储批量邻居信息
    struct BatchNeighbors
    {
        // 使用初始化列表构造函数，简化赋值操作并提高效率
        BatchNeighbors(unsigned num, int left_lower = -1, int left_upper = -1,
                       int right_lower = -1, int right_upper = -1)
            : batch(num),
              left_range{left_lower, left_upper},
              right_range{right_lower, right_upper} {}

        // 默认构造函数
        BatchNeighbors() : BatchNeighbors(0) {}

        // 存储邻居ID的向量
        std::vector<int> nns_id;

        // 批次ID
        unsigned batch;

        // 左右范围定义，使用std::pair代替单个int变量，提高代码清晰度
        std::pair<int, int> left_range;
        std::pair<int, int> right_range;

        unsigned size() const { return nns_id.size(); }

        bool if_outrange_equal(const pair<int, int> &outrange)
        {
            return outrange.first == left_range.first && outrange.second == right_range.second;
        }

        /**
         * @brief 计算以Pivot ID的内部范围以及更新外部范围
         *
         * 此函数计算给定的pivotID相对于其他点的最小和最大的外部ID，
         * 并更新整个范围内的左边界和右边界值。
         *
         * @param target_external_id 目标点的外部ID
         */
        void cal_innerrange(int target_external_id)
        {
            int tmp_min = std::numeric_limits<int>::max(); // 初始化临时最小值为int的最大值
            int tmp_max = -1;                              // 初始化临时最大值为-1

            left_range.second = target_external_id; // 设置左范围的上限为目标ID
            right_range.first = target_external_id; // 设置右范围的下限为目标ID

            for (auto &point_external_id : nns_id)
            { // 遍历所有邻居节点的外部ID
                if (point_external_id > target_external_id)
                    tmp_min = std::min(tmp_min, point_external_id); // 更新临时最小值

                if (point_external_id < target_external_id)
                    tmp_max = std::max(tmp_max, point_external_id); // 更新临时最大值

                // 更新外部范围
                right_range.second = std::max(right_range.second, point_external_id);
                left_range.first = std::min(left_range.first, point_external_id);
            }

            if (tmp_min != std::numeric_limits<int>::max()) // 如果有比目标大的ID，则设置右范围的下限
                right_range.first = tmp_min;

            if (tmp_max != -1) // 如果有比目标小的ID，则设置左范围的上限
                left_range.second = tmp_max;
        }

        bool judge_if_in_innerrange(int pivot_external_id, int point_external_id)
        {
            int left_bound = left_range.second == pivot_external_id ? left_range.first : left_range.second;
            int right_bound = right_range.first == pivot_external_id ? right_range.second : right_range.first;
            return point_external_id >= left_bound && point_external_id <= right_bound;
        }

        bool judge_if_in_outrange(int point_external_id){
            return point_external_id >= left_range.first && point_external_id <= right_range.second;
        }
    };

    struct DirectedBatchNeighbors
    {
        vector<BatchNeighbors> forward_nns;

        int countNeighborOfOneBatch()
        {
            int temp_size = 0;
            // 累加每个节点的前向邻居数量
            for (const auto &nns : forward_nns)
            {
                temp_size += nns.nns_id.size();
            }
            return temp_size;
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

        int max_external_id_ = -1;
        int min_external_id_ = std::numeric_limits<int>::max();

        // log
        size_t forward_batch_nn_amount = 0;
        size_t backward_batch_theoratical_nn_amount = 0;
        size_t drop_points_ = 0;

        // 指向BaseIndex::IndexParams类型的常量指针，存储索引参数
        const BaseIndex::IndexParams *params;

        // 存储指向段图邻居列表的指针，表示图结构中的边信息
        vector<DirectedBatchNeighbors> *compact_graph;

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
        void addNegativeEdges(const dist_t cur_dist, int cur_external_id, int target_external_id)
        {
            // if (target_external_id == 72974)
            //     cout << "hi";
            auto &target_batches = compact_graph->at(target_external_id).forward_nns;
            const int pre_nns_amount = compact_graph->at(target_external_id).countNeighborOfOneBatch();

            unsigned batch_size = target_batches.size();
            unsigned cnt = 0;
            pair<int, int> outrange = {min_external_id_, max_external_id_};
            vector<pair<int, dist_t>> to_insert_points = {{cur_external_id, cur_dist}};
            bool if_outrange_equal;
            while (cnt < batch_size)
            {
                auto &batch = target_batches[cnt];
                if_outrange_equal = batch.if_outrange_equal(outrange);
                if (if_outrange_equal && to_insert_points.size() == 0)
                    break;

                update_batch(outrange, batch, target_external_id, to_insert_points, if_outrange_equal);
                pair<int, int> cur_inner_range = {batch.left_range.second, batch.right_range.first};

                // TODO: Can it moved inside "If"
                // if updated means next outrange need to be aligned with current inner range
                if (cur_inner_range.first != target_external_id)
                    outrange.first = cur_inner_range.first + 1;
                if (cur_inner_range.second != target_external_id)
                    outrange.second = cur_inner_range.second - 1;
                cnt++;
            }

            // If still left points, create a new batch
            if (to_insert_points.size() > 0)
            {
                target_batches.emplace_back(batch_size, outrange.first, target_external_id, target_external_id, outrange.second);
                for (auto &point : to_insert_points)
                {
                    target_batches[batch_size].nns_id.emplace_back(point.first);
                }
                target_batches[batch_size].cal_innerrange(target_external_id);
            }

            const int now_nns_amount = compact_graph->at(target_external_id).countNeighborOfOneBatch();
            // assert(pre_nns_amount == now_nns_amount - 1);
        }

        void update_batch(pair<int, int> &outrange, BatchNeighbors &batch, int target_external_id, vector<pair<int, dist_t>> &to_insert_points, bool if_outrange_equal)
        {
            // Check whether need to update the outrange
            if (if_outrange_equal == false)
            {
                // Todo: Need to debate whether need to shrink the outrange to prune some points
                // batch.left_range.first = outrange.first;
                // batch.right_range.second = outrange.second;

                batch.left_range.first = std::min(outrange.first, batch.left_range.first);
                batch.right_range.second = std::max(outrange.second, batch.right_range.second);
            }

            // TODO: Maybe the passed cur_range is a larger range and curerent top_k_indices is not up to k elements (the last layer)
            // We can just push the outer elements into the top_k_indices (But it is not important to update the minimum range)

            std::priority_queue<std::pair<dist_t, int>> max_dist_heap;
            for (auto &nn_id : batch.nns_id)
            {
                if (nn_id >= batch.left_range.first && nn_id <= batch.right_range.second)
                {
                    dist_t cur_dist = fstdistfunc_(getDataByLabel((size_t)target_external_id), getDataByLabel((size_t)nn_id), dist_func_param_);
                    max_dist_heap.emplace(cur_dist, nn_id);
                }
            }
            size_t Mcurmax = 2 * maxM0_;

            // if it already not full, not need to check the distance
            auto max_dist = max_dist_heap.size() >= Mcurmax ? max_dist_heap.top().first : std::numeric_limits<dist_t>::max();
            vector<pair<int, dist_t>> to_pass_points;
            vector<pair<int, dist_t>> dropped_points;
            for (auto &point : to_insert_points)
            {
                if (point.second > max_dist)
                // if the point distance is too far, just drop it, except it is not in current compressed range
                {
                    if (batch.judge_if_in_innerrange(target_external_id, point.first))
                    {
                        to_pass_points.emplace_back(point);
                    }
                }
                else
                {
                    max_dist_heap.emplace(point.second, point.first);
                }
            }

            // check whether we need pruning? try not pruning
            if (max_dist_heap.size() > Mcurmax)
            {
                // We should just pop the maxheap
                // we need to prune first
                // while (max_dist_heap.size() > Mcurmax)
                // {
                //     auto [distance, point_index] = max_dist_heap.top(); // 解构赋值以提高可读性
                //     max_dist_heap.pop();
                //     dropped_points.emplace_back(point_index, distance);
                // }

                // Start pruning
                vector<std::pair<dist_t, int>> return_list;
                std::priority_queue<std::pair<dist_t, int>> queue_closest;
                while (max_dist_heap.size() > 0)
                {
                    queue_closest.emplace(-max_dist_heap.top().first,
                                          max_dist_heap.top().second);
                    max_dist_heap.pop();
                }
                while (queue_closest.size())
                {
                    // Reach limit just break
                    if (return_list.size() >= Mcurmax)
                        break;

                    // 获取当前队列顶部的元素
                    std::pair<dist_t, int> current_pair = queue_closest.top();

                    // 计算距离到查询点的实际值
                    dist_t dist_to_query = -current_pair.first;

                    // 移除队列顶部元素
                    queue_closest.pop();

                    // 判断当前元素是否满足条件
                    bool good = true;

                    // 检查当前元素与其他已选元素之间的距离
                    for (std::pair<dist_t, int> second_pair : return_list)
                    {
                        dist_t curdist = fstdistfunc_(
                            getDataByLabel(second_pair.second),
                            getDataByLabel(current_pair.second),
                            dist_func_param_);

                        // 如果发现更近的点，则标记为不满足条件
                        if (curdist < dist_to_query)
                        {
                            good = false;
                            // try try this will lead more points into the next batch
                            // if dominated point is dominated by a point outer than it, then we still need it if the query range is "inner"
                            // auto cur_external_id = current_pair.second;
                            // auto dominate_external_id = second_pair.second;
                            // if((cur_external_id > target_external_id && dominate_external_id > cur_external_id) || (cur_external_id < target_external_id && dominate_external_id < cur_external_id) )
                            dropped_points.emplace_back(current_pair.second, dist_to_query);
                            break;
                        }
                    }
                    if (good)
                    {
                        return_list.push_back(current_pair);
                    }
                }

                // push the rest points in queue_closest into dropped_points
                while (queue_closest.size())
                {
                    std::pair<dist_t, int> current_pair = queue_closest.top();
                    // remember in the queue closest the distance is negative
                    dropped_points.emplace_back(current_pair.second, -current_pair.first);
                    queue_closest.pop();
                }

                // update batch nnds_id
                batch.nns_id.clear();
                for (auto &point : return_list)
                {
                    batch.nns_id.emplace_back(point.second);
                }
                batch.cal_innerrange(target_external_id);
            }
            else
            {
                // not need to do prune
                // just push the maxheap list into the batch.nns_id
                batch.nns_id.clear();
                while (max_dist_heap.size() > 0)
                {
                    batch.nns_id.emplace_back(max_dist_heap.top().second);
                    max_dist_heap.pop();
                }
                batch.cal_innerrange(target_external_id);
            }

            // update outrange
            if (batch.left_range.second != target_external_id)
            {
                outrange.first = batch.left_range.second + 1;
            }
            if (batch.right_range.first != target_external_id)
            {
                outrange.second = batch.right_range.first - 1;
            }

            // try to put qualified points into the to_pass_point
            for (auto &point : dropped_points)
            {
                if (batch.judge_if_in_innerrange(target_external_id, point.first))
                    // if (point.first > batch.left_range.second and point.first < batch.right_range.first)
                    to_pass_points.emplace_back(point);
            }
            to_insert_points.swap(to_pass_points);
        }
        
        void addRangeEdges(const void *data_point, labeltype label){
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            auto currObj = label_c;
            auto top_candidates = searchBaseLayerLevel0(currObj, data_point, 0);
            getRangeEdges(data_point, label_c, top_candidates, 0, 0);
        }
        /**
         * @file src/compact_graph.h
         * @brief 互连新元素并递归地应用启发式剪枝算法并且对插入元素的attribute没有要求
         *
         * 此函数用于连接新的数据点到图中的现有节点，
         * 并通过优先队列处理候选邻居以优化连接过程。
         */

        void getRangeEdges(
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
            int external_id = getExternalLabel(cur_c);
            tableint next_closest_entry_point = 0; // 下一个最近入口点

            // 更新目前所有节点的最大和最小外部标签值
            if (external_id > max_external_id_)
                max_external_id_ = external_id;
            if (external_id < min_external_id_)
                min_external_id_ = external_id;

            // 邻居选择容器初始化 这个邻居是对整个图的邻居 也就是说对整个图他本身整个range的信息他有 但是sub range 的信息我是自己额外存的
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);

            {
                // MAX_POS recursive pruning, combining into connect function.

                // 处理优先级队列：从最近到最远排序候选者
                std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
                while (!top_candidates.empty())
                {
                    queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                    top_candidates.pop();
                }
                // Now top_candidates is empty

                // 初始化变量
                pair<int, int> external_lr_most = {min_external_id_, max_external_id_}; // keeps track of the furthest (bidirectional) external ID encountered
                int l_rightmost = external_id, r_leftmost = external_id;
                int tmp_min = std::numeric_limits<int>::max();
                int tmp_max = -1;
                unsigned iter_counter = 0;
                unsigned batch_counter = 0;
                std::vector<std::pair<dist_t, int>> return_list;
                std::vector<int> return_external_list;

                // Need a buffer candidates to store neighbors (external_id, dist_t,
                // internal_id)
                vector<pair<int, pair<dist_t, tableint>>> buffer_candidates;

                // 主循环处理候选者队列直到为空
                while (!queue_closest.empty())
                {
                    // If return list size meet M or current window exceed ef_construction, end this batch, enter the new batch
                    if (return_list.size() >= Mcurmax ||
                        iter_counter >= ef_basic_construction_)
                    {
                        // get l_rightmost r_leftmost if tmp_min or tmp_max has been triggerred, update r_leftmost or l_rightmost
                        if (tmp_min != std::numeric_limits<int>::max())
                            r_leftmost = tmp_min;
                        if (tmp_max != -1)
                            l_rightmost = tmp_max;

                        // reset batch, add current batch;
                        // no breaking because recursivly visiting the candidates.

                        for (pair<int, pair<dist_t, tableint>> curent_buffer :
                             buffer_candidates)
                        {
                            // 这些被pruned 掉的如果是 最里面的话 就要留着
                            if ((curent_buffer.first <= external_id && curent_buffer.first > tmp_max) || (curent_buffer.first >= external_id && curent_buffer.first < tmp_min))
                            {
                                // available in next batch, add back to the queue.
                                queue_closest.emplace(curent_buffer.second);
                            }
                        }

                        // current batch start position: [[last left + 1,current_leftmost], [current_rightmost, last right - 1 ]
                        BatchNeighbors one_batch(
                            batch_counter, external_lr_most.first, l_rightmost, r_leftmost, external_lr_most.second);

                        // only keep id, drop dists
                        forward_batch_nn_amount += return_external_list.size();
                        one_batch.nns_id.swap(return_external_list);
                        compact_graph->at(external_id).forward_nns.emplace_back(one_batch);
                        // add reverse edge
                        for (auto i = 0; i < one_batch.nns_id.size(); i++)
                        {
                            auto point_dist = -return_list[i].first;
                            auto target_external_id = one_batch.nns_id[i];
                            addNegativeEdges(point_dist, external_id, target_external_id);
                        }

                        backward_batch_theoratical_nn_amount += return_list.size();

                        return_list.clear(); // 这里会清空return list
                        return_external_list.clear();
                        iter_counter = 0;
                        batch_counter++;

                        if (l_rightmost != external_id)
                        {
                            external_lr_most.first = l_rightmost + 1;
                        }
                        if (r_leftmost != external_id)
                        {
                            external_lr_most.second = r_leftmost - 1;
                        }

                        // re initiliaze
                        l_rightmost = external_id;
                        r_leftmost = external_id;
                        tmp_min = std::numeric_limits<int>::max();
                        tmp_max = -1;
                    }

                    std::pair<dist_t, tableint> curent_pair = queue_closest.top(); // 当前离我最近的点
                    dist_t dist_to_query = -curent_pair.first;
                    queue_closest.pop();

                    int current_external_id = getExternalLabel(curent_pair.second);
                    if (current_external_id < external_lr_most.first || current_external_id > external_lr_most.second) // 因为是maxleap 所以我的batch之间没有重
                    {
                        // position in the previous batch, skip
                        continue;
                    }
                    iter_counter++;
                    bool good = true;

                    // 查看会不会被在return list的给prune掉
                    for (std::pair<dist_t, int> second_pair : return_list)
                    {
                        dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                                      getDataByInternalId(curent_pair.second),
                                                      dist_func_param_);

                        if (curdist < dist_to_query)
                        {
                            good = false;
                            break;
                        }
                    }

                    if (good)
                    {
                        return_list.emplace_back(curent_pair.first, (int)curent_pair.second);
                        return_external_list.push_back(current_external_id);
                        // 不断更新当前batch的inner range
                        if (current_external_id <= external_id)
                            tmp_max = std::max(tmp_max, (int)current_external_id);
                        if (current_external_id >= external_id)
                            tmp_min = std::min(tmp_min, (int)current_external_id);
                    }
                    else
                    {
                        if ((current_external_id <= external_id && current_external_id > tmp_max) || (current_external_id >= external_id && current_external_id < tmp_min))
                        {
                            buffer_candidates.emplace_back(
                                make_pair(current_external_id, curent_pair));
                        }
                    }
                }

                

                if (!return_list.empty() && external_lr_most.first <= external_id && external_id <= external_lr_most.second) // 这里肯定会进的 因为最后还是batch没打满 离插入点的attribute维度上最近的点们，很可能凑不齐M或者ef construction个 然后这里也得进OneSegmentNeighbors
                {
                    BatchNeighbors one_batch(
                        batch_counter, external_lr_most.first, external_id, external_id, external_lr_most.second);
                    forward_batch_nn_amount += return_external_list.size();
                    one_batch.nns_id.swap(return_external_list);
                    compact_graph->at(external_id).forward_nns.emplace_back(one_batch);

                    // // add reverse edges
                    for (auto i = 0; i < one_batch.nns_id.size(); i++)
                    {
                        auto point_dist = -return_list[i].first;
                        auto target_external_id = one_batch.nns_id[i];
                        addNegativeEdges(point_dist, external_id, target_external_id);
                    }
                    backward_batch_theoratical_nn_amount += return_list.size();
                }
            }
        }
    };

    class IndexCompactGraph : public BaseIndex
    {
    public:
        vector<DirectedBatchNeighbors> directed_indexed_arr;

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
            for (auto nns :
                 directed_indexed_arr[data_wrapper->data_size / 2].forward_nns)
            {
                cout << "Forward batch: " << nns.batch << "[(" << nns.left_range.first << "," << nns.left_range.second << ","
                     << nns.right_range.first << "," << nns.right_range.second << ")]" << endl;
                print_set(nns.nns_id);
                cout << endl;
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
            // 初始化计数器
            double batch_counter = 0;
            double max_batch_counter = 0;
            size_t max_reverse_nn = 0;

            // 如果有向图索引不为空，则开始处理
            if (!directed_indexed_arr.empty())
            {
                // 遍历所有节点的前向邻居列表
                for (unsigned j = 0; j < directed_indexed_arr.size(); j++)
                {
                    int temp_size = 0;
                    // 累加每个节点的前向邻居数量
                    for (const auto &nns : directed_indexed_arr[j].forward_nns)
                    {
                        temp_size += nns.nns_id.size();
                    }
                    // 更新总前向邻居数量和批处理计数
                    batch_counter += directed_indexed_arr[j].forward_nns.size();
                    index_info->nodes_amount += temp_size;
                }
            }

            // 计算平均前向邻居数
            index_info->avg_forward_nns = index_info->nodes_amount / static_cast<float>(data_wrapper->data_size);

            // 打印日志（如果启用）
            if (isLog)
            {

                cout << "Sum of forward batch nn #: " << index_info->nodes_amount << endl;
                cout << "Max. forward batch nn #: " << max_batch_counter << endl;
                cout << "Avg. forward nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
                cout << "Avg. forward batch #: " << batch_counter / static_cast<float>(data_wrapper->data_size) << endl;
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

            // multi-thread also work, but not guaranteed as the paper
            // may has minor recall decrement
            // #pragma omp parallel for schedule(monotonic : dynamic)
            // for (size_t i = 0; i < data_wrapper->data_size; ++i)
            // {
            //     hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
            // }

            // random add points
            // Step 1: Generate a sequence 0, 1, ..., data_size - 1
            std::vector<size_t> permutation(data_wrapper->data_size);
            std::iota(permutation.begin(), permutation.end(), 0);

            // Step 2: Shuffle the sequence
            std::random_device rd;    // obtain a random number from hardware
            unsigned int seed = 2024; // fix the seed for debug
            // std::mt19937 g(rd());
            std::mt19937 g(seed);     // seed the generator
            std::shuffle(permutation.begin(), permutation.end(), g);

            // Step 3: Traverse the shuffled sequence

            cout << "First point" << permutation[0] << endl;
            for (size_t i : permutation)
            {
                hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
            }

            // Step 4: Get Compressed Edges

            for (size_t i : permutation)
            {
                hnsw.addRangeEdges(data_wrapper->nodes.at(i).data(), i);
            }

            gettimeofday(&tt2, NULL);
            index_info->index_time = CountTime(tt1, tt2);

            cout << "All the forward batch nn #: " << hnsw.forward_batch_nn_amount << endl;
            cout << "Theoratical backward batch nn #: " << hnsw.backward_batch_theoratical_nn_amount << endl;

            // count neighbors number
            countNeighbrs();

            if (index_params->print_one_batch)
            {
                printOnebatch();
            }
        };

        void decompressForwardPath(vector<const vector<int> *> &neighbor_iterators,
                                   const vector<BatchNeighbors> &forward_nns, const int lbound, const int rbound)
        {
            // forward iterator
            auto forward_batch_it = forward_nns.begin();
            while (forward_batch_it != forward_nns.end())
            {
                if ((forward_batch_it->left_range.first <= lbound && lbound <= forward_batch_it->left_range.second) || (forward_batch_it->right_range.first <= rbound && rbound <= forward_batch_it->right_range.second))
                {
                    neighbor_iterators.emplace_back(&forward_batch_it->nns_id);
                }
                forward_batch_it++;
            }
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
            // 时间测量变量初始化
            timeval tt1, tt2, tt3, tt4;

            // 初始化访问列表
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            float lower_bound = std::numeric_limits<float>::max(); // 最低界限初始化为最大浮点数
            std::priority_queue<pair<float, int>> top_candidates;  // 优先队列存储候选结果
            std::priority_queue<pair<float, int>> candidate_set;   // 候选集优先队列

            // 数据大小
            const int data_size = data_wrapper->data_size;
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
                // if (current_node_id < query_bound.first || current_node_id > query_bound.second)
                // {
                //     cout << "no satisfied range point" << endl;
                //     continue;
                // }

                // search cw on the fly
                vector<const vector<int> *> neighbor_iterators;

                gettimeofday(&tt1, NULL);
                {
                    // TODO 这里可以改一下 就是不是有overlap的我就拿出来 而是看overlap的那些点 直到有k个我就不走了
                    decompressForwardPath(neighbor_iterators,
                                          directed_indexed_arr[current_node_id].forward_nns,
                                          query_bound.first, query_bound.second);
                }
                gettimeofday(&tt2, NULL);                              // 结束时间记录
                AccumulateTime(tt1, tt2, search_info->fetch_nns_time); // 累加邻居检索时间

                // 处理邻居集合
                gettimeofday(&tt1, NULL); // 开始时间记录
                unsigned cnt_positive_through_neighbors = 0;
                const auto Mcurmax = 2 * index_params_->K; // 也需要看看效果 不过现在反向边和正向边放在一起，当然得扩大成2K // 也可以试试不加这个条件，理论上应该只会提升时间和recall 不会降低recall和time吧
                for (auto batch_it : neighbor_iterators)
                {
                    for (auto candidate_id : *batch_it)
                    {
                        if (candidate_id < query_bound.first || candidate_id > query_bound.second) // 忽略越界节点
                            continue;

                        if (cnt_positive_through_neighbors < Mcurmax)
                            cnt_positive_through_neighbors++;
                        else
                            break;

                        if (!(visited_array[candidate_id] == visited_array_tag)) // 若未被访问过
                        {
                            visited_array[candidate_id] = visited_array_tag; // 标记为已访问

                            // 计算距离
                            float dist = fstdistfunc_(query.data(),
                                                      data_wrapper->nodes[candidate_id].data(),
                                                      dist_func_param_);

#ifdef LOG_DEBUG_MODE
                            // 输出调试信息
#endif

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