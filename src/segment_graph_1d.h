/**
 * @file segment_graph_1d.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for half-bounded range filtering search.
 * Lossless compression on N hnsw on search space
 * @date 2023-06-29; Revised 2023-12-29
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "utils.h"

using namespace base_hnsw;
// #define INT_MAX __INT_MAX__

namespace SeRF
{

    /**
     * @brief segment neighbor structure to store segment graph edge information
     * if id==end_id, means haven't got pruned
     * id: neighbor id
     * dist: neighbor dist
     * end_id: when got pruned
     */
    template <typename dist_t>
    struct SegmentNeighbor1D
    {
        SegmentNeighbor1D(int id) : id(id){};
        SegmentNeighbor1D(int id, dist_t dist, int end_id)
            : id(id), dist(dist), end_id(end_id){};
        int id;
        dist_t dist;
        int end_id;
    };

    // Inherit from basic HNSW, modify the 'heuristic pruning' procedure to record
    // the lifecycle for SegmentGraph
    /**
     * @brief 模板类SegmentGraph1DHNSW实现一维段图HNSW索引结构。
     *
     * 此类继承自HierarchicalNSW，用于构建基于距离度量的一维空间数据的层次邻近搜索树。
     *
     * @tparam dist_t 距离度量类型。
     */
    template <typename dist_t>
    class SegmentGraph1DHNSW : public HierarchicalNSW<float>
    {
    public:
        /**
         * @brief 构造函数初始化SegmentGraph1DHNSW对象。
         *
         * 根据给定参数创建一维段图HNSW索引结构实例。
         *
         * @param index_params 索引参数配置。
         * @param s 空间接口指针，提供距离计算方法。
         * @param max_elements 最大元素数量。
         * @param M 初始最大连接数。
         * @param ef_construction 构建过程中的查询效率因子。
         * @param random_seed 随机种子值。
         */
        SegmentGraph1DHNSW(const BaseIndex::IndexParams &index_params,
                           SpaceInterface<dist_t> *s, size_t max_elements,
                           size_t M = 16, size_t ef_construction = 200,
                           size_t random_seed = 100)
            : HierarchicalNSW(s, max_elements, M, index_params.ef_construction,
                              random_seed)
        {
            params = &index_params;
            // 在单边段图中，ef_max_等于ef_construction
            ef_max_ = index_params.ef_construction;
            ef_basic_construction_ = index_params.ef_construction;
            ef_construction = index_params.ef_construction;
        }

        // 存储索引参数的指针
        const BaseIndex::IndexParams *params;

        // 索引存储结构
        vector<vector<SegmentNeighbor1D<dist_t>>> *range_nns;

        /**
         * @brief 使用启发式算法获取并记录剪枝后的邻居列表。
         *
         * 根据优先队列中的候选者，筛选出满足条件的最近邻居，并将不符合条件但可能有用的邻居信息存入范围邻居列表。
         *
         * @param top_candidates 包含候选者的优先队列。
         * @param M 候选者数量上限。
         * @param back_nns 记录被剪枝的邻居列表。
         * @param end_pos_id 结束位置标识符。
         */
        void getNeighborsByHeuristic2RecordPruned(
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> &top_candidates,
            const size_t M, vector<SegmentNeighbor1D<dist_t>> *back_nns,
            const int end_pos_id)
        {
            /**
             * 如果候选列表大小小于M，则直接返回。
             */
            if (top_candidates.size() < M)
            {
                return;
            }

            /**
             * 使用优先队列存储最近点信息。
             * @param dist_t 距离类型
             * @param tableint 表格索引类型
             */
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;

            /**
             * 将候选列表中的元素转移到优先队列中，
             * 元素距离取反以实现最大堆效果。
             */
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first,
                                      top_candidates.top().second);
                top_candidates.pop();
            }

            /**
             * 处理优先队列直到找到M个最近邻或者队列为空。
             */
            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;

                /**
                 * 获取当前最近点对并恢复距离值。
                 */
                std::pair<dist_t, tableint> current_pair = queue_closest.top();
                dist_t dist_to_query = -current_pair.first;
                queue_closest.pop();

                bool good = true;

                /**
                 * 检查远的邻居离中心点的距离是否比更近的邻居近 是的话就prune掉他。
                 */
                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                                  getDataByInternalId(current_pair.second),
                                                  dist_func_param_);

                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }

                /**
                 * 如果满足条件，将点加入结果列表。
                 */
                if (good)
                {
                    return_list.push_back(current_pair);
                }
                else
                {
                    /**
                     * 记录被剪枝的近邻点，存入back_nns。
                     */
                    int external_nn = this->getExternalLabel(current_pair.second);
                    if (external_nn != end_pos_id)
                    {
                        SegmentNeighbor1D<dist_t> pruned_nn(external_nn, dist_to_query,
                                                            end_pos_id);
                        back_nns->emplace_back(pruned_nn);
                    }
                }
            }

            /**
             * 添加未访问过的近邻点到back_nns。 ？？？ 还是不理解？？？
             */
            while (queue_closest.size())
            {
                std::pair<dist_t, tableint> current_pair = queue_closest.top();
                int external_nn = this->getExternalLabel(current_pair.second);
                queue_closest.pop();

                if (external_nn != end_pos_id)
                {
                    SegmentNeighbor1D<dist_t> pruned_nn(external_nn, -current_pair.first,
                                                        end_pos_id);
                    back_nns->emplace_back(pruned_nn);
                }
            }

            /**
             * 更新top_candidates为最终的pruned过的最近邻。
             */
            for (std::pair<dist_t, tableint> current_pair : return_list)
            {
                top_candidates.emplace(-current_pair.first, current_pair.second);
            }
        }

        // since the order is important, SeRF use the external_id rather than the
        // inernal_id, but right now SeRF only supports building in one thread, so
        // acutally current external_id is equal to internal_id(cur_c).
        virtual tableint mutuallyConnectNewElementLevel0(
            const void *data_point, tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> &top_candidates,
            int level, bool isUpdate)
        {
            /**
             * @brief 更新当前元素与其最近邻的关系列表。
             *
             * 此方法用于更新数据结构中的链接列表，以反映当前元素（cur_c）与其他选定邻居之间的关系。
             * 它首先通过启发式方法获取候选邻居，然后遍历这些邻居并建立双向连接，
             * 确保每个邻居的链接列表都包含当前元素，除非它已经在列表中。
             *
             * @param top_candidates 优先队列，包含按距离排序的候选邻居。
             * @param cur_c 当前要处理的元素。
             * @param level 要操作的数据结构层级。
             * @param isUpdate 标志是否为更新操作。
             * @return tableint 返回下一个最接近的入口点。
             */

            // 获取最大允许的邻居数量
            size_t Mcurmax = this->maxM0_;

            // 使用启发式方法获取候选邻居
            getNeighborsByHeuristic2(top_candidates, this->M_);

            // 检查返回的候选者数量不应超过预设的最大值
            if (top_candidates.size() > this->M_)
                throw std::runtime_error("候选者的数量应不超过M_");

            // 获取当前元素的外部标签
            int external_id = this->getExternalLabel(cur_c);

            // 初始化存储选定邻居的向量
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(this->M_);

            // 将优先队列中的邻居转移到选定邻居向量中
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            // 设置下一个最接近的入口点
            tableint next_closest_entry_point = selectedNeighbors.back();

            // 更新当前元素的链接列表
            {
                linklistsizeint *ll_cur;

                // 根据层级选择正确的链接列表
                if (level == 0)
                    ll_cur = this->get_linklist0(cur_c);
                else
                    ll_cur = this->get_linklist(cur_c, level);

                // 验证链接列表的状态
                if (*ll_cur && !isUpdate)
                    throw std::runtime_error("新插入的元素的链接列表应该是空白的");

                // 设置链接列表的大小
                this->setListCount(ll_cur, selectedNeighbors.size());

                // 更新链接列表中的数据
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("可能的内存损坏");
                    if (level > this->element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("尝试在一个不存在的层级上创建链接");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            // 更新选定邻居的链接列表
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {
                // 锁定邻居的互斥锁
                std::unique_lock<std::mutex> lock(this->link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;

                // 根据层级选择正确的链接列表
                if (level == 0)
                    ll_other = this->get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = this->get_linklist(selectedNeighbors[idx], level);

                // 获取其他元素链接列表的大小
                size_t sz_link_list_other = this->getListCount(ll_other);

                // 进行各种验证
                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("sz_link_list_other 的值不正确");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("尝试将元素连接到自身");
                if (level > this->element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("尝试在一个不存在的层级上创建链接");

                // 获取链接列表中的数据
                tableint *data = (tableint *)(ll_other + 1);

                // 检查当前元素是否已存在于邻居的链接列表中
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

                // 如果当前元素不在邻居的链接列表中，则进行相应的更新
                if (!is_cur_c_present)
                {
                    if (sz_link_list_other < Mcurmax)
                    {
                        // 直接添加当前元素到链接列表末尾
                        data[sz_link_list_other] = cur_c;
                        this->setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else
                    {
                        // 找出“最弱”的元素以替换新的元素
                        dist_t d_max = fstdistfunc_(this->getDataByInternalId(cur_c), this->getDataByInternalId(selectedNeighbors[idx]), this->dist_func_param_);

                        // 构建优先级队列以找到最远的邻居
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(fstdistfunc_(this->getDataByInternalId(data[j]), this->getDataByInternalId(selectedNeighbors[idx]), this->dist_func_param_), data[j]);
                        }

                        // 更新邻居的链接列表
                        auto back_nns = &range_nns->at(this->getExternalLabel(selectedNeighbors[idx]));
                        getNeighborsByHeuristic2RecordPruned(candidates, Mcurmax, back_nns, external_id);

                        int indx = 0;
                        while (candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        this->setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
                        getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            // 返回下一个最接近的入口点
            return next_closest_entry_point;
        }
    };

    template <typename dist_t>
    class IndexSegmentGraph1D : public BaseIndex
    {
    public:
        vector<vector<SegmentNeighbor1D<dist_t>>> indexed_arr;

        IndexSegmentGraph1D(base_hnsw::SpaceInterface<dist_t> *s,
                            const DataWrapper *data)
            : BaseIndex(data)
        {
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            index_info = new IndexInfo();
            index_info->index_version_type = "IndexSegmentGraph1D";
        }

        base_hnsw::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;

        VisitedListPool *visited_list_pool_;
        IndexInfo *index_info;

        void printOnebatch(int pos = -1)
        {
            if (pos == -1)
            {
                pos = data_wrapper->data_size / 2;
            }

            cout << "nns at position: " << pos << endl;
            for (auto nns : indexed_arr[pos])
            {
                cout << nns.id << "->" << nns.end_id << ")\n"
                     << endl;
            }
            cout << endl;
        }

        void countNeighbors()
        {
            if (!indexed_arr.empty())
                for (unsigned j = 0; j < indexed_arr.size(); j++)
                {
                    int temp_size = 0;
                    temp_size += indexed_arr[j].size();
                    index_info->nodes_amount += temp_size;
                }
            index_info->avg_forward_nns =
                index_info->nodes_amount / (float)data_wrapper->data_size;
            if (isLog)
            {
                cout << "Avg. nn #: "
                     << index_info->nodes_amount / (float)data_wrapper->data_size << endl;
            }

            index_info->avg_reverse_nns = 0;
        }

        /**
         * 构建索引方法，根据给定的参数构建HNSW（Hierarchical Navigable Small World）图。
         *
         * @param index_params 指向包含索引构建参数的结构体指针。
         */
        void buildIndex(const IndexParams *index_params) override
        {
            cout << "Building Index using " << index_info->index_version_type << endl;
            timeval tt1, tt2;

            // 创建一个VisitedListPool实例用于管理访问列表。
            visited_list_pool_ =
                new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

            // 初始化L2空间度量。
            L2Space space(data_wrapper->data_dim);

            // 构建SegmentGraph1DHNSW实例，用于存储HNSW图。
            SegmentGraph1DHNSW<float> hnsw(
                *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
                index_params->ef_construction, index_params->random_seed);

            // 清空并重新分配indexed_arr数组大小为数据集大小。
            indexed_arr.clear();
            indexed_arr.resize(data_wrapper->data_size);

            // 设置hnsw的range_nns属性指向indexed_arr。
            hnsw.range_nns = &indexed_arr;

            // 获取当前时间作为计时开始点。
            gettimeofday(&tt1, NULL);

            // multi-thread also work, but not guaranteed as the paper
            // may has minor recall decrement
            // #pragma omp parallel for schedule(monotonic : dynamic)
            for (size_t i = 0; i < data_wrapper->data_size; ++i)
            {
                hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
            }

            // 将未被剪枝的HNSW图中的链接列表插入回indexed_arr。
            for (size_t i = 0; i < data_wrapper->data_size; ++i)
            {
                // 获取第i个节点的linklist0。
                linklistsizeint *ll_cur = hnsw.get_linklist0(i);

                // 计算链接列表中的元素数量。
                size_t link_list_count = hnsw.getListCount(ll_cur);
                tableint *data = (tableint *)(ll_cur + 1);

                // 遍历链接列表，创建邻接关系并添加到indexed_arr中。
                for (size_t j = 0; j < link_list_count; j++)
                {
                    int node_id = hnsw.getExternalLabel(data[j]);
                    SegmentNeighbor1D<dist_t> nn(node_id, 0, node_id);
                    indexed_arr.at(i).emplace_back(nn);
                }
            }

            // 打印构造索引的时间消耗。
            logTime(tt1, tt2, "Construct Time");

            // 再次获取当前时间以计算总耗时。
            gettimeofday(&tt2, NULL);

            // 更新索引信息中的索引构建时间。
            index_info->index_time = CountTime(tt1, tt2);

            // 统计邻居的数量。
            countNeighbors();

            // 如果设置打印一批，则调用printOnebatch()方法。
            if (index_params->print_one_batch)
            {
                printOnebatch();
            }
        }

        /**
         * @brief 范围过滤搜索，在范围内节点上仅计算距离。
         *
         * 此方法执行范围过滤搜索算法，它仅在指定查询边界内的节点上计算距离，
         * 并返回最接近查询点的节点列表。
         *
         * @param[in] search_params 搜索参数指针，包含搜索配置信息。
         * @param[out] search_info 搜索信息结构体指针，用于记录搜索过程中的统计信息。
         * @param[in] query 查询向量，表示要搜索的目标点。
         * @param[in] query_bound 查询边界，定义搜索的有效区间。
         *
         * @return 返回一个整数向量，其中包含了最邻近节点的ID。
         */
        vector<int> rangeFilteringSearchInRange(
            const SearchParams *search_params, SearchInfo *search_info,
            const vector<float> &query,
            const std::pair<int, int> query_bound) override
        {

            // 初始化时间变量和数据结构。
            timeval tt3, tt4;
            VisitedList *vl = visited_list_pool_->getFreeVisitedList(); // 获取空闲访问列表。
            vl_type *visited_array = vl->mass;                          // 访问数组。
            vl_type visited_array_tag = vl->curV;                       // 当前访问标记。
            float lower_bound = std::numeric_limits<float>::max();      // 最低界限初始化为最大浮点值。
            std::priority_queue<pair<float, int>> top_candidates;       // k近邻
            std::priority_queue<pair<float, int>> candidate_set;        // 要访问的点集合

            // 初始化搜索统计信息。
            search_info->total_comparison = 0;
            search_info->internal_search_time = 0;
            search_info->cal_dist_time = 0;
            search_info->fetch_nns_time = 0;
            num_search_comparison = 0;
            gettimeofday(&tt3, nullptr); // 开始计时。

            // 初始化三个入口点。
            vector<int> enter_list;
            {
                int lbound = query_bound.first;                   // 左界。
                int interval = (query_bound.second - lbound) / 3; // 区间间隔。
                for (size_t i = 0; i < 3; i++)
                {                                      // 遍历三次以确定入口点。
                    int point = lbound + interval * i; // 确定当前入口点位置。
                    float dist = fstdistfunc_(         // 计算查询点到该点的距离。
                        query.data(), data_wrapper->nodes[point].data(), dist_func_param_);
                    candidate_set.push(make_pair(-dist, point)); // 将负距离和点ID加入候选集。
                    enter_list.emplace_back(point);              // 添加入口点至列表。
                    visited_array[point] = visited_array_tag;    // 标记已访问。
                }
            }

            // 主循环：从候选集中提取并处理节点直到集合为空。
            while (!candidate_set.empty())
            {
                pair<float, int> current_node_pair = candidate_set.top(); // 获取当前节点对。
                int current_node_id = current_node_pair.second;           // 提取节点ID。

                if (-current_node_pair.first > lower_bound)
                { // 如果当前节点距离大于最低界限，则停止搜索。
                    break;
                }

#ifdef LOG_DEBUG_MODE
                cout << "current node: " << current_node_pair.second << "  -- "
                     << -current_node_pair.first << endl; // 打印调试信息。
#endif

                candidate_set.pop();                                        // 移除已处理的节点。
                auto neighbor_it = indexed_arr.at(current_node_id).begin(); // 获取邻居迭代器。

                while (neighbor_it != indexed_arr[current_node_id].end())
                {                                                  // 遍历所有邻居。
                    if ((neighbor_it->id < query_bound.second) &&  // 判断是否在查询范围内。
                        (neighbor_it->end_id == neighbor_it->id || // 或者结束ID等于ID或者超出右界。
                         neighbor_it->end_id >= query_bound.second))
                    {
                        int candidate_id = neighbor_it->id; // 获取候选节点ID。

                        if (!(visited_array[candidate_id] == visited_array_tag))
                        {                                                    // 如果未被访问过。
                            visited_array[candidate_id] = visited_array_tag; // 标记为已访问。
                            float dist = fstdistfunc_(                       // 计算距离。
                                query.data(), data_wrapper->nodes[candidate_id].data(), dist_func_param_);

                            num_search_comparison++;                                // 更新比较次数。
                            if (top_candidates.size() < search_params->search_ef || // 如果候选者数量小于限制。
                                lower_bound > dist)
                            {                                                       // 或者新距离更小。
                                candidate_set.push(make_pair(-dist, candidate_id)); // 加入候选集。
                                top_candidates.push(make_pair(dist, candidate_id)); // 加入最佳候选者队列。
                                if (top_candidates.size() > search_params->search_ef)
                                {                         // 如果超过限制。
                                    top_candidates.pop(); // 弹出最差的一个。
                                }
                                if (!top_candidates.empty())
                                {                                             // 如果队列非空。
                                    lower_bound = top_candidates.top().first; // 更新最低界限。
                                }
                            }
                        }
                    }
                    ++neighbor_it; // 迭代下一个邻居。
                }
            }

            // 处理结果。
            vector<int> res;
            while (top_candidates.size() > search_params->query_K)
            { // 调整候选者数量。
                top_candidates.pop();
            }

            while (!top_candidates.empty())
            { // 构建结果列表。
                res.emplace_back(top_candidates.top().second);
                top_candidates.pop();
            }
            search_info->total_comparison += num_search_comparison; // 更新总比较次数。

#ifdef LOG_DEBUG_MODE
            print_set(res);                            // 打印结果集。
            cout << l_bound << "," << r_bound << endl; // 输出边界信息。
            assert(false);                             // 断言失败（调试用）。
#endif

            visited_list_pool_->releaseVisitedList(vl); // 释放访问列表资源。

            gettimeofday(&tt4, nullptr);                            // 结束计时。
            CountTime(tt3, tt4, search_info->internal_search_time); // 统计内部搜索耗时。

            return res; // 返回结果列表。
        }

        // also calculate outbount dists, similar to knn-first
        // This is bad for half bounded search.
        vector<int> rangeFilteringSearchOutBound(
            const SearchParams *search_params, SearchInfo *search_info,
            const vector<float> &query,
            const std::pair<int, int> query_bound) override
        {
            // timeval tt1, tt2;
            timeval tt3, tt4;

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            float lower_bound = std::numeric_limits<float>::max();
            std::priority_queue<pair<float, int>> top_candidates;
            std::priority_queue<pair<float, int>> candidate_set;

            search_info->total_comparison = 0;
            search_info->internal_search_time = 0;
            search_info->cal_dist_time = 0;
            search_info->fetch_nns_time = 0;
            num_search_comparison = 0;
            // finding enters
            vector<int> enter_list;
            {
                int lbound = query_bound.first;
                int interval = (query_bound.second - lbound) / 3;
                for (size_t i = 0; i < 3; i++)
                {
                    int point = lbound + interval * i;
                    float dist = fstdistfunc_(
                        query.data(), data_wrapper->nodes[point].data(), dist_func_param_);
                    candidate_set.push(make_pair(-dist, point));
                    enter_list.emplace_back(point);
                    visited_array[point] = visited_array_tag;
                }
            }
            gettimeofday(&tt3, NULL);

            // size_t hop_counter = 0;

            while (!candidate_set.empty())
            {
                std::pair<float, int> current_node_pair = candidate_set.top();
                int current_node_id = current_node_pair.second;
                if (-current_node_pair.first > lower_bound)
                {
                    break;
                }

                // hop_counter++;

                candidate_set.pop();

                auto neighbor_it = indexed_arr.at(current_node_id).begin();
                // gettimeofday(&tt1, NULL);

                while (neighbor_it != indexed_arr[current_node_id].end())
                {
                    if ((neighbor_it->id < query_bound.second))
                    {
                        int candidate_id = neighbor_it->id;

                        if (!(visited_array[candidate_id] == visited_array_tag))
                        {
                            visited_array[candidate_id] = visited_array_tag;
                            float dist = fstdistfunc_(query.data(),
                                                      data_wrapper->nodes[candidate_id].data(),
                                                      dist_func_param_);

                            num_search_comparison++;
                            if (top_candidates.size() < search_params->search_ef ||
                                lower_bound > dist)
                            {
                                candidate_set.emplace(-dist, candidate_id);
                                // add to top_candidates only in range
                                if (candidate_id <= query_bound.second &&
                                    candidate_id >= query_bound.first)
                                {
                                    top_candidates.emplace(dist, candidate_id);
                                    if (top_candidates.size() > search_params->search_ef)
                                    {
                                        top_candidates.pop();
                                    }
                                    if (!top_candidates.empty())
                                    {
                                        lower_bound = top_candidates.top().first;
                                    }
                                }
                            }
                        }
                    }
                    neighbor_it++;
                }

                // gettimeofday(&tt2, NULL);
                // AccumulateTime(tt1, tt2, search_info->cal_dist_time);
            }

            vector<int> res;
            while (top_candidates.size() > search_params->query_K)
            {
                top_candidates.pop();
            }

            while (!top_candidates.empty())
            {
                res.emplace_back(top_candidates.top().second);
                top_candidates.pop();
            }
            search_info->total_comparison += num_search_comparison;

#ifdef LOG_DEBUG_MODE
            print_set(res);
            cout << l_bound << "," << r_bound << endl;
            assert(false);
#endif
            visited_list_pool_->releaseVisitedList(vl);

            gettimeofday(&tt4, NULL);
            CountTime(tt3, tt4, search_info->internal_search_time);

            return res;
        }

        // TODO: save and load segment graph 1d
        void save(const string &save_path) {}

        void load(const string &load_path) {}

        ~IndexSegmentGraph1D()
        {
            delete index_info;
            indexed_arr.clear();
            delete visited_list_pool_;
        }
    };

} // namespace SeRF