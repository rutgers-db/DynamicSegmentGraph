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

// 确保在包含 base_hnsw 之前已经定义了 vector
using std::vector;

#include "baselines/hnswalg.h"
#include "baselines/hnswlib.h"
#include "baselines/space_l2.h"
#include "infrastructure/io/data_loader.h"
#include "infrastructure/io/index_serializer.h"
#include "infrastructure/utils/utils.h"
#include "infrastructure/utils/distance.h"

using namespace hnswlib_incre;

 namespace SeRF {
 
 struct OneSegmentNeighbors {
     OneSegmentNeighbors();
     OneSegmentNeighbors(unsigned num);
     OneSegmentNeighbors(unsigned num, int start, int end);
 
     // vector<pair<int, float>> nns;
     vector<int> nns_id;
     unsigned batch; // batch id
     int start = -1; // left position
     int end = -2;   // right position
     const unsigned size();
 };
 
 // struct OneTupleNeighbor {
 //   int start = -1;
 //   int end = -2;
 //   int nn_id;
 // }
 
 struct DirectedSegNeighbors {
     vector<OneSegmentNeighbors> forward_nns;
     vector<int> reverse_nns;
 };
 
 template <typename dist_t>
 class SegmentGraph2DHNSW : public base_hnsw::HierarchicalNSW<float> {
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
     SegmentGraph2DHNSW(const BaseIndex::IndexParams &index_params,
                        base_hnsw::SpaceInterface<float> *s,
                        size_t max_elements,
                        size_t M = 16,
                        size_t ef_construction = 200,
                        size_t random_seed = 100);
 
     // 指向BaseIndex::IndexParams类型的常量指针，存储索引参数
     const BaseIndex::IndexParams *params;
 
    // 存储指向段图邻居列表的指针，表示图结构中的边信息
    vector<DirectedSegNeighbors> *segment_graph;

    // Search parameters
    size_t ef_max_ = 400;                    // Maximum expansion factor
    size_t ef_basic_construction_ = 200;     // Basic construction expansion factor

    /**
     * 在构建HNSW图时优化搜索过程，保留更多邻居节点信息。
      * 这个是基本就是原本的search 就是在整个图里当前层搜最近的
      * 或许可以结合（RNN-descent）以提升效率。
      *
      * @param ep_id 起始点ID
      * @param data_point 数据点指针
      * @param layer 当前层级
      * @return 返回一个优先队列，其中包含距离和节点ID对，按距离排序。
      */
     virtual std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                                 std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                                 base_hnsw::HierarchicalNSW<float>::CompareByFirst>
     searchBaseLayerLevel0(base_hnsw::tableint ep_id, const void *data_point, int layer);
 
     /**
      * @file src/segment_graph_2d.h
      * @brief 互连新元素并递归地应用启发式剪枝算法
      *
      * 此函数用于连接新的数据点到图中的现有节点，
      * 并通过优先队列处理候选邻居以优化连接过程。
      */
 
     virtual base_hnsw::tableint mutuallyConnectNewElementLevel0(
         const void *data_point, /**< 当前数据点 */
         base_hnsw::tableint cur_c,         /**< 当前节点的内部标识符 */
         std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                             std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                             base_hnsw::HierarchicalNSW<float>::CompareByFirst> &top_candidates, /**< 候选邻居列表 */
         int level,                                           /**< 连接级别 */
         bool isUpdate);                                       /**< 是否更新已存在的链接 */

 };
 
 class IndexSegmentGraph2D : public BaseIndex {
 public:
     vector<DirectedSegNeighbors> directed_indexed_arr;
 
     IndexSegmentGraph2D(base_hnsw::SpaceInterface<float> *s,
                         const DataWrapper *data);
     base_hnsw::DISTFUNC<float> fstdistfunc_;
     void *dist_func_param_;
 
     VisitedListPool *visited_list_pool_ = nullptr;
     IndexInfo *index_info;
     const BaseIndex::IndexParams *index_params_;
     SegmentGraph2DHNSW<float> *hnsw;
     unsigned index_k;
     // connect reverse neighbors, do pruning all sth else. In this base version,
     // just no prune and collect all reverse neighbor in one batch.
     void processReverseNeighbors();
 
     void processReverseNeighbors(vector<unsigned> & nodes_ids);
 
 
     void printOnebatch();
 
     /**
      * @brief 计算图中的邻居节点数量统计信息
      *
      * 此方法遍历有向图索引数组以计算平均前向邻居数、最大前向批量邻居数，
      * 平均反向邻居数、最大反向邻居数以及相关批处理计数。
      */
     void countNeighbrs();
 
     void buildIndex(const IndexParams *index_params);
 
     void initForScabilityExp(const IndexParams *index_params, L2Space *space);

     SearchResult searchKnn(
         const SearchParams *search_params,
         SearchInfo *search_info,
         const vector<float> &query) override;
 
     void insert_batch(vector<unsigned> &nodes_ids);
 
     vector<OneSegmentNeighbors>::const_iterator decompressForwardPath(
         const vector<OneSegmentNeighbors> &forward_nns,
         const int lbound);
 
     vector<OneSegmentNeighbors>::const_iterator decompressReversePath(
         const vector<OneSegmentNeighbors> &reverse_nns,
         const int rbound);
 
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
         const SearchParams *search_params,
         SearchInfo *search_info,
         const vector<float> &query,
         const std::pair<int, int> query_bound);
 
     // also calculate outbount dists, similar to knn-first
     vector<int> rangeFilteringSearchOutBound(
         const SearchParams *search_params,
         SearchInfo *search_info,
         const vector<float> &query,
         const std::pair<int, int> query_bound);
 
     void save(const string &file_path);
 
     void load(const string &file_path);
 
     ~IndexSegmentGraph2D();
 };
 } // namespace SeRF