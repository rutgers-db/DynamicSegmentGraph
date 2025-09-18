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

// 确保在包含 base_hnsw 之前已经定义了 vector
using std::vector;

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
    CompressedPoint(unsigned _external_id, unsigned _ll, unsigned _lr, unsigned _rl, unsigned _rr);
    CompressedPoint();

    unsigned external_id;
    unsigned ll, lr, rl, rr;

    inline bool const if_in_compressed_range(const unsigned &query_L, const unsigned &query_R);

    bool operator<(const CompressedPoint &other);
};

template <typename dist_t>
struct DirectedPointNeighbors {
    vector<CompressedPoint<dist_t>> nns;
    vector<CompressedPoint<dist_t>> rev_nns;

    size_t countNeighbors();
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
                size_t random_seed = 100);

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
    searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer);

    std::vector<tableint> selectedNeighbors;
    std::vector<tableint> return_list;
    unsigned iter_counter = 0;
    bool complete = false;
    tableint next_closest_entry_point;
    void init_selectedNeighbors();

    void get_selectedNeighbors(tableint passed_c, dist_t dist_to_query, const unsigned &Mcurmax);

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
    void dfs(vector<unsigned> &prefix_idx, unsigned PIVOT_ID, unsigned L, unsigned R, unsigned lr, unsigned rl);

    void generate_compressed_neighbors(
        std::priority_queue<std::pair<dist_t, tableint>> &queue_closest,
        unsigned center_external_id,
        const unsigned &index_k);

    void gen_rev_neighbors(unsigned center_external_id);

    virtual tableint mutuallyConnectNewElementLevel0(
        const void *data_point, 
        tableint cur_c,        
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates, 
        int level,                                           
        bool isUpdate);
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
        BaseIndex(data);

    void printOnebatch();

    void countNeighbrs();

    void buildIndex(const IndexParams *index_params);

    void initForScabilityExp(const IndexParams *index_params, L2Space *space);

    void rebuild_batchInHNSW(vector<unsigned> &nodes_ids);

    void insert_batch(vector<unsigned> &nodes_ids);

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
        const std::pair<int, int> query_bound);

    vector<int> rangeFilteringSearchOutBound(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound);

    // Save function to store the IndexCompactGraph to a file
    void save(const std::string &file_path);

    // Load function to load the IndexCompactGraph from a file
    void load(const std::string &file_path);

    ~IndexCompactGraph();
};
} // namespace Compact