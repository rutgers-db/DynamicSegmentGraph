/**
 * @file range_search.h
 * @brief 统一的范围过滤搜索接口和实现
 * @author DSG Team
 * @date 2024
 * 
 * 本文件提供了统一的范围过滤搜索抽象接口，将各种图结构中的
 * rangeFilteringSearch* 方法抽象为统一的接口和通用组件。
 */

#pragma once

#include <algorithm>
#include <queue>
#include <vector>
#include <memory>
#include <functional>
#include <sys/time.h>

#include "base_search.h"
#include "base_hnsw/visited_list_pool.h"
#include "base_hnsw/space_interface.h"
#include "data_wrapper.h"
#include "base_struct.h"

using std::vector;
using std::pair;
using std::priority_queue;
using std::function;
using namespace base_hnsw;

namespace rangeindex {

/**
 * @brief 搜索策略枚举
 * 定义不同的范围过滤搜索策略
 */
enum class RangeSearchStrategy {
    IN_RANGE_ONLY,      // 仅在范围内搜索
    OUT_BOUND_INCLUDED, // 包含范围外的搜索
    HYBRID              // 混合策略
};

/**
 * @brief 邻居解压策略枚举
 * 定义不同的邻居数据结构解压方式
 */
enum class NeighborDecompressionType {
    BATCH_NEIGHBORS,    // 批处理邻居（OneBatchNeighbors）
    SEGMENT_NEIGHBORS,  // 分段邻居（OneSegmentNeighbors）
    COMPACT_NEIGHBORS   // 压缩邻居（直接存储）
};

/**
 * @brief 搜索上下文结构
 * 封装搜索过程中的通用状态和数据结构
 */
struct RangeSearchContext {
    // 访问列表管理
    VisitedList* vl;
    vl_type* visited_array;
    vl_type visited_array_tag;
    
    // 搜索数据结构
    float lower_bound;
    priority_queue<pair<float, int>> top_candidates;    // 最大堆，存储最终结果
    priority_queue<pair<float, int>> candidate_set;     // 最小堆，存储待探索候选
    
    // 时间测量
    timeval tt1, tt2, tt3, tt4;
    
    // 缓存和临时数据
    vector<unsigned> fetched_nns;
    vector<int> enter_list;
    
    /**
     * @brief 初始化搜索上下文
     * @param visited_list_pool 访问列表池
     */
    void initialize(VisitedListPool* visited_list_pool) {
        vl = visited_list_pool->getFreeVisitedList();
        visited_array = vl->mass;
        visited_array_tag = vl->curV;
        lower_bound = std::numeric_limits<float>::max();
        
        // 清空数据结构
        while (!top_candidates.empty()) top_candidates.pop();
        while (!candidate_set.empty()) candidate_set.pop();
        fetched_nns.clear();
        enter_list.clear();
        
        // 预分配空间
        fetched_nns.reserve(100);
    }
    
    /**
     * @brief 释放搜索上下文资源
     * @param visited_list_pool 访问列表池
     */
    void cleanup(VisitedListPool* visited_list_pool) {
        visited_list_pool->releaseVisitedList(vl);
    }
};

/**
 * @brief 邻居迭代器抽象接口
 * 统一不同邻居数据结构的访问方式
 */
class NeighborIterator {
public:
    virtual ~NeighborIterator() = default;
    
    /**
     * @brief 检查是否还有更多邻居
     */
    virtual bool hasNext() const = 0;
    
    /**
     * @brief 获取下一个邻居ID
     */
    virtual int getNext() = 0;
    
    /**
     * @brief 跳转到下一个批次/段
     */
    virtual void moveToNextBatch() = 0;
    
    /**
     * @brief 检查当前批次/段是否在指定范围内
     */
    virtual bool isInRange(int lbound, int rbound) const = 0;
};

/**
 * @brief 批处理邻居迭代器实现
 */
template<typename BatchNeighborType>
class BatchNeighborIterator : public NeighborIterator {
private:
    typename vector<BatchNeighborType>::const_iterator current_batch_;
    typename vector<BatchNeighborType>::const_iterator end_batch_;
    size_t current_idx_in_batch_;
    
public:
    BatchNeighborIterator(
        typename vector<BatchNeighborType>::const_iterator begin,
        typename vector<BatchNeighborType>::const_iterator end)
        : current_batch_(begin), end_batch_(end), current_idx_in_batch_(0) {}
    
    bool hasNext() const override {
        return current_batch_ != end_batch_ && 
               (current_idx_in_batch_ < current_batch_->nns.size() || 
                std::next(current_batch_) != end_batch_);
    }
    
    int getNext() override {
        if (current_idx_in_batch_ >= current_batch_->nns.size()) {
            moveToNextBatch();
        }
        return current_batch_->nns[current_idx_in_batch_++];
    }
    
    void moveToNextBatch() override {
        if (current_batch_ != end_batch_) {
            ++current_batch_;
            current_idx_in_batch_ = 0;
        }
    }
    
    bool isInRange(int lbound, int rbound) const override {
        if (current_batch_ == end_batch_) return false;
        return !(rbound <= current_batch_->start || lbound >= current_batch_->end);
    }
};

/**
 * @brief 范围过滤搜索执行器
 * 核心搜索算法的统一实现
 */
class RangeFilteringSearchExecutor {
private:
    const DataWrapper* data_wrapper_;
    SpaceInterface<float>* space_;
    VisitedListPool* visited_list_pool_;
    
public:
    RangeFilteringSearchExecutor(
        const DataWrapper* data_wrapper,
        SpaceInterface<float>* space,
        VisitedListPool* visited_list_pool)
        : data_wrapper_(data_wrapper), space_(space), visited_list_pool_(visited_list_pool) {}
    
    /**
     * @brief 执行范围过滤搜索的核心算法
     * 
     * @param search_params 搜索参数
     * @param search_info 搜索统计信息
     * @param query 查询向量
     * @param query_bound 查询范围边界
     * @param strategy 搜索策略
     * @param forward_neighbor_provider 前向邻居提供者函数
     * @param reverse_neighbor_provider 反向邻居提供者函数
     * @param entry_point_finder 入口点查找函数
     * @return 搜索结果
     */
    vector<int> execute(
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        RangeSearchStrategy strategy,
        function<std::unique_ptr<NeighborIterator>(int)> forward_neighbor_provider,
        function<std::unique_ptr<NeighborIterator>(int)> reverse_neighbor_provider,
        function<vector<int>(const pair<int, int>&)> entry_point_finder
    );
    
private:
    /**
     * @brief 初始化入口点
     */
    void initializeEntryPoints(
        RangeSearchContext& ctx,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        const function<vector<int>(const pair<int, int>&)>& entry_point_finder,
        BaseIndex::SearchInfo* search_info
    );
    
    /**
     * @brief 执行贪心搜索
     */
    void performGreedySearch(
        RangeSearchContext& ctx,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        const BaseIndex::SearchParams* search_params,
        const function<std::unique_ptr<NeighborIterator>(int)>& forward_neighbor_provider,
        const function<std::unique_ptr<NeighborIterator>(int)>& reverse_neighbor_provider,
        RangeSearchStrategy strategy,
        BaseIndex::SearchInfo* search_info
    );
    
    /**
     * @brief 处理候选节点
     */
    void processCandidateNode(
        RangeSearchContext& ctx,
        int node_id,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        const BaseIndex::SearchParams* search_params,
        RangeSearchStrategy strategy,
        BaseIndex::SearchInfo* search_info
    );
    
    /**
     * @brief 探索邻居节点
     */
    void exploreNeighbors(
        RangeSearchContext& ctx,
        int node_id,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        const function<std::unique_ptr<NeighborIterator>(int)>& neighbor_provider,
        RangeSearchStrategy strategy,
        BaseIndex::SearchInfo* search_info
    );
    
    /**
     * @brief 检查节点是否在范围内
     */
    bool isNodeInRange(int node_id, const pair<int, int>& query_bound) const {
        return node_id >= query_bound.first && node_id <= query_bound.second;
    }
    
    /**
     * @brief 计算距离
     */
    float calculateDistance(const vector<float>& query, int node_id) {
        return space_->get_dist_func()(query.data(), data_wrapper_->getDataByInternalId(node_id), space_->get_dist_func_param());
    }
    
    /**
     * @brief 构建最终结果
     */
    vector<int> buildResult(
        const RangeSearchContext& ctx,
        const BaseIndex::SearchParams* search_params
    );
};

/**
 * @brief 范围搜索工厂类
 * 为不同的图结构创建相应的搜索组件
 */
class RangeSearchFactory {
public:
    /**
     * @brief 创建批处理邻居的前向迭代器
     */
    template<typename BatchNeighborType>
    static function<std::unique_ptr<NeighborIterator>(int)> 
    createBatchForwardProvider(const vector<vector<BatchNeighborType>>* neighbors, const pair<int, int>& query_bound) {
        return [neighbors, query_bound](int node_id) -> std::unique_ptr<NeighborIterator> {
            const auto& node_neighbors = (*neighbors)[node_id];
            
            // 找到第一个可能相交的批次
            auto forward_it = node_neighbors.begin();
            while (forward_it != node_neighbors.end()) {
                if (query_bound.first < forward_it->end) {
                    break;
                }
                ++forward_it;
            }
            
            return std::make_unique<BatchNeighborIterator<BatchNeighborType>>(forward_it, node_neighbors.end());
        };
    }
    
    /**
     * @brief 创建批处理邻居的反向迭代器
     */
    template<typename BatchNeighborType>
    static function<std::unique_ptr<NeighborIterator>(int)> 
    createBatchReverseProvider(const vector<vector<BatchNeighborType>>* neighbors, const pair<int, int>& query_bound) {
        return [neighbors, query_bound](int node_id) -> std::unique_ptr<NeighborIterator> {
            const auto& node_neighbors = (*neighbors)[node_id];
            
            // 找到第一个可能相交的批次
            auto reverse_it = node_neighbors.begin();
            while (reverse_it != node_neighbors.end()) {
                if (query_bound.second > reverse_it->start) {
                    break;
                }
                ++reverse_it;
            }
            
            return std::make_unique<BatchNeighborIterator<BatchNeighborType>>(reverse_it, node_neighbors.end());
        };
    }
    
    /**
     * @brief 创建标准入口点查找器
     */
    static function<vector<int>(const pair<int, int>&)> 
    createStandardEntryPointFinder(const DataWrapper* data_wrapper) {
        return [data_wrapper](const pair<int, int>& query_bound) -> vector<int> {
            vector<int> entries;
            int lbound = query_bound.first;
            int rbound = query_bound.second;
            int range_size = rbound - lbound + 1;
            
            if (range_size >= 3) {
                // 多入口点策略：选择范围内的3个均匀分布的点
                entries.push_back(lbound);
                entries.push_back(lbound + range_size / 2);
                entries.push_back(rbound);
            } else {
                // 小范围：使用所有点作为入口
                for (int i = lbound; i <= rbound; ++i) {
                    entries.push_back(i);
                }
            }
            
            return entries;
        };
    }
};

/**
 * @brief 具体的邻居适配器实现
 */

/**
 * @brief OneBatchNeighbors 适配器
 */
class OneBatchNeighborIterator : public NeighborIterator {
private:
    vector<OneBatchNeighbors>::const_iterator current_batch_;
    vector<OneBatchNeighbors>::const_iterator end_batch_;
    size_t current_idx_in_batch_;
    
public:
    OneBatchNeighborIterator(
        vector<OneBatchNeighbors>::const_iterator begin,
        vector<OneBatchNeighbors>::const_iterator end)
        : current_batch_(begin), end_batch_(end), current_idx_in_batch_(0) {}
    
    bool hasNext() const override {
        if (current_batch_ == end_batch_) return false;
        if (current_idx_in_batch_ < current_batch_->nns.size()) return true;
        return std::next(current_batch_) != end_batch_;
    }
    
    int getNext() override {
        if (current_idx_in_batch_ >= current_batch_->nns.size()) {
            moveToNextBatch();
        }
        return current_batch_->nns[current_idx_in_batch_++];
    }
    
    void moveToNextBatch() override {
        if (current_batch_ != end_batch_) {
            ++current_batch_;
            current_idx_in_batch_ = 0;
        }
    }
    
    bool isInRange(int lbound, int rbound) const override {
        if (current_batch_ == end_batch_) return false;
        return !(rbound <= current_batch_->start || lbound >= current_batch_->end);
    }
};

/**
 * @brief OneSegmentNeighbors 适配器
 */
class OneSegmentNeighborIterator : public NeighborIterator {
private:
    vector<OneSegmentNeighbors>::const_iterator current_segment_;
    vector<OneSegmentNeighbors>::const_iterator end_segment_;
    size_t current_idx_in_segment_;
    
public:
    OneSegmentNeighborIterator(
        vector<OneSegmentNeighbors>::const_iterator begin,
        vector<OneSegmentNeighbors>::const_iterator end)
        : current_segment_(begin), end_segment_(end), current_idx_in_segment_(0) {}
    
    bool hasNext() const override {
        if (current_segment_ == end_segment_) return false;
        if (current_idx_in_segment_ < current_segment_->nns.size()) return true;
        return std::next(current_segment_) != end_segment_;
    }
    
    int getNext() override {
        if (current_idx_in_segment_ >= current_segment_->nns.size()) {
            moveToNextBatch();
        }
        return current_segment_->nns[current_idx_in_segment_++];
    }
    
    void moveToNextBatch() override {
        if (current_segment_ != end_segment_) {
            ++current_segment_;
            current_idx_in_segment_ = 0;
        }
    }
    
    bool isInRange(int lbound, int rbound) const override {
        if (current_segment_ == end_segment_) return false;
        return !(rbound <= current_segment_->start || lbound >= current_segment_->end);
    }
};

/**
 * @brief 简单向量邻居适配器（用于压缩图等）
 */
class VectorNeighborIterator : public NeighborIterator {
private:
    vector<unsigned>::const_iterator current_;
    vector<unsigned>::const_iterator end_;
    
public:
    VectorNeighborIterator(
        vector<unsigned>::const_iterator begin,
        vector<unsigned>::const_iterator end)
        : current_(begin), end_(end) {}
    
    bool hasNext() const override {
        return current_ != end_;
    }
    
    int getNext() override {
        return static_cast<int>(*current_++);
    }
    
    void moveToNextBatch() override {
        // 对于简单向量，没有批次概念
    }
    
    bool isInRange(int lbound, int rbound) const override {
        // 对于简单向量，范围检查需要在外部进行
        return true;
    }
};

/**
 * @brief RangeFilteringSearchExecutor 的实现
 */

// 执行范围过滤搜索的核心算法
inline vector<int> RangeFilteringSearchExecutor::execute(
    const BaseIndex::SearchParams* search_params,
    BaseIndex::SearchInfo* search_info,
    const vector<float>& query,
    const pair<int, int>& query_bound,
    RangeSearchStrategy strategy,
    function<std::unique_ptr<NeighborIterator>(int)> forward_neighbor_provider,
    function<std::unique_ptr<NeighborIterator>(int)> reverse_neighbor_provider,
    function<vector<int>(const pair<int, int>&)> entry_point_finder
) {
    RangeSearchContext ctx;
    ctx.initialize(visited_list_pool_);
    
    try {
        // 初始化搜索统计信息
        search_info->total_comparison = 0;
        search_info->internal_search_time = 0;
        search_info->cal_dist_time = 0;
        search_info->fetch_nns_time = 0;
        
        // 初始化入口点
        initializeEntryPoints(ctx, query, query_bound, entry_point_finder, search_info);
        
        // 执行贪心搜索
        performGreedySearch(ctx, query, query_bound, search_params, 
                          forward_neighbor_provider, reverse_neighbor_provider, 
                          strategy, search_info);
        
        // 构建最终结果
        auto result = buildResult(ctx, search_params);
        
        ctx.cleanup(visited_list_pool_);
        return result;
        
    } catch (...) {
        ctx.cleanup(visited_list_pool_);
        throw;
    }
}

// 初始化入口点
inline void RangeFilteringSearchExecutor::initializeEntryPoints(
    RangeSearchContext& ctx,
    const vector<float>& query,
    const pair<int, int>& query_bound,
    const function<vector<int>(const pair<int, int>&)>& entry_point_finder,
    BaseIndex::SearchInfo* search_info
) {
    gettimeofday(&ctx.tt1, NULL);
    
    ctx.enter_list = entry_point_finder(query_bound);
    
    // 将入口点添加到候选集
    for (int entry : ctx.enter_list) {
        if (ctx.visited_array[entry] != ctx.visited_array_tag) {
            ctx.visited_array[entry] = ctx.visited_array_tag;
            
            float dist = calculateDistance(query, entry);
            search_info->total_comparison++;
            
            ctx.candidate_set.push({-dist, entry});
            ctx.top_candidates.push({dist, entry});
            
            if (dist < ctx.lower_bound) {
                ctx.lower_bound = dist;
            }
        }
    }
    
    gettimeofday(&ctx.tt2, NULL);
    search_info->fetch_nns_time += getInterval(&ctx.tt1, &ctx.tt2);
}

// 执行贪心搜索
inline void RangeFilteringSearchExecutor::performGreedySearch(
    RangeSearchContext& ctx,
    const vector<float>& query,
    const pair<int, int>& query_bound,
    const BaseIndex::SearchParams* search_params,
    const function<std::unique_ptr<NeighborIterator>(int)>& forward_neighbor_provider,
    const function<std::unique_ptr<NeighborIterator>(int)>& reverse_neighbor_provider,
    RangeSearchStrategy strategy,
    BaseIndex::SearchInfo* search_info
) {
    while (!ctx.candidate_set.empty()) {
        auto current = ctx.candidate_set.top();
        ctx.candidate_set.pop();
        
        float current_dist = -current.first;
        int current_node = current.second;
        
        // 如果当前距离大于下界且候选集足够大，则停止搜索
        if (current_dist > ctx.lower_bound && 
            ctx.top_candidates.size() >= search_params->search_ef) {
            break;
        }
        
        // 探索前向邻居
        if (forward_neighbor_provider) {
            exploreNeighbors(ctx, current_node, query, query_bound, 
                           forward_neighbor_provider, strategy, search_info);
        }
        
        // 探索反向邻居
        if (reverse_neighbor_provider) {
            exploreNeighbors(ctx, current_node, query, query_bound, 
                           reverse_neighbor_provider, strategy, search_info);
        }
        
        // 更新下界
        if (ctx.top_candidates.size() > search_params->search_ef) {
            while (ctx.top_candidates.size() > search_params->search_ef) {
                ctx.top_candidates.pop();
            }
            ctx.lower_bound = ctx.top_candidates.top().first;
        }
    }
}

// 探索邻居节点
inline void RangeFilteringSearchExecutor::exploreNeighbors(
    RangeSearchContext& ctx,
    int node_id,
    const vector<float>& query,
    const pair<int, int>& query_bound,
    const function<std::unique_ptr<NeighborIterator>(int)>& neighbor_provider,
    RangeSearchStrategy strategy,
    BaseIndex::SearchInfo* search_info
) {
    gettimeofday(&ctx.tt1, NULL);
    
    auto neighbor_iter = neighbor_provider(node_id);
    
    gettimeofday(&ctx.tt2, NULL);
    search_info->fetch_nns_time += getInterval(&ctx.tt1, &ctx.tt2);
    
    while (neighbor_iter && neighbor_iter->hasNext()) {
        // 检查当前批次/段是否与查询范围相交
        if (!neighbor_iter->isInRange(query_bound.first, query_bound.second)) {
            neighbor_iter->moveToNextBatch();
            continue;
        }
        
        int neighbor_id = neighbor_iter->getNext();
        
        // 检查是否已访问
        if (ctx.visited_array[neighbor_id] == ctx.visited_array_tag) {
            continue;
        }
        ctx.visited_array[neighbor_id] = ctx.visited_array_tag;
        
        // 根据策略决定是否计算距离
        bool should_calculate_distance = false;
        switch (strategy) {
            case RangeSearchStrategy::IN_RANGE_ONLY:
                should_calculate_distance = isNodeInRange(neighbor_id, query_bound);
                break;
            case RangeSearchStrategy::OUT_BOUND_INCLUDED:
                should_calculate_distance = true;
                break;
            case RangeSearchStrategy::HYBRID:
                // 混合策略：优先计算范围内的，但也考虑范围外的
                should_calculate_distance = true;
                break;
        }
        
        if (should_calculate_distance) {
            gettimeofday(&ctx.tt3, NULL);
            float dist = calculateDistance(query, neighbor_id);
            search_info->total_comparison++;
            gettimeofday(&ctx.tt4, NULL);
            search_info->cal_dist_time += getInterval(&ctx.tt3, &ctx.tt4);
            
            // 添加到候选集
            if (dist < ctx.lower_bound || ctx.top_candidates.size() < search_params->search_ef) {
                ctx.candidate_set.push({-dist, neighbor_id});
                ctx.top_candidates.push({dist, neighbor_id});
                
                if (dist < ctx.lower_bound) {
                    ctx.lower_bound = dist;
                }
            }
        }
    }
}

// 构建最终结果
inline vector<int> RangeFilteringSearchExecutor::buildResult(
    const RangeSearchContext& ctx,
    const BaseIndex::SearchParams* search_params
) {
    vector<pair<float, int>> result_candidates;
    
    // 从优先队列中提取结果
    auto temp_candidates = ctx.top_candidates;
    while (!temp_candidates.empty()) {
        result_candidates.push_back(temp_candidates.top());
        temp_candidates.pop();
    }
    
    // 按距离排序（升序）
    std::sort(result_candidates.begin(), result_candidates.end());
    
    // 提取前K个结果
    vector<int> result;
    size_t result_size = std::min(static_cast<size_t>(search_params->query_K), result_candidates.size());
    result.reserve(result_size);
    
    for (size_t i = 0; i < result_size; ++i) {
        result.push_back(result_candidates[i].second);
    }
    
    return result;
}

/**
 * @brief 扩展的工厂方法，支持具体的数据结构
 */
class ExtendedRangeSearchFactory : public RangeSearchFactory {
public:
    /**
     * @brief 为OneBatchNeighbors创建前向提供者
     */
    static function<std::unique_ptr<NeighborIterator>(int)> 
    createOneBatchForwardProvider(const vector<vector<OneBatchNeighbors>>* neighbors, const pair<int, int>& query_bound) {
        return [neighbors, query_bound](int node_id) -> std::unique_ptr<NeighborIterator> {
            const auto& node_neighbors = (*neighbors)[node_id];
            
            // 找到第一个可能相交的批次
            auto forward_it = node_neighbors.begin();
            while (forward_it != node_neighbors.end()) {
                if (query_bound.first < forward_it->end) {
                    break;
                }
                ++forward_it;
            }
            
            return std::make_unique<OneBatchNeighborIterator>(forward_it, node_neighbors.end());
        };
    }
    
    /**
     * @brief 为OneSegmentNeighbors创建前向提供者
     */
    static function<std::unique_ptr<NeighborIterator>(int)> 
    createOneSegmentForwardProvider(const vector<vector<OneSegmentNeighbors>>* neighbors, const pair<int, int>& query_bound) {
        return [neighbors, query_bound](int node_id) -> std::unique_ptr<NeighborIterator> {
            const auto& node_neighbors = (*neighbors)[node_id];
            
            // 找到第一个可能相交的段
            auto forward_it = node_neighbors.begin();
            while (forward_it != node_neighbors.end()) {
                if (query_bound.first < forward_it->end) {
                    break;
                }
                ++forward_it;
            }
            
            return std::make_unique<OneSegmentNeighborIterator>(forward_it, node_neighbors.end());
        };
    }
    
    /**
     * @brief 为简单向量邻居创建提供者（如压缩图）
     */
    static function<std::unique_ptr<NeighborIterator>(int)> 
    createVectorNeighborProvider(const vector<vector<unsigned>>* neighbors, const pair<int, int>& query_bound) {
        return [neighbors, query_bound](int node_id) -> std::unique_ptr<NeighborIterator> {
            const auto& node_neighbors = (*neighbors)[node_id];
            return std::make_unique<VectorNeighborIterator>(node_neighbors.begin(), node_neighbors.end());
        };
    }
};

/**
 * @brief 使用示例和适配器类
 * 
 * 这个类展示了如何将现有的各种图结构适配到统一的接口上
 */
class RangeSearchAdapter {
public:
    /**
     * @brief 为递归批处理HNSW创建适配器
     */
    template<typename GraphType>
    static vector<int> adaptRecursionBatchHNSW(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        RangeSearchStrategy strategy = RangeSearchStrategy::IN_RANGE_ONLY
    ) {
        RangeFilteringSearchExecutor executor(
            graph.data_wrapper, graph.space_, graph.visited_list_pool_
        );
        
        auto forward_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.forward_nns_batches, query_bound);
        auto reverse_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.reverse_nns_batches, query_bound);
        auto entry_finder = RangeSearchFactory::createStandardEntryPointFinder(graph.data_wrapper);
        
        return executor.execute(search_params, search_info, query, query_bound, 
                              strategy, forward_provider, reverse_provider, entry_finder);
    }
    
    /**
     * @brief 为2D分段图创建适配器
     */
    template<typename GraphType>
    static vector<int> adaptSegmentGraph2D(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        RangeSearchStrategy strategy = RangeSearchStrategy::IN_RANGE_ONLY
    ) {
        RangeFilteringSearchExecutor executor(
            graph.data_wrapper, graph.space_, graph.visited_list_pool_
        );
        
        auto forward_provider = ExtendedRangeSearchFactory::createOneSegmentForwardProvider(
            &graph.forward_nns, query_bound);
        auto reverse_provider = ExtendedRangeSearchFactory::createOneSegmentForwardProvider(
            &graph.reverse_nns, query_bound);
        auto entry_finder = RangeSearchFactory::createStandardEntryPointFinder(graph.data_wrapper);
        
        return executor.execute(search_params, search_info, query, query_bound, 
                              strategy, forward_provider, reverse_provider, entry_finder);
    }
    
    /**
     * @brief 为压缩图创建适配器
     */
    template<typename GraphType>
    static vector<int> adaptCompactGraph(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound,
        RangeSearchStrategy strategy = RangeSearchStrategy::IN_RANGE_ONLY
    ) {
        RangeFilteringSearchExecutor executor(
            graph.data_wrapper, graph.space_, graph.visited_list_pool_
        );
        
        auto forward_provider = ExtendedRangeSearchFactory::createVectorNeighborProvider(
            &graph.forward_neighbors, query_bound);
        auto reverse_provider = ExtendedRangeSearchFactory::createVectorNeighborProvider(
            &graph.reverse_neighbors, query_bound);
        auto entry_finder = RangeSearchFactory::createStandardEntryPointFinder(graph.data_wrapper);
        
        return executor.execute(search_params, search_info, query, query_bound, 
                              strategy, forward_provider, reverse_provider, entry_finder);
    }
};

} // namespace rangeindex
