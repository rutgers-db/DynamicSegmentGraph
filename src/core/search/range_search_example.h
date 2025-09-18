/**
 * @file range_search_example.h
 * @brief 统一范围过滤搜索接口的使用示例
 * @author DSG Team
 * @date 2024
 * 
 * 本文件展示了如何使用统一的范围过滤搜索接口来重构现有的各种图结构中的
 * rangeFilteringSearch* 方法。
 */

#pragma once

#include "range_search.h"
#include "algorithms/recursion_batch.h"

namespace rangeindex {

/**
 * @brief 示例：如何在现有图结构中集成统一接口
 * 
 * 这个示例展示了如何将现有的 rangeFilteringSearchInRange 方法
 * 重构为使用统一的接口实现。
 */

// 1. 原始的 rangeFilteringSearchInRange 方法（以递归批处理HNSW为例）
// 可以被替换为：

template<typename dist_t>
vector<int> RangeFilteringHNSW<dist_t>::rangeFilteringSearchInRange_NEW(
    const BaseIndex::SearchParams* search_params,
    BaseIndex::SearchInfo* search_info,
    const vector<float>& query,
    const pair<int, int>& query_bound) {
    
    // 使用统一接口的新实现
    return RangeSearchAdapter::adaptRecursionBatchHNSW(
        *this, search_params, search_info, query, query_bound,
        RangeSearchStrategy::IN_RANGE_ONLY
    );
}

// 2. 原始的 rangeFilteringSearchOutBound 方法可以被替换为：

template<typename dist_t>
vector<int> RangeFilteringHNSW<dist_t>::rangeFilteringSearchOutBound_NEW(
    const BaseIndex::SearchParams* search_params,
    BaseIndex::SearchInfo* search_info,
    const vector<float>& query,
    const pair<int, int>& query_bound) {
    
    // 使用统一接口的新实现，策略为包含范围外搜索
    return RangeSearchAdapter::adaptRecursionBatchHNSW(
        *this, search_params, search_info, query, query_bound,
        RangeSearchStrategy::OUT_BOUND_INCLUDED
    );
}

/**
 * @brief 自定义搜索策略示例
 * 
 * 展示如何创建自定义的搜索组件来满足特殊需求
 */
class CustomRangeSearchExample {
public:
    /**
     * @brief 自定义入口点查找策略
     * 
     * 这个示例展示如何创建自定义的入口点查找逻辑
     */
    static function<vector<int>(const pair<int, int>&)> 
    createCustomEntryPointFinder(const DataWrapper* data_wrapper, int max_entries = 5) {
        return [data_wrapper, max_entries](const pair<int, int>& query_bound) -> vector<int> {
            vector<int> entries;
            int lbound = query_bound.first;
            int rbound = query_bound.second;
            int range_size = rbound - lbound + 1;
            
            // 自定义策略：根据范围大小动态调整入口点数量
            int num_entries = std::min(max_entries, std::max(1, range_size / 100));
            
            for (int i = 0; i < num_entries; ++i) {
                int entry = lbound + (i * range_size) / num_entries;
                entries.push_back(entry);
            }
            
            return entries;
        };
    }
    
    /**
     * @brief 自定义搜索实现示例
     * 
     * 展示如何直接使用核心组件来创建完全自定义的搜索逻辑
     */
    template<typename GraphType>
    static vector<int> customRangeSearch(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound
    ) {
        RangeFilteringSearchExecutor executor(
            graph.data_wrapper, graph.space_, graph.visited_list_pool_
        );
        
        // 使用自定义入口点查找器
        auto custom_entry_finder = createCustomEntryPointFinder(graph.data_wrapper, 7);
        
        // 使用标准的邻居提供者
        auto forward_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.forward_nns_batches, query_bound);
        auto reverse_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.reverse_nns_batches, query_bound);
        
        // 执行搜索
        return executor.execute(
            search_params, search_info, query, query_bound,
            RangeSearchStrategy::HYBRID,  // 使用混合策略
            forward_provider, reverse_provider, custom_entry_finder
        );
    }
};

/**
 * @brief 性能优化示例
 * 
 * 展示如何通过不同的配置来优化搜索性能
 */
class PerformanceOptimizationExample {
public:
    /**
     * @brief 高性能搜索配置
     * 
     * 为大规模数据集优化的搜索配置
     */
    template<typename GraphType>
    static vector<int> highPerformanceSearch(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound
    ) {
        RangeFilteringSearchExecutor executor(
            graph.data_wrapper, graph.space_, graph.visited_list_pool_
        );
        
        // 优化的入口点策略：只使用单个中心点
        auto optimized_entry_finder = [](const pair<int, int>& bound) -> vector<int> {
            return {bound.first + (bound.second - bound.first) / 2};
        };
        
        // 只使用前向邻居以减少计算开销
        auto forward_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.forward_nns_batches, query_bound);
        
        return executor.execute(
            search_params, search_info, query, query_bound,
            RangeSearchStrategy::IN_RANGE_ONLY,
            forward_provider, nullptr, optimized_entry_finder
        );
    }
    
    /**
     * @brief 高精度搜索配置
     * 
     * 为高精度要求优化的搜索配置
     */
    template<typename GraphType>
    static vector<int> highAccuracySearch(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound
    ) {
        RangeFilteringSearchExecutor executor(
            graph.data_wrapper, graph.space_, graph.visited_list_pool_
        );
        
        // 高精度入口点策略：使用更多入口点
        auto high_accuracy_entry_finder = [](const pair<int, int>& bound) -> vector<int> {
            vector<int> entries;
            int range_size = bound.second - bound.first + 1;
            int num_entries = std::min(10, range_size);  // 最多10个入口点
            
            for (int i = 0; i < num_entries; ++i) {
                entries.push_back(bound.first + (i * range_size) / num_entries);
            }
            return entries;
        };
        
        // 使用前向和反向邻居
        auto forward_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.forward_nns_batches, query_bound);
        auto reverse_provider = ExtendedRangeSearchFactory::createOneBatchForwardProvider(
            &graph.reverse_nns_batches, query_bound);
        
        return executor.execute(
            search_params, search_info, query, query_bound,
            RangeSearchStrategy::OUT_BOUND_INCLUDED,  // 包含范围外搜索以提高精度
            forward_provider, reverse_provider, high_accuracy_entry_finder
        );
    }
};

/**
 * @brief 迁移指南
 * 
 * 展示如何将现有代码迁移到新的统一接口
 */
class MigrationGuide {
public:
    /**
     * @brief 迁移步骤说明
     * 
     * 1. 识别现有的 rangeFilteringSearch* 方法
     * 2. 确定使用的邻居数据结构类型（OneBatchNeighbors, OneSegmentNeighbors等）
     * 3. 选择适当的适配器方法
     * 4. 替换原有实现
     * 
     * 迁移前：
     * ```cpp
     * vector<int> MyGraph::rangeFilteringSearchInRange(...) {
     *     // 200+ 行的复杂实现
     *     // 包含重复的初始化、搜索、结果构建逻辑
     * }
     * ```
     * 
     * 迁移后：
     * ```cpp
     * vector<int> MyGraph::rangeFilteringSearchInRange(...) {
     *     return RangeSearchAdapter::adaptMyGraphType(
     *         *this, search_params, search_info, query, query_bound
     *     );
     * }
     * ```
     * 
     * 好处：
     * - 代码量减少90%以上
     * - 统一的性能优化
     * - 更好的可维护性
     * - 统一的错误处理
     * - 更容易添加新功能
     */
    
    /**
     * @brief 向后兼容性支持
     * 
     * 如果需要保持向后兼容性，可以保留原有方法名，内部调用新实现
     */
    template<typename GraphType>
    static vector<int> legacyRangeFilteringSearchInRange(
        GraphType& graph,
        const BaseIndex::SearchParams* search_params,
        BaseIndex::SearchInfo* search_info,
        const vector<float>& query,
        const pair<int, int>& query_bound
    ) {
        // 内部调用新的统一实现
        return RangeSearchAdapter::adaptRecursionBatchHNSW(
            graph, search_params, search_info, query, query_bound,
            RangeSearchStrategy::IN_RANGE_ONLY
        );
    }
};

} // namespace rangeindex
