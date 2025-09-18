#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <functional>

#include "base_hnsw/space_l2.h"
#include "data_wrapper.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

/// 默认参数值
static const unsigned default_K = 16;
static const unsigned default_ef_construction = 400;

/**
 * @brief 搜索策略枚举
 * 定义不同的范围过滤搜索策略
 */
enum class SearchStrategy {
    IN_RANGE_ONLY,      // 仅在范围内搜索
    OUT_BOUND_INCLUDED, // 包含范围外的搜索
    HYBRID,             // 混合策略
    KNN_SEARCH          // 标准K近邻搜索
};

/**
 * @brief 搜索结果结构
 * 封装搜索返回的结果信息
 */
struct SearchResult {
    vector<int> neighbors;              // 邻居节点ID列表
    vector<float> distances;            // 对应距离列表
    size_t comparisons_made = 0;        // 执行的距离比较次数
    double search_time = 0.0;          // 搜索耗时（毫秒）
    
    SearchResult() = default;
    SearchResult(vector<int> nns, vector<float> dists) 
        : neighbors(std::move(nns)), distances(std::move(dists)) {}
};

/**
 * @class BaseIndex
 * @brief 索引基类，提供索引构建和搜索的基本框架。
 */
class BaseIndex {
public:
    BaseIndex(const DataWrapper *data) {
        data_wrapper = data;
    }

    /// 搜索比较次数计数器
    int num_search_comparison = 0;

    /// 图出界计数器
    int k_graph_out_bound = 0;

    /// 日志开关
    bool isLog = true;

    /**
     * @struct IndexParams
     * @brief 索引参数结构体，存储索引构建过程中的配置参数。
     */
    struct IndexParams {
        /// 出度边界 (original params in hnsw, out degree boundary)
        unsigned K = default_K;

        /// 构建效率因子
        unsigned ef_construction = default_ef_construction;

        /// 随机种子
        unsigned random_seed = 100;

        /// TODO: 已废弃参数
        unsigned ef_large_for_pruning = 400;

        /// 最大效率因子
        unsigned ef_max = 2000;

        /// 替代 ef_max
        unsigned ef_construction_2d_max = 2000;

        /// 是否打印每批处理结果
        bool print_one_batch = false;

        /// 构造函数：允许用户自定义参数
        IndexParams(unsigned K, unsigned ef_construction, unsigned ef_large_for_pruning, unsigned ef_max) :
            K(K),
            ef_construction(ef_construction),
            ef_large_for_pruning(ef_large_for_pruning),
            ef_max(ef_max) {}

        /// 递归切分位置类型
        enum Recursion_Type_t {
            MIN_POS,
            MID_POS,
            MAX_POS,
            SMALL_LEFT_POS
        };
        
        Recursion_Type_t recursion_type = Recursion_Type_t::MAX_POS;
        
        /// 默认构造函数
        IndexParams() :
            K(default_K),
            ef_construction(default_ef_construction),
            random_seed(2023) {}
    };

    /**
     * @struct IndexInfo
     * @brief 索引信息结构体，存储索引构建后的统计信息。
     */
    struct IndexInfo {
        /// 索引版本类型
        string index_version_type;

        /// 索引构建耗时
        double index_time = 0.0;

        /// 窗口数量
        size_t window_count = 0;

        /// 节点总数
        size_t nodes_amount = 0;

        /// 平均正向近邻数量
        float avg_forward_nns = 0.0f;

        /// 平均反向近邻数量
        float avg_reverse_nns = 0.0f;
    };

    /**
     * @struct SearchParams
     * @brief 搜索参数结构体，存储搜索过程中的配置参数。
     */
    struct SearchParams {
        /// 查询返回的邻居数量
        unsigned query_K = 10;

        /// 查询效率因子
        unsigned search_ef = 50;

        /// 查询范围（用于范围过滤搜索）
        unsigned query_range = 0;

        /// 批次阈值控制
        float control_batch_threshold = 1.0f;
        
        /// 搜索策略
        SearchStrategy strategy = SearchStrategy::KNN_SEARCH;
        
        /// 最大搜索步数（防止无限搜索）
        unsigned max_search_steps = 1000;
        
        /// 默认构造函数
        SearchParams() = default;
        
        /// 构造函数
        SearchParams(unsigned k, unsigned ef, SearchStrategy strat = SearchStrategy::KNN_SEARCH)
            : query_K(k), search_ef(ef), strategy(strat) {}
    };

    /**
     * @struct SearchInfo
     * @brief 查询信息结构体，记录查询过程中的统计信息和日志。
     */
    struct SearchInfo {
        /// 构造函数：初始化数据包装器、索引参数、方法名称和版本号
        SearchInfo(const DataWrapper *data,
                   const BaseIndex::IndexParams *index_params,
                   const string &meth,
                   const string &ver) {
            data_wrapper = data;
            index = index_params;
            version = ver;
            method = meth;
            path_counter = 0;
            Path(ver + "-" + data->version);
        }

        const DataWrapper *data_wrapper;
        const BaseIndex::IndexParams *index;

        string version;
        string method;

        int index_k = 0;
        double time = 0.0;
        double precision = 0.0;
        double approximate_ratio = 0.0;

        int query_id = 0;
        double internal_search_time = 0.0;   // 单次查询时间
        double fetch_nns_time = 0.0;
        double cal_dist_time = 0.0;
        double other_process_time = 0.0;
        size_t total_comparison = 0;
        size_t path_counter = 0;

        size_t pos_point_traverse_counter = 0;
        size_t pos_point_used_counter = 0;
        size_t neg_point_traverse_counter = 0;
        size_t neg_point_used_counter = 0;
        float total_traversed_nn_amount = 0.0f;

        string investigate_path;
        string save_path;

        bool is_investigate = false;

        void Path(const string &ver) {
            version = ver;
            save_path = "../exp/" + version + "-" + method + "-" + 
                       data_wrapper->dataset + "-" + 
                       std::to_string(data_wrapper->data_size) + ".csv";
        }

        void RecordOneQuery(BaseIndex::SearchParams *search) {
            std::ofstream file;
            file.open(save_path, std::ios_base::app);
            if (file) {
                file << internal_search_time << "," << precision << "," << approximate_ratio
                     << "," << search->query_range << "," << search->search_ef << ","
                     << fetch_nns_time << "," << cal_dist_time << ","
                     << total_comparison << "," << std::to_string(index->recursion_type)
                     << "," << index->K << "," << index->ef_max << ","
                     << index->ef_large_for_pruning << "," << index->ef_construction;
                file << "\n";
            }
            file.close();
        }
    };

    const DataWrapper *data_wrapper;
    SearchInfo *search_info;

    // ========== 核心虚函数接口 ==========

    /**
     * @brief 构建索引
     * @param index_params 索引构建参数
     */
    virtual void buildIndex(const IndexParams *index_params) = 0;

    /**
     * @brief 标准K近邻搜索
     * @param search_params 搜索参数
     * @param search_info 搜索统计信息
     * @param query 查询向量
     * @return 搜索结果
     */
    virtual SearchResult searchKnn(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query) = 0;

    /**
     * @brief 范围内过滤搜索
     * @param search_params 搜索参数
     * @param search_info 搜索统计信息
     * @param query 查询向量
     * @param query_bound 查询范围边界
     * @return 邻居节点ID列表
     */
    virtual vector<int> rangeFilteringSearchInRange(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) = 0;

    /**
     * @brief 范围外过滤搜索
     * @param search_params 搜索参数
     * @param search_info 搜索统计信息
     * @param query 查询向量
     * @param query_bound 查询范围边界
     * @return 邻居节点ID列表
     */
    virtual vector<int> rangeFilteringSearchOutBound(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) = 0;

    /**
     * @brief 统一搜索接口（推荐使用）
     * @param search_params 搜索参数
     * @param search_info 搜索统计信息
     * @param query 查询向量
     * @param query_bound 查询范围边界（可选，用于范围过滤搜索）
     * @return 搜索结果
     */
    virtual SearchResult search(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> *query_bound = nullptr) {
        
        // 根据搜索策略选择相应的搜索方法
        switch (search_params->strategy) {
            case SearchStrategy::KNN_SEARCH:
                return searchKnn(search_params, search_info, query);
                
            case SearchStrategy::IN_RANGE_ONLY:
                if (query_bound) {
                    auto neighbors = rangeFilteringSearchInRange(search_params, search_info, query, *query_bound);
                    return SearchResult(std::move(neighbors), vector<float>());
                }
                break;
                
            case SearchStrategy::OUT_BOUND_INCLUDED:
                if (query_bound) {
                    auto neighbors = rangeFilteringSearchOutBound(search_params, search_info, query, *query_bound);
                    return SearchResult(std::move(neighbors), vector<float>());
                }
                break;
                
            default:
                break;
        }
        
        // 默认回退到KNN搜索
        return searchKnn(search_params, search_info, query);
    }

    // ========== 持久化接口 ==========

    /**
     * @brief 保存索引到文件
     * @param file_path 文件路径
     */
    virtual void save(const string &file_path) = 0;

    /**
     * @brief 从文件加载索引
     * @param file_path 文件路径
     */
    virtual void load(const string &file_path) = 0;

    // ========== 辅助方法 ==========

    /**
     * @brief 获取索引信息
     * @return 索引统计信息
     */
    virtual IndexInfo getIndexInfo() const {
        IndexInfo info;
        info.nodes_amount = data_wrapper ? data_wrapper->data_size : 0;
        return info;
    }

    /**
     * @brief 重置搜索计数器
     */
    virtual void resetSearchCounters() {
        num_search_comparison = 0;
        k_graph_out_bound = 0;
    }

    /**
     * @brief 虚析构函数
     */
    virtual ~BaseIndex() = default;
};