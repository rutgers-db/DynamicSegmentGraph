
#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include "infrastructure/utils/distance.h"
#include "infrastructure/io/data_loader.h"
#include "infrastructure/utils/utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

/// 默认参数值
static const unsigned default_K = 16;
static const unsigned default_ef_construction = 400;

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
    int num_search_comparison;

    /// 图出界计数器
    int k_graph_out_bound;

    bool isLog = true;

    /**
     * @struct IndexParams
     * @brief 索引参数结构体，存储索引构建过程中的配置参数。
     */
    struct IndexParams {
        /// 出度边界  original params in hnsw，out degree boundry
        unsigned K;

        /// 构建效率因子
        unsigned ef_construction = 400;
        ;

        /// 随机种子
        unsigned random_seed = 100;

        ///  TODO: Depratched parameter
        unsigned ef_large_for_pruning = 400;

        /// 最大效率因子
        unsigned ef_max = 2000;

        /// Replace ef_max
        unsigned ef_construction_2d_max;

        /// 是否打印每批处理结果
        bool print_one_batch = false;

        /// 构造函数：允许用户自定义参数
        IndexParams(unsigned K, unsigned ef_construction, unsigned ef_large_for_pruning, unsigned ef_max) :
            K(K),
            ef_construction(ef_construction),
            ef_large_for_pruning(ef_large_for_pruning),
            ef_max(ef_max){};

        // which position to cut during the recursion
        enum Recursion_Type_t {
            MIN_POS,
            MID_POS,
            MAX_POS,
            SMALL_LEFT_POS
        };
        Recursion_Type_t recursion_type = Recursion_Type_t::MAX_POS;
        IndexParams() :
            K(default_K),
            ef_construction(default_ef_construction),
            random_seed(2023) {
        }
    };

    struct IndexInfo {
        /// 索引版本类型
        string index_version_type;

        /// 索引构建耗时
        double index_time;

        /// 窗口数量
        size_t window_count;

        /// 节点总数
        size_t nodes_amount;

        /// 平均正向近邻数量
        float avg_forward_nns;

        /// 平均反向近邻数量
        float avg_reverse_nns;
    };

    const DataWrapper *data_wrapper;

    SearchInfo *search_info;

    virtual void buildIndex(const IndexParams *index_params) = 0;

    virtual vector<int> rangeFilteringSearchInRange(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) = 0;

    virtual vector<int> rangeFilteringSearchOutBound(
        const SearchParams *search_params,
        SearchInfo *search_info,
        const vector<float> &query,
        const std::pair<int, int> query_bound) = 0;
    virtual ~BaseIndex() {
    }

    virtual void save(const string &file_path) = 0;
    virtual void load(const string &file_path) = 0;
};
