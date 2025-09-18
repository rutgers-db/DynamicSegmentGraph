
/**
 * @file compressed_point.h
 * @author DSG Team
 * @brief 压缩点结构实现 - DSG算法的核心数据结构
 * 
 * 这个文件实现了动态段图(DSG)算法中的压缩点结构，用于：
 * 1. 优化2D范围查询的性能
 * 2. 减少存储空间占用
 * 3. 支持快速的范围判断操作
 * 4. 提供高效的点比较和排序功能
 * 
 * 压缩点的核心思想是将原始的2D坐标点压缩为四个边界值：
 * - ll, lr: 左边界的范围 [ll, lr]
 * - rl, rr: 右边界的范围 [rl, rr]
 * 
 * 这种设计使得范围查询可以通过简单的边界比较来快速确定点是否在查询范围内。
 * 
 * @date 2023-12-15
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

/**
 * @brief 压缩点结构 - DSG算法的核心数据结构
 * 
 * 压缩点将2D空间中的点表示为四个边界值，支持高效的范围查询操作。
 * 每个压缩点包含：
 * - external_id: 外部节点ID，用于标识原始数据点
 * - ll, lr: 左边界范围 [ll, lr]，表示该点在左边界上的投影范围
 * - rl, rr: 右边界范围 [rl, rr]，表示该点在右边界上的投影范围
 * 
 * 这种表示方式使得范围查询 [query_L, query_R] 可以通过简单的区间重叠判断来实现：
 * 点在查询范围内 ⟺ [ll, lr] ∩ [query_L, query_L] ≠ ∅ AND [rl, rr] ∩ [query_R, query_R] ≠ ∅
 */
struct CompressedPoint {
    /**
     * @brief 构造函数 - 创建压缩点
     * @param _external_id 外部节点ID
     * @param _ll 左边界起始位置
     * @param _lr 左边界结束位置
     * @param _rl 右边界起始位置
     * @param _rr 右边界结束位置
     * 
     * @pre _ll <= _lr && _rl <= _rr (边界范围必须有效)
     * @pre _external_id 应该是有效的节点ID
     */
    CompressedPoint(unsigned _external_id, unsigned _ll, unsigned _lr, unsigned _rl, unsigned _rr) :
        external_id(_external_id), ll(_ll), lr(_lr), rl(_rl), rr(_rr) {
        // 验证输入参数的有效性
        assert(_ll <= _lr && "左边界范围无效: ll > lr");
        assert(_rl <= _rr && "右边界范围无效: rl > rr");
    }

    /**
     * @brief 默认构造函数 - 创建无效的压缩点
     * 
     * 注意：默认构造的压缩点需要后续初始化才能使用
     */
    CompressedPoint() : external_id(0), ll(0), lr(0), rl(0), rr(0) {
    }

    // 成员变量
    unsigned external_id;  ///< 外部节点ID，用于标识原始数据点
    unsigned ll, lr;       ///< 左边界范围 [ll, lr]
    unsigned rl, rr;       ///< 右边界范围 [rl, rr]

    /**
     * @brief 检查点是否在压缩范围内
     * 
     * 判断给定的查询范围 [query_L, query_R] 是否与当前压缩点的范围重叠。
     * 重叠条件：左边界重叠 AND 右边界重叠
     * 
     * @param query_L 查询左边界
     * @param query_R 查询右边界
     * @return true 如果点在查询范围内
     * @return false 如果点不在查询范围内
     * 
     * @note 这是DSG算法中最核心的范围判断函数，性能要求极高
     */
    inline bool const if_in_compressed_range(const unsigned &query_L, const unsigned &query_R) const {
        return ((ll <= query_L && query_L <= lr) && (rl <= query_R && query_R <= rr));
    }

    /**
     * @brief 检查范围是否重叠（更通用的版本）
     * 
     * 判断查询范围 [query_L, query_R] 是否与压缩点的范围有任何重叠
     * 
     * @param query_L 查询范围起始
     * @param query_R 查询范围结束
     * @return true 如果有重叠
     * @return false 如果没有重叠
     */
    inline bool intersects_with_range(const unsigned &query_L, const unsigned &query_R) const {
        // 检查左边界重叠: [ll, lr] ∩ [query_L, query_R] ≠ ∅
        bool left_overlap = (ll <= query_R && query_L <= lr);
        // 检查右边界重叠: [rl, rr] ∩ [query_L, query_R] ≠ ∅  
        bool right_overlap = (rl <= query_R && query_L <= rr);
        return left_overlap && right_overlap;
    }

    /**
     * @brief 比较运算符 - 用于排序和搜索
     * 
     * 按照external_id进行排序，这样可以保持与原始数据的一致性
     * 
     * @param other 另一个压缩点
     * @return true 如果当前点的ID小于other的ID
     */
    bool operator<(const CompressedPoint &other) const {
        return this->external_id < other.external_id;
    }

    /**
     * @brief 相等比较运算符
     * 
     * @param other 另一个压缩点
     * @return true 如果两个点完全相同
     */
    bool operator==(const CompressedPoint &other) const {
        return external_id == other.external_id && 
               ll == other.ll && lr == other.lr && 
               rl == other.rl && rr == other.rr;
    }

    /**
     * @brief 不等比较运算符
     */
    bool operator!=(const CompressedPoint &other) const {
        return !(*this == other);
    }

    /**
     * @brief 验证压缩点的有效性
     * 
     * @return true 如果压缩点的所有边界都是有效的
     */
    bool is_valid() const {
        return (ll <= lr) && (rl <= rr);
    }

    /**
     * @brief 获取左边界范围的大小
     * 
     * @return unsigned 左边界范围大小 (lr - ll + 1)
     */
    unsigned get_left_range_size() const {
        return lr - ll + 1;
    }

    /**
     * @brief 获取右边界范围的大小
     * 
     * @return unsigned 右边界范围大小 (rr - rl + 1)
     */
    unsigned get_right_range_size() const {
        return rr - rl + 1;
    }

    /**
     * @brief 获取总的覆盖范围大小
     * 
     * @return unsigned 总覆盖范围 (左边界范围 * 右边界范围)
     */
    unsigned get_total_coverage() const {
        return get_left_range_size() * get_right_range_size();
    }

    /**
     * @brief 输出流运算符重载 - 用于调试
     * 
     * @param os 输出流
     * @param point 压缩点
     * @return std::ostream& 输出流引用
     */
    friend std::ostream& operator<<(std::ostream& os, const CompressedPoint& point) {
        os << "CompressedPoint{id=" << point.external_id 
           << ", left=[" << point.ll << "," << point.lr << "]"
           << ", right=[" << point.rl << "," << point.rr << "]}";
        return os;
    }
};

/**
 * @brief 压缩点集合的工具函数
 */
namespace CompressedPointUtils {
    
    /**
     * @brief 按external_id排序压缩点集合
     * 
     * @param points 压缩点集合
     */
    inline void sort_by_id(std::vector<CompressedPoint>& points) {
        std::sort(points.begin(), points.end());
    }

    /**
     * @brief 在排序的压缩点集合中查找指定ID的点
     * 
     * @param points 已排序的压缩点集合
     * @param external_id 要查找的外部ID
     * @return 指向找到的点的迭代器，如果未找到则返回end()
     */
    inline std::vector<CompressedPoint>::const_iterator find_by_id(
        const std::vector<CompressedPoint>& points, unsigned external_id) {
        CompressedPoint target(external_id, 0, 0, 0, 0);
        return std::lower_bound(points.begin(), points.end(), target);
    }

    /**
     * @brief 过滤在指定范围内的压缩点
     * 
     * @param points 压缩点集合
     * @param query_L 查询左边界
     * @param query_R 查询右边界
     * @return std::vector<CompressedPoint> 在范围内的点集合
     */
    inline std::vector<CompressedPoint> filter_in_range(
        const std::vector<CompressedPoint>& points, 
        unsigned query_L, unsigned query_R) {
        std::vector<CompressedPoint> result;
        result.reserve(points.size() / 4); // 预估结果大小
        
        for (const auto& point : points) {
            if (point.if_in_compressed_range(query_L, query_R)) {
                result.push_back(point);
            }
        }
        return result;
    }

    /**
     * @brief 统计在指定范围内的压缩点数量
     * 
     * @param points 压缩点集合
     * @param query_L 查询左边界
     * @param query_R 查询右边界
     * @return size_t 在范围内的点数量
     */
    inline size_t count_in_range(
        const std::vector<CompressedPoint>& points, 
        unsigned query_L, unsigned query_R) {
        size_t count = 0;
        for (const auto& point : points) {
            if (point.if_in_compressed_range(query_L, query_R)) {
                ++count;
            }
        }
        return count;
    }
}