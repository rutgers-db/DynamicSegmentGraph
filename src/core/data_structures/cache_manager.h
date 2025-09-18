/**
 * @file cache_manager.h
 * @author DSG Team
 * @brief 高效缓存管理器 - 替代低效的unordered_map缓存
 * 
 * 这个文件实现了DSG算法中支配关系计算的高效缓存机制，用于：
 * 1. 替代compact_graph.h中低效的unordered_map<unsigned, bool>缓存
 * 2. 提供更高效的缓存访问性能，减少哈希计算开销
 * 3. 优化内存使用，减少内存分配和碎片化
 * 4. 支持批量清理和容量管理
 * 
 * 原始问题分析（来自compact_graph.h第298行）：
 * - std::unordered_map<unsigned, bool> calculated_pair 存在性能瓶颈
 * - 频繁的哈希计算和内存分配影响DFS搜索效率
 * - 缓存命中率虽高，但访问开销大
 * 
 * 优化策略：
 * 1. 使用线性探测哈希表替代std::unordered_map
 * 2. 预分配连续内存，减少内存碎片
 * 3. 优化哈希函数，针对点对编码特性设计
 * 4. 提供高效的批量操作接口
 * 
 * @date 2023-12-15
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <iostream>

/**
 * @brief 支配关系缓存条目
 * 
 * 存储点对的支配关系计算结果，使用紧凑的内存布局
 */
struct DominationCacheEntry {
    uint32_t encoded_pair;  ///< 编码的点对：(pre_nb_idx << 16) + i
    bool domination_result; ///< 支配关系结果：true表示支配，false表示不支配
    bool is_valid;          ///< 条目是否有效（用于线性探测）
    
    DominationCacheEntry() : encoded_pair(0), domination_result(false), is_valid(false) {}
    
    DominationCacheEntry(uint32_t pair, bool result) 
        : encoded_pair(pair), domination_result(result), is_valid(true) {}
};

/**
 * @brief 高效的支配关系缓存管理器
 * 
 * 使用线性探测哈希表实现的高效缓存，专门优化了点对支配关系的存储和查询。
 * 相比std::unordered_map，具有以下优势：
 * 1. 更少的内存分配和释放
 * 2. 更好的缓存局部性
 * 3. 更快的插入和查询速度
 * 4. 可预测的性能表现
 */
class DominationCache {
private:
    std::vector<DominationCacheEntry> cache_;  ///< 缓存存储数组
    size_t capacity_;                          ///< 缓存容量（必须是2的幂）
    size_t size_;                              ///< 当前存储的条目数
    size_t mask_;                              ///< 位掩码，用于快速取模：mask_ = capacity_ - 1
    
    // 性能统计（可选，用于调试）
    mutable size_t hit_count_;                 ///< 缓存命中次数
    mutable size_t miss_count_;                ///< 缓存未命中次数
    mutable size_t collision_count_;           ///< 哈希冲突次数

    /**
     * @brief 专门优化的哈希函数
     * 
     * 针对点对编码 (pre_nb_idx << 16) + i 的特性设计的哈希函数。
     * 使用乘法哈希和位移操作，避免昂贵的除法运算。
     * 
     * @param encoded_pair 编码的点对
     * @return size_t 哈希值
     */
    inline size_t hash_function(uint32_t encoded_pair) const {
        // 使用乘法哈希：h(k) = (a * k) >> (32 - log2(capacity))
        // 选择黄金比例的近似值作为乘数
        constexpr uint32_t HASH_MULTIPLIER = 2654435769U; // 2^32 / φ
        return ((encoded_pair * HASH_MULTIPLIER) >> (32 - __builtin_ctz(capacity_))) & mask_;
    }

    /**
     * @brief 线性探测查找下一个可用位置
     * 
     * @param start_pos 起始位置
     * @return size_t 下一个可用位置的索引
     */
    inline size_t find_next_slot(size_t start_pos) const {
        size_t pos = start_pos;
        while (cache_[pos].is_valid) {
            pos = (pos + 1) & mask_;  // 线性探测，使用位运算快速取模
            if (pos == start_pos) {   // 表已满
                return capacity_;     // 返回无效位置
            }
        }
        return pos;
    }

    /**
     * @brief 扩容缓存（当负载因子过高时）
     * 
     * 将缓存容量翻倍，并重新哈希所有条目
     */
    void resize() {
        size_t old_capacity = capacity_;
        auto old_cache = std::move(cache_);
        
        // 扩容到原来的2倍
        capacity_ *= 2;
        mask_ = capacity_ - 1;
        cache_.clear();
        cache_.resize(capacity_);
        size_ = 0;
        
        // 重新插入所有有效条目
        for (const auto& entry : old_cache) {
            if (entry.is_valid) {
                insert_internal(entry.encoded_pair, entry.domination_result);
            }
        }
    }

    /**
     * @brief 内部插入方法（不检查负载因子）
     * 
     * @param encoded_pair 编码的点对
     * @param domination_result 支配关系结果
     */
    void insert_internal(uint32_t encoded_pair, bool domination_result) {
        size_t pos = hash_function(encoded_pair);
        
        // 线性探测找到合适位置
        while (cache_[pos].is_valid) {
            if (cache_[pos].encoded_pair == encoded_pair) {
                // 更新已存在的条目
                cache_[pos].domination_result = domination_result;
                return;
            }
            pos = (pos + 1) & mask_;
            ++collision_count_;
        }
        
        // 插入新条目
        cache_[pos] = DominationCacheEntry(encoded_pair, domination_result);
        ++size_;
    }

public:
    /**
     * @brief 构造函数
     * 
     * @param initial_capacity 初始容量（会调整为最近的2的幂）
     * @param max_load_factor 最大负载因子（默认0.75）
     */
    explicit DominationCache(size_t initial_capacity = 1024) 
        : capacity_(next_power_of_two(initial_capacity))
        , size_(0)
        , mask_(capacity_ - 1)
        , hit_count_(0)
        , miss_count_(0)
        , collision_count_(0) {
        cache_.resize(capacity_);
    }

    /**
     * @brief 获取缓存中的支配关系结果
     * 
     * @param encoded_pair 编码的点对
     * @param result 输出参数，存储查找到的结果
     * @return true 缓存命中
     * @return false 缓存未命中
     */
    inline bool get(uint32_t encoded_pair, bool& result) const {
        size_t pos = hash_function(encoded_pair);
        size_t start_pos = pos;
        
        do {
            if (!cache_[pos].is_valid) {
                // 空位置，说明不存在
                ++miss_count_;
                return false;
            }
            
            if (cache_[pos].encoded_pair == encoded_pair) {
                // 找到匹配的条目
                result = cache_[pos].domination_result;
                ++hit_count_;
                return true;
            }
            
            // 线性探测下一个位置
            pos = (pos + 1) & mask_;
        } while (pos != start_pos);
        
        // 遍历了整个表都没找到
        ++miss_count_;
        return false;
    }

    /**
     * @brief 设置支配关系结果到缓存
     * 
     * @param encoded_pair 编码的点对
     * @param domination_result 支配关系结果
     */
    inline void set(uint32_t encoded_pair, bool domination_result) {
        // 检查是否需要扩容（负载因子 > 0.75）
        if (size_ * 4 >= capacity_ * 3) {
            resize();
        }
        
        insert_internal(encoded_pair, domination_result);
    }

    /**
     * @brief 清空缓存
     * 
     * 快速清空所有缓存条目，为新一轮DFS搜索做准备
     */
    inline void clear() {
        std::fill(cache_.begin(), cache_.end(), DominationCacheEntry());
        size_ = 0;
        // 注意：不重置统计信息，便于性能分析
    }

    /**
     * @brief 获取缓存使用统计信息
     * 
     * @return 包含命中率、冲突率等信息的结构体
     */
    struct CacheStats {
        size_t capacity;
        size_t size;
        size_t hit_count;
        size_t miss_count;
        size_t collision_count;
        double hit_rate;
        double load_factor;
        double collision_rate;
    };

    CacheStats get_stats() const {
        size_t total_access = hit_count_ + miss_count_;
        return CacheStats{
            capacity_,
            size_,
            hit_count_,
            miss_count_,
            collision_count_,
            total_access > 0 ? static_cast<double>(hit_count_) / total_access : 0.0,
            static_cast<double>(size_) / capacity_,
            size_ > 0 ? static_cast<double>(collision_count_) / size_ : 0.0
        };
    }

    /**
     * @brief 重置性能统计
     */
    void reset_stats() {
        hit_count_ = 0;
        miss_count_ = 0;
        collision_count_ = 0;
    }

    /**
     * @brief 预留缓存容量
     * 
     * @param expected_size 预期的条目数量
     */
    void reserve(size_t expected_size) {
        size_t required_capacity = next_power_of_two(expected_size * 4 / 3); // 考虑负载因子
        if (required_capacity > capacity_) {
            capacity_ = required_capacity;
            mask_ = capacity_ - 1;
            cache_.clear();
            cache_.resize(capacity_);
            size_ = 0;
        }
    }

    /**
     * @brief 获取当前缓存大小
     */
    size_t size() const { return size_; }

    /**
     * @brief 获取缓存容量
     */
    size_t capacity() const { return capacity_; }

    /**
     * @brief 检查缓存是否为空
     */
    bool empty() const { return size_ == 0; }

private:
    /**
     * @brief 计算大于等于n的最小2的幂
     */
    static size_t next_power_of_two(size_t n) {
        if (n <= 1) return 2;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }
};

/**
 * @brief 简化的缓存接口 - 与原始unordered_map接口兼容
 * 
 * 提供与std::unordered_map类似的接口，便于从原有代码迁移
 */
class CompatibleDominationCache {
private:
    DominationCache cache_;

public:
    explicit CompatibleDominationCache(size_t initial_capacity = 1024) 
        : cache_(initial_capacity) {}

    /**
     * @brief 检查键是否存在（模拟unordered_map::find）
     */
    bool find(uint32_t encoded_pair) const {
        bool result;
        return cache_.get(encoded_pair, result);
    }

    /**
     * @brief 获取值（模拟unordered_map::operator[]）
     */
    bool get(uint32_t encoded_pair) const {
        bool result;
        if (cache_.get(encoded_pair, result)) {
            return result;
        }
        // 如果不存在，返回默认值
        return false;
    }

    /**
     * @brief 设置值（模拟unordered_map::operator[]）
     */
    void set(uint32_t encoded_pair, bool value) {
        cache_.set(encoded_pair, value);
    }

    /**
     * @brief 清空缓存
     */
    void clear() {
        cache_.clear();
    }

    /**
     * @brief 获取统计信息
     */
    auto get_stats() const {
        return cache_.get_stats();
    }
};

/**
 * @brief 用于compact_graph.h的直接替换宏
 * 
 * 可以通过简单的宏定义实现无缝替换：
 * 
 * 原代码：
 * std::unordered_map<unsigned, bool> calculated_pair;
 * if (calculated_pair.find(encoded_pair) != calculated_pair.end()) {
 *     const auto &domination_result = calculated_pair[encoded_pair];
 * }
 * calculated_pair[encoded_pair] = domination_result;
 * calculated_pair.clear();
 * 
 * 替换后：
 * CompatibleDominationCache calculated_pair;
 * if (calculated_pair.find(encoded_pair)) {
 *     const auto domination_result = calculated_pair.get(encoded_pair);
 * }
 * calculated_pair.set(encoded_pair, domination_result);
 * calculated_pair.clear();
 */

// 性能测试和调试工具
namespace CacheManagerUtils {
    
    /**
     * @brief 打印缓存性能统计
     */
    inline void print_cache_stats(const DominationCache::CacheStats& stats) {
        std::cout << "=== 缓存性能统计 ===" << std::endl;
        std::cout << "容量: " << stats.capacity << std::endl;
        std::cout << "已用: " << stats.size << std::endl;
        std::cout << "负载因子: " << stats.load_factor * 100 << "%" << std::endl;
        std::cout << "命中次数: " << stats.hit_count << std::endl;
        std::cout << "未命中次数: " << stats.miss_count << std::endl;
        std::cout << "命中率: " << stats.hit_rate * 100 << "%" << std::endl;
        std::cout << "冲突次数: " << stats.collision_count << std::endl;
        std::cout << "冲突率: " << stats.collision_rate * 100 << "%" << std::endl;
        std::cout << "===================" << std::endl;
    }

    /**
     * @brief 基准测试：比较新旧缓存性能
     */
    void benchmark_cache_performance(size_t test_size = 100000);
}
