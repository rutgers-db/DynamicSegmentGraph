/**
 * @file memory_pool.h
 * @brief 高性能内存池实现，解决vector<vector<float>>内存碎片化问题
 * 
 * 本文件实现了针对DynamicSegmentGraph项目的高性能内存管理解决方案：
 * 1. 连续内存布局替代嵌套vector结构
 * 2. 对齐分配器提高缓存命中率
 * 3. 内存池复用减少分配开销
 * 4. 优化的缓存机制
 * 
 * @date 2024-01-15
 * @copyright Copyright (c) 2024
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <new>

namespace DSG {
namespace Infrastructure {
namespace Memory {

/**
 * @brief 内存对齐工具类
 */
class AlignUtils {
public:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t DEFAULT_ALIGNMENT = 16;
    
    /**
     * @brief 计算对齐后的地址
     * @param ptr 原始指针
     * @param alignment 对齐字节数
     * @return 对齐后的指针
     */
    static inline void* align_ptr(void* ptr, size_t alignment) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<void*>(aligned);
    }
    
    /**
     * @brief 计算对齐所需的总大小
     * @param size 原始大小
     * @param alignment 对齐字节数
     * @return 对齐后的总大小
     */
    static inline size_t align_size(size_t size, size_t alignment) {
        return size + alignment - 1;
    }
};

/**
 * @brief 高性能对齐分配器
 * @tparam T 数据类型
 * @tparam Alignment 对齐字节数，默认为缓存行大小
 */
template<typename T, size_t Alignment = AlignUtils::CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    AlignedAllocator() = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}
    
    pointer allocate(size_type n) {
        if (n > max_size()) {
            throw std::bad_alloc();
        }
        
        size_t total_size = n * sizeof(T);
        size_t aligned_size = AlignUtils::align_size(total_size, Alignment);
        
        void* raw_ptr = std::aligned_alloc(Alignment, aligned_size);
        if (!raw_ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(raw_ptr);
    }
    
    void deallocate(pointer ptr, size_type) {
        std::free(ptr);
    }
    
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
    
    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};

/**
 * @brief 连续内存存储类，替代vector<vector<float>>
 * 
 * 使用连续内存布局存储多维数据，提高缓存命中率和内存访问效率
 */
class ContinuousDataStorage {
public:
    using value_type = float;
    using pointer = float*;
    using const_pointer = const float*;
    using reference = float&;
    using const_reference = const float&;
    using size_type = size_t;
    
    /**
     * @brief 构造函数
     * @param num_points 数据点数量
     * @param dimension 数据维度
     */
    ContinuousDataStorage(size_type num_points, size_type dimension)
        : num_points_(num_points), dimension_(dimension), 
          total_size_(num_points * dimension) {
        if (total_size_ > 0) {
            data_ = allocator_.allocate(total_size_);
            std::fill(data_, data_ + total_size_, 0.0f);
        }
    }
    
    /**
     * @brief 析构函数
     */
    ~ContinuousDataStorage() {
        if (data_) {
            allocator_.deallocate(data_, total_size_);
        }
    }
    
    // 禁用拷贝构造和赋值
    ContinuousDataStorage(const ContinuousDataStorage&) = delete;
    ContinuousDataStorage& operator=(const ContinuousDataStorage&) = delete;
    
    // 支持移动构造和赋值
    ContinuousDataStorage(ContinuousDataStorage&& other) noexcept
        : data_(other.data_), num_points_(other.num_points_), 
          dimension_(other.dimension_), total_size_(other.total_size_) {
        other.data_ = nullptr;
        other.num_points_ = 0;
        other.dimension_ = 0;
        other.total_size_ = 0;
    }
    
    ContinuousDataStorage& operator=(ContinuousDataStorage&& other) noexcept {
        if (this != &other) {
            if (data_) {
                allocator_.deallocate(data_, total_size_);
            }
            data_ = other.data_;
            num_points_ = other.num_points_;
            dimension_ = other.dimension_;
            total_size_ = other.total_size_;
            
            other.data_ = nullptr;
            other.num_points_ = 0;
            other.dimension_ = 0;
            other.total_size_ = 0;
        }
        return *this;
    }
    
    /**
     * @brief 获取指定点的数据指针
     * @param point_idx 点索引
     * @return 指向该点数据的指针
     */
    pointer get_point(size_type point_idx) {
        return data_ + point_idx * dimension_;
    }
    
    const_pointer get_point(size_type point_idx) const {
        return data_ + point_idx * dimension_;
    }
    
    /**
     * @brief 获取指定点的指定维度值
     * @param point_idx 点索引
     * @param dim_idx 维度索引
     * @return 该点的指定维度值
     */
    reference operator()(size_type point_idx, size_type dim_idx) {
        return data_[point_idx * dimension_ + dim_idx];
    }
    
    const_reference operator()(size_type point_idx, size_type dim_idx) const {
        return data_[point_idx * dimension_ + dim_idx];
    }
    
    /**
     * @brief 设置指定点的数据
     * @param point_idx 点索引
     * @param data 数据指针
     */
    void set_point(size_type point_idx, const float* data) {
        std::memcpy(get_point(point_idx), data, dimension_ * sizeof(float));
    }
    
    /**
     * @brief 获取数据点数量
     */
    size_type num_points() const { return num_points_; }
    
    /**
     * @brief 获取数据维度
     */
    size_type dimension() const { return dimension_; }
    
    /**
     * @brief 获取总数据大小
     */
    size_type total_size() const { return total_size_; }
    
    /**
     * @brief 获取原始数据指针（只读）
     */
    const_pointer data() const { return data_; }
    
    /**
     * @brief 获取原始数据指针（可写）
     */
    pointer data() { return data_; }

private:
    AlignedAllocator<float> allocator_;
    pointer data_;
    size_type num_points_;
    size_type dimension_;
    size_type total_size_;
};

/**
 * @brief 高效缓存管理器，替代unordered_map<unsigned, bool>
 * 
 * 使用位图和哈希表组合，提供O(1)的查找和插入性能
 */
class DominationCache {
public:
    using key_type = std::pair<unsigned, unsigned>;
    using value_type = bool;
    
    /**
     * @brief 构造函数
     * @param capacity 预期容量
     */
    explicit DominationCache(size_t capacity = 10000) 
        : capacity_(capacity), size_(0) {
        // 使用位图存储布尔值，节省内存
        bit_map_.resize((capacity + 63) / 64, 0); // 每个uint64_t存储64个bool
        key_to_index_.reserve(capacity);
    }
    
    /**
     * @brief 获取缓存值
     * @param a 第一个键
     * @param b 第二个键
     * @param result 输出结果
     * @return 是否找到缓存值
     */
    bool get(unsigned a, unsigned b, bool& result) const {
        auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = key_to_index_.find(key);
        if (it != key_to_index_.end()) {
            size_t index = it->second;
            result = (bit_map_[index / 64] >> (index % 64)) & 1;
            return true;
        }
        return false;
    }
    
    /**
     * @brief 设置缓存值
     * @param a 第一个键
     * @param b 第二个键
     * @param value 要设置的值
     */
    void set(unsigned a, unsigned b, bool value) {
        auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = key_to_index_.find(key);
        
        if (it != key_to_index_.end()) {
            // 更新现有值
            size_t index = it->second;
            if (value) {
                bit_map_[index / 64] |= (1ULL << (index % 64));
            } else {
                bit_map_[index / 64] &= ~(1ULL << (index % 64));
            }
        } else if (size_ < capacity_) {
            // 插入新值
            size_t index = size_++;
            key_to_index_[key] = index;
            
            if (value) {
                bit_map_[index / 64] |= (1ULL << (index % 64));
            } else {
                bit_map_[index / 64] &= ~(1ULL << (index % 64));
            }
        }
        // 如果容量已满，忽略新插入
    }
    
    /**
     * @brief 清空缓存
     */
    void clear() {
        key_to_index_.clear();
        std::fill(bit_map_.begin(), bit_map_.end(), 0);
        size_ = 0;
    }
    
    /**
     * @brief 获取当前大小
     */
    size_t size() const { return size_; }
    
    /**
     * @brief 获取容量
     */
    size_t capacity() const { return capacity_; }

private:
    std::vector<uint64_t> bit_map_;  // 位图存储布尔值
    std::unordered_map<key_type, size_t> key_to_index_;  // 键到索引的映射
    size_t capacity_;
    size_t size_;
};

/**
 * @brief 线程安全的内存池
 * 
 * 提供高效的内存分配和回收机制，减少频繁的malloc/free调用
 */
template<typename T>
class ThreadSafeMemoryPool {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = size_t;
    
    /**
     * @brief 构造函数
     * @param block_size 每个内存块的大小
     * @param initial_blocks 初始内存块数量
     */
    explicit ThreadSafeMemoryPool(size_type block_size = 1024, size_type initial_blocks = 10)
        : block_size_(block_size), total_blocks_(0), free_blocks_(0) {
        
        // 预分配初始内存块
        for (size_type i = 0; i < initial_blocks; ++i) {
            allocate_new_block();
        }
    }
    
    /**
     * @brief 析构函数
     */
    ~ThreadSafeMemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : blocks_) {
            std::free(block);
        }
    }
    
    // 禁用拷贝
    ThreadSafeMemoryPool(const ThreadSafeMemoryPool&) = delete;
    ThreadSafeMemoryPool& operator=(const ThreadSafeMemoryPool&) = delete;
    
    /**
     * @brief 分配内存
     * @return 分配的内存指针
     */
    pointer allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_blocks_ == 0) {
            allocate_new_block();
        }
        
        pointer ptr = free_list_.back();
        free_list_.pop_back();
        --free_blocks_;
        
        return ptr;
    }
    
    /**
     * @brief 释放内存
     * @param ptr 要释放的内存指针
     */
    void deallocate(pointer ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push_back(ptr);
        ++free_blocks_;
    }
    
    /**
     * @brief 获取统计信息
     */
    struct Stats {
        size_type total_blocks;
        size_type free_blocks;
        size_type used_blocks;
        size_type block_size;
    };
    
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {total_blocks_, free_blocks_, total_blocks_ - free_blocks_, block_size_};
    }

private:
    void allocate_new_block() {
        pointer block = static_cast<pointer>(std::aligned_alloc(AlignUtils::DEFAULT_ALIGNMENT, 
                                                               block_size_ * sizeof(T)));
        if (!block) {
            throw std::bad_alloc();
        }
        
        blocks_.push_back(block);
        
        // 将新块中的所有内存单元加入空闲列表
        for (size_type i = 0; i < block_size_; ++i) {
            free_list_.push_back(block + i);
        }
        
        total_blocks_ += block_size_;
        free_blocks_ += block_size_;
    }
    
    mutable std::mutex mutex_;
    std::vector<pointer> blocks_;
    std::vector<pointer> free_list_;
    size_type block_size_;
    size_type total_blocks_;
    size_type free_blocks_;
};

/**
 * @brief 内存池管理器
 * 
 * 统一管理各种类型的内存池，提供全局访问接口
 */
class MemoryPoolManager {
public:
    static MemoryPoolManager& instance() {
        static MemoryPoolManager instance;
        return instance;
    }
    
    /**
     * @brief 获取连续数据存储
     * @param num_points 数据点数量
     * @param dimension 数据维度
     * @return 连续数据存储对象
     */
    std::unique_ptr<ContinuousDataStorage> create_continuous_storage(
        size_t num_points, size_t dimension) {
        return std::make_unique<ContinuousDataStorage>(num_points, dimension);
    }
    
    /**
     * @brief 获取支配关系缓存
     * @param capacity 容量
     * @return 支配关系缓存对象
     */
    std::unique_ptr<DominationCache> create_domination_cache(size_t capacity = 10000) {
        return std::make_unique<DominationCache>(capacity);
    }
    
    /**
     * @brief 获取线程安全内存池
     * @tparam T 数据类型
     * @param block_size 块大小
     * @param initial_blocks 初始块数量
     * @return 内存池对象
     */
    template<typename T>
    std::unique_ptr<ThreadSafeMemoryPool<T>> create_memory_pool(
        size_t block_size = 1024, size_t initial_blocks = 10) {
        return std::make_unique<ThreadSafeMemoryPool<T>>(block_size, initial_blocks);
    }

private:
    MemoryPoolManager() = default;
};

} // namespace Memory
} // namespace Infrastructure
} // namespace DSG
