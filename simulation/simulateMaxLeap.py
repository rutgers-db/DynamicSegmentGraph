import random
import numpy as np
from util import *

def search_top_k_ranges(array, k, pivot, start, end):
    """
    递归查找数组的压缩范围，以包含至少 k 个最小元素。
    
    参数:
    array   -- 输入数组
    k       -- 最小元素数量
    pivot   -- 当前搜索区间的中心点
    start   -- 当前搜索区间的起始索引
    end     -- 当前搜索区间的结束索引
    
    返回:
    包含压缩范围的列表，每个范围是一个四元组（start, l_rightmost, r_leftmost, end）
    """
    if end - start + 1 < k: # 如果搜索区间长度小于 k，则直接返回从 start 到 end 整个range
        top_k_indices = list(range(start, end + 1))  # 从 start 一直到 end 的 indices
        return [(start, pivot, pivot, end, top_k_indices)]

    top_k_elements = get_top_k_minimums(array[start:end+1], k)
    top_k_indices = [idx + start for _, idx in top_k_elements]
    # print("Top-k Elements:", top_k_elements)
    # print("Top-k Indices:", top_k_indices)
    
    l_rightmost = pivot
    tmp_l_rightmost = -1
    tmp_r_leftmost = float('inf')
    r_leftmost = pivot

    for idx in top_k_indices:
        if idx <= pivot:
            tmp_l_rightmost = max(tmp_l_rightmost, idx)
        else:
            tmp_r_leftmost = min(tmp_r_leftmost, idx)

    if tmp_l_rightmost != -1:
        l_rightmost = tmp_l_rightmost

    if tmp_r_leftmost != float('inf'):
        r_leftmost = tmp_r_leftmost

    compress_range = (start, l_rightmost, r_leftmost, end, top_k_indices)

    if tmp_l_rightmost != -1:
        start = l_rightmost + 1
    if tmp_r_leftmost != float('inf'):
        end = r_leftmost - 1

    inner_ranges = search_top_k_ranges(array, k, pivot, start, end)

    return [compress_range] + inner_ranges

def optimize_search_top_k_ranges(array, pivot, k):
    """
    优化后的版本，通过排序和滑动窗口来生成压缩范围。

    参数:
    array -- 输入数组
    k     -- 最小元素数量

    返回:
    包含压缩范围的列表，每个范围是一个五元组（start, l_rightmost, r_leftmost, end, top_k_indices）
    """
    n = len(array)
    if n == 0:
        return []

    sorted_array = sorted((val, idx) for idx, val in enumerate(array))
    result_ranges = []

    start, end = 0, n - 1
    cnt = 0
    while start <= end and cnt < len(sorted_array):
        top_k_elements = []
        top_k_indices = []

        while cnt < len(sorted_array):
            val, idx = sorted_array[cnt]
            cnt += 1
            if start <= idx <= end:
                top_k_elements.append(val)
                top_k_indices.append(idx)
                if len(top_k_indices) == k:
                    break

        # 设置 l_rightmost 和 r_leftmost 的默认值为 pivot 如果两边哪个是空的就会返回default值
        l_rightmost = max((idx for idx in top_k_indices if idx <= pivot), default=pivot)
        r_leftmost = min((idx for idx in top_k_indices if idx > pivot), default=pivot)

        compress_range = (start, l_rightmost, r_leftmost, end, top_k_indices)
        result_ranges.append(compress_range)
        
        if len(top_k_indices) < k:
            break

        if l_rightmost != pivot:
            start = l_rightmost + 1

        if r_leftmost != pivot:
            end = r_leftmost - 1

        print(f"start:{start} end:{end} cnt:{cnt}")
    return result_ranges

def create_top_k_compress_ranges(array, pivot, k):
    """
    根据输入数组和指定的 k 值，生成压缩范围列表。
    
    参数:
    array -- 输入数组
    k     -- 最小元素数量
    
    返回:
    包含压缩范围的列表
    """
    length = len(array)
    if length == 0:
        return []

    return search_top_k_ranges(array, k, pivot, 0 , length - 1)

def query_top_k_from_ranges(L, R, k, compress_ranges):
    """
    在给定的压缩范围内查询 [L, R] 范围内的 top-k 最小值的索引。
    
    参数:
    L                -- 查询范围的起始索引
    R                -- 查询范围的结束索引
    k                -- 最小值的数量
    compress_ranges  -- 预先计算好的压缩范围
    
    返回:
    在 [L, R] 范围内的 top-k 最小值的索引列表
    """
    # 检查范围是否合法
    if L > R:
        return []
    
    if L<0 or R>=len(array):
        return []

    result = []
    for compress_range in compress_ranges:
        start, l_rightmost, r_leftmost, end, top_k_indices = compress_range
        
        if start <= L <= l_rightmost or r_leftmost <= R <= end:
            for idx in top_k_indices:
                if L <= idx <= R:
                    result.append(idx)
                    if len(result) >= k:
                        return result
    return result

# 示例输入
n = 8192
k = 13
seed = 42
DEBUG = False

random.seed(seed)
array = generate_random_float_array(n)
pivot = len(array) // 2
print(f"Array Len:{n} k:{k}")
# 调用函数并输出压缩区间
# compress_ranges = create_top_k_compress_ranges(array, pivot, k)
compress_ranges = optimize_search_top_k_ranges(array, pivot, k)

if DEBUG:
    # 输出压缩区间
    print(f"There {len(compress_ranges)} Top-k Compress Ranges:")
    for cr in compress_ranges:
        print(cr)

# 生成 100 个随机查询
num_queries = 100
random.seed(seed)  # 确保查询的随机性与生成数组的一致
queries = [(random.randint(0, pivot), random.randint(pivot + 1, n - 1)) for _ in range(num_queries)]
print(f"There are {num_queries} random queries:")

# 计算召回率
recalls = []
for L, R in queries:
    query_topk_indices = query_top_k_from_ranges(L, R, k, compress_ranges)
    query_topk_indices.sort()
    ground_truth = get_top_k_minimums(array[L:R+1], k)
    truth_indices = [idx + L for _, idx in ground_truth]
    recall = cal_recall_of_two_sorted_arr(query_topk_indices, truth_indices)
    recalls.append(recall)

# 统计召回率
max_recall = max(recalls)
min_recall = min(recalls)
avg_recall = np.mean(recalls)

print(f"最高召回率: {max_recall:.2f}")
print(f"最低召回率: {min_recall:.2f}")
print(f"平均召回率: {avg_recall:.2f}")

# # 查询范围
# L = 490
# R = 510

# # 查询 [L, R] 范围内的 top-k 最小值的索引
# query_topk_indices = query_top_k_from_ranges(L, R, k, compress_ranges)
# # 排序下 query_topk_indices
# query_topk_indices.sort()

# print(f"Query [{L}, {R}] 范围内的 top-{k} 最小值的索引:")
# print(query_topk_indices)
# ground_truth = get_top_k_minimums(array[L:R+1], k)
# truth_indices = [idx + L for _, idx in ground_truth]
# print(truth_indices)
# print(f"recall of approximate Top-{k}: {cal_recall_of_two_sorted_arr(query_topk_indices, truth_indices)}")

# if DEBUG:
#     print(f"approximate Top-{k} 最小值:")
#     print([array[idx] for idx in query_topk_indices])
#     print("实际最小值是")
#     print(ground_truth)