# util.py

"""
本模块提供了一系列实用工具函数，用于处理数组、哈希表和其他常见数据结构。

### 函数列表

1. `generate_random_float_array(n)`
   - 生成一个包含随机正浮点数的数组。
   
2. `top_k_min_positions(arr, k)`
   - 找到数组中前 K 个最小元素的位置。
   
3. `hash_top_k_min_positions(positions)`
   - 对一组位置进行哈希处理以创建唯一标识符。

4. `cal_recall_of_two_sorted_arr(arr_1, arr_2)`
   - 计算第一个数组中有多少元素出现在第二个数组中，并返回召回率。

5. `get_top_k_minimums(array, k)`
   - 返回数组中最小的 k 个元素及其对应的索引。

6. `generate_frequency_ranges(max_freq)`
   - 根据最大频次生成合适的频次范围。

7. `calculate_range_percentage_distribution(hash_dict)`
   - 计算给定字典中每个频次范围的百分比分布。

8. `analyze_top_k_distribution(top_k_min_hash_map)`
   - 分析并打印前 K 个最小哈希值的分布情况。

### 示例用法

```python
from util import generate_random_float_array, top_k_min_positions

# 生成随机数组
random_array = generate_random_float_array(10)
print(random_array)

# 查找前 K 个小元素的位置
positions = top_k_min_positions(random_array, 3)
print(positions)
"""

import random
from collections import Counter
import heapq
import hashlib

def generate_random_float_array(n):
    """
    生成一个包含随机正浮点数的数组。
    
    参数:
    n -- 数组的长度
    
    返回:
    包含随机正浮点数的列表
    """
    return [random.uniform(0, 100) for _ in range(n)]

def top_k_min_positions(arr, k):
    """
    找到数组中前K个最小元素的位置。
    
    参数:
    arr -- 输入数组
    k -- 要找的最小元素的数量
    
    返回:
    前K个最小元素的位置列表
    """
    if len(arr) <= k:
        return list(range(len(arr)))
    return sorted(range(len(arr)), key=lambda x: arr[x])[:k]

def hash_top_k_min_positions(positions):
    """
    对一组位置进行哈希处理以创建唯一标识符。
    
    参数:
    positions -- 需要被哈希的一组位置
    
    返回:
    哈希后的十六进制字符串
    """
    m = hashlib.md5()
    for pos in positions:
        m.update(str(pos).encode('utf-8'))
    return m.hexdigest()

def cal_recall_of_two_sorted_arr(arr_1, arr_2):
    """
    返回 arr_1 中有百分之多少的元素在 arr_2 中。
    arr_1 和 arr_2 要在函数内排序下。
    
    参数:
    arr_1 -- 第一个排序数组
    arr_2 -- 第二个排序数组
    
    返回:
    recall -- arr_1 中存在于 arr_2 中的元素num / len(arr_2) (介于 0.0 和 1.0 之间)
    """
    if not arr_1:
        return 0.0  # 如果 arr_1 为空，召回率为 0.0
    
    arr_1.sort()
    arr_2.sort()

    len_1 = len(arr_1)
    len_2 = len(arr_2)
    count = 0
    j = 0

    for i in range(len_1):
        while j < len_2 and arr_2[j] < arr_1[i]:
            j += 1
        if j < len_2 and arr_2[j] == arr_1[i]:
            count += 1
            j += 1

    recall = count / len_2
    return recall

def get_top_k_minimums(array, k):
    """
    返回数组中最小的 k 个元素及其对应的索引。
    
    参数:
    array -- 输入数组
    k     -- 最小值的数量
    
    返回:
    包含元组（索引，值）的列表，表示最小的 k 个元素及其在原数组中的位置
    并且返回的结果按照位置排序。
    """
    if not array:
        return []

    max_heap = []  # 创建一个最大堆来存储前 k 小的元素
    for index, value in enumerate(array):
        if len(max_heap) < k:
            heapq.heappush(max_heap, (-value, index))
        elif value < -max_heap[0][0]:
            heapq.heapreplace(max_heap, (-value, index))

    return sorted([(-val, idx) for val, idx in max_heap], key=lambda x: x[0])

def generate_frequency_ranges(max_freq):
    """
    根据最大频次生成合适的频次范围。

    参数:
    max_freq (int): 最大频次。

    返回:
    list of tuple: 频次范围列表。
    """
    ranges = [(1, 1), (2, 2)]
    step = 2
    start = 3
    while start <= max_freq:
        end = min(start + step - 1, max_freq)
        ranges.append((start, end))
        start += step
        step *= 2  # 增加步长以生成更大的范围
    return ranges


def calculate_range_percentage_distribution(hash_dict):
    """
    计算给定字典中每个频次范围的百分比分布。

    参数:
    hash_dict (dict): 输入的字典，键为哈希值，值为出现的频次。

    返回:
    dict: 每个频次范围占总数的百分比。
    """
    # 统计每个范围值出现的频次
    value_counts = Counter(hash_dict.values())
    # 总频次数
    total_counts = sum(value_counts.values())
    # 最大频次
    max_freq = max(value_counts.keys())

    # 生成频次范围
    ranges = generate_frequency_ranges(max_freq)

    # 初始化范围分布字典
    range_distribution = {range_label: 0 for range_label in [f"{r[0]}-{r[1]}" for r in ranges]}

    # 计算每个范围内的频次数量
    for value, count in value_counts.items():
        for start, end in ranges:
            if start <= value <= end:
                range_label = f"{start}-{end}"
                range_distribution[range_label] += count
    
    # 将频次数量转换为百分比
    percentage_distribution = {k: v / total_counts * 100 for k, v in range_distribution.items()}
    
    return percentage_distribution

def analyze_top_k_distribution(top_k_min_hash_map):
    """
    分析并打印前K个最小哈希值的分布情况。
    
    :param top_k_min_hash_map: 字典，键是最小哈希值，值是出现次数。
    """
    unique_top_k_count = len(top_k_min_hash_map)
    print(f"总独特前-K个最小值: {unique_top_k_count}")
    
    # 创建一个新的字典，只包含计数信息
    simplified_top_k_min_hash_map = {}
    # 创建一个新的计数器来有多少position会在prunedKNN里
    position_counts = Counter()
    # 遍历现有的 top_k_min_hash_map
    for key, value in top_k_min_hash_map.items():
        # 假设 value 是一个形如 (list_of_data, count) 的元组
        pruned_topk_nnids, count = value
        position_counts.update(pruned_topk_nnids)
        
        # 将简化后的值存入新的字典
        simplified_top_k_min_hash_map[key] = count

    # 此时，simplified_top_k_min_hash_map 只包含了数字值
    count_each_hash = dict(simplified_top_k_min_hash_map)  # 确保副本，虽然在这里不是必需
    # 计算频次范围分布占比
    distribution = calculate_range_percentage_distribution(count_each_hash)
    
    print("频次范围分布占比:")
    for range_label, percentage in distribution.items():
        print(f"{range_label} 次出现: {percentage:.2f}%")
    print(f"pos unique的数量: {len(position_counts)}")