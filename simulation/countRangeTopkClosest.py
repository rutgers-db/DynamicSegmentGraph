"""
模拟计算数组中所有可能范围内的前K个最小值的不同组合数量以及每个组合出现的次数。

问题描述：对于一个长度为N的随机正浮点数数组，考虑一个位置Pivot（0 <= Pivot < N），
定义一个范围是[l, r]其中0 <= l <= Pivot < r < N。目标是找到所有这些范围内
的前K个最低数值的位置组合有多少种不同的情况，并记录每种组合对应多少个范围。

方法：通过哈希映射存储每个独特的前K个最小值的位置组合，并统计其出现频率。

"""
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter
from util import *

def visualize_counter(counter_data):
    """
    接收一个Counter对象，绘制其条形图。
    
    参数:
    counter_data (Counter): 包含位置和计数的Counter对象。
    """
    # 准备 x 和 y 的数据
    x = list(counter_data.keys())
    y = list(counter_data.values())

    # 创建条形图
    plt.figure(figsize=(10, 6))  # 可根据需要调整图形大小
    plt.bar(x, y)

    # 添加标题和轴标签
    plt.title('Position Counts Visualization')
    plt.xlabel('Position')
    plt.ylabel('Count')

    # 自动旋转x轴的标签以防止重叠
    plt.xticks(rotation=90)

    # 调整布局以适应所有的x轴标签
    plt.tight_layout()

    # 显示图形
    plt.show()

def bf_all_ranges(arr, n, pivot, k):
    """
    对给定数组arr，在所有可能的子数组范围内查找前k个最小值的位置，
    并统计这些位置的频率。
    
    参数：
    arr (list): 输入数组
    n (int): 数组长度
    pivot (int): 中心点索引
    k (int): 前k个最小值
    
    返回：
    Counter: 子数组内前k个最小值的位置计数器
    """

    def validate_pivot(pivot, n):
        """验证中心点是否有效"""
        if pivot < -1 or pivot >= n:
            raise ValueError("pivot 必须在 -1 和 n-1 之间")

    def update_position_counts(left, right, arr, k, position_counts, top_k_min_hash_map):
        """更新位置计数器和哈希映射"""
        sub_arr = arr[left:right + 1]
        top_k_min_pos = top_k_min_positions(sub_arr, min(k, len(sub_arr)))
        adjusted_pos = [left + pos for pos in top_k_min_pos]
        position_counts.update(adjusted_pos)
        hash_value = hash_top_k_min_positions(adjusted_pos)
        top_k_min_hash_map[hash_value] = top_k_min_hash_map.get(hash_value, 0) + 1

    validate_pivot(pivot, n)

    top_k_min_hash_map = defaultdict(int)
    position_counts = Counter()

    if 0 <= pivot < n - 1:
        for left in range(pivot + 2):
            for right in range(pivot, n):
                if left > right:
                    continue
                update_position_counts(left, right, arr, k, position_counts, top_k_min_hash_map)

    elif pivot == -1:
        for right in range(n):
            update_position_counts(0, right, arr, k, position_counts, top_k_min_hash_map)

    else:  # pivot == n - 1
        for left in range(n):
            update_position_counts(left, n - 1, arr, k, position_counts, top_k_min_hash_map)

    unique_top_k_count = len(top_k_min_hash_map)
    count_each_hash = dict(top_k_min_hash_map)
    
    print(f"总独特前-K个最小值: {unique_top_k_count}")
    distribution = calculate_range_percentage_distribution(count_each_hash)
    print("频次范围分布占比:")
    for range_label, percentage in distribution.items():
        print(f"{range_label} 次出现: {percentage:.2f}%")
    print(f"pos unique的数量: {len(position_counts)}")
    print(f"pos unique的分布: {position_counts}")

    # visualize_counter(position_counts)
    return position_counts

def simulate(arr, n, pivot, k):
    """
    根据定义找到所有位置，即k近邻中每个邻居的位置。
    
    参数同上。
    
    返回：
    Counter: 子数组内前k个最小值位置的计数器
    """

    if pivot < -1 or pivot >= n:
        raise ValueError("pivot 必须在 -1 和 n-1 之间")

    count_valid_i = 0
    position_counts = Counter()

    for i in range(pivot + 1):
        right_numbers = np.array(arr[i + 1:pivot + 1])
        count_smaller = np.sum(right_numbers < arr[i])
        if count_smaller <= k - 1:
            count_valid_i += 1
            position_counts.update([i])

    for i in range(pivot + 1, n):
        left_numbers = np.array(arr[pivot + 1:i + 1])
        count_smaller = np.sum(left_numbers < arr[i])
        if count_smaller <= k - 1:
            count_valid_i += 1
            position_counts.update([i])

    print(f"验证前k个唯一位置的数量: {count_valid_i}")
    print(f"pos unique 的数量: {len(position_counts)}")
    print(f"pos unique 的分布: {position_counts}")

    return position_counts

if __name__ == "__main__":
    
    seed = 42
    random.seed(seed)
    n = 1024 # 1 << 12 # 1 << 10 # Example value for n, which is 1024  # 760
    k = 8 # k = int(math.log2(n))  # the value of k is log2(n)
    # pivot = int(n / 2)  # Simulate the case where pivot is in the middle of the array
    # pivot = -1 # Simulate the case where pivot is at the beginning of the array
    pivot = n - 1 # Simulate the case where pivot is at the end of the array

    print(f"Simulate n:{n} pivot:{pivot}")
    
    # 步骤1：生成随机正浮点数数组
    arr = generate_random_float_array(n)
    # print(f"生成的数组: {arr}")
    
    bf_result = bf_all_ranges(arr, n, pivot, k)
    sim_result = simulate(arr, n, pivot, k)
    
    diff = sim_result - bf_result
    print(f"diff: {diff}")