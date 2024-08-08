"""
模拟计算数组中所有可能范围内的前K个最小值的不同组合数量以及每个组合出现的次数。

问题描述：对于一个长度为N的随机正浮点数数组，考虑一个位置Pivot（0 <= Pivot < N），
定义一个范围是[l, r]其中0 <= l <= Pivot < r < N。目标是找到所有这些范围内
的前K个最低数值的位置组合有多少种不同的情况，并记录每种组合对应多少个范围。

方法：通过哈希映射存储每个独特的前K个最小值的位置组合，并统计其出现频率。

"""

import random
import hashlib
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter

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


def simulate(n, pivot):
    if pivot < -1 or pivot >= n:
        raise ValueError("pivot must be between -1 and n-1")
    
    k = int(math.log2(n)) # 前K个最小值，k = log2(n)
    
    random.seed(42)
    
    # 步骤1：生成随机正浮点数数组
    arr = generate_random_float_array(n)
    # print(f"生成的数组: {arr}")
    
    # 步骤2：创建哈希映射以存储独特的前K个最小值位置
    top_k_min_hash_map = defaultdict(int)
    position_counts = Counter()

    # 步骤3：遍历所有可能的范围
    # 这个步骤3 需要考虑 pivot为-1 和pivot 为 n-1的情况
    if 0 <= pivot < n - 1:
        # 遍历以pivot为中心的所有可能子数组
        for left in range(pivot + 1):
            for right in range(pivot + 1, n):
                sub_arr = arr[left:right + 1]
                top_k_min_pos = top_k_min_positions(sub_arr, min(k, len(sub_arr)))
                adjusted_pos = [left + pos for pos in top_k_min_pos]  # 调整位置相对于原数组
                position_counts.update(adjusted_pos)
                hash_value = hash_top_k_min_positions(adjusted_pos)
                top_k_min_hash_map[hash_value] = top_k_min_hash_map.get(hash_value, 0) + 1

    elif pivot == -1:
        # 处理pivot为-1的情况
        for right in range(n):
            sub_arr = arr[:right + 1]
            top_k_min_pos = top_k_min_positions(sub_arr, min(k, len(sub_arr)))
            position_counts.update(top_k_min_pos)
            hash_value = hash_top_k_min_positions(top_k_min_pos)
            top_k_min_hash_map[hash_value] = top_k_min_hash_map.get(hash_value, 0) + 1

    else:  # pivot == n - 1
        # 处理pivot为n-1的情况
        for left in range(n):
            sub_arr = arr[left:]
            top_k_min_pos = top_k_min_positions(sub_arr, min(k, len(sub_arr)))
            adjusted_pos = [left + pos for pos in top_k_min_pos]  # 调整位置相对于原数组
            position_counts.update(adjusted_pos)
            hash_value = hash_top_k_min_positions(adjusted_pos)
            top_k_min_hash_map[hash_value] = top_k_min_hash_map.get(hash_value, 0) + 1

    # 步骤4：计算结果
    unique_top_k_count = len(top_k_min_hash_map)
    count_each_hash = dict(top_k_min_hash_map)
    
    print(f"总独特前-K个最小值: {unique_top_k_count}")
    # print(f"每个哈希对应的范围数量: {count_each_hash}") 
    distribution = calculate_range_percentage_distribution(count_each_hash)
    print("频次范围分布占比:")
    for range_label, percentage in distribution.items():
        print(f"{range_label} 次出现: {percentage:.2f}%")
    print(f"pos unique的数量: {len(position_counts)}")
    print(f"pos unique的分布: {position_counts}")
    visualize_counter(position_counts)

if __name__ == "__main__":
    n = 1 << 20 # Example value for n, which is 1024
    # the value of k is log2(n)
    pivot = int(n / 2)  # Simulate the case where pivot is in the middle of the array
    # pivot = -1 # Simulate the case where pivot is at the beginning of the array
    # pivot = n - 1 # Simulate the case where pivot is at the end of the array
    print(f"Simulate n:{n} pivot:{pivot}")
    simulate(n, pivot)