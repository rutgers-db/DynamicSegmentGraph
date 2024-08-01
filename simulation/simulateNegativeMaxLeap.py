"""
    这是模拟反向边的点插入到了正向边上的场景4️⃣
    插入的点经过处理，是否等同于当初maxleap建立正向边时插入点就在的效果
    脚本会有两种生成方式：
    1. 先插入到已经生成的batch上
    2. 放在当初排好序的里面然后再生成batch
    最后比较二者是否一样
"""

import heapq
import random
import numpy as np
import math

def generate_random_float_array(n):
    """
    生成一个包含随机正浮点数的数组。
    
    参数:
    n -- 数组的长度
    
    返回:
    包含随机正浮点数的列表
    """
    return [random.uniform(0, 100) for _ in range(n)]

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
        l_rightmost = max((idx for idx in top_k_indices if idx <= pivot), default=pivot+1)
        r_leftmost = min((idx for idx in top_k_indices if idx > pivot), default=pivot)

        compress_range = (start, l_rightmost, r_leftmost, end, top_k_indices)
        result_ranges.append(compress_range)
        
        if len(top_k_indices) < k:
            break

        if l_rightmost != pivot+1:
            start = l_rightmost + 1

        if r_leftmost != pivot:
            end = r_leftmost - 1

        # print(f"start:{start} end:{end} cnt:{cnt}")
    return result_ranges

def insert_point2arr(array, insert_point, insert_pos):
    """
    在数组中插入一个点。
    
    参数:
    array -- 输入数组
    insert_point -- 要插入的点
    insert_pos -- 要插入的点的位置
    
    返回:
    插入点后的数组
    """
    array.insert(insert_pos, insert_point)
    return array

def insert_point2compress_ranges(compress_ranges, insert_points, array, k, pivot):
    """
    在这些压缩区间中插入一个点。
    
    参数:
    compress_ranges -- 输入压缩区间
    insert_point -- 要插入的点
    insert_pos -- 要插入的点的位置
    
    返回:
    插入点后的压缩区间
    """
    compress_ranges_len = len(compress_ranges)
    cnt = 0
    limit_range = [0, len(array)-1]
    range_if_update = False
    while (len(insert_points) > 0 or range_if_update) and cnt < compress_ranges_len :
        range_if_update = False
        cr = compress_ranges[cnt]
        
        updated_cr, insert_points = insert_points2cr(limit_range, cr, insert_points, array, k, pivot)
        _, l_rightmost, r_leftmost ,_ ,_ = updated_cr
        if l_rightmost != pivot+1:
            limit_range[0] = l_rightmost + 1

        if r_leftmost != pivot:
            limit_range[1] = r_leftmost - 1
        if [l_rightmost , r_leftmost] != [cr[1], cr[2]]:
            range_if_update = True
        compress_ranges[cnt] = updated_cr     
        cnt += 1

    return compress_ranges
    
def insert_points2cr(cur_range, cr, insert_points, array, k, pivot):
    """
    尝试对一个cr插入
    """
    # print(cur_range)
    # print(cr)
    start, l_rightmost, r_leftmost, end, top_k_indices = cr
    if start != cur_range[0] or end != cur_range[1]:
        start = cur_range[0]
        end = cur_range[1]
    
    # TODO: Maybe the passed cur_range is a larger range and curerent top_k_indices is not up to k elements (the last layer)
    # We can just push the outer elements into the top_k_indices (But it is not important to update the minimum range)
    # build a max heap for top_k_indices that the top point has maximum value array[indice]
    max_heap = []
    for indice in top_k_indices:
        if(indice >= start and indice <= end):
            heapq.heappush(max_heap, (-array[indice], indice))

    max_topkpoint = -max_heap[0][0]

    to_pass_points = []
    dropped_points = []
    for insert_point, insert_point_pos in insert_points:
        # check whether smaller than any points in top k indices
        if(insert_point > max_topkpoint):
            if insert_point_pos > l_rightmost and insert_point_pos < r_leftmost:
                to_pass_points.append((insert_point, insert_point_pos))
            # otherwise just drop it
            continue

        # means this batch need to be updated
        heapq.heappush(max_heap, (-insert_point, insert_point_pos))

    while len(max_heap)>k:
        pop_point = heapq.heappop(max_heap)
        dropped_points.append((-pop_point[0], pop_point[1]))

    # insert points of heap into top_k_indices
    heap_len = len(max_heap)
    top_k_indices = [heapq.heappop(max_heap)[1] for _ in range(heap_len)]
    # reverse top_k_indices
    top_k_indices.reverse()

    # renew l_rightmost and r_leftmost
    l_rightmost = max((idx for idx in top_k_indices if idx <= pivot), default=pivot+1)
    r_leftmost = min((idx for idx in top_k_indices if idx > pivot), default=pivot)

    updated_cr = (start, l_rightmost, r_leftmost, end, top_k_indices)
    
    for point,pos in dropped_points:
        if pos > l_rightmost and pos < r_leftmost:
            to_pass_points.append((point, pos))

    return updated_cr, to_pass_points

def update_compress_ranges(ori_compress_ranges: list[tuple[int, int, int, int, list[int]]], insert_pos: int) -> list[tuple[int, int, int, int, list[int]]]:
    """
    更新压缩范围列表，以反映在指定位置插入新元素的影响。

    参数:
    - ori_compress_ranges: 列表，其中每个元素都是一个五元组 (start, l_rightmost, r_leftmost, end, top_k_indices)，表示一个压缩范围及其属性。
    - insert_pos: 整数，表示新元素的插入位置。

    返回:
    - 更新后的压缩范围列表，其中每个范围的起始、结束、左右最远点以及top_k_indices已根据插入位置进行了相应调整。
    """
    moved_compress_ranges = []
    for cr in ori_compress_ranges:
        start, l_rightmost, r_leftmost, end, top_k_indices = cr
        
        # 根据insert_pos更新start, end, l_rightmost, r_leftmost
        start = start + 1 if start >= insert_pos else start
        end = end + 1 if end >= insert_pos else end
        l_rightmost = l_rightmost + 1 if l_rightmost >= insert_pos else l_rightmost
        r_leftmost = r_leftmost + 1 if r_leftmost >= insert_pos else r_leftmost
        
        # 更新top_k_indices
        updated_top_k_indices = [i + 1 if i >= insert_pos else i for i in top_k_indices]

        # 创建更新后的压缩范围并添加到列表
        moved_cr = (start, l_rightmost, r_leftmost, end, updated_top_k_indices)
        moved_compress_ranges.append(moved_cr)
    
    return moved_compress_ranges


# 示例输入
n = 128
k = int(math.log2(n)) #k = log2(n)
seed = 2
random.seed(seed)
start_test_id = 72
DEBUG = True
num_tests = 100
all_passed = True
for test_id in range(num_tests):
    insert_point = random.uniform(0, 100)
    insert_pos = random.randint(0, n-1)
    array = generate_random_float_array(n)
    pivot = len(array) // 2
    if test_id < start_test_id:
        continue
    # 调用函数并输出压缩区间
    ori_compress_ranges = optimize_search_top_k_ranges(array, pivot, k)

    # if DEBUG:
    #     # 输出压缩区间
    #     print(f"Orignal Compress Rnages {len(ori_compress_ranges)}:")
    #     for cr in ori_compress_ranges:
    #         print(cr)

    # 插入点
    updated_array = insert_point2arr(array, insert_point, insert_pos)
    pivot = pivot if insert_pos >= pivot else pivot + 1

    # Move position of original compress range
    moved_compress_ranges = update_compress_ranges(ori_compress_ranges, insert_pos)

    updated_compress_ranges = insert_point2compress_ranges(moved_compress_ranges, [(insert_point, insert_pos)], updated_array, k, pivot)
    generated_crs = optimize_search_top_k_ranges(array, pivot, k)
    # Search topk compress ranges

    if not generated_crs == updated_compress_ranges:
        all_passed = False
        if DEBUG:
            print(f"Array Len:{n} k:{k}, insert pos:{insert_pos} in test: {test_id}")
            print(f"Orignal Compress Rnages {len(ori_compress_ranges)}:")
            for cr in ori_compress_ranges:
                print(cr)
            # 输出压缩区间
            print(f"There {len(updated_compress_ranges)} Updated Top-k Compress Ranges:")
            for cr in updated_compress_ranges:
                print(cr)

            # 输出压缩区间
            print(f"Generated Top-k Compress Ranges: {len(generated_crs)} Updated Top-k Compress Ranges:")
            for cr in generated_crs:
                print(cr)
        break       

print(f"All tests passed: {all_passed}")
            

