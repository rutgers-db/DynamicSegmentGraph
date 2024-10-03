from bisect import bisect_left
from collections import defaultdict
import json
from IO import process_lines
from util import hash_top_k_min_positions, analyze_top_k_distribution
import time

# 全局常量定义
K = 8
PIVOT_ID = 2048  # 基准点ID
DOMINATION_FILE_PATH = './sample_data/sampled_neighbors_domination.txt'
IF_SAVE = True
IF_LOAD_DOMINATE = True

# 数据读取
nns, points_dominate_me = process_lines(DOMINATION_FILE_PATH, IF_LOAD_DOMINATE)

# 数据预处理
sorted_nns = sorted(nns, key=lambda x: x[0])
sorted_nn_ids = [x[0] for x in sorted_nns]
pivot_pos = bisect_left(sorted_nns, (PIVOT_ID, 0))

# 确定基准点附近的邻居ID
if pivot_pos < len(sorted_nn_ids):
    pivot_nn_id_after = sorted_nn_ids[pivot_pos]
else:
    pivot_nn_id_after = PIVOT_ID

print(f"基准点位置 {pivot_pos} 是在：{sorted_nn_ids[pivot_pos - 1]} 和 {pivot_nn_id_after} 之间")
print(f"第一个点的nnid {sorted_nn_ids[0]} 和 最后一个点的nnid {sorted_nn_ids[-1]}")


def get_pruned_topk_nn_ids(start, end):
    """
    获取指定范围内，经过修剪的前K个最近邻的位置。

    :param start: 子范围的起始索引。
    :param end: 子范围的结束索引。
    :return: 经过修剪的前K个最近邻的位置列表。
    """
    start = max(0, start)
    end = min(len(sorted_nns) - 1, end)
    
    sub_range_nns = sorted_nns[start:end + 1]
    sorted_nns_sub_range = sorted(sub_range_nns, key=lambda x: x[1])
    pruned_topk_nn_pos = []
    pruned_topk_nn_ids = []
    for k, nn_tuple in enumerate(sorted_nns_sub_range):
        nn_id = nn_tuple[0]
        nn_ori_arr_pos = nns.index(nn_tuple)

        good = True
        # Check if the nearest neighbor is not pruned by any previous ones.        
        for pre_nn_id_pos in pruned_topk_nn_pos:
                pre_nn_id = nns[pre_nn_id_pos][0]
                if pre_nn_id in points_dominate_me[nn_ori_arr_pos]:
                    good = False
                    break

        if good:
            pruned_topk_nn_pos.append(nn_ori_arr_pos)
            pruned_topk_nn_ids.append(nn_id)

        if len(pruned_topk_nn_pos) == K:
            break

    return pruned_topk_nn_ids


# 生成所有剪枝后的topK值
top_k_min_hash_map = defaultdict(lambda: [[], 0])

# 暴力法计算
for i in range(0, pivot_pos + 1):
    for j in range(pivot_pos - 1, len(sorted_nns)):
        if i > j:
            continue
        
        # 获取剪枝后的topK值
        pruned_topk_nnids = get_pruned_topk_nn_ids(i, j)
        
        # 添加到哈希表
        key = hash_top_k_min_positions(pruned_topk_nnids)
        top_k_min_hash_map[key][0] = pruned_topk_nnids
        top_k_min_hash_map[key][1] += 1

# 输出topK分布分析结果
analyze_top_k_distribution(top_k_min_hash_map)

# 保存结果
if IF_SAVE:
    with open('topk_results/top_k_bruteforce.json', 'w') as f:
        json.dump(top_k_min_hash_map, f)
        


def process_range(pivot_pos, sorted_nns, sorted_nn_ids, pruned_topk_nn_ids, dominate_nums, L_stack, R_stack, top_k_min_hash_map):
    """
    处理指定范围内的数据，并更新相关统计信息。
    
    参数：
    pivot_pos: 中心点的位置
    sorted_nns: 排序后的邻居列表
    sorted_nn_ids: 排序后的邻居ID列表
    pruned_topk_nn_ids: 剪枝后的最近邻ID列表
    dominate_nums: 被支配点的数量统计数组
    L_stack: 左边界栈
    R_stack: 右边界栈
    top_k_min_hash_map: 存储top-k最小值分布的哈希表
    """

    cnt_searchRange = 0
    
    # 初始化左右边界栈
    L_stack.append(-1)
    R_stack.append(len(sorted_nns))
    
    while L_stack[-1] <= R_stack[-1]:
        if (L_stack[-1] == pivot_pos and pivot_pos == len(sorted_nns)) or \
           (pivot_pos == 0 and R_stack[-1] == -1):
            break
        
        # 获取剪枝后的最近邻ID列表
        pruned_topk_nn_ids = get_pruned_topk_nn_ids(L_stack[-1], R_stack[-1])
        cnt_searchRange = cnt_searchRange + 1
        
        # 获取最左侧和最右侧的位置索引
        sorted_topk_nnids = sorted(pruned_topk_nn_ids)
        leftmost_nn_id = sorted_topk_nnids[0]
        rightmost_nn_id = sorted_topk_nnids[-1]

        leftmost_pos = sorted_nn_ids.index(leftmost_nn_id)
        rightmost_pos = sorted_nn_ids.index(rightmost_nn_id)

        leftmost_rank, rightmost_rank = -1, -1
        
        # 找到左侧和右侧的排名
        if leftmost_pos < pivot_pos:
            leftmost_rank = pruned_topk_nn_ids.index(leftmost_nn_id)
            
        if rightmost_pos >= pivot_pos:
            rightmost_rank = pruned_topk_nn_ids.index(rightmost_nn_id)
            
        if rightmost_rank > leftmost_rank:
            new_R = rightmost_pos - 1
            R_stack.append(new_R)
            
            if leftmost_rank != -1:
                dominate_nums[leftmost_pos] += 1
                
            while dominate_nums[rightmost_pos] > 0:
                dominate_nums[rightmost_pos] -= 1
                L_stack.pop()
                
        else:
            new_L = leftmost_pos + 1
            L_stack.append(new_L)
            
            if rightmost_rank != -1:
                dominate_nums[rightmost_pos] += 1
                
            while dominate_nums[leftmost_pos] > 0:
                dominate_nums[leftmost_pos] -= 1
                R_stack.pop()
                
        # 更新哈希表
        key = hash_top_k_min_positions(pruned_topk_nn_ids)
        top_k_min_hash_map.setdefault(key, [None, 0])
        top_k_min_hash_map[key][0] = pruned_topk_nn_ids
        top_k_min_hash_map[key][1] += 1
    
    return cnt_searchRange

def optimized_generating_method(sorted_nns, sorted_nn_ids, pivot_pos, top_k_min_hash_map, if_save=True):
    """
    使用优化算法生成所有的prunedTopK。
    
    参数：
    sorted_nns: 排序后的邻居列表
    sorted_nn_ids: 排序后的邻居ID列表
    pivot_pos: 中心点的位置
    top_k_min_hash_map: 存储top-k最小值分布的哈希表
    if_save: 是否保存结果，默认为True
    """
    dominate_nums = [0] * len(sorted_nns)
    L_stack, R_stack = [], []

    # 清空哈希表
    top_k_min_hash_map.clear()

    cnt_searchRange = process_range(pivot_pos, sorted_nns, sorted_nn_ids, None, dominate_nums, L_stack, R_stack, top_k_min_hash_map)
    print(cnt_searchRange)
    
    # 输出分析结果
    analyze_top_k_distribution(top_k_min_hash_map)

    if if_save:
        file_path = 'topk_results/top_k_Optimization.json'
        with open(file_path, 'w') as f:
            json.dump(top_k_min_hash_map, f)

# start_time = time.time()  # 记录开始时间
# optimized_generating_method(sorted_nns, sorted_nn_ids, pivot_pos, top_k_min_hash_map, IF_SAVE)
# end_time = time.time()  # 记录结束时间
# print(f"函数耗时: {end_time - start_time:.6f} 秒")