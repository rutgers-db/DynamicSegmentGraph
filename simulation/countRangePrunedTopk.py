from bisect import bisect_left
from countRangeTopkClosest import hash_top_k_min_positions, calculate_range_percentage_distribution
from collections import defaultdict
from collections import Counter
import json

def process_lines(file_path, if_load_dominate = True):
    """
    处理每一行数据，解析并存储邻居信息及支配关系。
    
    :param lines: 包含数据行的列表，每行数据格式为 'id distance id1 id2 ...'
    :return: (nns, points_dominate_me)，其中：
             nns 是一个元组列表，每个元组包含 (neighbor_id, distance)；
             points_dominate_me 是一个列表，每个元素对应一个点，
             表示哪些点支配当前点。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    nns = []
    points_dominate_me = []

    for line in lines:
        parts = line.strip().split()
        first_part = int(parts[0])
        second_part = float(parts[1])
        nns.append((first_part, second_part))
        
        # 跳过距离字段，直接处理支配关系
        dominated_nns = list(map(int, parts[2:]))
        if if_load_dominate == True:
            points_dominate_me.append(dominated_nns)
        else:
            # TODO: We can see the result if there is no dominationship
            points_dominate_me.append([])
        
        # if first_part == 2597:
        #     print(f'Neighbor ID: {first_part}, Distance: {second_part}, NNs dominate it: {dominated_nns}')

    return nns, points_dominate_me

K = 8
pivot_id = 2048 # 2048
domination_file_path = '/Users/zhencan/WorkPlace/Serf_V2/simulation/sample_data/sampled_neighbors_domination.txt'
if_save = False
if_load_dominate = False
# Data Read
nns, points_dominate_me = process_lines(domination_file_path, if_load_dominate)

# Data Preprocessing
sorted_nns = sorted(nns, key=lambda x: x[0])
sorted_nn_ids = [x[0] for x in sorted_nns]
pivot_pos = bisect_left(sorted_nns, (pivot_id, 0))
if pivot_pos < len(sorted_nn_ids):
    pivot_nn_id_after = sorted_nn_ids[pivot_pos]
else:
    pivot_nn_id_after = pivot_id
print(f"基准点位置 {pivot_pos} 是在：{sorted_nn_ids[pivot_pos - 1]} and {pivot_nn_id_after} 之间的")
print(f"第一个点的nnid {sorted_nn_ids[0]}  and 最后一个点的nnid {sorted_nn_ids[-1]}")

def get_pruned_topk_nn_ids(start, end): # 闭区间
    """
    获取指定范围内，经过修剪的前K个最近邻的位置。
    
    :param sorted_nns: 排序后的最近邻列表。
    :param nns: 原始未排序的最近邻列表。
    :param points_dominate_me: 一个字典，键是点的索引，值是支配该点的所有点的列表。
    :param start: 子范围的起始索引。
    :param end: 子范围的结束索引。
    :param K: 需要返回的前K个位置的数量。
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
    
# Generate all the pruned topK values
top_k_min_hash_map = defaultdict(lambda: [[], 0])


# # Bruteforce Method
for i in range(0, pivot_pos + 1):
    for j in range(pivot_pos - 1, len(sorted_nns)):
        if i > j:
            continue
        # Get the topK values With Pruning
        pruned_topk_nnids = get_pruned_topk_nn_ids(i, j)
        
        # Add to the hash map
        key = hash_top_k_min_positions(pruned_topk_nnids)
        top_k_min_hash_map[key][0] = pruned_topk_nnids
        top_k_min_hash_map[key][1] += 1

# Output the analysis of topk distribution
analyze_top_k_distribution(top_k_min_hash_map)

# Save the result
if if_save:
    with open('topk_results/top_k_bruteforce.json', 'w') as f:
        json.dump(top_k_min_hash_map, f)
    
# Optimized Generating Method
dominate_nums = [0] * len(sorted_nns)
top_k_min_hash_map.clear()

# # Push the initial boundary into the heap
L_stack = []
L_stack.append(-1)
R_stack = []
R_stack.append(len(sorted_nns))

while L_stack[-1] <= R_stack[-1]:
    if L_stack[-1] == pivot_pos and pivot_pos == len(sorted_nns):
        break
    if pivot_pos == 0 and R_stack[-1] == -1:
        break    
    
    pruned_topk_nn_ids = get_pruned_topk_nn_ids(L_stack[-1], R_stack[-1])
    # Get the leftmost and rightmost positions
    sorted_topk_nnids = sorted(pruned_topk_nn_ids)
    leftmost_nn_id = sorted_topk_nnids[0]
    rightmost_nn_id = sorted_topk_nnids[-1]
    
    leftmost_pos = sorted_nn_ids.index(leftmost_nn_id)
    rightmost_pos = sorted_nn_ids.index(rightmost_nn_id)

    leftmost_rank = -1
    rightmost_rank = -1
    
    # Find the rank larger one
    if leftmost_pos < pivot_pos:
        leftmost_rank = pruned_topk_nn_ids.index(leftmost_nn_id)  #在pruned_topk_nn_ids中的位置（也就是rank）因为这个topk是按照距离排序的
    if rightmost_pos >= pivot_pos:
        rightmost_rank = pruned_topk_nn_ids.index(rightmost_nn_id)
        
    if rightmost_rank > leftmost_rank:
        # means we gonna skip rightmost one
        new_R = rightmost_pos - 1
        R_stack.append(new_R)
        
        # exist elements in the left 
        if leftmost_rank != -1:
            dominate_nums[leftmost_pos] += 1
            
        # release the dominated points by rightmost one
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
    
    # Add to the hash map
    key = hash_top_k_min_positions(pruned_topk_nn_ids)
    top_k_min_hash_map[key][0] = pruned_topk_nn_ids
    top_k_min_hash_map[key][1] += 1

# Output the analysis
analyze_top_k_distribution(top_k_min_hash_map)

# Save the result
if if_save:
    with open('topk_results/top_k_Optimization.json', 'w') as f:
        json.dump(top_k_min_hash_map, f)