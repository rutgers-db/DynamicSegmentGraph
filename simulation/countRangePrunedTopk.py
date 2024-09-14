from bisect import bisect_left
from countRangeTopkClosest import hash_top_k_min_positions, calculate_range_percentage_distribution
from collections import defaultdict
from collections import Counter
import json
from IO import process_lines


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