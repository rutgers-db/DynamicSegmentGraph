# 这个script 是用来simulate 在HNSW 压缩的时候
# 我已经得到了ANNSearch的KNN结果
# 在这K个元素里进行压缩该怎么做

from bisect import bisect_left
from collections import defaultdict
import json
from IO import process_lines
from util import hash_top_k_min_positions, analyze_top_k_distribution
import time

# 全局常量定义
M = 8
PIVOT_ID = 2048  # 基准点ID
DOMINATION_FILE_PATH = '/Users/zhencan/WorkPlace/Serf_V2/simulation/sample_data/sampled_neighbors_domination.txt'
IF_SAVE = False
IF_LOAD_DOMINATE = True

# 数据读取
nns, points_dominate_me = process_lines(DOMINATION_FILE_PATH, IF_LOAD_DOMINATE)

# 数据预处理
sorted_nns = sorted(nns, key=lambda x: x[0])
sorted_nn_ids = [x[0] for x in sorted_nns]
pivot_pos = bisect_left(sorted_nns, (PIVOT_ID, 0))
max_nn_id = sorted_nn_ids[-1]

# 确定基准点附近的邻居ID
if pivot_pos < len(sorted_nn_ids):
    pivot_nn_id_after = sorted_nn_ids[pivot_pos]
else:
    pivot_nn_id_after = PIVOT_ID

print(f"基准点位置 {pivot_pos} 是在：{sorted_nn_ids[pivot_pos - 1]} 和 {pivot_nn_id_after} 之间")
print(f"第一个点的nnid {sorted_nn_ids[0]} 和 最后一个点的nnid {sorted_nn_ids[-1]}")

# DFS traverse all the pruned_topknn
current_V = [True] * len(nns) # initiliaze current_V as an array with len(nns_length) full of true
top_k_min_hash_map = defaultdict(lambda: [[], 0])
cal_domination_count = 0 # 计算 domination 的次数

# calculate how many calculation is needed
# initilize a dictionary to store the calculate pair key is a pair of int and the value is default 0
calculate_pair = defaultdict(lambda: 0)
prefix_set = set()

def dfs(prefix_nbr_idx, M, L, R, lr, rl):
    global cal_domination_count, top_k_min_hash_map, current_V
    if len(prefix_nbr_idx) == M:
        prefix_nbr = [nns[i][0] for i in prefix_nbr_idx]
        key = hash_top_k_min_positions(prefix_nbr)
        top_k_min_hash_map.setdefault(key, [None, 0])
        top_k_min_hash_map[key][0] = prefix_nbr
        top_k_min_hash_map[key][1] += 1
        return

    last_nbr_idx = prefix_nbr_idx[-1] if prefix_nbr_idx else -1
    for i in range(last_nbr_idx + 1, len(nns)):
        if nns[i][0] <= R and nns[i][0] >= L:
            if current_V[i]:
                # check new dominate
                tmp_dominated_arr = []
                for j in range(i+1, len(nns)):
                    if nns[j][0] <= R and nns[j][0] >= L:
                        if current_V[j]:
                            if nns[i][0] in points_dominate_me[j]:
                                current_V[j] = False
                                tmp_dominated_arr += [j]
                                cal_domination_count += 1
                                cal_pair = i*len(nns) + j
                                calculate_pair[cal_pair] += 1
                            
                # 更新左侧范围边界
                next_lr = min(nns[i][0], lr) if nns[i][0] < PIVOT_ID else lr
                # 更新右侧范围边界
                next_rl = max(nns[i][0], rl) if nns[i][0] > PIVOT_ID else rl
                
                dfs(prefix_nbr_idx + [i], M, L, R, next_lr, next_rl)
                prefix_set.add(hash_top_k_min_positions(prefix_nbr_idx + [i]))
                # uncheck domination
                for j in tmp_dominated_arr:
                    current_V[j] = True
                            
                # update L and R
                if nns[i][0] < lr:
                    L = nns[i][0] + 1
                elif nns[i][0] > rl:
                    R = nns[i][0] - 1
                else:
                    break
                    
                

start_time = time.time()  # 记录开始时间
dfs([], M, -1, max_nn_id + 1, PIVOT_ID, PIVOT_ID)
end_time = time.time()  # 记录结束时间
print(f"函数耗时: {end_time - start_time:.6f} 秒")
print(f"计算domination的次数: {cal_domination_count}")
# 输出分析结果
analyze_top_k_distribution(top_k_min_hash_map)
 # 打印 calculate_pair 的长度
print(f"calculate_pair 的长度: {len(calculate_pair)}")

# 打印 prefix_set 的长度
print(f"prefix_set 的长度: {len(prefix_set)}")
# print(top_k_min_hash_map)
if IF_SAVE:
        file_path = 'topk_results/top_k_DFS.json'
        with open(file_path, 'w') as f:
            json.dump(top_k_min_hash_map, f)
