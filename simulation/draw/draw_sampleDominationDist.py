import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IO import process_lines

# 数据路径
DOMINATION_FILE_PATH = '/Users/zhencan/WorkPlace/Serf_V2/simulation/sample_data/sampled_neighbors_domination.txt'
IF_LOAD_DOMINATE = True

# 数据读取
nns, points_dominate_me = process_lines(DOMINATION_FILE_PATH, IF_LOAD_DOMINATE)
nn_ids = [x[0] for x in nns]
# 计算每个点被多少点 dominat
dominated_counts = [len(dominators) for dominators in points_dominate_me]

# 计算每个点 dominate 的其他点数量
dominating_counts = [0] * len(points_dominate_me)
for dominators in points_dominate_me:
    for dominator in dominators:
        dominator_idx = nn_ids.index(dominator)
        dominating_counts[dominator_idx] += 1

# 绘图
plt.figure(figsize=(12, 6))

# 绘制被支配点数量分布
plt.subplot(1, 2, 1)
plt.plot(range(len(dominated_counts)), dominated_counts, marker='o', linestyle='-', color='blue')
plt.title('Distribution of Points Dominate Me')
plt.xlabel('索引')
plt.ylabel('被支配点数')

# 绘制支配其他点数量分布
plt.subplot(1, 2, 2)
plt.plot(range(len(dominating_counts)), dominating_counts, marker='o', linestyle='-', color='red')
plt.title('Distribution of Dominating Points')
plt.xlabel('Number of Dominated Points')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 计算平均被支配数量
average_dominated = np.mean(dominated_counts)
average_dominating = np.mean(dominating_counts)
print(f'Average number of points dominating me: {average_dominated:.2f}')
print(f'Average number of points I dominate: {average_dominating:.2f}')

# 计算我不支配别人的点的数量 和 我不被任何人支配的点的数量
# 我不被任何人支配的点的数量 = dominated_counts 里的 0 的数量
# 我不支配别人的点的数量 = dominating_counts 里的 0 的数量
non_dominating_points = dominating_counts.count(0)
non_dominated_points = dominated_counts.count(0)

print(f'Number of points I do not dominate: {non_dominating_points}')
print(f'Number of points I am not dominated by anyone: {non_dominated_points}')




