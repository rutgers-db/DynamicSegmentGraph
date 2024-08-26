# import matplotlib.pyplot as plt
# import numpy as np

# # 数据定义
# ranges = ["100", "500", "1K", "5K", "10K", "50K", "100K"]
# current_recall = [0.3338, 0.9878, 0.9980, 0.9984, 0.9985, 0.9977, 0.9832] # Current Method recall values
# current_qps = [55903, 5107, 3933, 2449, 2243, 1961, 2199] # Current Method QPS values
# baseline_recall = [0.2409, 0.9059, 0.9879, 0.9993, 0.9988, 0.9996, 0.9998] # Baseline recall values
# baseline_qps = [52108, 4698, 3766, 2531, 2059, 1365, 1107] # Baseline QPS values


# # 准备表格数据
# table_data = []
# header_row = ["Range", "Recall", "QPS"]
# for idx, rng in enumerate(ranges):
#     row = [
#         rng,
#         "{:.4f} / {:.4f}".format(baseline_recall[idx], current_recall[idx]),
#         "{} / {}".format(baseline_qps[idx], current_qps[idx])
#     ]
#     table_data.append(row)

# # 创建绘图
# fig, ax = plt.subplots(figsize=(10, 6))  # 图形大小

# # 隐藏坐标轴
# ax.axis('off')

# # 创建表格
# table = ax.table(
#     cellText=table_data,
#     colLabels=header_row,
#     cellLoc='center',
#     loc='center',
#     colWidths=[0.2, 0.4, 0.4]
# )

# # 设置表格样式
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1.5, 1.5)

# # 添加标题
# plt.title("Performance Metrics Comparison", y=1.08)

# # 显示图表
# plt.show()
import matplotlib.pyplot as plt

# 定义数据点
points = '100k'
avg_nn_baseline = 452
avg_nn_current_method = 96
sum_nn_baseline = '45.2M'
sum_nn_current_method = '9.6M'
index_time_baseline = '76s'
index_time_current_method = '35s'
index_size_baseline = '208Mb' 
index_size_current_method = '115Mb'

# 准备表格数据
table_data = [
    ['Points', points, points],
    ['Avg.nn', avg_nn_baseline, avg_nn_current_method],
    ['Sum.nn', sum_nn_baseline, sum_nn_current_method],
    ['Index Time', index_time_baseline, index_time_current_method],
    ['Index Size', index_size_baseline, index_size_current_method],
]

headers = ['Metric', 'Serf (MinLeap)', 'Current Method']


# 使用matplotlib绘制表格
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
table = ax.table(cellText=table_data[0:], colLabels=headers, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2) 
plt.title('Comparison between Baseline and Current Method for Index Properties')
plt.show()
