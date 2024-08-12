import matplotlib.pyplot as plt
import numpy as np

# 数据定义
ranges = ["1K", "5K", "10K", "50K", "100K", "500K", "1M"]
baseline_recall = [0.1774, 0.7573, 0.9150, 0.9781, 0.9787, 0.9736, 0.9806]
current_recall = [0.0196, 0.8067, 0.9322, 0.9892, 0.9902, 0.9907, 0.9917]
baseline_qps = [23961, 4436, 3463, 2162, 1844, 1358, 1136]
current_qps = [81800, 4032, 3200, 1841, 1467, 921, 710]

# 准备表格数据
table_data = []
header_row = ["Range", "Recall", "QPS"]
for idx, rng in enumerate(ranges):
    row = [
        rng,
        "{:.4f} / {:.4f}".format(baseline_recall[idx], current_recall[idx]),
        "{} / {}".format(baseline_qps[idx], current_qps[idx])
    ]
    table_data.append(row)

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 6))  # 图形大小

# 隐藏坐标轴
ax.axis('off')

# 创建表格
table = ax.table(
    cellText=table_data,
    colLabels=header_row,
    cellLoc='center',
    loc='center',
    colWidths=[0.2, 0.4, 0.4]
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)

# 添加标题
plt.title("Performance Metrics Comparison", y=1.08)

# 显示图表
plt.show()
