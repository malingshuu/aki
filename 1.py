import seaborn as sns
import matplotlib.pyplot as plt

# 示例距离矩阵（对应tabular_feature_augment.py中的dist_matrix）
distance_matrix = [
    [0.0, 1.2, 3.4, 2.8],
    [1.2, 0.0, 2.9, 3.1],
    [3.4, 2.9, 0.0, 0.7],
    [2.8, 3.1, 0.7, 0.0]
]

# 绘制热力图
plt.figure(figsize=(8,6))
sns.heatmap(distance_matrix,
            annot=True,
            cmap="YlOrRd",  # 黄-橙-红色系
            fmt=".1f",
            linewidths=.5,
            vmin=0,
            vmax=4)
plt.title("Sample Distance Matrix\n(Euclidean Distance)", fontsize=12)
plt.savefig("distance_matrix.png", dpi=300)