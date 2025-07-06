import sys, os
# 确保能导入项目根目录的模块
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from config import config

# 1. 加载表格特征级 SHAP 值和数据
try:
    shap_tab_in = np.load('shap_tab_input_values.npy', allow_pickle=True)
    X_tab_sample = np.load('X_tab_sample.npy')
except FileNotFoundError:
    print("[ERROR] 找不到表格特征级 SHAP 输出文件：shap_tab_input_values.npy 或 X_tab_sample.npy")
    sys.exit(1)

# 2. 加载特征名称
feature_names = config['SELECTED_COLS']
if len(feature_names) != shap_tab_in.shape[1]:
    print(f"[WARN] 特征名数量({len(feature_names)})与 SHAP 数据维度({shap_tab_in.shape[1]})不匹配，可能可视化异常")

# 3. 计算平均绝对 SHAP 值并排序
mean_abs = np.mean(np.abs(shap_tab_in), axis=0)
order = np.argsort(mean_abs)[::-1]
sorted_means = mean_abs[order]
sorted_features = [feature_names[i] for i in order]

# 4. 绘制水平柱状图
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(sorted_features, sorted_means)
ax.invert_yaxis()
ax.set_title('特征重要性（平均 |SHAP值|）', fontsize=14)
ax.set_xlabel('平均 |SHAP值|（模型输出幅度平均影响）', fontsize=12)

# 5. 保存可视化结果到文件
output_path = 'shap_tab_input_summary.png'
fig.savefig(output_path, dpi=300)
print(f"[OK] 已保存 SHAP 可视化图片：{output_path}", flush=True)

# 6. 显示
plt.tight_layout()
plt.show()

# ===== 1. 蜂群图 (Bee Swarm Plot) =====
plt.figure()
shap.summary_plot(shap_tab_in, X_tab_sample, feature_names=feature_names, show=False)
plt.title("特征影响分布", fontsize=14)
plt.tight_layout()
plt.savefig('shap_bee_swarm.png', dpi=300)
plt.close()

# ===== 2. 单个样本决策图 =====
sample_idx = 0  # 选择第一个样本，可改为其他有代表性的索引
plt.figure()
shap.force_plot(
    np.mean(shap_tab_in, axis=0),
    shap_tab_in[sample_idx],
    X_tab_sample[sample_idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title(f"样本 {sample_idx} 的决策过程", fontsize=12)
plt.tight_layout()
plt.savefig('shap_force_plot_sample.png', dpi=300)
plt.close()

# ===== 3. 依赖图 =====
if len(feature_names) > 1:  # 只有多个特征时才绘制依赖图
    top_feature = sorted_features[0]
    plt.figure()
    shap.dependence_plot(
        top_feature,
        shap_tab_in,
        X_tab_sample,
        feature_names=feature_names,
        interaction_index='auto',
        show=False
    )
    plt.title(f"{top_feature} 的SHAP值依赖关系", fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_dependence_plot.png', dpi=300)
    plt.close()

# ===== 4. 交互式HTML报告 =====
shap.initjs()
force_plot_html = shap.force_plot(
    np.mean(shap_tab_in, axis=0),
    shap_tab_in,
    X_tab_sample,
    feature_names=feature_names
)
shap.save_html('shap_interactive_force_plot.html', force_plot_html)

# 继续原有代码
plt.tight_layout()
plt.show()


# 在 4_visualize_shap.py 的 plt.show() 之后添加：

# ===== 图像SHAP可视化 =====
def visualize_image_shap():
    if not (os.path.exists('shap_img_values.npy') and os.path.exists('X_img_sample.npy')):
        print("[INFO] 未检测到图像SHAP数据，跳过图像可视化")
        return

    shap_img = np.load('shap_img_values.npy')
    X_img = np.load('X_img_sample.npy')

    print(f"\n[INFO] 开始可视化图像SHAP（共{len(X_img)}个样本）")

    # 创建保存目录
    os.makedirs('shap_img_plots', exist_ok=True)

    # 可视化前3个样本
    for i in range(min(3, len(X_img))):
        plt.figure(figsize=(12, 5))

        # 原始图像
        plt.subplot(1, 2, 1)
        img = X_img[i].transpose(1, 2, 0)
        # 反归一化（假设使用ImageNet均值和标准差）
        img = img * np.array(config['IMG_STD']) + np.array(config['IMG_MEAN'])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"样本 {i} 原始图像")
        plt.axis('off')

        # SHAP热力图
        plt.subplot(1, 2, 2)
        shap.image_plot([shap_img[i]], [img], show=False)
        plt.title("SHAP热力图（红色表示正影响）")

        plt.tight_layout()
        plt.savefig(f'shap_img_plots/sample_{i}_shap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"-> 已保存样本 {i} 的SHAP可视化")


# 执行图像可视化
visualize_image_shap()
