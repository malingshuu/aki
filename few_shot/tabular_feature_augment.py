# 文件：tabular_feature_augment.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from config import config
from few_shot.swav_anchor import SwAVAnchorExtractor
import torch
import torch.nn.functional as F


def augment_positive_samples(df, n_aug=1, dist_thresh=2.5, visualize=True, max_total=None):
    """
    基于表格特征对正类样本进行增强：
    - df 包含 'label', 'image_filename' 以及 config['SELECTED_COLS'] 中的列
    - n_aug: 每对样本生成的增强样本数
    - dist_thresh: 只对距离 <= 阈值的对进行增强
    - visualize: 是否保存距离直方图
    - max_total: 增强后正类样本的最大总数，None表示不限制
    返回扩增后的 DataFrame，如果没有有效对则直接返回原 df。
    """
    pos_df = df[df['label'] == 1].reset_index(drop=True)
    neg_df = df[df['label'] == 0].reset_index(drop=True)
    
    # 如果正类样本数量已经很多，可以考虑不增强
    if max_total and len(pos_df) >= max_total:
        print(f"正类样本数量({len(pos_df)})已达到设定上限({max_total})，跳过增强")
        return df
        
    if len(pos_df) < 2:
        print("Warning: not enough positive samples to augment.")
        return df

    # 准备原始特征矩阵
    selected_cols = config['SELECTED_COLS']
    X_raw = pos_df[selected_cols].astype(np.float32).values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 计算两两距离，仅取上三角
    dist_matrix = pairwise_distances(X, metric='euclidean')
    idxs = np.triu_indices_from(dist_matrix, k=1)
    dist_vals = dist_matrix[idxs]

    # 过滤 NaN
    valid = dist_vals[~np.isnan(dist_vals)]
    if valid.size == 0:
        print("Warning: no valid distances, skip augmentation.")
        return df

    # 打印距离范围
    print(f"Distance range among positive samples: min={valid.min():.3f}, max={valid.max():.3f}")

    # 可视化距离分布
    if visualize:
        os.makedirs(config['RESULT_PATH'], exist_ok=True)
        plt.figure()
        plt.hist(valid, bins=30, edgecolor='black')
        plt.title("Positive–Positive Distance Distribution")
        plt.savefig(os.path.join(config['RESULT_PATH'], 'pp_distance_hist.png'))
        plt.close()

    # 优化：使用自适应距离阈值
    if dist_thresh == 'auto':
        # 自动选择距离阈值为距离分布的25%分位数
        dist_thresh = np.percentile(valid, 25)
        print(f"自动选择距离阈值: {dist_thresh:.3f}")
    
    # 筛选距离小于等于阈值的对
    mask = valid <= dist_thresh
    if not mask.any():
        print(f"Warning: no pairs under dist_thresh={dist_thresh}, skip augmentation.")
        return df
    pairs = list(zip(idxs[0][mask], idxs[1][mask]))
    
    # 对于小样本学习，限制生成的对数
    max_pairs = min(len(pairs), 50)  # 最多使用50对样本进行增强
    pairs = pairs[:max_pairs]
    print(f"使用 {len(pairs)} 对样本进行增强 (从 {len(mask[mask])} 对符合条件的样本中选择)")

    # 线性插值生成增强样本
    aug_rows = []
    for i, j in pairs:
        for _ in range(n_aug):
            alpha = np.random.rand()
            new_feat = alpha * X_raw[i] + (1 - alpha) * X_raw[j]
            row = pos_df.iloc[i].copy()
            for k, col in enumerate(selected_cols):
                row[col] = float(new_feat[k])
            aug_rows.append(row)

    # 在生成增强样本后，检查总数是否超过上限
    if max_total and len(pos_df) + len(aug_rows) > max_total:
        # 只取部分增强样本，使总数不超过上限
        samples_to_take = max_total - len(pos_df)
        if samples_to_take > 0:
            aug_rows = aug_rows[:samples_to_take]
        else:
            aug_rows = []
    
    # 合并增强样本
    if not aug_rows:
        return df
        
    aug_df = pd.DataFrame(aug_rows)
    aug_df['label'] = 1
    aug_df['image_filename'] = np.random.choice(pos_df['image_filename'], size=len(aug_df))

    df_new = pd.concat([neg_df, pos_df, aug_df], ignore_index=True)
    print(f"🔁 正类样本扩增：原始 {len(pos_df)} → 增强后 {len(aug_df)} → 总共 {len(df_new[df_new['label']==1])}")
    return df_new

# 添加负类样本增强函数
def augment_negative_samples(df, target_count, selected_cols, dist_thresh=2.5, visualize=True):
    """
    基于表格特征对负类样本进行增强：
    - df 包含 'label', 'image_filename' 以及 selected_cols 中的列
    - target_count: 目标负类样本数量
    - selected_cols: 用于增强的特征列
    - dist_thresh: 只对距离 <= 阈值的对进行增强
    - visualize: 是否保存距离直方图
    返回扩增后的 DataFrame，如果没有有效对则直接返回原 df。
    """
    neg_df = df[df['label'] == 0].reset_index(drop=True)
    pos_df = df[df['label'] == 1].reset_index(drop=True)
    
    # 如果负类样本已经足够，直接返回原始数据
    if len(neg_df) >= target_count:
        return df
    
    if len(neg_df) < 2:
        print("Warning: not enough negative samples to augment.")
        return df

    # 准备原始特征矩阵
    X_raw = neg_df[selected_cols].astype(np.float32).values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 计算两两距离，仅取上三角
    dist_matrix = pairwise_distances(X, metric='euclidean')
    idxs = np.triu_indices_from(dist_matrix, k=1)
    dist_vals = dist_matrix[idxs]

    # 过滤 NaN
    valid = dist_vals[~np.isnan(dist_vals)]
    if valid.size == 0:
        print("Warning: no valid distances, skip augmentation.")
        return df

    # 打印距离范围
    print(f"Distance range among negative samples: min={valid.min():.3f}, max={valid.max():.3f}")

    # 可视化距离分布
    if visualize:
        os.makedirs(config['RESULT_PATH'], exist_ok=True)
        plt.figure()
        plt.hist(valid, bins=30, edgecolor='black')
        plt.title("Negative–Negative Distance Distribution")
        plt.savefig(os.path.join(config['RESULT_PATH'], 'nn_distance_hist.png'))
        plt.close()

    # 自动选择距离阈值为距离分布的25%分位数
    if dist_thresh == 'auto':
        dist_thresh = np.percentile(valid, 25)
        print(f"自动选择距离阈值: {dist_thresh:.3f}")

    # 筛选距离小于等于阈值的对
    mask = valid <= dist_thresh
    if not mask.any():
        print(f"Warning: no pairs under dist_thresh={dist_thresh}, skip augmentation.")
        return df
    pairs = list(zip(idxs[0][mask], idxs[1][mask]))
    
    # 对于小样本学习，限制生成的对数
    max_pairs = min(len(pairs), 50)  # 最多使用50对样本进行增强
    pairs = pairs[:max_pairs]
    print(f"使用 {len(pairs)} 对样本进行增强 (从 {len(mask[mask])} 对符合条件的样本中选择)")

    # 计算需要的增强样本数量
    samples_needed = target_count - len(neg_df)
    n_aug = max(1, int(np.ceil(samples_needed / len(pairs))))
    
    # 线性插值生成增强样本
    aug_rows = []
    for i, j in pairs:
        for _ in range(n_aug):
            alpha = np.random.rand() * 0.8 + 0.1  # 限制在0.1-0.9之间
            new_feat = alpha * X_raw[i] + (1 - alpha) * X_raw[j]
            row = neg_df.iloc[i].copy()
            for k, col in enumerate(selected_cols):
                row[col] = float(new_feat[k])
            aug_rows.append(row)
            
            # 如果已经生成足够的样本，就停止
            if len(aug_rows) >= samples_needed:
                break
        if len(aug_rows) >= samples_needed:
            break

    # 合并增强样本
    if not aug_rows:
        return df
        
    aug_df = pd.DataFrame(aug_rows)
    aug_df['label'] = 0  # 确保标签为负类
    
    # 确保image_filename列存在
    if 'image_filename' in neg_df.columns:
        aug_df['image_filename'] = np.random.choice(neg_df['image_filename'], size=len(aug_df))

    df_new = pd.concat([pos_df, neg_df, aug_df], ignore_index=True)
    print(f"🔁 负类样本扩增：原始 {len(neg_df)} → 增强后 {len(aug_df)} → 总共 {len(df_new[df_new['label']==0])}")
    return df_new




