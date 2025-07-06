import os
import pandas as pd
import torch
import os; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from few_shot.few_shot_trainer_proto import train_few_shot_proto  # 添加这个导入
from few_shot.proto_loss import compute_prototypical_loss, FocalProtoLoss
from dual_input_model import MultiModalFewShotNet
from config import config
from sklearn.neural_network import MLPClassifier
import joblib
from torchvision import models
from PIL import Image
from torchvision import transforms
import torch.nn as nn
# 中文显示配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

def main(use_augmentation=False):
    print("加载数据并启动 Few-Shot 训练...")
    # 读取并预处理数据
    data_path = config['DATA_CSV']
    if data_path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data_path, engine='openpyxl')
    else:
        df = pd.read_csv(data_path, encoding='gb18030')
    
    # 只使用原始特征，不添加额外特征
    df = df.rename(columns={'急性肾损伤术后': 'label'})
    df['image_filename'] = df['序号'].astype(str) + ".jpg"

    # ① 生成高血压二值特征（阈值可调整）
    df['高血压'] = (
            (df['高压（入院）'] >= 140) |
            (df['低压（入院）'] >= 90)
    ).astype(int)
    # ② 生成糖尿病（二值，基于空腹血糖）
    bg = df['血糖（术前）'].copy()
    # ——单位自动识别：若最大值明显 >40，说明是 mg/dL，需要换算——
    if bg.max() > 40:
        bg = bg / 18.0           # mg/dL → mmol/L
    # ——空腹阈值 7.0 mmol/L（126 mg/dL）——
    df['糖尿病'] = (bg >= 7.0).astype(int)


    # 使用config中定义的特征列
    selected_cols = config['SELECTED_COLS']
    print(f"使用的特征列: {selected_cols}")
    
    # 数据预处理
    for col in selected_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    # 只保留需要的列并处理缺失值
    df = df[['image_filename', 'label'] + selected_cols].copy()
    
    # 使用中位数填充缺失值
    for col in selected_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)


    # 修复：确保label列的数据类型一致
    df['label'] = df['label'].astype(str)
    positive_values = ['1', '1.0', '是', 'yes', 'true', 'True']
    df['label'] = df['label'].apply(lambda x: 1 if x.lower() in [v.lower() for v in positive_values] else 0)
    print(f"处理后的label值分布: {df['label'].value_counts()}")

    # 统计原始数据分布
    orig_pos = int((df['label'] == 1).sum())
    orig_neg = int((df['label'] == 0).sum())
    print(f"原始数据分布：正类 {orig_pos}，负类 {orig_neg}，总计 {len(df)}")


    # ——① 先用原始分布算类别权重 ——
    total = orig_pos + orig_neg
    w0 = total / (2 * orig_neg)
    w1 = total / (2 * orig_pos)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"计算权重: 负类={w0:.3f}, 正类={w1:.3f}")


    # —— 数据增强（只改变样本，不影响已算的 class_weights） ——
    if use_augmentation:
        from few_shot.tabular_feature_augment import augment_negative_samples
        df = augment_negative_samples(
            df,
            target_count=orig_pos,
            selected_cols=selected_cols
        )
        print("增强后 label 值分布：")
        print(df['label'].value_counts())




    # 构建 Few-Shot 模型
    model = MultiModalFewShotNet(
        tabular_input_dim=len(selected_cols),
        embedding_dim=128,
        dropout_rate=0.2
    )

    criterion = FocalProtoLoss(weight=class_weights)
    print(f"使用权重: 负类={w0:.3f}, 正类={w1:.3f}")


    # 训练
    train_few_shot_proto(
        model=model,
        dataset=df,
        device=device,
        n_way=2,
        k_shot=10,
        q_query=10,
        num_episodes=config['EPOCHS'],
        criterion=criterion,
        learning_rate=1e-4,
        weight_decay=0.0001,
        use_scheduler=True
    )

    # 保存模型
    os.makedirs(os.path.dirname(config['MODEL_SAVE_PATH']), exist_ok=True)
    torch.save(model.state_dict(), config['MODEL_SAVE_PATH'])
    print(f"✅ 模型已保存到: {config['MODEL_SAVE_PATH']}")



if __name__ == "__main__":
    # 手动选择是否进行增强
    print("请选择训练方式：")
    print("选项1：不使用增强直接训练")
    print("选项2：使用增强后训练")
    choice = input("请输入选项（1 或 2）：")

    if choice == '2':
        use_augmentation = True
    else:
        use_augmentation = False

    main(use_augmentation)