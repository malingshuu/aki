import sys, os
# 确保先查找项目根目录
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from pandas.errors import ParserError
import numpy as np
from config import config
from sklearn.preprocessing import StandardScaler
import joblib


# 1) 读取原始数据，支持 Excel 和多编码 CSV
data_path = config['DATA_CSV']
if data_path.lower().endswith(('.xls', '.xlsx')):
    df = pd.read_excel(data_path, engine='openpyxl')
    print("[OK] 成功读取 Excel 文件")
else:
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            print(f"[OK] 使用编码 {enc} 成功读取 CSV 文件")
            break
        except (UnicodeDecodeError, ParserError):
            print(f"[WARN] 编码 {enc} 无法读取，尝试下一个…")
    if df is None:
        raise ValueError(f"无法使用以下编码读取文件：{encodings}")

# 2) 重命名标签列并生成二值特征
df = df.rename(columns={'急性肾损伤术后': 'label'})
df['高血压'] = ((df['高压（入院）'] >= 140) | (df['低压（入院）'] >= 90)).astype(int)
# ② 生成糖尿病（二值，基于空腹血糖）
bg = df['血糖（术前）'].copy()
# ——单位自动识别：若最大值明显 >40，说明是 mg/dL，需要换算——
if bg.max() > 40:
    bg = bg / 18.0  # mg/dL → mmol/L
# ——空腹阈值 7.0 mmol/L（126 mg/dL）——
df['糖尿病'] = (bg >= 7.0).astype(int)
# 仅保留必要列并填充缺失值
df = df[['序号', 'label'] + config['SELECTED_COLS']]
df = df.fillna(df.median())

# 3) 划分训练/验证集，仅用于训练集分层抽样
from sklearn.model_selection import train_test_split
train_df, _ = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
# 验证集使用全量
val_df = df.copy()

# 4) 提取并保存验证集特征&标签
X_val = val_df[config['SELECTED_COLS']].astype(float)
y_val = val_df['label'].values
ids   = val_df['序号'].values
np.save('X_val.npy', X_val.values)
np.save('val_labels.npy', y_val)
np.save('val_ids.npy', ids)
X_val.to_csv('X_val.csv', index=False)

print(f"[OK] 验证集准备完成：{X_val.shape[0]} 样本，{X_val.shape[1]} 特征")