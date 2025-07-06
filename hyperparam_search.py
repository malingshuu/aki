import copy
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import config
from dual_input_model import MultiModalFewShotNet
from few_shot.few_shot_trainer_proto import train_few_shot_proto, AKIDataset
from few_shot.proto_loss import FocalProtoLoss
from few_shot.episode_generator import create_episode

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
print("Current device:", device)
if device.type == 'cuda':
    print("GPU name:", torch.cuda.get_device_name(device))


def prepare_dataset():
    """
    读取并预处理数据，与 main.py 保持一致
    """
    df = pd.read_csv(config['DATA_CSV'])
    df = df.rename(columns={'急性肾损伤术后': 'label'})
    df['image_filename'] = df['序号'].astype(str) + '.jpg'
    for col in config['SELECTED_COLS']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df = df[['image_filename', 'label'] + config['SELECTED_COLS']].dropna()
    df['label'] = df['label'].astype(int)
    return df


def quick_validate(model, val_dataset, device, n_way=2, k_shot=5, q_query=5, best_t=0.5):
    """
    使用 AKIDataset 构造 support/query，并评估单折 F1
    """
    # 构造一个 episode
    support_set, query_set = create_episode(val_dataset, n_way, k_shot, q_query)

    # support 特征
    support_ds = AKIDataset(support_set, config['TRAIN_IMG_PATH'])
    sup_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    model.eval()
    with torch.no_grad():
        batch = next(iter(sup_loader))
        imgs, tabs, lbls = batch['image'].to(device), batch['tabular'].to(device), batch['label'].to(device)
        feats_s = model(imgs, tabs).cpu()
        lbls_s = lbls.cpu()

    # 计算原型
    prototypes = []
    for i in range(n_way):
        prototypes.append(feats_s[lbls_s == i].mean(dim=0))
    prototypes = torch.stack(prototypes)

    # query 特征 + 预测
    query_ds = AKIDataset(query_set, config['TRAIN_IMG_PATH'])
    qry_loader = DataLoader(query_ds, batch_size=len(query_ds), shuffle=False)
    with torch.no_grad():
        batch_q = next(iter(qry_loader))
        imgs_q, tabs_q, lbls_q = batch_q['image'].to(device), batch_q['tabular'].to(device), batch_q['label'].cpu().numpy()
        feats_q = model(imgs_q, tabs_q).cpu()
        dists = torch.cdist(feats_q, prototypes)
        probs = torch.softmax(-dists, dim=1)[:, 1].numpy()
        preds = (probs >= best_t).astype(int)

    return f1_score(lbls_q, preds)


def search():
    df = prepare_dataset()
    # 分层抽样取 50 条做验证集
    val_dataset, _ = train_test_split(
        df,
        test_size=50,
        random_state=42,
        stratify=df['label']
    )

    # 初始化模型并备份初始权重
    model = MultiModalFewShotNet(tabular_input_dim=len(config['SELECTED_COLS']))
    base_state = copy.deepcopy(model.state_dict())

    best_cfg = {'w': None, 'gamma': None, 'f1': 0.0}
    for w in [2.0, 3.0, 4.0]:
        for g in [1.5, 2.0, 2.5]:
            print(f"\n>>> Testing weight={w}, gamma={g}")
            model.load_state_dict(base_state)
            # 构造对应 Focal 原型损失
            cw = torch.tensor([1.0, w], dtype=torch.float32, device=device)
            criterion = FocalProtoLoss(gamma=g, weight=cw)
            # 快速训练少量 episode
            train_few_shot_proto(
                model=model,
                dataset=df,
                device=device,
                n_way=2, k_shot=5, q_query=5,
                num_episodes=30,
                criterion=criterion
            )
            # 验证集评估
            f1 = quick_validate(model, val_dataset, device)
            print(f"Config (w={w}, γ={g}) → F1: {f1:.4f}")
            if f1 > best_cfg['f1']:
                best_cfg.update({'w': w, 'gamma': g, 'f1': f1})

    print("\n=== Best config ===")
    print(best_cfg)


if __name__ == '__main__':
    search()
