import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve)
from sklearn.neural_network import MLPClassifier

from config import config
from dual_input_model import MultiModalFewShotNet

warnings.filterwarnings('ignore', category=UserWarning)
plt.switch_backend('Agg')

RESULT_DIR = config['RESULT_PATH']
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_prototypes(feats: torch.Tensor, lbls: torch.Tensor, n_cls: int = 2):
    return torch.stack([
        feats[lbls == c].mean(dim=0) if (lbls == c).any()
        else torch.zeros_like(feats[0])
        for c in range(n_cls)
    ])


def best_thresh_by_youden(y_true, prob):
    fpr, tpr, thresh = roc_curve(y_true, prob)
    return thresh[np.argmax(tpr - fpr)]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AKIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, cols):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.cols = cols
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pth = os.path.join(self.img_dir, row['image_filename'])
        try:
            img = Image.open(pth).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        img = self.tf(img)
        tab = torch.tensor(row[self.cols].values.astype(np.float32))
        label = int(row['label'])
        return img, tab, label

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(model, df, img_dir, cols, device):
    loader = DataLoader(
        AKIDataset(df, img_dir, cols), batch_size=64,
        shuffle=False, num_workers=4, pin_memory=True
    )
    model.eval()
    outs = []
    with torch.no_grad():
        for img, tab, _ in loader:
            outs.append(model(img.to(device), tab.to(device)).cpu())
    return torch.cat(outs)

# ---------------------------------------------------------------------------
# Image‑only CNN loader
# ---------------------------------------------------------------------------
def load_img_branch(path: str, device):
    cnn = models.resnet50(weights=None)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, 1)
    cnn.load_state_dict(torch.load(path, map_location=device))
    cnn.to(device).eval()
    return cnn

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def nested_cv_evaluate(img_dir: str, use_aug: bool = False):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load dataframe
    data_path = config['DATA_CSV']
    df = (
        pd.read_excel(data_path, engine='openpyxl')
        if data_path.lower().endswith(('xls', 'xlsx'))
        else pd.read_csv(data_path, encoding='gb18030')
    )
    df.rename(columns={'急性肾损伤术后': 'label'}, inplace=True)
    df['image_filename'] = df['序号'].astype(str) + '.jpg'
    # ① 生成高血压二值特征（阈值可调整）
    df['高血压'] = (
            (df['高压（入院）'] >= 140) |
            (df['低压（入院）'] >= 90)
    ).astype(int)
    # ② 生成糖尿病（二值，基于空腹血糖）
    bg = df['血糖（术前）'].copy()
    # ——单位自动识别：若最大值明显 >40，说明是 mg/dL，需要换算——
    if bg.max() > 40:
        bg = bg / 18.0  # mg/dL → mmol/L
    # ——空腹阈值 7.0 mmol/L（126 mg/dL）——
    df['糖尿病'] = (bg >= 7.0).astype(int)


    cols = config['SELECTED_COLS']
    for c in cols:
        df[c] = df[c].astype(str).str.replace(',', '.').astype(float)
    df = df[['image_filename', 'label'] + cols].fillna(df[cols].median())
    # —— 修改后：先强制转数值，再填 NaN（如读到“是”/“否”时才映射），最后转 int ——
    # 如果你的原始数据里只有 0 和 1，下面这一行就足够了：
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    # （可选）如果还要兼容中文“是/否”或其他文本：
    # df['label'] = df['label'].map({'是':1, '否':0}).fillna(df['label'])
     # df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    print('Label distribution:', df['label'].value_counts().to_dict())

    # 2. Standardize & augment
    df[cols] = StandardScaler().fit_transform(df[cols])
    if use_aug:
        from few_shot.tabular_feature_augment import augment_negative_samples
        df = augment_negative_samples(
            df,
            target_count=(df['label'] == 1).sum(),
            selected_cols=cols
        )
        print('After augmentation:', df['label'].value_counts().to_dict())

    # 3. Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proto_model = MultiModalFewShotNet(len(cols), 128, 0.2).to(device)
    proto_model.load_state_dict(
        torch.load(config['MODEL_PATH'], map_location=device)
    )

    tab_mlp_global = (
        joblib.load(config['TAB_SAVE_PATH'])
        if os.path.exists(config['TAB_SAVE_PATH']) else None
    )
    if tab_mlp_global is None:
        print(f'⚠️  {config["TAB_SAVE_PATH"]} not found → train per fold')

    img_cnn = (
        load_img_branch(config['IMG_SAVE_PATH'], device)
        if os.path.exists(config['IMG_SAVE_PATH']) else None
    )
    if img_cnn is None:
        print(f'⚠️  {config["IMG_SAVE_PATH"]} not found → skip image branch')

    # 4. Pre‑compute embeddings
    emb = extract_embeddings(proto_model, df, img_dir, cols, device)
    labels = df['label'].values
    full_ds = AKIDataset(df, img_dir, cols)

    skf_outer = StratifiedKFold(5, shuffle=True, random_state=42)
    metrics = {k: [] for k in ['acc', 'bal_acc', 'prec', 'rec', 'f1', 'auc', 'mcc']}
    all_probs, all_lbls = [], []
    global_thresh = None

    for fold, (tr_idx, te_idx) in enumerate(
        skf_outer.split(emb, labels), 1
    ):
        print(f'—— Fold {fold}/5 ——')
        e_trv, e_te = emb[tr_idx].to(device), emb[te_idx].to(device)
        y_trv, y_te = labels[tr_idx], labels[te_idx]
        df_trv, df_te = df.iloc[tr_idx], df.iloc[te_idx]

        # Prototype branch
        proto = compute_prototypes(
            e_trv, torch.tensor(y_trv, device=device)
        )
        p_proto_te = torch.softmax(
            -torch.cdist(e_te, proto), dim=1
        )[:, 1].cpu().numpy()

        # Tabular branch
        if tab_mlp_global is not None:
            p_tab_te = tab_mlp_global.predict_proba(
                df_te[cols]
            )[:, 1]
            p_tab_tr = tab_mlp_global.predict_proba(
                df_trv[cols]
            )[:, 1]
        else:
            tab_mlp_local = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42
            )
            tab_mlp_local.fit(df_trv[cols], y_trv)
            p_tab_te = tab_mlp_local.predict_proba(
                df_te[cols]
            )[:, 1]
            p_tab_tr = tab_mlp_local.predict_proba(
                df_trv[cols]
            )[:, 1]

        # Image branch
        if img_cnn is not None:
            def img_probs(indices):
                loader = DataLoader(
                    Subset(full_ds, indices), batch_size=64,
                    shuffle=False
                )
                out = []
                img_cnn.eval()
                with torch.no_grad():
                    for img, _, _ in loader:
                        out.append(torch.sigmoid(
                            img_cnn(img.to(device)).squeeze(1)
                        ).cpu())
                return torch.cat(out).numpy()
            p_img_tr = img_probs(tr_idx)
            p_img_te = img_probs(te_idx)
        else:
            p_img_tr = np.zeros_like(p_tab_tr)
            p_img_te = np.zeros_like(p_tab_te)

        # Ensemble
        p_ens_te = (p_proto_te + p_tab_te + p_img_te) / 3
        #p_ens_te = p_proto_te
        all_probs.extend(p_ens_te.tolist())
        all_lbls.extend(y_te.tolist())

        # Global threshold (Youden)
        if global_thresh is None:
            p_proto_tr = torch.softmax(
                -torch.cdist(e_trv, proto), dim=1
            )[:, 1].cpu().numpy()
            p_ens_tr = (p_proto_tr + p_tab_tr + p_img_tr) / 3
            global_thresh = best_thresh_by_youden(y_trv, p_ens_tr)
            print(f'◎ Global Youden threshold = {global_thresh:.3f}')

        preds = (p_ens_te >= global_thresh).astype(int)

        # Metrics
        acc = accuracy_score(y_te, preds)
        bal = balanced_accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds)
        f1 = f1_score(y_te, preds)
        aucv = roc_auc_score(y_te, p_ens_te)
        mccv = matthews_corrcoef(y_te, preds)
        for k, v in zip(metrics.keys(), [acc, bal, prec, rec, f1, aucv, mccv]):
            metrics[k].append(v)
        print(f'  Acc={acc:.3f}, AUC={aucv:.3f}, Prec={prec:.3f}, Rec={rec:.3f}')

        # Per-fold ROC
        fpr, tpr, _ = roc_curve(y_te, p_ens_te)
        plt.figure(); plt.plot(fpr, tpr, label=f'Fold{fold} AUC={aucv:.2f}');
        plt.plot([0,1],[0,1],'--'); plt.legend();
        plt.savefig(os.path.join(RESULT_DIR, f'roc_fold{fold}.png'));
        plt.close()

        # Per-fold confusion matrix
        cm = confusion_matrix(y_te, preds)
        plt.figure(); plt.imshow(cm, cmap='Blues');
        plt.title(f'CM Fold{fold}'); plt.colorbar();
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i,j], ha='center', va='center',
                         color='white' if cm[i,j]>cm.max()/2 else 'black')
        plt.xlabel('Pred'); plt.ylabel('True');
        plt.savefig(os.path.join(RESULT_DIR, f'cm_fold{fold}.png'));
        plt.close()

    # Summary
    print('=== Summary ===')
    for k, v in metrics.items():
        print(f'{k}: {np.mean(v):.3f} ± {np.std(v):.3f}')

    # Overall ROC
    fpr_all, tpr_all, _ = roc_curve(all_lbls, all_probs)
    plt.figure(); plt.plot(fpr_all, tpr_all, label='Overall ROC');
    plt.plot([0,1],[0,1],'--'); plt.legend();
    plt.savefig(os.path.join(RESULT_DIR, 'roc_overall.png'));
    plt.close()


if __name__ == '__main__':
    choice = input('是否进行数据增强? (1-否, 2-是): ').strip() == '2'
    nested_cv_evaluate(config['TRAIN_IMG_PATH'], use_aug=choice)
