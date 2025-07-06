import sys, os
# Ensure project root is first in PYTHONPATH
sys.path.insert(0, os.path.abspath('..'))
import joblib, os
import torch, shap, numpy as np
from dual_input_model import MultiModalFewShotNet
from config import config
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# 0. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n", flush=True)

# 1. Prompt: include image branch?
choice = input("是否加入图像分支？ 1: 加入  2: 不加入\n请输入选项: ")
include_image = (choice.strip() == '1')
print(f"Include image branch: {include_image}\n", flush=True)

# 2. Prompt: compute pixel-level SHAP?
compute_pixel = False
if include_image:
    choice2 = input("是否计算像素级图像SHAP？该过程较慢，1: 计算 2: 跳过\n请输入选项: ")
    compute_pixel = (choice2.strip() == '1')
    print(f"Compute pixel-level SHAP: {compute_pixel}\n", flush=True)

# SHAP availability
has_kernel = hasattr(shap, 'KernelExplainer')
has_explainer = hasattr(shap, 'Explainer')
has_gradient = hasattr(shap, 'GradientExplainer')

# 3. Load tabular data and labels
X_tab_all = np.load('X_val.npy')            # shape: (N, features)
id_list = np.load('val_ids.npy')           # shape: (N,)
y_tab_all = np.load('val_labels.npy')      # shape: (N,)
print(f"[INFO] 总共检测到 {X_tab_all.shape[0]} 个验证样本（表格）", flush=True)

# 4. Prompt: select sample size
def select_sample_size(y_tab_all, K=50):
    choice = input("选择样本数量：1. 随机50个样本  2. 全部样本\n请输入选项: ")
    use_all_samples = (choice.strip() == '2')
    if use_all_samples:
        return np.arange(len(y_tab_all))  # 使用全部样本
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=K, random_state=42)
        for _, sample_idx in splitter.split(np.zeros_like(y_tab_all), y_tab_all):
            return sample_idx

sampled_idx = select_sample_size(y_tab_all)

# 5. Subset arrays
X_tab_all = X_tab_all[sampled_idx]
id_list   = id_list[sampled_idx]
y_tab_all = y_tab_all[sampled_idx]

# 6. Prepare samples for SHAP
if include_image:
    img_dir = config['TRAIN_IMG_PATH']
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    paths, valid_idx = [], []
    for idx, seq in enumerate(id_list):
        seq_str = str(int(seq))
        for ext in exts:
            p = os.path.join(img_dir, seq_str + ext)
            if os.path.exists(p):
                paths.append(p)
                valid_idx.append(idx)
                break
    if not valid_idx:
        print("[WARN] 未找到任何图像，跳过图像分支", flush=True)
        include_image = False
        valid_idx = []
    K_img = len(valid_idx)
    valid_idx = valid_idx[:K_img]
    X_tab_sample = X_tab_all[valid_idx]
    paths = paths[:K_img]
else:
    X_tab_sample = X_tab_all
    K_img = 0

# 7. Image preprocessing if needed
if include_image:
    tf = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(config['IMG_MEAN'], config['IMG_STD'])
    ])
    imgs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        # optional cropping by annotation
        txt = os.path.splitext(p)[0] + '.txt'
        if os.path.exists(txt):
            with open(txt, 'r') as f:
                line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    cls, xc, yc, w, h = map(float, parts[:5])
                    W, H = img.size
                    xc, yc = xc * W, yc * H
                    w, h = w * W, h * H
                    left = max(0, xc - w/2); top = max(0, yc - h/2)
                    right = min(W, xc + w/2); bottom = min(H, yc + h/2)
                    img = img.crop((left, top, right, bottom))
        imgs.append(tf(img).unsqueeze(0))
    X_img_sample = torch.cat(imgs, dim=0).to(device)

# 8. Load model to GPU
tab_input_dim = len(config['SELECTED_COLS'])
model = MultiModalFewShotNet(tabular_input_dim=tab_input_dim)
model.load_state_dict(torch.load(config['MODEL_PATH'], map_location=device))
model.to(device).eval()
tab_enc = getattr(model, 'tabular_encoder', None) or model.tab_encoder
img_enc = getattr(model, 'image_encoder', None) or model.img_encoder

# 9. Precompute image embedding mean if needed
if include_image:
    with torch.no_grad():
        emb = img_enc(X_img_sample)
    img_mean = emb.mean(0, keepdim=True)

# 10. Define combined forward for SHAP
def combined_forward(x_np):
    x = torch.tensor(x_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        t_emb = tab_enc(x)
        if include_image:
            i_emb = img_mean.repeat(t_emb.size(0), 1)
            out = torch.cat([t_emb, i_emb], dim=1)
        else:
            out = t_emb
    return out.cpu().numpy()

# 11. Run embedding-level SHAP
background = X_tab_sample[:min(20, X_tab_sample.shape[0])]
shap_emb = None
if has_kernel:
    expl = shap.KernelExplainer(combined_forward, background)
    shap_emb = expl.shap_values(X_tab_sample)
elif has_explainer:
    expl = shap.Explainer(combined_forward, background, algorithm='kernel')
    shap_emb = expl(X_tab_sample).values
if shap_emb is not None:
    np.save('shap_tab_values.npy', shap_emb)
    print("[OK] Saved shap_tab_values.npy", flush=True)

# 12. Input-level SHAP for tabular features
if has_kernel:
    def tabular_forward(x_np):
        tx = torch.tensor(x_np, dtype=torch.float32).to(device)
        with torch.no_grad(): e = tab_enc(tx)
        return e[:,0].cpu().numpy()
    expl2 = shap.KernelExplainer(tabular_forward, background)
    stv = expl2.shap_values(X_tab_sample)
    np.save('shap_tab_input_values.npy', stv)
    print("[OK] Saved shap_tab_input_values.npy", flush=True)

# 13. Pixel-level image SHAP if chosen
if include_image and compute_pixel and has_gradient:
    grad = shap.GradientExplainer(img_enc, X_img_sample[:min(20, X_img_sample.shape[0])])
    siv = grad.shap_values(X_img_sample)
    np.save('shap_img_values.npy', siv)
    print("[OK] Saved shap_img_values.npy", flush=True)
elif include_image and compute_pixel:
    print("[WARN] 当前 SHAP 版本不支持 GradientExplainer，跳过像素级 SHAP", flush=True)

# 14. Save samples
np.save('X_tab_sample.npy', X_tab_sample)
if include_image:
    np.save('X_img_sample.npy', X_img_sample.cpu().numpy())

print(f"[OK] Completed SHAP for {len(X_tab_sample)} samples.", flush=True)