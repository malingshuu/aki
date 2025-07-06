import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from few_shot.episode_generator import create_episode
from few_shot.proto_loss import FocalProtoLoss, anchor_alignment_loss
from few_shot.swav_anchor import SwAVAnchorExtractor
from few_shot.anchor_feature_generator import AnchorFeatureGenerator
from sklearn.cluster import KMeans
import pandas as pd
from config import config
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
print("Current device:", device)
if device.type == 'cuda':
    print("GPU name:", torch.cuda.get_device_name(device))


def compute_multi_prototypes(features: torch.Tensor,
                             labels:   torch.Tensor,
                             n_classes: int = 2,
                             k_protos:  int = 3):
    """
    每类生成 k_protos 个原型，返回 list，长度 = n_classes，
    每个元素 Tensor 形状 [k_protos, dim]。
    """
    protos = []
    feats_np = features.detach().cpu().numpy()
    lbls_np  = labels.cpu().numpy()
    for c in range(n_classes):
        # 取出该类对应的 feature Tensor
        cls_feats = features[labels == c]  # Tensor [Nc, D]
        D = features.size(1)
        if cls_feats.size(0) == 0:
            centers = torch.zeros((k_protos, D), device=features.device)
        else:
            n_centers = min(k_protos, cls_feats.size(0))
            array_feats = cls_feats.detach().cpu().numpy()
            km = KMeans(n_clusters=n_centers, random_state=0).fit(array_feats)
            centers = torch.tensor(km.cluster_centers_, device=features.device, dtype=torch.float32)
            if n_centers < k_protos:
                avg = cls_feats.mean(dim=0, keepdim=True)
                extra = avg.repeat(k_protos - n_centers, 1)
                centers = torch.cat([centers, extra], dim=0)

        protos.append(F.normalize(centers, p=2, dim=1))

    return protos


def batch_hard_triplet_loss(embeddings: torch.Tensor,
                            labels:     torch.Tensor,
                            margin:     float = 0.5):
    """
    手动实现 batch-hard triplet loss：
    对每个样本， hardest positive: 同类最大距离；
                     hardest negative: 异类最小距离。
    """
    # pairwise distance
    dist_mat = torch.cdist(embeddings, embeddings)
    loss = 0.0
    count = 0
    for i in range(labels.size(0)):
        label = labels[i]
        mask_pos = (labels == label)
        mask_neg = (labels != label)
        # hardest positive (exclude self)
        pos_dists = dist_mat[i][mask_pos]
        # remove zero self-distance
        pos_dists = pos_dists[pos_dists > 0] if pos_dists.numel() > 1 else pos_dists
        hardest_pos = pos_dists.max() if pos_dists.numel() > 0 else torch.tensor(0.0, device=embeddings.device)
        # hardest negative
        neg_dists = dist_mat[i][mask_neg]
        hardest_neg = neg_dists.min() if neg_dists.numel() > 0 else torch.tensor(0.0, device=embeddings.device)
        loss += F.relu(hardest_pos - hardest_neg + margin)
        count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=embeddings.device)


class AKIDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.selected_cols = config['SELECTED_COLS']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_filename'])
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        # 根据图片路径构造对应的 .txt 标注路径
        ann_path = img_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(ann_path):
            # 逐行读 YOLO 格式：class x_center y_center width height（都归一化到 [0,1]）
            with open(ann_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                boxes = []
                W,H = img.size
                for line in lines:
                    parts = line.split()
                    if len(parts)>=5:
                        _,x_c,y_c,w_r,h_r = map(float, parts[:5])
                        # 把归一化坐标转换成像素坐标并收集所有 box
                        x1 = int((x_c-w_r/2)*W); y1=int((y_c-h_r/2)*H)
                        x2 = int((x_c+w_r/2)*W); y2=int((y_c+h_r/2)*H)
                        boxes.append((max(0,x1),max(0,y1),min(W,x2),min(H,y2)))
                # 如果解析出了 box，就把图片裁剪到所有 box 的并集区域
                if boxes:
                    x1,y1=min(b[0] for b in boxes),min(b[1] for b in boxes)
                    x2,y2=max(b[2] for b in boxes),max(b[3] for b in boxes)
                    img=img.crop((x1,y1,x2,y2))
        img = self.transform(img)
        tab = torch.tensor(row[self.selected_cols].astype(np.float32).values, dtype=torch.float32)
        lbl = torch.tensor(int(row['label']), dtype=torch.long)
        return {'image': img, 'tabular': tab, 'label': lbl}


def train_few_shot_proto(model, dataset, device,
                         n_way=2, k_shot=5, q_query=5,
                         num_episodes=100, criterion=None,
                         learning_rate=1e-3, weight_decay=1e-4,
                         use_scheduler=False):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # 默认损失
    if criterion is None:
        lbls_all=dataset['label'].values
        neg_cnt=np.sum(lbls_all==0); pos_cnt=np.sum(lbls_all==1)
        tot=neg_cnt+pos_cnt
        w0=tot/(2*neg_cnt) if neg_cnt>0 else 1.0
        w1=tot/(2*pos_cnt) if pos_cnt>0 else 1.0
        criterion=FocalProtoLoss(gamma=2.0, weight=torch.tensor([w0,w1],device=device)).to(device)
    # 全局原型
    model.eval()
    all_feats=[]
    loader_all=DataLoader(AKIDataset(dataset,config['TRAIN_IMG_PATH']),batch_size=32,shuffle=False)
    with torch.no_grad():
        for b in loader_all:
            feats=model(b['image'].to(device),b['tabular'].to(device)); all_feats.append(feats)
    anchor_extractor=SwAVAnchorExtractor(n_clusters=n_way,device=device)
    print('[Debug] SwAVAnchorExtractor from:', anchor_extractor.__module__,
          anchor_extractor.__class__.__dict__.get('__file__'))
    anchor_vectors=anchor_extractor.extract_anchor_vectors(torch.cat(all_feats,dim=0))
    gen=AnchorFeatureGenerator(input_dim=anchor_vectors.size(1),hidden_dims=[128,64],noise_dim=16).to(device)
    model.train()
    # Episodes
    for ep in range(num_episodes):
        support_set,query_set=create_episode(dataset,n_way,k_shot,q_query)
        # 加权采样支持集
        sup_lbls=support_set['label'].values
        sup_ws=[criterion.weight[0].item() if l==0 else criterion.weight[1].item() for l in sup_lbls]
        sup_sampler=WeightedRandomSampler(sup_ws,len(sup_ws),replacement=True)
        support_loader=DataLoader(AKIDataset(support_set,config['TRAIN_IMG_PATH']),batch_size=len(support_set),sampler=sup_sampler)
        # 加权采样查询集
        qry_lbls=query_set['label'].values
        qry_ws=[criterion.weight[0].item() if l==0 else criterion.weight[1].item() for l in qry_lbls]
        qry_sampler=WeightedRandomSampler(qry_ws,len(qry_ws),replacement=True)
        query_loader=DataLoader(AKIDataset(query_set,config['TRAIN_IMG_PATH']),batch_size=len(query_set),sampler=qry_sampler)
        # 支持特征
        for b in support_loader:
            s_feat=model(b['image'].to(device),b['tabular'].to(device)); s_lbl=b['label'].to(device)
        # 查询特征
        for b in query_loader:
            q_feat=model(b['image'].to(device),b['tabular'].to(device)); q_lbl=b['label'].to(device)
        # 原型对齐合成
        synth_feats=[]; synth_lbls=[]
        for c in range(n_way):
            for _ in range(5):
                noise=torch.randn(1,gen.noise_dim,device=device)
                fake=gen(anchor_vectors[c:c+1],noise); synth_feats.append(fake.squeeze(0)); synth_lbls.append(c)
        synth_feats=torch.stack(synth_feats,dim=0).to(device)
        synth_lbls=torch.tensor(synth_lbls,device=device)
        # 多原型 proto loss
        protos_list=compute_multi_prototypes(s_feat,s_lbl,n_way,3)
        dists=[torch.cdist(q_feat,p).min(dim=1).values for p in protos_list]
        logits=torch.stack([-d for d in dists],dim=1)
        loss_proto=F.cross_entropy(logits,q_lbl,weight=criterion.weight)
        pred=logits.argmax(dim=1); acc=(pred==q_lbl).float().mean().item()
        # 对齐 loss
        align_loss=anchor_alignment_loss(synth_feats,synth_lbls,anchor_vectors.to(device))
        # batch-hard triplet loss
        batch_feats=torch.cat([s_feat,q_feat],dim=0); batch_labels=torch.cat([s_lbl,q_lbl],dim=0)
        hard_loss=batch_hard_triplet_loss(batch_feats,batch_labels,margin=0.5)
        # 总 loss
        loss=loss_proto+0.2*align_loss+0.5*hard_loss
        optimizer.zero_grad();
        loss.backward();
        if ep == 0:
            conv1_grad = model.img_encoder.conv1.weight.grad  # ResNet 第一层
            print('[Grad‑Check] img_encoder.conv1 grad mean:',
                  conv1_grad.abs().mean().item())
        optimizer.step()
        if use_scheduler: scheduler.step()
        if ep%10==0: print(f"[Episode {ep}] Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    return model
