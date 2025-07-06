import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiModalFewShotNet(nn.Module):
    def __init__(self, tabular_input_dim, embedding_dim=128, dropout_rate=0.2):
        super(MultiModalFewShotNet, self).__init__()
        # 图像编码器：ResNet50 去掉最后一层 fc，输出 2048 维特征
        self.img_encoder = models.resnet50(pretrained=True)
        self.img_encoder.fc = nn.Identity()
        img_feat_dim = 2048

        # 表格编码器：两层 MLP，将 tabular_input_dim 映射到 128 维
        self.tab_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        tab_feat_dim = 128

        # ——1. 用独立 sigmoid 门控取代原来的 softmax 注意力——
        self.gate = nn.Sequential(
            nn.Linear(img_feat_dim + tab_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()       # 两个通道分别对图像/表格做独立加权
        )

        # 融合层：将加权后的 [img_feat_dim + tab_feat_dim] 映射到 embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + tab_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, img, tab):
        # 1) 各自编码
        img_feat = self.img_encoder(img)    # [B, 2048]
        tab_feat = self.tab_encoder(tab)    # [B, 128]
        if not hasattr(self, '_debug_printed'):  # 只打印第一批就够
            print('img_feat mean/std:', img_feat.mean().item(), img_feat.std().item())
            print('tab_feat mean/std:', tab_feat.mean().item(), tab_feat.std().item())
            self._debug_printed = True

        # 2) 拼接产出门控权重
        combined = torch.cat([img_feat, tab_feat], dim=1)  # [B, 2176]
        gates = self.gate(combined)
        # [B, 2] in (0,1)
        if not hasattr(self, '_dbg'):
            print('gate sample (img/tab):', gates[:4].cpu().detach().numpy())
            self._dbg = True

        # 3) 分别加权
        weighted_img = img_feat * gates[:, 0].unsqueeze(1)  # [B,2048]
        weighted_tab = tab_feat * gates[:, 1].unsqueeze(1)  # [B,128]

        # 4) 融合并投影到 embedding_dim
        fused = torch.cat([weighted_img, weighted_tab], dim=1)  # [B,2176]
        embedding = self.fusion(fused)                         # [B,embedding_dim]

        # 5) L2 归一化输出（原型网络常用）
        return F.normalize(embedding, p=2, dim=1)
