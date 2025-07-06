
import torch
import torch.nn as nn
import torchvision.models as models

class OptimizedMultiModalNet(nn.Module):
    def __init__(self, tabular_input_dim, dropout_rate=0.3):
        super(OptimizedMultiModalNet, self).__init__()

        # 图像编码器：ResNet50，解冻后几层用于微调
        self.img_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in list(self.img_encoder.parameters())[:-20]:
            param.requires_grad = False
        self.img_encoder.fc = nn.Identity()
        self.img_feat_dim = 2048

        # 表格编码器
        self.tab_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 融合层 + gating attention
        self.gate = nn.Sequential(
            nn.Linear(self.img_feat_dim + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()  # gating attention
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.img_feat_dim + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feat = self.img_encoder(img)
        tab_feat = self.tab_encoder(tab)
        combined = torch.cat([img_feat, tab_feat], dim=1)

        gates = self.gate(combined)
        weighted_img = img_feat * gates[:, 0].unsqueeze(1)
        weighted_tab = tab_feat * gates[:, 1].unsqueeze(1)
        fused = torch.cat([weighted_img, weighted_tab], dim=1)

        out = self.classifier(fused)
        return out.squeeze(1)

    def extract_features(self, img, tab):
        img_feat = self.img_encoder(img)
        tab_feat = self.tab_encoder(tab)
        gates = self.gate(torch.cat([img_feat, tab_feat], dim=1))
        weighted_img = img_feat * gates[:, 0].unsqueeze(1)
        weighted_tab = tab_feat * gates[:, 1].unsqueeze(1)
        return torch.cat([weighted_img, weighted_tab], dim=1)

