import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class SwAVAnchorExtractor:
    def __init__(self, n_clusters=2, temperature=0.1, device='cpu'):
        """
        n_clusters: number of anchor vectors (e.g., 2 for binary classification)
        temperature: softmax temperature
        device: 'cuda' or 'cpu'
        """
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.device = device

    def extract_anchor_vectors(self, features):
        """
        对输入特征进行KMeans聚类，提取聚类中心作为anchor vectors
        features: torch.Tensor, shape (N, D)，已L2归一化
        return: anchor_vectors: torch.Tensor, shape (n_clusters, D)
        """
        if isinstance(features, torch.Tensor):
            feats_np = features.cpu().numpy()
        else:
            feats_np = np.asarray(features)

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        features_np = features.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(features_np)
        centers = kmeans.cluster_centers_
        centers = torch.tensor(centers, dtype=torch.float32).to(self.device)
        centers = F.normalize(centers, p=2, dim=1)
        return centers

    def ssl_loss(self, x1, x2, q1, q2, anchors):
        """
        自监督loss：x1和x2是相同样本的不同增强；q1, q2是soft cluster assignments
        anchors: (K, D) 的 anchor 向量
        return: scalar loss
        """
        logits_1 = torch.mm(x1, anchors.t()) / self.temperature
        logits_2 = torch.mm(x2, anchors.t()) / self.temperature
        loss_1 = -torch.sum(q2 * F.log_softmax(logits_1, dim=1), dim=1).mean()
        loss_2 = -torch.sum(q1 * F.log_softmax(logits_2, dim=1), dim=1).mean()
        return loss_1 + loss_2

    def get_soft_assignments(self, features, anchors):
        """
        给定特征和anchor，返回softmax后的软分配概率 (batch_size, n_clusters)
        """
        logits = torch.mm(features, anchors.t()) / self.temperature
        q = F.softmax(logits, dim=1)
        return q
