import torch
import torch.nn.functional as F

#“原型网络的标准损失函数
def compute_prototypical_loss(
    support_feats, support_labels,
    query_feats,   query_labels,
    n_way, k_shot,
    weight=None
):
    # 1) 计算原型，将支持集中每一类的特征向量求均值， [n_way, 特征维度] 的原型矩阵
    prototypes = []
    for i in range(n_way):
        cls_feats = support_feats[support_labels == i]
        prototypes.append(cls_feats.mean(dim=0))
    prototypes = torch.stack(prototypes)  # [n_way, feat_dim]

    # 2) 距离 → log_prob,
    dists = torch.cdist(query_feats, prototypes)      # [B, n_way]，计算与原型之间的欧氏距离

    # 3) Loss
    from few_shot.tim_loss import tim_loss
    logits = -dists
    loss = tim_loss(logits, gamma=0.5)

    # 4) 计算准确率
    # 修复准确率计算
    pred = logits.argmax(dim=1)
    acc = (pred == query_labels).float().mean()
    return loss, acc.item()


class FocalProtoLoss(torch.nn.Module):
    """
    在原型网络基础上，使用 Focal Loss 聚焦难例。
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        #控制模型对难样本的关注程度
        self.gamma = gamma
        self.weight = weight

    def forward(self,
        support_feats, support_labels,
        query_feats,   query_labels,
        n_way, k_shot
    ):
        # 1) 原型
        prototypes = []
        for i in range(n_way):
            prototypes.append(support_feats[support_labels == i].mean(dim=0))
        prototypes = torch.stack(prototypes)  # [n_way, D]

        # 2) log softmax
        dists   = torch.cdist(query_feats, prototypes)   # [B, n_way]
        log_p_y = F.log_softmax(-dists, dim=1)           # [B, n_way]

        # 3) per-sample CE
        ce = F.nll_loss(
            log_p_y,
            query_labels,
            weight=self.weight,
            reduction='none'
        )  # [B]
        pt = torch.exp(-ce)

        # 4) focal factor
        #如果样本越难（pt 越小），它的 (1 - pt)^γ 就越大 → 损失越大 → 越被关注。
        loss = ((1 - pt) ** self.gamma * ce).mean()

        # 5) accuracy
        pred = log_p_y.argmax(dim=1)
        acc  = (pred == query_labels).float().mean()
        return loss, acc.item()


def anchor_alignment_loss(synthetic_feats, synthetic_labels, anchor_vectors):
    """
    LSSL3 损失：合成样本与其所属 anchor vector 之间的余弦相似性损失
    synthetic_feats: Tensor (N, D) - 合成特征
    synthetic_labels: List[int] 或 Tensor (N,) - 每个特征的类别索引
    anchor_vectors: Tensor (K, D) - 每个类别的 anchor 中心
    返回：平均余弦距离（越小越好）
    """
    if isinstance(synthetic_labels, list):
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long, device=synthetic_feats.device)
    anchor_targets = anchor_vectors[synthetic_labels]  # (N, D)
    cosine_sim = F.cosine_similarity(synthetic_feats, anchor_targets, dim=1)  # (N,)
    loss = 1 - cosine_sim.mean()  # 余弦相似性越高越好，目标是最小化 1 - sim
    return loss
