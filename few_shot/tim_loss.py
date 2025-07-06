# 可保存为 few_shot/tim_loss.py
import torch
import torch.nn.functional as F

def tim_loss(logits, gamma=0.5):
    """
    logits: (B, C) → 来自 query 样本对原型的 -distance 或 logits
    gamma: 权重超参
    """
    probs = F.softmax(logits, dim=1)          # [B, C]
    log_probs = torch.log(probs + 1e-8)        # 避免 log0
    H_y_given_x = -torch.mean(torch.sum(probs * log_probs, dim=1))       # 条件熵
    H_y = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-8))  # 边际熵
    ce = F.cross_entropy(logits, torch.argmax(probs, dim=1))  # γ 用于平衡
    return gamma * ce - H_y + H_y_given_x
