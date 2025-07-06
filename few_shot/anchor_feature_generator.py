import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorFeatureGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], noise_dim=16):
        """
        一个简单的 MLP 特征生成器，用于从 anchor 向量生成伪样本。

        参数：
        - input_dim: 表征的维度（与 anchor 向量一致）
        - hidden_dims: 两层 MLP 的隐藏单元数量
        - noise_dim: 随机噪声维度
        """
        super(AnchorFeatureGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.fc1 = nn.Linear(input_dim + noise_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_out = nn.Linear(hidden_dims[1], input_dim)

    def forward(self, anchor_vec, noise=None):
        """
        输入：
          - anchor_vec: Tensor of shape (B, D) 或 (1, D)，B 为 batch_size
          - noise: 可选 Tensor of shape (B, noise_dim); 如果为 None 则自动生成
        返回：
          - Tensor of shape (B, D)
        """
        # anchor_vec: [B, D]
        batch_size = anchor_vec.size(0)
        device = anchor_vec.device

        # 如果外部没传 noise，则自己生成
        if noise is None:
            # 直接传两个整数即可，不要用 tuple 作为第一个参数
            noise = torch.randn(batch_size, self.noise_dim, device=device)
        else:
            # 使用传入的 noise，并确保在同一 device
            noise = noise.to(device)

        # 拼接 anchor 与 noise
        x = torch.cat([anchor_vec, noise], dim=1)  # [B, D+noise_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)                         # [B, D]
        return x


def generate_synthetic_samples(generator, anchor_vecs, num_per_class=10):
    """
    给定多个 anchor vectors，为每个类生成 num_per_class 个合成样本。

    参数：
    - generator: AnchorFeatureGenerator 实例
    - anchor_vecs: Tensor of shape (K, D)
    - num_per_class: 每个类别生成多少个样本

    返回：
    - all_features: Tensor of shape (K * num_per_class, D)
    - all_labels: list[int] 长度为 K * num_per_class
    """
    all_features = []
    all_labels = []
    for i, anchor in enumerate(anchor_vecs):
        # 直接用 generator.forward(anchor_vec_batch, noise) 生成
        # 这里 anchor.unsqueeze(0) 保证是 (1, D)，返回 (num_per_class, D)
        synth_feats = generator(anchor.unsqueeze(0), noise=None if num_per_class == 1 else None).repeat(num_per_class, 1)
        # 或者更清晰地：
        # synth_feats = generator(anchor.unsqueeze(0), None).repeat(num_per_class, 1)
        # 但常规做法是在 trainer 里按需调用 generator(anchor_vec, noise)
        all_features.append(synth_feats)
        all_labels.extend([i] * num_per_class)
    return torch.cat(all_features, dim=0), all_labels
