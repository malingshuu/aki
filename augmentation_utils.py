# augmentation_utils.py

from torchvision import transforms

def get_transform(label: int):
    """
    返回一个 torchvision.Compose，正/负类各自不同增强。
    label: 0=Non‑AKI, 1=AKI
    """
    if label == 1:
        # AKI: 丰富增强
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        # Non‑AKI: 温和增强
        return transforms.Compose([
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.02, contrast=0.02),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
