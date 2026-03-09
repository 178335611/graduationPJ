"""
数据模块：支持CSV标注的数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import copy
from config import *

class FlowerCSVDataset(Dataset):
    """Oxford 102 Flower CSV格式数据集"""
    def __init__(self, root_dir, csv_path, transform=None):
        """
        Args:
            root_dir: 包含train文件夹和csv的目录
            csv_path: labels.csv路径
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'train')  # 图片在train子目录

        # 读取CSV
        self.df = pd.read_csv(csv_path)
        print(f"📄 CSV列: {list(self.df.columns)}")
        print(f"📊 总样本数: {len(self.df)}")

        # 使用实际列名
        self.name_col = 'fname'  # 文件名列
        self.label_col = 'label'  # 标签列（文本格式）

        # 文本标签映射到数字
        unique_labels = sorted(self.df[self.label_col].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # 类别名列表
        self.class_names = unique_labels  # 直接使用文本标签如"passion flower"

        print(f"🎯 类别数: {len(self.class_names)}")
        print(f"   前5类: {self.class_names[:5]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 图片路径
        img_name = row[self.name_col]
        img_path = os.path.join(self.image_dir, img_name)

        # 读取图片
        image = Image.open(img_path).convert('RGB')

        # 标签转换
        label = self.label_to_idx[row[self.label_col]]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset():
    """加载Oxford 102 Flower CSV数据集"""

    # 硬编码你的实际路径
    if IS_KAGGLE:
        DATA_ROOT = '/kaggle/input/datasets/hishamkhdair/102flowers-data/102flowers'
        CSV_PATH = os.path.join(DATA_ROOT, 'labels.csv')
    else:
        DATA_ROOT = r'D:\01bishe\pj001\datasets\flower2'
        CSV_PATH = os.path.join(DATA_ROOT, 'labels.csv')

    print(f"📂 数据根目录: {DATA_ROOT}")
    print(f"📄 CSV路径: {CSV_PATH}")

    # 检查路径存在
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV不存在: {CSV_PATH}")
    if not os.path.exists(os.path.join(DATA_ROOT, 'train')):
        raise FileNotFoundError(f"train目录不存在: {os.path.join(DATA_ROOT, 'train')}")

    # 获取变换
    transforms_dict = get_transforms()

    # 创建完整数据集（训练变换）
    full_dataset = FlowerCSVDataset(
        root_dir=DATA_ROOT,
        csv_path=CSV_PATH,
        transform=transforms_dict['train']
    )

    num_classes = len(full_dataset.class_names)
    class_names = full_dataset.class_names

    # 划分训练验证（80/20）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )

    # 创建训练集（直接用subset）
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)

    # 创建验证集（需要验证变换）
    val_df = full_dataset.df.iloc[val_indices.indices].reset_index(drop=True)

    # 保存临时CSV用于验证集
    temp_csv = os.path.join(DATA_ROOT, '_val_temp.csv')
    val_df.to_csv(temp_csv, index=False)

    val_dataset = FlowerCSVDataset(
        root_dir=DATA_ROOT,
        csv_path=temp_csv,
        transform=transforms_dict['val']
    )

    # 删除临时文件
    os.remove(temp_csv)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # 标签映射（文本->数字）
    class_mapping = full_dataset.label_to_idx

    print(f"\n📊 数据划分:")
    print(f"   训练集: {dataset_sizes['train']}")
    print(f"   验证集: {dataset_sizes['val']}")

    return train_loader, val_loader, class_names, num_classes, dataset_sizes, class_mapping