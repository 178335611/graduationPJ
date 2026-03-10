"""
数据模块：支持CSV标注的数据集 (Oxford 102 Flower专用)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from config_k import BATCH_SIZE, NUM_WORKERS, is_kaggle


class FlowerCSVDataset(Dataset):
    """Oxford 102 Flower CSV格式数据集"""
    def __init__(self, df, image_dir, transform=None):  # 【修改】传入image_dir
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir  # 【修改】使用传入的路径
        self.transform = transform

        # 标签映射
        unique = sorted(self.df['label'].unique())
        self.label_to_idx = {l: i for i, l in enumerate(unique)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}
        self.class_names = unique
        self.num_classes = len(unique)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['fname'])
        image = Image.open(img_path).convert('RGB')
        label = self.label_to_idx[row['label']]

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():  # 【确保】函数在模块级别定义
    """获取数据预处理变换"""
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def load_dataset():
    """加载Oxford 102 Flower CSV数据集"""

    # 【定义】局部变量DATA_ROOT
    if is_kaggle():
        print("--检测到Kaggle环境--")
        DATA_ROOT = '/kaggle/input/datasets/hishamkhdair/102flowers-data/102flowers'
        CSV_PATH = os.path.join(DATA_ROOT, 'labels.csv')
        IMAGE_DIR = os.path.join(DATA_ROOT, 'train')  # 【定义】图片目录
    else:
        # 本地路径
        DATA_ROOT = r'D:\01bishe\pj001\datasets'
        CSV_PATH = os.path.join(DATA_ROOT, 'labels.csv')
        IMAGE_DIR = os.path.join(DATA_ROOT, 'flower1\jpg')  # 【定义】图片目录

    print(f"📂 数据目录: {DATA_ROOT}")
    print(f"📄 CSV: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV不存在: {CSV_PATH}")

    # 【调用】模块级别的函数
    transforms_dict = get_transforms()

    # 读取CSV
    full_df = pd.read_csv(CSV_PATH)
    print(f"📊 总样本: {len(full_df)}, 类别: {full_df['label'].nunique()}")

    # 划分
    train_size = int(0.8 * len(full_df))
    torch.manual_seed(42)
    perm = torch.randperm(len(full_df))
    train_idx = perm[:train_size].tolist()
    val_idx = perm[train_size:].tolist()

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    # 【修改】传入IMAGE_DIR
    train_dataset = FlowerCSVDataset(train_df, IMAGE_DIR, transforms_dict['train'])
    val_dataset = FlowerCSVDataset(val_df, IMAGE_DIR, transforms_dict['val'])

    # DataLoader
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = train_dataset.num_classes
    class_names = train_dataset.class_names
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_mapping = train_dataset.label_to_idx

    print(f"✅ 训练: {dataset_sizes['train']}, 验证: {dataset_sizes['val']}")

    return train_loader, val_loader, class_names, num_classes, dataset_sizes, class_mapping