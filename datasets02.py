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
from sklearn.model_selection import train_test_split


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


def get_transforms():
    """获取数据预处理变换"""
    # 验证集和测试集的预处理方式通常完全一致
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': val_test_transform,
        'test': val_test_transform  # 【新增】测试集变换
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

    print(f" 数据目录: {DATA_ROOT}")
    print(f" CSV: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV不存在: {CSV_PATH}")

    # 【调用】模块级别的函数
    transforms_dict = get_transforms()

    # 读取CSV
    full_df = pd.read_csv(CSV_PATH)
    print(f" 总样本: {len(full_df)}, 类别: {full_df['label'].nunique()}")

    # ================= 【核心修改：8:1:1 分层抽样】 =================
    # 第一步：从总集中切分出 10% 作为 Test (保证每个类别都有 10% 进入 Test)
    train_val_df, test_df = train_test_split(
        full_df, test_size=0.10, stratify=full_df['label'], random_state=42
    )

    # 第二步：从剩下的 90% 中，再切分出 11.1% (即占总数的 10%) 作为 Val
    # 此时 train_val_df 占总数的 90%，取它的 1/9 就是总数的 10%
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.111, stratify=train_val_df['label'], random_state=42
    )

    # 重置索引
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 打印各个集合的样本数，验证比例
    print(f"训练集: {len(train_df)} ({len(train_df) / len(full_df) * 100:.1f}%)")
    print(f"验证集: {len(val_df)} ({len(val_df) / len(full_df) * 100:.1f}%)")
    print(f"测试集: {len(test_df)} ({len(test_df) / len(full_df) * 100:.1f}%)")

    # ... 后面的 Dataset 和 DataLoader 创建代码保持不变 ...
    transforms_dict = get_transforms()
    train_dataset = FlowerCSVDataset(train_df, IMAGE_DIR, transforms_dict['train'])
    val_dataset = FlowerCSVDataset(val_df, IMAGE_DIR, transforms_dict['val'])
    test_dataset = FlowerCSVDataset(test_df, IMAGE_DIR, transforms_dict['test'])

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = train_dataset.num_classes
    class_names = train_dataset.class_names
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_mapping = train_dataset.label_to_idx

    return train_loader, val_loader, test_loader, class_names, num_classes, dataset_sizes, class_mapping