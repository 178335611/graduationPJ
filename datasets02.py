"""
数据模块：数据集加载、预处理、子集划分
"""
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms
import copy
from config_k import *


class RemapDataset(Dataset):
    """把旧label动态映射到0~N-1"""
    def __init__(self, subset, remap):
        self.subset = subset
        self.remap = remap

    def __getitem__(self, i):
        x, y_old = self.subset[i]
        return x, self.remap[y_old]

    def __len__(self):
        return len(self.subset)


def get_transforms():
    """获取数据预处理变换"""
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 先resize
        transforms.RandomCrop(224),  # 再crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # 移除：Rotation, ColorJitter, Blur, Erasing
    ])
    val_transform= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return {'train': train_transform, 'val': val_transform}


def load_dataset():
    """
    加载并处理数据集
    如果 WANTED_CLASSES 为空，则使用全部类别

    返回: train_loader, val_loader, class_names, num_classes, dataset_sizes, class_mapping
    """
    # 加载全集
    transforms_dict = get_transforms()
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transforms_dict['train'])
    class_names_all = full_dataset.classes  # 全部类别名列表

    # ===== 智能类别选择 =====
    if WANTED_CLASSES:  # 指定了特定类别
        # 映射文件夹名到label index
        name2idx = {name: idx for idx, name in enumerate(class_names_all)}

        # 检查类别是否存在
        for cls in WANTED_CLASSES:
            if cls not in name2idx:
                raise ValueError(f"类别 '{cls}' 不存在于数据集中。可用类别: {class_names_all[:10]}...")

        wanted_labels = [name2idx[n] for n in WANTED_CLASSES]
        selected_names = WANTED_CLASSES
        print(f'🎯 选中 {len(WANTED_CLASSES)} 个指定类别：{WANTED_CLASSES}')

    else:  # 使用全部类别
        wanted_labels = list(range(len(class_names_all)))
        selected_names = class_names_all
        print(f'🎯 使用全部 {len(class_names_all)} 个类别')

    print(f'   Label indices: {wanted_labels[:5]}{"..." if len(wanted_labels) > 5 else ""}')

    # ===== 过滤数据 =====
    if len(wanted_labels) < len(class_names_all):  # 需要过滤
        indices = [i for i, (_, y) in enumerate(full_dataset) if y in wanted_labels]
        subset_dataset = Subset(full_dataset, indices)

        # 重新映射label到 0~N-1
        old2new = {old: new for new, old in enumerate(wanted_labels)}
        remapped_dataset = RemapDataset(subset_dataset, old2new)
    else:  # 使用全部，无需过滤和映射
        remapped_dataset = full_dataset
        old2new = {i: i for i in range(len(class_names_all))}  # 恒等映射

    num_classes = len(wanted_labels)
    print(f'📊 实际参与训练的类别数：{num_classes}')
    print(f'📊 样本总数：{len(remapped_dataset)}')

    # ===== 划分训练集和验证集 =====
    train_size = int(0.8 * len(remapped_dataset))
    val_size = len(remapped_dataset) - train_size
    train_dataset, val_dataset = random_split(remapped_dataset, [train_size, val_size])

    # 验证集使用不同的transform
    if isinstance(remapped_dataset, datasets.ImageFolder):
        # 全部类别时，需要复制并修改transform
        val_dataset.dataset = copy.copy(remapped_dataset)
        val_dataset.dataset.transform = transforms_dict['val']
    else:  # RemapDataset
        val_dataset.dataset = copy.copy(remapped_dataset)
        val_dataset.dataset.subset = copy.copy(remapped_dataset.subset)
        val_dataset.dataset.subset.dataset = copy.copy(remapped_dataset.subset.dataset)
        val_dataset.dataset.subset.dataset.transform = transforms_dict['val']

    # 获取类别名
    class_names = [class_names_all[i] for i in wanted_labels]

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    return train_loader, val_loader, class_names, num_classes, dataset_sizes, old2new