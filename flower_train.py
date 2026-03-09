# 保存为 flower_train.py，上传到Kaggle

"""
Kaggle单文件花卉分类训练
"""

# ========== 1. 配置 ==========
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, models, transforms
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import copy
from datetime import datetime

# 检测环境
IS_KAGGLE = os.path.exists('/kaggle')
DATA_DIR = '/kaggle/input/oxford-102-flower-dataset/oxford-102-flower-dataset' if IS_KAGGLE else './data'
OUTPUT_DIR = '/kaggle/working/experiments' if IS_KAGGLE else './experiments'

# 超参数
BATCH_SIZE = 64 if IS_KAGGLE else 32
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
USE_AMP = IS_KAGGLE  # Kaggle启用AMP


# ========== 2. 数据集类 ==========
class RemapDataset(Dataset):
    def __init__(self, subset, remap):
        self.subset = subset
        self.remap = remap

    def __getitem__(self, i):
        x, y_old = self.subset[i]
        return x, self.remap[y_old]

    def __len__(self):
        return len(self.subset)


def get_transforms():
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


def load_data():
    """加载数据"""
    tfms = get_transforms()
    full = datasets.ImageFolder(DATA_DIR, transform=tfms['train'])

    # 使用全部类别
    wanted = list(range(len(full.classes)))
    indices = [i for i, (_, y) in enumerate(full) if y in wanted]
    subset = Subset(full, indices)

    # 映射label
    old2new = {old: new for new, old in enumerate(wanted)}
    dataset = RemapDataset(subset, old2new) if len(wanted) < len(full.classes) else full

    # 划分
    train_sz = int(0.8 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])

    # 验证集换transform
    val_ds.dataset = copy.copy(dataset)
    if hasattr(val_ds.dataset, 'subset'):
        val_ds.dataset.subset = copy.copy(dataset.subset)
        val_ds.dataset.subset.dataset = copy.copy(dataset.subset.dataset)
        val_ds.dataset.subset.dataset.transform = tfms['val']
    else:
        val_ds.dataset.transform = tfms['val']

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, len(wanted), full.classes


# ========== 3. 模型 ==========
def build_model(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V2')

    # 冻结layer1-2
    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 修改FC
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.cuda()


def get_optimizer(model):
    # 分层学习率
    params = [
        {'params': [], 'lr': LEARNING_RATE * 0.1},  # layer3
        {'params': [], 'lr': LEARNING_RATE * 0.5},  # layer4
        {'params': [], 'lr': LEARNING_RATE},  # fc
    ]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'layer3' in name:
            params[0]['params'].append(p)
        elif 'layer4' in name:
            params[1]['params'].append(p)
        else:
            params[2]['params'].append(p)

    return optim.Adam(params, weight_decay=WEIGHT_DECAY)


# ========== 4. 训练 ==========
def train_epoch(model, loader, criterion, optimizer, scaler, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    correct = 0
    start = time.time()

    for x, y in loader:
        x, y = x.cuda(), y.cuda()

        if is_train:
            optimizer.zero_grad()

            if USE_AMP:
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                out = model(x)
                loss = criterion(out, y)

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    return loss_sum / len(loader.dataset), correct / len(loader.dataset), time.time() - start


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = GradScaler() if USE_AMP else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    patience = 0

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')

        # 训练
        t_loss, t_acc, t_time = train_epoch(model, train_loader, criterion, optimizer, scaler, True)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        print(f'Train | Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | {t_time:.1f}s')

        # 验证
        v_loss, v_acc, v_time = train_epoch(model, val_loader, criterion, optimizer, scaler, False)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        print(f'Val   | Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | {v_time:.1f}s')

        # 保存最佳
        if v_acc > best_acc:
            best_acc = v_acc
            patience = 0
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_model.pth')
            print(f'✨ New best: {best_acc:.4f}')
        else:
            patience += 1
            print(f'⏳ Patience: {patience}/10')
            if patience >= 10:
                print('Early stopping!')
                break

        scheduler.step(v_loss)

    return history, best_acc


# ========== 5. 可视化 ==========
def plot(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_acc']) + 1)

    # Acc
    axes[0, 0].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 1].set_title('Loss')
    axes[0, 1].legend()

    # Gap
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 0].plot(epochs, gap, 'g-')
    axes[1, 0].set_title('Overfit Gap')
    axes[1, 0].axhline(0.1, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved plot to {save_path}')


# ========== 6. 主函数 ==========
def main():
    print(f'🌸 Flower Classification on {"Kaggle" if IS_KAGGLE else "Local"}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print('\nLoading data...')
    train_loader, val_loader, num_classes, class_names = load_data()
    print(f'Classes: {num_classes}, Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}')

    # 模型
    print('\nBuilding model...')
    model = build_model(num_classes)

    # 训练
    print('\nTraining...')
    history, best_acc = train(model, train_loader, val_loader)

    # 保存
    print('\nSaving results...')
    plot(history, f'{OUTPUT_DIR}/training_curves.png')

    # CSV
    df = pd.DataFrame(history)
    df.to_csv(f'{OUTPUT_DIR}/history.csv', index=False)

    print(f'\n🎉 Done! Best acc: {best_acc:.4f}')
    print(f'Output: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()