"""
训练模块：训练循环和检查点管理
"""
import os
import time
import copy
import torch
from config import *


def train_epoch(model, dataloader, criterion, optimizer, device, phase):
    """单个epoch的训练或验证"""
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                scheduler, device, paths, timestamp, resume_from=None):
    """
    完整训练流程（支持断点续训）

    Args:
        resume_from: 检查点路径，None则从头训练
    """

    since = time.time()

    # 初始化状态
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # ===== 断点续训：加载检查点 =====
    if resume_from and os.path.isfile(resume_from):
        print(f"🔍 发现检查点，正在加载：{resume_from}")

        checkpoint = torch.load(resume_from, map_location=device)

        # 【关键修复】自动检测文件格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点格式（checkpoint_epoch_X.pth）
            print("📦 检测到完整检查点格式")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_acc = checkpoint.get('best_acc', 0.0)
            best_model_wts = checkpoint.get('best_model_wts', copy.deepcopy(model.state_dict()))

        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 某些框架保存的格式
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1

        else:
            # 纯权重格式（best_model.pth）
            print("📦 检测到纯权重格式（仅模型参数）")
            model.load_state_dict(checkpoint)
            # 无法恢复优化器状态，从头开始优化
            print("⚠️  注意：优化器状态无法恢复，将使用初始学习率")
            start_epoch = 0  # 或从命令行参数指定

        print(f"📌 从 epoch {start_epoch} 开始训练")
        if best_acc > 0:
            print(f"   历史最佳准确率: {best_acc:.4f}")

    else:
        if resume_from:
            print(f"⚠️  未找到检查点: {resume_from}，将从头训练")
        else:
            print("🚀 从头开始训练")

    # ===== 训练循环 =====
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'val']:
            dataloader = dataloaders[phase]
            epoch_loss, epoch_acc = train_epoch(
                model, dataloader, criterion, optimizer, device, phase
            )

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, paths['best_model'])
                print(f"✓ 新的最佳模型 (Acc: {best_acc:.4f})")

        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # ===== 保存检查点（包含最佳模型权重） =====
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(
                os.path.dirname(paths['best_model']),
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_model_wts': best_model_wts,  # 保存最佳权重，随时可恢复
                'history': history
            }, checkpoint_path)
            print(f"💾 检查点已保存: checkpoint_epoch_{epoch + 1}.pth")

    time_elapsed = time.time() - since
    print(f'\n{"="*50}')
    print(f'训练完成！耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')
    print(f'{"="*50}')

    return best_model_wts, history