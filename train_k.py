"""
训练模块：Kaggle优化版（支持AMP）
"""
import os
import time
import copy
import torch
from torch.cuda.amp import autocast, GradScaler  # AMP支持
from config_k import *


def train_epoch(model, dataloader, criterion, optimizer, device, phase, scaler=None):
    """单个epoch的训练或验证，支持AMP"""
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    batch_count = 0

    epoch_start = time.time()

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        print(f"Batch {i}: input shape={inputs.shape}, device={inputs.device}, label sum={labels.sum()}")

        optimizer.zero_grad()

        # AMP自动混合精度
        with torch.set_grad_enabled(phase == 'train'):
            if USE_AMP and phase == 'train':
                with autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 正常精度
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        batch_count += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_time = time.time() - epoch_start

    return epoch_loss, epoch_acc, epoch_time


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                device, paths, timestamp, resume_from=None, scheduler=None):
    """
    Kaggle优化训练流程
    """
    since = time.time()

    # AMP初始化
    scaler = GradScaler() if USE_AMP else None
    if USE_AMP:
        print("⚡ 启用自动混合精度(AMP)加速")

    # 初始化状态
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    # 断点续训
    if resume_from and os.path.isfile(resume_from):
        print(f"🔍 加载检查点: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            best_acc = checkpoint.get('best_acc', 0.0)
            best_model_wts = checkpoint.get('best_model_wts', copy.deepcopy(model.state_dict()))

            # Kaggle：显示恢复信息
            print(f"📌 从 epoch {start_epoch} 继续 | 历史最佳: {best_acc:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("⚠️ 纯权重格式，优化器状态未恢复")
            start_epoch = 0
    else:
        print("🚀 从头开始训练")

    # 训练循环
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'\n{"="*70}')
        if is_kaggle():
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | GPU: {torch.cuda.get_device_name(0)}')
        print(f'{"="*70}')

        # 训练和验证
        epoch_times = {}
        for phase in ['train', 'val']:
            dataloader = dataloaders[phase]

            epoch_loss, epoch_acc, epoch_time = train_epoch(
                model, dataloader, criterion, optimizer, device, phase, scaler
            )

            epoch_times[phase] = epoch_time

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # 格式化输出
            time_str = f"{epoch_time:.1f}s"
            print(f'{phase.upper():5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {time_str}')

            # 保存最佳模型
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_wts,
                        'best_acc': best_acc,
                        'history': history,
                    }, paths['best_model'])
                    print(f"✨ 新的最佳模型! Acc: {best_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"⏳ 早停计数: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # 统计信息
        total_time = epoch_times['train'] + epoch_times['val']
        elapsed = time.time() - since
        eta = (NUM_EPOCHS - epoch - 1) * total_time / 60

        # GPU内存信息（Kaggle专用）
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_info = f" | GPU内存: {mem_alloc:.1f}G/{mem_reserved:.1f}G"
        else:
            gpu_info = ""

        print(f"⏱️  Epoch总耗时: {total_time:.1f}s | 已用: {elapsed/60:.1f}min | 预估剩余: {eta:.1f}min{gpu_info}")

        # 学习率调整
        if scheduler is not None:
            current_val_loss = history['val_loss'][-1]
            scheduler.step(current_val_loss)

            lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
            print(f"📉 学习率: {lrs}")

        # 早停检查
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 早停触发！连续{EARLY_STOPPING_PATIENCE}epoch无提升")
            print(f"   最佳epoch: {epoch - patience_counter}, Acc: {best_acc:.4f}")
            break

        # 保存检查点
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(
                os.path.dirname(paths['best_model']),
                f'checkpoint_epoch_{epoch + 1}.pth'
            )

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_model_wts': best_model_wts,
                'history': history
            }
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict, ckpt_path)
            print(f"💾 检查点: epoch_{epoch + 1}.pth")

    # 训练结束
    time_elapsed = time.time() - since

    print(f'\n{"="*70}')
    print(f'🎉 训练完成!')
    print(f'   总耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'   最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)')
    print(f'   平均每epoch: {time_elapsed/(epoch+1):.1f}s')
    print(f'{"="*70}')

    # Kaggle提交信息
    if is_kaggle():
        from utils import kaggle_commit_message
        kaggle_commit_message(os.path.dirname(paths['best_model']), best_acc)

    return best_model_wts, history