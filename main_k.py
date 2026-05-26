#!/usr/bin/env python
"""
Kaggle Notebook主入口
Usage: 直接复制到Kaggle Notebook的cell中运行
"""

import os
import sys,torch
import torch.nn as nn

# 自动检测并添加路径（Kaggle上不需要，本地需要）
if '/kaggle' in os.getcwd():
    sys.path.append('/kaggle/working')

from config_k import *
from datasets02 import load_dataset
from models_k import build_model, get_optimizer, get_scheduler
from train_k import train_model
from evaluate import evaluate_model, predict_single_image
from utils import *


# 【新增】专门用于计算测试集最终指标的函数
def get_test_metrics(model, test_loader, criterion, device):
    """计算测试集的Loss和Acc"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    return test_loss, test_acc.item()

def main():
    """Kaggle主函数"""
    print(f"\n{'#'*70}")
    print(f"# 花卉分类训练 - {'Kaggle' if is_kaggle() else '本地'}环境")
    print(f"{'#'*70}")
    
    # 创建实验目录
    exp_dir, timestamp = create_experiment_dir()
    paths = get_paths(exp_dir)
    
    print(f"实验目录: {exp_dir}")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ================= 【修改 1：接收 test_loader】 =================
    print(f"\n{'─' * 70}")
    print("加载数据集...")
    # 注意这里多了一个 test_loader
    train_loader, val_loader, test_loader, class_names, num_classes, dataset_sizes, class_mapping = load_dataset()

    # 训练过程只需要 train 和 val（测试集不参与训练和早停）
    dataloaders = {'train': train_loader, 'val': val_loader}
    print(f"{'─' * 70}")
    
    # 保存配置
    config = {
        'timestamp': timestamp,
        'environment': 'kaggle' if is_kaggle() else 'local',
        'data_dir': DATA_DIR,
        'num_classes': num_classes,
        'class_names': class_names[:5] + ['...'] if len(class_names) > 5 else class_names,
        'dataset_sizes': dataset_sizes,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'model': MODEL_NAME,
        'use_amp': USE_AMP,
        'device': str(device),
    }
    save_config(config, paths['config'])
    
    # 构建模型
    print(f"\n{'─'*70}")
    print("构建模型...")
    model = build_model(num_classes, device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params/1e6:.2f}M")
    print(f"   可训练参数量: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    print(f"{'─'*70}")
    
    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    # Kaggle断点续训（从上次保存继续）
    RESUME_CHECKPOINT = None
    # 自动查找最新的检查点
    if is_kaggle() and os.path.exists('/kaggle/working'):
        import glob
        checkpoints = glob.glob('/kaggle/working/**/checkpoint_epoch_*.pth', recursive=True)
        if RESUME_CHECKPOINT != None:
            if checkpoints:
                # 找最新的
                RESUME_CHECKPOINT = max(checkpoints, key=os.path.getmtime)
                print(f"🔍 自动找到检查点: {RESUME_CHECKPOINT}")
    
    # 训练
    print(f"\n{'='*70}")
    print("开始训练")
    print(f"{'='*70}")
    
    best_model_wts, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer,
        device, paths, timestamp,
        resume_from=RESUME_CHECKPOINT,
        scheduler=scheduler
    )

    # 2. 训练结束后的最终评估
    print(f"\n{'─' * 70}")
    print("最终验证集评估 (Validation Set)...")
    model.load_state_dict(best_model_wts)
    evaluate_model(model, val_loader, criterion, device, paths['results'], class_names)

    print(f"\n{'─' * 70}")
    print("最终测试集盲测评估 (Test Set)...")
    # test_results_path = paths['results'].replace('results', 'test_results')
    test_results_path = paths['results'].replace('.txt', '_test.txt')
    evaluate_model(model, test_loader, criterion, device, test_results_path, class_names)

    # 3. 【核心修改】获取 Test 指标并追加到 history 中，以便保存进 Log
    test_loss, test_acc = get_test_metrics(model, test_loader, criterion, device)
    print(f"🏆 最终测试集盲测结果 -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

    # 将最终结果以标量形式存入 history 字典
    history['final_test_loss'] = test_loss
    history['final_test_acc'] = test_acc

    # 4. 保存结果 (此时 save_history 会将 test 结果一并写入 JSON/CSV)
    print(f"\n{'─' * 70}")
    print("保存结果与日志...")
    save_history(history, paths['history_csv'], paths['history_json'])
    plot_curves(history, paths['plot'])  # 画图函数依然只画 train 和 val 的曲线
    
    # Kaggle：输出下载链接
    if is_kaggle():
        print(f"\n{'='*70}")
        print("下载链接（右侧Output文件夹）：")
        for name, path in paths.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024**2
                print(f"   {name}: {os.path.basename(path)} ({size:.1f}MB)")
        print(f"{'='*70}")
    
    return exp_dir, best_acc


# Kaggle Notebook直接运行
if __name__ == '__main__':
    exp_dir, best_acc = main()