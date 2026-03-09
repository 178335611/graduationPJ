"""
工具模块：日志记录、可视化、文件管理
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from config02 import *


def create_experiment_dir():
    """
    创建实验目录结构
    返回: 实验根目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)

    # 创建子目录
    subdirs = ['models', 'logs', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir, timestamp


def get_paths(exp_dir):
    """获取所有文件路径"""
    return {
        'best_model': os.path.join(exp_dir, 'models', 'best_model.pth'),
        'final_model': os.path.join(exp_dir, 'models', 'final_model.pth'),
        'plot': os.path.join(exp_dir, 'plots', 'training_curves.png'),
        'history_csv': os.path.join(exp_dir, 'logs', 'training_history.csv'),
        'history_json': os.path.join(exp_dir, 'logs', 'training_history.json'),
        'config': os.path.join(exp_dir, 'logs', 'config.json'),
        'results': os.path.join(exp_dir, 'results', 'evaluation_results.txt'),
    }


def save_config(config, path):
    """保存配置到JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {path}")


def save_history(history, csv_path, json_path):
    """保存训练历史"""
    # CSV格式
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_acc']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    df.to_csv(csv_path, index=False)
    print(f"History (CSV) saved to: {csv_path}")

    # JSON格式
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History (JSON) saved to: {json_path}")


def plot_curves(history, save_path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # 准确率
    ax1.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    ax1.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # 损失
    ax2.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax2.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")


def log_result(path, content):
    """追加记录结果"""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')