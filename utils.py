"""
工具模块：Kaggle优化版
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from config_k import is_kaggle

def create_experiment_dir():
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)

    for subdir in ['models', 'logs', 'plots', 'results']:
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

    # Kaggle：同时输出到控制台便于查看
    if is_kaggle():
        print(f"\n{'='*60}")
        print("实验配置：")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")


def save_history(history, csv_path, json_path):
    """保存训练历史"""
    # CSV
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_acc']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    df.to_csv(csv_path, index=False)

    # JSON
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Kaggle：输出CSV预览
    if is_kaggle():
        print(f"\n📊 训练历史预览（最后5epoch）：")
        print(df.tail().to_string(index=False))


def plot_curves(history, save_path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_acc']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2布局，更多信息

    # 准确率
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax1.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax2.set_title('Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 学习率（如果有）
    ax3 = axes[1, 0]
    # 这里可以传入lr_history绘制

    # 过拟合指标
    ax4 = axes[1, 1]
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    ax4.plot(epochs, gap, 'g-', linewidth=2)
    ax4.axhline(y=0.1, color='r', linestyle='--', label='Overfit threshold')
    ax4.set_title('Train-Val Gap (Overfitting)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Gap')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Kaggle用150dpi足够
    plt.close()

    print(f"训练曲线已保存: {save_path}")

    # Kaggle：显示图片
    if is_kaggle():
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=save_path))


def log_result(path, content, print_console=True):
    """记录结果"""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

    if print_console and is_kaggle():
        print(content)


def kaggle_commit_message(exp_dir, best_acc):
    """生成Kaggle提交信息"""
    msg = f"""
    {'='*60}
    KAGGLE训练完成报告
    {'='*60}
    实验目录: {exp_dir}
    最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)
    
    输出文件:
    - 最佳模型: {os.path.join(exp_dir, 'models', 'best_model.pth')}
    - 训练曲线: {os.path.join(exp_dir, 'plots', 'training_curves.png')}
    - 历史数据: {os.path.join(exp_dir, 'logs', 'training_history.csv')}
    
    记得点击"Save Version"保存输出！
    {'='*60}
    """
    print(msg)
    return msg