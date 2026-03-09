"""
主程序入口：协调各个模块完成训练流程
"""
import os
import torch
import copy

# 导入自定义模块
from config02 import *
from datasets02 import load_dataset
from models02 import build_model, get_optimizer
from train05 import train_model
from evaluate import evaluate_model, predict_single_image
from utils import *


def main():
    # 创建实验目录
    exp_dir, timestamp = create_experiment_dir()
    paths = get_paths(exp_dir)

    print(f"--- Experiment: {os.path.basename(exp_dir)} ---")
    print(f"--- Output: {exp_dir} ---")

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, val_loader, class_names, num_classes, dataset_sizes, class_mapping = load_dataset()
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 保存配置
    config = {
        'timestamp': timestamp,
        'data_dir': DATA_DIR,
        'wanted_classes': WANTED_CLASSES,
        'num_classes': num_classes,
        'class_names': class_names,
        'class_mapping': {str(k): v for k, v in class_mapping.items()},
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'train_size': dataset_sizes['train'],
        'val_size': dataset_sizes['val'],
        'device': str(device),
        'model': MODEL_NAME,
        'pretrained': PRETRAINED,
        'resume_checkpoint': RESUME_CHECKPOINT

    }
    save_config(config, paths['config'])

    # 构建模型
    model = build_model(num_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)

    # 训练
    print("\n--- Starting Training ---")
    best_model_wts, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer,
         device, paths, timestamp,
        resume_from=RESUME_CHECKPOINT
    )

    # 保存训练历史
    save_history(history, paths['history_csv'], paths['history_json'])

    # 绘制曲线
    plot_curves(history, paths['plot'])

    # 评估
    print("\n--- Evaluating Model ---")
    if os.path.exists(paths['best_model']):
        model.load_state_dict(torch.load(paths['best_model']))
        print("Loaded best model for evaluation")
    else:
        print("Using final model for evaluation")

    evaluate_model(model, val_loader, criterion, device, paths['results'], class_names)

    # 保存最终模型
    torch.save(model.state_dict(), paths['final_model'])

    # 推理示例
    print("\n--- Sample Inference ---")
    sample_dir = os.path.join(DATA_DIR, class_names[0])
    sample_img = os.listdir(sample_dir)[0]
    sample_path = os.path.join(sample_dir, sample_img)

    predict_single_image(model, sample_path, class_names, device, paths['results'])

    # 完成
    print(f"\n{'=' * 60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()