import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
# [新增] 引入混合精度训练模块
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import os
import time
import copy
import json  # [新增] 用于保存类别名称

# --- 1. 配置参数 ---
DATA_DIR = r'D:\01bishe\pj001\datasets\flower2'
MODEL_SAVE_DIR = r'D:\01bishe\pj001\model'

# 超参数
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_EPOCHS = 50  # [修改] 增加Epoch上限，反正有早停机制，不用担心跑太久
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7  # [新增] 如果验证集Loss连续7个Epoch不下降，就停止训练


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--- Running experiment with timestamp: {timestamp} ---")

    BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f'best_model_{timestamp}.pth')
    # [新增] 类别名称保存路径
    CLASS_JSON_PATH = os.path.join(MODEL_SAVE_DIR, f'classes_{timestamp}.json')
    PLOT_SAVE_PATH = f'training_curves_{timestamp}.png'

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查是否支持AMP
    use_amp = torch.cuda.is_available()
    if use_amp:
        print("AMP (Automatic Mixed Precision) is enabled.")
    else:
        print("CUDA not available. AMP disabled.")

    # --- 2. 数据加载与预处理 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # [新增] 随机旋转，增加对角度的鲁棒性
            transforms.RandomRotation(15),
            # [新增] 颜色抖动，增加对光照变化的鲁棒性
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    except FileNotFoundError:
        print(f"!! 错误：找不到数据集文件夹 '{DATA_DIR}'")
        return

    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    print(f"自动检测到 {NUM_CLASSES} 个类别。")

    # [新增] 保存类别名称映射到JSON文件
    with open(CLASS_JSON_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"类别名称映射已保存至: {CLASS_JSON_PATH}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.transform = data_transforms['val']

    # tiny数据集用于测试
    # 如果 tiny 集能冲到 95 %+ → 模型容量够，问题在 数据量/正则太强；
    # 如果 tiny 集也卡在 60 %左右 → 模型或优化信号被“冻住”，直接看第 2 步。
    wanted = set(range(10))  # 0-9
    idxs = [i for i, (_, y) in enumerate(full_dataset) if y in wanted]
    tiny_dataset = torch.utils.data.Subset(full_dataset, idxs)
    # 设置种子
    torch.manual_seed(42)
    random.seed(42)
    tiny, _ = random_split(tiny_dataset, [50, len(tiny_dataset) - 50])
    tiny_loader = DataLoader(tiny, batch_size=10, shuffle=True)
    # 设置种子
    torch.manual_seed(3367)
    random.seed(3367)
    tiny2, _ = random_split(tiny_dataset, [50, len(tiny_dataset) - 50])
    tiny_loader2=DataLoader(tiny2, batch_size=10, shuffle=True)
    dataset_sizes = {
        'train': 50,
        'val': 50
    }

    # [修改] pin_memory=True 加速CPU到GPU的数据传输
    # dataloaders = {
    #     'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    #     'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    # }
    dataloaders = {
        'train': tiny_loader,
        'val': tiny_loader2
    }


    # --- 3. 模型构建 ---
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 策略选择：这里暂时保持冻结，仅训练全连接层。
    # 如果想进一步提升精度，可以将下面这两行注释掉，进行全参数微调（Fine-tuning）
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # [修改] 使用 AdamW 优化器，通常比 SGD 收敛更快
    # 注意：如果全参数微调，建议 lr 设为 1e-4
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # [修改] 使用 CosineAnnealingLR 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- 4. 训练模型 ---
    print("\n--- Starting Training ---")
    best_model_wts, history = train_model(
        model, criterion, optimizer, scheduler, dataloaders,
        dataset_sizes, device, BEST_MODEL_SAVE_PATH, timestamp, use_amp
    )

    plot_training_curves(history, PLOT_SAVE_PATH)

    # --- 5. 评估与推理 ---
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        print("\n--- Evaluating Best Model ---")
        model.load_state_dict(best_model_wts)
        evaluate_model(model, dataloaders['val'], criterion, device)

        print("\n--- Performing Inference on a Sample Image ---")
        sample_image_path = os.path.join(DATA_DIR, class_names[0],
                                         os.listdir(os.path.join(DATA_DIR, class_names[0]))[0])
        predict_single_image(model, sample_image_path, class_names, device)  # 注意这里我修改了传参方式
    else:
        print("No best model was saved.")


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, best_model_save_path,
                timestamp, use_amp):
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # [新增] 早停相关变量
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # [新增] 初始化 GradScaler 用于混合精度训练
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # [修改] 开启混合精度上下文
                    with autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        # [修改] 使用 scaler 进行反向传播和参数更新
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳准确率模型
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, best_model_save_path)
                    print(f"Found better accuracy. Model saved.")

                # [新增] 早停逻辑 (Early Stopping)
                # 监控 Val Loss，如果不下降则计数
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"Early Stopping Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        # [新增] 检查早停条件
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping triggered! No improvement in validation loss for {EARLY_STOPPING_PATIENCE} epochs.")
            break

        # 定期保存checkpoint
        if (epoch + 1) % 5 == 0:
            # 这里的路径也可以改为 MODEL_SAVE_DIR
            pass

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return best_model_wts, history


def plot_training_curves(history, plot_save_path):
    # 保持原样
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    epochs = range(1, len(train_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(epochs, train_acc, 'bo-', label='Training Acc')
    ax1.plot(epochs, val_acc, 'ro-', label='Validation Acc')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_loss, 'bo-', label='Training Loss')
    ax2.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Training curves plot saved to {plot_save_path}")


def evaluate_model(model, dataloader, criterion, device):
    # 保持原样
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)
    print(f'Evaluation Loss: {total_loss:.4f} Acc: {total_acc:.4f}')


# [修改] 优化了推理函数，直接接收已经加载好的 model 对象
def predict_single_image(model, image_path, class_names, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Could not read image: {e}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item()
    print(f"Image path: {image_path}")
    print(f"Predicted class: '{predicted_class}' with confidence: {confidence:.2%}")


if __name__ == '__main__':
    main()