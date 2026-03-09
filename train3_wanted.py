import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import os
import time
import copy
import json
import pandas as pd

# --- 1. 配置参数 ---
# 使用 r'' 字符串来处理Windows路径，避免反斜杠问题
DATA_DIR = r'D:\01bishe\pj001\datasets\flower2'
BASE_OUTPUT_DIR = r'D:\01bishe\pj001\experiments'  # 【改】所有实验结果的根目录

# 超参数
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001


# ========== 文件头部新增一个顶层类 ==========
class RemapDataset(torch.utils.data.Dataset):
    """pickle 可见：把旧 label 动态映射到 0~N-1"""

    def __init__(self, subset, remap):
        self.subset = subset
        self.remap = remap

    def __getitem__(self, i):
        x, y_old = self.subset[i]
        return x, self.remap[y_old]

    def __len__(self):
        return len(self.subset)


# ========== 下面是你原来的 main() 等代码 ==========

def main():
    # 【改】创建带时间戳的实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_{timestamp}"
    EXP_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_name)

    # 【改】创建子目录结构
    MODEL_SAVE_DIR = os.path.join(EXP_DIR, 'models')
    LOG_DIR = os.path.join(EXP_DIR, 'logs')
    PLOT_DIR = os.path.join(EXP_DIR, 'plots')
    RESULT_DIR = os.path.join(EXP_DIR, 'results')

    for dir_path in [MODEL_SAVE_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 【改】定义所有输出文件的完整路径
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'final_model.pth')
    PLOT_PATH = os.path.join(PLOT_DIR, 'training_curves.png')
    HISTORY_CSV_PATH = os.path.join(LOG_DIR, 'training_history.csv')
    HISTORY_JSON_PATH = os.path.join(LOG_DIR, 'training_history.json')
    CONFIG_PATH = os.path.join(LOG_DIR, 'config.json')
    RESULT_LOG_PATH = os.path.join(RESULT_DIR, 'evaluation_results.txt')

    print(f"--- Running experiment: {experiment_name} ---")
    print(f"--- Output directory: {EXP_DIR} ---")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 数据加载与预处理 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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

    # ===== 1. 先正常加载全集 =====
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    class_names_all = full_dataset.classes  # 102 个字符串

    # ===== 2. 指定想要的几类（填文件夹名字） =====
    wanted_folder_names = ['000', '003', '007', '010', '015',
                           '101', '044', '055', '056', '057']  # 随意挑
    # 把文件夹名 → 原始 label index
    name2idx = {name: idx for idx, name in enumerate(class_names_all)}
    wanted_labels = [name2idx[n] for n in wanted_folder_names]
    print(f'选中类别：{wanted_folder_names}  ->  label index：{wanted_labels}')

    # ===== 3. 过滤索引 =====
    indices = [i for i, (_, y) in enumerate(full_dataset) if y in wanted_labels]

    # ===== 4. 用 Subset 抽出来 =====
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)

    # ===== 5. 重新映射 label → 0~N-1 （关键，否则 label 会断档） =====
    old2new = {old: new for new, old in enumerate(wanted_labels)}
    remapped_dataset = RemapDataset(subset_dataset, old2new)
    NUM_CLASSES = len(wanted_labels)
    print(f'实际参与训练的类别数：{NUM_CLASSES}')

    # 划分 & 数据加载
    train_size = int(0.8 * len(remapped_dataset))
    val_size = len(remapped_dataset) - train_size
    train_dataset, val_dataset = random_split(remapped_dataset, [train_size, val_size])

    # ========== 保存新类别名映射 ==========
    class_names = [class_names_all[i] for i in wanted_labels]

    # 验证集要换 transform
    val_dataset.dataset = copy.copy(remapped_dataset)
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # 【改】保存实验配置信息
    config = {
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'data_dir': DATA_DIR,
        'wanted_classes': wanted_folder_names,
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'train_size': train_size,
        'val_size': val_size,
        'device': str(device),
        'class_mapping': {old: new for old, new in old2new.items()},
        'class_names': class_names
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {CONFIG_PATH}")

    # --- 3. 模型构建 ---
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. 训练模型 ---
    print("\n--- Starting Training ---")
    best_model_wts, history = train_model(
        model, criterion, optimizer, scheduler, dataloaders,
        dataset_sizes, device, BEST_MODEL_PATH, MODEL_SAVE_DIR, timestamp
    )

    # 【改】保存训练历史数据（CSV和JSON双格式）
    save_training_history(history, HISTORY_CSV_PATH, HISTORY_JSON_PATH)

    # 【改】绘制并保存图表
    plot_training_curves(history, PLOT_PATH)

    # --- 5. 评估模型 ---
    print("\n--- Evaluating Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        print("Loading best model for evaluation...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else:
        print("No best model found, using current (last-epoch) model for evaluation.")

    eval_loss, eval_acc = evaluate_model(model, dataloaders['val'], criterion, device, RESULT_LOG_PATH, class_names)

    # 【改】保存最终模型（最后epoch的权重）
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")

    # --- 6. 推理 ---
    print("\n--- Performing Inference on a Sample Image ---")
    if os.path.exists(BEST_MODEL_PATH):
        print("Loading best model for inference...")
        infer_model = copy.deepcopy(model)
        infer_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else:
        print("No best model found, using current model for inference.")
        infer_model = model

    sample_image_path = os.path.join(DATA_DIR, class_names[0],
                                     os.listdir(os.path.join(DATA_DIR, class_names[0]))[0])
    predict_single_image(infer_model, sample_image_path, class_names, device, NUM_CLASSES, RESULT_LOG_PATH)

    print(f"\n{'=' * 50}")
    print(f"实验完成！所有结果保存在: {EXP_DIR}")
    print(f"{'=' * 50}")


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                device, best_model_path, model_save_dir, timestamp):
    start_epoch = 0
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            for inputs, labels in dataloaders[phase]:
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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, best_model_path)
                print(f"✓ New best model saved (Acc: {best_acc:.4f})")

        # 【改】每5个epoch保存检查点，放在models子目录
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    return best_model_wts, history


def save_training_history(history, csv_path, json_path):
    """【新增】保存训练历史到CSV和JSON"""
    # 保存为CSV（便于Excel查看）
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_acc']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    df.to_csv(csv_path, index=False)
    print(f"Training history (CSV) saved to: {csv_path}")

    # 保存为JSON（便于程序读取）
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history (JSON) saved to: {json_path}")


def plot_training_curves(history, plot_path):
    """【改】使用传入的完整路径"""
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
    plt.savefig(plot_path, dpi=300)  # 【改】增加分辨率
    plt.close()  # 【改】关闭图形释放内存
    print(f"Training curves plot saved to: {plot_path}")


def evaluate_model(model, dataloader, criterion, device, log_path, class_names):
    """【改】增加日志保存和返回指标"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    result_str = f'Evaluation Loss: {total_loss:.4f} Acc: {total_acc:.4f}\n'
    print(result_str)

    # 【改】保存评估结果到文件
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Evaluation Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(result_str)

    return total_loss, total_acc


def predict_single_image(model_or_path, image_path, class_names, device, num_classes, log_path=None):
    """【改】增加日志记录"""
    if isinstance(model_or_path, str):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_or_path, map_location=device))
        model = model.to(device)
    else:
        model = model_or_path

    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item()

    result_str = (
        f"Image path: {image_path}\n"
        f"Predicted class: '{predicted_class}' with confidence: {confidence:.2%}\n"
    )
    print(result_str)

    # 【改】保存推理结果
    if log_path:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Inference ---\n")
            f.write(result_str)


if __name__ == '__main__':
    main()