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

# --- 1. 配置参数 ---
# 使用 r'' 字符串来处理Windows路径，避免反斜杠问题
DATA_DIR = r'D:\01bishe\pj001\datasets\flower2'
MODEL_SAVE_DIR = r'D:\01bishe\pj001\model'
CHECKPOINT = r'D:\01bishe\pj001\model\checkpoint_epoch_25_20260105_223118.pth'

# 超参数
NUM_CLASSES = 10 #不能在函数内部修改
BATCH_SIZE = 32
NUM_EPOCHS = 10
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--- Running experiment with timestamp: {timestamp} ---")

    BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f'best_model_{timestamp}.pth')
    PLOT_SAVE_PATH = f'training_curves_{timestamp}.png'

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

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
    remapped_dataset = RemapDataset(subset_dataset, old2new)  # 【改】用顶层类
    NUM_CLASSES = len(wanted_labels)# 模型最后一层自动跟随
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
        dataset_sizes, device, BEST_MODEL_SAVE_PATH, timestamp
    )

    plot_training_curves(history, PLOT_SAVE_PATH)

    # --- 5. 评估模型 ---
    print("\n--- Evaluating Model ---")
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        print("Loading best model for evaluation...")
        model.load_state_dict(best_model_wts)
    else:
        print("No best model found, using current (last-epoch) model for evaluation.")
    evaluate_model(model, dataloaders['val'], criterion, device)

    # --- 6. 推理 ---
    print("\n--- Performing Inference on a Sample Image ---")
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        print("Loading best model for inference...")
        infer_model = copy.deepcopy(model)  # 复用架构，加载 best 权重
        infer_model.load_state_dict(best_model_wts)
    else:
        print("No best model found, using current model for inference.")
        infer_model = model  # 直接用最新模型

    sample_image_path = os.path.join(DATA_DIR, class_names[0],
                                     os.listdir(os.path.join(DATA_DIR, class_names[0]))[0])
    predict_single_image(infer_model, sample_image_path, class_names, device,NUM_CLASSES)  # 传模型对象


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                device, best_model_save_path,timestamp, checkpoint_path=None):
    start_epoch = 0

    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # ===== 断点续训 =====
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"🔍 发现断点 checkpoint，正在加载：{checkpoint_path}")
        ckp = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckp['model_state'])
        optimizer.load_state_dict(ckp['optimizer_state'])
        scheduler.load_state_dict(ckp['scheduler_state'])
        start_epoch = ckp['epoch'] + 1  # 接着下一 epoch
        history = ckp['history']  # 恢复曲线
        best_acc = ckp.get('best_acc', 0.0)
        best_model_wts = ckp.get('best_model_wts', copy.deepcopy(model.state_dict()))
        print(f"📌 已从 epoch {start_epoch - 1} 继续训练")

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

            # 【修复】这里是完整的循环和计算逻辑
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

            # 【修复】这里是完整的 epoch_loss 和 epoch_acc 计算
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, best_model_save_path)
                print(f"Best validation accuracy achieved. Model saved to {best_model_save_path}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch + 1}_{timestamp}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return best_model_wts, history


def plot_training_curves(history, plot_save_path):
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


def predict_single_image(model_or_path, image_path, class_names, device, num_classes=None):
    # 1. 统一得到“已加载的模型”
    if isinstance(model_or_path, str):          # 传进来的是路径
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_or_path, map_location=device))
        model = model.to(device)
    else:                                       # 传进来的是 nn.Module
        model = model_or_path

    model.eval()

    # 2. 后续推理代码不动
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
    print(f"Image path: {image_path}")
    print(f"Predicted class: '{predicted_class}' with confidence: {confidence:.2%}")

if __name__ == '__main__':
    main()