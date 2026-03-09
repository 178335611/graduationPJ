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

# 超参数
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001


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

    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    except FileNotFoundError:
        print(f"!! 错误：找不到数据集文件夹 '{DATA_DIR}'")
        print("!! 请确保路径正确，并且该文件夹下是按类别分好的子文件夹。")
        return

    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    if NUM_CLASSES == 0:
        print(f"!! 错误：在 '{DATA_DIR}' 中没有找到任何类别的子文件夹。")
        return
    print(f"自动检测到 {NUM_CLASSES} 个类别。")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 这一步非常重要，确保验证集使用正确的（非增强的）transform
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes. First few: {class_names[:5]}")
    print(f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")

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
    print("\n--- Evaluating Best Model ---")
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        model.load_state_dict(best_model_wts)
        evaluate_model(model, dataloaders['val'], criterion, device)
    else:
        print("No best model was saved during training.")

    # --- 6. 推理 ---
    print("\n--- Performing Inference on a Sample Image ---")
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        sample_image_path = os.path.join(DATA_DIR, class_names[0],
                                         os.listdir(os.path.join(DATA_DIR, class_names[0]))[0])
        predict_single_image(BEST_MODEL_SAVE_PATH, sample_image_path, class_names, device)
    else:
        print("No model available for inference.")


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, best_model_save_path,
                timestamp):
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


def predict_single_image(model_path, image_path, class_names, device):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
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
    print(f"Image path: {image_path}")
    print(f"Predicted class: '{predicted_class}' with confidence: {confidence:.2%}")


if __name__ == '__main__':
    main()