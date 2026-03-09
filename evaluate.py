"""
评估模块：模型评估和单图推理
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils import log_result


def evaluate_model(model, dataloader, criterion, device, log_path, class_names):
    """评估模型性能"""
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

    result = f'Evaluation Loss: {total_loss:.4f} Acc: {total_acc:.4f}'
    print(result)

    # 记录到文件
    log_result(log_path, f"\n{'=' * 50}\nEvaluation Results\n{'=' * 50}\n{result}")

    return total_loss, total_acc


def predict_single_image(model, image_path, class_names, device, log_path=None):
    """单图推理"""
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

    result = (
        f"Image: {image_path}\n"
        f"Predicted: '{predicted_class}' (confidence: {confidence:.2%})"
    )
    print(result)

    if log_path:
        log_result(log_path, f"\n--- Inference ---\n{result}")

    return predicted_class, confidence


def load_model_for_inference(model_path, num_classes, device):
    """加载训练好的模型用于推理"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model