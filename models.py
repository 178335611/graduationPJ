"""
模型模块：模型构建和修改
"""
import torch
import torch.nn as nn
from torchvision import models
from config import *


def build_model(num_classes, device):
    """构建模型（针对细粒度分类优化）"""
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 策略：少冻结，多训练（102类需要更多可训练参数）
    if num_classes > 50:  # 大数据集
        # 只冻结前2层，训练后3层
        for name, param in model.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print(f'🔓 解冻策略：训练layer3, layer4和FC层（{num_classes}类）')
    else:  # 小数据集，完全冻结
        for param in model.parameters():
            param.requires_grad = False

    # 修改FC层
    num_ftrs = model.fc.in_features

    # 针对102类，添加Dropout防止过拟合
    if num_classes > 50:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # 添加Dropout
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model

def get_optimizer(model):
    """获取优化器"""
    return torch.optim.SGD(
        model.fc.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM
    )


def get_scheduler(optimizer):
    """获取学习率调度器"""
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA
    )