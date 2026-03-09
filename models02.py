"""
模型模块：模型构建和修改
"""
import torch
import torch.nn as nn
from torchvision import models
from config02 import *


def build_model(num_classes, device):
    """保守策略：冻结特征，只训练分类头"""
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 全部冻结！
    for param in model.parameters():
        param.requires_grad = False

    # 只替换FC层，使用Dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    print(f'🔒 完全冻结Backbone，仅训练FC层 ({num_classes}类)')
    return model.to(device)


def get_optimizer(model):
    """只用Adam，单一学习率"""
    fc_params = model.fc.parameters()
    return torch.optim.Adam(fc_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)