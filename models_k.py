"""
模型模块：模型构建和修改
"""
import torch
import torch.nn as nn
from torchvision import models
from config_k import *


def build_model(num_classes, device):
    """构建ResNet50，解冻layer3和layer4"""

    print(f'🚀 使用ResNet50（{num_classes}类）')
    model = models.resnet50(weights='IMAGENET1K_V2')

    # 冻结layer1和layer2，解冻layer3和layer4
    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print(f'🔓 冻结layer1-2，训练layer3-4和FC层')
    print(f'   可训练参数占比: {sum(p.requires_grad for p in model.parameters()) / len(list(model.parameters())):.1%}')

    # 修改FC层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.to(device)


def get_optimizer(model):
    """分层学习率优化"""
    # 分层设置学习率
    layer3_params = []
    layer4_params = []
    fc_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'layer3' in name:
            layer3_params.append(param)
        elif 'layer4' in name:
            layer4_params.append(param)
        else:
            fc_params.append(param)

    param_groups = [
        {'params': layer3_params, 'lr': LEARNING_RATE * 0.1},  # 特征层慢
        {'params': layer4_params, 'lr': LEARNING_RATE * 0.5},  # 高层稍快
        {'params': fc_params, 'lr': LEARNING_RATE}  # 分类层最快
    ]

    return torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)


def get_scheduler(optimizer):
    """Plateau调度器，patience=3"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控loss（更敏感）
        factor=0.5,  # 温和衰减
        patience=3,  # 3epoch不提升就降lr
    )