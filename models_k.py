"""
模型模块：模型构建和修改
"""
import torch
import torch.nn as nn
from torchvision import models
from config_k import *


class ChannelAttention1D(nn.Module):
    """作用于展平特征向量的通道注意力 [B, C] -> [B, C]"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

def build_model(num_classes, device):
    """构建ResNet50，解冻layer3和layer4"""
    if MODEL_NAME == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.vgg16(weights=weights)
        # VGG16 的 classifier 第6层是原1000类全连接层
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif MODEL_NAME.startswith('efficientnet'):
        # 以 efficientnet_b0 为例，可替换 b1~b7
        weights = models.EfficientNet_B0_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet 的 classifier 第1层是全连接层
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif MODEL_NAME == 'resnet50':
        # 1. 加载预训练权重（torchvision 新版 API）
        weights = models.ResNet50_Weights.DEFAULT if USE_PRETRAINED else None
        model = models.resnet50(weights=weights)

        feat_dim = model.fc.in_features  # ResNet50 固定为 2048

        # 2. 动态重建分类头
        new_head = []
        if USE_ATTENTION:
            new_head.append(ChannelAttention1D(feat_dim, reduction=16))
        new_head.append(nn.Linear(feat_dim, num_classes))

        # 替换原 model.fc（保持 forward 流程不变：avgpool -> flatten -> new_head）
        model.fc = nn.Sequential(*new_head)

        return model.to(device)

    return model.to(device)

def get_optimizer(model):
    #  方案1/2：单学习率
    print(f"   [Config] MODEL={MODEL_NAME}, OPT={OPTIMIZER}, LR={LEARNING_RATE}")
    print(f"   [Config] USE_HIER_LR={USE_HIER_LR},"
          f" USE_PRETRAINED={USE_PRETRAINED}, USE_ATTENTION={USE_ATTENTION}")
    if not USE_HIER_LR:
        print(f"   [Optimizer] 单学习率模式 (lr={LEARNING_RATE})")
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 方案3/4：ResNet50 分层学习率
    early, mid, late, head = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'layer4' in name:
            late.append(param)
        elif 'layer3' in name:
            mid.append(param)
        elif 'fc' in name:
            head.append(param)
        else:
            early.append(param)  # layer1, layer2, conv1, bn1

    param_groups = [
        {'params': early, 'lr': LEARNING_RATE * 0.1, 'weight_decay': WEIGHT_DECAY},
        {'params': mid, 'lr': LEARNING_RATE * 0.5, 'weight_decay': WEIGHT_DECAY},
        {'params': late, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
        {'params': head, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ]
    print(f"   [Optimizer] 分层学习率模式 (4个参数组)")
    return torch.optim.AdamW(param_groups)

def get_scheduler(optimizer):
    """Plateau调度器，patience=3"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控loss（更敏感）
        factor=0.5,  # 温和衰减
        patience=3,  # 3epoch不提升就降lr
    )