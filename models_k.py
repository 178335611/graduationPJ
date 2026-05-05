"""
模型模块：模型构建和修改
"""
import torch
import torch.nn as nn
from torchvision import models
# from config_k import *
from config_k import (
    MODEL_NAME,
    USE_PRETRAINED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    OPTIMIZER,
    MOMENTUM,      # SGD 需要
    # SCHEDULER,     # get_scheduler 需要
    NUM_EPOCHS     # scheduler 可能需要
)

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
        print(f'-- 使用ResNet50（{num_classes}类）')
        model = models.resnet50(weights='IMAGENET1K_V2')
        # 冻结更多层（只训练FC）
        for param in model.parameters():
            param.requires_grad = False

        # 只解冻FC层
        for param in model.fc.parameters():
            param.requires_grad = True

        print(f'-- 冻结全部特征层，仅训练FC层')
        # 修改FC
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    return model.to(device)

def get_optimizer(model):
    """分层学习率优化"""
    # # 分层设置学习率
    # layer3_params = []
    # layer4_params = []
    # fc_params = []
    #
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
    #     if 'layer3' in name:
    #         layer3_params.append(param)
    #     elif 'layer4' in name:
    #         layer4_params.append(param)
    #     else:
    #         fc_params.append(param)
    #
    # param_groups = [
    #     {'params': layer3_params, 'lr': LEARNING_RATE * 0.1},  # 特征层慢
    #     {'params': layer4_params, 'lr': LEARNING_RATE * 0.5},  # 高层稍快
    #     {'params': fc_params, 'lr': LEARNING_RATE}  # 分类层最快
    # ]
    #
    # return torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)

    """分层学习率优化 (自动适配 ResNet/VGG/EfficientNet)"""
    # 调试：确认配置已正确加载
    print(f"   [Config] MODEL={MODEL_NAME}, OPT={OPTIMIZER}, LR={LEARNING_RATE}")
    early_backbone = []  # 底层/主干特征 (学习率最低)
    late_backbone = []  # 高层特征 (学习率中等，仅部分模型使用)
    head_params = []  # 分类头/全连接层 (学习率最高)

    model_name = MODEL_NAME.lower()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 🟦 ResNet 系列 (layer1~4 + fc)
        if 'resnet' in model_name:
            if 'fc' in name:
                head_params.append(param)
            elif 'layer4' in name:
                late_backbone.append(param)
            else:  # layer1, layer2, layer3, conv1, bn1 等
                early_backbone.append(param)

        # 🟨 VGG 系列 (features + classifier)
        elif 'vgg' in model_name:
            if 'classifier' in name:
                head_params.append(param)
            else:
                early_backbone.append(param)

        # 🟩 EfficientNet 系列 (features + classifier)
        elif 'efficientnet' in model_name:
            if 'classifier' in name:
                head_params.append(param)
            else:
                early_backbone.append(param)

        # ⚪ 兜底：无法识别的层默认归入分类头
        else:
            head_params.append(param)

    # 构建参数组 (自动过滤空列表)
    param_groups = []
    if early_backbone:
        param_groups.append({'params': early_backbone, 'lr': LEARNING_RATE * 0.1, 'weight_decay': WEIGHT_DECAY})
    if late_backbone:
        param_groups.append({'params': late_backbone, 'lr': LEARNING_RATE * 0.5, 'weight_decay': WEIGHT_DECAY})
    if head_params:
        param_groups.append({'params': head_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY})

    # 动态绑定优化器
    if OPTIMIZER == 'AdamW':
        return torch.optim.AdamW(param_groups)
    elif OPTIMIZER == 'SGD':
        return torch.optim.SGD(param_groups, momentum=MOMENTUM)
    else:
        return torch.optim.Adam(param_groups)



def get_scheduler(optimizer):
    """Plateau调度器，patience=3"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控loss（更敏感）
        factor=0.5,  # 温和衰减
        patience=3,  # 3epoch不提升就降lr
    )