"""
Kaggle专用配置
"""
import os

# --- 自动检测环境 ---
def is_kaggle():
    """检测是否在Kaggle环境"""
    return os.path.exists('/kaggle')

# --- 路径配置 ---
if is_kaggle():
    # Kaggle路径
    DATA_DIR = '/kaggle/input/datasets/hishamkhdair/102flowers-data/102flowers'
    BASE_OUTPUT_DIR = '/kaggle/working/experiments'
    print(" 检测到Kaggle环境")
else:
    # 本地路径
    DATA_DIR = r'D:\01bishe\pj001\datasets\flower2'
    BASE_OUTPUT_DIR = r'D:\01bishe\pj001\experiments'

# --- 数据配置 ---
WANTED_CLASSES = []  # 空列表使用全部102类

# --- 模型配置 ---
MODEL_NAME = 'efficientnet_b0'
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9

# 'vgg16' 或 'efficientnet_b0' or 'resnet50'
IMAGE_SIZE = 224
USE_PRETRAINED = True
PRETRAINED = True

# --- 调度器 ---
SCHEDULER_PATIENCE = 3
CHECKPOINT_INTERVAL = 10  # Kaggle保存间隔大些，减少IO

# --- 早停 ---
EARLY_STOPPING_PATIENCE = 5

# --- Kaggle优化参数 ---
if is_kaggle():
    # Kaggle P100/T4优化
    BATCH_SIZE = 64          # P100显存大，增大batch
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-4     # 稍大，收敛快
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4          # Kaggle CPU核心数
    USE_AMP = True         # 自动混合精度加速
    if MODEL_NAME.startswith('efficientnet'):
        LEARNING_RATE = 1e-3
        OPTIMIZER = 'AdamW'
        WEIGHT_DECAY = 1e-4
    elif MODEL_NAME == 'vgg16':
        LEARNING_RATE = 1e-2
        OPTIMIZER = 'SGD'
        BATCH_SIZE=16#vgg显存消耗高
        WEIGHT_DECAY = 1e-4
        MOMENTUM = 0.9
else:
    # 本地保守设置
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    USE_AMP = False


