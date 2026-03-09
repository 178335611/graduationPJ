"""
配置文件：集中管理所有参数
"""
import os

# --- 路径配置 ---
DATA_DIR = r'D:\01bishe\pj001\datasets\flower2'
BASE_OUTPUT_DIR = r'D:\01bishe\pj001\experiments'

# --- 数据配置 ---
WANTED_CLASSES =[]
    #['000', '003', '007', '010', '015','101', '044', '055', '056', '057']

# --- 训练超参数 ---
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005# 降低学习率
WEIGHT_DECAY = 5e-4      # 强正则化
MOMENTUM = 0.9
SCHEDULER_STEP_SIZE = 15#（学习率衰减步长）
SCHEDULER_GAMMA = 0.1
# 关键：冻结所有特征层，只训练FC！
FREEZE_BACKBONE = True

# --- 模型配置 ---
MODEL_NAME = 'resnet18'
PRETRAINED = True
RESUME_CHECKPOINT = None # 从头训练

# --- 其他配置 ---
NUM_WORKERS = 4
CHECKPOINT_INTERVAL = 5  # 每5个epoch保存检查点