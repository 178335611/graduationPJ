import torch

def check_device():
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # 获取可用的CUDA设备数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # 获取当前设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 如果有GPU，打印GPU的名称
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if __name__ == "__main__":
    check_device()
