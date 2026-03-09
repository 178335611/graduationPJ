import os
import shutil
from datasets import load_from_disk

# --- 配置参数 ---
# 原始数据集的路径 (包含dataset_infos.json的那个)
ORIGINAL_DATASET_PATH = 'D:\\01bishe\\pj001\\datasets'
# 你之前提取出来的、未分类的图片所在的文件夹
UNORGANIZED_IMAGES_FOLDER = r'D:\01bishe\pj001\datasets\flower1'

# 我们要创建的、整理好的目标文件夹
ORGANIZED_OUTPUT_FOLDER = r'D:\01bishe\pj001\datasets'


def organize_existing_images():
    print("--- 开始整理已提取的图片 ---")

    # 1. 加载数据集元信息，获取标签
    print(f"正在从 '{ORIGINAL_DATASET_PATH}' 加载数据集信息以获取标签...")
    try:
        dataset = load_from_disk(ORIGINAL_DATASET_PATH)
        train_split = dataset['train']
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保路径正确，并且数据集文件夹完整。")
        return

    # 获取标签名称列表，例如 ['pink primrose', 'hard-leaved pocket orchid', ...]
    label_names = train_split.features['label'].names
    # 获取所有样本的标签列表，例如 [7, 23, 45, ...]
    labels = train_split['label']

    # 检查样本数量是否匹配
    num_labels = len(labels)
    num_images = len(os.listdir(UNORGANIZED_IMAGES_FOLDER))
    if num_labels != num_images:
        print(f"警告：标签数量 ({num_labels}) 与图片文件数量 ({num_images}) 不匹配！")
        print("请确保 '{UNORGANIZED_IMAGES_FOLDER}' 文件夹中的图片是你从该数据集中完整提取的。")
        # 即使不匹配，我们仍然尝试继续整理
    else:
        print(f"标签数量与图片数量匹配 ({num_labels})，可以开始整理。")

    # 2. 创建目标文件夹
    os.makedirs(ORGANIZED_OUTPUT_FOLDER, exist_ok=True)

    # 3. 遍历未整理的图片文件夹
    print(f"正在遍历 '{UNORGANIZED_IMAGES_FOLDER}' 文件夹...")
    moved_count = 0
    for filename in os.listdir(UNORGANIZED_IMAGES_FOLDER):
        # 假设文件名是 'flower_XXXXX.png'
        if filename.startswith('flower_') and filename.endswith('.png'):
            try:
                # 从文件名 'flower_00001.png' 中解析出索引 0
                # 文件名从1开始，所以索引要减1
                index = int(filename.split('_')[1].split('.')[0]) - 1

                # 检查索引是否越界
                if index < 0 or index >= num_labels:
                    print(f"警告：从文件名 '{filename}' 解析出的索引 {index} 超出范围，已跳过。")
                    continue

                # 4. 查找对应的类别名称
                label_index = labels[index]
                class_name = label_names[label_index]

                # 5. 创建类别的子文件夹
                class_folder = os.path.join(ORGANIZED_OUTPUT_FOLDER, class_name)
                os.makedirs(class_folder, exist_ok=True)

                # 6. 移动文件
                source_path = os.path.join(UNORGANIZED_IMAGES_FOLDER, filename)
                destination_path = os.path.join(class_folder, filename)

                # 使用 shutil.move 来移动文件
                shutil.move(source_path, destination_path)
                moved_count += 1

            except (ValueError, IndexError) as e:
                print(f"无法解析文件名 '{filename}' 或查找标签失败: {e}，已跳过。")

    print(f"\n整理完成！总共移动了 {moved_count} 个图片文件到 '{ORGANIZED_OUTPUT_FOLDER}' 文件夹中。")
    print("现在你可以使用这个文件夹进行模型训练了。")


if __name__ == "__main__":
    organize_existing_images()