#!/usr/bin/env python3
# move_102flower_by_mat.py
from pathlib import Path
from scipy.io import loadmat
import shutil

src_dir = Path(r'D:\01bishe\pj001\datasets\flower2\jpg')        # 已解压的 8189 张图
dst_dir = Path(r'D:\01bishe\pj001\datasets\flower2') # 输出根目录
mat_file = 'imagelabels.mat'   # 下载好的标签文件

labels = loadmat(mat_file)['labels'][0]  # 1-102
for idx, label in enumerate(labels, 1):   # 文件名 1-based
    folder = dst_dir / f'{label-1:03d}'   # 0-101
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / f'image_{idx:05d}.jpg',
                folder / f'image_{idx:05d}.jpg')

print('✅ 完成，已按 mat 标签分到', len(set(labels)), '个文件夹')