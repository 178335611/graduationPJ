# pull.py  —— Python 3.7 实测可用
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import os

target_dir = r'D:\01bishe\pj001\data\flower'   # 想放哪就改哪
os.makedirs(target_dir, exist_ok=True)

# 旧版 snapshot_download 只要装过 tqdm 就会自动出现进度条
snapshot_download(repo_id='keremberke/oxford-flower-102-yolo',
                  repo_type='dataset',
                  local_dir=target_dir,
                  local_dir_use_symlinks=False)

print('✅ 下载完成，路径：', target_dir)