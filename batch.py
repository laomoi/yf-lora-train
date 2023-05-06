
import config
import train_preview


import os
import fnmatch

path = config.config_data['path']['batch_lora_path'] # 目录路径
ext = "safetensors"  # 扩展名
file_paths = []
# 枚举目录下的所有文件和子目录
for root, dirs, files in os.walk(path):
    # 遍历文件
    for filename in fnmatch.filter(files, f"*.{ext}"):
        file_path = os.path.join(root, filename)
        # 将符合条件的文件路径添加到列表中
        file_paths.append(file_path)

for path in file_paths:
    train_preview.on_train_finish(path, 0 , False)

