import shutil
import os

# 定义源文件夹和目标文件夹路径
source_folder = 'results/OFFset'
destination_folder = '/home/sda/cjm/Results/Offset'

for folder in os.listdir(source_folder):
    name = folder
    original_path = os.path.join(source_folder, name, 'visualization', 'RealDataset')
    target_path = os.path.join(destination_folder, name)

    # 使用shutil.copytree()来复制整个文件夹
    shutil.copytree(original_path, target_path)

print(f'文件夹 {source_folder} 已成功复制到 {destination_folder}')
