import os
import torch
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import json
from pointnet2_ops import pointnet2_utils as futils

def farthest_point_sampling_cuda(points, num_samples):
    idx = futils.furthest_point_sample(torch.tensor(points, dtype=torch.float32).unsqueeze(0).cuda(), num_samples).long()
    return idx.cpu().numpy()[0]

def preprocess_data(data_root='./sampled_data/splited', num_samples=2000):
    # 定义四个目录
    split_dirs = ['train', 'val', 'test_intra', 'test_inter']
    
    for split_dir in split_dirs:
        print(f"Processing {split_dir} data...")
        
        # 获取当前split目录的pth文件
        pth_files = glob(str(Path(data_root) / split_dir / "pth") + "/*.pth")
        
        # 创建idx目录
        idx_dir = Path(data_root) / split_dir / 'idx'
        os.makedirs(idx_dir, exist_ok=True)

        for pth_file in tqdm(pth_files, desc=f"Processing {split_dir}"):
            # 读取点云数据
            pc_data = torch.load(pth_file)
            # meta_path = str(Path(data_root) / split_dir / "meta") + '/' + pth_file.split('/')[-1].split('.')[0] + '.json'

            # 点云数据在pc_data[0]中
            points = pc_data[0]
            
            # 执行FPS
            fps_idx = farthest_point_sampling_cuda(points, num_samples)
            
            # 保存索引
            idx_file = idx_dir / (Path(pth_file).stem + '_idx.pth')
            torch.save(fps_idx, idx_file)

        print(f"{split_dir.capitalize()} data processing completed.")

if __name__ == "__main__":
    preprocess_data('/16T/zhangran/GAPartNet_re_rendered')
