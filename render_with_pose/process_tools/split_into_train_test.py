from glob import glob
from typing import List
from pathlib import Path
import os
import shutil
import random
from tqdm import tqdm
os.chdir(os.path.dirname(__file__))

random.seed(42)

seen_category = ['Box', 'Bucket', 'Camera', 'CoffeeMachine', 'Dishwasher', 'Keyboard', 'Microwave', 'Printer', 'Remote', 'StorageFurniture', 'Toaster', 'Toilet', 'WashingMachine']
unseen_category = ['Door', 'KitchenPot', 'Laptop', 'Oven', 'Phone', 'Refrigerator', 'Safe', 'Suitcase', 'Table', 'TrashCan']

# structure:
# ./sampled_data
#     - gt
#         -- Box_47645_0_0.txt
#         -- ...
#         -- Box_47645_0_31.txt
#         -- Box_47645_1_0.txt
#         -- CoffeeMachine_103064_0_7.txt
#         -- ...
#     - pth
#         -- Box_47645_0_0.pth
#         -- ...
#         -- Box_47645_0_31.pth
#         -- Box_47645_1_0.pth
#         -- CoffeeMachine_103064_0_7.pth
#         -- ...
#     - meta
#         -- Box_47645_0_0.json
#         -- ...
#         -- Box_47645_0_31.json
#         -- Box_47645_1_0.json
#         -- CoffeeMachine_103064_0_7.json
#         -- ...

# target:
# ./sampled_data/splited/
#         - train (85% of seen)
#             - meta
#                 ...(xxx.json)
#             - pth
#                 ...(xxx.pth)
#             - gt
#                 ...(xxx.txt)
#         - val (10% of seen)
#         - test_intra (5% of seen)
#         - test_inter (all of unseen)

def split_data(data_root='./sampled_data', target_dir='./sampled_data/splited', seen_category: List[str] = None, unseen_category: List[str] = None):
    seen_objs = []
    for cat in seen_category:
        seen_objs += glob(str(Path(data_root) / "pth") + "/" + cat + "*.pth")
    
    unseen_objs = []
    for cat in unseen_category:
        unseen_objs += glob(str(Path(data_root) / "pth") + "/" + cat + "*.pth")
    
    random.shuffle(seen_objs)

    dir_to_postfix = {
        'meta': 'json',
        'pth': 'pth',
        'gt': 'txt',
    }
    total_seen = len(seen_objs)
    train_num = int(0.85 * total_seen)
    val_num = int(0.10 * total_seen)
    test_intra_num = total_seen - train_num - val_num

    train_objs = seen_objs[:train_num]
    val_objs = seen_objs[train_num:train_num + val_num]
    test_intra_objs = seen_objs[train_num + val_num:]
    test_inter_objs = unseen_objs

    split_dirs = ['train', 'val', 'test_intra', 'test_inter']
    for split_dir in split_dirs:
        for sub_dir in ['meta', 'pth', 'gt']:
            os.makedirs(Path(target_dir) / split_dir / sub_dir, exist_ok=True)

    def copy_files(file_list, split_type):
        for file_path in tqdm(file_list, desc=f"Copying files to {split_type}"):
            base_name = Path(file_path).stem
            for sub_dir in ['meta', 'pth', 'gt']:
                src_path = Path(data_root) / sub_dir / f"{base_name}.{dir_to_postfix[sub_dir]}"
                dst_path = Path(target_dir) / split_type / sub_dir / f"{base_name}.{dir_to_postfix[sub_dir]}"
                shutil.copy(src_path, dst_path)
    
    copy_files(train_objs, 'train')
    copy_files(val_objs, 'val')
    copy_files(test_intra_objs, 'test_intra')
    copy_files(test_inter_objs, 'test_inter')

    print(f"Training files: {len(train_objs)}")
    print(f"Validation files: {len(val_objs)}")
    print(f"Test intra files: {len(test_intra_objs)}")
    print(f"Test inter files: {len(test_inter_objs)}")

if __name__ == "__main__":
    split_data('./sampled_data', './GAPartNet_fix_small', seen_category, unseen_category)
    