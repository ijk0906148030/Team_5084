import os
import glob
import shutil
import argparse
import random
from tqdm import tqdm

# # 定義工作路徑
# working_directory = r'E:\AI_CUP\AI-driven_Future\AICUP_Baseline_BoT-SORT-main'
# os.chdir(working_directory)
# print("Current Working Directory:", os.getcwd())

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--AICUP_dir', type=str, default='trainv2', help='your AICUP train dataset path')
    parser.add_argument('--YOLOv7_dir', type=str, default='yolo_datasets_final', help='converted dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='The ratio of the train set when splitting the train set and the validation set')

    opt = parser.parse_args()
    return opt

def aicup_to_yolo(args):
    # 創建訓練集和驗證集的目錄
    train_dir = os.path.join(args.YOLOv7_dir, 'train')
    valid_dir = os.path.join(args.YOLOv7_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    
    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)
    
    # 獲取所有圖片和標籤
    all_image_files = sorted(glob.glob(os.path.join(args.AICUP_dir, 'images', '**', '*.jpg'), recursive=True))
    all_label_files = sorted(glob.glob(os.path.join(args.AICUP_dir, 'labels', '**', '*.txt'), recursive=True))
    
    # 確保圖片和標籤文件是成對的
    assert len(all_image_files) == len(all_label_files), "The number of images does not match the number of label files."
    
    # 打亂文件列表
    paired_files = list(zip(all_image_files, all_label_files))
    random.shuffle(paired_files)
    
    # 分割成訓練集和驗證集
    total_count = len(paired_files)
    train_count = int(total_count * args.train_ratio)
    
    train_files = paired_files[:train_count]
    valid_files = paired_files[train_count:]
    
    # 複製文件到相應的新目錄並重命名
    def copy_and_rename(files, dest_dir, start_index=0):
        for i, (img_file, label_file) in enumerate(tqdm(files, desc=f'Copying to {dest_dir}')):
            new_base_name = f'image_{start_index + i}'
            shutil.copy2(img_file, os.path.join(dest_dir, 'images', f'{new_base_name}.jpg'))
            shutil.copy2(label_file, os.path.join(dest_dir, 'labels', f'{new_base_name}.txt'))

    copy_and_rename(train_files, train_dir)
    copy_and_rename(valid_files, valid_dir, start_index=train_count)

    return 0

def delete_track_id(labels_dir):
    for file_path in tqdm(glob.glob(os.path.join(labels_dir, '*.txt')), desc='Deleting track ID'):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) > 5:
                new_lines.append(' '.join(parts[:-1]) + '\n')

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    return 0

if __name__ == '__main__':
    args = arg_parse()

    # # 修改這些路徑和比例，根據需要
    # args.AICUP_dir = 'trainv2'  
    # args.YOLOv7_dir = 'yolo_datasets_random'
    # args.train_ratio = 0.8

    aicup_to_yolo(args)

    train_dir = os.path.join(args.YOLOv7_dir, 'train', 'labels')
    val_dir = os.path.join(args.YOLOv7_dir, 'valid', 'labels')
    delete_track_id(train_dir)
    delete_track_id(val_dir)
