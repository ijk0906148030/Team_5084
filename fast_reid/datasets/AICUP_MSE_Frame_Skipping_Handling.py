import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np

# 計算兩張圖片的均方誤差（MSE）
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# 處理文件夾中的圖片
def process_images_in_folders(input_folder, output_folder, threshold, num_to_compare=1):
    # 創建輸出文件夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取輸入文件夾中的所有子文件夾（每個子文件夾都包含要處理的圖片）
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # 使用 tqdm 顯示總進度條
    with tqdm(total=len(subfolders), desc="Processing Subfolders") as pbar:
        # 遍歷每個子文件夾
        for subfolder in subfolders:
            # 創建當前子文件夾對應的輸出路徑
            relative_subfolder = os.path.relpath(subfolder, input_folder)  # 相對路徑
            output_subfolder = os.path.join(output_folder, relative_subfolder)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # 獲取當前子文件夾的所有圖片路徑
            image_paths = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.jpg')]
            num_images = len(image_paths)
            
            # 使用 tqdm 顯示當前子文件夾中的圖片處理進度
            for img_path in tqdm(image_paths, desc=f"Processing {relative_subfolder}", leave=False):
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))  # 調整圖片尺寸，加快計算速度
                output_path = os.path.join(output_subfolder, img_name)
                
                # 標記是否保留當前圖片
                keep_image = True
                
                # 加載後續圖片並與當前圖片比較相似度
                img_index = image_paths.index(img_path)
                for j in range(img_index + 1, min(img_index + num_to_compare + 1, num_images)):
                    next_img_path = image_paths[j]
                    next_img = cv2.imread(next_img_path)
                    next_img = cv2.resize(next_img, (100, 100))  # 調整圖片尺寸，加快計算速度
                    mse_value = mse(img, next_img)
                    
                    if mse_value < threshold:
                        keep_image = False
                        break
                
                # 如果符合條件且是第一張圖片，則複製到輸出文件夾
                if keep_image:
                    # 檢查輸出文件夾中是否已存在同名文件
                    if not os.path.exists(output_path):
                        shutil.copy(img_path, output_path)
            
            # 更新進度條
            pbar.update(1)

    print("圖片處理完成！")

# 指定輸入文件夾、輸出文件夾、閾值和相似度比較數量
input_folder = "trainv2/images"
output_folder = "trainv2_Frame_Skipping_Handling/images"
threshold = 50  # 根據需要調整閾值
num_to_compare = 1  # 只與當前圖片後面的1張圖片比較相似度

# 處理圖片
process_images_in_folders(input_folder, output_folder, threshold, num_to_compare)
