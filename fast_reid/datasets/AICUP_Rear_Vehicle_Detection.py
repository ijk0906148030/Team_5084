#抓取車尾部分
import os
import cv2
import numpy as np
from tqdm import tqdm

# 檢查一個點是否在多邊形內的函數
def is_point_in_poly(point, poly):
    x, y = point
    n = len(poly)
    inside = False
    
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# 將bbox的數據轉換為圖像上的坐標函數
def convert_bbox(img_width, img_height, data):
    bb_width = float(data[3]) * img_width
    bb_height = float(data[4]) * img_height
    bb_left = float(data[1]) * img_width - bb_width / 2
    bb_top = float(data[2]) * img_height - bb_height / 2
    return (bb_left, bb_top, bb_left + bb_width, bb_top + bb_height)

camera_polygons = {
    '0': np.array([[750, 0], [800, 0], [1060, 180], [1280, 210], [1280, 720], [680, 720]]),
    '2': np.array([[530, 80], [600, 80], [1060, 180], [1280, 210], [1280, 720], [680, 720]]),
    '4': np.array([[790, 0], [800, 0], [1060, 180], [1280, 210], [1280, 720], [680, 720]]),
    '6': np.array([[400, 150], [800, 150], [1060, 180], [1280, 210], [1280, 720], [400, 720]])
}

# 定義數據集路徑和輸出文件夾
dataset_path = 'trainv2_Frame_Skipping_Handling'
output_folder = os.path.join(dataset_path, 'inside_poly')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 圖像文件夾路徑和標籤文件夾路徑
image_folder = os.path.join(dataset_path, 'images')
label_folder = os.path.join(dataset_path, 'labels')

# 遍歷圖像文件夾內所有類別文件夾
for class_folder in os.listdir(image_folder):
    # 創建輸出子文件夾
    output_class_folder = os.path.join(output_folder, class_folder)
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)
    
    # 圖像文件夾路徑和標籤文件夾路徑
    class_image_folder = os.path.join(image_folder, class_folder)
    class_label_folder = os.path.join(label_folder, class_folder)
    
    # 遍歷類別文件夾內所有圖像文件
    for filename in tqdm(os.listdir(class_image_folder)):
        camera_id = filename.split('_')[0]  # 提取相機ID
        if camera_id not in camera_polygons:
            # 如果相機ID不在多邊形列表中，創建空的txt文件
            label_filename = filename.split('.')[0] + '.txt'
            open(os.path.join(output_class_folder, label_filename), 'w').close()
            continue
        
        # 讀取圖像尺寸
        image_path = os.path.join(class_image_folder, filename)
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        # 讀取標籤並處理需要判斷的圖片
        label_filename = filename.split('.')[0] + '.txt'
        label_path = os.path.join(class_label_folder, label_filename)
        output_label_path = os.path.join(output_class_folder, label_filename)
        
        poly = camera_polygons[camera_id]  # 獲取相機對應的多邊形
        with open(output_label_path, 'w') as output_file, open(label_path, 'r') as label_file:
            labels = label_file.readlines()
            for label in labels:
                data = label.strip().split()
                bbox = convert_bbox(img_width, img_height, data)  # 轉換邊界格式
                bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                # 如果邊界框中心在多邊形內，寫入標籤
                if is_point_in_poly(bbox_center, poly):
                    output_file.write(label)

