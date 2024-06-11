import os
import shutil
from tqdm import tqdm


def copy_files_with_structure(source_folder, destination_folder):
    # 遍歷源文件夾中的所有子文件夾和文件
    for root, dirs, files in os.walk(source_folder):
        # 構建目標文件夾中對應的子文件夾路徑
        relative_root = os.path.relpath(root, source_folder)
        dest_root = os.path.join(destination_folder, relative_root)

        # 確保目標子文件夾存在，如果不存在則創建
        os.makedirs(dest_root, exist_ok=True)

        # 複製源文件夾中的文件到目標文件夾中的相應位置
        for file in files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_root, file)
            shutil.copy(source_file_path, dest_file_path)

# 定義主圖像文件夾、主標籤文件夾和目標主文件夾的路徑
main_image_folder = 'trainv2_Frame_Skipping_Handling/images'
main_label_folder = 'trainv2/labels'
output_main_folder = 'trainv2_Frame_Skipping_Handling/labels'

# 確保輸出主文件夾存在，如果不存在則創建
os.makedirs(output_main_folder, exist_ok=True)

# 獲取主標籤文件夾中所有的子文件夾（每個子文件夾代表一個類別）
subfolders = [f.path for f in os.scandir(main_label_folder) if f.is_dir()]

# 遍歷每個子文件夾（每個子文件夾對應一個類別）
for subfolder in subfolders:
    class_name = os.path.basename(subfolder)  # 類別名稱（子文件夾名稱）

    # 獲取當前類別對應的主圖像文件夾和主標籤文件夾的路徑
    class_image_folder = os.path.join(main_image_folder, class_name)
    class_label_folder = os.path.join(main_label_folder, class_name)

    # 確保當前類別對應的主圖像文件夾存在
    if os.path.exists(class_image_folder):
        # 創建當前類別對應的目標輸出文件夾路徑
        output_class_folder = os.path.join(output_main_folder, class_name)
        os.makedirs(output_class_folder, exist_ok=True)

        # 獲取當前類別對應的主圖像文件夾中所有的圖像文件名
        image_files = [f for f in os.listdir(class_image_folder) if f.endswith('.jpg') or f.endswith('.png')]

        # 遍歷當前類別對應的主標籤文件夾中的所有標籤文件
        label_files = [f for f in os.listdir(class_label_folder) if f.endswith('.txt')]

        # 遍歷當前類別對應的主標籤文件夾中的每個標籤文件
        for label_file in tqdm(label_files, desc=f"Processing {class_name}"):
            label_name = os.path.splitext(label_file)[0]  # 提取標籤文件名（不含擴展名）

            # 假設標籤文件名與圖像文件名相同，擴展名為 .txt
            image_file = f'{label_name}.jpg'  # 假設圖像文件名與標籤文件名對應
            image_path = os.path.join(class_image_folder, image_file)

            # 檢查當前標籤文件是否與圖像文件名相對應
            if os.path.exists(image_path):
                # 如果相對應，複製標籤文件到目標輸出文件夾中的對應類別文件夾
                shutil.copy(os.path.join(class_label_folder, label_file), os.path.join(output_class_folder, label_file))

print("文件處理完成！")
