# AI 驅動出行未來：跨相機多目標車輛追蹤競賽 － 模型組

## 安裝環境

1. 建立環境
    ```
    conda create -n star_botsort python=3.7
    conda activate star_botsort
    ```
2. 安裝torch
    ```
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
3. 安裝其他套件
    ```
    <cd Team_5084-main>
    pip install -r requirements.txt
    conda install pycocotools -c conda-forge
    conda install faiss-gpu -c conda-forge
    ```
## Yolov7訓練
    
1. 準備Yolov7資料集
    - 訓練集([trainv2](https://drive.google.com/file/d/1mTDO0SYJ_yzT7PEYYfcCVxsZ1g-AYUgd/view?usp=sharing))
    - 測試集([32_33_AI_CUP_testdataset](https://drive.google.com/file/d/10pBjA7Pc_i6ccnoGLDEsj_eLzTvklizR/view?usp=drive_link)) 
    #### 輸出路徑：`yolo_datasets_random`
    ```
    python yolov7/tools/AICUP_to_YOLOv7_datasets_random.py
    ```
2. 開始訓練
    
    #### pretrain載點及放置路徑：yolov7/pretrained/[yolov7-w6_training.pt](https://drive.google.com/drive/folders/1kSvA6zpf8AKdbX3ffw6SJvmFUaJXvAoL?usp=drive_link)
    訓練結果圖以及權重路徑：`runs/train/Yolov7_datasets_random_epoch_100`
    ```
    python yolov7/train_aux.py
    ```

## ReID資料集準備
為了準備**ReID**資料集，請依照以下步驟操作：

### 移除卡幀圖片並轉換為**ReID**資料集
在數據預處理過程中，我們使用均方誤差（MSE）計算來移除卡幀的圖片，然後將處理過的圖片轉換成**ReID**資料集格式。具體步驟如下：
1. **移除卡幀圖片**
    #### 首先，使用 AICUP_MSE_Frame_Skipping_Handling.py來計算並移除卡幀圖片，並輸出到一個新的資料夾。
    #### 輸出路徑：`trainv2_Frame_Skipping_Handling/images`
    ```
    python fast_reid/datasets/AICUP_MSE_Frame_Skipping_Handling.py
    ```
2. 生成處理後的資料集標籤
    #### 接著使用AICUP_MSE_Frame_Skipping_Handling_label.py來輸出處理後的標籤。
    #### 輸出路徑：`trainv2_Frame_Skipping_Handling/labels`
    ```
    python fast_reid/datasets/AICUP_MSE_Frame_Skipping_Handling_label.py
    ```
3. 轉換成**ReID**資料集
    #### 最後，使用 generate_AICUP_patches.py 將圖片轉換成 **ReID**資料集格式。這一步將處理過的圖片和標籤轉換為**ReID** 模型所需的數據格式。
    #### 輸出路徑：`REID_datasets/day-night_card_detection_ESRGAN/AICUP-ReID/bounding_box_train`
    ```
    python fast_reid/datasets/generate_AICUP_patches.py
    ```
### 抓取車尾部分數據增強並轉換為**ReID**資料集
在數據處理過程中，我們還進行了車尾部分的抓取並轉換為**ReID**資料集格式，具體步驟如下：
1.	抓取車尾部分並生成標籤
    #### 使用 AICUP_Rear_Vehicle_Detection.py，人工設置一個邊界框來抓取進入該框內的車輛，並生成對應的標籤（label.txt）。
    #### 輸出路徑：`trainv2_Frame_Skipping_Handling/inside_poly`
    ```
    python fast_reid/datasets/AICUP_Rear_Vehicle_Detection.py
    ```
2.	轉換標籤為**ReID**資料集格式
    #### 輸出路徑：`REID_datasets/Rear_Vehicle/AICUP-ReID/bounding_box_train`
    ```
    python fast_reid/datasets/generate_Rear_Vehicle_Label_AICUP.py
    ```
3.	數據增強
    #### 使用 Rear_Vehicle_Data_Augmentation.py對轉換後的 ReID 資料集進行數據增強（如旋轉、平移），並將增強後的數據存回**ReID**資料集。
    #### 輸出路徑：`REID_datasets/day-night_card_detection_ESRGAN/AICUP-ReID/bounding_box_train`
    ```    
    python fast_reid/datasets/Rear_Vehicle_Data_Augmentation.py
    ```
### 使用 Real-ESRGAN 增強**ReID**資料集
最後，我們使用來自其他作者的 Real-ESRGAN 對**ReID**資料集進行增強，會先輸出一個臨時資料夾讓使用者觀察增強情況，並將增強後的數據存回**ReID**資料集。
#### 權重載點(weights)及放置路徑：Real-ESRGAN-master/weights/[RealESRGAN_x4plus.pth](https://drive.google.com/drive/folders/1ME5_t9Lut-ZJ7qS5vuvvq8w7Gs2wj6Cv?usp=drive_link)
#### 臨時資料夾路徑：`REID_datasets/temp_1024_Real_ESRGAN`
#### 輸出路徑：`REID_datasets/day-night_card_detection_ESRGAN/AICUP-ReID/bounding_box_train`
```
python Real-ESRGAN-master/inference_realesrgan.py -n RealESRGAN_x4plus --face_enhance
```
### 有關 Real-ESRGAN 的更多資訊，請參考：
- [xinntao Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master)


## ReID訓練

1. 開始訓練
    #### 輸出路徑：`logs/AICUP_115/final_data_resnext50_V19_CE_TRI_COS`
    #### 參數調整路徑：`fast_reid/configs/AICUP/bagtricks_R50-ibn.yml`
    ```
    python fast_reid/tools/train_net.py
    ```
2. 觀察訓練日誌

    #### 為了觀察和分析訓練過程中的各種指標，可以透過以下路徑的檔案來觀察。
    `logs/AICUP_115/final_data_resnext50_V19_CE_TRI_COS/metrics.json`


## Inference
#### 本項目的推理（**Inference**）是使用`trake_all_timestamps.bat`來執行`final_mc_demo_yolov7.py`，**請使用者自行更改執行路徑**。
#### 參數調整路徑：`tools/final_mc_demo_yolov7.py`
#### 輸出路徑：`runs/detect/final_public_private/e6e_test_0.65_box_200`
```
.\trake_all_timestamps.bat
```
使用的模型權重：
在推理過程中，我們使用了自己訓練好的權重文件。
  - YOLOv7 權重文件路徑：`runs/train/Yolov7_datasets_random_epoch_100/weights/best.pt`
  - ReID 模型配置文件路徑：`logs/AICUP_115/final_data_resnext50_V19_CE_TRI_COS/config.yaml`
  - ReID 權重文件路徑：`logs/AICUP_115/final_data_resnext50_V19_CE_TRI_COS/model_final.pth`

## Evaluate

1. 準備數據
    #### 在進行評估之前，需要執行AICUP_to_MOT15.py來將真實數據轉換為提交格式。
    #### 輸出路徑：`AICUP_MOT15`
    ```
    python tools/datasets/AICUP_to_MOT15.py
    ```
2. 開始評估
    #### 輸出路徑：`detect_MOT15/final_public_private/e6e_test_0.65_box_200`
    ```
    python tools/evaluate.py
    ```

## Public & Private

#### 在Public & Private的推理中，我們最後使用的模型權重：
#### 測試時，請注意不要跟訓練時的路徑產生衝突

權重載點及放置路徑：
   - YOLOv7 權重路徑：YOLO_WEIGHT/[best.pt](https://drive.google.com/drive/folders/1XWDbEw2Z9C2leSAiNn7hhLxgf1eNK6QP?usp=drive_link)
   - ReID 模型配置文件路徑：REID_WEIGHT/final_data_resnext50_V19_CE_TRI_COS/config.yaml
   - ReID 權重路徑：REID_WEIGHT/final_data_resnext50_V19_CE_TRI_COS/[model_final.pth](https://drive.google.com/drive/folders/1E_x8MCk6kAjdOUD_HaWGZ1nBTYLtf5qP?usp=drive_link)
#### 最終提交的評估答案：

`detect_MOT15/Public_Priavte_MOT15`
