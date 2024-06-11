#數據增強 翻轉旋轉 平移 (將REID_DATASETS車尾丟進總DATA)
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from tqdm import tqdm

# Define a preprocessing function that will resize images to 256x256
def resize_image(image):
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

# Include the preprocessing function in the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    featurewise_center=True,
    preprocessing_function=resize_image
)

# Original and destination image directories
source_dir = 'REID_datasets/Rear_Vehicle/AICUP-ReID/bounding_box_train'
dest_dir = 'REID_datasets/day-night_card_detection_ESRGAN/AICUP-ReID/bounding_box_train'
os.makedirs(dest_dir, exist_ok=True)  # Create the destination folder if it doesn't exist

# Iterate through all the images in the original image folder
for filename in tqdm(os.listdir(source_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        filebase = os.path.splitext(filename)[0]  # Get the base file name without extension
        img_path = os.path.join(source_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = np.expand_dims(img, axis=0)

        # Data augmentation iteration generation
        aug_iter = datagen.flow(
            img_tensor,
            batch_size=1,
            save_to_dir=dest_dir,
            save_prefix=filebase + '_aug',  # Add '_aug' to the original filebase
            save_format='bmp'
        )

        # Generate and save a certain number of augmented images
        num_augmented_images = 2
        for i in range(num_augmented_images):
            next(aug_iter)

# Data augmentation complete
print("Data augmentation is complete and saved to '{}' folder.".format(dest_dir))