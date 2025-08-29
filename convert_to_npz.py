import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Paths
data_dir = 'dataset_classification'  # Your root dir
images_subdir = os.path.join(data_dir, 'images')
masks_subdir = os.path.join(data_dir, 'masks')
output_dir = 'custom_polyp/polyp_npz'
list_dir = 'custom_polyp/lists_Polyp'

# Get all image files
image_files = [f for f in os.listdir(images_subdir) if f.endswith('.jpeg')]
image_files.sort()  # Ensure consistent order

# Split (random seed for reproducibility)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

def convert_file(file_list, split):
    names = []
    for img_file in file_list:
        base_name = os.path.splitext(img_file)[0]  # e.g., hash123
        mask_file = img_file  # Assume same name as image; adjust if masks have different naming (e.g., f"{base_name}_mask.jpeg")
        img_path = os.path.join(images_subdir, img_file)
        mask_path = os.path.join(masks_subdir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Skipping {img_file}: mask not found at {mask_path}")
            continue

        # Load image (RGB, C=3, H, W)
        image = cv2.imread(img_path)[..., ::-1].transpose(2, 0, 1).astype(np.float32)  # To RGB, CHW

        # Load mask, convert colors to labels
        mask = cv2.imread(mask_path)
        label = np.zeros(mask.shape[:2], dtype=np.uint8)
        # Adenoma (red-ish: high R, low G/B; threshold >100 for robustness)
        adenoma = (mask[:,:,2] > 100) & (mask[:,:,1] < 50) & (mask[:,:,0] < 50)
        # Hyperplastic (green-ish: high G, low R/B)
        hyper = (mask[:,:,1] > 100) & (mask[:,:,2] < 50) & (mask[:,:,0] < 50)
        label[adenoma] = 1
        label[hyper] = 2
        # Rest is background=0

        # Save .npz
        npz_path = os.path.join(output_dir, f"{base_name}.npz")
        np.savez(npz_path, image=image, label=label)
        names.append(base_name)
    # Write list
    with open(os.path.join(list_dir, f"{split}.txt"), 'w') as f:
        f.write('\n'.join(names))

convert_file(train_files, 'train')
convert_file(val_files, 'test_vol')