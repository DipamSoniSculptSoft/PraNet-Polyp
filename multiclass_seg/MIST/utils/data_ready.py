import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
image_dir = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data\images"
mask_dir = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data\masks"
output_dir = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data\train_npz_new"
list_dir = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data\lists_Synapse"
img_size = (256, 256)   # model input size
val_split = 0.2         # 20% validation
test_split = 0.1        # 10% test

os.makedirs(output_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

# ---- COLOR TO LABEL MAPPING ----
COLOR_MAP = {
    (255, 0, 0): 1,   # red → adenoma
    (0, 255, 0): 2,   # green → hyperplastic
}

def mask_to_class(mask):
    """
    Convert color mask to class-indexed mask (0=bg, 1=adenoma, 2=hyperplastic)
    using a robust thresholding method to handle JPEG artifacts.
    """
    # Ensure the mask is a NumPy array
    mask_arr = np.array(mask, dtype=np.int32)
    h, w, _ = mask_arr.shape
    label = np.zeros((h, w), dtype=np.uint8)

    # --- Define thresholds for each color ---
    # For red (adenoma), we want high R, low G, low B
    red_mask = (mask_arr[:, :, 0] > 150) & (mask_arr[:, :, 1] < 100) & (mask_arr[:, :, 2] < 100)
    label[red_mask] = 1  # Assign class 1 (adenoma)

    # For green (hyperplastic), we want high G, low R, low B
    green_mask = (mask_arr[:, :, 1] > 150) & (mask_arr[:, :, 0] < 100) & (mask_arr[:, :, 2] < 100)
    label[green_mask] = 2  # Assign class 2 (hyperplastic)
    
    return label

# ---- Collect image files ----
all_images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))])
print(f"Found {len(all_images)} images")

# ---- Split into train, val, test ----
train_imgs, val_test = train_test_split(all_images, test_size=val_split + test_split, random_state=42)
val_imgs, test_imgs = train_test_split(val_test, test_size=test_split / (val_split + test_split), random_state=42)

splits = {
    "train": train_imgs,
    "val_vol": val_imgs,
    "test_vol": test_imgs
}

# ---- Process and save ----
for split_name, file_list in splits.items():
    print(f"Processing {split_name} set ({len(file_list)} samples)...")
    for fname in file_list:
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            print(f"⚠️ Missing mask for {fname}, skipping.")
            continue

        # Read image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # single channel
        mask = cv2.imread(mask_path)  # color image (BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Convert mask colors → class indices
        label = mask_to_class(mask)

        # Resize both
        image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, img_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Save as .npz
        name = os.path.splitext(fname)[0]
        save_path = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(save_path, image=image, label=label)

    # Save list file
    list_path = os.path.join(list_dir, f"{split_name}.txt")
    with open(list_path, "w") as f:
        for fname in file_list:
            name = os.path.splitext(fname)[0]
            f.write(name + "\n")
    print(f"Saved {list_path}")

print("\n✅ Dataset preparation complete.")
print(f"NPZ files saved to: {output_dir}")
print(f"List files saved to: {list_dir}")
