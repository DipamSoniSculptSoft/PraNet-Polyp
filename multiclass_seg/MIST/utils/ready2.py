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
    mask_arr = np.array(mask, dtype=np.int32)
    h, w, _ = mask_arr.shape
    label = np.zeros((h, w), dtype=np.uint8)
    red_mask = (mask_arr[:, :, 0] > 150) & (mask_arr[:, :, 1] < 100) & (mask_arr[:, :, 2] < 100)
    label[red_mask] = 1
    green_mask = (mask_arr[:, :, 1] > 150) & (mask_arr[:, :, 0] < 100) & (mask_arr[:, :, 2] < 100)
    # Condition 2: Pixel is white (high values in all R, G, and B channels)
    # A threshold of 220 is used to be safe against off-white colors from compression artifacts.
    white_condition = (mask_arr[:, :, 0] > 220) & (mask_arr[:, :, 1] > 220) & (mask_arr[:, :, 2] > 220)

    # A pixel belongs to the hyperplastic class if it meets EITHER the green OR the white condition.
    # The `|` symbol represents the logical OR operation for NumPy arrays.
    hyperplastic_mask = green_mask | white_condition
    label[hyperplastic_mask] = 2  # Assign class 2 (hyperplastic)
    
    return label

# ==============================================================================
# ---- NEW HELPER FUNCTION TO FIND MASK WITH ANY EXTENSION ----
# ==============================================================================
def find_mask_path(image_filename, mask_directory):
    """
    Finds the corresponding mask file for a given image, checking multiple extensions.
    Returns the full path to the mask if found, otherwise returns None.
    """
    base_name = os.path.splitext(image_filename)[0]
    possible_extensions = ['.png', '.jpg', '.jpeg', '.tif']
    for ext in possible_extensions:
        mask_filename = base_name + ext
        potential_path = os.path.join(mask_directory, mask_filename)
        if os.path.exists(potential_path):
            return potential_path  # Found it! Return the path.
    return None  # If loop finishes, no mask was found.
# ==============================================================================

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

        # ==============================================================================
        # ---- MODIFIED LOGIC TO FIND THE MASK ----
        # ==============================================================================
        # Old way: mask_path = os.path.join(mask_dir, fname)
        # New way: Use the helper function to find the mask regardless of extension.
        mask_path = find_mask_path(fname, mask_dir)

        if mask_path is None: # The function returns None if no mask is found
            print(f"⚠️ Missing mask for {fname}, skipping.")
            continue
        # ==============================================================================

        # Read image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path)
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