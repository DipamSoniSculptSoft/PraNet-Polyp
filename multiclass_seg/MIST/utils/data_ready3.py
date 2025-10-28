# # =====================================================================================
# # DATASET PREPARATION SCRIPT FOR MULTI-CLASS SEGMENTATION
# #
# # DESCRIPTION:
# # This script is designed to process a raw dataset of images and their corresponding
# # colored masks, preparing them for training a multi-class segmentation model.
# # It handles data loading, splitting, preprocessing, and saving in a format
# # that is efficient for model training pipelines.
# #
# # KEY FEATURES:
# #   -   Environment-Based Configuration: Uses a `.env` file to manage all file
# #       paths, making the script portable and easy to configure without changing
# #       the code itself.
# #   -   Automatic Data Splitting: Splits the dataset into training, validation,
# #       and test sets based on configurable ratios.
# #   -   Color Mask to Class Conversion: Converts RGB color masks into integer-based
# #       class labels (e.g., Red -> Class 1, Green -> Class 2). The logic is
# #       robust to handle minor color variations from image compression (like JPEG).
# #   -   Handles Maskless Images: If an image does not have a corresponding mask file,
# #       it is automatically treated as a "background-only" sample and included in
# #       the dataset. This is crucial for teaching the model what negative samples
# #       look like.
# #   -   Flexible File Extensions: Automatically searches for mask files with
# #       different extensions (.png, .jpg, .jpeg) to match an image.
# #   -   Standardized Output: Resizes all images and masks to a specified input
# #       size and saves them as compressed NumPy archives (`.npz`), which is
# #       highly efficient for loading during training. It also generates `.txt`
# #       files listing the contents of each data split.
# #
# # HOW TO USE (FOR A NEW USER FETCHING FROM GITHUB):
# #   1.  Install Dependencies:
# #       pip install opencv-python numpy scikit-learn python-dotenv
# #
# #   2.  Create a `.env` File: In the same directory as this script, create a
# #       file named `.env`. This is the MOST IMPORTANT step. Your file should
# #       look like this, with the paths updated for your machine:
# #
# #       # --- Example .env file ---
# #       dataset_image_path="C:/path/to/your/images"
# #       dataset_mask_path="C:/path/to/your/masks"
# #       dataset_npz_path="C:/path/to/save/processed_npz_files"
# #       dataset_synapse_path="C:/path/to/save/list_files"
# #       # -------------------------
# #
# #   3.  Organize Your Data:
# #       -   Place all your original images in the `dataset_image_path` folder.
# #       -   Place all your corresponding color masks in the `dataset_mask_path` folder.
# #       -   Image and mask files should share the same base name (e.g., `img1.jpg`
# #         and `img1.png`).
# #
# #   4.  Adjust Configuration (Optional):
# #       -   You can change the `img_size`, `val_split`, or `test_split` variables
# #         in the "CONFIG" section of this script if needed.
# #
# #   5.  Run the Script:
# #       Execute the script from your terminal: python your_script_name.py
# #
# # =====================================================================================

# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from dotenv import load_dotenv  # Import the dotenv library

# # Load environment variables from a .env file
# load_dotenv()
# image_dir = os.getenv('dataset_image_path')
# mask_dir = os.getenv('dataset_mask_path')
# output_dir = os.getenv('dataset_npz_path')
# list_dir = os.getenv('dataset_synapse_path')
# # ---- CONFIG ----
# # These parameters can be adjusted as needed.
# img_size = (256, 256)   # The target size for model input (width, height)
# val_split = 0.2         # 20% of the data will be used for validation
# test_split = 0.1        # 10% of the data will be used for testing
# # The remaining 70% will be used for training.


# # Create output directories if they don't already exist
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(list_dir, exist_ok=True)

# # ---- COLOR TO LABEL MAPPING ----
# # Defines how RGB colors in the mask files correspond to integer class labels.
# COLOR_MAP = {
#     (255, 0, 0): 1,   # red → class 1 (e.g., adenoma)
#     (0, 255, 0): 2,   # green → class 2 (e.g., hyperplastic)
# }

# def mask_to_class(mask):
#     """
#     Convert a color mask to a class-indexed mask (0=background, 1=class_1, 2=class_2).
#     This uses a robust thresholding method to handle compression artifacts (e.g., from JPEG).
#     """
#     mask_arr = np.array(mask, dtype=np.int32)
#     h, w, _ = mask_arr.shape
#     label = np.zeros((h, w), dtype=np.uint8)

#     # Condition for Class 1 (predominantly red)
#     red_mask = (mask_arr[:, :, 0] > 150) & (mask_arr[:, :, 1] < 100) & (mask_arr[:, :, 2] < 100)
#     label[red_mask] = 1

#     # Condition for Class 2 (predominantly green or pure white)
#     green_mask = (mask_arr[:, :, 1] > 150) & (mask_arr[:, :, 0] < 100) & (mask_arr[:, :, 2] < 100)
#     white_condition = (mask_arr[:, :, 0] > 220) & (mask_arr[:, :, 1] > 220) & (mask_arr[:, :, 2] > 220)
#     hyperplastic_mask = green_mask | white_condition
#     label[hyperplastic_mask] = 2

#     return label

# def find_mask_path(image_filename, mask_directory):
#     """
#     Finds the corresponding mask file for a given image, checking multiple common extensions.
#     Returns the full path to the mask if found, otherwise returns None.
#     """
#     base_name = os.path.splitext(image_filename)[0]
#     possible_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
#     for ext in possible_extensions:
#         mask_filename = base_name + ext
#         potential_path = os.path.join(mask_directory, mask_filename)
#         if os.path.exists(potential_path):
#             return potential_path
#     return None

# # ---- 1. Collect all image files ----
# all_images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))])
# print(f"Found {len(all_images)} total images in '{image_dir}'")

# # ---- 2. Split into training, validation, and test sets ----
# train_imgs, val_test = train_test_split(all_images, test_size=val_split + test_split, random_state=42, shuffle=True)
# val_imgs, test_imgs = train_test_split(val_test, test_size=test_split / (val_split + test_split), random_state=42, shuffle=True)

# splits = {
#     "train": train_imgs,
#     "valid": val_imgs,
#     "test": test_imgs
# }

# # ---- 3. Process and save each split ----
# for split_name, file_list in splits.items():
#     print(f"\nProcessing '{split_name}' set ({len(file_list)} samples)...")
#     for fname in file_list:
#         img_path = os.path.join(image_dir, fname)

#         # Read the image as grayscale.
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             print(f"⚠️  Could not read image {fname}, skipping.")
#             continue

#         # Use the helper function to find the corresponding mask path.
#         mask_path = find_mask_path(fname, mask_dir)

#         # --- Logic to handle both masked and maskless images ---
#         if mask_path is not None:
#             # If a mask was found, load it and convert it to class labels.
#             mask = cv2.imread(mask_path)
#             mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#             label = mask_to_class(mask_rgb)

#             # Verification Step: Check if the mask file was interpreted correctly.
#             if not np.any(label > 0):
#                 print(f"⚠️  WARNING: Mask found for '{fname}', but no classes were detected after color conversion.")
#                 print("    This could mean the colors in the mask are not pure red/green or the thresholds need adjustment.")
#                 print("    The file will be saved as a background-only sample.")
#         else:
#             # If no mask was found, create a background-only label (all zeros).
#             # print(f"🔹 No mask found for {fname}, creating a background-only label.") # Uncomment for verbose output
#             h, w = image.shape
#             label = np.zeros((h, w), dtype=np.uint8)
#         # --- End of logic ---

#         # Resize both the image and the final label to the target dimensions.
#         image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
#         label_resized = cv2.resize(label, img_size, interpolation=cv2.INTER_NEAREST)

#         # Normalize image pixel values to the [0, 1] range.
#         image_normalized = image_resized.astype(np.float32) / 255.0

#         # Save the processed image and label as a compressed .npz file.
#         name = os.path.splitext(fname)[0]
#         save_path = os.path.join(output_dir, f"{name}.npz")
#         np.savez_compressed(save_path, image=image_normalized, label=label_resized)

#     # After processing all files in a split, save the list of filenames.
#     list_path = os.path.join(list_dir, f"{split_name}.txt")
#     with open(list_path, "w") as f:
#         for fname in file_list:
#             name = os.path.splitext(fname)[0]
#             f.write(name + "\n")
#     print(f"Saved list file: {list_path}")

# print("\n✅ Dataset preparation complete.")
# print(f"   - NPZ files saved to: {output_dir}")
# print(f"   - List files saved to: {list_dir}")
# =====================================================================================
# DATASET PREPARATION SCRIPT FOR MULTI-CLASS SEGMENTATION (AUTO-DETECT VERSION)
# =====================================================================================

# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# # ---- AUTO-DETECT PROJECT PATHS ----
# # Automatically resolve all dataset paths based on your folder structure.

# # Get the absolute path to this script
# current_file = os.path.abspath(__file__)

# # Go up three levels: utils → MIST → multiclass_seg → PraNet-V2 (root)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

# # Dataset folder path
# dataset_root = os.path.join(project_root, "Dataset", "Dataset_2_0")

# # Subdirectories
# image_dir = os.path.join(dataset_root, "images")
# mask_dir = os.path.join(dataset_root, "masks")
# output_dir = os.path.join(dataset_root, "train_npz_new")
# list_dir = os.path.join(dataset_root, "list_Synapse")

# # Create output directories if they don't already exist
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(list_dir, exist_ok=True)

# print("📂 Auto-detected dataset structure:")
# print(f"   Images Path : {image_dir}")
# print(f"   Masks Path  : {mask_dir}")
# print(f"   Output Path : {output_dir}")
# print(f"   Lists Path  : {list_dir}")

# # ---- CONFIG ----
# img_size = (256, 256)
# val_split = 0.2
# test_split = 0.1

# # ---- COLOR TO LABEL MAPPING ----
# COLOR_MAP = {
#     (255, 0, 0): 1,   # red → class 1 (e.g., adenoma)
#     (0, 255, 0): 2,   # green → class 2 (e.g., hyperplastic)
# }

# def mask_to_class(mask):
#     """Convert color mask to class-indexed mask (0=background, 1=class1, 2=class2)."""
#     mask_arr = np.array(mask, dtype=np.int32)
#     h, w, _ = mask_arr.shape
#     label = np.zeros((h, w), dtype=np.uint8)

#     # Red detection
#     red_mask = (mask_arr[:, :, 0] > 150) & (mask_arr[:, :, 1] < 100) & (mask_arr[:, :, 2] < 100)
#     label[red_mask] = 1

#     # Green or white detection
#     green_mask = (mask_arr[:, :, 1] > 150) & (mask_arr[:, :, 0] < 100) & (mask_arr[:, :, 2] < 100)
#     white_condition = (mask_arr[:, :, 0] > 220) & (mask_arr[:, :, 1] > 220) & (mask_arr[:, :, 2] > 220)
#     label[green_mask | white_condition] = 2

#     return label

# def find_mask_path(image_filename, mask_directory):
#     """Find the corresponding mask for an image (supports multiple extensions)."""
#     base_name = os.path.splitext(image_filename)[0]
#     for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
#         path = os.path.join(mask_directory, base_name + ext)
#         if os.path.exists(path):
#             return path
#     return None

# # ---- 1. Collect all image files ----
# all_images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
# print(f"\n🖼️  Found {len(all_images)} total images in '{image_dir}'")

# # ---- 2. Split into training, validation, and test sets ----
# train_imgs, val_test = train_test_split(all_images, test_size=val_split + test_split, random_state=42, shuffle=True)
# val_imgs, test_imgs = train_test_split(val_test, test_size=test_split / (val_split + test_split), random_state=42, shuffle=True)

# splits = {"train": train_imgs, "valid": val_imgs, "test": test_imgs}

# # ---- 3. Process and save each split ----
# for split_name, file_list in splits.items():
#     print(f"\n🚀 Processing '{split_name}' set ({len(file_list)} samples)...")
#     for fname in file_list:
#         img_path = os.path.join(image_dir, fname)

#         # Read image as grayscale
#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             print(f"⚠️ Could not read image {fname}, skipping.")
#             continue

#         # Find and process mask
#         mask_path = find_mask_path(fname, mask_dir)
#         if mask_path:
#             mask = cv2.imread(mask_path)
#             mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#             label = mask_to_class(mask_rgb)
#             if not np.any(label > 0):
#                 print(f"⚠️ Mask found for '{fname}' but no valid class detected.")
#                 label = np.zeros_like(image, dtype=np.uint8)
#         else:
#             label = np.zeros_like(image, dtype=np.uint8)

#         # Resize and normalize
#         image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
#         label_resized = cv2.resize(label, img_size, interpolation=cv2.INTER_NEAREST)
#         image_normalized = image_resized.astype(np.float32) / 255.0

#         # Save compressed file
#         save_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.npz")
#         np.savez_compressed(save_path, image=image_normalized, label=label_resized)

#     # Save filename list for split
#     list_path = os.path.join(list_dir, f"{split_name}.txt")
#     with open(list_path, "w") as f:
#         for fname in file_list:
#             f.write(os.path.splitext(fname)[0] + "\n")
#     print(f"✅ Saved list file: {list_path}")

# print("\n🎯 Dataset preparation complete!")
# print(f"   NPZ files saved to: {output_dir}")
# print(f"   List files saved to: {list_dir}")
# =====================================================================================
# DATASET PREPARATION SCRIPT FOR MULTI-CLASS SEGMENTATION (AUTO-DETECT VERSION)
#
# DESCRIPTION:
# This updated script now explicitly handles a separate folder for negative samples
# in addition to the main images and masks folders.
# =====================================================================================

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ---- AUTO-DETECT PROJECT PATHS ----
# Automatically resolve all dataset paths based on your folder structure.

# Get the absolute path to this script
current_file = os.path.abspath(__file__)

# Go up three levels: utils → MIST → multiclass_seg → PraNet-V2 (root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

# Dataset folder path
dataset_root = os.path.join(project_root, "Dataset", "classification_dataset")

# Subdirectories
image_dir = os.path.join(dataset_root, "images")
mask_dir = os.path.join(dataset_root, "masks")
# <-- MODIFIED SECTION START -->
# 1. ADDED PATH FOR THE NEGATIVE SAMPLES DIRECTORY
negative_dir = os.path.join(dataset_root, "negative_samples")
# <-- MODIFIED SECTION END -->
output_dir = os.path.join(dataset_root, "train_npz_new")
list_dir = os.path.join(dataset_root, "list_Synapse")


# Create output directories if they don't already exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

print("📂 Auto-detected dataset structure:")
print(f"   Images Path          : {image_dir}")
print(f"   Masks Path           : {mask_dir}")
# <-- MODIFIED SECTION START -->
print(f"   Negative Samples Path: {negative_dir}") # Print the new path
# <-- MODIFIED SECTION END -->
print(f"   Output Path          : {output_dir}")
print(f"   Lists Path           : {list_dir}")


# ---- CONFIG ----
img_size = (256, 256)
val_split = 0.2
test_split = 0.1

# ---- COLOR TO LABEL MAPPING ----
COLOR_MAP = {
    (255, 0, 0): 1,   # red → class 1 (e.g., adenoma)
    (0, 255, 0): 2,   # green → class 2 (e.g., hyperplastic)
}

def mask_to_class(mask):
    """Convert color mask to class-indexed mask (0=background, 1=class1, 2=class2)."""
    mask_arr = np.array(mask, dtype=np.int32)
    h, w, _ = mask_arr.shape
    label = np.zeros((h, w), dtype=np.uint8)

    # Red detection
    red_mask = (mask_arr[:, :, 0] > 150) & (mask_arr[:, :, 1] < 100) & (mask_arr[:, :, 2] < 100)
    label[red_mask] = 1

    # Green or white detection
    green_mask = (mask_arr[:, :, 1] > 150) & (mask_arr[:, :, 0] < 100) & (mask_arr[:, :, 2] < 100)
    white_condition = (mask_arr[:, :, 0] > 220) & (mask_arr[:, :, 1] > 220) & (mask_arr[:, :, 2] > 220)
    label[green_mask | white_condition] = 2

    return label

def find_mask_path(image_filename, mask_directory):
    """Find the corresponding mask for an image (supports multiple extensions)."""
    base_name = os.path.splitext(image_filename)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
        path = os.path.join(mask_directory, base_name + ext)
        if os.path.exists(path):
            return path
    return None

# <-- MODIFIED SECTION START -->
# ---- 1. Collect all image files from both 'images' and 'negative_samples' folders ----
all_image_paths = []
valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')

# Add images that are expected to have masks
for f in sorted(os.listdir(image_dir)):
    if f.lower().endswith(valid_extensions):
        all_image_paths.append(os.path.join(image_dir, f))

# Add images that are explicitly negative samples
for f in sorted(os.listdir(negative_dir)):
    if f.lower().endswith(valid_extensions):
        all_image_paths.append(os.path.join(negative_dir, f))

print(f"\n🖼️  Found {len(all_image_paths)} total images from all sources.")
# <-- MODIFIED SECTION END -->


# ---- 2. Split into training, validation, and test sets ----
# The list now contains full paths, which is what we split
train_paths, val_test_paths = train_test_split(all_image_paths, test_size=val_split + test_split, random_state=42, shuffle=True)
val_paths, test_paths = train_test_split(val_test_paths, test_size=test_split / (val_split + test_split), random_state=42, shuffle=True)

splits = {"train": train_paths, "valid": val_paths, "test": test_paths}

# ---- 3. Process and save each split ----
for split_name, path_list in splits.items():
    print(f"\n🚀 Processing '{split_name}' set ({len(path_list)} samples)...")
    # <-- MODIFIED SECTION START -->
    # 3. LOOP OVER FULL PATHS INSTEAD OF JUST FILENAMES
    for img_path in path_list:
        # Extract just the filename for searching for masks and for saving
        fname = os.path.basename(img_path)
    # <-- MODIFIED SECTION END -->

        # Read image as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Could not read image {fname}, skipping.")
            continue

        # Find and process mask (this logic remains the same)
        # It will find masks for images from the 'images' folder
        # and fail to find them for images from the 'negative_samples' folder, which is correct.
        mask_path = find_mask_path(fname, mask_dir)
        if mask_path:
            mask = cv2.imread(mask_path)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            label = mask_to_class(mask_rgb)
            if not np.any(label > 0):
                print(f"⚠️ Mask found for '{fname}' but no valid class detected.")
                label = np.zeros_like(image, dtype=np.uint8)
        else:
            # This correctly creates a blank mask for all images in 'negative_samples'
            label = np.zeros_like(image, dtype=np.uint8)

        # Resize and normalize
        image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
        label_resized = cv2.resize(label, img_size, interpolation=cv2.INTER_NEAREST)
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Save compressed file
        save_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.npz")
        np.savez_compressed(save_path, image=image_normalized, label=label_resized)

    # Save filename list for split
    list_path = os.path.join(list_dir, f"{split_name}.txt")
    with open(list_path, "w") as f:
        for img_path in path_list:
            fname = os.path.basename(img_path)
            f.write(os.path.splitext(fname)[0] + "\n")
    print(f"✅ Saved list file: {list_path}")

print("\n🎯 Dataset preparation complete!")
print(f"   NPZ files saved to: {output_dir}")
print(f"   List files saved to: {list_dir}")