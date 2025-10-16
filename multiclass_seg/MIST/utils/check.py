# --- START OF FILE check_data.py ---

import os
import numpy as np
# --- START OF FILE debug_color.py ---

import cv2
import numpy as np
import os

# ---- !! IMPORTANT !! ----
# ---- CHANGE THIS to the full path of one of your RED mask images ----
mask_path = r"C:\Users\jaydu\Downloads\Unlicensed - Classification Dataset\Unlicensed - Classification Dataset\masks\00f9b6c911c4aa179344271fa54aa1b2.jpeg"

# ---------------------------------------------------------------------

if not os.path.exists(mask_path):
    print(f"ERROR: The file was not found at the path: {mask_path}")
    print("Please update the 'mask_path' variable in this script.")
else:
    # Read the mask image in BGR format, just like data_ready.py does
    mask_bgr = cv2.imread(mask_path)
    
    # Convert to RGB, just like data_ready.py does
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    
    # Find all unique color values in the image
    # We reshape the image to be a list of pixels, then find the unique rows
    pixels = mask_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    print(f"--- Analyzing color values in: {os.path.basename(mask_path)} ---")
    print("Found the following unique RGB color(s):")
    for color in unique_colors:
        print(f"  - {tuple(color)}")

    # Check if the expected red color is present
    expected_red = np.array([255, 0, 0])
    is_present = any(np.array_equal(color, expected_red) for color in unique_colors)

    if is_present:
        print("\nSUCCESS: The expected red color (255, 0, 0) was found!")
    else:
        print("\nERROR: The expected red color (255, 0, 0) was NOT found.")
        print("Please update the COLOR_MAP in data_ready.py with one of the RGB values listed above.")
# ---- CONFIG (Match these to your project) ----
output_dir = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data_with_non_polp\classification_data\train_npz_new"
list_dir = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data_with_non_polp\classification_data\lists_Synapse"

splits_to_check = ["train", "val_vol", "test_vol"]

print("--- Starting Data Distribution and Pixel Analysis ---\n")

for split in splits_to_check:
    list_path = os.path.join(list_dir, f"{split}.txt")
    if not os.path.exists(list_path):
        print(f"List file for '{split}' not found. Skipping.")
        continue

    with open(list_path, "r") as f:
        file_list = [line.strip() for line in f.readlines()]

    # --- New Pixel Counters ---
    # Dictionary to hold the total pixel count for each class
    pixel_counts = {0: 0, 1: 0, 2: 0} 
    total_pixels_in_split = 0

    # --- Original Image Counters ---
    class_1_image_count = 0  # Adenoma
    class_2_image_count = 0  # Hyperplastic
    total_files = len(file_list)

    if total_files == 0:
        print(f"--- Results for '{split}' split ---")
        print("No files found in this split.")
        print("-" * 45 + "\n")
        continue

    for fname in file_list:
        npz_path = os.path.join(output_dir, f"{fname}.npz")
        try:
            data = np.load(npz_path)
            label = data['label']
            
            # --- Image-level analysis ---
            unique_labels = np.unique(label)
            if 1 in unique_labels:
                class_1_image_count += 1
            if 2 in unique_labels:
                class_2_image_count += 1

            # --- Pixel-level analysis ---
            pixel_counts[0] += np.sum(label == 0)
            pixel_counts[1] += np.sum(label == 1)
            pixel_counts[2] += np.sum(label == 2)
            total_pixels_in_split += label.size

        except FileNotFoundError:
            print(f"Warning: Could not find file {npz_path}")

    # --- Print Report for the Split ---
    print(f"--- Results for '{split}' split ---")
    print(f"Total images: {total_files}")
    print("\n[Image-level Counts]")
    print(f"Images containing Class 1 (Adenoma):     {class_1_image_count} ({class_1_image_count/total_files:.1%})")
    print(f"Images containing Class 2 (Hyperplastic): {class_2_image_count} ({class_2_image_count/total_files:.1%})")
    
    print("\n[Pixel-level Counts]")
    print(f"Total pixels in this split: {total_pixels_in_split:,}")
    
    bg_pixels = pixel_counts[0]
    c1_pixels = pixel_counts[1]
    c2_pixels = pixel_counts[2]

    bg_percent = (bg_pixels / total_pixels_in_split) * 100
    c1_percent = (c1_pixels / total_pixels_in_split) * 100
    c2_percent = (c2_pixels / total_pixels_in_split) * 100

    print(f"  - Class 0 (Background):   {bg_pixels:12,} pixels ({bg_percent:.4f}%)")
    print(f"  - Class 1 (Adenoma):      {c1_pixels:12,} pixels ({c1_percent:.4f}%)")
    print(f"  - Class 2 (Hyperplastic): {c2_pixels:12,} pixels ({c2_percent:.4f}%)")
    print("-" * 45 + "\n")

print("--- Analysis Complete ---")