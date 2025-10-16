# =====================================================================================
# DATASET ANALYSIS & VERIFICATION SCRIPT
#
# DESCRIPTION:
# This script is a utility for analyzing and verifying a dataset that has already
# been processed into the `.npz` format by a data preparation script. Its primary
# goal is to generate detailed statistical reports for each data split (train,
# validation, test).
#
# This analysis is crucial for understanding the dataset's characteristics before
# model training. It helps answer key questions such as:
#   - Is there a significant class imbalance at the pixel level?
#   - How many images in each split contain a specific class?
#   - Were images without masks (background-only) correctly processed and included?
#
# KEY FEATURES:
#   -   Statistical Reporting: Provides both image-level and pixel-level counts
#       and percentages for each class.
#   -   Class Balance Analysis: Clearly shows the distribution of classes, which
#       can inform decisions about loss weighting or data augmentation.
#   -   Background Verification: Specifically reports on the number and percentage
#       of "background-only" images, confirming that negative samples are present.
#   -   Environment-Based Configuration: Uses a `.env` file to load necessary
#       paths, making it easy to run without modifying the code.
#
# HOW TO USE (FOR A NEW USER FETCHING FROM GITHUB):
#   1.  Run the Data Preparation Script First: This script analyzes the OUTPUT of
#       your data preparation process. You must run the script that generates the
#       `.npz` files and the `lists_Synapse` text files *before* using this one.
#
#   2.  Install Dependencies:
#       pip install numpy python-dotenv
#
#   3.  Verify Your `.env` File: This script uses the SAME `.env` file as your
#       data preparation script. Ensure the paths are correct, especially:
#       - `dataset_npz_path`: Path to the folder with your `.npz` files.
#       - `dataset_synapse_path`: Path to the folder with your `train.txt`,
#         `val_vol.txt`, etc., files.
#
#   4.  Run the Script:
#       Execute from your terminal: python your_script_name.py
#
#   5.  Review the Output: The script will print a detailed report for each
#       data split, allowing you to assess the quality and balance of your dataset.
#
# =====================================================================================

import os
import numpy as np
from dotenv import load_dotenv

# ---- CONFIG (Loads paths from your .env file) ----
load_dotenv()
output_dir = os.getenv('dataset_npz_path')
list_dir = os.getenv('dataset_synapse_path')

splits_to_check = ["train", "val_vol", "test_vol"]

print("--- Starting Data Distribution and Pixel Analysis ---\n")

for split in splits_to_check:
    list_path = os.path.join(list_dir, f"{split}.txt")
    if not os.path.exists(list_path):
        print(f"List file for '{split}' not found at '{list_path}'. Skipping.")
        continue

    with open(list_path, "r") as f:
        file_list = [line.strip() for line in f.readlines()]

    # --- Counters for statistics ---
    pixel_counts = {0: 0, 1: 0, 2: 0}
    total_pixels_in_split = 0
    class_1_image_count = 0
    class_2_image_count = 0
    background_only_files = []
    total_files = len(file_list)

    if total_files == 0:
        print(f"--- Results for '{split}' split ---")
        print("No files found in this split's list file.")
        print("-" * 45 + "\n")
        continue

    for fname in file_list:
        npz_path = os.path.join(output_dir, f"{fname}.npz")
        try:
            data = np.load(npz_path)
            label = data['label']

            # --- Image-level analysis ---
            unique_labels = np.unique(label)

            # Check for presence of each class
            has_class_1 = 1 in unique_labels
            has_class_2 = 2 in unique_labels

            if has_class_1:
                class_1_image_count += 1
            if has_class_2:
                class_2_image_count += 1

            # Check if the image contains ONLY the background class
            if not has_class_1 and not has_class_2:
                background_only_files.append(fname)

            # --- Pixel-level analysis ---
            pixel_counts[0] += np.sum(label == 0)
            pixel_counts[1] += np.sum(label == 1)
            pixel_counts[2] += np.sum(label == 2)
            total_pixels_in_split += label.size

        except FileNotFoundError:
            print(f"Warning: Could not find file {npz_path}")
        except Exception as e:
            print(f"An error occurred while processing {npz_path}: {e}")


    # --- Print the final report for the split ---
    print(f"--- Results for '{split}' split ---")
    print(f"Total images: {total_files}")

    # Report on Background-Only Images
    bg_only_count = len(background_only_files)
    print("\n[Background-Only Image Analysis]")
    print(f"Images containing ONLY background: {bg_only_count} ({bg_only_count/total_files:.1%})")
    if bg_only_count > 0:
        # Print first 10 filenames for brevity
        display_files = ", ".join(background_only_files[:10])
        if bg_only_count > 10:
            display_files += "..."
        print(f"  - Sample Files: [{display_files}]")

    # Report on Per-Class Image Counts
    print("\n[Per-Class Image Counts]")
    print(f"Images containing Class 1 (e.g., Adenoma):     {class_1_image_count} ({class_1_image_count/total_files:.1%})")
    print(f"Images containing Class 2 (e.g., Hyperplastic): {class_2_image_count} ({class_2_image_count/total_files:.1%})")

    # Report on Pixel-level Distribution
    print("\n[Pixel-level Distribution]")
    if total_pixels_in_split > 0:
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