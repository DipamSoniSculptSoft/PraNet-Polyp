# =====================================================================================
# DATASET PREPARATION SCRIPT FOR MULTI-CLASS SEGMENTATION (PROFESSIONAL VERSION)
#
# DESCRIPTION:
# This script has been upgraded to include professional software engineering practices:
#   - Argument Parsing: Configure paths and settings via the command line.
#   - Logging: Professional status and error reporting.
#   - Modularity: Code is organized into functions for clarity and reuse.
#   - Modern Tooling: Uses the `pathlib` library for clean path management.
#
# HOW TO RUN:
# 1. From your terminal, you can run it with default paths (if your structure matches):
#    python data_ready_professional.py
#
# 2. Or, you can override any setting from the command line:
#    python data_ready_professional.py \
#        --image_dir "/path/to/your/images" \
#        --mask_dir "/path/to/your/masks" \
#        --negative_dir "/path/to/your/negative_samples" \
#        --output_dir "/path/to/save/npz" \
#        --img_size 384
# =====================================================================================

import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# 1. SETUP LOGGING FRAMEWORK (Replaces print())
# This provides structured, controllable output for status and errors.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---- COLOR TO LABEL MAPPING ----
COLOR_MAP = {
    (255, 0, 0): 1,   # red → class 1 (e.g., adenoma)
    (0, 255, 0): 2,   # green → class 2 (e.g., hyperplastic)
}

def mask_to_class(mask: np.ndarray) -> np.ndarray:
    """Convert color mask to class-indexed mask (0=background, 1=class1, 2=class2)."""
    h, w, _ = mask.shape
    label = np.zeros((h, w), dtype=np.uint8)
    mask_int = mask.astype(np.int32) # Use a temporary int array for safe arithmetic

    # Red detection
    red_mask = (mask_int[:, :, 0] > 150) & (mask_int[:, :, 1] < 100) & (mask_int[:, :, 2] < 100)
    label[red_mask] = 1

    # Green or white detection
    green_mask = (mask_int[:, :, 1] > 150) & (mask_int[:, :, 0] < 100) & (mask_int[:, :, 2] < 100)
    white_condition = (mask_int[:, :, 0] > 220) & (mask_int[:, :, 1] > 220) & (mask_int[:, :, 2] > 220)
    label[green_mask | white_condition] = 2

    return label

def find_mask_path(image_name: str, mask_dir: Path) -> Optional[Path]:
    """Find the corresponding mask for an image using pathlib."""
    base_name = Path(image_name).stem
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
        path = mask_dir / (base_name + ext)
        if path.exists():
            return path
    return None

def main():
    # 2. SETUP ARGUMENT PARSING (For command-line configuration)
    # This makes the script highly reusable without needing to edit the code.
    parser = argparse.ArgumentParser(description="Prepare a multi-class segmentation dataset.")

    # Define project root to provide sensible defaults for paths
    project_root = Path(__file__).resolve().parents[3] # Assumes script is 3 levels deep
    dataset_root = project_root / "Dataset" / "Dataset_2_0"

    parser.add_argument('--image_dir', type=str, default=str(dataset_root / "images"), help="Directory containing images with masks.")
    parser.add_argument('--mask_dir', type=str, default=str(dataset_root / "masks"), help="Directory containing corresponding color masks.")
    parser.add_argument('--negative_dir', type=str, default=str(dataset_root / "negative_samples"), help="Directory for images with no masks.")
    parser.add_argument('--output_dir', type=str, default=str(dataset_root / "train_npz_new"), help="Directory to save the processed .npz files.")
    parser.add_argument('--list_dir', type=str, default=str(dataset_root / "list_Synapse"), help="Directory to save the train/valid/test list files.")
    parser.add_argument('--img_size', type=int, default=256, help="The edge size to which images and masks will be resized (e.g., 256 -> 256x256).")
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of the data to be used for validation.")
    parser.add_argument('--test_split', type=float, default=0.1, help="Fraction of the data to be used for testing.")
    args = parser.parse_args()

    # 3. USE PATHLIB FOR FILE SYSTEM OPERATIONS
    # Convert string paths from argparse into modern Path objects.
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    negative_dir = Path(args.negative_dir)
    output_dir = Path(args.output_dir)
    list_dir = Path(args.list_dir)
    img_size_tuple = (args.img_size, args.img_size)

    logger.info("📂 Starting dataset preparation with the following configuration:")
    logger.info(f"   Images Path          : {image_dir}")
    logger.info(f"   Masks Path           : {mask_dir}")
    logger.info(f"   Negative Samples Path: {negative_dir}")
    logger.info(f"   Output NPZ Path      : {output_dir}")
    logger.info(f"   Output Lists Path    : {list_dir}")
    logger.info(f"   Image Size           : {img_size_tuple}")

    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Collect all image files ----
    all_image_paths: List[Path] = []
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tif']
    all_image_paths.extend(p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in valid_extensions)
    all_image_paths.extend(p for p in sorted(negative_dir.iterdir()) if p.suffix.lower() in valid_extensions)
    logger.info(f"🖼️  Found {len(all_image_paths)} total images from all sources.")

    # ---- 2. Split into training, validation, and test sets ----
    train_paths, val_test_paths = train_test_split(all_image_paths, test_size=args.val_split + args.test_split, random_state=42, shuffle=True)
    val_paths, test_paths = train_test_split(val_test_paths, test_size=args.test_split / (args.val_split + args.test_split), random_state=42, shuffle=True)
    splits = {"train": train_paths, "valid": val_paths, "test": test_paths}

    # ---- 3. Process and save each split ----
    for split_name, path_list in splits.items():
        logger.info(f"🚀 Processing '{split_name}' set ({len(path_list)} samples)...")
        for img_path in path_list:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Could not read image {img_path.name}, skipping.")
                continue

            mask_path = find_mask_path(img_path.name, mask_dir)
            if mask_path:
                mask = cv2.imread(str(mask_path))
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                label = mask_to_class(mask_rgb)
                if not np.any(label > 0):
                    logger.warning(f"Mask found for '{img_path.name}' but no valid class pixels were detected.")
                    label = np.zeros_like(image, dtype=np.uint8)
            else:
                label = np.zeros_like(image, dtype=np.uint8)

            image_resized = cv2.resize(image, img_size_tuple, interpolation=cv2.INTER_AREA)
            label_resized = cv2.resize(label, img_size_tuple, interpolation=cv2.INTER_NEAREST)
            image_normalized = image_resized.astype(np.float32) / 255.0

            save_path = output_dir / f"{img_path.stem}.npz"
            np.savez_compressed(save_path, image=image_normalized, label=label_resized)

        list_path = list_dir / f"{split_name}.txt"
        with list_path.open("w") as f:
            for img_path in path_list:
                f.write(img_path.stem + "\n")
        logger.info(f"✅ Saved list file: {list_path}")

    logger.info("🎯 Dataset preparation complete!")

# 4. MAIN EXECUTION BLOCK
# This standard Python construct ensures the main() function is called only when
# the script is executed directly.
if __name__ == "__main__":
    main()