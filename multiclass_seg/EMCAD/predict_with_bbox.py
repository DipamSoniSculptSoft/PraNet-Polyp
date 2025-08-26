# Save this file as `predict_with_boxes.py` inside the `multiclass_seg/EMCAD/` directory.

import os
import sys
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np

# Local imports from the EMCAD directory
from lib.networks import EMCADNet

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================

# --- OS-Agnostic Path Construction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# 1. Paths
MODEL_PATH = os.path.join(project_root, "trained_models", "custom_binary_session", "epoch_15.pth") # <-- IMPORTANT: Point to your best model
IMAGE_DIR = os.path.join(project_root, "predict", "input_images")
OUTPUT_DIR = os.path.join(project_root, "predict", "output_masks")

# 2. Model Parameters (MUST match training)
NUM_CLASSES = 2
ENCODER_NAME = 'pvt_v2_b2'
IMG_SIZE = 512

# 3. Bounding Box Style
BOX_COLOR = "red"
BOX_THICKNESS = 3
# =====================================================================================


def predict_with_boxes():
    """
    Main function to run inference and draw bounding boxes on the original images.
    """
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}")
    model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER_NAME, pretrain=False, dual=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Define Image Transformations for the model
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Find all images to process
    image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_filenames)} images to process.")

    # 5. Loop through images and predict
    with torch.no_grad():
        for filename in tqdm(image_filenames, desc="Predicting and drawing boxes"):
            img_path = os.path.join(IMAGE_DIR, filename)
            
            # --- Step A: Load original image and prepare for model ---
            original_image = Image.open(img_path).convert("RGB")
            original_size = original_image.size # (width, height)
            input_tensor = transform(original_image).unsqueeze(0).to(device)

            # --- Step B: Get model prediction ---
            predictions = model(input_tensor)
            logits = predictions[3] # Use the final, highest-resolution output
            
            # --- Step C: Create a binary mask and resize it to original image size ---
            # Get class index with the highest probability for each pixel (0=background, 1=polyp)
            pred_mask_tensor = torch.argmax(logits, dim=1).squeeze(0).cpu()
            # Convert to a PIL Image for easy resizing
            mask_image_small = Image.fromarray(pred_mask_tensor.numpy().astype(np.uint8))
            # Resize mask back to the original image's dimensions using NEAREST interpolation
            mask_image_full = mask_image_small.resize(original_size, Image.NEAREST)
            mask_np_full = np.array(mask_image_full)

            # --- Step D: Find bounding box from the full-size mask ---
            # Find all coordinates where the mask is 1 (the polyp)
            # np.where returns (row_indices, column_indices) which correspond to (y, x)
            y_coords, x_coords = np.where(mask_np_full == 1)

            # --- Step E: Draw the bounding box on the original image ---
            if len(x_coords) > 0: # Check if any polyp was detected
                # Calculate the corners of the box
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                # Create a drawable version of the original image
                draw = ImageDraw.Draw(original_image)
                # Draw the rectangle
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    outline=BOX_COLOR,
                    width=BOX_THICKNESS
                )

            # --- Step F: Save the final image ---
            # The image will be saved with a bounding box if a polyp was found,
            # otherwise, the original image will be saved.
            output_path = os.path.join(OUTPUT_DIR, filename)
            original_image.save(output_path)

    print(f"\nPrediction complete. All images with bounding boxes saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    predict_with_boxes()