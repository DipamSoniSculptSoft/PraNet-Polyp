# Save this file as `predict.py` inside the `multiclass_seg/EMCAD/` directory.

import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add the parent directory to the path to find the model libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EMCAD.lib.networks import EMCADNet

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================

# --- OS-Agnostic Path Construction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# 1. Paths
MODEL_PATH = os.path.join(project_root, "trained_models", "custom_binary_session", "epoch_15.pth") # <-- IMPORTANT: Point this to your best saved model
IMAGE_DIR = os.path.join(project_root, "predict", "input_images")
OUTPUT_DIR = os.path.join(project_root, "predict", "output_masks")

# 2. Model Parameters (MUST match the parameters used for training)
NUM_CLASSES = 2
ENCODER_NAME = 'pvt_v2_b2'
IMG_SIZE = 512
# =====================================================================================


def predict():
    """
    Main function to run inference on a folder of images.
    """
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}")
    model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER_NAME, pretrain=False, dual=False) # pretrain=False as we are loading custom weights
    
    # Load the state dictionary. `map_location` allows loading a GPU-trained model on a CPU.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # IMPORTANT: Set the model to evaluation mode

    # 3. Define Image Transformations (must be the same as training)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Find all images in the input directory
    image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_filenames)} images to process.")

    # 5. Loop through images and predict
    with torch.no_grad(): # Disable gradient calculations for inference
        for filename in tqdm(image_filenames, desc="Predicting masks"):
            img_path = os.path.join(IMAGE_DIR, filename)
            
            # Open and transform the image
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

            # Get model prediction
            predictions = model(input_tensor)
            logits = predictions[3] # Use the final, highest-resolution output

            # Post-process the output to create a segmentation mask
            # Get the class index with the highest probability for each pixel
            pred_mask = torch.argmax(logits, dim=1).squeeze(0) # Remove batch and channel dims
            
            # Convert to a NumPy array
            mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            
            # For visualization, map the polyp class (1) to white (255)
            # The background class (0) will remain black (0)
            mask_np[mask_np == 1] = 255
            
            # Convert NumPy array to a PIL Image
            mask_image = Image.fromarray(mask_np)
            
            # Save the resulting mask
            output_path = os.path.join(OUTPUT_DIR, filename)
            mask_image.save(output_path)

    print(f"\nPrediction complete. All masks saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # The fix to networks.py is no longer needed if we set pretrain=False,
    # but it's good practice to have it fixed anyway.
    predict()