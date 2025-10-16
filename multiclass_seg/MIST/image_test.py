import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================

# 1. Path to the image you want to test
# --- IMPORTANT: UPDATE THIS PATH TO YOUR TEST IMAGE ---
IMAGE_PATH = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\Dataset\classification_data\images\0bd55b1393e2ef89424de1556a26c8eb.jpeg"
# 2. Path to your exported RGB ONNX model
# This should point to the ONNX file we tried to create before.
ONNX_MODEL_PATH = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\models\Synapse\run_2025-10-09_14_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo20_bs4_lr1e-05_256_s2222\mist_cam_polyp_rgb.onnx"

# 3. Path to save the output images
# The script will create this folder if it doesn't exist.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_output")

# 4. Model Input Size (must match the trained model)
IMG_SIZE = 256

# 5. Visualization Settings (BGR color codes)
# We add a color for the background to make the mask image clearer
CLASS_COLORS = {
    0: (0, 0, 0),          # Class 0 (Background) -> Black
    1: (0, 255, 0),      # Class 1 (Adenoma) -> Green
    2: (0, 0, 255),      # Class 2 (Hyperplastic) -> Red
}
# =====================================================================================


def preprocess_rgb(image_pil, img_size):
    """ Prepares a PIL image for the 3-channel RGB ONNX model. """
    # Resize and convert to numpy array
    resized_image = image_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
    image_np = np.array(resized_image, dtype=np.float32)

    # Normalize pixel values to the [0, 1] range
    normalized_frame = image_np / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_frame = (normalized_frame - mean) / std
    
    # Transpose channels from HWC to CHW
    transposed_frame = normalized_frame.transpose(2, 0, 1)
    
    # Add a batch dimension: [1, 3, H, W]
    input_tensor = np.expand_dims(transposed_frame, axis=0)
    
    return input_tensor


def create_color_mask(pred_mask, class_colors):
    """ Creates a colored segmentation mask from a class index mask. """
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[pred_mask == class_id] = color
    return color_mask


def main():
    print("--- Running Inference on a Single Image ---")
    
    # --- 1. Validation ---
    if not os.path.exists(IMAGE_PATH):
        print(f"\n[ERROR] Test image not found at: {IMAGE_PATH}")
        print("Please update the 'IMAGE_PATH' variable in the script.")
        return
        
    if not os.path.exists(ONNX_MODEL_PATH) or "YOUR_RUN_FOLDER_HERE" in ONNX_MODEL_PATH:
        print(f"\n[ERROR] ONNX model not found at: {ONNX_MODEL_PATH}")
        print("Please update the 'ONNX_MODEL_PATH' variable.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # --- 2. Load Model ---
    print(f"Loading ONNX model...")
    # NOTE: This will still show the GPU error and fall back to CPU if not fixed,
    # but it will work if the ONNX model is compatible with the CPU provider.
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model loaded to run on: {session.get_providers()[0]}")

    # --- 3. Load and Preprocess Image ---
    print(f"Loading and preprocessing image: {os.path.basename(IMAGE_PATH)}")
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor = preprocess_rgb(original_image, IMG_SIZE)

    # --- 4. Run Inference ---
    print("Running inference...")
    result = session.run([output_name], {input_name: input_tensor})

    # --- 5. Post-process and Save Results ---
    print("Post-processing results...")
    
    # Get the class predictions by finding the max probability
    pred_mask = np.argmax(result[0], axis=1).squeeze() # Shape: [IMG_SIZE, IMG_SIZE]
    
    # Create a color mask from the prediction
    color_mask_small = create_color_mask(pred_mask, CLASS_COLORS)
    
    # Resize the color mask and the original image for saving
    original_w, original_h = original_image.size
    color_mask_resized = cv2.resize(color_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Convert original PIL image to OpenCV format for overlay
    original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    # Create a blended overlay image
    alpha = 0.5 # Transparency of the mask
    overlay_image = cv2.addWeighted(original_image_cv, 1, color_mask_resized, alpha, 0)

    # Save the output files
    base_filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_original.png"), original_image_cv)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_mask.png"), color_mask_resized)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_overlay.png"), overlay_image)
    
    print("\n✅ Inference complete. The following files have been saved:")
    print(f"  - {base_filename}_original.png")
    print(f"  - {base_filename}_mask.png (Green=Adenoma, Red=Hyperplastic)")
    print(f"  - {base_filename}_overlay.png")


if __name__ == "__main__":
    main()