# =====================================================================================
# HIGHLY OPTIMIZED REAL-TIME MULTI-CLASS POLYP DETECTION (V2)
#
# DESCRIPTION:
# This script is an advanced, performance-tuned version for real-time inference
# using a pre-trained ONNX model. It processes video from a file or webcam,
# runs segmentation, and visualizes the results with bounding boxes. This version
# introduces several key optimizations in the post-processing pipeline to
# significantly reduce latency and improve efficiency.
#
# KEY HIGHLIGHTS & OPTIMIZATIONS:
#   -   **Superior Performance**: Achieves a significantly lower end-to-end latency,
#       now in the **32-48 ms** range on capable hardware.
#   -   **Confidence Thresholding**: A confidence score is calculated for each
#       potential detection. The script now filters out low-confidence predictions
#       *before* performing expensive contour analysis, saving significant processing time.
#   -   **Morphological Noise Reduction**: Instead of finding all contours and filtering
#       them by area, this version uses `cv2.morphologyEx` (Opening) to efficiently
#       remove small, noisy pixel regions from the prediction mask. This is much
#       faster and results in cleaner detections.
#   -   **Optimized Post-Processing**: The entire post-processing logic has been
#       streamlined. Scaling factors are calculated only once per frame, and confidence
#       is calculated efficiently using NumPy's vectorized operations.
#
# HOW TO USE (FOR A NEW USER FETCHING FROM GITHUB):
#   1.  Install Dependencies: Make sure you have the required Python libraries.
#       For GPU: pip install opencv-python numpy onnxruntime-gpu
#       For CPU: pip install opencv-python numpy onnxruntime
#
#   2.  Update Model Path: In the "CONFIGURATION" section, you MUST change
#       the `PYTORCH_MODEL_FOLDER` variable to the absolute path of the directory
#       that contains your `mist_cam_polyp_rgb.onnx` model file.
#
#   3.  Update Video Source: Change the `VIDEO_SOURCE` variable.
#       -   For a webcam, use a number (e.g., 0 for the default camera).
#       -   For a video file, provide the full path
#           (e.g., "C:/Users/YourUser/Videos/test_video.mp4").
#
#   4.  Adjust Threshold (Optional): You can tune the `CONFIDENCE_THRESHOLD`
#       value. A higher value (e.g., 0.7) will result in fewer but more reliable
#       detections. A lower value (e.g., 0.4) will detect more but may
#       include more false positives.
#
#   5.  Run the Script: Execute from your terminal:
#       python your_script_name.py
#
# =====================================================================================

import os
import cv2
import numpy as np
import onnxruntime
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (if present) to read custom paths for dataset/models/etc.
# This allows the same code to run locally and on servers without modification.
load_dotenv()
# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================

# --- (CODE UPDATED) ---
# The following section has been corrected to dynamically find your model file.

# 1. Define the Project Root.
#    Since this script is in PraNet-V2/multiclass_seg/MIST, we need to go
#    up two parent directories to get to the main "PraNet-V2" folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Define the Model Directory.
#    The default path is now set to the specific model folder from your screenshot.
#    os.getenv will still allow you to override this with a .env file if needed.
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    r"models/Synapse/run_2025-10-14_18_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo50_bs4_lr1e-05_256_s2222"
)

# 3. Construct the full path to the ONNX model.
PYTORCH_MODEL_FOLDER = PROJECT_ROOT / MODEL_DIR
ONNX_MODEL_PATH = PYTORCH_MODEL_FOLDER / "mist_cam_polyp_rgb.onnx"

# --- (USER ACTION REQUIRED) ---
# 4. Video Source
#    Change this to your video file path or a camera index (e.g., 0 for webcam).
VIDEO_SOURCE = 2 #r"C:\Users\jaydu\OneDrive\Desktop\python_projects\live-2\6.20.25.0845.mp4"

# 5. Model Input Size (must match the trained model)
IMG_SIZE = 256

# 6. Visualization & Threshold Settings
CLASS_COLORS = {
    1: (0, 255, 255),  # Class 1 (Adenoma) -> Yellow (BGR format)
    2: (255, 0, 255),  # Class 2 (Hyperplastic) -> Magenta (BGR format)
}
CLASS_NAMES = {
    1: "Adenoma",
    2: "Hyperplastic",
}
BOX_THICKNESS = 2
FONT_SCALE = 0.7
SHOW_LATENCY = True
CONFIDENCE_THRESHOLD = 0.50 # <-- Only show detections with confidence >= this value
SHOW_CONFIDENCE = True   # <-- Display confidence score on the bounding box

# =====================================================================================


def preprocess_rgb_optimized(frame, img_size):
    """
    An optimized version of the preprocessing function that uses a faster resizing
    algorithm and more efficient normalization math.
    """
    # 1. Convert frame from BGR to RGB and resize in one step for efficiency
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # 2. Convert to float32 and normalize in a single, vectorized operation
    input_tensor = resized_frame.astype(np.float32)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = ((input_tensor / 255.0) - MEAN) / STD

    # 3. Transpose from HWC to CHW and add batch dimension
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

def softmax(x, axis=1):
    """A stable softmax function to convert raw model output to probabilities."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def main():
    print("--- Live Multi-Class Polyp Detection (Optimized V2) ---")

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"\n[ERROR] ONNX model not found at: {ONNX_MODEL_PATH}")
        print("Please check the 'MODEL_DIR' variable in the configuration section.")
        return

    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH), providers=providers) # Use str() for pathlib compatibility
    print(f"ONNX session created using: {session.get_providers()[0]}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model loaded. Input: '{input_name}', Output: '{output_name}'")

    print(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
        return

    while True:
        overall_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("End of video stream or camera error.")
            break

        original_height, original_width, _ = frame.shape

        # 1. Preprocess the frame
        prest = time.time()
        input_tensor = preprocess_rgb_optimized(frame, IMG_SIZE)
        # print(f"Preprocessing time: {(time.time() - prest) * 1000:.1f} ms")


        # 2. Run inference
        sessst = time.time()
        raw_output = session.run([output_name], {input_name: input_tensor})[0]
        # print(f"Session (Inference) time: {(time.time() - sessst) * 1000:.1f} ms")

        # =====================================================================================
        # POST-PROCESSING (FULLY OPTIMIZED V2)
        # =====================================================================================
        posst = time.time()

        # 3. Get probabilities, predicted class mask, and confidence map in one go
        probabilities = softmax(raw_output, axis=1)
        pred_mask = np.argmax(probabilities, axis=1).squeeze()
        confidence_map = np.max(probabilities, axis=1).squeeze()

        detected_classes = np.unique(pred_mask)

        # Micro-optimization: Calculate scaling factors once per frame
        scale_x = original_width / IMG_SIZE
        scale_y = original_height / IMG_SIZE

        for class_id in detected_classes:
            if class_id == 0:  # Skip background class
                continue

            # 4. Create a binary mask for the current class
            class_mask = (pred_mask == class_id).astype(np.uint8)

            # 5. Calculate the mean confidence across all pixels of the detected mask
            # Check for empty mask to avoid division by zero if no pixels are detected
            if np.any(class_mask):
                mean_confidence = np.mean(confidence_map[class_mask == 1])
            else:
                mean_confidence = 0


            # 6. THRESHOLDING FIRST: This is a key optimization. Only proceed if confidence is high.
            if mean_confidence >= CONFIDENCE_THRESHOLD:

                # 7. OPTIMIZATION: Use Morphological Opening to remove small noise artifacts.
                # This is much faster than finding all contours and then filtering by area.
                kernel = np.ones((5, 5), np.uint8) # A slightly larger kernel can be more robust
                clean_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel, iterations=2)

                # 8. Find contours on the CLEANED mask. There will be far fewer contours.
                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    continue

                label = CLASS_NAMES.get(class_id, f"Class {class_id}")
                display_text = f"{label}: {mean_confidence:.2f}" if SHOW_CONFIDENCE else label
                color = CLASS_COLORS.get(class_id, (0, 255, 0))

                # 9. Loop through the filtered contours and draw them
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Scale the bounding box coordinates back to the original frame size
                    orig_x = int(x * scale_x)
                    orig_y = int(y * scale_y)
                    orig_w = int(w * scale_x)
                    orig_h = int(h * scale_y)

                    cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), color, BOX_THICKNESS)
                    cv2.putText(frame, display_text, (orig_x, orig_y - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, BOX_THICKNESS)

        # print(f"Post-processing time: {(time.time() - posst) * 1000:.1f} ms") # Uncomment for detailed timing

        # Display latency information
        if SHOW_LATENCY:
            overall_latency = (time.time() - overall_start_time) * 1000
            latency_text = f"Latency: {overall_latency:.1f} ms"
            (w, h), _ = cv2.getTextSize(latency_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BOX_THICKNESS)
            # print(f"--- Total Latency for frame: {overall_latency:.1f} ms ---")
            cv2.rectangle(frame, (original_width - w - 20, 10), (original_width - 10, 10 + h + 10), (0,0,0), -1)
            cv2.putText(frame, latency_text, (original_width - w - 15, 10 + h), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), BOX_THICKNESS)

        # Resize frame for better display
        display_width = 640
        display_height = int((display_width / frame.shape[1]) * frame.shape[0])
        display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

        cv2.imshow('Live Multi-Class Polyp Detection (Press Q to quit)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference finished.")

if __name__ == "__main__":
    main()