
# =====================================================================================
# REAL-TIME MULTI-CLASS POLYP DETECTION INFERENCE SCRIPT
#
# DESCRIPTION:
# This script performs real-time inference using a pre-trained and exported ONNX
# model for multi-class polyp segmentation. It captures video from a file or a
# webcam, preprocesses each frame, runs it through the ONNX model, and then
# post-processes the output to draw bounding boxes around detected polyps.
#
# KEY HIGHLIGHTS:
#   - Real-Time Performance: The model demonstrates efficient performance, with
#     an end-to-end latency typically in the range of 40-57 ms on appropriate
#     hardware (including preprocessing, inference, and post-processing).
#   - Multi-Class Detection: The script is configured to distinguish between
#     different classes of polyps (e.g., "Adenoma" and "Hyperplastic").
#   - Optimized Preprocessing: Uses an optimized function (`preprocess_rgb_optimized`)
#     to prepare video frames for the model, ensuring high throughput.
#   - Hardware Acceleration: It is configured to prioritize GPU (CUDA) for
#     inference via ONNX Runtime, falling back to CPU if a GPU is not available.
#
# HOW TO USE (FOR A NEW USER FETCHING FROM GITHUB):
#   1. Install Dependencies: Ensure you have the required Python libraries installed:
#      pip install opencv-python numpy onnxruntime-gpu  (or onnxruntime for CPU)
#
#   2.  Update Model Path: In the "CONFIGURATION" section below, you MUST change
#       the `PYTORCH_MODEL_FOLDER` variable to the absolute path of the directory
#       containing your `mist_cam_polyp_rgb.onnx` model file.
#
#   3.  Update Video Source: Change the `VIDEO_SOURCE` variable.
#       - To use a webcam, set it to 0 for the primary camera, 1 for the next, etc.
#       - To use a video file, provide the full path to your video
#         (e.g., "C:/Users/YourUser/Videos/test_video.mp4").
#
#   4.  Run the Script: Execute the script from your terminal:
#       python your_script_name.py
#
# =====================================================================================

import os
import cv2
import numpy as np
import onnxruntime
import time

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================

# --- (USER ACTION REQUIRED) ---
# 1. Path to your exported RGB ONNX model folder.
#    This path MUST be updated to point to the folder on your local machine where
#    the 'mist_cam_polyp_rgb.onnx' file is located.
PYTORCH_MODEL_FOLDER = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\models\Synapse\run_2025-10-14_18_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo50_bs4_lr1e-05_256_s2222"
ONNX_MODEL_PATH = os.path.join(PYTORCH_MODEL_FOLDER, "mist_cam_polyp_rgb.onnx")

# --- (USER ACTION REQUIRED) ---
# 2. Video Source
#    Change this to your video file path or a camera index (e.g., 0 for webcam).
VIDEO_SOURCE = r"C:\Users\jaydu\Downloads\8.4.25.1410_hyperplastic.mp4"


# 3. Model Input Size (must match the trained model)
IMG_SIZE = 256

# 4. Visualization Settings
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
# =====================================================================================

def preprocess_rgb_optimized(frame, img_size):
    """
    An optimized version of the preprocessing function that uses a faster resizing
    algorithm and more efficient normalization math.
    """
    # 1. Convert frame from BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Resize using a faster interpolation method.
    #    cv2.INTER_LINEAR is much faster than cv2.INTER_AREA and the quality
    #    difference is negligible for neural network inputs.
    resized_frame = cv2.resize(rgb_frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # 3. Convert to float32. This is done once at the beginning.
    input_tensor = resized_frame.astype(np.float32)

    # 4. Combine normalization to [0, 1] and standardization into one step.
    #    This is mathematically equivalent to (frame / 255.0 - MEAN) / STD
    #    but can be slightly more efficient by reducing the number of separate
    #    full-array operations.
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = ((input_tensor / 255.0) - MEAN) / STD

    # 5. Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    input_tensor = input_tensor.transpose(2, 0, 1)

    # 6. Add a batch dimension to create the final shape: [1, 3, H, W]
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

def main():
    print("--- Live Multi-Class Polyp Detection ---")

    # Check if the ONNX model file exists
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"\n[ERROR] ONNX model not found at: {ONNX_MODEL_PATH}")
        print("Please update the 'PYTORCH_MODEL_FOLDER' variable in the script's CONFIGURATION section.")
        return

    # Load the ONNX runtime session, preferring the GPU (CUDA) over CPU
    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    print(f"ONNX session created using: {session.get_providers()[0]}")

    # Get the names of the input and output layers
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model loaded. Input: '{input_name}', Output: '{output_name}'")

    # Setup video capture
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

        # 1. Preprocess the frame for the model
        prest = time.time()
        input_tensor = preprocess_rgb_optimized(frame, IMG_SIZE)
        print(f"Preprocessing time: {(time.time() - prest) * 1000:.1f} ms")

        # 2. Run inference
        sessst = time.time()
        result = session.run([output_name], {input_name: input_tensor})
        print(f"Session (Inference) time: {(time.time() - sessst) * 1000:.1f} ms")

        # 3. Post-process the output
        posst = time.time()
        pred_mask = np.argmax(result[0], axis=1).squeeze()
        resized_mask = cv2.resize(pred_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Draw bounding boxes for each detected class
        for class_id, color in CLASS_COLORS.items():
            class_mask = (resized_mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # Filter small detections
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)
                    cv2.putText(frame, CLASS_NAMES[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, BOX_THICKNESS)
        print(f"Postprocessing time: {(time.time() - posst) * 1000:.1f} ms")

        # Display latency information
        if SHOW_LATENCY:
            overall_latency = (time.time() - overall_start_time) * 1000
            latency_text = f"Latency: {overall_latency:.1f} ms"
            print(f"--- Total Latency for frame: {overall_latency:.1f} ms ---")
            (w, h), _ = cv2.getTextSize(latency_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BOX_THICKNESS)
            cv2.rectangle(frame, (original_width - w - 20, 10), (original_width - 10, 10 + h + 10), (0, 0, 0), -1)
            cv2.putText(frame, latency_text, (original_width - w - 15, 10 + h), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), BOX_THICKNESS)

        # Resize frame for better display
        display_width = 960
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