
# =====================================================================================
# HIGHLY OPTIMIZED REAL-TIME MULTI-CLASS POLYP DETECTION (V3 - WITH TRACKING)
#
# DESCRIPTION:
# This script integrates a Centroid Tracker for temporal consistency. It builds on V2
# by adding a "memory" to the system. Detections are now tracked across frames.
#
# KEY HIGHLIGHTS & UPGRADES:
#   -   **Temporal Filtering**: A Centroid Tracker assigns a unique ID to each
#       detected object, tracking it across frames.
#   -   **False Positive Reduction**: An object is only displayed if it has been
#       consistently detected for a set number of frames (e.g., 5), effectively
#       eliminating fleeting detections like air bubbles.
#   -   **Classification Stability**: The displayed class for a tracked object is
#       determined by a majority vote over its last 10 classifications, preventing
#       the label from flickering between types.
#   -   **Smoothed Visuals**: Bounding circle size and confidence scores are
#       averaged over the last few frames, providing a more stable visualization.
#
# =====================================================================================

import os
import cv2
import numpy as np
import onnxruntime
import time
from pathlib import Path
from dotenv import load_dotenv
from tracking.Centroid import CentroidTracker

# Load .env file (if present)
load_dotenv()

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# MODEL_DIR = os.getenv(
#     "MODEL_DIR",
#     r"models/Synapse/run_2025-10-14_18_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo50_bs4_lr1e-05_256_s2222"
# )
# PYTORCH_MODEL_FOLDER = PROJECT_ROOT / MODEL_DIR
# ONNX_MODEL_PATH = PYTORCH_MODEL_FOLDER / "mist_cam_polyp_rgb.onnx"
ONNX_MODEL_PATH=r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\multiclass_seg\MERIT\model_pth\ACDC\run_2025-10-16_13-27_MERIT_Cascaded_Small_loss_MUTATION_w3_7_256_pretrain_epo10_bs4_lr0.0001_256_s2222\mist_cam_polyp_rgb.onnx"
VIDEO_SOURCE = 2 # e.g., 0 for webcam or "path/to/video.mp4"
IMG_SIZE = 256

# Visualization & Threshold Settings
CLASS_COLORS = {1: (0, 255, 255), 2: (255, 0, 255)}
CLASS_NAMES = {1: "Adenoma", 2: "Hyperplastic"}
BOX_THICKNESS = 2
FONT_SCALE = 0.7
SHOW_LATENCY = True
CONFIDENCE_THRESHOLD = 0.50
SHOW_CONFIDENCE = True

# =====================================================================================
def preprocess_rgb_optimized(frame, img_size):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    input_tensor = resized_frame.astype(np.float32)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_tensor = ((input_tensor / 255.0) - MEAN) / STD
    input_tensor = input_tensor.transpose(2, 0, 1)
    return np.expand_dims(input_tensor, axis=0)

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def main():
    print("--- Live Multi-Class Polyp Detection (V3 with Tracking) ---")
    # if not ONNX_MODEL_PATH.exists():
    #     print(f"\n[ERROR] ONNX model not found at: {ONNX_MODEL_PATH}")
    #     return

    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
    print(f"ONNX session created using: {session.get_providers()[0]}")
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name

    print(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
        return

    # --- NEW: Instantiate the Centroid Tracker ---
    ct = CentroidTracker(maxDisappeared=15, confirm_hits=5, history_size=10)

    while True:
        overall_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or camera error.")
            break

        original_height, original_width, _ = frame.shape
        input_tensor = preprocess_rgb_optimized(frame, IMG_SIZE)
        raw_output = session.run([output_name], {input_name: input_tensor})[0]

        probabilities = softmax(raw_output, axis=1)
        pred_mask = np.argmax(probabilities, axis=1).squeeze()
        confidence_map = np.max(probabilities, axis=1).squeeze()
        
        # --- MODIFIED: The post-processing logic is now split ---
        # Part 1: Collect all raw detections from the current frame.
        current_frame_detections = []
        detected_classes = np.unique(pred_mask)
        for class_id in detected_classes:
            if class_id == 0: continue
            
            class_mask = (pred_mask == class_id).astype(np.uint8)
            if np.any(class_mask):
                mean_confidence = np.mean(confidence_map[class_mask == 1])
            else:
                mean_confidence = 0

            if mean_confidence >= CONFIDENCE_THRESHOLD:
                kernel = np.ones((5, 5), np.uint8)
                clean_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if radius > 5:
                        # Collect detection data: (centroid, radius, class_id, confidence)
                        centroid = (int(x), int(y))
                        current_frame_detections.append((centroid, int(radius), class_id, mean_confidence))

        # Part 2: Update the tracker with the raw detections
        confirmed_objects = ct.update(current_frame_detections)

        # Part 3: Draw ONLY the confirmed and temporally smoothed objects
        scale_x = original_width / IMG_SIZE
        scale_y = original_height / IMG_SIZE
        
        for objectID, data in confirmed_objects.items():
            # Get smoothed data from the tracker
            centroid_256 = data['centroid']
            radius_256 = data['radius']
            voted_class_id = data['class_id']
            avg_confidence = data['confidence']

            # Scale to original frame size
            orig_center_x = int(centroid_256[0] * scale_x)
            orig_center_y = int(centroid_256[1] * scale_y)
            orig_radius = int(radius_256 * scale_x)

            # Get display info
            color = CLASS_COLORS.get(voted_class_id, (0, 255, 0))
            label = CLASS_NAMES.get(voted_class_id, f"Class {voted_class_id}")
            display_text = f"ID {objectID}: {label}: {avg_confidence:.2f}" if SHOW_CONFIDENCE else f"ID {objectID}: {label}"

            # Draw the circle and label
            cv2.circle(frame, (orig_center_x, orig_center_y), orig_radius, color, BOX_THICKNESS)
            text_y = orig_center_y - orig_radius - 10
            if text_y < 15: text_y = orig_center_y + orig_radius + 20
            cv2.putText(frame, display_text, (orig_center_x - orig_radius, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, BOX_THICKNESS)

        if SHOW_LATENCY:
            latency = (time.time() - overall_start_time) * 1000
            cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        display_width = 800
        display_height = int((display_width / frame.shape[1]) * frame.shape[0])
        display_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow('Live Multi-Class Polyp Detection (V3 with Tracking)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference finished.")

if __name__ == "__main__":
    main()