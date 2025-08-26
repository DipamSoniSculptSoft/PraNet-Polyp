# # Save this file as `live_predict_onnx.py` inside the `multiclass_seg/EMCAD/` directory.

# import os
# import cv2
# import numpy as np
# import onnxruntime
# from PIL import Image

# # =====================================================================================
# # CONFIGURATION - EDIT THIS SECTION
# # =====================================================================================
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# ONNX_MODEL_PATH = os.path.join(project_root, "trained_models", "custom_binary_session", "emcad_polyp_model.onnx")
# VIDEO_SOURCE = 2 # 0 for webcam, or path to a video file
# IMG_SIZE = 512
# BOX_COLOR = (0, 0, 255) # BGR: Red
# BOX_THICKNESS = 4
# # =====================================================================================


# def preprocess(image_pil, img_size):
#     """ Prepares a PIL image for the ONNX model. """
#     resized_img = image_pil.resize((img_size, img_size))
#     img_np = np.array(resized_img, dtype=np.float32) / 255.0
    
#     # --- CHANGE: Explicitly set the dtype to float32 to prevent mismatch error ---
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
#     normalized_img = (img_np - mean) / std
#     input_tensor = normalized_img.transpose(2, 0, 1)[np.newaxis, :, :, :]
#     return input_tensor

# def main():
#     # Load the ONNX runtime session. It will automatically use the CPU.
#     print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
#     session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
#     input_name = session.get_inputs()[0].name
    
#     # The original model returns a list of 4 outputs. We need to get the name of the one we want.
#     # In this case, it's the 4th output (index 3), which corresponds to p1.
#     output_names = [output.name for output in session.get_outputs()]
#     final_output_name = output_names[3] # This corresponds to 'p1' from the original model
    
#     print("ONNX model loaded successfully.")

#     # Setup video capture
#     print(f"Opening video source: {VIDEO_SOURCE}")
#     cap = cv2.VideoCapture(VIDEO_SOURCE)
#     if not cap.isOpened():
#         print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         original_height, original_width, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image_pil = Image.fromarray(frame_rgb)
#         input_tensor = preprocess(image_pil, IMG_SIZE)

#         # Run inference
#         result = session.run([final_output_name], {input_name: input_tensor})[0]

#         # Post-process the result
#         pred_mask = np.argmax(result, axis=1).squeeze()
#         resized_mask = cv2.resize(pred_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)

#         # Find contours and draw bounding box
#         contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             if cv2.contourArea(largest_contour) > 100:
#                 x, y, w, h = cv2.boundingRect(largest_contour)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
#                 cv2.putText(frame, 'Polyp', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)

#         cv2.imshow('Live Polyp Detection (Press Q to quit)', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import time
import torch
import torchvision.transforms as transforms

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

ONNX_MODEL_PATH = os.path.join(project_root, "trained_models", "custom_binary_session", "emcad_polyp_model.onnx")
VIDEO_SOURCE = 2 # 0 for webcam, or path to a video file
IMG_SIZE = 512
BOX_COLOR = (0, 0, 255) # BGR: Red
BOX_THICKNESS = 4
SHOW_LATENCY = True # Toggle to show/hide latency overlay
# =====================================================================================

def preprocess(image_pil, img_size):
    """ Prepares a PIL image for the ONNX model using PyTorch and GPU. """
    start_time = time.time()
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Converts to [C, H, W] and normalizes to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    input_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension: [1, C, H, W]
    
    # Move to GPU if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    
    # Convert to numpy for ONNX compatibility
    input_tensor = input_tensor.cpu().numpy()
    
    preprocess_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return input_tensor, preprocess_time

def main():
    # Load the ONNX runtime session. It will automatically use the CPU.
    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # The original model returns a list of 4 outputs. We need to get the name of the one we want.
    # In this case, it's the 4th output (index 3), which corresponds to p1.
    output_names = [output.name for output in session.get_outputs()]
    final_output_name = output_names[3] # This corresponds to 'p1' from the original model
    
    print("ONNX model loaded successfully.")

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
            break

        original_height, original_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        
        # Preprocess
        input_tensor, preprocess_time = preprocess(image_pil, IMG_SIZE)

        # Run inference
        inference_start_time = time.time()
        result = session.run([final_output_name], {input_name: input_tensor})[0]
        inference_time = (time.time() - inference_start_time) * 1000  # Convert to milliseconds

        # Post-process
        postprocess_start_time = time.time()
        pred_mask = np.argmax(result, axis=1).squeeze()
        resized_mask = cv2.resize(pred_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Find contours and draw bounding box
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
                cv2.putText(frame, 'Polyp', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)
        postprocess_time = (time.time() - postprocess_start_time) * 1000  # Convert to milliseconds

        # Calculate overall latency
        overall_latency = (time.time() - overall_start_time) * 1000  # Convert to milliseconds

        # Log timings
        print(f"Preprocess time: {preprocess_time:.2f} ms")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Postprocess time: {postprocess_time:.2f} ms")
        print(f"Overall latency: {overall_latency:.2f} ms")
        print("-" * 50)

        # Overlay latency in top right corner if enabled
        if SHOW_LATENCY:
            latency_text = f"Latency: {overall_latency:.2f} ms"
            text_size = cv2.getTextSize(latency_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = original_width - text_size[0] - 10  # 10 pixels from right edge
            text_y = 30  # 30 pixels from top
            cv2.putText(frame, latency_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Live Polyp Detection (Press Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()