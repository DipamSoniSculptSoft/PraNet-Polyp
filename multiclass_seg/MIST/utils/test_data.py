#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from PIL import Image

# --- ONNX Requirement ---
import onnxruntime

# --- Local Imports ---
from dataset_synapse import Synapse_dataset
from dotenv import load_dotenv

# --- Clear GPU Cache ---
import gc
gc.collect()
torch.cuda.empty_cache()

# --- Load Environment Variables ---
print("Loading environment variables from .env file...")
load_dotenv()
NPZ_DATA_PATH = os.getenv('dataset_npz_path')
LIST_DIR_PATH = os.getenv('dataset_synapse_path')
print("Paths loaded successfully.")

# =====================================================================================
# SCRIPT ARGUMENTS & CONFIGURATION
# =====================================================================================
parser = argparse.ArgumentParser(description="ONNX Model Testing Script")

parser.add_argument('--volume_path', type=str, default=NPZ_DATA_PATH)
parser.add_argument('--list_dir', type=str, default=LIST_DIR_PATH)
parser.add_argument('--dataset', type=str, default='Synapse')
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=2222)
parser.add_argument('--deterministic', type=int,  default=1)
parser.add_argument('--is_savefig', default=True, action="store_true")
parser.add_argument('--test_save_dir', type=str, default='onnx_test_predictions')

args, unknown = parser.parse_known_args()


# =====================================================================================
# METRIC & HELPER FUNCTIONS
# =====================================================================================

def softmax(x, axis=1):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def calculate_metrics_np(pred_mask, gt_mask, num_classes):
    """Calculates Dice and Jaccard for each class using NumPy."""
    metrics = []
    
    for i in range(1, num_classes):
        pred_class = (pred_mask == i).astype(np.float32)
        gt_class = (gt_mask == i).astype(np.float32)
        
        intersection = np.sum(pred_class * gt_class)
        pred_sum = np.sum(pred_class)
        gt_sum = np.sum(gt_class)
        
        dice = (2. * intersection + 1e-8) / (pred_sum + gt_sum + 1e-8)
        union = pred_sum + gt_sum - intersection
        jaccard = (intersection + 1e-8) / (union + 1e-8)
        
        metrics.append([dice, jaccard])
        
    return metrics

# =====================================================================================
# ONNX INFERENCE FUNCTION
# =====================================================================================

def inference_onnx(args, session, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    
    logging.info(f"{len(testloader)} test iterations per epoch")
    print(f"Starting ONNX inference on {len(testloader)} test images...")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    all_metrics = []
    classes = ['Adenoma', 'Hyperplastic']

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_tensor, label_tensor, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        input_np = image_tensor.cpu().numpy()
        label_np = label_tensor.squeeze().cpu().numpy()

        # ============================================================================
        # *** ROBUST SHAPE CORRECTION IS HERE ***
        # This block guarantees the input array has the correct shape for the model.
        # ============================================================================
        # STEP 1: Ensure the input is 4D (add batch dimension if missing)
        if input_np.ndim == 3:
            # Shape was (C, H, W), add batch dimension to make it (1, C, H, W)
            input_np = np.expand_dims(input_np, axis=0)

        # STEP 2: Ensure the input has 3 channels (repeat grayscale if needed)
        # This runs after we are sure the input is 4D.
        if input_np.shape[1] == 1:
            # Shape was (1, 1, H, W), repeat channel to make it (1, 3, H, W)
            input_np = np.repeat(input_np, 3, axis=1)
        # ============================================================================

        # Run ONNX Inference
        raw_output = session.run([output_name], {input_name: input_np})[0]

        probabilities = softmax(raw_output, axis=1)
        pred_mask = np.argmax(probabilities, axis=1).squeeze()

        metric_i = calculate_metrics_np(pred_mask, label_np, args.num_classes)
        all_metrics.append(metric_i)

        dice_scores = [f"{m[0]:.4f}" for m in metric_i]
        jaccard_scores = [f"{m[1]:.4f}" for m in metric_i]
        logging.info(f"idx {i_batch} case {case_name} -> Dice: {dice_scores}, Jaccard: {jaccard_scores}")

        if test_save_path:
            color_mask = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
            color_mask[pred_mask == 1] = [255, 255, 0] # Yellow
            color_mask[pred_mask == 2] = [255, 0, 255] # Magenta
            
            img = Image.fromarray(color_mask)
            img.save(os.path.join(test_save_path, f"{case_name}_pred.png"))

    avg_metrics = np.mean(all_metrics, axis=0)
    
    for i in range(args.num_classes - 1):
        logging.info(f"--- Mean metrics for Class {i+1} ({classes[i]}) ---")
        logging.info(f"  - Mean Dice: {avg_metrics[i][0]:.4f}")
        logging.info(f"  - Mean Jaccard (IoU): {avg_metrics[i][1]:.4f}")

    overall_dice = np.mean([item[0] for item in avg_metrics])
    overall_jaccard = np.mean([item[1] for item in avg_metrics])
    
    logging.info('--- Overall ONNX Test Performance ---')
    logging.info(f'Mean Dice Score: {overall_dice:.4f}')
    logging.info(f'Mean Jaccard (IoU): {overall_jaccard:.4f}')
    
    print("\n--- Overall ONNX Test Performance ---")
    print(f"Mean Dice Score: {overall_dice:.4f}")
    print(f"Mean Jaccard (IoU): {overall_jaccard:.4f}")
    
    return "Testing Finished!"

# =====================================================================================
# MAIN EXECUTION BLOCK
# =====================================================================================

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    onnx_model_path = r"C:\Users\jaydu\OneDrive\Desktop\python_projects\polpy classification\PraNet-V2\models\Synapse\run_2025-10-13_15_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_bs4_lr1e-05_256_s2222\mist_cam_polyp_rgb.onnx"
    
    if not os.path.exists(onnx_model_path):
        print(f"[ERROR] ONNX model file not found at: {onnx_model_path}")
        sys.exit(1)

    print(f"Loading ONNX model from: {onnx_model_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    print(f"ONNX session created using: {ort_session.get_providers()[0]}")

    log_folder = 'test_log'
    os.makedirs(log_folder, exist_ok=True)
    model_name = os.path.basename(onnx_model_path).replace('.onnx', '')
    log_filename = f"onnx_test_log_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    logging.basicConfig(filename=os.path.join(log_folder, log_filename), level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Testing ONNX model: {model_name}")
    
    if args.is_savefig:
        test_save_path = os.path.join(args.test_save_dir, "onnx_test_vol_predictions")
        os.makedirs(test_save_path, exist_ok=True)
        print(f"Predicted masks will be saved to: {os.path.abspath(test_save_path)}")
    else:
        test_save_path = None
        
    inference_onnx(args, ort_session, test_save_path)