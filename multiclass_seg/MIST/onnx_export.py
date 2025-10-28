# =====================================================================================
# MODEL EXPORT SCRIPT: PyTorch MIST_CAM to ONNX
#
# DESCRIPTION:
# This script is designed to export a pre-trained PyTorch MIST_CAM model to the 
# ONNX (Open Neural Network Exchange) format. The MIST_CAM model is intended for 
# multi-class segmentation tasks.
#
# The script performs the following key functions:
#   1.  Loads a trained MIST_CAM model architecture.
#   2.  Loads the saved weights from a specified PyTorch model file (.pth).
#   3.  Wraps the model to produce a single, final output tensor containing
#       softmax probabilities, which is more convenient for inference.
#   4.  Exports the wrapped model to an ONNX file, which can then be used for
#       cross-platform inference in various environments (e.g., C++, C#, Java).
#
# HOW TO USE:
#   1.  Ensure you have all the required libraries installed (torch, onnx).
#   2.  Update the configuration variables in the "CONFIGURATION" section below.
#       Most importantly, you MUST change 'PYTORCH_MODEL_PATH' to point to your
#       own trained model file.
#   3.  Run the script from your terminal: python your_script_name.py
#
# =====================================================================================

import os
import torch
import torch.onnx
import sys

# Add local libraries to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from lib.networks import MIST_CAM
# =====================================================================================
# CONFIGURATION
# =====================================================================================
# --- (USER ACTION REQUIRED) ---
# The following paths and parameters will likely need to be changed when you
# run this on a new machine or with a different model.

# 1. project_root: This path is determined relative to the script's location.
#    If your "models" directory is not two levels above this script, you may
#    need to adjust this path accordingly.
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# 2. PYTORCH_MODEL_PATH: This is the MOST IMPORTANT path to change.
#    You MUST update this to the absolute path of your trained .pth model file.
#    Example: "/home/user/my_project/models/my_model_epoch_final.pth"
PYTORCH_MODEL_PATH = os.path.join(project_root, 
                                  "models", 
                                  "Synapse", 
                                  "run_2025-10-14_18_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo50_bs4_lr1e-05_256_s2222",
                                  "epoch_49.pth")

# 3. ONNX_MODEL_PATH: This is the destination path for the output ONNX model.
#    By default, it saves the ONNX file in the same directory as the PyTorch model.
#    You can change this to any location you prefer.
ONNX_MODEL_PATH = os.path.join(os.path.dirname(PYTORCH_MODEL_PATH), "mist_cam_polyp_rgb.onnx")

# 4. Model-specific parameters: These values must match the parameters used
#    during the training of your model.
NUM_CLASSES = 3
IMG_SIZE = 256
DUAL_SUPERVISION = True
ONNX_OPSET_VERSION = 14
# =====================================================================================


class ModelWrapper(torch.nn.Module):
    """
    A wrapper class to process the multiple outputs of the MIST_CAM model
    and return a single final tensor of probabilities. This simplifies inference.
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.dual = model.dual

    def forward(self, x):
        all_predictions = self.model(x)
        if self.dual:
            p_foreground = all_predictions[:4]
            outputs = 0.0
            for p in p_foreground:
                outputs += p
        else:
            outputs = 0.0
            for p in all_predictions:
                outputs += p
        final_output = torch.softmax(outputs, dim=1)
        return final_output


def export_model_to_onnx():
    """
    Main function to handle the model loading, wrapping, and ONNX export process.
    """
    print("--- Starting ONNX Export Process for MIST_CAM (RGB) ---")

    # --- (USER ACTION RECOMMENDED) ---
    # This check contains a hardcoded path specific to the original developer's machine.
    # It is recommended to either REMOVE this check or update the path if you wish
    # to use a similar check for your own environment.
    if not os.path.exists(PYTORCH_MODEL_PATH) or ONNX_MODEL_PATH in PYTORCH_MODEL_PATH:
        print(f"\n[ERROR] PyTorch model not found. Please update the 'PYTORCH_MODEL_PATH' variable in the script.")
        return

    print("Initializing MIST_CAM model architecture...")
    # Initialize the model with the parameters defined in the configuration section.
    model = MIST_CAM(
        n_class=NUM_CLASSES, 
        img_size_s1=(IMG_SIZE, IMG_SIZE), 
        img_size_s2=(224, 224), 
        model_scale='small', 
        decoder_aggregation='additive', 
        interpolation='bilinear',
        dual=DUAL_SUPERVISION
    )

    print(f"Loading trained weights from: {PYTORCH_MODEL_PATH}")
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
    model.eval()

    print("Wrapping model for a single inference output...")
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    # The dummy input tensor should match the expected input shape of the model.
    # The 3 channels represent an RGB image.
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print(f"Created a dummy RGB input tensor of shape: {dummy_input.shape}")

    print(f"Exporting model to ONNX (opset {ONNX_OPSET_VERSION}) at: {ONNX_MODEL_PATH}")
    # Export the model to the ONNX format.
    torch.onnx.export(wrapped_model,
                      dummy_input,
                      ONNX_MODEL_PATH,
                      export_params=True,
                      opset_version=ONNX_OPSET_VERSION,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output_probabilities'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output_probabilities': {0: 'batch_size'}}
    )
    
    print("\n✅ ONNX model exported successfully!")
    print(f"   Model saved to: {ONNX_MODEL_PATH}")
    print(f"   Input node name: 'input', Shape: [batch_size, 3, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"   Output node name: 'output_probabilities', Shape: [batch_size, {NUM_CLASSES}, {IMG_SIZE}, {IMG_SIZE}]")


if __name__ == '__main__':
    export_model_to_onnx()