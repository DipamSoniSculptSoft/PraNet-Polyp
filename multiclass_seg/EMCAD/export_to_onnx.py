# Save this file as `export_to_onnx.py` inside the `multiclass_seg/EMCAD/` directory.

import os
import torch
import sys

# Add local libraries to path
from lib.networks import EMCADNet

# =====================================================================================
# CONFIGURATION - EDIT THIS SECTION
# =====================================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# 1. Path to your trained PyTorch model (.pth file)
PYTORCH_MODEL_PATH = os.path.join(project_root, "trained_models", "custom_binary_session", "epoch_15.pth")

# 2. Desired path for the output ONNX model
ONNX_MODEL_PATH = os.path.join(project_root, "trained_models", "custom_binary_session", "emcad_polyp_model.onnx")

# 3. Model parameters (MUST match the model you trained)
NUM_CLASSES = 2
ENCODER_NAME = 'pvt_v2_b2'
IMG_SIZE = 512
# =====================================================================================


def export_model_to_onnx():
    """
    Loads a trained PyTorch model and exports it to the ONNX format.
    """
    print("Starting ONNX export process...")
    
    # 1. Initialize the model architecture
    print(f"Initializing EMCADNet with {ENCODER_NAME} encoder...")
    model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER_NAME, pretrain=False, dual=False)

    # 2. Load the trained weights
    print(f"Loading trained weights from: {PYTORCH_MODEL_PATH}")
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu')) # Load to CPU for export
    model.eval() # Set the model to evaluation mode

    # 3. Create a dummy input tensor
    # This is a sample input that matches the model's expected input shape.
    # The batch size is 1, with 3 color channels, and IMG_SIZE x IMG_SIZE dimensions.
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, requires_grad=True)
    print(f"Created a dummy input tensor of shape: {dummy_input.shape}")

    # 4. Export the model
    print(f"Exporting model to ONNX format at: {ONNX_MODEL_PATH}")
    torch.onnx.export(model,
                      dummy_input,
                      ONNX_MODEL_PATH,
                      export_params=True,
                      opset_version=12, # A widely compatible version
                      do_constant_folding=True,
                      input_names=['input'], # Name for the input layer in the ONNX model
                      output_names=['outputs'], # Name for the output layer in the ONNX model
                      dynamic_axes={'input': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}} # Allow variable batch size
    )
    
    print("\nONNX model exported successfully!")
    print(f"Model saved to: {ONNX_MODEL_PATH}")

if __name__ == '__main__':
    export_model_to_onnx()