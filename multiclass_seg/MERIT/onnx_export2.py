# =====================================================================================
# MODEL EXPORT SCRIPT: PyTorch MERIT to ONNX
#
# DESCRIPTION:
# This script is designed to export a pre-trained PyTorch MERIT model to the
# ONNX (Open Neural Network Exchange) format for cross-platform inference.
#
# HOW TO USE:
#   1.  Ensure you have all required libraries installed (torch, onnx).
#   2.  Run the script from your terminal, providing the required paths and flags.
#
#      EXAMPLE (for a dual-supervision model):
#      python your_export_script_name.py ^
#          --pytorch_model_path "C:\path\to\your\model.pth" ^
#          --onnx_model_path "C:\path\to\save\model.onnx" ^
#          --dual
#
# =====================================================================================

import os
import torch
import torch.onnx
import sys
import argparse

# Add local libraries to path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes the 'lib' folder is one directory level up from this script
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import the correct model architectures from the network library
from lib.networks import MERIT_Cascaded, MERIT_Cascaded_dual

# =====================================================================================
# CONFIGURATION VIA COMMAND-LINE ARGUMENTS
# =====================================================================================
parser = argparse.ArgumentParser(description='Export a PyTorch MERIT model to ONNX format.')

parser.add_argument('--pytorch_model_path', type=str, required=True,
                    help='Absolute path to your trained .pth model file.')

parser.add_argument('--onnx_model_path', type=str, required=True,
                    help='Destination path for the output .onnx model file.')

args = parser.parse_args()


# Model-specific parameters: These must match the parameters used during training.
NUM_CLASSES = 3
IMG_SIZE = 256
DUAL_SUPERVISION = True
ONNX_OPSET_VERSION = 14
# =====================================================================================

#
class ModelWrapper(torch.nn.Module):
    """
    A wrapper class to process the multiple outputs of the MERIT model
    and return a single final tensor of probabilities, simplifying inference.
    """
    def __init__(self, model, is_dual):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.dual = is_dual

    def forward(self, x):
        all_predictions = self.model(x)
        if self.dual:
            # CORRECTED LOGIC: Matches the validation logic from train_ACDC.py
            p_foreground = all_predictions[:4]
            p_background = all_predictions[-4:]
            outputs = 0.0
            for fg, bg in zip(p_foreground, p_background):
                outputs += (fg - bg)
        else:
            # Logic for single supervision models
            outputs = 0.0
            for p in all_predictions:
                outputs += p
        
        final_output = torch.softmax(outputs, dim=1)
        return final_output


def export_model_to_onnx():
    """
    Main function to handle the model loading, wrapping, and ONNX export process.
    """
    print("--- Starting ONNX Export Process for MERIT Model ---")

    if not os.path.exists(args.pytorch_model_path):
        print(f"\n[ERROR] PyTorch model not found at: {args.pytorch_model_path}")
        print("Please check that the --pytorch_model_path argument is correct.")
        return

    print("Initializing MERIT model architecture...")
    # Conditionally initialize the correct model architecture based on the --dual flag
    if args.dual:
        print("Loading DUAL supervision model: MERIT_Cascaded_dual")
        model = MERIT_Cascaded_dual(
            n_class=NUM_CLASSES,
            img_size_s1=(IMG_SIZE, IMG_SIZE),
            img_size_s2=(224, 224),
            model_scale='small',
            decoder_aggregation='additive',
            interpolation='bilinear'
        )
    else:
        print("Loading SINGLE supervision model: MERIT_Cascaded")
        model = MERIT_Cascaded(
            n_class=NUM_CLASSES,
            img_size_s1=(IMG_SIZE, IMG_SIZE),
            img_size_s2=(224, 224),
            model_scale='small',
            decoder_aggregation='additive',
            interpolation='bilinear'
        )

    print(f"Loading trained weights from: {args.pytorch_model_path}")
    # Load weights onto the CPU for portability
    model.load_state_dict(torch.load(args.pytorch_model_path, map_location='cpu'))
    model.eval()

    print("Wrapping model for a single inference output...")
    wrapped_model = ModelWrapper(model, is_dual=args.dual)
    wrapped_model.eval()

    # Create a dummy input tensor that matches the expected model input shape
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print(f"Created a dummy RGB input tensor of shape: {dummy_input.shape}")

    print(f"Exporting model to ONNX (opset {ONNX_OPSET_VERSION}) at: {args.onnx_model_path}")
    # Export the wrapped model to the ONNX format
    torch.onnx.export(wrapped_model,
                      dummy_input,
                      args.onnx_model_path,
                      export_params=True,
                      opset_version=ONNX_OPSET_VERSION,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output_probabilities'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output_probabilities': {0: 'batch_size'}}
    )

    print("\n✅ ONNX model exported successfully!")
    print(f"   Model saved to: {args.onnx_model_path}")
    print(f"   Input node name: 'input', Shape: [batch_size, 3, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"   Output node name: 'output_probabilities', Shape: [batch_size, {NUM_CLASSES}, {IMG_SIZE}, {IMG_SIZE}]")


if __name__ == '__main__':
    export_model_to_onnx()