# import os
# import torch
# from lib.networks import MIST_CAM

# # =====================================================
# # 1️⃣ Path to your model weights (.pth)
# # =====================================================
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# MODEL_PATH = os.path.join(
#     project_root,
#     "models",
#     "Synapse",
#     "run_2025-10-09_14_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo20_bs4_lr1e-05_256_s2222",
#     "epoch_19.pth"
# )

# # =====================================================
# # 2️⃣ Initialize and load model
# # =====================================================
# print(f"🔍 Loading model from: {MODEL_PATH}")

# # 🧠 Use the correct number of output classes (3 for multiclass)
# model = MIST_CAM(num_classes=3)

# state_dict = torch.load(MODEL_PATH, map_location='cpu')

# # Handle different checkpoint formats
# if isinstance(state_dict, dict) and 'model' in state_dict:
#     state_dict = state_dict['model']
# elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
#     state_dict = state_dict['state_dict']

# # Load weights
# missing, unexpected = model.load_state_dict(state_dict, strict=False)
# print(f"✅ Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

# model.eval()

# # =====================================================
# # 3️⃣ Count parameters
# # =====================================================
# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n✅ Total parameters: {total_params:,}")
#     print(f"🧠 Trainable parameters: {trainable_params:,}")

# count_parameters(model)
# # import torch
# # from thop import profile
# # from lib.MIST import CAM, FCT1, FCT2

# # # Fake args object
# # class Args:
# #     num_classes = 3

# # args = Args()

# # # Dummy input (batch=1, 3-channel RGB image, 256x256)
# # dummy_input = torch.randn(1, 3, 256, 256)

# # models = {
# #     "CAM": CAM(args),
# #     "FCT1": FCT1(args),
# #     "FCT2": FCT2(args)
# # }

# # for name, model in models.items():
# #     total_params = sum(p.numel() for p in model.parameters())
# #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    
# #     print(f"\n📊 {name} MODEL")
# #     print(f"✅ Total Parameters: {total_params:,}")
# #     print(f"🧠 Trainable Parameters: {trainable_params:,}")
# #     print(f"⚡ MACs: {macs / 1e9:.2f} GMac")
import os
import torch
from lib.networks import MIST_CAM

# =====================================================
# 1️⃣ Path to your model weights (.pth)
# =====================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(
    project_root,
    "models",
    "Synapse",
    "run_2025-10-09_14_Dual_MIST_CAM_loss_MUTATION_w3_7_256_pretrain_epo20_bs4_lr1e-05_256_s2222",
    "epoch_19.pth"
)

print(f"🔍 Loading model from: {MODEL_PATH}")

# =====================================================
# 2️⃣ Initialize and load model
# =====================================================
model = MIST_CAM(num_classes=3)

# Load checkpoint
state_dict = torch.load(MODEL_PATH, map_location="cpu")

# Handle different checkpoint formats
if isinstance(state_dict, dict):
    for key in ["model", "state_dict", "net"]:
        if key in state_dict:
            state_dict = state_dict[key]
            break

# Load weights
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"✅ Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

model.eval()

# =====================================================
# 3️⃣ Count Parameters
# =====================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Total Parameters: {total_params:,}")
    print(f"🧠 Trainable Parameters: {trainable_params:,}")

count_parameters(model)

# =====================================================
# 4️⃣ (Optional) Compute MACs/FLOPs
# =====================================================
# Uncomment this section if you have `thop` installed
"""
from thop import profile

dummy_input = torch.randn(1, 3, 256, 256)
macs, params = profile(model, inputs=(dummy_input,), verbose=False)

print(f"\n⚡ MACs (Multiply–Accumulate Ops): {macs / 1e9:.2f} GMac")
print(f"📦 Params: {params / 1e6:.2f} M")
"""
