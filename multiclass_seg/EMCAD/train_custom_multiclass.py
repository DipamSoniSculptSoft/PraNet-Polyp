# Save this file as `train_custom_multiclass.py` inside the `multiclass_seg/EMCAD/` directory.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import logging

# Local imports
from lib.networks import EMCADNet
from utils.utils import DiceLoss

# =====================================================================================
# CONFIGURATION
# =====================================================================================
# --- OS-Agnostic Path Construction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# 1. Dataset Paths
IMAGE_DIR = os.path.join(project_root, "dataset_classification", "images")
MASK_DIR = os.path.join(project_root, "dataset_classification", "masks")

# 2. Output Path
SNAPSHOT_PATH = os.path.join(project_root, "trained_models", "custom_multiclass_session")

# 3. Model and Training Parameters
NUM_CLASSES = 3 # 0: Background, 1: Adenoma (Red), 2: Hyperplastic (Green)
ENCODER_NAME = 'pvt_v2_b2'
IMG_SIZE = 512
EPOCHS = 15
BATCH_SIZE = 7
LEARNING_RATE = 0.0001
# =====================================================================================


class MultiClassSegDatasetJPEG(Dataset):
    """
    Custom Dataset for loading JPEG images and JPEG color-coded masks.
    This version is robust to JPEG compression artifacts in the masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        # --- CHANGE: Look for .jpg or .jpeg files ---
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Assume mask has the same name and extension
        mask_path = os.path.join(self.mask_dir, img_name) 

        # Load image and mask as RGB
        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        # --- CHANGE: Robust color-to-class mapping for JPEG masks ---
        mask_np_rgb = np.array(mask_rgb)
        label_mask = np.zeros((mask_np_rgb.shape[0], mask_np_rgb.shape[1]), dtype=np.uint8)

        # Identify pixels that are "mostly red"
        # Condition: Red channel is high AND significantly higher than Green and Blue
        red_pixels = (mask_np_rgb[:, :, 0] > 150) & (mask_np_rgb[:, :, 1] < 100) & (mask_np_rgb[:, :, 2] < 100)
        
        # Identify pixels that are "mostly green"
        # Condition: Green channel is high AND significantly higher than Red and Blue
        green_pixels = (mask_np_rgb[:, :, 1] > 150) & (mask_np_rgb[:, :, 0] < 100) & (mask_np_rgb[:, :, 2] < 100)

        # Assign class indices. All other pixels will remain 0 (background).
        label_mask[red_pixels] = 1    # Adenoma
        label_mask[green_pixels] = 2  # Hyperplastic
        
        # Convert the numpy label map back to a PIL Image for transformations
        mask = Image.fromarray(label_mask)

        # Apply standard transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0)
            
        return image, mask

def trainer_custom(model, snapshot_path):
    """ Main training function """
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Starting multi-class training with JPEG masks...")

    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])

    logging.info("Loading dataset...")
    dataset = MultiClassSegDatasetJPEG(IMAGE_DIR, MASK_DIR, image_transform, mask_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    logging.info(f"Dataset loaded with {len(dataset)} images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

    logging.info(f"Starting training for {EPOCHS} epochs.")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            predictions = model(images)
            final_pred = predictions[3]
            
            loss_ce = criterion_ce(final_pred, masks)
            loss_dice = criterion_dice(final_pred, masks, softmax=True)
            total_loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} --- Average Loss: {avg_epoch_loss:.4f}")
        
        if (epoch + 1) % 25 == 0 or epoch == EPOCHS - 1:
            save_path = os.path.join(snapshot_path, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Checkpoint saved to {save_path}")

    logging.info("Training finished!")

if __name__ == '__main__':
    print(f"Initializing EMCADNet for multi-class segmentation with {NUM_CLASSES} classes...")
    model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER_NAME, pretrain=True, dual=False)
    
    trainer_custom(model=model, snapshot_path=SNAPSHOT_PATH)