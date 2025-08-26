# Save this file as `train_custom.py` inside the `multiclass_seg/EMCAD/` directory.

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
import tifffile
import logging

# Local imports from the EMCAD directory
from lib.networks import EMCADNet
from utils.utils import DiceLoss

# =====================================================================================
# CONFIGURATION - UPDATED
# =====================================================================================

# --- OS-Agnostic Path Construction ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# 1. Dataset Paths
IMAGE_DIR = os.path.join(project_root, "dataset", "images")
MASK_DIR = os.path.join(project_root, "dataset", "masks")

# 2. Output Path for models and logs
SNAPSHOT_PATH = os.path.join(project_root, "trained_models", "custom_binary_session")

# 3. Model and Training Parameters
# --- IMPORTANT ---
# For a binary segmentation task (one object class + background), NUM_CLASSES must be 2.
# Your masks should contain pixel values of 0 (background) and 1 (your object).
NUM_CLASSES = 2

ENCODER_NAME = 'pvt_v2_b2'
IMG_SIZE = 512
EPOCHS = 15
BATCH_SIZE = 5  # Note: A batch size of 5 might be slow if you have a powerful GPU. Adjust if needed.
LEARNING_RATE = 0.0001
# =====================================================================================


class CustomSegDataset(Dataset):
    """ Loads .png images and .tif masks """
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".tif"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask_array = tifffile.imread(mask_path)
        mask = Image.fromarray(mask_array)

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
    logging.info("Starting custom training...")

    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])

    logging.info("Loading dataset...")
    dataset = CustomSegDataset(IMAGE_DIR, MASK_DIR, image_transform, mask_transform)
    # Note: Set num_workers to 0 if you are on Windows and encounter DataLoader errors.
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
        
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1: # Save more frequently
            save_path = os.path.join(snapshot_path, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Checkpoint saved to {save_path}")

    logging.info("Training finished!")

if __name__ == '__main__':
    print(f"Initializing EMCADNet with {ENCODER_NAME} encoder...")
    model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER_NAME, pretrain=True, dual=False)
    
    trainer_custom(model=model, snapshot_path=SNAPSHOT_PATH)