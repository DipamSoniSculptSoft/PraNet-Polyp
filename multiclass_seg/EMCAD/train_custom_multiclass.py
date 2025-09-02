# Save this file as `train_custom_multiclass.py` inside the `multiclass_seg/EMCAD/` directory.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import numpy as np
import logging
import torch.optim.lr_scheduler as lr_scheduler

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
NUM_CLASSES = 3  # 0: Background, 1: Adenoma (Red), 2: Hyperplastic (Green)
ENCODER_NAME = 'pvt_v2_b2'
IMG_SIZE = 352
EPOCHS = 50  # Increased for more robust training
BATCH_SIZE = 10  # Increase if VRAM allows
LEARNING_RATE = 1e-4
DUAL_SUPERVISION = True
WEIGHT_DECAY = 1e-4 # Added for regularization

# =====================================================================================


class MultiClassSegDatasetJPEG(Dataset):
    """
    Custom Dataset for loading JPEG images and JPEG color-coded masks
    with robust color-to-class mapping. Now uses Albumentations for augmentations.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        label_mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
        red_pixels = (mask_rgb[:, :, 0] > 150) & (mask_rgb[:, :, 1] < 100) & (mask_rgb[:, :, 2] < 100)
        green_pixels = (mask_rgb[:, :, 1] > 150) & (mask_rgb[:, :, 0] < 100) & (mask_rgb[:, :, 2] < 100)
        label_mask[red_pixels] = 1
        label_mask[green_pixels] = 2

        if self.transform:
            augmented = self.transform(image=image, mask=label_mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

def compute_dice(pred, target, classes):
    dice_scores = []
    pred = pred.argmax(dim=1)
    for c in range(1, classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        dice = (2 * intersection + 1e-6) / (p.sum() + t.sum() + 1e-6)
        dice_scores.append(dice.item())
    return np.mean(dice_scores) if dice_scores else 0.0

def trainer_custom(model, snapshot_path):
    """ Main training function with validation, early stopping, and learning rate scheduling. """
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Starting multi-class training with JPEG masks...")

    # --- Enhanced Data Augmentation with Albumentations ---
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    logging.info("Loading dataset...")
    full_dataset = MultiClassSegDatasetJPEG(IMAGE_DIR, MASK_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply respective transforms to the datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    logging.info(f"Dataset loaded with {len(train_dataset)} train and {len(val_dataset)} val images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_dice = 0.0
    epochs_no_improve = 0
    patience = 10  # For early stopping

    logging.info(f"Starting training for {EPOCHS} epochs.")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)", unit="batch")

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            predictions = model(images)
            final_pred = predictions[3] if not DUAL_SUPERVISION else predictions[0] - predictions[-1]
            
            loss_ce = criterion_ce(final_pred, masks)
            loss_dice = criterion_dice(final_pred, masks, softmax=True)
            total_loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} --- Average Train Loss: {avg_epoch_loss:.4f}")

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)", unit="batch")
            for images, masks in val_progress:
                images, masks = images.to(device), masks.to(device, dtype=torch.long)
                predictions = model(images)
                final_pred = predictions[3] if not DUAL_SUPERVISION else predictions[0] - predictions[-1]
                dice = compute_dice(final_pred, masks, NUM_CLASSES)
                val_dice += dice
            
            val_dice /= len(val_loader)
            logging.info(f"Epoch {epoch+1}/{EPOCHS} --- Average Val Dice: {val_dice:.4f}")
        
        # Save best model and Early Stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved to {save_path} with Val Dice: {val_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        scheduler.step()

    logging.info("Training finished!")

if __name__ == '__main__':
    print(f"Initializing EMCADNet for multi-class segmentation with {NUM_CLASSES} classes...")
    model = EMCADNet(num_classes=NUM_CLASSES, encoder=ENCODER_NAME, pretrain=True, dual=DUAL_SUPERVISION)
    
    trainer_custom(model=model, snapshot_path=SNAPSHOT_PATH)