# snake_spider_classifier/features.py

import sys
import numpy as np
import torch
from torchvision import transforms
from snake_spider_classifier.configs.logger import logging
from snake_spider_classifier.configs.exceptions import CustomException

# -----------------------------
# 1️⃣ Define data transforms
# -----------------------------
try:
    logging.info("Initializing data transformations...")

    # Training data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                      # Resize images to 224x224
        transforms.RandomHorizontalFlip(),                  # Random horizontal flip
        transforms.RandomRotation(20),                      # Random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color jitter
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # Random crop
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),  # Random affine
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),             # Perspective transform
        transforms.ToTensor(),                               # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # Random erasing
    ])

    # Validation / Test data transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    logging.info("Data transformations initialized successfully.")

except Exception as e:
    logging.error("Error occurred during transforms initialization")
    raise CustomException(e, sys)


# -----------------------------
# 2️⃣ MixUp and CutMix helper functions
# -----------------------------
def rand_bbox(size, lam):
    """Generate bounding box for CutMix"""
    try:
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
        y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2
    except Exception as e:
        logging.error("Error occurred in rand_bbox function")
        raise CustomException(e, sys)


def apply_mixup_cutmix(images, labels, device, p_mixup=0.5, p_cutmix=0.25, alpha_mixup=0.4, alpha_cutmix=1.0):
    """
    Apply MixUp or CutMix augmentation with probabilities p_mixup and p_cutmix.
    Returns transformed images and labels with mixing ratio.
    """
    try:
        rand_val = np.random.rand()
        if rand_val < p_mixup:
            lam = np.random.beta(alpha_mixup, alpha_mixup)
            index = torch.randperm(images.size(0)).to(device)
            mixed_x = lam * images + (1 - lam) * images[index, :]
            y_a, y_b = labels, labels[index]
            return mixed_x, y_a, y_b, lam, "mixup"
        elif rand_val < p_mixup + p_cutmix:
            lam = np.random.beta(alpha_cutmix, alpha_cutmix)
            index = torch.randperm(images.size(0)).to(device)
            y_a, y_b = labels, labels[index]
            x1, y1, x2, y2 = rand_bbox(images.size(), lam)
            images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
            return images, y_a, y_b, lam, "cutmix"
        else:
            return images, labels, labels, 1.0, "none"

    except Exception as e:
        logging.error("Error occurred in apply_mixup_cutmix")
        raise CustomException(e, sys)
