# snake_spider_classifier/modeling/train.py

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from snake_spider_classifier.dataset import get_train_val_dataloaders
from snake_spider_classifier.models.vgg16_se import CustomVGG16
from snake_spider_classifier.features import apply_mixup_cutmix
from snake_spider_classifier.configs.logger import logging
from snake_spider_classifier.configs.exceptions import CustomException

# -----------------------------
# 1️⃣ Hyperparameters
# -----------------------------
batch_size = 128
learning_rate = 1e-4
num_epochs = 50
p_mixup = 0.5
p_cutmix = 0.25
alpha_mixup = 0.4
alpha_cutmix = 1.0
early_lr_min, early_lr_max = 1e-6, 5e-6  # ramp-up

dataset_path = os.path.join(os.getcwd(), "data/raw")  # adjust as needed
reports_path = os.path.join(os.getcwd(), "reports")
os.makedirs(reports_path, exist_ok=True)
metrics_csv_path = os.path.join(reports_path, "training_metrics.csv")

# -----------------------------
# 2️⃣ Device and Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    train_loader, val_loader, num_classes = get_train_val_dataloaders(dataset_path, batch_size=batch_size)
    model = CustomVGG16(num_classes=num_classes).to(device)
except Exception as e:
    raise CustomException(e, sys)

# -----------------------------
# 3️⃣ Freeze first 3 blocks initially
# -----------------------------
for param in list(model.features[:16].parameters()):
    param.requires_grad = False

# -----------------------------
# 4️⃣ Optimizer, Scheduler, Scaler
# -----------------------------
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
scaler = torch.amp.GradScaler()

# -----------------------------
# 5️⃣ Loss with class weights
# -----------------------------
targets = [label for _, label in train_loader.dataset.dataset.dataset.samples]
class_counts = torch.tensor([targets.count(i) for i in range(num_classes)], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# -----------------------------
# 6️⃣ Checkpointing
# -----------------------------
checkpoint_path = os.path.join(reports_path, "vgg16_se_checkpoint.pth")
best_val_acc = 0.0
start_epoch = 0

if os.path.exists(checkpoint_path):
    logging.info("🔄 Restoring checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_acc"]
    logging.info(f"✅ Checkpoint restored (epoch {start_epoch}, val_acc={best_val_acc:.2f})")

# -----------------------------
# 7️⃣ SWA setup
# -----------------------------
use_swa = True
swa_start = int(0.7 * num_epochs)
swa_model = AveragedModel(model)
swa_scheduler = None
ramp_up_epochs = 5  # Gradual LR ramp-up

# -----------------------------
# 8️⃣ CSV logging setup
# -----------------------------
if not os.path.exists(metrics_csv_path):
    with open(metrics_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc", "val_acc", "train_loss", "val_loss", "lr"])

# -----------------------------
# 9️⃣ Training Loop
# -----------------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Gradual layer-wise fine-tuning
    if epoch >= 30:
        if epoch == 30:
            for param in model.features.parameters():
                param.requires_grad = True
            logging.info("🔓 Unfrozen all layers for fine-tuning.")

        ramp_progress = min(epoch - 30 + 1, ramp_up_epochs) / ramp_up_epochs
        early_lr = early_lr_min + (early_lr_max - early_lr_min) * ramp_progress
        optimizer = optim.AdamW([
            {'params': model.features[:16].parameters(), 'lr': early_lr},
            {'params': model.features[16:].parameters(), 'lr': learning_rate},
            {'params': model.classifier.parameters(), 'lr': learning_rate}
        ], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ---------- Training ----------
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, targets_a, targets_b, lam, mode = apply_mixup_cutmix(imgs, labels,
                                                                  p_mixup=p_mixup,
                                                                  p_cutmix=p_cutmix,
                                                                  alpha_mixup=alpha_mixup,
                                                                  alpha_cutmix=alpha_cutmix)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = lam*criterion(outputs, targets_a) + (1-lam)*criterion(outputs, targets_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # ---------- Validation ----------
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            correct_val += (preds==labels).sum().item()
            total_val += labels.size(0)

    val_acc = 100 * correct_val / total_val
    val_loss = val_loss / len(val_loader)

    # Update scheduler
    scheduler.step()

    # SWA update
    if use_swa and epoch >= swa_start:
        swa_model.update_parameters(model)
        if swa_scheduler is None:
            swa_scheduler = SWALR(optimizer, swa_lr=5e-6)

    # Save checkpoint if best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_val_acc
        }, checkpoint_path)

    # Save metrics to CSV
    current_lr = optimizer.param_groups[0]['lr']
    with open(metrics_csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_acc, val_acc, train_loss, val_loss, current_lr])

    logging.info(f"Epoch {epoch+1} → Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

# Apply SWA at the end
if use_swa:
    update_bn(train_loader, swa_model, device=device)
    model = swa_model
    logging.info("SWA applied to model.")

logging.info("✅ Training finished successfully.")
