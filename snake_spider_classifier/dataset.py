# snake_spider_classifier/dataset.py

import sys
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from snake_spider_classifier.features import train_transform, val_transform
from snake_spider_classifier.configs.logger import logging
from snake_spider_classifier.configs.exceptions import CustomException

# -----------------------------
# 1️⃣ Custom dataset loader
# -----------------------------
class ImageFolderWithTransform(Dataset):
    """
    Wrapper around torchvision.datasets.ImageFolder to apply transforms dynamically.
    """
    def __init__(self, root_dir, transform=None):
        try:
            self.dataset = datasets.ImageFolder(root=root_dir)
            self.transform = transform
            self.classes = self.dataset.classes
            logging.info(f"Loaded dataset from {root_dir} with classes: {self.classes}")
        except Exception as e:
            logging.error("Error initializing ImageFolderWithTransform")
            raise CustomException(e, sys)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            logging.error(f"Error in __getitem__ at index {idx}")
            raise CustomException(e, sys)


# -----------------------------
# 2️⃣ Dataset splits
# -----------------------------
def get_train_val_dataloaders(dataset_path, batch_size=32, val_split=0.2, num_workers=2, pin_memory=True):
    """
    Returns train and validation dataloaders.
    """
    try:
        # Load full dataset
        full_dataset = ImageFolderWithTransform(root_dir=dataset_path, transform=train_transform)

        # Compute sizes
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        # Split dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply validation transform to val_dataset
        val_dataset.dataset.transform = val_transform

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

        logging.info(f"Train size: {train_size}, Val size: {val_size}")
        return train_loader, val_loader, len(full_dataset.classes)

    except Exception as e:
        logging.error("Error creating train/val dataloaders")
        raise CustomException(e, sys)


# -----------------------------
# 3️⃣ Example usage
# -----------------------------
if __name__ == "__main__":
    try:
        dataset_path = os.path.join(os.getcwd(), "data/raw")  # adjust path if needed
        train_loader, val_loader, num_classes = get_train_val_dataloaders(dataset_path, batch_size=16)
        logging.info(f"Number of classes: {num_classes}")
        logging.info("Dataset and DataLoaders initialized successfully.")
    except Exception as e:
        logging.error("Failed to initialize dataset and DataLoaders")
        raise CustomException(e, sys)
