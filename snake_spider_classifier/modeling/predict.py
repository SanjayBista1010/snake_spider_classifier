# snake_spider_classifier/modeling/predict.py

import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from snake_spider_classifier.models.vgg16_se import CustomVGG16
from snake_spider_classifier.features import get_transform
from snake_spider_classifier.configs.logger import logging
from snake_spider_classifier.configs.exceptions import CustomException

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(os.getcwd(), "reports", "vgg16_se_checkpoint.pth")
class_names = ["snake", "spider"]  # adjust according to your dataset

# -----------------------------
# 2️⃣ Load Model
# -----------------------------
try:
    model = CustomVGG16(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    raise CustomException(e, sys)

# -----------------------------
# 3️⃣ Image Prediction Function
# -----------------------------
def predict_image(image_path: str):
    """
    Predict the class of a single image.
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Transform image
        transform = get_transform(train=False)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        result = {
            "prediction": class_names[pred_idx.item()],
            "confidence": confidence.item(),
            "probabilities": {cls: float(probs[0][i].item()) for i, cls in enumerate(class_names)}
        }

        logging.info(f"Prediction for {os.path.basename(image_path)}: {result['prediction']} "
                     f"({result['confidence']*100:.2f}%)")
        return result

    except Exception as e:
        raise CustomException(e, sys)

# -----------------------------
# 4️⃣ Example Usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict image class using trained VGG16_SE model")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    prediction = predict_image(args.image)
    logging.info(f"Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']*100:.2f}%)")
    print("📊 Class probabilities:")

