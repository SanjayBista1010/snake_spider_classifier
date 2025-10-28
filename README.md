# 🐍🕷️ Snake vs Spider Classifier

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

A **deep learning project** for classifying images of **snakes** and **spiders** using a **custom VGG16 + Squeeze-and-Excitation (SE) model** with advanced training techniques like **MixUp, CutMix, Gradual Fine-tuning, Layer-wise LR Ramp-up, and Stochastic Weight Averaging (SWA)**.

---

## 🔗 Live Demos

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WwaLVpm7ENUtUU1iWtY1-_8omTFOhD2A)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://snakespiderclassifier-r8szaq6993lwfdurucxjgb.streamlit.app/)

- **Colab Notebook**: Run the full training pipeline
- **Streamlit Web App**: Test the model with your own images

---

## 📈 Project Highlights  

- **Model:** Custom VGG16 + SE blocks  
- **Input size:** 224 × 224 images  
- **Number of classes:** 2 (Snake, Spider)  
- **Dataset:** 34,000 images (17,000 per class)
- **Data augmentation:** Random horizontal flip, rotation, perspective, color jitter, affine transforms, Random Erasing  
- **Advanced training:**  
  - MixUp and CutMix  
  - Gradual layer-wise fine-tuning  
  - Layer-wise learning rate ramp-up  
  - SWA for final model  
- **Training metrics logged:** Train loss, train/val accuracy, learning rate, best validation accuracy, frozen/trainable layers & parameters per epoch 

---

## 🏆 Performance  

Here's a snapshot of **training results over 50 epochs**:

| Epoch | Train Acc (%) | Val Acc (%) | Learning Rate | Best Val Acc (%) | Trainable Layers |
|-------|---------------|-------------|---------------|------------------|------------------|
| 1     | 68.55         | 80.05       | 9.76E-05      | 0                | 48               |
| 10    | 82.76         | 90.52       | 0.0001        | 89.87            | 48               |
| 20    | 85.30         | 93.27       | 5.05E-05      | 94.30            | 48               |
| 30    | 85.21         | 94.93       | 0.0001        | 95.30            | 68               |
| 50    | 81.78         | 98.64       | 4.90E-06      | 98.68            | 68               |

> Maximum validation accuracy achieved: **98.68%**

## 📊 Classification Report

The performance of the Snake vs Spider Classifier on the test dataset is as follows:

```text
               precision    recall  f1-score   support

           0        0.99      0.99      0.99      3398
           1        1.00      0.99      0.99      3584

    accuracy                            0.99      6982
    macro avg       0.99      0.99      0.99      6982
 weighted avg       0.99      0.99      0.99      6982
```

## 📂 Dataset Structure
data/raw/
├── snake/
│ ├── snake1.jpg
│ └── ...
└── spider/
├── spider1.jpg
└── ...


Place images in their respective class folders (snake or spider).

Split is handled automatically: 80% train / 20% validation.

## 🏗️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/SanjayBista1010/snake-spider-classifier.git
cd snake-spider-classifier

# Create and activate conda environment
conda env create -f environment.yml
conda activate snake_spider_classifier

# (Optional) Manual installation of dependencies
pip install torch torchvision numpy pandas scikit-learn tqdm
```
Environment includes: PyTorch, torchvision, NumPy, Pandas, Scikit-learn, tqdm.


## ⚡ Features

    1️⃣ Gradual Fine-tuning - Initial blocks frozen, gradually unfrozen for better performance

    2️⃣ MixUp & CutMix - Data augmentation techniques for generalization

    3️⃣ Layer-wise Learning Rate Ramp-up - Maintains high first-epoch accuracy and stabilizes training

    4️⃣ Stochastic Weight Averaging (SWA) - Smooths final weights for improved validation accuracy

    5️⃣ CSV Logging - Tracks train/val accuracy, loss, learning rate, best accuracy, trainable/frozen parameters & layers per epoch


## 🚀 Training

To train the model:
bash

    python -m snake_spider_classifier.modeling.train

Training Scale:

- 34,000 total images

- 27,200 training images

- 6,800 validation images

- 50 epochs with advanced augmentation

## 🌐 Web App
Streamlit App (Recommended)

Try the live classification web app:

[🐍🕷️ Snake vs Spider Classifier Web App](https://snakespiderclassifier-r8szaq6993lwfdurucxjgb.streamlit.app/)

The web app allows you to:

- Upload snake or spider images

- Get real-time predictions

- See confidence scores

- Test the model easily without any setup

# Local Flask App

Run the Flask app locally:
bash

    python flask_app.py

The Flask app will start a local server at http://localhost:5000 and provides:

- Web interface for image classification

- REST API endpoints

- File upload functionality

- Real-time predictions

##  📊 Results

First-epoch accuracy: ~68–70%

Final validation accuracy: ~98.6%

Training is robust across multiple epochs with advanced augmentation

##  📝 License

This project is licensed under the MIT License – see the LICENSE file for details.

## 💡 Acknowledgments

Inspired by VGG16 and Squeeze-and-Excitation Networks

Uses PyTorch, torchvision, and SWA utilities
    
