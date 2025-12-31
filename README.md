# 🚀 High-Performance CIFAR-10 Classification with Vision Transformers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.87%25-brightgreen)
![Model](https://img.shields.io/badge/Model-ViT--Base-purple)

This project implements a State-of-the-Art (SOTA) image classification pipeline on the **CIFAR-10** dataset using the **Vision Transformer (ViT)** architecture. By leveraging Transfer Learning from ImageNet-21k and upscaling inputs, the model achieved a remarkable **Top-1 Accuracy of 98.87%**, surpassing traditional CNN baselines.

## 📊 Key Results

| Metric | Result |
| :--- | :--- |
| **Top-1 Test Accuracy** | **98.87%** |
| **Best Epoch** | 6 |
| **Training Loss** | 1.0188 (with Mixup/Cutmix) |
| **Model Architecture** | `vit_base_patch16_224` |

> **Note:** The model demonstrated rapid convergence, reaching >96% accuracy within the first epoch due to robust pre-training.

---

## 🧠 Methodology

### 1. Model Architecture: ViT-Base
Instead of using standard CNNs (like ResNet), we utilized the **Vision Transformer (ViT-Base)**.
* **Pre-training:** Initialized with weights from **ImageNet-21k** (14M images) to capture rich semantic features.
* **Input Adaptation:** Although CIFAR-10 images are 32x32, we **upscaled them to 224x224** using bicubic interpolation. This bridges the resolution gap and allows the ViT patch embeddings to function optimally.

### 2. Strong Training Recipe
To prevent overfitting on the relatively small CIFAR-10 dataset (50k images), we applied a "Strong Recipe":
* **Optimizer:** AdamW (Weight Decay: 0.05)
* **Scheduler:** Cosine Annealing Learning Rate.
* **Regularization:**
    * **Mixup ($\alpha=0.8$):** Blends images and labels.
    * **Cutmix ($\alpha=1.0$):** Swaps patches between images.
    * **RandAugment:** Automated augmentation policy.
    * **Label Smoothing:** Prevents over-confidence.

### 3. Test-Time Augmentation (TTA)
During inference/testing, we averaged predictions from the original image and its horizontally flipped version to maximize accuracy.

---

## 📈 Performance Visualization

*(The learning curve below demonstrates the training stability and rapid convergence.)*

![Learning Curves](learning_curves.png)

---

## 🛠️ How to Run

The implementation is provided as a Jupyter Notebook designed to run on Google Colab (with T4/P100 GPU).

1. Open `CIFAR10_ViT_Implementation.ipynb` in this repository.
2. Click the "Open in Colab" button (if available) or download and upload to Google Colab.
3. Install dependencies:
   ```bash
   !pip install timm
