# CIFAR10-Vision-Transformer-SOTA
High-Performance Image Classification on CIFAR-10 using Vision Transformers (ViT). Achieved 98.87% accuracy.
# 🏎️ Stanford Cars Classification with EfficientNetV2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-92.40%25-brightgreen)
![Model](https://img.shields.io/badge/Model-EfficientNetV2--S-purple)

This project tackles the **Fine-Grained Visual Classification (FGVC)** task on the **Stanford Cars** dataset, distinguishing between **196 distinct car classes** (e.g., *Audi A5 Coupe* vs. *Audi S5 Coupe*). 

By leveraging **EfficientNetV2-S** and modern training techniques like **Hugging Face Streaming** and **PyTorch Compilation**, the model achieved a high Top-1 Accuracy of **92.40%**.

## 📊 Key Results

| Metric | Result |
| :--- | :--- |
| **Top-1 Test Accuracy** | **92.40%** |
| **Model Architecture** | `tf_efficientnetv2_s.in21k_ft_in1k` |
| **Training Time** | ~45 Minutes (15 Epochs on T4 GPU) |
| **Dataset Size** | 16,185 Images (196 Classes) |

> **Highlight:** The model utilizes `torch.compile()` to optimize the computation graph, significantly speeding up training while maintaining SOTA-level accuracy for a lightweight model.

---

## 🧠 Methodology

### 1. Model Selection: EfficientNetV2-S
Instead of heavy Transformers, we chose **EfficientNetV2-Small** (~24M parameters) for its balance of speed and accuracy.
* **Pre-training:** Initialized with **ImageNet-21k** weights to capture rich feature representations.
* **Fine-Tuning:** The classifier head was adapted for 196 classes.

### 2. Modern Data Pipeline
* **Streaming:** Used **Hugging Face Datasets** to stream data directly into RAM, bypassing the need for time-consuming downloads and unzip operations.
* **Augmentation:** Applied **RandAugment** (`num_ops=2`, `magnitude=9`) and **Mixup** ($\alpha=0.8$) to prevent overfitting on the fine-grained classes.

### 3. Optimization Strategy
* **Optimizer:** AdamW with Cosine Annealing Scheduler.
* **Compilation:** PyTorch 2.0 `torch.compile` mode was enabled for faster execution.
* **Test-Time Augmentation (TTA):** Evaluation was performed by averaging predictions of the original and flipped images.

---

## 📈 Performance Visualization

*(Training progress showing the correlation between Loss and Accuracy over 15 epochs.)*

![Learning Curves](stanford_cars_final_chart.png)

---

## 🛠️ How to Run

The project is designed to run on **Google Colab** (GPU required).

1. Open `StanfordCars_EfficientNetV2_Implementation.ipynb`.
2. Install dependencies:
   ```bash
   !pip install timm datasets accelerate
