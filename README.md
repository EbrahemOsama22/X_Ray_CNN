
---

## Chest X-Ray Pneumonia Detection 🫁🩻

A deep learning computer vision project that classifies chest X-ray images as **Normal** or **Pneumonia** using transfer learning with a pre-trained **ResNet18** model.

📎 **Dataset:** [Kaggle — Chest X-Ray and OCT Medical Image Dataset](https://www.kaggle.com/datasets/gallo33henrique/chest-x-ray-and-oct-medical-image-dataset)

---

### 📌 What This Project Does

- **Data Exploration** — Analyzes image counts per class, calculates per-class min/max dimensions, and computes aspect ratio statistics (min, max, mean, std) for both `NORMAL` and `PNEUMONIA` images.
- **Duplicate Detection** — Uses **MD5 hashing** to identify and remove duplicate images before training, ensuring data integrity.
- **Data Preprocessing & Augmentation** — Applies different transformation pipelines for training and evaluation:
  - *Training:* Resize → CenterCrop (224×224) → Random Rotation → Random Horizontal Flip → Normalize
  - *Validation/Test:* Resize (224×224) → Normalize
- **Transfer Learning** — Fine-tunes a pre-trained **ResNet18** (ImageNet weights) by freezing all layers and replacing only the final fully connected layer for binary classification.
- **Training** — Trains with **Adam optimizer** and **CrossEntropyLoss**, tracking loss and accuracy per epoch with best model checkpointing.
- **Evaluation** — Generates a full **Classification Report** and **Confusion Matrix** on the test set.
- **Model Export** — Saves the trained model as `resnet18_pneumonia_classifier.pth` for future inference.

---

### 🤖 Model Architecture

| Detail | Value |
|---|---|
| Base Model | ResNet18 (pretrained on ImageNet) |
| Strategy | Feature Extraction (frozen backbone) |
| Output Classes | 2 — `NORMAL` / `PNEUMONIA` |
| Optimizer | Adam (lr = 0.001) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 (train) / 64 (test) |

---

### 🛠️ Tech Stack

`Python` · `PyTorch` · `Torchvision` · `Pillow` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `tqdm`

---

