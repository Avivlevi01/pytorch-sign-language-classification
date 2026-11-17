# pytorch-sign-language-classification
Final ML Project: CNN vs. Logistic Regression for 36-class sign language classification. Code is private for academic integrity.

[README.md](https://github.com/user-attachments/files/34965045/README.md)
# 🤟 Sign Language Alphanumeric Classification (CNN vs. Logistic Regression)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-blue?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Multiclass%20Classification-success?style=for-the-badge)

**Authors:** Aviv Levi & Ion Vasile Cojocari  
**Course:** Introduction to Machine Learning (SS 2026)  

> 📄 **[Click here to view the full Technical Project Report (PDF)](./Final_Project_Innsbruck_University_ML.pdf)**

---

## 📌 Project Overview
This project tackles a **36-class image classification problem**: recognizing human hand shapes representing alphanumeric sign language characters (`0–9` and `a–z`). 

Using a balanced dataset of **9,680 grayscale images** ($128 \times 128$ pixels each), we designed, trained, and evaluated two distinct machine learning architectures in **PyTorch**:
1. **Multinomial Logistic Regression** (Baseline flat linear model)
2. **Deep Convolutional Neural Network (CNN)** (Spatial feature extractor)

Our primary goal was to demonstrate the importance of architectural inductive bias in computer vision—specifically how local receptive fields and spatial pooling overcome the high variance and overfitting of flat linear models.

---

## 📊 Dataset & Preprocessing
* **Total Samples:** 9,680 unique grayscale images (no missing data, evenly distributed across all 36 classes).
* **Input Resolution:** $128 \times 128$ pixels ($16,384$ raw pixel features per image).
* **Target Output:** Integer index `0–35` mapped to `[0-9, a-z]`.
* **Preprocessing Pipeline:**
  * Converted raw image arrays to floating-point PyTorch tensors.
  * Scaled pixel intensities from `[0, 255]` to `[0, 1]`.
  * Standardized inputs using $\mu = 0.5$ and $\sigma = 0.5$.
  * Preserved native $128 \times 128$ resolution to comply with the official evaluation environment constraints.

---

## 🧠 Model Architectures

### 1. Baseline: Multinomial Logistic Regression
Maps the flattened $16,384$ pixel vector directly to 36 class logits via a single linear transformation layer followed by a Softmax activation.

### 2. Primary Model: Convolutional Neural Network (CNN)
Designed to capture spatial hierarchies, edge orientations, and hand contours with translation invariance:

    Input Image (1 x 128 x 128)
       │
       ├──► Conv2d (16 filters, 3x3 kernel, padding=1) + ReLU
       ├──► MaxPool2d (2x2) ──► Output: 16 x 64 x 64
       │
       ├──► Conv2d (32 filters, 3x3 kernel, padding=1) + ReLU
       ├──► MaxPool2d (2x2) ──► Output: 32 x 32 x 32
       │
       ├──► Flatten + Dropout (p = 0.50)
       ├──► Fully Connected Layer (256 units) + ReLU
       └──► Output Linear Layer (36 classes)

### ⚙️ Training & Hyperparameters
* **Optimizer:** `AdamW` (Adaptive learning rates + decoupled weight decay)
* **Learning Rate:** `0.001`
* **Weight Decay:** `0.01`
* **Batch Size:** `64`
* **Regularization:** Dropout (`0.50`) + Model Checkpointing (saving weights at peak validation loss)

---

## 🏆 Results & Performance Analysis

| Model | Training Accuracy | Validation Accuracy | Test Set Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (Baseline)** | 94.48% | 72.50% | — | Severe overfitting; ignores spatial dependencies |
| **CNN (Best Checkpoint / Final)** | **94.55%** | **85.33%** | **95.36%** | Strong generalization on unseen data |

### Training & Validation Curves (20 Epochs)
> *Note: Validation loss plateaued around Epoch 6–7 while training loss continued to decrease, highlighting the importance of early model checkpointing.*

<p align="center">
  <img src="./assets/accuracy_curve.png" alt="CNN Training and Validation Accuracy" width="48%" />
  <img src="./assets/loss_curve.png" alt="CNN Training and Validation Loss" width="48%" />
</p>

---

## 💡 Key Takeaways & Future Improvements
* **Inductive Bias Matters:** Flattening images into $16,384$ independent features causes linear models to memorize background noise and static pixel positions. Convolutional filters and spatial pooling allow the CNN to learn true geometric hand contours.
* **Future Work:**
  * **Real-Time Data Augmentation:** Adding random rotations, slight translations, horizontal flips, and brightness shifts to curb post-epoch-7 overfitting.
  * **Batch Normalization:** Inserting `BatchNorm2d` layers after convolutional blocks to stabilize internal covariate shifts and accelerate convergence.

---

## 🔒 Source Code & Academic Integrity Notice
To comply with university academic integrity policies and prevent plagiarism in future course iterations, the full source code, training scripts, and dataset loaders are hosted in a **private repository**.

💼 **For Recruiters & Hiring Managers:**  
If you would like to review the original PyTorch implementation, training notebooks, or model weights, please feel free to reach out directly and I will gladly grant you access or share a code walkthrough.
