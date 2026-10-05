# Synthetic Media Forensics Engine
## EfficientNet-B0 Convolutional Analysis

An end-to-end computer vision pipeline built with PyTorch to detect AI-generated human faces (StyleGAN / StyleGAN2). This project leverages a pretrained **EfficientNet-B0** backbone fine-tuned for high-accuracy binary classification, featuring strict **leakage-safe dataset splitting**, comprehensive augmentation audits, and robustness analysis.

---

## Project Overview

With the rapid progression of generative adversarial networks (GANs) and diffusion models, detecting synthetic face imagery has become critical for media forensics and digital security. This project implements a deep learning pipeline to distinguish between authentic human portraits and synthetic face images.

### Key Highlights
* **Pretrained Backbone:** Fine-tuned `EfficientNet-B0` architecture balancing parameter efficiency with feature extraction capacity.
* **Leakage-Safe Partitioning:** Group-based train/val/test splitting using canonical base filenames to prevent identity and augmentation leakage.
* **Robust Evaluation:** Comprehensive post-training evaluation, including confusion matrices, ROC-AUC, classification reports, and failure-case analysis.

---

## Dataset & Integrity Audit

The project uses a combined StyleGAN / StyleGAN2 dataset with labeled `Real` and `Fake` facial images.

| Category | Sample Count | Percentage |
| :--- | :--- | :--- |
| **Synthetic (AI-Generated)** | 7,000 | 54.31% |
| **Authentic (Real)** | 5,890 | 45.69% |
| **Total Images** | **12,890** | **100.0%** |

### Data Leakage Mitigation
A detailed audit of the dataset revealed that exactly **50% of the images (6,445 images)** are augmented derivatives containing regex patterns like `_aug_\d+`. 

* **Unique Underlying Base Images:** 7,734
* **Distribution:** 6,445 images exist as single instances; 1,289 base images have 5 augmented variants.
* **Architecture Solution:** Data splits are grouped by underlying `base_name`. All augmented copies of any base image are confined to the same partition (Train, Validation, or Test) to ensure evaluation metrics reflect true generalization rather than memorized transformations.

---

## Pipeline Architecture

1. **Environment Setup & Verification:** Hardware detection (CUDA, NVIDIA Tensor Core GPU verification) and high-throughput data pipeline mounting.
2. **Data Inspection & Decontamination:** Image loading, metadata parsing, label distribution checks, and filename deduplication.
3. **Leakage-Safe Splitting:** Stratified group-based dataset partition into Training, Validation, and Holdout Test sets.
4. **Data Transforms & Augmentation:** Input normalization according to ImageNet standards, spatial resizing, and real-time training augmentations (rotations, flips, color jitter).
5. **Model Training:**
   * Backbone: `EfficientNet-B0` (Pretrained weights)
   * Custom Head: Adaptive pooling, dropout regularization, and binary classification projection.
   * Optimization: AdamW / Adam optimizer with cross-entropy loss and learning rate scheduling.
6. **Evaluation & Error Analysis:** Evaluation on the unseen holdout split, threshold calibration, and false-positive / false-negative inspection.

---

## Tech Stack & Environment

* **Framework:** PyTorch (`torch`, `torchvision`)
* **Computer Vision:** OpenCV, PIL, Albumentations / Torchvision Transforms
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Compute Platform:** Linux x86_64
* **Tested Hardware:** NVIDIA L4 (24GB VRAM) / NVIDIA T4 / A100

---

## Deployment Configuration

This repository is configured for automated deployment via Streamlit Community Cloud.

### Running Locally
```bash
git clone https://github.com/RohitSharma69/deepfake-detector.git
cd deepfake-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Evaluation & Results

The fine-tuned EfficientNet-B0 model was evaluated on the unseen test set:

* **Accuracy:** High binary classification discriminability between synthetic GAN artifacts and authentic skin textures.
* **Precision & Recall:** Balanced performance across both classes, minimizing false positives.
* **Inference Speed:** Real-time throughput (~15–25ms per image on modern GPUs), suitable for batch video frame processing.

---

## Future Architecture Roadmap

- Add frequency-domain feature extractors (DCT / FFT analysis) to detect GAN blending boundary artifacts.
- Benchmark against larger foundation models (ConvNeXt, Swin Transformer).
- Extend classification to video streams via face extraction (MTCNN / RetinaFace) and temporal sequence modeling.

---
*Developed by Rohit Sharma.*