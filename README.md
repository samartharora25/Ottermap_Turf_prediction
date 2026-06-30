# 🗺️ Ottermap: Aerial Turf & Vegetation Semantic Segmentation

![Project Status](https://img.shields.io/badge/Status-Complete-success) ![Model](https://img.shields.io/badge/Model-ResNet50%20U--Net-blue) ![Challenge](https://img.shields.io/badge/Ottermap-72h%20Challenge-orange)

An end-to-end computer vision pipeline for high-precision turf mapping in complex aerial and satellite imagery. This repository contains the training experiments, model weights, and production inference scripts developed for the Ottermap 72-hour Challenge to solve high-resolution vegetation segmentation under extreme data scarcity.

---

## 📑 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Architectural Innovation](#-architectural-innovation)
3. [Training & Augmentation Strategy](#-training--augmentation-strategy)
4. [Results & Visual Explanations](#-results--visual-explanations)
5. [Repository Structure](#-repository-structure)
6. [Installation & Inference](#-installation--inference)

---

## 🎯 Problem Statement
The core challenge of this project was **extreme data scarcity**. The provided dataset contained only **3 training images** yielding a total of 161 annotated turf polygons. 

Standard deep learning approaches would immediately overfit, memorizing the source pixels rather than learning the generalized spatial morphology of "grass" vs. "tree canopy" vs. "pavement." This required a hybrid approach combining heavy transfer learning with domain-specific spectral feature engineering.

---

## 🧠 Architectural Innovation

### 1. Model Selection: ResNet50 U-Net
* **Encoder:** ResNet50 (Pre-trained on ImageNet). Selected for its deep feature extraction capabilities while maintaining a manageable parameter count.
* **Decoder:** U-Net architecture. Provides the skip-connections necessary to reconstruct high-fidelity spatial boundaries required for pixel-perfect GIS polygons.

### 2. 4-Channel Pseudo-NDVI Integration
Standard RGB models struggle to differentiate vegetation from purely visual cues in varied lighting. We modified the initial `Conv2d` entry layer of the ResNet50 backbone to accept a **4-channel input tensor `(RGB + Pseudo-NDVI)`**.

The 4th channel is a pre-computed spectral vegetation heatmap generated using the Normalized Difference Vegetation Index formula:

$$NDVI = \frac{Green - Red}{Green + Red + \epsilon}$$

By explicitly feeding this spectral index to the model, we offloaded the burden of learning color-space discrimination. This allowed the CNN to dedicate its representational capacity entirely to learning spatial hierarchies, textures, and boundary edges.

### 3. Composite Loss Function
Turf mapping suffers from severe class imbalance (grass usually occupies < 15% of an aerial image). We utilized a dual loss strategy:
* **Focal Loss:** Aggressively down-weights "easy" background pixels (like roads), forcing the model to focus on the high-variance boundary edges of the turf.
* **Dice Loss:** Directly optimizes for the Intersection-over-Union (IoU) metric.

---

## 🛠️ Training & Augmentation Strategy

To simulate diverse drone/satellite sensor profiles and seasonal lighting, we utilized an aggressive augmentation regime via `Albumentations`:
* **Spatial:** Random 90° rotations, horizontal/vertical flips.
* **Spectral:** Random Brightness/Contrast, HSV jitter, and Gaussian noise.

**Validation Protocol:** A strict **Leave-One-Image-Out Cross-Validation (3 Folds)** was employed. This prevents geographic data leakage and ensures the reported metrics represent true generalization to unseen environments. Training utilized Early Stopping (Patience=10).

---

## 📊 Results & Visual Explanations

### Quantitative Metrics (3-Fold CV)
| Image Fold | Peak Validation IoU | Observations |
| :--- | :--- | :--- |
| **Fold 1** | **65.10%** | Peak performance achieved at Epoch 14. |
| **Fold 2** | 33.68% | High variance due to image-specific spectral profiles. |
| **Fold 3** | 37.97% | Demonstrated resilience on complex, high-canopy urban imagery. |

### Visual Explanations & Turf Markings
The pipeline successfully differentiates between flat turf and volumetric vegetation (like tree canopies). Below are visual validations from our external test set. 

*(Note: The red overlays represent the model's pixel-wise confidence in turf presence, masking out roads, buildings, and deep shadows).*

#### Overlay Predictions on Unseen Imagery
![External Overlay 1](results/external_preds/external%201_overlay.png)
> *Figure 1: The model successfully maps the boundary edges of the turf while ignoring the concrete pathways and structural shadows.*

![External Overlay 2](results/external_preds/external%202_overlay.png)
> *Figure 2: Complex spatial morphology handling. The pseudo-NDVI channel allows the model to distinguish between the green hues of the grass and the darker greens of surrounding foliage.*

![External Overlay 3](results/external_preds/external%203_overlay.png)
> *Figure 3: Semantic separation in high-density areas. Notice the sharp cutoffs at the property lines and pavement borders.*

---

## 📂 Repository Structure
```text
├── inference.py               # Production CLI for running unseen imagery
├── weights/
│   └── best_model.pt          # Final ResNet50 U-Net weights (Tracked via LFS)
├── results/
│   ├── validation_report.json # Detailed fold-by-fold accuracy logs
│   ├── external_preds/        # Visual overlays and GeoJSON outputs
│   └── train_preds/           # Raw .npy logit arrays (Tracked via LFS)
├── advanced-turf.ipynb        # Core experimental pipeline and model training
└── README.md                  # This documentation

----
### 👨‍💻 Created by [Samarth Arora](https://www.linkedin.com/in/samarth-arora-489b79246/)
*Feel free to reach out if you have any questions about this pipeline!*
