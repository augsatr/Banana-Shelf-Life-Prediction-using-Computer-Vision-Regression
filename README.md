<div align="center">
  <h1>🍌 Banana Shelf-Life Prediction</h1>
  <p><strong>AI-Powered Computer Vision System for Banana Freshness Estimation</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch">
    <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green" alt="OpenCV">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </p>
</div>

---

## 📋 Overview

An end-to-end **Computer Vision + Machine Learning** system that predicts banana ripeness stage and remaining shelf life (in days) from a single image. The system combines **traditional feature engineering** (color, texture, shape) with **deep learning** (CNN, Vision Transformer) and **ensemble regression** (XGBoost, LightGBM, Random Forest) to produce accurate predictions with **uncertainty quantification**.

> **R² = 0.94 | MAE = ~0.5 days | 95% CI** on synthetic evaluation set

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Input Image │───▶│ Preprocessing │───▶│ Feature Pipeline │───▶│   Ensemble   │
│  (224×224)   │    │               │    │                  │    │   Predictor  │
└─────────────┘    └───────────────┘    └──────────────────┘    └──────────────┘
                         │                       │                      │
                         ▼                       ▼                      ▼
                  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
                  │ • Background │    │ • Color (HSV/    │    │ • CNN (Eff-  │
                  │   Removal    │    │   LAB/YCrCb)    │    │   NetV2)    │
                  │ • CLAHE     │    │ • GLCM Texture  │    │ • Vision     │
                  │ • Crop+Resize│   │ • LBP/HOG/Edge  │    │   Transformer│
                  │ • Contour    │    │ • Spots/Blobs   │    │ • Stacked ML │
                  │   Extraction │    │ • Curvature/FD  │    │   Ensemble   │
                  │              │    │ • Morphology    │    │ • SWA + MC   │
                  │              │    │ • Deep Features │    │   Dropout    │
                  └──────────────┘    └──────────────────┘    └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │  Prediction  │
                                                              │ • Shelf Life │
                                                              │ • Stage      │
                                                              │ • 95% CI     │
                                                              │ • Grad-CAM   │
                                                              └──────────────┘
```

---

## 🧪 Ripeness Stages

| Stage | Image Example | Description | Shelf Life |
|-------|-------------|-------------|------------|
| **0 - Green** | ![Green] | Firm, green peel, no spots | 7–10 days |
| **1 - Yellow** | ![Yellow] | Bright yellow, slight give | 3–5 days |
| **2 - Spotted** | ![Spotted] | Yellow with brown speckles | 1–2 days |
| **3 - Brown** | ![Brown] | Dark brown, soft, bruised | <1 day |

<img width="3262" height="1826" alt="Image" src="https://github.com/user-attachments/assets/477f4445-a18a-474b-815b-78c2b2590588" />

<img width="685" height="421" alt="Image" src="https://github.com/user-attachments/assets/35ace905-66f8-4e72-b453-aaa4a892b8b2" />

<img width="974" height="664" alt="Image" src="https://github.com/user-attachments/assets/60504ab7-52d7-44d9-a9e3-cf649e20e45c" />


---

## 🔬 Technical Deep Dive

### 1. Data Pipeline

**Synthetic Banana Generator** using Perlin noise for realistic textures:
- Procedural banana shapes with natural curvature
- HSV-based color transitions across ripeness stages
- Brown spot generation at overripe stages
- Wrinkle lines for spoiled stages
- Stem and tip modeling

**Data Augmentation:**
- `RandAugment`: random rotation, shear, translation, color jitter, sharpness
- `MixUp` / `CutMix`: for CNN training robustness
- `CLAHE`: illumination normalization

### 2. Feature Engineering (234 dimensions)

| Category | Features | Count |
|----------|----------|-------|
| **Color** | HSV/LAB/YCrCb stats (mean, std, percentiles), color ratios, ripening index | 113 |
| **Texture** | GLCM (6 props × 3 distances × 4 angles), LBP (3 radii), HOG (9-bin stats) | 66 |
| **Spots** | Adaptive thresholding, contour analysis, convexity, area distribution | 10 |
| **Shape** | Circularity, solidity, extent, elongation, equivalent diameter | 10 |
| **Curvature** | Mean/std/max curvature, bending energy | 5 |
| **Fourier** | 12 Fourier descriptors from contour | 12 |
| **Dominant Colors** | MiniBatch K-Means (6 clusters × RGB + proportion) | 24 |
| **Deep** | Pretrained EfficientNet/ConvNeXt embeddings (optional) | 128+ |

### 3. Model Zoo

| Model | Type | Key Features |
|-------|------|-------------|
| **EfficientNet-V2** | CNN | Regression + classification + aleatoric uncertainty heads |
| **Vision Transformer** | Transformer | ViT-Base with learned regression head |
| **ConvNeXt** | CNN | Modern convnet with layer norm, GELU |
| **XGBoost** | Gradient Boost | Optuna-optimized hyperparameters |
| **LightGBM** | Gradient Boost | Leaf-wise tree growth with histogram bins |
| **Random Forest** | Bagging | 500 trees with sqrt feature sampling |
| **Advanced Ensemble** | Stacking | RobustScaler + weighted average + Ridge meta-model |

### 4. Training & Optimization

```
Training Pipeline:
1. Synthetic Data Generation (Perlin noise)
2. RandAugment → MixUp/CutMix
3. Model-Specific Training:
   - CNN/ViT: Cosine warmup + SWA + AMP + Early Stopping
   - Ensemble: Optuna hyperparameter search (5 trials/model)
4. Model Averaging → Weighted Stacking
5. Evaluation: 16+ metrics, error analysis, calibration
```

**Loss Function:**
```
L = 0.5 * MSE(shelf_life) + 0.5 * Uncertainty_NLL + 0.3 * CrossEntropy(stage)
```

### 5. Uncertainty Quantification

- **Aleatoric**: Heteroscedastic regression (learned variance head)
- **Epistemic**: Monte Carlo Dropout (30 forward passes)
- **Ensemble**: Standard deviation across component models
- **Output**: 95% prediction intervals via normal approximation

---

## 📊 Performance

| Metric | Ensemble | CNN | ViT |
|--------|----------|-----|-----|
| MAE ↓ | **0.52** | 0.61 | 0.58 |
| RMSE ↓ | **0.73** | 0.84 | 0.80 |
| R² ↑ | **0.94** | 0.91 | 0.92 |
| Within 1 day | **88%** | 83% | 85% |
| Within 2 days | **97%** | 94% | 95% |
| Inference | **4ms** | 12ms | 35ms |

*Evaluated on 500 synthetic samples. Real-world performance may vary.*

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/augsatr/Banana-Shelf-Life-Prediction-using-Computer-Vision-Regression.git
cd Banana-Shelf-Life-Prediction-using-Computer-Vision-Regression

# 2. Install
pip install -r requirements.txt

# 3. Train (generates 2000 synthetic images)
python scripts/train.py --model all --samples 2000

# 4. Predict
python scripts/predict.py path/to/banana.jpg --visualize --gradcam

# 5. Launch Web App
python app.py
# → http://localhost:8000

# 6. Full Evaluation
python scripts/evaluate.py --model ensemble --samples 500 --save metrics.json

# 7. Export to ONNX
python scripts/export.py --model cnn

# 8. Docker
docker-compose up --build
```

---

## 🖥️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check + model metadata |
| `/predict` | POST | Predict from image file |
| `/predict/batch` | POST | Batch prediction (multiple files) |

---

## 📁 Project Structure

```
📦 Banana-Shelf-Life-Prediction
├── 📂 config/
│   ├── config.yaml           # Central configuration
│   └── model/                # Model-specific YAML configs
├── 📂 data/
│   ├── dataset.py            # PyTorch Dataset
│   ├── preprocessing.py      # Background removal, CLAHE, contour
│   ├── augmentations.py      # RandAugment, MixUp, CutMix
│   └── synthetic.py          # Perlin-noise procedural generator
├── 📂 features/
│   ├── extractor.py          # 234-dim feature extraction (10 families)
│   └── deep_features.py      # Deep CNN embeddings
├── 📂 models/
│   ├── cnn_model.py          # EfficientNet/ResNeXt/ConvNeXt
│   ├── vit_model.py          # Vision Transformer
│   ├── ensemble.py           # Stacked ensemble + Optuna
│   ├── uncertainty.py        # MC Dropout, prediction intervals
│   ├── factory.py            # Model creation factory
│   └── train.py              # Trainer (SWA, AMP, early stopping)
├── 📂 explain/
│   ├── gradcam.py            # Grad-CAM visualizations
│   └── shap_explain.py       # SHAP feature importance
├── 📂 serving/
│   ├── inference.py          # Production inference engine
│   └── onnx_export.py        # ONNX export
├── 📂 utils/
│   ├── metrics.py            # 16+ evaluation metrics
│   └── visualization.py      # Training curves, error analysis
├── 📂 scripts/
│   ├── train.py              # Training entry point
│   ├── predict.py            # CLI prediction
│   ├── evaluate.py           # Full evaluation suite
│   └── export.py             # ONNX export
├── 📜 app.py                 # FastAPI web application
├── 📜 Dockerfile             # Docker build
├── 📜 docker-compose.yml     # Docker compose
├── 📜 requirements.txt       # Python dependencies
├── 📜 setup.py               # Package installer
└── 📜 README.md              # This file
```

---

## 🛠️ Tech Stack

```
Computer Vision     │ OpenCV, scikit-image, albumentations
Deep Learning       │ PyTorch, torchvision, transformers (ViT)
ML Models           │ scikit-learn, XGBoost, LightGBM, Optuna
Feature Engineering │ GLCM, LBP, HOG, Fourier Descriptors, K-Means
Explainability      │ Grad-CAM, SHAP
Backend             │ FastAPI, uvicorn, python-multipart
DevOps              │ Docker, docker-compose, ONNX
Visualization       │ Matplotlib, OpenCV
Experiment Tracking │ (Extensible: MLflow/W&B)
```

---

andddd me sohan
