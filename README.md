# ALS Speech Disorder Classification System

A machine learning project that detects Amyotrophic Lateral Sclerosis (ALS) from speech samples using acoustic feature analysis. The project includes multiple model approaches (traditional ML and deep learning) plus a web application for real-time predictions.

## Overview

This system analyzes voice recordings to classify whether a speaker has ALS or is healthy. We extract 115+ acoustic features from audio (pitch, formants, voice quality, etc.) and use trained models to make predictions. The project explores different modeling approaches to find the best accuracy.

## Dataset

**VOC-ALS Dataset**: 763 voice samples from patients with ALS and healthy controls
- **Vowels**: A, E, I, O, U
- **Split**: Training / Validation / Test
- **Features**: Audio spectrograms and extracted acoustic features
- **Metadata**: Age, sex, severity scores (in `VOC-ALS/metadata.csv`)

## Project Structure

### Core Components

- **Feature Extraction** (`extract_features.py`): Extracts 115+ acoustic features using Praat
- **Machine Learning Models**:
  - `train_logistic_regression.py` - Simple baseline model
  - `train_random_forest.py` - Tree-based model with feature importance
  - `train_xgboost.py` - Gradient boosting model
  - `train_svm.py` - Support Vector Machine
  - `train_lr_xgboost_top_features.py` - LR/XGBoost using selected features

- **Deep Learning** (`CNNS/` folder):
  - MobileNetV2 (`mobilenetv2.ipynb`)
  - EfficientNet + ResNet-50 combination (`ellficientnet+Resent.ipynb`)
  - Models trained on spectrograms

- **Web Application** (`app.py`): Flask-based UI for uploading audio and getting predictions

### Datasets & Results

- `dataset/` - Preprocessed train/val/test splits with spectrograms
- `VOC-ALS/` - Raw dataset with voiced segments and spectrograms
- `*_results.csv` - Performance metrics for each model
- `*_features*.csv` - Feature importance and selection results

## Features Extracted

The system extracts several categories of acoustic features:

- **Fundamental Frequency (F0)**: Mean, median, standard deviation, quantiles, range
- **Formants (F1-F5)**: Frequency, bandwidth, slopes
- **Voice Quality**: Jitter, shimmer measurements
- **Harmonicity**: Harmonics-to-noise ratio (HNR)
- **Intensity**: Mean, standard deviation, coefficient of variation
- **MFCCs**: Mel-frequency cepstral coefficients
- **Temporal Features**: Complexity measures from trajectory analysis

## Getting Started

### Requirements

```bash
pip install -r requirements_web.txt
```

Key dependencies: Flask, scikit-learn, XGBoost, PyTorch, Parselmouth, pandas

### Running the Web App

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

**Upload an audio file** → **Click Analyze** → **Get prediction with confidence score**

## Model Performance

Each model has been trained and evaluated. Results saved in:
- `logistic_regression_results.csv`
- `random_forest_results.csv`
- `xgboost_results.csv`
- `svm_rbf_results.csv`
- `lr_xgboost_top_features_results.csv`

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask web application for predictions |
| `extract_features.py` | Acoustic feature extraction pipeline |
| `train_*.py` | Training scripts for different models |
| `feature_extraction_helpers/` | Praat scripts for voiced segment extraction |
| `CNNS/` | Deep learning notebooks and models |
| `VOC-ALS/` | Main dataset directory |

## How It Works

1. **Input**: User uploads a `.wav` audio file
2. **Processing**: Praat extracts 115+ acoustic features in real-time
3. **Prediction**: Pre-trained model classifies as "Disease" or "No Disease"
4. **Output**: Classification result with confidence score

## Limitations

- Audio files must be `.wav` format
- Maximum file size: 50 MB
- Processing time: 10-30 seconds (depends on audio length)
- Model trained on vowel sounds specifically
- Requires good audio quality for accurate feature extraction

## Project Files

- `main.tex` - Academic paper about the project
- `methodology.html` - Detailed methodology documentation
- Jupyter notebooks in `CNNS/` - Deep learning model development

---

**Note**: Pre-trained models must be in the project root directory for the web app to function. The system uses the Logistic Regression model by default for predictions.
