# Infant State Recognition System (Audio-Based)

## Problem Statement
Developing an automated system to classify infant states based on audio signals (crying, discomfort, hunger, sleeping). Identifying infant needs accurately can assist parents and caregivers by providing immediate feedback using sound analysis.

## Objectives
1. Build an end-to-end pipeline handling audio data.
2. Extract meaningful audio features (MFCC, Spectrograms, ZCR, Chroma).
3. Train and evaluate Baseline (Logistic Regression, Decision Tree), Advanced ML (Random Forest, SVM), and Deep Learning (CNN) models.
4. Provide an edge-ready TFLite model and a simple web-based inference UI.

## Methodology & Architecture
1. **Data Handling**: A script generates dummy audio data for different classes to simulate a dataset. The system can easily be adapted for real `.wav` files by placing them in `data/raw/`.
2. **Preprocessing**: Noise addition, normalisation, and pitch shifting using `librosa`.
3. **Feature Engineering**: Extraction of 1D features (MFCC, ZCR, Chroma) for ML models and 2D Spectrograms for CNNs.
4. **Modeling**: 
   - **Baseline**: Logistic Regression and Decision Tree
   - **Advanced ML**: Random Forest with hyperparameter tuning, SVM
   - **Deep Learning**: CNN built with TensorFlow using Mel Spectrograms.
5. **Deployment**: Flask API to handle file uploads and return predictions.

## Project Structure
```text
infant-state-recognition/
├── app/
│   ├── app.py                     # Flask application
│   └── templates/
│       └── index.html             # Frontend UI
├── data/
│   ├── processed/                 # Store processed files
│   └── raw/                       # Raw datasets
├── models/                        # Saved models (.pkl, .h5, .tflite)
├── notebooks/                     # Interactive Jupyter notebooks for EDA and modeling
├── outputs/                       # Metrics, confusion matrices, and uploaded audio
├── src/
│   ├── evaluate.py                # Evaluation metrics and plotting
│   ├── feature_engineering.py     # Feature extraction (MFCC, Spectrogram, etc.)
│   ├── preprocessing.py           # Denoise, normalize, augment audio
│   ├── train_baseline.py          # Train Logistic Regression / Decision Tree
│   ├── train_dl.py                # Train CNN model
│   └── train_ml.py                # Train Random Forest / SVM
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── create_notebooks.py            # Generates the Jupyter notebooks
└── generate_dummy_data.py         # Synthesizes test dataset
```

## How to Run the Project (Step-by-Step)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Since no real dataset is provided, run this to generate dummy wav files for testing the pipeline:
```bash
python generate_dummy_data.py
```

### 3. Generate Interactive Notebooks
```bash
python create_notebooks.py
```
You can view them in the `notebooks/` directory.

### 4. Train Models
Run the training scripts to fit and evaluate models. They will output metrics to the console and save plots in `outputs/`.
```bash
python src/train_baseline.py
python src/train_ml.py
python src/train_dl.py
```

### 5. Run the Web Application
```bash
python app/app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`. You can upload a generated `.wav` file from `data/raw/crying/` to test the prediction.

## Model Comparison

| Model | Architecture | Speed | Memory/Size | Interpretability |
|-------|--------------|-------|-------------|------------------|
| Logistic Regression | Linear | Very Fast | Very Low | High |
| Decision Tree | Tree | Very Fast | Very Low | High |
| Random Forest | Ensemble Trees | Fast | Low | Medium |
| SVM | RBF Kernel | Medium | Medium | Low |
| CNN | Conv2D Layers | Slow (Train) | Mod/High | None |

## Future Improvements
- Train on large real-world datasets (e.g. Donate-a-cry dataset).
- Add robust noise cancellation using Deep Learning.
- Implement streaming real-time prediction using WebSockets and the microphone.
- Deploy the `.tflite` model directly on Edge devices (mobile app or Raspberry Pi).
