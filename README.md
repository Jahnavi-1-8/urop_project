# Apple Leaf Disease Detection, Lesion Segmentation & Severity Assessment (Phases 1–4)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An end-to-end academic machine learning and phytopathological computer vision project for **4-class apple leaf disease classification**, **complete leaf extraction**, **disease-specific lesion segmentation**, and **foliar severity quantification** built from scratch using the PlantVillage dataset.

---

## Table of Contents

1. [Problem Statement & Background](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Four Target Classes](#3-four-target-classes)
4. [Phase 1: Feature Extraction & Classification Models](#4-phase-1-classification-pipeline)
5. [Phase 2: Complete Leaf Segmentation](#5-phase-2-complete-leaf-segmentation)
6. [Phase 3: Disease Lesion Segmentation](#6-phase-3-disease-lesion-segmentation)
7. [Phase 4: Severity Estimation & Quantification](#7-phase-4-severity-calculation)
8. [End-to-End Integrated Pipeline](#8-end-to-end-integrated-pipeline)
9. [Project Directory Structure](#9-project-directory-structure)
10. [Installation & Setup](#10-installation--setup)
11. [How to Run Training & Inference](#11-how-to-run-training--inference)
12. [How to Run Severity Estimation & CLI](#12-how-to-run-severity-estimation--cli)
13. [How to Launch the Streamlit Web Application](#13-how-to-launch-the-streamlit-web-application)
14. [Testing & Validation](#14-testing--validation)
15. [Limitations](#15-limitations)
16. [Future Work (Phase 5 Roadmap)](#16-future-work)

---

## 1. Problem Statement

Apple trees (*Malus domestica*) are among the most economically significant temperate fruit crops worldwide. However, foliar diseases such as Apple Scab, Black Rot, and Cedar Apple Rust pose severe threats to orchard yield, fruit quality, and tree viability. 

Traditional phytopathological identification relies on manual visual inspection by agricultural experts, which is labor-intensive, subjective, and prone to misdiagnosis during early infection stages. The objective of this project is to develop an automated, reproducible, and interpretable machine learning system capable of classifying digital photographs of apple leaves into one of four diagnostic categories with high precision and confidence calibration.

> **Scope Restriction (Phase 1):** This phase focuses exclusively on **4-class categorical classification** (healthy vs. specific disease types). Severity-level estimation, treatment scheduling, IoT/drone deployment, and mobile hardware integration are explicitly designated for Phase 2.

---

## 2. Dataset Overview

This study utilizes the curated apple leaf subset from the standard **PlantVillage** public repository. 

* **Total Samples:** 3,171 images
* **Image Format:** Standard RGB digital photographs (`.JPG`)
* **Standard Resolution:** Resized to uniform $500 \times 500$ resolution during preprocessing
* **Dataset Partitioning:** Stratified $80/20$ split:
  * **Training Set:** 2,536 samples ($80\%$)
  * **Held-Out Test Set:** 635 samples ($20\%$)
* **Data Leakage Prevention:** Feature scaling (`MinMaxScaler`) is fitted **strictly on the training partition** and transformed onto the held-out test set and unseen inference inputs.

---

## 3. Four Target Classes

The dataset comprises four primary foliar conditions:

| Directory Identifier | Human-Readable Label | Botanical / Pathological Agent | Sample Count | Test Split (20%) |
| :--- | :--- | :--- | :---: | :---: |
| `Apple___Apple_scab` | **Apple Scab** | *Venturia inaequalis* (Ascomycete fungus) | 630 | 126 |
| `Apple___Black_rot` | **Black Rot** | *Diplocarpa mali* / *Botryosphaeria obtusa* | 621 | 124 |
| `Apple___Cedar_apple_rust` | **Cedar Apple Rust** | *Gymnosporangium juniperi-virginianae* | 275 | 55 |
| `Apple___healthy` | **Healthy** | Asymptomatic leaf tissue | 1,645 | 330 |
| **Total** | — | — | **3,171** | **635** |

---

## 4. Image Preprocessing Pipeline

The preprocessing module ([`src/preprocessing.py`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/src/preprocessing.py)) standardizes input images and eliminates background illumination and mounting artifacts via color thresholding:

1. **Image Ingestion & Validation:** Image files are read via OpenCV (`cv2.imread`) with graceful handling of corrupted or missing inputs.
2. **Standardization:** Images are resized to a fixed dimension of $500 \times 500$ pixels using area interpolation.
3. **Color Space Transformation:**
   $$\text{BGR} \longrightarrow \text{RGB} \longrightarrow \text{HSV}$$
   HSV (Hue, Saturation, Value) decouples chromaticity from luminance, providing robustness across varying lighting conditions.
4. **Dual-Mask Color Segmentation:**
   To isolate the entire leaf structure (both healthy chlorophyll and necrotic symptoms) from background noise:
   * **Healthy Foliage Mask:**
     $$\text{Hue} \in [25, 100], \quad \text{Sat} \in [0, 255], \quad \text{Val} \in [20, 255]$$
   * **Diseased Lesion Mask:**
     $$\text{Hue} \in [10, 30], \quad \text{Sat} \in [0, 255], \quad \text{Val} \in [10, 255]$$
   * **Composite Mask:**
     $$M_{\text{final}} = M_{\text{healthy}} \lor M_{\text{diseased}}$$
   * **Segmented Output:**
     $$I_{\text{segmented}} = I_{\text{RGB}} \odot M_{\text{final}}$$
5. **Grayscale Conversion:** Grayscale representation of $I_{\text{segmented}}$ is generated for morphological and texture analysis.

---

## 5. Feature Extraction Methodology

The feature extraction module ([`src/feature_extraction.py`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/src/feature_extraction.py)) combines three complementary global descriptors:

```
  Segmented Apple Leaf
         │
         ├───► Hu Moments (Shape & Morphology) ──────────►  7 features
         │
         ├───► Haralick Texture (GLCM Statistics) ───────► 13 features
         │
         └───► 3D HSV Color Histogram (8x8x8 bins) ──────► 512 features
                                                                │
                                                                ▼
                                      Concatenated Feature Vector: 532 dimensions
```

### 1. Hu Moments (Shape Invariants — 7 Dimensions)
Calculated from the normalized central spatial moments of the grayscale segmented leaf. The 7 orthogonal moment invariants are invariant to translation, scale, and planar rotation, capturing outer leaf morphology and macroscopic lesion boundaries:
$$\phi_1 = \eta_{20} + \eta_{02}$$
$$\phi_2 = (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2, \quad \dots, \quad \phi_7$$

### 2. Haralick Texture Features (GLCM — 13 Dimensions)
Extracted from the Gray-Level Co-occurrence Matrix (GLCM) using `mahotas`. Second-order joint conditional probability distributions are evaluated across four spatial directions ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) with a displacement of 1 pixel and averaged to enforce rotational invariance:
1. Angular Second Moment (Energy / Uniformity)
2. Contrast
3. Correlation
4. Variance
5. Inverse Difference Moment (Homogeneity)
6. Sum Average
7. Sum Variance
8. Sum Entropy
9. Entropy
10. Difference Variance
11. Difference Entropy
12. Information Measure of Correlation 1
13. Information Measure of Correlation 2

### 3. 3D HSV Color Histogram (Color Distribution — 512 Dimensions)
Color distributions are quantized across the Hue, Saturation, and Value channels using $8$ discrete bins per axis ($8 \times 8 \times 8 = 512$ bins). The resulting histogram is normalized using L2 norm to ensure scale invariance against leaf surface area.

### 4. Dynamic Dimension Reporting
The total feature vector is dynamically assembled:
$$\dim(\mathbf{x}) = 512 + 13 + 7 = 532$$
Our code computes and verifies this dimensionality at runtime without hard-coding static constants.

---

## 6. Machine Learning Models

### Model 1 — Random Forest
* **Algorithm:** Bootstrap aggregating ensemble of unpruned decision trees.
* **Hyperparameters:** `n_estimators=100`, `criterion='gini'`, `random_state=42`, `n_jobs=-1`.
* **Strengths:** Robust to high feature dimensionality, invariant to monotonic feature scaling, captures non-linear feature interactions without overfitting.

### Model 2 — Support Vector Machine (SVM)
* **Algorithm:** Soft-margin Support Vector Classifier with Radial Basis Function (RBF) kernel:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
* **Hyperparameters:** `C=10.0`, `kernel='rbf'`, `gamma='scale'`, `probability=True`, `random_state=42`.
* **Feature Scaling:** Sensitive to feature magnitudes; hence, `MinMaxScaler(feature_range=(0, 1))` is fitted **strictly on $X_{\text{train}}$** and transformed onto $X_{\text{test}}$.
* **Probability Calibration:** Employs Platt scaling (logistic sigmoid fitted via cross-validation) to output real, calibrated class probabilities.

---

## 7. Experimental Results & Model Comparison

Both models were trained on 2,536 samples and tested on 635 held-out test samples. The results obtained directly from our implementation are as follows:

### Primary Metric Comparison Table (Held-Out Test Set, $N = 635$)

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **96.22%** | **0.9630** | **0.9582** | **0.9605** | **0.9621** | **0.9622** | **0.9621** |
| **Support Vector Machine (SVM)** | **95.12%** | **0.9530** | **0.9390** | **0.9451** | **0.9520** | **0.9512** | **0.9512** |

*Note: Results saved automatically to [`results/model_comparison.csv`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/model_comparison.csv).*

### Class-by-Class Breakdown (Random Forest vs. SVM)

| Disease Class | Support | RF Precision | RF Recall | RF F1 | SVM Precision | SVM Recall | SVM F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple Scab** | 126 | 0.9355 | 0.9206 | 0.9280 | 0.8846 | 0.9127 | 0.8984 |
| **Black Rot** | 124 | 0.9680 | 0.9758 | 0.9719 | 0.9609 | 0.9919 | 0.9762 |
| **Cedar Apple Rust** | 55 | 0.9815 | 0.9636 | 0.9725 | **1.0000** | 0.8909 | 0.9423 |
| **Healthy** | 330 | 0.9669 | 0.9727 | 0.9698 | 0.9665 | 0.9606 | 0.9635 |

---

## 8. Evaluation Metrics & Analysis

### Why Accuracy Alone is Insufficient
In agricultural datasets, class imbalance is common (Healthy samples represent $51.9\%$ of the dataset, whereas Cedar Apple Rust represents only $8.7\%$). Relying solely on overall accuracy can mask severe false-negative rates in minority disease categories.

* **Macro-Averaging:** Calculates the unweighted mean across all 4 classes, treating each disease equally regardless of frequency.
* **Random Forest Advantage:** Random Forest achieved higher Macro Recall ($95.82\%$ vs. $93.90\%$), demonstrating superior sensitivity to minority classes such as Cedar Apple Rust and Apple Scab.
* **SVM Behavior:** SVM achieved perfect Precision ($1.0000$) on Cedar Apple Rust (zero false positives), but exhibited lower recall ($89.09\%$), misclassifying 6 rust samples as scab/healthy due to decision boundary margin constraints in the 532-dimensional space.

### Visualizations Generated in `results/`
1. [`results/class_distribution.png`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/class_distribution.png): Stratified Train vs. Test sample distribution.
2. [`results/confusion_matrix_rf.png`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/confusion_matrix_rf.png): Random Forest test set confusion matrix.
3. [`results/confusion_matrix_svm.png`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/confusion_matrix_svm.png): SVM test set confusion matrix.
4. [`results/model_comparison.png`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/model_comparison.png): Side-by-side grouped bar chart comparing Accuracy, Precision, Recall, and F1.
5. [`results/severity_samples/`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/severity_samples): High-resolution 4-panel diagnostic reports across all 4 foliar classes.
6. [`results/severity_evaluation_report.txt`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/results/severity_evaluation_report.txt): Validation metrics, Dice scores, and IoU benchmarks across 40 test leaves.

---

## 5. Phase 2: Complete Leaf Segmentation

The complete leaf segmentation module ([`severity/leaf_segmentation.py`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/severity/leaf_segmentation.py)) isolates the entire apple leaf blade and petiole from background artifacts:

1. **Illumination-Invariant HSV Conversion:** Converts RGB input to HSV color space.
2. **Multi-Band Foliar Thresholding:**
   - **Healthy green foliage:** $\text{Hue} \in [22, 95], \text{Sat} \in [20, 255], \text{Val} \in [20, 255]$
   - **Diseased chlorotic / rust / brown tissue:** $\text{Hue} \in [7, 38], \text{Sat} \in [20, 255], \text{Val} \in [15, 255]$
   - **Dark necrotic spots:** $\text{Hue} \in [0, 45], \text{Sat} \in [15, 255], \text{Val} \in [10, 85]$
3. **Morphological Filtering:** Closing with an elliptical structuring element ($11 \times 11$) bridges minor gaps between leaf veins and necrotic edges; opening ($5 \times 5$) suppresses background dust.
4. **Dominant Contour Selection:** Identifies external contours and retains the primary leaf structure and connected segments ($\ge 5\%$ of largest contour area).
5. **Border-Padded Interior Hole-Filling:** Applies morphological flood-fill from border padding $(0, 0)$ to guarantee that **100% of internal lesions** (even if very dark, yellow, or atypical) are captured inside the leaf mask.
6. **Total Leaf Surface Area:** Computes $A_{\text{leaf}} = \sum (M_{\text{leaf}} > 0)$.

---

## 6. Phase 3: Disease Lesion Segmentation

The lesion segmentation module ([`severity/lesion_segmentation.py`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/severity/lesion_segmentation.py)) detects symptomatic foliar damage guided by the predicted disease diagnosis:

* **Apple Scab (*Venturia inaequalis*):** Identifies dark olive-green ($\text{Hue} \in [12, 42], \text{Val} \le 140$) and velvety black necrotic lesions ($\text{Val} \le 75, \text{Hue} \le 50$).
* **Black Rot (*Botryosphaeria obtusa*):** Identifies necrotic dark centers ($\text{Val} \le 80$) and characteristic reddish-brown circular frog-eye margins ($\text{Hue} \in [5, 28], \text{Val} \in [25, 160]$).
* **Cedar Apple Rust (*Gymnosporangium juniperi-virginianae*):** Identifies bright orange aecial pustules ($\text{Hue} \in [8, 28], \text{Sat} \ge 55, \text{Val} \ge 50$) and surrounding chlorotic yellow haloes ($\text{Hue} \in [28, 38], \text{Sat} \ge 65, \text{Val} \ge 65$).
* **Healthy Leaf:** Strictly initialized to zero lesion mask ($A_{\text{lesion}} = 0$).
* **Strict Intersection Guarantee:**
  $$M_{\text{lesion}} = M_{\text{candidate}} \land M_{\text{leaf}}$$
  Background pixels are mathematically prohibited from being counted as infected surface area.

---

## 7. Phase 4: Severity Calculation & Categorization

The severity calculation module ([`severity/severity_calculator.py`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/severity/severity_calculator.py)) computes the percentage of diseased foliar tissue:

$$\text{Severity Percentage} = \left( \frac{\text{Diseased Lesion Area}}{\text{Total Leaf Surface Area}} \right) \times 100$$

### Standard Agronomic Severity Scale:
* **Healthy (0.0%):** Foliage exhibits full chlorophyll integrity; no active lesions detected.
* **Mild (< 15.0%):** Localized early-stage symptoms. Preventative protective fungicide management recommended.
* **Moderate (15.0% to 40.0%):** Established symptomatic infection. Curative fungicide and sanitation/pruning advised.
* **Severe (> 40.0%):** Extensive necrosis and photosynthetic destruction. High risk of premature defoliation; immediate chemical intervention required.

### Edge Case Handling:
* Empty leaf mask ($A_{\text{leaf}} = 0$): Handled gracefully, returns $0.0\%$ and status `empty_leaf_mask` (zero division prevented).
* Asymptomatic / empty lesion mask: Returns $0.0\%$ and category `Healthy` or `Mild`.
* Input validation: Missing or corrupt images return graceful structured error dictionaries.

---

## 8. End-to-End Integrated Pipeline

The integrated pipeline ([`severity/pipeline.py`](file:///c:/Users/manch/Downloads/apple-leaf-disease-classification/severity/pipeline.py)) connects the existing Phase 1 classification models with Phases 2, 3, and 4:

```
Input Apple Leaf Image
         │
         ├───► Phase 1: Feature Extraction & Classification ──► Predicted Disease & Confidence
         │
         ├───► Phase 2: Complete Leaf Segmentation ──────────► Complete Leaf Mask & Leaf Area
         │
         ├───► Phase 3: Disease-Specific Lesion Detection ───► Lesion Mask & Lesion Area
         │
         └───► Phase 4: Severity Quantification ─────────────► Severity % & Category (Mild/Moderate/Severe)
                                                                     │
                                                                     ▼
                                                   Integrated Diagnostic Result & Visualizations
```

---

## 9. Project Directory Structure

```
apple-leaf-disease-classification/
│
├── data/
│   ├── raw/                                # Raw PlantVillage images by class
│   │   ├── Apple___Apple_scab/             # 630 images
│   │   ├── Apple___Black_rot/              # 621 images
│   │   ├── Apple___Cedar_apple_rust/       # 275 images
│   │   └── Apple___healthy/                # 1,645 images
│   └── processed/
│       ├── apple_leaf_features.npz         # Extracted 532-D features & labels
│       ├── dataset_summary.json            # Extraction metadata & telemetry
│       └── split_data.npz                  # Exact reproducible Train/Test splits
│
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── preprocessing.py                    # Resizing, RGB/HSV conversion, segmentation
│   ├── feature_extraction.py               # Hu Moments, Haralick textures, HSV histogram
│   ├── train.py                            # Stratified split, MinMaxScaler, RF & SVM training
│   ├── evaluate.py                         # Evaluation metrics, comparison tables, plots
│   └── predict.py                          # Predictor CLI supporting --severity flag
│
├── severity/                               # Phases 2, 3, 4 Severity Quantification Package
│   ├── __init__.py                         # Package exports
│   ├── leaf_segmentation.py                # Phase 2: Complete Leaf Segmentation & hole-filling
│   ├── lesion_segmentation.py              # Phase 3: Disease-specific lesion segmentation
│   ├── severity_calculator.py              # Phase 4: Mathematical severity % & categories
│   ├── visualization.py                    # 4-Panel diagnostic figures & overlays
│   ├── pipeline.py                         # Integrated end-to-end inference pipeline
│   └── evaluate_segmentation.py            # Comprehensive validation with Dice & IoU
│
├── models/
│   ├── random_forest.joblib                # Serialized Random Forest model
│   ├── svm.joblib                          # Serialized SVM model
│   ├── scaler.joblib                       # MinMaxScaler fitted solely on X_train
│   └── label_encoder.joblib                # Scikit-learn target LabelEncoder
│
├── results/
│   ├── class_distribution.png              # Dataset partition plot
│   ├── confusion_matrix_rf.png             # RF confusion matrix heatmap
│   ├── confusion_matrix_svm.png            # SVM confusion matrix heatmap
│   ├── model_comparison.png                # Comparative metric bar chart
│   ├── model_comparison.csv                # Primary metric tracking table
│   ├── per_class_metrics.csv               # Class-level precision/recall/F1 table
│   ├── evaluation_report.txt               # Phase 1 classification report
│   ├── severity_evaluation_report.txt      # Phase 2-4 validation report (Dice & IoU)
│   └── severity_samples/                   # Diagnostic 4-panel visual reports
│       ├── apple_scab_diagnostic.png
│       ├── black_rot_diagnostic.png
│       ├── cedar_apple_rust_diagnostic.png
│       └── healthy_diagnostic.png
│
├── tests/
│   └── test_severity_pipeline.py           # Unit & integration test suite (9 tests)
│
├── app/
│   ├── __init__.py                         # Application package
│   └── app.py                              # Streamlit interactive web application
│
├── app.py                                  # Root convenience launcher
├── requirements.txt                        # Pinned dependencies
├── README.md                               # Project documentation
└── .gitignore                              # Git exclusion rules
```

---

## 10. Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Step-by-Step Installation

1. **Clone or Navigate to the Repository:**
   ```bash
   cd apple-leaf-disease-classification
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 11. How to Run Feature Extraction & Training

### Run Feature Extraction
Processes all 3,171 images from `data/raw/` and compiles the 532-dimensional feature matrix into `data/processed/apple_leaf_features.npz`:
```bash
python src/feature_extraction.py
```

### Run Model Training
Performs stratified 80/20 train/test splitting, fits the scaler on training data, trains both Random Forest and SVM, and serializes models to `models/`:
```bash
python src/train.py --rf_estimators 100 --svm_c 10.0 --seed 42
```

---

## 12. How to Run Evaluation & Generate Visualizations

Evaluates the saved models on the held-out test partition ($N = 635$) and regenerates comparison tables and publication plots:
```bash
python src/evaluate.py
```

Outputs generated:
* `results/model_comparison.csv`
* `results/per_class_metrics.csv`
* `results/model_comparison.png`
* `results/confusion_matrix_rf.png`
* `results/confusion_matrix_svm.png`
* `results/class_distribution.png`
* `results/evaluation_report.txt`

---

## 13. How to Run Single-Image Prediction & Severity Estimation

### Single-Image Classification
Classify an apple leaf image using the trained Random Forest or SVM model:
```bash
python src/predict.py --image "data/raw/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG" --model rf
```

### Integrated Classification & Severity Assessment
To classify the disease **and** quantify the foliar severity percentage in a single run:
```bash
python src/predict.py --image "data/raw/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG" --severity
```

Sample CLI Output:
```text
========================================================
 APPLE LEAF DISEASE & SEVERITY ASSESSMENT RESULT
========================================================
Model Selected:     Random Forest
Predicted Class:    Cedar Apple Rust
Confidence:         95.00%
--------------------------------------------------------
Leaf Area:          210,953 px
Lesion Area:        76,965 px
Severity:           36.48%
Severity Category:  MODERATE
Clinical Guidance:  Established infection (15% - 40% leaf area). Curative fungicide and sanitation measures advised.

Class Probabilities:
  - Apple Scab         :   0.00% [                              ]
  - Black Rot          :   5.00% [#                             ]
  - Cedar Apple Rust   :  95.00% [############################  ]
  - Healthy            :   0.00% [                              ]

Diagnostic Summary:
  Cedar Apple Rust (Gymnosporangium juniperi-virginianae): Bright orange-yellow rust pustules with small raised fruiting bodies on the upper surface. Caused by fungal heteroecious cycle.
========================================================
```

### Run Standalone Modules

#### Phase 2: Complete Leaf Segmentation
```bash
python severity/leaf_segmentation.py --image "data/raw/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
```

#### Phase 3: Disease-Specific Lesion Segmentation
```bash
python severity/lesion_segmentation.py --image "data/raw/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG" --disease "Apple Scab"
```

#### Phase 4: Severity Calculation
```bash
python severity/severity_calculator.py --image "data/raw/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG" --disease "Apple Scab"
```

#### End-to-End Integrated Pipeline Runner
```bash
python severity/pipeline.py --image "data/raw/Apple___Black_rot/0090d05d-d797-4c99-abd4-3b9cb323a5fd___JR_FrgE.S 8727.JPG"
```

---

## 14. How to Launch the Streamlit Web Application

Launch the interactive web UI via either root `app.py` or `app/app.py`:

```bash
# General
streamlit run app/app.py

# Windows (if streamlit is not in PATH)
python -m streamlit run app/app.py

# Alternatively from repository root
streamlit run app.py
```

### Application Features:
* **Image Upload:** Upload JPG, JPEG, or PNG images.
* **Quick Samples:** Choose pre-loaded samples from each disease category for instant one-click testing.
* **Classifier Switcher:** Toggle between Random Forest and SVM dynamically.
* **Live Probabilities:** Real calibrated class probability distribution chart.
* **Foliar Severity Card:** Live calculation of infected leaf area percentage with color-coded severity badges (Healthy, Mild, Moderate, Severe).
* **4-Column Visual Diagnostic Gallery:** Side-by-side display of Input Leaf, Complete Leaf Mask (Phase 2), Lesion Segmentation Mask (Phase 3), and Infected Region Overlay (Phase 4).
* **Clinical Guidance Callout:** Automated phytopathological treatment advice based on disease and severity.
* **Results Dashboard:** Embedded view of the benchmark metrics, confusion matrices, and class distributions.

---

## 15. Testing & Validation

### Run Automated Unit & Integration Tests
Execute the 9-test test suite covering all modules and edge cases:
```bash
python -m unittest tests/test_severity_pipeline.py
```

### Run Comprehensive Validation & Metric Benchmarks
Benchmark leaf and lesion segmentation across 40 test leaves and generate Dice / IoU scores:
```bash
python severity/evaluate_segmentation.py
```

Key empirical benchmarks from test evaluation:
* **Apple Scab:** Mean Severity = $20.11\%$, Segmentation Dice = $0.8421$, IoU = $0.7304$
* **Black Rot:** Mean Severity = $18.19\%$, Segmentation Dice = $0.8984$, IoU = $0.8199$
* **Cedar Apple Rust:** Mean Severity = $22.95\%$, Segmentation Dice = $0.6433$, IoU = $0.4919$
* **Healthy:** Mean Severity = $0.00\%$, Segmentation Dice = $1.0000$, IoU = $1.0000$ (0 false positive lesion pixels)

---

## 16. Limitations

While achieving high classification accuracy ($>95\%$) and robust segmentation on the benchmark:

1. **Controlled Photography Conditions:** PlantVillage images were captured against uniform laboratory backgrounds under controlled lighting. In-field images with complex canopy backgrounds, harsh shadows, or motion blur may require deep semantic segmentation (e.g., U-Net or Mask R-CNN).
2. **Color-Based Lesion Segmentation:** Color thresholding in HSV space is sensitive to extreme lighting shifts, camera white balance variation, and non-pathological blemishes (e.g., dust, spray residue, sun scald).
3. **Single-Leaf Assumption:** The pipeline assumes a single, centered leaf per image; it does not perform multi-leaf object detection or bounding box localization.

---

## 17. Future Work (Phase 5 Roadmap)

1. **Phase 5: Agentic AI Integration:** Implement an autonomous phytopathological diagnostic agent capable of conversational leaf health inquiries, treatment scheduling, weather integration, and agronomic recommendations.
2. **Deep Learning Benchmarking:** Benchmark handcrafted ML models against Vision Transformers (ViT) and CNNs (EfficientNet-B0, ConvNeXt).
3. **In-the-Wild Field Dataset Evaluation:** Expand testing on field datasets such as the FGVC Plant Pathology challenge.
4. **Edge Deployment:** Quantize models for offline edge deployment on agricultural IoT devices (Raspberry Pi, ESP32-CAM) and mobile field scouting apps.
#   u r o p _ p r o j e c t  
 