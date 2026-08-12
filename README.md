# Interpretable Feature Selection for Speech Emotion Recognition in New Zealand English

This repository contains the code and notebooks for our **Speech Emotion Recognition (SER)** project using the **JL Corpus**, a New Zealand English speech corpus.

The main focus of the project was not just to train a classifier, but to understand which acoustic features are useful for recognizing emotions. We extracted a combination of traditional speech features, used **Recursive Feature Elimination (RFE)** for feature selection, optimized a **Random Forest** classifier using **GridSearchCV**, and used **SHAP** to make the final model more interpretable.

The work was presented at **ICTCS 2025** and published as a Springer book chapter:

> **Sidharrth, V., Pranav, A., Khanna, M. (2026). Interpretable Feature Selection for Speech Emotion Recognition in New Zealand English with Grid Search Optimization for Effective Hyperparameter Tuning.**
>
> In: Joshi, A., Ragel, R.G., S., K. (eds) *ICT: Applications and Social Interfaces. ICTCS 2025*. Lecture Notes in Networks and Systems, vol. 1873. Springer, Cham.
>
> DOI: **10.1007/978-3-032-19675-0_18**

---

## What We Worked On

The overall pipeline was:

```text
JL Corpus
    │
    ▼
Audio Files
    │
    ▼
Emotion Label Extraction
    │
    ▼
Acoustic Feature Extraction
    │
    ├── MFCC
    ├── Delta MFCC
    ├── Delta-Delta MFCC
    ├── Chroma
    ├── Spectral Centroid
    ├── Spectral Rolloff
    ├── Spectral Bandwidth
    ├── Spectral Contrast
    ├── Pitch
    ├── Tempo
    ├── Zero Crossing Rate
    ├── RMS Energy
    ├── Jitter
    ├── Shimmer
    └── HNR
    │
    ▼
132-dimensional Feature Vector
    │
    ├───────────────► t-SNE Visualization
    │
    ▼
Train/Test Split
    │
    ▼
Random Forest + GridSearchCV
    │
    ▼
Recursive Feature Elimination (RFE)
    │
    ▼
Selected Feature Subsets
    │
    ▼
Random Forest Classification
    │
    ├── Accuracy
    ├── Classification Report
    ├── Confusion Matrix
    └── Feature Importance
    │
    ▼
SHAP Explainability
```

---

# Dataset

We used the **JL Corpus**, downloaded through Kaggle using:

```python
kagglehub.dataset_download("tli725/jl-corpus")
```

The notebooks read the `.wav` files from the JL Corpus and obtain the emotion label from the filename.

The label extraction used:

```python
label = file.split("_")[1]
```

The labels were then converted to numerical values using `LabelEncoder`.

The exact class distribution and dataset statistics can be reproduced using the feature-extraction notebook.

---

# Feature Extraction

A major part of this project was extracting a reasonably broad set of acoustic features before doing feature selection.

The final feature vector contains **132 features**.

## 1. MFCC

We extracted **13 MFCC coefficients**, and for each coefficient we calculated:

- Mean
- Standard deviation

This gives:

```text
13 × 2 = 26 features
```

MFCCs are commonly used in speech processing because they provide a compact representation of the spectral characteristics of speech.

---

## 2. Delta MFCC

We calculated the first-order temporal derivatives of the MFCCs.

For each of the 13 MFCCs:

- Delta mean
- Delta standard deviation

This gives another:

```text
13 × 2 = 26 features
```

These features capture how the speech characteristics change over time.

---

## 3. Delta-Delta MFCC

We also calculated second-order derivatives of the MFCCs.

Again, for all 13 coefficients:

- Delta-delta mean
- Delta-delta standard deviation

This adds:

```text
13 × 2 = 26 features
```

So MFCC + delta + delta-delta provide:

```text
26 + 26 + 26 = 78 features
```

---

## 4. Chroma

We extracted 12 chroma features and calculated:

- Mean
- Standard deviation

giving:

```text
12 × 2 = 24 features
```

Chroma features describe the distribution of energy across the 12 pitch classes.

---

## 5. Spectral Features

We extracted several spectral characteristics.

### Spectral Centroid

The spectral centroid provides information about where the center of spectral energy lies.

We used:

- Mean
- Standard deviation

### Spectral Rolloff

Spectral rolloff describes the frequency below which a specified percentage of the spectral energy is concentrated.

We used:

- Mean
- Standard deviation

### Spectral Bandwidth

Spectral bandwidth measures the spread of the frequency distribution around the spectral centroid.

We used:

- Mean
- Standard deviation

### Spectral Contrast

We extracted 7 spectral contrast bands and calculated:

- Mean
- Standard deviation

These features provide additional information about the structure of the speech spectrum.

---

# Pitch and Prosodic Features

We also extracted features related to pitch and speech dynamics.

### Pitch

Pitch values were obtained using `librosa.piptrack()`.

We used:

- Mean pitch
- Standard deviation of pitch

### Tempo

The estimated tempo was also included as a feature.

### Zero Crossing Rate

We calculated:

- Mean ZCR
- Standard deviation of ZCR

ZCR can provide information related to the frequency characteristics of the speech signal.

### RMS Energy

We calculated:

- Mean RMS energy
- Standard deviation of RMS energy

This gives information about the overall energy/intensity of the speech signal.

---

# Voice Quality Features

We also included three voice-quality-related features using **OpenSMILE eGeMAPSv02**:

- Jitter
- Shimmer
- Harmonics-to-Noise Ratio (HNR)

Specifically, the notebook extracts:

```text
jitterLocal_sma3nz_amean
shimmerLocaldB_sma3nz_amean
HNRdBACF_sma3nz_amean
```

These features were included because changes in voice quality can be relevant to emotional speech.

---

# Final Feature Vector

The complete feature vector has:

```text
132 features
```

The feature names are generated in the notebook so that every selected feature can later be identified.

Examples include:

```text
mfcc_mean_0
mfcc_std_0
delta_mean_0
delta_std_0
delta2_mean_0
chroma_mean_0
spectral_centroid_mean
spectral_rolloff_mean
pitch_mean
tempo
zcr_mean
rms_mean
jitter_local_mean
shimmer_localdB_mean
hnr_dBACF_mean
```

---

# Exploratory Analysis

Before classification, we also performed some basic analysis of the dataset and extracted features.

The notebook includes:

- Sample counts per emotion
- Class-distribution visualization
- Audio duration statistics
- Example waveforms for each emotion
- t-SNE visualization of the extracted features

For t-SNE, the features are first standardized using `StandardScaler`.

```text
132-dimensional features
        ↓
StandardScaler
        ↓
t-SNE
        ↓
2-dimensional representation
        ↓
Visualization by emotion
```

This gives a visual way to inspect whether different emotion classes form distinguishable regions in the feature space.

---

# Random Forest Classifier

We used **Random Forest** as the main classification model.

Random Forest combines multiple decision trees and aggregates their predictions.

A simplified view is:

```text
                 Feature Vector
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Tree 1        Tree 2       Tree N
          │            │            │
          ▼            ▼            ▼
      Prediction   Prediction   Prediction
          └────────────┼────────────┘
                       ▼
                 Final Prediction
```

We selected Random Forest because it works well with tabular feature representations and also provides feature-importance information that can be useful for interpretation.

---

# Grid Search for Hyperparameter Tuning

Instead of using the default Random Forest configuration, we performed **GridSearchCV** to find a better combination of hyperparameters.

The search included:

```python
n_estimators:
[100, 200, 300, 400, 500]

max_depth:
[10, 20, 30, 40]

min_samples_split:
[2, 4, 5, 10]

min_samples_leaf:
[1, 2, 4, 8]
```

The search used:

```python
scoring='accuracy'
n_jobs=-1
```

The best configuration obtained from GridSearchCV was then used in the later RFE experiments.

The final Random Forest configuration used in the RFE pipeline was:

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

---

# Recursive Feature Elimination (RFE)

One of the main contributions of the project was feature selection.

Instead of automatically using all 132 features, we used **Recursive Feature Elimination (RFE)** to investigate different feature subsets.

The basic idea is:

```text
132 Features
     │
     ▼
Train Random Forest
     │
     ▼
Rank Features
     │
     ▼
Remove Less Important Features
     │
     ▼
Repeat
     │
     ▼
Selected Feature Subset
```

We tested:

```text
30 features
50 features
75 features
80 features
100 features
132 features
```

This allowed us to study how classification accuracy changes as the number of selected features changes.

---

# RFE Results

The notebook contains the following recorded accuracy values:

| Number of Features | Accuracy |
|---:|---:|
| 30 | 72.50% |
| 50 | 72.08% |
| 75 | 73.54% |
| **80** | **74.17%** |
| 100 | 71.04% |
| 132 | 73.13% |

The best result among the tested feature subsets was obtained using **80 selected features**, with an accuracy of:

> **74.17%**

This was one of the main findings of our feature-selection experiment.

Rather than assuming that more features would automatically produce better classification, we compared different feature subset sizes and found that the 80-feature configuration performed best among the tested settings.

---

# Why Feature Selection?

The purpose of RFE in this project was not simply to reduce the number of features.

We also wanted to understand:

- Which acoustic features contribute to emotion recognition?
- Does removing features improve classification?
- How does performance change with different feature subset sizes?
- Can a smaller feature representation provide competitive performance?

This makes the feature-selection stage important for the interpretability aspect of the work.

---

# Feature Importance

After selecting the features using RFE and training the Random Forest, we also plotted the Random Forest feature importances.

This allows us to inspect which of the selected acoustic features contributed more strongly to the model's decisions.

The repository's RFE pipeline generates feature-importance plots for each tested feature count.

---

# SHAP Explainability

We also used **SHAP (SHapley Additive exPlanations)** to further investigate the model.

The RFE pipeline uses:

```python
shap.TreeExplainer(rf)
```

because Random Forest is a tree-based model.

Two types of SHAP visualizations are generated:

1. SHAP feature-importance bar plot
2. SHAP summary plot

The idea is to move beyond:

```text
"Accuracy = 74.17%"
```

and investigate:

```text
Which features influenced the model?
How strongly did they influence predictions?
How are the selected acoustic features related to the predictions?
```

This is particularly important for a speech-emotion recognition system because the model's decisions should ideally be understandable rather than treated as a black box.

---

# Final Pipeline

The complete experimental pipeline can be summarized as:

```text
              JL Corpus
                  │
                  ▼
             WAV Files
                  │
                  ▼
          Emotion Extraction
                  │
                  ▼
        Acoustic Feature Extraction
                  │
        ┌─────────┴─────────┐
        │                   │
     librosa             OpenSMILE
        │                   │
        └─────────┬─────────┘
                  ▼
          132 Features
                  │
                  ▼
            t-SNE Analysis
                  │
                  ▼
          Train/Test Split
            80 / 20
                  │
                  ▼
          GridSearchCV
                  │
                  ▼
       Optimized Random Forest
                  │
                  ▼
                 RFE
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
       30        80         132
    features  features   features
        │         │          │
        └─────────┼──────────┘
                  ▼
          Model Evaluation
                  │
        ┌─────────┼─────────────┐
        ▼         ▼             ▼
    Accuracy   Confusion      Feature
               Matrix         Importance
                                │
                                ▼
                               SHAP
```

---

# Train/Test Split

The data was divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

This means:

- 80% of the data was used for training.
- 20% was used for testing.
- Stratification was used to preserve the class distribution.
- `random_state=42` was used for reproducibility.

---

# Repository Files

The repository currently contains the two main notebooks used for the work.

```text
Speech-Emotion-Recognition-JL-Corpus/
│
├── featureextraction&gridsearch1.ipynb
│
└── RFEPipeline.ipynb
```

## `featureextraction&gridsearch1.ipynb`

This notebook contains the main feature-engineering and hyperparameter-tuning pipeline.

It includes:

- Dataset download
- Dataset inspection
- Emotion label extraction
- Class distribution analysis
- Waveform visualization
- Duration analysis
- Acoustic feature extraction
- OpenSMILE feature extraction
- 132-dimensional feature creation
- t-SNE visualization
- Random Forest setup
- GridSearchCV
- Saving the feature matrix and labels

The extracted arrays are saved as:

```text
X_features.pkl
y_labels.pkl
```

---

## `RFEPipeline.ipynb`

This notebook contains the feature-selection and interpretation stage.

It includes:

- Loading the extracted features
- Train/test split
- Feature-name generation
- RFE
- Random Forest classification
- Accuracy evaluation
- Classification reports
- Confusion matrices
- t-SNE visualization after feature selection
- Random Forest feature importance
- SHAP analysis
- Accuracy comparison for different feature counts

The feature counts evaluated were:

```text
30, 50, 75, 80, 100, 132
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/VSidharrth/Speech-Emotion-Recognition-JL-Corpus.git
cd Speech-Emotion-Recognition-JL-Corpus
```

Install the main Python packages used in the notebooks:

```bash
pip install numpy pandas librosa matplotlib seaborn scikit-learn
pip install opensmile shap joblib kagglehub tqdm
```

The notebooks were developed in a Google Colab-style environment, so some cells contain:

```python
!pip install ...
```

and:

```python
from google.colab import files
```

If running locally, those Colab-specific cells may need to be changed.

---

# Running the Project

### Step 1 — Feature Extraction

Open:

```text
featureextraction&gridsearch1.ipynb
```

Run the notebook to:

1. Download the JL Corpus.
2. Read the audio files.
3. Extract emotion labels.
4. Extract the 132 acoustic features.
5. Create the feature matrix.
6. Run exploratory analysis.
7. Perform GridSearchCV.
8. Save the features and labels.

The notebook saves:

```text
X_features.pkl
y_labels.pkl
```

---

### Step 2 — RFE and Classification

Open:

```text
RFEPipeline.ipynb
```

Load the saved feature and label files and run the RFE pipeline.

The notebook evaluates:

```text
30 features
50 features
75 features
80 features
100 features
132 features
```

and generates the corresponding:

- Accuracy
- Classification report
- Confusion matrix
- t-SNE visualization
- Feature importance
- SHAP plots

---

# Results

The best recorded result in the repository is:

### Random Forest + RFE

**80 selected features**

**Accuracy: 74.17%**

The complete comparison recorded in the notebook is:

```text
30 features   → 72.50%
50 features   → 72.08%
75 features   → 73.54%
80 features   → 74.17%
100 features  → 71.04%
132 features  → 73.13%
```
---

# Authors
**V. Sidharrth**  
**Amara Pranav**  
---

**Year:** 2026
