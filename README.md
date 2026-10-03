# 🌾 Smart Crop Recommendation System — AgroSmart AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.32%25-brightgreen)](#results)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **An intelligent agricultural decision-support system** that recommends the most suitable crop for a given set of soil and climatic conditions using a Random Forest classifier trained on 2,200 records across 22 crop classes.

---

## 📸 Demo

| Input Panel | Recommendation Output |
|---|---|
| Enter 7 soil & climate parameters | Instant crop prediction with confidence scores |

> 🚀 **Live App:** Run locally with `streamlit run app.py`

---

## 🧠 Problem Statement

Farmers in developing countries like India often rely on traditional knowledge and guesswork when deciding which crop to grow. This leads to sub-optimal yields and economic losses. AgroSmart AI provides a **data-driven recommendation** using machine learning.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Records | 2,200 |
| Features | 7 continuous (N, P, K, temperature, humidity, pH, rainfall) |
| Classes | 22 crops |
| Missing Values | 0 |

**Features:**

| Feature | Description | Unit |
|---|---|---|
| N | Nitrogen content in soil | mg/kg |
| P | Phosphorus content in soil | mg/kg |
| K | Potassium content in soil | mg/kg |
| temperature | Average ambient temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH level | 0–14 |
| rainfall | Average annual rainfall | mm |

---

## 🌲 Model Architecture

```
Random Forest Classifier
├── n_estimators : 100 trees
├── random_state : 42
├── n_jobs       : -1  (parallel)
└── Split        : 80:20 stratified train-test
```

### Why Random Forest?
- Handles non-linear relationships between soil/climate features
- Robust to outliers (real-world sensor noise)
- Provides feature importances (interpretability)
- No need for feature scaling
- Naturally handles multi-class classification

---

## 📈 Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **99.32%** |
| Macro F1 Score | 0.9926 |
| Macro Precision | 0.9926 |
| Macro Recall | 0.9933 |
| Train samples | 1,760 (80%) |
| Test samples | 440 (20%) |

### Feature Importances (Mean Decrease in Impurity)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Rainfall | 22.7% |
| 2 | Humidity | 21.1% |
| 3 | Potassium (K) | 18.1% |
| 4 | Phosphorus (P) | 14.4% |
| 5 | Nitrogen (N) | 10.9% |
| 6 | Temperature | 7.6% |
| 7 | Soil pH | 5.2% |

---

## 🗂️ Project Structure

```
smart-crop-recommendation/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── Crop_recommendation.csv     # Dataset
├── requirements.txt            # Python dependencies
│
├── model/
│   └── crop_model.pkl          # Trained Random Forest model
│
├── assets/
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   └── feature_importance.png  # Feature importance chart
│
└── notebooks/
    └── EDA_and_Modelling.ipynb # Exploratory data analysis notebook
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/smart-crop-recommendation.git
cd smart-crop-recommendation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (or use pre-trained)
```bash
python train_model.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🌾 Supported Crops (22 Classes)

Apple · Banana · Blackgram · Chickpea · Coconut · Coffee · Cotton · Grapes · Jute · Kidney Beans · Lentil · Maize · Mango · Moth Beans · Mung Bean · Muskmelon · Orange · Papaya · Pigeon Peas · Pomegranate · Rice · Watermelon

---

## 🔬 Methodology

```
Raw Data (2,200 records)
    │
    ├─► Exploratory Data Analysis
    │       Distribution plots, correlation heatmap, class balance
    │
    ├─► Data Preprocessing
    │       Feature/target separation, 80:20 stratified split
    │
    ├─► Model Training
    │       Random Forest (100 trees, random_state=42)
    │
    ├─► Evaluation
    │       Accuracy, F1, Precision, Recall, Confusion Matrix
    │
    └─► Deployment
            Streamlit web app with real-time inference
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Library | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Pickle |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Sanjana Singh**  
B.Tech — Data Science, 4th Year  
Shri Ramswaroop Memorial University, Lucknow  
Data Science Internship Project — EISystems Technologies, 2026

---

*Built as part of a 6-week Data Science Industrial Internship at EISystems Technologies, NCR.*
