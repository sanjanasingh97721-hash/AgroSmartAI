"""
AgroSmart AI — Model Training Script
=====================================
Trains a Random Forest classifier on the Crop Recommendation dataset.
Saves the trained model as model/crop_model.pkl.

Dataset  : Crop_recommendation.csv  (2,200 records x 8 columns)
Features : N, P, K, temperature, humidity, ph, rainfall
Target   : label (22 crop classes)
Accuracy : 99.32% on 20% holdout test set
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

os.makedirs("model",  exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ── 1. Load Dataset ───────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("Crop_recommendation.csv")
print(f"  Shape   : {df.shape}")
print(f"  Classes : {df['label'].nunique()} crops")
print(f"  Missing : {df.isnull().sum().sum()}")

# ── 2. Feature / Target Split ─────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]
y = df["label"]

# ── 3. 80:20 Stratified Train-Test Split ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

# ── 4. Train Random Forest ────────────────────────────────────
print("\nTraining Random Forest Classifier (100 estimators)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred   = model.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nTest Accuracy  : {acc * 100:.2f}%")
print(f"Macro F1 Score : {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 6. Feature Importances ────────────────────────────────────
fi = dict(zip(FEATURES, model.feature_importances_.round(4)))
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
print("Feature Importances (MDI):")
for feat, imp in fi_sorted.items():
    bar = "=" * int(imp * 40)
    print(f"  {feat:12s}  {imp:.4f}  [{bar}]")

# ── 7. Save Model ─────────────────────────────────────────────
with open("model/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved -> model/crop_model.pkl")

# ── 8. Confusion Matrix Plot ──────────────────────────────────
plt.figure(figsize=(14, 11))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlGn",
    xticklabels=model.classes_, yticklabels=model.classes_,
    linewidths=0.5, linecolor="#ccc"
)
plt.title("Confusion Matrix - Random Forest (99.32% Accuracy)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> assets/confusion_matrix.png")

# ── 9. Feature Importance Bar Chart ──────────────────────────
plt.figure(figsize=(8, 5))
colors = ["#218c45" if i == 0 else "#5cb85c" for i in range(len(fi_sorted))]
plt.barh(list(fi_sorted.keys())[::-1], list(fi_sorted.values())[::-1], color=colors[::-1])
plt.xlabel("Mean Decrease in Impurity (MDI)")
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> assets/feature_importance.png")

print("\nDone.")
