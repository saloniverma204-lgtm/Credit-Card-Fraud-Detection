
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset loaded successfully!")
print("Total transactions:", len(df))

# ==========================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("Model training completed!")

# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# Fraud probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_probability)
avg_precision = average_precision_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)
print("Average Precision:", avg_precision)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/model.joblib")

# ==========================================
# 9. SAVE MODEL METRICS
# ==========================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
    "confusion_matrix": cm.tolist()
}

with open("reports/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved in models/model.joblib")
print("Metrics saved in reports/metrics.json")
print("\nTraining process completed successfully!")
