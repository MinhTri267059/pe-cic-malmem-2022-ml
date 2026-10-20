"""
Demo quy trình phát hiện mã độc (Scenario 1 - Binary: Benign vs Malware).
Dùng model XGBoost đã train trong Models/Scenario_1/XGBoost.pkl.

Xuất ra:
  - demo_outputs/confusion_matrix.png
  - demo_outputs/roc_curve.png
  - demo_outputs/sample_predictions.csv (vài mẫu ngẫu nhiên + xác suất dự đoán)
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCENARIO = 1
MODEL_PATH = f"Models/Scenario_{SCENARIO}/XGBoost.pkl"
LABEL_MAP_PATH = f"Models/Scenario_{SCENARIO}/label_mapping.json"
OUT_DIR = "demo_outputs"


def load_data():
    df = pd.read_csv("MalMem2022.csv").drop_duplicates()
    y = df["Class"]
    X = df.drop(columns=[c for c in ["Filename", "Class", "Category"] if c in df.columns])

    with open(LABEL_MAP_PATH) as f:
        label_mapping = json.load(f)
    y_mapped = y.map(label_mapping)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # ponytail: cùng random_state/test_size dùng khi train, để tái lập đúng tập test gốc
    _, X_test, _, y_test = train_test_split(
        X_scaled, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    return X_test, y_test, inv_label_mapping


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    X_test, y_test, inv_label_mapping = load_data()
    class_names = [inv_label_mapping[i] for i in sorted(inv_label_mapping)]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - XGBoost (Binary: Benign vs Malware)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba, name="XGBoost")
    plt.title("ROC Curve - XGBoost (Binary: Benign vs Malware)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # Vài mẫu dự đoán minh họa (demo "soi 1 bản ghi")
    rng = np.random.default_rng(42)
    pos = rng.choice(len(X_test), size=10, replace=False)
    sample = X_test.iloc[pos].copy()
    sample["True_Label"] = [inv_label_mapping[v] for v in y_test.iloc[pos]]
    sample["Predicted_Label"] = [inv_label_mapping[v] for v in model.predict(X_test.iloc[pos])]
    sample["Malware_Probability"] = model.predict_proba(X_test.iloc[pos])[:, 1]
    sample[["True_Label", "Predicted_Label", "Malware_Probability"]].to_csv(
        f"{OUT_DIR}/sample_predictions.csv"
    )

    print(f"Đã lưu: {OUT_DIR}/confusion_matrix.png, roc_curve.png, sample_predictions.csv")
    print(sample[["True_Label", "Predicted_Label", "Malware_Probability"]])


if __name__ == "__main__":
    main()
