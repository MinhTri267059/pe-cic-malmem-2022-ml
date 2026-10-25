"""
Demo quy trình phát hiện mã độc (Scenario 1 - Binary: Benign vs Malware).
Dùng model XGBoost đã train trong Models/Scenario_1/XGBoost.pkl.

Xuất ra:
  - demo_outputs/confusion_matrix.png
  - demo_outputs/roc_curve.png
  - demo_outputs/sample_predictions.csv (vài mẫu ngẫu nhiên + xác suất dự đoán)
  - demo_outputs/sample_card_benign.png, sample_card_malware.png
    (tái hiện nội dung dashboard app_demo.py cho 1 ca Benign và 1 ca Malware,
     dùng khi không chụp được màn hình trình duyệt)
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

SCENARIO = 1
MODEL_PATH = f"Models/Scenario_{SCENARIO}/XGBoost.pkl"
SCALER_PATH = f"Models/Scenario_{SCENARIO}/XGBoost_scaler.pkl"
LABEL_MAP_PATH = f"Models/Scenario_{SCENARIO}/label_mapping.json"
OUT_DIR = "demo_outputs"

TOP_FEATURES = [
    "dlllist.avg_dlls_per_proc",
    "handles.nevent",
    "handles.nthread",
    "handles.nmutant",
    "pslist.nppid",
    "malfind.commitCharge",
    "malfind.ninjections",
]


def load_data():
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv("MalMem2022.csv").drop_duplicates().reset_index(drop=True)
    y = df["Class"]
    X = df.drop(columns=[c for c in ["Filename", "Class", "Category"] if c in df.columns])

    with open(LABEL_MAP_PATH) as f:
        label_mapping = json.load(f)
    y_mapped = y.map(label_mapping)

    # ponytail: cùng random_state/test_size dùng khi train, để tái lập đúng tập test gốc;
    # split trên X thô rồi transform bằng scaler đã fit lúc train (chỉ trên X_train gốc)
    _, X_test_raw, _, y_test = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns, index=X_test_raw.index)
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    return X_test, y_test, X_test_raw, df, inv_label_mapping


def plot_sample_card(model, x_sample, raw_sample, true_label, inv_label_mapping, means, out_path):
    """Tái hiện 1 'thẻ chi tiết mẫu' giống dashboard app_demo.py: nhãn + xác suất + so sánh đặc trưng."""
    proba = model.predict_proba(x_sample)[0]
    malware_idx = [k for k, v in inv_label_mapping.items() if v == "Malware"][0]
    pred_label = inv_label_mapping[int(np.argmax(proba))]
    malware_proba = proba[malware_idx]

    fig, (ax_info, ax_bar) = plt.subplots(
        2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1, 2]}
    )

    ax_info.axis("off")
    status = "Đúng" if pred_label == true_label else "Sai"
    ax_info.text(0.0, 0.8, f"Nhãn thật: {true_label}", fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.0, 0.5, f"Dự đoán: {pred_label} ({status})", fontsize=14, transform=ax_info.transAxes)
    ax_info.text(
        0.0, 0.2, f"Xác suất Malware: {malware_proba*100:.2f}%", fontsize=14,
        transform=ax_info.transAxes, color="crimson" if malware_proba > 0.5 else "seagreen",
    )

    x = np.arange(len(TOP_FEATURES))
    width = 0.25
    ax_bar.bar(x - width, raw_sample[TOP_FEATURES].values, width, label="Mẫu hiện tại", color="#1f77b4")
    ax_bar.bar(x, means.loc["Benign", TOP_FEATURES].values, width, label="Trung bình Benign", color="#87CEEB")
    ax_bar.bar(x + width, means.loc["Malware", TOP_FEATURES].values, width, label="Trung bình Malware", color="red")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(TOP_FEATURES, rotation=45, ha="right")
    ax_bar.set_title("So sánh đặc trưng then chốt với giá trị trung bình Benign / Malware")
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    X_test, y_test, X_raw_test, full_df, inv_label_mapping = load_data()
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

    # Thẻ chi tiết 1 ca Benign + 1 ca Malware (tái hiện dashboard app_demo.py)
    means = full_df.groupby("Class")[TOP_FEATURES].mean()
    y_test_labels = y_test.map(inv_label_mapping)
    benign_pos = int(np.where(y_test_labels.values == "Benign")[0][0])
    malware_pos = int(np.where(y_test_labels.values == "Malware")[0][0])

    for pos, name in [(benign_pos, "benign"), (malware_pos, "malware")]:
        plot_sample_card(
            model,
            X_test.iloc[[pos]],
            X_raw_test.iloc[pos],
            inv_label_mapping[y_test.iloc[pos]],
            inv_label_mapping,
            means,
            f"{OUT_DIR}/sample_card_{name}.png",
        )
    print(f"Đã lưu: {OUT_DIR}/sample_card_benign.png, sample_card_malware.png")


if __name__ == "__main__":
    main()
