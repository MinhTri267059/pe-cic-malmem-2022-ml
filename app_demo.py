"""
Dashboard demo trực quan: Phát hiện mã độc từ đặc trưng bộ nhớ RAM (CIC-MalMem-2022).
Chạy: streamlit run app_demo.py

Dùng model XGBoost đã train, hỗ trợ cả 3 kịch bản (Binary / 4 nhóm / 16 lớp).
Chọn 1 mẫu từ tập test, hoặc từ các file đặc trưng thật trong "demo test/"
(dữ liệu lab của nhóm), xem xác suất dự đoán + so sánh đặc trưng với
giá trị trung bình theo lớp.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Demo phát hiện mã độc", layout="wide")

SCENARIOS = {
    "Kịch bản 1 - Binary (Benign vs Malware)": 1,
    "Kịch bản 2 - 4 nhóm (Benign, Ransomware, Spyware, Trojan)": 2,
    "Kịch bản 3 - 16 lớp (Benign + 15 họ mã độc)": 3,
}

# Các đặc trưng có tương quan mạnh nhất với nhãn Malware (theo EDA_Report.md)
TOP_FEATURES = [
    "dlllist.avg_dlls_per_proc",
    "handles.nevent",
    "handles.nthread",
    "handles.nmutant",
    "pslist.nppid",
    "malfind.commitCharge",
    "malfind.ninjections",
]


def build_label(df, scenario):
    """Tái hiện đúng logic gán nhãn của reproduce_experiment.py::preprocess_data."""
    if scenario == 1:
        return df["Class"]
    if scenario == 2:
        def to_group(category):
            for name in ["Benign", "Ransomware", "Spyware", "Trojan"]:
                if name in category:
                    return name
            return "Unknown"
        return df["Category"].apply(to_group)
    # scenario == 3
    def to_family(category):
        if category == "Benign":
            return "Benign"
        parts = str(category).split("-")
        return parts[1] if len(parts) > 1 else category
    return df["Category"].apply(to_family)


@st.cache_resource
def load_model_and_data(scenario):
    model = joblib.load(f"Models/Scenario_{scenario}/XGBoost.pkl")
    scaler = joblib.load(f"Models/Scenario_{scenario}/XGBoost_scaler.pkl")
    with open(f"Models/Scenario_{scenario}/label_mapping.json") as f:
        label_mapping = json.load(f)
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    df = pd.read_csv("MalMem2022.csv").drop_duplicates().reset_index(drop=True)
    y_label = build_label(df, scenario)
    y = y_label.map(label_mapping)
    X_raw = df.drop(columns=[c for c in ["Filename", "Class", "Category"] if c in df.columns])

    # ponytail: cùng random_state/test_size dùng khi train (reproduce_experiment.py), để khớp tập test gốc;
    # split trên X thô rồi transform bằng scaler đã fit lúc train (chỉ trên X_train gốc)
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_raw.columns, index=X_test_raw.index)
    return model, inv_label_mapping, X_test, y_test, X_test_raw, df, y_label, scaler, X_raw.columns.tolist()


DEMO_TEST_DIR = Path("demo test")


@st.cache_data
def list_demo_test_files():
    if not DEMO_TEST_DIR.is_dir():
        return []
    return sorted(p.name for p in DEMO_TEST_DIR.glob("*.csv"))


st.title("🔍 Demo: Phát hiện mã độc từ đặc trưng bộ nhớ RAM")
st.caption("Dataset: CIC-MalMem-2022 · Model: XGBoost · 55 đặc trưng Volatility")

st.sidebar.header("Chọn kịch bản & mẫu")
scenario_name = st.sidebar.selectbox("Kịch bản phân loại", list(SCENARIOS.keys()))
scenario = SCENARIOS[scenario_name]

(model, inv_label_mapping, X_test, y_test, X_raw_test, full_df, full_label,
 scaler, feature_cols) = load_model_and_data(scenario)

if "scenario" not in st.session_state or st.session_state.scenario != scenario:
    st.session_state.scenario = scenario
    st.session_state.sample_pos = 0

tab_detect, tab_training = st.tabs(["🔍 Phát hiện 1 mẫu", "📊 Kết quả huấn luyện"])

with tab_detect:
    demo_test_files = list_demo_test_files()
    source_options = ["Lấy ngẫu nhiên từ tập test", "Chọn theo vị trí (tập test)"]
    if demo_test_files:
        source_options.append("Demo của nhóm")
    source = st.sidebar.radio("Nguồn mẫu", source_options)

    if source == "Demo của nhóm":
        demo_file = st.sidebar.selectbox("File trong demo test/", demo_test_files)
        demo_df = pd.read_csv(DEMO_TEST_DIR / demo_file)
        row_idx = st.sidebar.slider("Dòng trong file", 0, len(demo_df) - 1, 0) if len(demo_df) > 1 else 0

        raw_sample = demo_df.iloc[row_idx].reindex(feature_cols)
        if raw_sample.isna().any():
            st.error(
                "File này thiếu một số cột đặc trưng so với model "
                f"(thiếu: {raw_sample[raw_sample.isna()].index.tolist()}). Không thể dự đoán."
            )
            st.stop()

        x_sample = pd.DataFrame(
            scaler.transform(raw_sample.to_frame().T), columns=feature_cols
        )
        true_label = str(demo_df.iloc[row_idx].get("Class", "Không rõ"))
        category = demo_df.iloc[row_idx].get("Category", "")
        filename = demo_df.iloc[row_idx].get("Filename", demo_file)
        st.caption(f"Mẫu từ `{demo_file}` — Category gốc: `{category}` — Filename: `{filename}`")
        if true_label not in inv_label_mapping.values():
            st.warning(
                f"Nhãn gốc '{true_label}' của file này không khớp tập nhãn của kịch bản đang chọn "
                f"({list(inv_label_mapping.values())}) — chỉ hiển thị kết quả dự đoán, không so sánh đúng/sai."
            )
            true_label = None
    else:
        if source == "Lấy ngẫu nhiên từ tập test":
            if st.sidebar.button("🎲 Lấy mẫu mới"):
                st.session_state.sample_pos = int(np.random.randint(0, len(X_test)))
        else:
            st.session_state.sample_pos = st.sidebar.slider("Vị trí mẫu trong tập test", 0, len(X_test) - 1, 0)

        pos = st.session_state.sample_pos
        x_sample = X_test.iloc[[pos]]
        true_label = inv_label_mapping[y_test.iloc[pos]]
        raw_sample = X_raw_test.iloc[pos]

    proba = model.predict_proba(x_sample)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = inv_label_mapping[pred_idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Nhãn thật", true_label or "Không rõ")
    if true_label is not None:
        col2.metric("Dự đoán", pred_label, delta="✅ Đúng" if pred_label == true_label else "❌ Sai")
    else:
        col2.metric("Dự đoán", pred_label)
    col3.metric("Độ tin cậy dự đoán", f"{proba[pred_idx]*100:.2f}%")

    st.progress(float(proba[pred_idx]), text=f"Độ tin cậy vào lớp '{pred_label}': {proba[pred_idx]*100:.1f}%")

    top_k = pd.Series(proba, index=[inv_label_mapping[i] for i in range(len(proba))]).sort_values(ascending=False).head(5)
    st.bar_chart(top_k, horizontal=True, x_label="Xác suất", y_label="Lớp")

    st.divider()
    ref_label = true_label or pred_label
    st.subheader(f"So sánh đặc trưng then chốt với giá trị trung bình theo lớp (so với: {ref_label})")

    # Trung bình theo nhãn trên toàn bộ dữ liệu (không chỉ test) để so sánh ổn định hơn
    means = full_df.assign(**{"__label__": full_label})[TOP_FEATURES + ["__label__"]].groupby("__label__").mean()
    other_labels = [c for c in means.index if c != ref_label][:2]

    compare_df = pd.DataFrame({"Mẫu hiện tại": raw_sample[TOP_FEATURES]})
    compare_df[f"TB {ref_label}"] = means.loc[ref_label]
    for lbl in other_labels:
        compare_df[f"TB {lbl}"] = means.loc[lbl]
    st.bar_chart(compare_df)

    with st.expander("Xem toàn bộ 55 đặc trưng thô của mẫu này"):
        st.dataframe(raw_sample.to_frame(name="Giá trị"))

    st.caption(
        f"Lưu ý: model '{scenario_name}' được train trên dữ liệu đã chuẩn hóa (StandardScaler); "
        "biểu đồ so sánh dùng giá trị thô để dễ đọc."
    )

with tab_training:
    st.subheader(f"Bảng chỉ số đánh giá — {scenario_name}")
    metrics_df = pd.read_csv(f"Metrics_Scenario_{scenario}.csv")
    st.dataframe(
        metrics_df.style.format({"Accuracy": "{:.4f}", "Recall": "{:.4f}", "Precision": "{:.4f}", "F1-Score": "{:.4f}"}),
        width="stretch",
    )

    split_acc = metrics_df[metrics_df["Test_Method"] == "80% Split"].set_index("Model")["Accuracy"]
    st.bar_chart(split_acc, x_label="Model", y_label="Accuracy (80% Split)")

    st.divider()
    st.subheader("Confusion Matrix — XGBoost trên tập test")
    y_pred_all = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_all)
    class_names = [inv_label_mapping[i] for i in sorted(inv_label_mapping)]
    fig, ax = plt.subplots(figsize=(6 + 0.3 * len(class_names), 6 + 0.3 * len(class_names)))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax, xticks_rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
