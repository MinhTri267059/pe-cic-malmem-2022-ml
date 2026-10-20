"""
Dashboard demo trực quan: Phát hiện mã độc từ đặc trưng bộ nhớ RAM (CIC-MalMem-2022).
Chạy: streamlit run app_demo.py

Dùng model XGBoost đã train (Scenario 1 - Binary: Benign vs Malware).
Chọn 1 mẫu từ tập test, xem xác suất dự đoán + so sánh đặc trưng với giá trị
trung bình của Benign/Malware.
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Demo phát hiện mã độc", layout="wide")

MODEL_PATH = "Models/Scenario_1/XGBoost.pkl"
LABEL_MAP_PATH = "Models/Scenario_1/label_mapping.json"

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


@st.cache_resource
def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    with open(LABEL_MAP_PATH) as f:
        label_mapping = json.load(f)
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    df = pd.read_csv("MalMem2022.csv").drop_duplicates().reset_index(drop=True)
    y = df["Class"].map(label_mapping)
    X_raw = df.drop(columns=[c for c in ["Filename", "Class", "Category"] if c in df.columns])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

    # ponytail: cùng random_state/test_size dùng khi train (reproduce_experiment.py), để khớp tập test gốc
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_raw_test = X_raw.loc[X_test.index]
    return model, inv_label_mapping, X_test, y_test, X_raw_test


model, inv_label_mapping, X_test, y_test, X_raw_test = load_model_and_data()

st.title("🔍 Demo: Phát hiện mã độc từ đặc trưng bộ nhớ RAM")
st.caption("Dataset: CIC-MalMem-2022 · Model: XGBoost (Kịch bản 1 - Binary) · 55 đặc trưng Volatility")

st.sidebar.header("Chọn mẫu để kiểm tra")
mode = st.sidebar.radio("Nguồn mẫu", ["Lấy ngẫu nhiên từ tập test", "Chọn theo vị trí"])

if "sample_pos" not in st.session_state:
    st.session_state.sample_pos = 0

if mode == "Lấy ngẫu nhiên từ tập test":
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
malware_idx = [k for k, v in inv_label_mapping.items() if v == "Malware"][0]

col1, col2, col3 = st.columns(3)
col1.metric("Nhãn thật", true_label)
col2.metric("Dự đoán", pred_label, delta="✅ Đúng" if pred_label == true_label else "❌ Sai")
col3.metric("Xác suất Malware", f"{proba[malware_idx]*100:.2f}%")

st.progress(float(proba[malware_idx]), text=f"Mức độ nghi ngờ mã độc: {proba[malware_idx]*100:.1f}%")

st.divider()
st.subheader("So sánh đặc trưng then chốt với giá trị trung bình Benign / Malware")

# Trung bình theo nhãn thật trên toàn bộ dữ liệu (không chỉ test) để so sánh ổn định hơn
full_df = pd.read_csv("MalMem2022.csv").drop_duplicates().reset_index(drop=True)
means = full_df.groupby("Class")[TOP_FEATURES].mean()

compare_df = pd.DataFrame(
    {
        "Mẫu hiện tại": raw_sample[TOP_FEATURES],
        "Trung bình Benign": means.loc["Benign"],
        "Trung bình Malware": means.loc["Malware"],
    }
)
st.bar_chart(compare_df)

with st.expander("Xem toàn bộ 55 đặc trưng thô của mẫu này"):
    st.dataframe(raw_sample.to_frame(name="Giá trị"))

st.divider()
st.caption(
    "Lưu ý: model được train trên dữ liệu đã chuẩn hóa (StandardScaler); "
    "biểu đồ so sánh dùng giá trị thô để dễ đọc."
)
