import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# 22 đặc trưng cốt lõi thuần hành vi kháng môi trường OS
FEATURES_22 = [
    'ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem', 
    'ldrmodules.not_in_load_avg', 'ldrmodules.not_in_init_avg', 'ldrmodules.not_in_mem_avg', 
    'malfind.ninjections', 'malfind.uniqueInjections', 
    'psxview.not_in_pslist', 'psxview.not_in_eprocess_pool', 'psxview.not_in_ethread_pool', 
    'psxview.not_in_pspcid_list', 'psxview.not_in_csrss_handles', 'psxview.not_in_session', 
    'psxview.not_in_deskthrd', 'psxview.not_in_pslist_false_avg', 'psxview.not_in_eprocess_pool_false_avg', 
    'psxview.not_in_ethread_pool_false_avg', 'psxview.not_in_pspcid_list_false_avg', 
    'psxview.not_in_csrss_handles_false_avg', 'psxview.not_in_session_false_avg', 'psxview.not_in_deskthrd_false_avg'
]

MODEL_NAMES = [
    'Random_Forest',
    'XGBoost',
    'J_48',
    'Random_Tree',
    'Naive_Bayes'
]

MODEL_DISPLAY_NAMES = {
    'Random_Forest': 'Random Forest (RF)',
    'XGBoost': 'XGBoost',
    'J_48': 'J-48 (Decision Tree)',
    'Random_Tree': 'Random Tree (RT)',
    'Naive_Bayes': 'Naive Bayes (NB)'
}

def calculate_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    FPR_array = np.divide(FP, (FP + TN), out=np.zeros_like(FP, dtype=float), where=(FP+TN)!=0)
    FPR = np.mean(FPR_array)
    
    return acc, recall, precision, f1, FPR, cm

def evaluate_test_dataset(test_csv_path="features_all_x64_19041.csv"):
    print("=" * 80)
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH ĐÃ HUẤN LUYỆN TRÊN FILE TEST: {test_csv_path}")
    print("=" * 80)
    
    df = pd.read_csv(test_csv_path)
    print(f"📌 Tổng số mẫu kiểm thử: {len(df)}")
    print(f"📌 Phân bố nhãn Class (KB1): {df['Class'].value_counts().to_dict()}")
    print(f"📌 Phân bố 4 nhóm (KB2): {df['Category'].apply(lambda x: x.split('-')[0]).value_counts().to_dict()}")
    print(f"📌 Số lượng họ chi tiết (KB3): {df['Category'].nunique()} họ khác nhau")
    
    # Chuẩn bị dữ liệu đặc trưng 22 features (điền 0 cho 6 cột psxview không khả dụng trên Win10 x64)
    X_22 = df[FEATURES_22].fillna(0)
    
    all_metrics = []
    
    # -------------------------------------------------------------------------
    # 1. ĐÁNH GIÁ TRÊN BỘ MÔ HÌNH 22 ĐẶC TRƯNG KHÁNG MÔI TRƯỜNG (Models_22features)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1️⃣ ĐÁNH GIÁ MÔ HÌNH 22 ĐẶC TRƯNG BẤT BIẾN (Models_22features)")
    print("=" * 80)
    
    predictions_table = pd.DataFrame({
        'Filename': df['Filename'],
        'True_Class': df['Class'],
        'True_Group': df['Category'].apply(lambda x: x.split('-')[0]),
        'True_Category': df['Category']
    })
    
    for scenario in [1, 2, 3]:
        model_dir = f"Models_22features/Scenario_{scenario}"
        with open(f"{model_dir}/label_mapping.json") as f:
            label_mapping = json.load(f)
        inv_map = {v: k for k, v in label_mapping.items()}
        labels_list = list(label_mapping.keys())
        
        if scenario == 1:
            y_true = df['Class']
            sc_name = "Kịch bản 1 (Nhị phân Benign vs Malware)"
        elif scenario == 2:
            y_true = df['Category'].apply(lambda x: x.split('-')[0])
            sc_name = "Kịch bản 2 (Đa lớp 4 nhóm hành vi)"
        else:
            y_true = df['Category']
            sc_name = "Kịch bản 3 (Chi tiết 16 họ malware)"
            
        print(f"\n--- {sc_name} ---")
        
        for m_name in MODEL_NAMES:
            model_file = f"{model_dir}/{m_name}.pkl"
            if not os.path.exists(model_file):
                continue
            model = joblib.load(model_file)
            
            # Dự đoán
            pred_indices = model.predict(X_22)
            y_pred = [inv_map[i] for i in pred_indices]
            
            # Ghi nhận dự đoán vào bảng chi tiết cho RF và XGBoost
            if m_name in ['Random_Forest', 'XGBoost']:
                short_m = 'RF' if m_name == 'Random_Forest' else 'XGB'
                predictions_table[f'KB{scenario}_{short_m}_Pred'] = y_pred
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_22)
                    max_probs = np.max(probs, axis=1) * 100
                    predictions_table[f'KB{scenario}_{short_m}_Conf'] = [f"{p:.1f}%" for p in max_probs]
            
            # Tính metrics
            acc, rec, prec, f1, fpr, cm = calculate_metrics(y_true, y_pred, labels_list)
            
            all_metrics.append({
                'Model_Set': '22 Features (Invariant)',
                'Scenario': f'Scenario {scenario}',
                'Model': MODEL_DISPLAY_NAMES[m_name],
                'Accuracy': round(acc * 100, 2),
                'Precision': round(prec * 100, 2),
                'Recall': round(rec * 100, 2),
                'F1-Score': round(f1 * 100, 2),
                'FPR': round(fpr * 100, 2) if isinstance(fpr, (int, float)) else fpr
            })
            
            print(f"[{MODEL_DISPLAY_NAMES[m_name]:22s}] Acc: {acc*100:6.2f}% | Precision: {prec*100:6.2f}% | Recall: {rec*100:6.2f}% | F1: {f1*100:6.2f}% | FPR: {fpr*100:5.2f}%")
            if scenario == 1 and m_name in ['Random_Forest', 'XGBoost']:
                print(f"   Confusion Matrix [Benign, Malware]:\n   {cm.tolist()}")

    # -------------------------------------------------------------------------
    # 2. SO SÁNH ĐỐI CHỨNG VỚI MÔ HÌNH 55 ĐẶC TRƯNG LÀM SẠCH (Models_Cleaned)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2️⃣ SO SÁNH ĐỐI CHỨNG VỚI MÔ HÌNH 55 ĐẶC TRƯNG LÀM SẠCH (Models_Cleaned)")
    print("=" * 80)
    
    df_train_cleaned = pd.read_csv('Dataset_Cleaned_Scenario_1.csv')
    cleaned_features = [c for c in df_train_cleaned.columns if c not in ['Class', 'Category', 'Filename']]
    X_cleaned = df[cleaned_features].fillna(0)
    
    for scenario in [1, 2]:
        model_dir = f"Models_Cleaned/Scenario_{scenario}"
        with open(f"{model_dir}/label_mapping.json") as f:
            label_mapping = json.load(f)
        inv_map = {v: k for k, v in label_mapping.items()}
        labels_list = list(label_mapping.keys())
        
        y_true = df['Class'] if scenario == 1 else df['Category'].apply(lambda x: x.split('-')[0])
        
        for m_name in ['Random_Forest', 'XGBoost']:
            model = joblib.load(f"{model_dir}/{m_name}.pkl")
            pred_indices = model.predict(X_cleaned)
            y_pred = [inv_map[i] for i in pred_indices]
            
            acc, rec, prec, f1, fpr, cm = calculate_metrics(y_true, y_pred, labels_list)
            
            all_metrics.append({
                'Model_Set': '55 Features (Cleaned)',
                'Scenario': f'Scenario {scenario}',
                'Model': MODEL_DISPLAY_NAMES[m_name],
                'Accuracy': round(acc * 100, 2),
                'Precision': round(prec * 100, 2),
                'Recall': round(rec * 100, 2),
                'F1-Score': round(f1 * 100, 2),
                'FPR': round(fpr * 100, 2) if isinstance(fpr, (int, float)) else fpr
            })
            print(f"[Models_Cleaned - SC{scenario}] {MODEL_DISPLAY_NAMES[m_name]:20s} Acc: {acc*100:6.2f}% | F1: {f1*100:6.2f}%")

    # -------------------------------------------------------------------------
    # 3. LƯU CÁC TẬP TIN KẾT QUẢ
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = "Metrics_x64_19041_Evaluation.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n✅ Đã xuất bảng tổng hợp chỉ số đánh giá ra: {metrics_csv}")
    
    pred_csv = "Predictions_200dumps_x64_19041.csv"
    predictions_table.to_csv(pred_csv, index=False)
    print(f"✅ Đã xuất bảng dự đoán chi tiết 200 bản dump ra: {pred_csv}")
    
    return metrics_df, predictions_table

if __name__ == "__main__":
    evaluate_test_dataset()

