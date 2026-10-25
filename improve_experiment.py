import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. TIỀN XỬ LÝ DỮ LIỆU
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path, scenario):
    print(f"\n==================================================================")
    print(f"📊 CHUẨN BỊ DỮ LIỆU CHO KỊCH BẢN {scenario}")
    print(f"==================================================================")
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    
    labels = []
    for _, row in df.iterrows():
        cat = str(row['Category'])
        if scenario == 2:
            if 'Benign' in cat:
                labels.append('Benign')
            elif 'Ransomware' in cat:
                labels.append('Ransomware')
            elif 'Spyware' in cat:
                labels.append('Spyware')
            elif 'Trojan' in cat:
                labels.append('Trojan')
            else:
                labels.append('Unknown')
        elif scenario == 3:
            if cat == 'Benign':
                labels.append('Benign')
            else:
                parts = cat.split('-')
                labels.append(parts[1] if len(parts) > 1 else cat)
                
    df['Label'] = labels
    
    cols_to_drop = ['Label']
    for col in ['Filename', 'Class', 'Category']:
        if col in df.columns:
            cols_to_drop.append(col)
            
    X = df.drop(columns=cols_to_drop)
    y = df['Label']

    unique_classes = sorted(list(y.unique()))
    label_mapping = {label: idx for idx, label in enumerate(unique_classes)}
    y_mapped = y.map(label_mapping)

    print(f"✓ Tổng số mẫu: {len(X):,} | Số đặc trưng: {X.shape[1]}")
    print(f"✓ Số lớp bài toán ({len(unique_classes)}): {unique_classes}")

    # Chia 80% Train - 20% Test phân tầng (stratified) tương tự baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )

    # ponytail: scaler fit CHỈ trên X_train, tránh rò rỉ thống kê từ X_test
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, label_mapping

# -----------------------------------------------------------------------------
# 2. HÀM ĐÁNH GIÁ MÔ HÌNH
# -----------------------------------------------------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"  ⏳ Đang huấn luyện và đánh giá: {name} ...", end="", flush=True)
    t0 = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Flatten if needed
    if len(y_pred.shape) > 1:
        y_pred = y_pred.ravel()
        
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    elapsed = time.time() - t0
    print(f" Done! ({elapsed:.1f}s) -> Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}%")
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Macro_Precision': macro_prec,
        'Macro_Recall': macro_rec,
        'Macro_F1': macro_f1,
        'Weighted_F1': weighted_f1,
        'Train_Time_Sec': round(elapsed, 2)
    }

# -----------------------------------------------------------------------------
# 3. THỰC THI SO SÁNH PHƯƠNG ÁN 1, 2, 3
# -----------------------------------------------------------------------------
def run_improvement_benchmark(scenario):
    dataset_file = 'MalMem2022.csv'
    X_train, X_test, y_train, y_test, label_mapping = load_and_preprocess(dataset_file, scenario)
    
    results = []
    
    print(f"\n--- [1] KHỞI ĐỘNG CÁC MÔ HÌNH BASELINE (MẶC ĐỊNH HIỆN TẠI) ---")
    base_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    results.append(evaluate_model("Baseline: XGBoost (Default)", base_xgb, X_train, y_train, X_test, y_test))
    
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    results.append(evaluate_model("Baseline: Random Forest (Default)", base_rf, X_train, y_train, X_test, y_test))
    
    print(f"\n--- [2] PHƯƠNG ÁN 1: TINH CHỈNH THAM SỐ (HYPERPARAMETER TUNING) ---")
    # Tinh chỉnh độ sâu, learning rate, subsampling và chính quy hóa để chống overfitting
    if scenario == 2:
        tuned_xgb = XGBClassifier(
            n_estimators=350,
            learning_rate=0.07,
            max_depth=9,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.2,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        tuned_rf = RandomForestClassifier(
            n_estimators=350,
            max_depth=26,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    else: # Scenario 3 (16 lớp khó)
        tuned_xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.06,
            max_depth=10,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.15,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        tuned_rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=28,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
    results.append(evaluate_model("Phương án 1: Tuned XGBoost", tuned_xgb, X_train, y_train, X_test, y_test))
    results.append(evaluate_model("Phương án 1: Tuned Random Forest", tuned_rf, X_train, y_train, X_test, y_test))
    
    print(f"\n--- [3] PHƯƠNG ÁN 2: THỬ NGHIỆM MÔ HÌNH MỚI (EXTRA TREES, LIGHTGBM, CATBOOST) ---")
    # 2.1 Extra Trees
    et_model = ExtraTreesClassifier(
        n_estimators=350,
        max_depth=28 if scenario == 3 else 26,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    results.append(evaluate_model("Phương án 2: Extra Trees Classifier", et_model, X_train, y_train, X_test, y_test))
    
    # 2.2 LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=350,
        learning_rate=0.07,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    results.append(evaluate_model("Phương án 2: LightGBM Classifier", lgb_model, X_train, y_train, X_test, y_test))
    
    # 2.3 CatBoost
    cat_model = CatBoostClassifier(
        iterations=350,
        learning_rate=0.08,
        depth=7,
        random_seed=42,
        thread_count=-1,
        verbose=0
    )
    results.append(evaluate_model("Phương án 2: CatBoost Classifier", cat_model, X_train, y_train, X_test, y_test))
    
    print(f"\n--- [4] PHƯƠNG ÁN 3: KẾT HỢP ĐA MÔ HÌNH (VOTING & STACKING ENSEMBLE) ---")
    # 3.1 Soft Voting Ensemble: kết hợp 4 mô hình họ cây mạnh nhất
    voting_ensemble = VotingClassifier(
        estimators=[
            ('xgb', tuned_xgb),
            ('rf', tuned_rf),
            ('et', et_model),
            ('lgb', lgb_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    results.append(evaluate_model("Phương án 3: Soft Voting Ensemble (XGB+RF+ET+LGB)", voting_ensemble, X_train, y_train, X_test, y_test))
    
    # 3.2 Stacking Ensemble: Meta-Learner Logistic Regression
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('xgb', tuned_xgb),
            ('rf', tuned_rf),
            ('et', et_model),
            ('lgb', lgb_model)
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=-1
    )
    results.append(evaluate_model("Phương án 3: Stacking Ensemble (Meta-Logistic)", stacking_ensemble, X_train, y_train, X_test, y_test))
    
    # Bảng tổng kết kết quả
    res_df = pd.DataFrame(results)
    out_csv = f"Improvement_Scenario_{scenario}.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Đã lưu kết quả so sánh vào: {out_csv}")
    
    baseline_acc = res_df.loc[0, 'Accuracy']
    baseline_f1 = res_df.loc[0, 'Macro_F1']
    res_df['Delta_Acc_%'] = (res_df['Accuracy'] - baseline_acc) * 100
    res_df['Delta_F1_%'] = (res_df['Macro_F1'] - baseline_f1) * 100
    
    print(f"\n🏆 BẢNG TỔNG HỢP SO SÁNH HIỆU QUẢ CẢI THIỆN - KỊCH BẢN {scenario}:")
    display_df = res_df[['Model', 'Accuracy', 'Macro_F1', 'Delta_Acc_%', 'Delta_F1_%', 'Train_Time_Sec']].copy()
    display_df['Accuracy'] = (display_df['Accuracy'] * 100).map("{:.2f}%".format)
    display_df['Macro_F1'] = (display_df['Macro_F1'] * 100).map("{:.2f}%".format)
    display_df['Delta_Acc_%'] = display_df['Delta_Acc_%'].map("{:+.2f}%".format)
    display_df['Delta_F1_%'] = display_df['Delta_F1_%'].map("{:+.2f}%".format)
    print(display_df.to_string(index=False))
    
    return res_df

# -----------------------------------------------------------------------------
# CHẠY CHO CẢ KỊCH BẢN 2 VÀ KỊCH BẢN 3
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    t_start = time.time()
    print("🚀 BẮT ĐẦU THỰC HIỆN THỬ NGHIỆM CẢI THIỆN CHO KỊCH BẢN 2 VÀ KỊCH BẢN 3...")
    
    res_s2 = run_improvement_benchmark(scenario=2)
    res_s3 = run_improvement_benchmark(scenario=3)
    
    print(f"\n🎉 HOÀN TẤT TOÀN BỘ CÁC THỰC NGHIỆM SO SÁNH TRONG: {(time.time() - t_start)/60:.2f} phút!")
