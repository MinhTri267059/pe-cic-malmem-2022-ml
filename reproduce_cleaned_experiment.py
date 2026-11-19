import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

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
    
    return acc, recall, precision, f1, FPR

def preprocess_cleaned_data(df, scenario):
    # Loại bỏ các cột phi đặc trưng
    df_clean = df.drop(columns=['Filename'], errors='ignore')
    
    if scenario == 1:
        # Kịch bản 1: Phân loại nhị phân (Benign vs Malware)
        df_clean = df_clean.drop(columns=['Category'], errors='ignore')
        df_clean = df_clean.drop_duplicates()
        X = df_clean.drop(columns=['Class'])
        y = df_clean['Class']
    elif scenario == 2:
        # Kịch bản 2: Phân loại đa lớp 4 nhóm (Benign, Ransomware, Trojan, Spyware)
        df_clean['Category'] = df_clean['Category'].apply(lambda x: x.split('-')[0])
        df_clean = df_clean.drop(columns=['Class'], errors='ignore')
        df_clean = df_clean.drop_duplicates()
        X = df_clean.drop(columns=['Category'])
        y = df_clean['Category']
    elif scenario == 3:
        # Kịch bản 3: Phân loại 16 họ chi tiết
        df_clean = df_clean.drop(columns=['Class'], errors='ignore')
        df_clean = df_clean.drop_duplicates()
        X = df_clean.drop(columns=['Category'])
        y = df_clean['Category']
    else:
        raise ValueError("Chỉ chấp nhận scenario 1, 2 hoặc 3.")
        
    return X, y, df_clean

def run_cleaned_experiment(scenario, data_path="MalMem2022_Cleaned.csv"):
    print(f"\n{'='*70}")
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM VỚI DỮ LIỆU ĐÃ LÀM SẠCH ĐẶC TRƯNG - KỊCH BẢN {scenario}")
    print(f"{'='*70}")
    
    df = pd.read_csv(data_path)
    X, y, df_preprocessed = preprocess_cleaned_data(df, scenario)
    
    csv_filename = f"Dataset_Cleaned_Scenario_{scenario}.csv"
    df_preprocessed.to_csv(csv_filename, index=False)
    print(f"✅ Đã lưu tập dữ liệu làm sạch tiền xử lý vào: {csv_filename}")
    print(f"Số lượng đặc trưng sử dụng: {X.shape[1]}")
    print(f"Số lượng mẫu: {len(X)}")
    unique_classes = list(y.unique())
    print(f"Các lớp mục tiêu ({len(unique_classes)}): {unique_classes}")

    label_mapping = {label: idx for idx, label in enumerate(unique_classes)}
    y_mapped = y.map(label_mapping)

    models = {
        'Random Tree (RT)': DecisionTreeClassifier(random_state=42),
        'Random Forest (RF)': RandomForestClassifier(random_state=42, n_jobs=-1),
        'J-48 (Decision Tree)': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'Naive Bayes (NB)': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1)
    }

    model_dir = f'Models_Cleaned/Scenario_{scenario}'
    os.makedirs(model_dir, exist_ok=True)
    with open(f'{model_dir}/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=4)

    results = []

    for name, model in models.items():
        print(f"---> Đang huấn luyện: {name}")
        
        # 80% Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        safe_model_name = name.split(' (')[0].replace(' ', '_').replace('-', '_')
        joblib.dump(model, f'{model_dir}/{safe_model_name}.pkl')
        
        labels_mapped_list = list(label_mapping.values())
        acc_split, rec_split, prec_split, f1_split, fpr_split = calculate_metrics(
            y_test, y_pred, labels_mapped_list
        )
        
        # 10-Fold CV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model, X, y_mapped, cv=cv, 
            scoring=['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted'], 
            n_jobs=-1
        )
        
        acc_cv = cv_scores['test_accuracy'].mean()
        rec_cv = cv_scores['test_recall_weighted'].mean()
        prec_cv = cv_scores['test_precision_weighted'].mean()
        f1_cv = cv_scores['test_f1_weighted'].mean()
        
        results.append({
            'Model': name, 'Test_Method': '80% Split',
            'Accuracy': acc_split, 'Recall': rec_split, 'Precision': prec_split, 'F1-Score': f1_split, 'FPR': fpr_split
        })
        results.append({
            'Model': name, 'Test_Method': '10-Fold CV',
            'Accuracy': acc_cv, 'Recall': rec_cv, 'Precision': prec_cv, 'F1-Score': f1_cv, 'FPR': '-'
        })

    results_df = pd.DataFrame(results)
    output_filename = f"Metrics_Cleaned_Scenario_{scenario}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"📊 Kết quả Kịch bản {scenario} đã lưu vào: {output_filename}")
    print(results_df.to_string(index=False))
    return results_df, X.columns.tolist()

if __name__ == "__main__":
    feature_cols = None
    for sc in [1, 2, 3]:
        _, cols = run_cleaned_experiment(sc)
        if feature_cols is None:
            feature_cols = cols

    # Đánh giá kiểm thử trực tiếp trên file test Win 64-bit (features_maze_test.csv)
    print(f"\n{'='*70}")
    print("🔬 KIỂM THỬ TRỰC TIẾP TRÊN TẬP DỮ LIỆU WINDOWS 64-BIT (features_maze_test.csv)")
    print(f"{'='*70}")
    
    if os.path.exists("features_maze_test.csv"):
        df_maze = pd.read_csv("features_maze_test.csv")
        X_maze = df_maze[feature_cols]
        
        test_eval_results = []
        for sc in [1, 2, 3]:
            model_dir = f'Models_Cleaned/Scenario_{sc}'
            with open(f'{model_dir}/label_mapping.json') as f:
                lmap = json.load(f)
            inv_map = {v: k for k, v in lmap.items()}
            
            xgb_model = joblib.load(f'{model_dir}/XGBoost.pkl')
            rf_model = joblib.load(f'{model_dir}/Random_Forest.pkl')
            
            p_xgb = [inv_map[p] for p in xgb_model.predict(X_maze)]
            p_rf = [inv_map[p] for p in rf_model.predict(X_maze)]
            
            test_eval_results.append({
                'Scenario': f'Kịch bản {sc}',
                'True_Label': df_maze['Class'].iloc[0] if sc == 1 else ('Ransomware' if sc == 2 else 'Ransomware-Maze'),
                'XGBoost_Preds': p_xgb,
                'RF_Preds': p_rf
            })
            
        print("\nKết quả dự đoán trên 10 mẫu Maze 64-bit qua 3 kịch bản:")
        for r in test_eval_results:
            print(f"[{r['Scenario']}] Nhãn thực tế: {r['True_Label']}")
            print(f"   • XGBoost: {r['XGBoost_Preds']}")
            print(f"   • Random Forest: {r['RF_Preds']}")
            
        eval_df = pd.DataFrame(test_eval_results)
        eval_df.to_csv("Cleaned_Maze_Test_Evaluation.csv", index=False)
        print("\n✅ Đã lưu kết quả kiểm thử Win 64-bit vào: Cleaned_Maze_Test_Evaluation.csv")
