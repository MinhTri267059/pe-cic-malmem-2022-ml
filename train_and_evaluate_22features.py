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

def preprocess_data(df, scenario):
    cols_to_keep = FEATURES_22 + ['Class', 'Category']
    df_sub = df[cols_to_keep].copy()
    
    if scenario == 1:
        # Kịch bản 1: Phân loại nhị phân (Benign vs Malware)
        df_sub = df_sub.drop(columns=['Category']).drop_duplicates()
        X = df_sub[FEATURES_22]
        y = df_sub['Class']
    elif scenario == 2:
        # Kịch bản 2: Phân loại đa lớp 4 nhóm (Benign, Ransomware, Trojan, Spyware)
        df_sub['Category'] = df_sub['Category'].apply(lambda x: x.split('-')[0])
        df_sub = df_sub.drop(columns=['Class']).drop_duplicates()
        X = df_sub[FEATURES_22]
        y = df_sub['Category']
    elif scenario == 3:
        # Kịch bản 3: Phân loại 16 họ chi tiết
        df_sub = df_sub.drop(columns=['Class']).drop_duplicates()
        X = df_sub[FEATURES_22]
        y = df_sub['Category']
    else:
        raise ValueError("Scenario phải là 1, 2 hoặc 3")
        
    return X, y, df_sub

def run_experiment_22_features():
    df_raw = pd.read_csv("MalMem2022.csv")
    print(f"Tổng số mẫu nạp từ MalMem2022.csv: {len(df_raw)}")
    print(f"Số lượng đặc trưng sử dụng cố định: {len(FEATURES_22)}")
    
    # Lưu bộ data 22 đặc trưng
    df_22_full = df_raw[FEATURES_22 + ['Class', 'Category', 'Filename']]
    df_22_full.to_csv("Dataset_22features.csv", index=False)
    print("✅ Đã lưu Dataset_22features.csv")
    
    models_dict = {
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1),
        'Random Forest (RF)': RandomForestClassifier(random_state=42, n_jobs=-1),
        'J-48 (Decision Tree)': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'Random Tree (RT)': DecisionTreeClassifier(random_state=42),
        'Naive Bayes (NB)': GaussianNB()
    }
    
    scenario_metrics = {}
    
    for scenario in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"🚀 HUẤN LUYỆN VÀ ĐÁNH GIÁ: KỊCH BẢN {scenario} (VỚI 22 ĐẶC TRƯNG BẤT BIẾN)")
        print(f"{'='*70}")
        
        X, y, df_pre = preprocess_data(df_raw, scenario)
        df_pre.to_csv(f"Dataset_22features_Scenario_{scenario}.csv", index=False)
        
        unique_classes = sorted(list(y.unique()))
        label_mapping = {label: idx for idx, label in enumerate(unique_classes)}
        y_mapped = y.map(label_mapping)
        
        print(f"Số mẫu sau khi loại trùng lặp: {len(X)}")
        print(f"Số lớp phân loại: {len(unique_classes)} -> {unique_classes}")
        
        model_dir = f"Models_22features/Scenario_{scenario}"
        os.makedirs(model_dir, exist_ok=True)
        with open(f"{model_dir}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=4)
            
        results = []
        labels_list = list(label_mapping.values())
        
        # Split train/test 80:20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
        )
        
        for name, model in models_dict.items():
            print(f"--> Đang huấn luyện: {name}")
            model.fit(X_train, y_train)
            
            # Predict test 80/20
            y_pred = model.predict(X_test)
            acc, rec, prec, f1, fpr = calculate_metrics(y_test, y_pred, labels_list)
            
            # Save trained model
            safe_name = name.split(' (')[0].replace(' ', '_').replace('-', '_')
            joblib.dump(model, f"{model_dir}/{safe_name}.pkl")
            
            # 10-Fold CV
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = cross_validate(
                model, X, y_mapped, cv=cv, 
                scoring=['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted'], 
                n_jobs=-1
            )
            
            results.append({
                'Model': name, 'Test_Method': '80% Split',
                'Accuracy': acc, 'Recall': rec, 'Precision': prec, 'F1-Score': f1, 'FPR': fpr
            })
            results.append({
                'Model': name, 'Test_Method': '10-Fold CV',
                'Accuracy': cv_scores['test_accuracy'].mean(),
                'Recall': cv_scores['test_recall_weighted'].mean(),
                'Precision': cv_scores['test_precision_weighted'].mean(),
                'F1-Score': cv_scores['test_f1_weighted'].mean(),
                'FPR': '-'
            })
            
        res_df = pd.DataFrame(results)
        res_df.to_csv(f"Metrics_22features_Scenario_{scenario}.csv", index=False)
        scenario_metrics[scenario] = res_df
        print(f"\n📊 Kết quả Kịch bản {scenario} trên tập Test:")
        print(res_df.to_string(index=False))

    # -------------------------------------------------------------
    # KIỂM THỬ THỰC TẾ TRÊN BỘ DỮ LIỆU 10 BẢN DUMP 64-BIT
    # -------------------------------------------------------------
    print(f"\n{'='*70}")
    print("🔬 KIỂM THỬ THỰC TẾ LẦN CUỐI TRÊN 10 BẢN DUMP WINDOWS 64-BIT (features_maze_test.csv)")
    print(f"{'='*70}")
    
    df_maze = pd.read_csv("features_maze_test.csv")
    X_maze = df_maze[FEATURES_22]
    
    dumps_results = []
    
    for i in range(len(df_maze)):
        fname = df_maze['Filename'].iloc[i]
        sample_row = X_maze.iloc[[i]]
        
        row_res = {'Filename': fname}
        
        for sc in [1, 2, 3]:
            model_dir = f"Models_22features/Scenario_{sc}"
            with open(f"{model_dir}/label_mapping.json") as f:
                lmap = json.load(f)
            inv_map = {v: k for k, v in lmap.items()}
            
            xgb = joblib.load(f"{model_dir}/XGBoost.pkl")
            rf = joblib.load(f"{model_dir}/Random_Forest.pkl")
            
            p_xgb = inv_map[xgb.predict(sample_row)[0]]
            p_rf = inv_map[rf.predict(sample_row)[0]]
            
            prob_xgb = xgb.predict_proba(sample_row)[0]
            
            if sc == 1:
                row_res['KB1_True'] = 'Malware'
                row_res['KB1_XGB_Pred'] = p_xgb
                row_res['KB1_RF_Pred'] = p_rf
                row_res['KB1_XGB_Prob'] = f"{prob_xgb[lmap['Malware']]*100:.2f}%"
            elif sc == 2:
                row_res['KB2_True'] = 'Ransomware'
                row_res['KB2_XGB_Pred'] = p_xgb
                row_res['KB2_RF_Pred'] = p_rf
                if 'Ransomware' in lmap:
                    row_res['KB2_XGB_Ransomware_Prob'] = f"{prob_xgb[lmap['Ransomware']]*100:.2f}%"
            elif sc == 3:
                row_res['KB3_True'] = 'Ransomware-Maze'
                row_res['KB3_XGB_Pred'] = p_xgb
                row_res['KB3_RF_Pred'] = p_rf
                if 'Ransomware-Maze' in lmap:
                    row_res['KB3_XGB_Maze_Prob'] = f"{prob_xgb[lmap['Ransomware-Maze']]*100:.2f}%"
                    
        dumps_results.append(row_res)
        
    eval_10_df = pd.DataFrame(dumps_results)
    eval_10_df.to_csv("Predictions_10dumps_22features.csv", index=False)
    print("✅ Đã lưu kết quả dự đoán chi tiết 10 bản dump vào: Predictions_10dumps_22features.csv")
    print(eval_10_df.to_string(index=False))

if __name__ == "__main__":
    run_experiment_22_features()
