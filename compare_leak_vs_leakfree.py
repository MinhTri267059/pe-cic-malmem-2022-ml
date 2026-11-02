# -*- coding: utf-8 -*-
"""
So sánh định lượng giữa:
1. Naive Random Split (Rò rỉ snapshot dump bộ nhớ 99.92%)
2. Leak-Free Group Split (Phân chia theo Mã băm tệp nhị phân - SampleID, 0% rò rỉ mã độc)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def extract_sample_id(fn):
    m = re.match(r'^(.*)-(\d+)\.raw$', str(fn))
    return m.group(1) if m else str(fn)

def run_leak_comparison():
    csv_master = r"D:\Tai Lieu Hoc Tap Cao Hoc\Chuyên đề nghiên cứu về Phân loại mã độc\pe-cic-malmem-2022-ml\new feature\Dataset_22features.csv"
    df = pd.read_csv(csv_master)
    
    # Rút gọn Category 16 lớp về 4 nhóm (Scenario 2: Benign, Ransomware, Spyware, Trojan)
    df['SuperCategory'] = df['Category'].apply(lambda x: x.split('-')[0])
    df_s2 = df[df['SuperCategory'].isin(['Benign', 'Ransomware', 'Spyware', 'Trojan'])].copy()
    df_s2['SampleID'] = df_s2['Filename'].apply(extract_sample_id)
    
    feat_cols = [c for c in df_s2.columns if c not in ['Class', 'Category', 'Label', 'Filename', 'SampleID', 'SuperCategory']]
    classes = sorted(df_s2['SuperCategory'].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df_s2['label'] = df_s2['SuperCategory'].map(label_map)

    print("=" * 80)
    print("⚖️  THỰC NGHIỆM ĐỐI SÁNH: NAIVE RANDOM SPLIT (RÒ RỈ) VS LEAK-FREE GROUP SPLIT")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. NAIVE RANDOM SPLIT (Có rò rỉ snapshot)
    # -------------------------------------------------------------
    df_train_n, df_test_n = train_test_split(
        df_s2, test_size=0.2, random_state=42, stratify=df_s2['label']
    )
    
    mal_tr_samples = set(df_train_n[df_train_n['Class'] != 'Benign']['SampleID'])
    mal_te_samples = set(df_test_n[df_test_n['Class'] != 'Benign']['SampleID'])
    overlap_binaries_naive = len(mal_tr_samples.intersection(mal_te_samples))
    total_test_binaries = len(mal_te_samples)

    scaler_n = StandardScaler()
    X_tr_n_sc = scaler_n.fit_transform(df_train_n[feat_cols])
    X_te_n_sc = scaler_n.transform(df_test_n[feat_cols])
    y_tr_n = df_train_n['label'].values
    y_te_n = df_test_n['label'].values

    rf_n = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_n.fit(X_tr_n_sc, y_tr_n)
    pred_rf_n = rf_n.predict(X_te_n_sc)
    acc_rf_n = accuracy_score(y_te_n, pred_rf_n)
    f1_rf_n = f1_score(y_te_n, pred_rf_n, average='macro')

    xgb_n = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1)
    xgb_n.fit(X_tr_n_sc, y_tr_n)
    pred_xgb_n = xgb_n.predict(X_te_n_sc)
    acc_xgb_n = accuracy_score(y_te_n, pred_xgb_n)
    f1_xgb_n = f1_score(y_te_n, pred_xgb_n, average='macro')

    # -------------------------------------------------------------
    # 2. LEAK-FREE GROUP SPLIT (Không rò rỉ snapshot)
    # - Benign: chia ngẫu nhiên 80/20
    # - Malware: chia theo Group (SampleID) 80/20
    # -------------------------------------------------------------
    df_benign = df_s2[df_s2['Class'] == 'Benign']
    df_malware = df_s2[df_s2['Class'] != 'Benign']

    b_tr, b_te = train_test_split(df_benign, test_size=0.2, random_state=42)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    m_tr_idx, m_te_idx = next(gss.split(df_malware, df_malware['label'], groups=df_malware['SampleID']))
    m_tr = df_malware.iloc[m_tr_idx]
    m_te = df_malware.iloc[m_te_idx]

    df_train_g = pd.concat([b_tr, m_tr], ignore_index=True)
    df_test_g = pd.concat([b_te, m_te], ignore_index=True)

    mal_tr_samples_g = set(df_train_g[df_train_g['Class'] != 'Benign']['SampleID'])
    mal_te_samples_g = set(df_test_g[df_test_g['Class'] != 'Benign']['SampleID'])
    overlap_binaries_group = len(mal_tr_samples_g.intersection(mal_te_samples_g))
    total_test_binaries_g = len(mal_te_samples_g)

    scaler_g = StandardScaler()
    X_tr_g_sc = scaler_g.fit_transform(df_train_g[feat_cols])
    X_te_g_sc = scaler_g.transform(df_test_g[feat_cols])
    y_tr_g = df_train_g['label'].values
    y_te_g = df_test_g['label'].values

    rf_g = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_g.fit(X_tr_g_sc, y_tr_g)
    pred_rf_g = rf_g.predict(X_te_g_sc)
    acc_rf_g = accuracy_score(y_te_g, pred_rf_g)
    f1_rf_g = f1_score(y_te_g, pred_rf_g, average='macro')

    xgb_g = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1)
    xgb_g.fit(X_tr_g_sc, y_tr_g)
    pred_xgb_g = xgb_g.predict(X_te_g_sc)
    acc_xgb_g = accuracy_score(y_te_g, pred_xgb_g)
    f1_xgb_g = f1_score(y_te_g, pred_xgb_g, average='macro')

    print("\n📊 BẢNG TỔNG HỢP SO SÁNH TÁC ĐỘNG CỦA DATA LEAKAGE (KỊCH BẢN 2 - 4 LỚP):")
    print(f"{'Tiêu chí đánh giá':<40} | {'Naive Random Split (Bị rò rỉ)':<30} | {'Leak-Free Group Split':<25}")
    print("-" * 102)
    print(f"{'Tỷ lệ rò rỉ mẫu mã độc trong Test':<40} | {overlap_binaries_naive}/{total_test_binaries} ({overlap_binaries_naive/total_test_binaries*100:.2f}%) | {overlap_binaries_group}/{total_test_binaries_g} (0.00%)")
    print(f"{'Random Forest Accuracy':<40} | {acc_rf_n*100:.2f}% | {acc_rf_g*100:.2f}%")
    print(f"{'Random Forest Macro-F1':<40} | {f1_rf_n*100:.2f}% | {f1_rf_g*100:.2f}%")
    print(f"{'XGBoost Accuracy':<40} | {acc_xgb_n*100:.2f}% | {acc_xgb_g*100:.2f}%")
    print(f"{'XGBoost Macro-F1':<40} | {f1_xgb_n*100:.2f}% | {f1_xgb_g*100:.2f}%")
    print("-" * 102)

if __name__ == '__main__':
    run_leak_comparison()
