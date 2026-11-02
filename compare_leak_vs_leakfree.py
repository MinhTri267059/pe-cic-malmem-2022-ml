# -*- coding: utf-8 -*-
"""
So sánh định lượng (Kịch bản 2 & 3, 22 đặc trưng):
  1. Naive Random Split (rò rỉ snapshot dump bộ nhớ)
  2. Leak-Free Group Split (StratifiedGroupKFold theo SampleID)

SỬA LỖI so với phiên bản trước:
  • Đường dẫn tuyệt đối D:\\... -> tương đối.
  • GroupShuffleSplit KHÔNG phân tầng -> tỷ lệ lớp test lệch; nay dùng StratifiedGroupKFold.
  • Chỉ 1 lần chia -> nay dùng 5-fold cho cả hai giao thức và báo cáo mean ± std.
  • Bỏ StandardScaler thừa cho mô hình cây.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from common import read_22feature_dataset, feature_columns, group_kfold, check_no_group_overlap, RESULTS_DIR

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def _cv(X, y, folds, make):
    accs, f1s = [], []
    for tr, te in folds:
        p = make().fit(X[tr], y[tr]).predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average='macro'))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def run_leak_comparison(scenarios=(2, 3), n_splits=5, output_dir=RESULTS_DIR):
    rows = []
    models = {
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': lambda: XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=8,
                                         tree_method='hist', random_state=42, n_jobs=-1, verbosity=0),
    }
    print("=" * 80)
    print("⚖️  NAIVE SPLIT (RÒ RỈ) vs LEAK-FREE GROUP SPLIT – 22 đặc trưng")
    print("=" * 80)
    for sc in scenarios:
        df = read_22feature_dataset(sc)
        feats = feature_columns(df)
        classes = sorted(df['Category'].unique())
        y = df['Category'].map({c: i for i, c in enumerate(classes)}).values
        X = df[feats].values
        groups = df['SampleID'].values
        naive = list(StratifiedKFold(n_splits, shuffle=True, random_state=42).split(X, y))
        grp = group_kfold(y, groups, n_splits=n_splits, seed=42)
        for tr, te in grp:
            check_no_group_overlap(groups, tr, te)
        mal = df['Class'].values != 'Benign'
        leak = np.mean([len(set(groups[te][mal[te]]) & set(groups[tr][mal[tr]])) /
                        max(1, len(set(groups[te][mal[te]]))) for tr, te in naive])
        print(f"\nKịch bản {sc}: tệp mã độc trong test đã có dump ở train (naive) = {leak*100:.2f}%")
        for name, make in models.items():
            for proto, folds in [('Naive (leaky)', naive), ('Group SampleID (leak-free)', grp)]:
                a, sa, f, sf = _cv(X, y, folds, make)
                rows.append({'Scenario': sc, 'Model': name, 'Protocol': proto,
                             'Accuracy_Mean': a, 'Accuracy_Std': sa, 'Macro_F1_Mean': f, 'Macro_F1_Std': sf,
                             'Naive_Binary_Leak_Rate': leak})
                print(f"   {name:14s} | {proto:27s} | Acc {a*100:6.2f}±{sa*100:.2f}% | F1 {f*100:6.2f}±{sf*100:.2f}%")
    out = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, 'leak_vs_leakfree_comparison.csv'), index=False)
    return out


if __name__ == '__main__':
    run_leak_comparison()
