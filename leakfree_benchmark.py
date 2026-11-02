# -*- coding: utf-8 -*-
"""
BENCHMARK KHÔNG RÒ RỈ & TỐI ƯU ĐỘ CHÍNH XÁC (CIC-MalMem-2022)
===============================================================

Mục tiêu: đạt độ chính xác cao nhất NHƯNG trung thực (không rò rỉ dữ liệu).

Giao thức:
  • 5-fold StratifiedGroupKFold theo SampleID (hash tệp mã độc): 10 dump bộ nhớ của
    cùng một mẫu luôn nằm cùng một phía -> mô hình phải tổng quát hóa sang MẪU MỚI.
  • Scaler / mô hình chỉ fit trên fold train.

Các hướng cải tiến được so sánh (ablation):
  A. Tập đặc trưng:  22 tàng hình  |  55 đầy đủ  |  55 + đặc trưng kỹ nghệ (ratio)
  B. Mô hình:        RF, ExtraTrees, XGBoost, LightGBM, CatBoost, Soft-Voting Ensemble
  C. Mức suy luận:   theo từng dump (dump-level)  |  theo mẫu mã độc (sample-level:
                     trung bình xác suất của mọi dump cùng SampleID – mô phỏng việc
                     chụp RAM nhiều lần khi phân tích một tệp nghi vấn)
  D. Tham chiếu:     Naive random split (có rò rỉ) để định lượng độ lạc quan giả.

Chạy:  python3 leakfree_benchmark.py [--scenarios 2 3] [--quick]
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from common import (load_raw, raw_feature_columns, scenario_labels, STEALTH_22_FEATURES,
                    group_kfold, check_no_group_overlap, set_seed, RESULTS_DIR)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
warnings.filterwarnings('ignore')
N_JOBS = os.cpu_count() or 4


# -----------------------------------------------------------------------------
# 1. KỸ NGHỆ ĐẶC TRƯNG (chỉ dùng phép biến đổi theo từng dòng -> không rò rỉ)
# -----------------------------------------------------------------------------
def engineer_features(X):
    X = X.copy()
    eps = 1.0
    nh = X.get('handles.nhandles')
    if nh is not None:
        for c in [c for c in X.columns if c.startswith('handles.n') and c != 'handles.nhandles']:
            X[f'ratio.{c}_per_handle'] = X[c] / (nh + eps)
    if 'pslist.nproc' in X:
        npr = X['pslist.nproc'] + eps
        for c in ['dlllist.ndlls', 'handles.nhandles', 'malfind.ninjections', 'ldrmodules.not_in_load',
                  'svcscan.process_services', 'modules.nmodules', 'callbacks.ncallbacks']:
            if c in X:
                X[f'ratio.{c}_per_proc'] = X[c] / npr
        if 'pslist.nppid' in X:
            X['ratio.ppid_per_proc'] = X['pslist.nppid'] / npr
    if 'malfind.ninjections' in X:
        inj = X['malfind.ninjections'] + eps
        for c in ['malfind.commitCharge', 'malfind.protection', 'malfind.uniqueInjections']:
            if c in X:
                X[f'ratio.{c}_per_inj'] = X[c] / inj
    if 'svcscan.nservices' in X:
        ns = X['svcscan.nservices'] + eps
        for c in ['svcscan.kernel_drivers', 'svcscan.fs_drivers', 'svcscan.process_services',
                  'svcscan.shared_process_services', 'svcscan.nactive']:
            if c in X:
                X[f'ratio.{c}_per_svc'] = X[c] / ns
    ldr = [c for c in ['ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem'] if c in X]
    if ldr:
        X['sum.ldrmodules'] = X[ldr].sum(axis=1)
        if 'dlllist.ndlls' in X:
            X['ratio.ldr_hidden_per_dll'] = X['sum.ldrmodules'] / (X['dlllist.ndlls'] + eps)
    psx = [c for c in X.columns if c.startswith('psxview.not_in_') and not c.endswith('_avg')]
    if psx:
        X['sum.psxview_hidden'] = X[psx].sum(axis=1)
    if 'callbacks.nanonymous' in X and 'callbacks.ncallbacks' in X:
        X['ratio.anon_callbacks'] = X['callbacks.nanonymous'] / (X['callbacks.ncallbacks'] + eps)
    return X


# -----------------------------------------------------------------------------
# 2. MÔ HÌNH
# -----------------------------------------------------------------------------
def get_models(n_classes, quick=False):
    n_est = 300 if quick else 600
    obj = 'binary' if n_classes == 2 else 'multiclass'
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=n_est // 2, min_samples_leaf=1,
                                               max_features='sqrt', n_jobs=N_JOBS, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=n_est // 2, min_samples_leaf=1,
                                           max_features='sqrt', n_jobs=N_JOBS, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=8,
                                 subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
                                 reg_lambda=1.0, tree_method='hist', n_jobs=N_JOBS,
                                 random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(objective=obj, n_estimators=n_est, learning_rate=0.04,
                                   num_leaves=63, min_child_samples=20, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                   n_jobs=N_JOBS, random_state=42, verbose=-1),
    }
    if not quick:
        models['CatBoost'] = CatBoostClassifier(iterations=800, learning_rate=0.08, depth=8,
                                                random_seed=42, thread_count=N_JOBS, verbose=0)
    return models


def predict_proba_aligned(model, X, n_classes):
    p = model.predict_proba(X)
    if p.shape[1] != n_classes:  # lớp vắng mặt trong fold train
        full = np.zeros((len(X), n_classes))
        full[:, model.classes_.astype(int)] = p
        p = full
    return p


def aggregate_by_group(proba, groups):
    """Trung bình xác suất của mọi dump cùng SampleID (sample-level inference)."""
    df = pd.DataFrame(proba)
    df['g'] = np.asarray(groups)
    mean = df.groupby('g').transform('mean').values
    return mean


# -----------------------------------------------------------------------------
# 3. ĐÁNH GIÁ GROUP-CV
# -----------------------------------------------------------------------------
def run_cv(X, y, groups, model_names, n_classes, folds, quick=False):
    """Trả về OOF-probabilities cho từng mô hình (và ensemble)."""
    oof = {m: np.zeros((len(y), n_classes)) for m in model_names}
    times = {m: 0.0 for m in model_names}
    for k, (tr, te) in enumerate(folds):
        check_no_group_overlap(groups, tr, te)
        models = get_models(n_classes, quick)
        for m in model_names:
            t0 = time.time()
            mdl = models[m]
            mdl.fit(X.iloc[tr], y[tr])
            oof[m][te] = predict_proba_aligned(mdl, X.iloc[te], n_classes)
            times[m] += time.time() - t0
    return oof, times


def score(y, proba, groups=None):
    pred = proba.argmax(1)
    res = {'Accuracy': accuracy_score(y, pred),
           'Macro_F1': f1_score(y, pred, average='macro'),
           'Weighted_F1': f1_score(y, pred, average='weighted')}
    if groups is not None:
        agg = aggregate_by_group(proba, groups)
        pa = agg.argmax(1)
        res.update({'Sample_Accuracy': accuracy_score(y, pa),
                    'Sample_Macro_F1': f1_score(y, pa, average='macro')})
    return res


def benchmark_scenario(df, scenario, quick=False, n_splits=5):
    y_raw = scenario_labels(df, scenario)
    classes = sorted(y_raw.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(cmap).values
    groups = df['SampleID'].values
    n_classes = len(classes)

    full_cols = raw_feature_columns(df)
    feature_sets = {
        '22_stealth': df[STEALTH_22_FEATURES],
        '55_full': df[full_cols],
        '55_full+eng': engineer_features(df[full_cols]),
    }
    folds = group_kfold(y, groups, n_splits=n_splits, seed=42)
    model_names = list(get_models(n_classes, quick).keys())

    rows, best = [], None
    for fs_name, X in feature_sets.items():
        print(f"\n  ▶ Kịch bản {scenario} | Tập đặc trưng: {fs_name} ({X.shape[1]} cột)", flush=True)
        oof, times = run_cv(X, y, groups, model_names, n_classes, folds, quick)
        # Soft-voting ensemble của các mô hình boosting + ExtraTrees
        ens_members = [m for m in ['LightGBM', 'XGBoost', 'CatBoost', 'ExtraTrees'] if m in oof]
        oof['SoftVote(' + '+'.join(ens_members) + ')'] = np.mean([oof[m] for m in ens_members], axis=0)
        for m, p in oof.items():
            r = score(y, p, groups)
            r.update({'Scenario': scenario, 'Feature_Set': fs_name, 'Model': m,
                      'Protocol': f'{n_splits}-fold StratifiedGroupKFold (SampleID)',
                      'Train_Time_Sec': round(times.get(m, sum(times[e] for e in ens_members)), 1)})
            rows.append(r)
            print(f"     {m:45s} Acc {r['Accuracy']*100:6.2f}% | F1 {r['Macro_F1']*100:6.2f}% "
                  f"| Sample-Acc {r['Sample_Accuracy']*100:6.2f}% | Sample-F1 {r['Sample_Macro_F1']*100:6.2f}%",
                  flush=True)
            if best is None or r['Accuracy'] > best[0]['Accuracy']:
                best = (r, p)

    # Tham chiếu: naive split (rò rỉ) với LightGBM trên 55_full+eng
    X = feature_sets['55_full+eng']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_naive = np.zeros((len(y), n_classes))
    for tr, te in skf.split(X, y):
        mdl = get_models(n_classes, quick=True)['LightGBM']
        mdl.fit(X.iloc[tr], y[tr])
        oof_naive[te] = predict_proba_aligned(mdl, X.iloc[te], n_classes)
    r = score(y, oof_naive)
    r.update({'Scenario': scenario, 'Feature_Set': '55_full+eng', 'Model': 'LightGBM',
              'Protocol': f'{n_splits}-fold Naive StratifiedKFold (LEAKY - tham chiếu)'})
    rows.append(r)
    print(f"     [Tham chiếu NAIVE/rò rỉ] LightGBM Acc {r['Accuracy']*100:.2f}% | F1 {r['Macro_F1']*100:.2f}%")

    return pd.DataFrame(rows), best, classes, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--quick', action='store_true', help='Ít cây hơn, bỏ CatBoost (chạy nhanh)')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_raw()
    print(f"📂 MalMem2022: {len(df):,} dòng | {df['SampleID'].str.startswith('benign_').sum():,} benign | "
          f"{df.loc[df.Class == 'Malware', 'SampleID'].nunique():,} mẫu mã độc (SampleID) duy nhất")

    all_rows, summary = [], {}
    for sc in args.scenarios:
        t0 = time.time()
        res, best, classes, y = benchmark_scenario(df, sc, quick=args.quick, n_splits=args.folds)
        res.to_csv(os.path.join(RESULTS_DIR, f'LeakFree_Benchmark_Scenario_{sc}.csv'), index=False)
        all_rows.append(res)
        b = best[0]
        summary[sc] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in b.items()}
        cm = confusion_matrix(y, best[1].argmax(1))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            os.path.join(RESULTS_DIR, f'LeakFree_BestModel_ConfusionMatrix_Scenario_{sc}.csv'))
        print(f"  ✅ Kịch bản {sc}: tốt nhất = {b['Model']} / {b['Feature_Set']} "
              f"-> Acc {b['Accuracy']*100:.2f}%, Sample-Acc {b['Sample_Accuracy']*100:.2f}% "
              f"({(time.time()-t0)/60:.1f} phút)")

    # Gộp với kết quả các kịch bản đã chạy trước đó (không ghi đè khi chạy từng kịch bản riêng)
    all_path = os.path.join(RESULTS_DIR, 'LeakFree_Benchmark_All.csv')
    best_path = os.path.join(RESULTS_DIR, 'LeakFree_Benchmark_Best.json')
    new = pd.concat(all_rows)
    if os.path.exists(all_path):
        old = pd.read_csv(all_path)
        new = pd.concat([old[~old['Scenario'].isin(args.scenarios)], new])
    new.sort_values(['Scenario', 'Protocol', 'Feature_Set']).to_csv(all_path, index=False)
    if os.path.exists(best_path):
        with open(best_path, encoding='utf-8') as f:
            old_best = json.load(f)
        old_best.update({str(k): v for k, v in summary.items()})
        summary = old_best
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
