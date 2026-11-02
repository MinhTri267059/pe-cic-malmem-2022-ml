# -*- coding: utf-8 -*-
"""
Kiểm định Rò rỉ Dữ liệu (Data Leakage Audit) trên CIC-MalMem-2022 (22 đặc trưng + 55 đặc trưng).

  1. Exact duplicate rows (trùng lặp tuyệt đối giữa train/test)
  2. Label contradictions (cùng vector đặc trưng nhưng khác nhãn -> trần độ chính xác)
  3. Snapshot / SampleID group leakage (10 dump -1.raw..-10.raw của cùng một tệp)
  4. So sánh tỷ lệ rò rỉ: Naive split vs Group split (StratifiedGroupKFold)
  5. Trần độ chính xác lý thuyết (Bayes ceiling) do xung đột nhãn

SỬA LỖI: đường dẫn tuyệt đối D:\\... -> tương đối; tự sinh dataset 22 đặc trưng; bổ sung
đo đạc trên Group split (trước chỉ đo naive split nên không kiểm chứng được giải pháp).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (read_22feature_dataset, feature_columns,
                    group_train_test_split, load_raw, raw_feature_columns, RESULTS_DIR)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def bayes_ceiling(df, feat_cols, label_col):
    """Độ chính xác tối đa đạt được nếu với mỗi vector đặc trưng ta đoán nhãn phổ biến nhất."""
    key = pd.util.hash_pandas_object(df[feat_cols], index=False)
    ct = pd.crosstab(key, df[label_col])
    return ct.max(axis=1).sum() / len(df)


def exact_overlap(df, feat_cols, tr, te):
    key = pd.util.hash_pandas_object(df[feat_cols], index=False).values
    tr_set = set(key[tr])
    return float(np.mean([k in tr_set for k in key[te]]))


def audit_dataset_leakage(output_dir=RESULTS_DIR):
    print("=" * 80)
    print("🔍 KIỂM ĐỊNH TOÀN DIỆN CÁC ĐIỂM RÒ RỈ DỮ LIỆU (DATA LEAKAGE AUDIT)")
    print("=" * 80)
    report = {}
    df = read_22feature_dataset()            # Category = 16 lớp
    feat_cols = feature_columns(df)
    print(f"\n📂 Tổng số mẫu: {len(df):,} | Số đặc trưng: {len(feat_cols)}")

    # 1. Trùng lặp
    dup = df.duplicated(subset=feat_cols, keep=False)
    n_unique = df[feat_cols].drop_duplicates().shape[0]
    print(f"\n[1] TRÙNG LẶP TRÊN 22 ĐẶC TRƯNG: {dup.sum():,} dòng ({dup.mean()*100:.2f}%) | "
          f"{n_unique:,} vector duy nhất")
    report['dup_rows_22'] = int(dup.sum())
    report['unique_vectors_22'] = int(n_unique)

    # 2. Xung đột nhãn + trần Bayes
    g = df.groupby(feat_cols)['Category'].nunique()
    report['conflict_patterns_22'] = int((g > 1).sum())
    raw = load_raw()
    full_cols = raw_feature_columns(raw)
    print(f"\n[2] XUNG ĐỘT NHÃN: {report['conflict_patterns_22']:,} vector có >1 nhãn (16 lớp)")
    print("    Trần độ chính xác lý thuyết (Bayes ceiling – kể cả khi mô hình hoàn hảo):")
    for sc, col in [(1, 'Class'), (2, 'Group'), (3, 'Family')]:
        c22 = bayes_ceiling(raw, [c for c in feat_cols], col)
        c55 = bayes_ceiling(raw, full_cols, col)
        report[f'bayes_ceiling_s{sc}_22'] = c22
        report[f'bayes_ceiling_s{sc}_55'] = c55
        print(f"      • Kịch bản {sc}: 22 đặc trưng = {c22*100:.2f}% | 55 đặc trưng = {c55*100:.2f}%")
    print("    => 22 đặc trưng tàng hình có rất nhiều vector toàn 0 / giống nhau giữa các họ -> trần thấp.")

    # 3. Snapshot leakage
    mal = df[df['Class'] != 'Benign']
    n_bin = mal['SampleID'].nunique()
    print(f"\n[3] SNAPSHOT: {len(mal):,} dòng mã độc từ {n_bin:,} tệp ({len(mal)/n_bin:.1f} dump/tệp)")
    report['malware_binaries'] = int(n_bin)

    # 4. Naive vs Group split
    print("\n[4] RÒ RỈ KHI CHIA 80/20:")
    y = df['Category'].values
    groups = df['SampleID'].values
    tr_n, te_n = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
    tr_g, te_g = group_train_test_split(y, groups, test_size=0.2, seed=42)
    for name, tr, te in [('Naive', tr_n, te_n), ('Group(SampleID)', tr_g, te_g)]:
        ms = df['Class'].values != 'Benign'
        te_bins = set(groups[te][ms[te]])
        ov = te_bins & set(groups[tr][ms[tr]])
        ex = exact_overlap(df, feat_cols, tr, te)
        print(f"   • {name:16s}: tệp mã độc test đã có ở train = {len(ov):,}/{len(te_bins):,} "
              f"({len(ov)/max(1,len(te_bins))*100:.2f}%) | dòng test trùng tuyệt đối với train = {ex*100:.2f}%")
        report[f'{name}_binary_overlap'] = len(ov) / max(1, len(te_bins))
        report[f'{name}_exact_row_overlap'] = ex
    print("   (Trùng tuyệt đối còn lại ở Group split là do các tệp KHÁC NHAU cho cùng vector – chủ yếu"
          " vector toàn 0 – không phải rò rỉ, mà là giới hạn biểu diễn của 22 đặc trưng.)")

    print("\n🛠️ GIẢI PHÁP ĐÃ ÁP DỤNG TRONG PIPELINE:")
    print("  1. StratifiedGroupKFold theo SampleID cho mọi thực nghiệm (common.group_train_test_split).")
    print("  2. Scaler/GAN/encoder chỉ fit trên train (hoặc chỉ trên support trong few-shot).")
    print("  3. Zero-day: class-disjoint + group-disjoint support/query.")
    print("  4. CSV tăng cường có cột Is_Synthetic / Split_Role.")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'leakage_audit.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == '__main__':
    audit_dataset_leakage()
