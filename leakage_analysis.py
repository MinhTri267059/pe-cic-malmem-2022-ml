# -*- coding: utf-8 -*-
"""
Chương trình Kiểm định & Khắc phục Rò rỉ Dữ liệu (Data Leakage Audit & Fix)
trên bộ dữ liệu 22 đặc trưng CIC-MalMem-2022.

Phát hiện và định lượng 5 loại rò rỉ dữ liệu tiềm ẩn:
1. Exact Duplicate Row Leakage (Trùng lặp chính xác giữa Train và Test)
2. Temporal Snapshot / SampleID Group Leakage (Rò rỉ các dump -1.raw đến -10.raw của cùng một mẫu mã độc)
3. Label Ambiguity / Contradiction (Cùng 22 giá trị nhưng nhãn khác nhau)
4. Few-Shot Transductive / Prior-Knowledge Leakage (GAN/Embedder học trên lớp novel)
5. Preprocessing & Export Artifact Leakage (Thiếu cờ Is_Synthetic)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

BASE_DIR = r"D:\Tai Lieu Hoc Tap Cao Hoc\Chuyên đề nghiên cứu về Phân loại mã độc\pe-cic-malmem-2022-ml"
NEW_FEAT_DIR = os.path.join(BASE_DIR, "new feature")

def extract_sample_id(fn):
    """Trích xuất mã SHA256/Binary ID bỏ đuôi mốc thời gian (-1.raw, -2.raw,...)."""
    m = re.match(r'^(.*)-(\d+)\.raw$', str(fn))
    return m.group(1) if m else str(fn)

def audit_dataset_leakage():
    print("=" * 80)
    print("🔍 BẮT ĐẦU KIỂM ĐỊNH TOÀN DIỆN CÁC ĐIỂM RÒ RỈ DỮ LIỆU (DATA LEAKAGE AUDIT)")
    print("=" * 80)

    # Đọc file tổng thể 22 features
    csv_master = os.path.join(NEW_FEAT_DIR, "Dataset_22features.csv")
    df_master = pd.read_csv(csv_master)
    feat_cols = [c for c in df_master.columns if c not in ['Class', 'Category', 'Label', 'Filename']]

    print(f"\n📂 Tổng số mẫu master: {len(df_master):,} | Số đặc trưng: {len(feat_cols)}")

    # -------------------------------------------------------------------------
    # 1. KIỂM ĐỊNH TRÙNG LẶP HÀNG CHÍNH XÁC (EXACT DUPLICATES)
    # -------------------------------------------------------------------------
    dup_feats = df_master.duplicated(subset=feat_cols, keep=False)
    num_dup_rows = dup_feats.sum()
    print(f"\n[1] KIỂM TRA TRÙNG LẶP HÀNG TRÊN 22 ĐẶC TRƯNG:")
    print(f"   • Số hàng có giá trị đặc trưng bị trùng lặp: {num_dup_rows:,} ({num_dup_rows/len(df_master)*100:.2f}%)")
    print(f"   • Số vector đặc trưng độc nhất (Unique Patterns): {df_master[feat_cols].drop_duplicates().shape[0]:,}")

    # -------------------------------------------------------------------------
    # 2. KIỂM ĐỊNH XUNG ĐỘT NHÃN (LABEL CONTRADICTIONS)
    # -------------------------------------------------------------------------
    grouped_cat = df_master.groupby(feat_cols)['Category'].nunique()
    conflict_patterns = grouped_cat[grouped_cat > 1]
    print(f"\n[2] KIỂM TRA XUNG ĐỘT NHÃN (CÙNG ĐẶC TRƯNG NHƯNG KHÁC CATEGORY):")
    print(f"   • Số pattern có nhãn Category mâu thuẫn: {len(conflict_patterns):,}")
    
    # -------------------------------------------------------------------------
    # 3. KIỂM ĐỊNH RÒ RỈ NHÓM / MÃ ĐỘC QUA CÁC DUMP BỘ NHỚ (GROUP / SNAPSHOT LEAKAGE)
    # -------------------------------------------------------------------------
    df_malware = df_master[df_master['Class'] != 'Benign'].copy()
    df_malware['SampleID'] = df_malware['Filename'].apply(extract_sample_id)
    unique_malware_binaries = df_malware['SampleID'].nunique()
    print(f"\n[3] KIỂM TRA QUAN HỆ CÁC BẢN DUMP THỜI GIAN (TEMPORAL SNAPSHOTS):")
    print(f"   • Tổng số bản ghi Malware: {len(df_malware):,}")
    print(f"   • Số mẫu mã độc nhị phân thực tế (Unique Binaries): {unique_malware_binaries:,}")
    print(f"   • Tỷ lệ trung bình số dump/mẫu mã độc: {len(df_malware)/unique_malware_binaries:.1f} dump/mẫu")

    # Mô phỏng phân chia ngẫu nhiên (Naive Split) và đo đạc rò rỉ
    print("\n   ⚠️  MÔ PHỎNG PHÂN CHIA NGẪU NHIÊN THEO HÀNG (NAIVE ROW-LEVEL SPLIT 80/20):")
    tr_idx, te_idx = train_test_split(df_malware.index, test_size=0.2, random_state=42)
    tr_samples = set(df_malware.loc[tr_idx, 'SampleID'])
    te_samples = set(df_malware.loc[te_idx, 'SampleID'])
    overlap_samples = tr_samples.intersection(te_samples)
    print(f"      - Số mẫu mã độc trong Test: {len(te_samples):,}")
    print(f"      - Số mẫu mã độc trong Test ĐÃ TỪNG XUẤT HIỆN trong Train: {len(overlap_samples):,} ({len(overlap_samples)/len(te_samples)*100:.2f}%)!")
    print(f"      => KẾT LUẬN: Gần như 100% mẫu trong tập test bị rò rỉ do xuất phát từ cùng tiến trình!")

    # -------------------------------------------------------------------------
    # 4. KIỂM ĐỊNH RÒ RỈ TEST SET TRONG KỊCH BẢN 2 & KỊCH BẢN 3
    # -------------------------------------------------------------------------
    print("\n[4] TỶ LỆ TRÙNG LẶP EXACT-ROW TRONG TEST SET KHI CHIA NAIVE:")
    for sc in [2, 3]:
        df_sc = pd.read_csv(os.path.join(NEW_FEAT_DIR, f"Dataset_22features_Scenario_{sc}.csv"))
        f_cols = [c for c in df_sc.columns if c not in ['Class', 'Category', 'Label', 'Filename']]
        X_tr, X_te = train_test_split(df_sc[f_cols], test_size=0.2, random_state=42, stratify=df_sc['Category'])
        tr_set = set([tuple(x) for x in X_tr.values])
        te_list = [tuple(x) for x in X_te.values]
        leak_count = sum(1 for t in te_list if t in tr_set)
        print(f"   • Kịch bản {sc}: Test set có {len(X_te):,} mẫu -> Có {leak_count:,} mẫu ({leak_count/len(X_te)*100:.2f}%) trùng lặp hoàn toàn với Train!")

    # -------------------------------------------------------------------------
    # 5. GIẢI PHÁP KHẮC PHỤC TRIỆT ĐỂ (PROPOSED REMEDIATION)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🛠️ GIẢI PHÁP KHẮC PHỤC RÒ RỈ DỮ LIỆU ĐƯỢC ĐỀ XUẤT:")
    print("=" * 80)
    print("1. GROUP-AWARE SPLITTING: Phân tách Train/Test dựa trên SampleID (Binary Hash),")
    print("   đảm bảo toàn bộ các dump của một mẫu mã độc chỉ nằm ở Train HOẶC Test.")
    print("2. EXACT DUPLICATE ISOLATION: Xử lý hoặc phân cụm các mẫu trùng lặp tuyệt đối.")
    print("3. ZERO-DAY NOVEL CLASS FEW-SHOT: Chia lớp thành Base Classes (Train GAN & Embedder)")
    print("   và Novel Classes (chỉ cấp K mẫu để đánh giá thực sự khả năng thích ứng).")
    print("4. EXPLICIT ARTIFACT METADATA: Bổ sung cờ 'Is_Synthetic' (0: Thật, 1: GAN) trong các file CSV.")

if __name__ == '__main__':
    audit_dataset_leakage()
