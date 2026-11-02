# -*- coding: utf-8 -*-
"""
Tiện ích dùng chung cho toàn bộ pipeline "new feature" (22 đặc trưng tàng hình) và
benchmark không rò rỉ:

- Đường dẫn TƯƠNG ĐỐI theo vị trí repo (bỏ đường dẫn tuyệt đối D:\\... cũ).
- Đọc MalMem2022.csv gốc, tách Category -> (Family, SampleID/hash, DumpIdx).
- Sinh các tập Dataset_22features*.csv (trước đây không có script nào tạo ra).
- Chia train/test KHÔNG rò rỉ theo nhóm SampleID (mọi dump của cùng một mã độc
  chỉ nằm ở train HOẶC test) + phân tầng theo nhãn (StratifiedGroupKFold).
- Đặt seed cho numpy / random / torch để kết quả tái lập được.
"""

import os
import re
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_FEAT_DIR = os.path.join(ROOT_DIR, "new feature")
RESULTS_DIR = os.path.join(NEW_FEAT_DIR, "results")


def _resolve_raw_csv():
    """Cho phép đặt MalMem2022.csv ở gốc repo hoặc chỉ định qua biến môi trường MALMEM_CSV."""
    env = os.environ.get("MALMEM_CSV")
    if env:
        return env
    for name in ("MalMem2022.csv", "Obfuscated-MalMem2022.csv"):
        p = os.path.join(ROOT_DIR, name)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT_DIR, "MalMem2022.csv")


RAW_CSV = _resolve_raw_csv()

# 22 đặc trưng "tàng hình" (ldrmodules + malfind + psxview)
STEALTH_22_FEATURES = [
    'ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem',
    'ldrmodules.not_in_load_avg', 'ldrmodules.not_in_init_avg', 'ldrmodules.not_in_mem_avg',
    'malfind.ninjections', 'malfind.uniqueInjections',
    'psxview.not_in_pslist', 'psxview.not_in_eprocess_pool', 'psxview.not_in_ethread_pool',
    'psxview.not_in_pspcid_list', 'psxview.not_in_csrss_handles', 'psxview.not_in_session',
    'psxview.not_in_deskthrd',
    'psxview.not_in_pslist_false_avg', 'psxview.not_in_eprocess_pool_false_avg',
    'psxview.not_in_ethread_pool_false_avg', 'psxview.not_in_pspcid_list_false_avg',
    'psxview.not_in_csrss_handles_false_avg', 'psxview.not_in_session_false_avg',
    'psxview.not_in_deskthrd_false_avg',
]
# Đặc trưng dạng tỷ lệ (nằm trong [0, 1]) – dùng để chặn trên mẫu sinh bởi GAN
RATIO_FEATURES = [c for c in STEALTH_22_FEATURES if c.endswith('_avg')]
META_COLS = ['Class', 'Category', 'Label', 'Filename', 'SampleID', 'Family', 'DumpIdx',
             'Is_Synthetic', 'Split_Role']

_CAT_RE = re.compile(r'^([^-]+)-([^-]+)-(.+)-(\d+)\.raw$')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def extract_sample_id(fn):
    """Mã băm của tệp nhị phân, bỏ hậu tố dump '-<n>.raw'."""
    m = re.match(r'^(.*)-(\d+)\.raw$', str(fn))
    return m.group(1) if m else str(fn)


def load_raw(path=None, drop_constant=True):
    """
    Đọc MalMem2022.csv gốc và bổ sung metadata:
      Group     : Benign / Ransomware / Spyware / Trojan
      Family    : Benign hoặc '<Group>-<Family>' (16 lớp)
      SampleID  : hash của tệp mã độc (Benign: id riêng cho từng dòng)
      Filename  : '<hash>-<n>.raw' (Benign: rỗng)
    """
    path = path or RAW_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy tải MalMem2022.csv đặt vào gốc repo "
            f"hoặc đặt biến môi trường MALMEM_CSV=<đường dẫn>.")
    df = pd.read_csv(path)
    parsed = df['Category'].astype(str).str.extract(_CAT_RE)
    is_mal = parsed[0].notna()
    df['Group'] = np.where(is_mal, parsed[0], 'Benign')
    df['Family'] = np.where(is_mal, parsed[0] + '-' + parsed[1], 'Benign')
    df['Filename'] = np.where(is_mal, parsed[2] + '-' + parsed[3] + '.raw', '')
    df['DumpIdx'] = np.where(is_mal, pd.to_numeric(parsed[3], errors='coerce'), -1).astype(int)
    # Benign không có hash -> mỗi dòng là một nhóm riêng (giả định độc lập)
    df['SampleID'] = np.where(is_mal, parsed[2], 'benign_' + df.index.astype(str))
    feat_cols = [c for c in df.columns
                 if c not in ('Category', 'Class', 'Group', 'Family', 'Filename', 'DumpIdx', 'SampleID')]
    if drop_constant:
        nunique = df[feat_cols].nunique()
        const = nunique[nunique <= 1].index.tolist()
        df = df.drop(columns=const)
    return df


def raw_feature_columns(df):
    return [c for c in df.columns
            if c not in ('Category', 'Class', 'Group', 'Family', 'Filename', 'DumpIdx', 'SampleID')]


def scenario_labels(df, scenario):
    if scenario == 1:
        return df['Class']
    if scenario == 2:
        return df['Group']
    if scenario == 3:
        return df['Family']
    raise ValueError(scenario)


def build_22feature_datasets(df_raw=None, out_dir=NEW_FEAT_DIR, verbose=True):
    """
    Tạo:
      new feature/Dataset_22features.csv              (Class, Category=Family 16 lớp, Filename)
      new feature/Dataset_22features_Scenario_2.csv   (Category = 4 nhóm)
      new feature/Dataset_22features_Scenario_3.csv   (Category = 16 lớp)
    Giữ nguyên toàn bộ dòng (KHÔNG drop_duplicates) để có thể chia theo nhóm SampleID;
    việc xử lý trùng lặp được thực hiện sau khi chia (xem leakage_analysis.py).
    """
    df_raw = load_raw() if df_raw is None else df_raw
    os.makedirs(out_dir, exist_ok=True)
    base = df_raw[STEALTH_22_FEATURES].copy()
    base['Class'] = df_raw['Class'].values
    base['Filename'] = df_raw['Filename'].values
    base['SampleID'] = df_raw['SampleID'].values

    master = base.copy()
    master['Category'] = df_raw['Family'].values
    master.to_csv(os.path.join(out_dir, 'Dataset_22features.csv'), index=False)

    s2 = base.copy()
    s2['Category'] = df_raw['Group'].values
    s2.to_csv(os.path.join(out_dir, 'Dataset_22features_Scenario_2.csv'), index=False)

    s3 = master
    s3.to_csv(os.path.join(out_dir, 'Dataset_22features_Scenario_3.csv'), index=False)
    if verbose:
        print(f"✓ Đã tạo 3 file Dataset_22features*.csv ({len(master):,} dòng) tại: {out_dir}")
    return master


def ensure_22feature_datasets():
    need = [os.path.join(NEW_FEAT_DIR, f) for f in
            ('Dataset_22features.csv', 'Dataset_22features_Scenario_2.csv',
             'Dataset_22features_Scenario_3.csv')]
    if not all(os.path.exists(p) for p in need):
        build_22feature_datasets()


def read_22feature_dataset(scenario=None):
    """Đọc tập 22 đặc trưng; tự sinh nếu chưa có. Bổ sung SampleID nếu file cũ thiếu."""
    ensure_22feature_datasets()
    name = 'Dataset_22features.csv' if scenario is None else f'Dataset_22features_Scenario_{scenario}.csv'
    df = pd.read_csv(os.path.join(NEW_FEAT_DIR, name))
    if 'SampleID' not in df.columns:
        sid = df['Filename'].fillna('').astype(str).map(extract_sample_id)
        df['SampleID'] = np.where(sid == '', 'benign_' + df.index.astype(str), sid)
    df['Filename'] = df['Filename'].fillna('')
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def group_train_test_split(y, groups, test_size=0.2, seed=42):
    """
    Chia 80/20 KHÔNG rò rỉ: phân tầng theo nhãn + mọi dump cùng SampleID ở cùng một phía.
    Trả về (train_idx, test_idx) dạng vị trí (iloc).
    """
    n_splits = int(round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups)))
    return tr, te


def group_kfold(y, groups, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups)))


def check_no_group_overlap(groups, tr_idx, te_idx):
    g = np.asarray(groups)
    overlap = set(g[tr_idx]) & set(g[te_idx])
    assert not overlap, f"Rò rỉ nhóm: {len(overlap)} SampleID xuất hiện ở cả train và test!"
    return True
