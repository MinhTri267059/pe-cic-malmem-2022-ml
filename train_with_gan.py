# -*- coding: utf-8 -*-
"""
Huấn luyện các mô hình phân loại trên tập 22 đặc trưng (new feature), kết hợp làm phong phú
dữ liệu bằng Conditional WGAN-GP với các mức 200 - 2000 mẫu/lớp.

SỬA LỖI so với phiên bản trước:
  1. Chia train/test KHÔNG rò rỉ theo SampleID (StratifiedGroupKFold) thay vì
     train_test_split ngẫu nhiên theo dòng (cũ: ~100% mẫu mã độc trong test đã có dump ở train).
  2. Đường dẫn tương đối (bỏ D:\\...), tự sinh Dataset_22features_*.csv nếu chưa có.
  3. Đặt seed cho numpy/torch -> tái lập được.
  4. Xuất CSV tăng cường có cột provenance Is_Synthetic / Split_Role (code cũ không ghi,
     nhưng file CSV đã commit lại có -> code và kết quả không khớp).
  5. Đánh giá chất lượng GAN thêm TSTR (Train-on-Synthetic, Test-on-Real) và khoảng tin cậy
     bootstrap 95% cho chênh lệch Macro-F1 so với baseline (phân biệt cải thiện thật vs nhiễu).
  6. Không chuẩn hóa lại cho mô hình cây (không cần) – chỉ MLP dùng StandardScaler fit trên train.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from common import (read_22feature_dataset, feature_columns, group_train_test_split,
                    check_no_group_overlap, set_seed, RESULTS_DIR)
from gan_augmentation import TabularCWGAN_GP, evaluate_gan_quality

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
warnings.filterwarnings('ignore')
N_JOBS = os.cpu_count() or 4


# -----------------------------------------------------------------------------
# 1. MÔ HÌNH DEEP NEURAL NETWORK (MLP CẢI TIẾN)
# -----------------------------------------------------------------------------
class DeepTabularClassifier(nn.Module):
    """Mạng học sâu MLP với Residual connections, LayerNorm và Dropout."""
    def __init__(self, in_features=22, num_classes=4, hidden_dim=256):
        super(DeepTabularClassifier, self).__init__()
        self.in_block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_ln = nn.LayerNorm(hidden_dim)
        self.res1_act = nn.LeakyReLU(0.2, inplace=True)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.out_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h = self.in_block(x)
        res = self.res1_act(self.res1_ln(self.res1_fc1(h)))
        h = h + self.res1_fc2(res)
        return self.out_block(h)


class PyTorchMLPWrapper:
    """Wrapper chuẩn sklearn cho DeepTabularClassifier (tự chuẩn hóa bên trong, fit trên train)."""
    def __init__(self, in_features=22, num_classes=4, epochs=25, batch_size=256, lr=1e-3,
                 device=None, seed=42):
        self.in_features = in_features
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        self.model = DeepTabularClassifier(self.in_features, self.num_classes).to(self.device)
        self.model.train()
        loader = DataLoader(TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                                          torch.tensor(np.asarray(y), dtype=torch.long)),
                            batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                              total_steps=self.epochs * len(loader))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        for _ in range(self.epochs):
            for b_x, b_y in loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(b_x), b_y)
                loss.backward()
                optimizer.step()
                sched.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        with torch.no_grad():
            out = self.model(torch.tensor(Xs, dtype=torch.float32, device=self.device))
            return torch.softmax(out, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# -----------------------------------------------------------------------------
# 2. HÀM TẢI VÀ CHUẨN BỊ DỮ LIỆU (KHÔNG RÒ RỈ)
# -----------------------------------------------------------------------------
def load_scenario_data(data_path=None, scenario=2, seed=42, return_groups=False):
    """
    Đọc dữ liệu 22 features và chia train/test 80/20 theo NHÓM SampleID (không rò rỉ snapshot).
    data_path được giữ để tương thích ngược (bỏ qua nếu None).
    """
    print(f"\n==================================================================")
    print(f"📂 ĐỌC DỮ LIỆU 22 FEATURES - KỊCH BẢN {scenario}")
    print(f"==================================================================")
    df = pd.read_csv(data_path) if data_path and os.path.exists(data_path) else read_22feature_dataset(scenario)
    if 'SampleID' not in df.columns:
        df = read_22feature_dataset(scenario)
    feat_cols = feature_columns(df)
    y_raw = df['Category']
    classes = sorted(y_raw.unique())
    label_mapping = {c: i for i, c in enumerate(classes)}
    y_mapped = y_raw.map(label_mapping)
    groups = df['SampleID'].values

    print(f"✓ Tổng số mẫu: {len(df):,} | Số đặc trưng: {len(feat_cols)}")
    print(f"✓ Phân phối các lớp ({len(classes)} lớp):")
    for c in classes:
        print(f"    • {c:25s}: {(y_raw == c).sum():,} mẫu ({(y_raw == c).mean()*100:.1f}%)")

    tr, te = group_train_test_split(y_mapped.values, groups, test_size=0.2, seed=seed)
    check_no_group_overlap(groups, tr, te)
    print(f"✓ Chia theo nhóm SampleID: train {len(tr):,} | test {len(te):,} | "
          f"0 SampleID mã độc trùng giữa train/test")

    X_train, X_test = df[feat_cols].iloc[tr].reset_index(drop=True), df[feat_cols].iloc[te].reset_index(drop=True)
    y_train, y_test = y_mapped.iloc[tr].reset_index(drop=True), y_mapped.iloc[te].reset_index(drop=True)
    if return_groups:
        return X_train, X_test, y_train, y_test, label_mapping, feat_cols, groups[tr], groups[te]
    return X_train, X_test, y_train, y_test, label_mapping, feat_cols


# -----------------------------------------------------------------------------
# 3. ĐÁNH GIÁ
# -----------------------------------------------------------------------------
def evaluate_single_model(name, model, X_tr, y_tr, X_te, y_te, sample_weight=None):
    t0 = time.time()
    if sample_weight is not None:
        try:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    y_pred = np.asarray(model.predict(X_te)).ravel().astype(int)
    elapsed = time.time() - t0
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_te, y_pred),
        'Macro_Precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y_te, y_pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(y_te, y_pred, average='macro', zero_division=0),
        'Weighted_F1': f1_score(y_te, y_pred, average='weighted', zero_division=0),
        'Train_Time_Sec': round(elapsed, 2)
    }, y_pred


def bootstrap_delta_f1(y_true, pred_a, pred_b, groups, n_boot=300, seed=0):
    """CI 95% cho Macro-F1(b) - Macro-F1(a), bootstrap theo NHÓM SampleID (dump tương quan)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == g)[0] for g in range(len(ug))]
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[g] for g in pick])
        deltas.append(f1_score(y_true[idx], pred_b[idx], average='macro', zero_division=0) -
                      f1_score(y_true[idx], pred_a[idx], average='macro', zero_division=0))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def get_models(n_feat, num_classes):
    return {
        'XGBoost (Tuned)': XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=8,
                                         subsample=0.85, colsample_bytree=0.85, random_state=42,
                                         tree_method='hist', n_jobs=N_JOBS, verbosity=0),
        'LightGBM (Tuned)': LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                           subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
                                           random_state=42, n_jobs=N_JOBS, verbose=-1),
        'CatBoost (Tuned)': CatBoostClassifier(iterations=400, learning_rate=0.08, depth=7,
                                               random_seed=42, thread_count=N_JOBS, verbose=0),
        'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                                random_state=42, n_jobs=N_JOBS),
        'Deep Neural Net (MLP)': PyTorchMLPWrapper(in_features=n_feat, num_classes=num_classes,
                                                   epochs=20, batch_size=512, lr=2e-3),
    }


# -----------------------------------------------------------------------------
# 4. THỬ NGHIỆM GAN AUGMENTATION (200 - 2000 MẪU) TRÊN GIAO THỨC KHÔNG RÒ RỈ
# -----------------------------------------------------------------------------
def run_gan_enrichment_experiment(scenario=2, gan_levels=(200, 500, 1000, 2000),
                                  epochs_gan=60, output_dir=RESULTS_DIR, seed=42,
                                  models_filter=None):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    (X_train, X_test, y_train, y_test, label_mapping, feat_cols,
     g_train, g_test) = load_scenario_data(None, scenario, seed=seed, return_groups=True)
    num_classes = len(label_mapping)
    inv_mapping = {v: k for k, v in label_mapping.items()}

    # 1. Huấn luyện Conditional WGAN-GP CHỈ trên tập Train
    gan = TabularCWGAN_GP(num_features=len(feat_cols), num_classes=num_classes,
                          z_dim=64, embed_dim=16, hidden_dim=256, lr=2e-4)
    gan.fit(X_train, y_train, epochs=epochs_gan, batch_size=512, n_critic=5, verbose_every=10)
    gan.save_checkpoint(os.path.join(output_dir, f"cwgan_gp_scenario_{scenario}.pt"))

    # 2. Chất lượng mẫu sinh: thống kê + PCA + TSTR
    eval_dict = {c: 300 for c in range(num_classes)}
    df_eval_x, s_eval_y = gan.generate(eval_dict, seed=seed)
    gan_quality = evaluate_gan_quality(X_train, y_train, df_eval_x, s_eval_y, label_mapping, output_dir,
                                       tag=f"scenario_{scenario}")
    n_tstr = {c: int(max(300, (y_train == c).sum())) for c in range(num_classes)}
    X_syn_only, y_syn_only = gan.generate(n_tstr, seed=seed + 1)
    tstr = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=N_JOBS,
                          random_state=42, verbose=-1).fit(X_syn_only, y_syn_only)
    trtr = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=N_JOBS,
                          random_state=42, verbose=-1).fit(X_train, y_train)
    tstr_f1 = f1_score(y_test, tstr.predict(X_test), average='macro')
    trtr_f1 = f1_score(y_test, trtr.predict(X_test), average='macro')
    print(f"   • TSTR Macro-F1 (train GAN-only, test real): {tstr_f1*100:.2f}% "
          f"vs TRTR (train real): {trtr_f1*100:.2f}%  -> tỷ lệ {tstr_f1/max(trtr_f1,1e-9):.2f}")
    gan_quality.update({'TSTR_Macro_F1': tstr_f1, 'TRTR_Macro_F1': trtr_f1})
    pd.Series(gan_quality).to_json(os.path.join(output_dir, f"gan_quality_scenario_{scenario}.json"),
                                   indent=2)

    all_results = []
    preds = {}   # (model, level) -> y_pred

    def run_level(level, X_tr, y_tr, desc):
        models = get_models(len(feat_cols), num_classes)
        if models_filter:
            models = {k: v for k, v in models.items() if k in models_filter}
        for name, model in models.items():
            print(f"  ⏳ {name} [{desc}] ...", end="", flush=True)
            res, y_pred = evaluate_single_model(name, model, X_tr, y_tr, X_test, y_test)
            res.update({'Augmentation_Level': level, 'Augmentation_Desc': desc, 'Scenario': scenario,
                        'Split': 'StratifiedGroup 80/20 (SampleID)'})
            if level != 0:
                lo, hi = bootstrap_delta_f1(y_test, preds[(name, 0)], y_pred, g_test)
                res.update({'Delta_Macro_F1_vs_Baseline': res['Macro_F1'] - all_results[
                    [r['Model'] for r in all_results].index(name)]['Macro_F1'],
                            'Delta_CI95_Low': lo, 'Delta_CI95_High': hi,
                            'Significant': bool(lo > 0 or hi < 0)})
            all_results.append(res)
            preds[(name, level)] = y_pred
            extra = ''
            if level != 0:
                extra = f" | ΔF1 {res['Delta_Macro_F1_vs_Baseline']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]"
            print(f" Acc {res['Accuracy']*100:.2f}% | Macro-F1 {res['Macro_F1']*100:.2f}%{extra} "
                  f"({res['Train_Time_Sec']}s)", flush=True)

    # A. BASELINE (chỉ dữ liệu thật)
    print(f"\n📊 [A] BASELINE TRÊN DỮ LIỆU THỰC TẾ (KHÔNG GAN)")
    run_level(0, X_train, y_train, 'Baseline (Real Only)')

    # B. TĂNG CƯỜNG BẰNG GAN
    class_counts = y_train.value_counts()
    for n_samples in gan_levels:
        print(f"\n✨ [B] TĂNG CƯỜNG: +{n_samples} MẪU GAN / LỚP THIỂU SỐ")
        gen_dict = {c: (n_samples if class_counts.get(c, 0) < class_counts.max() else 0)
                    for c in range(num_classes)}
        df_synth_x, s_synth_y = gan.generate(gen_dict, seed=seed + n_samples)
        X_train_aug = pd.concat([X_train, df_synth_x], axis=0, ignore_index=True)
        y_train_aug = pd.concat([y_train, s_synth_y], axis=0, ignore_index=True)
        print(f"  ✓ Train: {len(X_train):,} -> {len(X_train_aug):,} (+{len(df_synth_x):,} mẫu sinh)")

        if n_samples == 1000:
            exp = X_train_aug.copy()
            exp['Category'] = y_train_aug.map(inv_mapping).values
            exp['Is_Synthetic'] = np.r_[np.zeros(len(X_train), int), np.ones(len(df_synth_x), int)]
            exp['Split_Role'] = 'train'
            test_exp = X_test.copy()
            test_exp['Category'] = y_test.map(inv_mapping).values
            test_exp['Is_Synthetic'] = 0
            test_exp['Split_Role'] = 'test'
            pd.concat([exp, test_exp], ignore_index=True).to_csv(
                os.path.join(output_dir, f"Dataset_22features_Scenario_{scenario}_GAN_augmented_{n_samples}.csv"),
                index=False)

        run_level(n_samples, X_train_aug, y_train_aug, f'+{n_samples} GAN Samples/Class')

    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(output_dir, f"Metrics_GAN_Scenario_{scenario}.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ ĐÃ LƯU: {out_csv}")

    # Biểu đồ Macro-F1 theo mức GAN
    plt.figure(figsize=(10, 6))
    for m_name in res_df['Model'].unique():
        sub = res_df[res_df['Model'] == m_name]
        plt.plot(sub['Augmentation_Level'], sub['Macro_F1'] * 100, marker='s', linewidth=2, label=m_name)
    plt.title(f'GAN augmentation vs Macro-F1 (Kịch bản {scenario}, chia theo SampleID)', fontsize=12)
    plt.xlabel('Số mẫu GAN trên mỗi lớp thiểu số')
    plt.ylabel('Macro-F1 (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"gan_macro_f1_curve_scenario_{scenario}.png"), dpi=150)
    plt.close()

    ref = 'XGBoost (Tuned)' if ('XGBoost (Tuned)', 0) in preds else res_df['Model'].iloc[0]
    if (ref, 0) in preds and (ref, 1000) in preds:
        c_labels = [inv_mapping[i] for i in range(num_classes)]
        c_labels = [c.replace('Ransomware-', 'R-').replace('Spyware-', 'S-').replace('Trojan-', 'T-')
                    for c in c_labels]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, lvl, cmap, title in [(axes[0], 0, 'Blues', 'Baseline'), (axes[1], 1000, 'Greens', '+1000 GAN')]:
            sns.heatmap(confusion_matrix(y_test, preds[(ref, lvl)]), annot=num_classes <= 6, fmt='d',
                        cmap=cmap, ax=ax, xticklabels=c_labels, yticklabels=c_labels)
            ax.set_title(f'{ref} {title} - Kịch bản {scenario}')
            ax.set_xlabel('Dự đoán')
        axes[0].set_ylabel('Thực tế')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_compare_scenario_{scenario}.png"), dpi=150)
        plt.close()

    print(f"\n🏆 MACRO-F1 (%) – KỊCH BẢN {scenario}:")
    print((res_df.pivot(index='Model', columns='Augmentation_Level', values='Macro_F1') * 100).round(2).to_string())
    return res_df, gan


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--epochs', type=int, default=60)
    args = ap.parse_args()
    for sc in args.scenarios:
        run_gan_enrichment_experiment(sc, epochs_gan=args.epochs)
