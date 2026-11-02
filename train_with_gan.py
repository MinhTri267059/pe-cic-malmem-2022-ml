# -*- coding: utf-8 -*-
"""
Chương trình huấn luyện các mô hình phân loại mới trên tập 22 đặc trưng (new feature),
kết hợp làm phong phú dữ liệu (Data Augmentation) bằng Conditional WGAN-GP với các mức 200 - 2000 mẫu.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gan_augmentation import TabularCWGAN_GP, evaluate_gan_quality

warnings.filterwarnings('ignore')


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
        out = self.out_block(h)
        return out


class PyTorchMLPWrapper:
    """Wrapper tuân thủ chuẩn sklearn cho DeepTabularClassifier."""
    def __init__(self, in_features=22, num_classes=4, epochs=25, batch_size=256, lr=1e-3, device=None):
        self.in_features = in_features
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepTabularClassifier(in_features, num_classes).to(self.device)

    def fit(self, X, y):
        # Khởi tạo lại model để huấn luyện từ đầu
        self.model = DeepTabularClassifier(self.in_features, self.num_classes).to(self.device)
        self.model.train()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        y_t = torch.tensor(np.array(y), dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            for b_x, b_y in loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
            out = self.model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
            out = self.model(X_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs


# -----------------------------------------------------------------------------
# 2. HÀM TẢI VÀ CHUẨN BỊ DỮ LIỆU
# -----------------------------------------------------------------------------
def load_scenario_data(data_path, scenario=2):
    """Đọc dữ liệu 22 features, trích xuất đặc trưng và chia tập train/test 80/20."""
    print(f"\n==================================================================")
    print(f"📂 ĐỌC DỮ LIỆU 22 FEATURES - KỊCH BẢN {scenario}: {os.path.basename(data_path)}")
    print(f"==================================================================")
    df = pd.read_csv(data_path)
    feat_cols = [c for c in df.columns if c not in ['Class', 'Category', 'Label', 'Filename']]

    target_col = 'Category' if 'Category' in df.columns else 'Class'
    y_raw = df[target_col]

    classes = sorted(y_raw.unique())
    label_mapping = {c: i for i, c in enumerate(classes)}
    y_mapped = y_raw.map(label_mapping)

    print(f"✓ Tổng số mẫu: {len(df):,} | Số đặc trưng: {len(feat_cols)}")
    print(f"✓ Phân phối các lớp ({len(classes)} lớp):")
    for c in classes:
        print(f"    • {c:25s}: {(y_raw == c).sum():,} mẫu ({(y_raw == c).mean()*100:.1f}%)")

    # Chia Train (80%) - Test (20%) phân tầng
    X_train, X_test, y_train, y_test = train_test_split(
        df[feat_cols], y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )

    return X_train, X_test, y_train, y_test, label_mapping, feat_cols


# -----------------------------------------------------------------------------
# 3. HÀM ĐÁNH GIÁ MÔ HÌNH VỚI NHIỀU THƯỚC ĐO
# -----------------------------------------------------------------------------
def evaluate_single_model(name, model, X_tr, y_tr, X_te, y_te, is_scaled=False, scaler=None):
    t0 = time.time()
    if not is_scaled and scaler is not None:
        X_tr_fit = scaler.fit_transform(X_tr)
        X_te_fit = scaler.transform(X_te)
    else:
        X_tr_fit = X_tr
        X_te_fit = X_te

    model.fit(X_tr_fit, y_tr)
    y_pred = model.predict(X_te_fit)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.ravel()

    elapsed = time.time() - t0
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    macro_prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_te, y_pred, average='macro', zero_division=0)

    return {
        'Model': name,
        'Accuracy': acc,
        'Macro_Precision': macro_prec,
        'Macro_Recall': macro_rec,
        'Macro_F1': macro_f1,
        'Weighted_F1': weighted_f1,
        'Train_Time_Sec': round(elapsed, 2)
    }, y_pred


# -----------------------------------------------------------------------------
# 4. CHẠY THỬ NGHIỆM ĐẦY ĐỦ VỚI CÁC MỨC GAN AUGMENTATION (200 - 2000 MẪU)
# -----------------------------------------------------------------------------
def run_gan_enrichment_experiment(scenario=2, gan_levels=[200, 500, 1000, 2000],
                                  epochs_gan=250, output_dir="new feature/results"):
    os.makedirs(output_dir, exist_ok=True)
    base_dir = r"D:\Tai Lieu Hoc Tap Cao Hoc\Chuyên đề nghiên cứu về Phân loại mã độc\pe-cic-malmem-2022-ml\new feature"
    csv_file = os.path.join(base_dir, f"Dataset_22features_Scenario_{scenario}.csv")

    X_train, X_test, y_train, y_test, label_mapping, feat_cols = load_scenario_data(csv_file, scenario)
    num_classes = len(label_mapping)
    inv_mapping = {v: k for k, v in label_mapping.items()}

    # 1. Huấn luyện Conditional WGAN-GP CHỈ trên tập Train
    gan = TabularCWGAN_GP(num_features=len(feat_cols), num_classes=num_classes,
                          z_dim=64, embed_dim=16, hidden_dim=256, lr=2e-4)
    gan.fit(X_train, y_train, epochs=epochs_gan, batch_size=512, n_critic=2, verbose_every=5)

    # Lưu checkpoint GAN
    gan_ckpt_path = os.path.join(output_dir, f"cwgan_gp_scenario_{scenario}.pt")
    gan.save_checkpoint(gan_ckpt_path)

    # 2. Đánh giá chất lượng mẫu sinh ban đầu
    eval_dict = {c: 300 for c in range(num_classes)}
    df_eval_x, s_eval_y = gan.generate(eval_dict)
    gan_quality = evaluate_gan_quality(X_train, y_train, df_eval_x, s_eval_y, label_mapping, output_dir)

    # 3. Thiết lập các mô hình phân loại mới để đánh giá
    def get_models():
        return {
            'XGBoost (Tuned)': XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=8,
                                             subsample=0.85, colsample_bytree=0.85, random_state=42,
                                             eval_metric='mlogloss', n_jobs=-1),
            'LightGBM (Tuned)': LGBMClassifier(n_estimators=250, learning_rate=0.07, num_leaves=63,
                                               subsample=0.85, colsample_bytree=0.85, random_state=42,
                                               n_jobs=-1, verbose=-1),
            'CatBoost (Tuned)': CatBoostClassifier(iterations=250, learning_rate=0.08, depth=7,
                                                   random_seed=42, thread_count=-1, verbose=0),
            'Random Forest': RandomForestClassifier(n_estimators=250, max_depth=25, min_samples_split=4,
                                                    random_state=42, n_jobs=-1),
            'Deep Neural Net (MLP)': PyTorchMLPWrapper(in_features=len(feat_cols), num_classes=num_classes,
                                                       epochs=20, batch_size=512, lr=1.5e-3)
        }

    # Danh sách kết quả chi tiết
    all_results = []
    best_preds_per_level = {}

    # -------------------------------------------------------------
    # GIAI ĐOẠN A: BASELINE (0 mẫu GAN - Dữ liệu thực tế 100%)
    # -------------------------------------------------------------
    print(f"\n==================================================================")
    print(f"📊 [A] ĐÁNH GIÁ CÁC MÔ HÌNH BASELINE TRÊN DỮ LIỆU THỰC TẾ (KHÔNG GAN)")
    print(f"==================================================================")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_baseline = get_models()
    for name, model in models_baseline.items():
        print(f"  ⏳ Đang chạy {name} ...", end="", flush=True)
        res, y_pred = evaluate_single_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test, is_scaled=True)
        res['Augmentation_Level'] = 0
        res['Augmentation_Desc'] = 'Baseline (Real Only)'
        res['Scenario'] = scenario
        all_results.append(res)
        print(f" Xong! -> Acc: {res['Accuracy']*100:.2f}% | Macro-F1: {res['Macro_F1']*100:.2f}% ({res['Train_Time_Sec']}s)")
        if name == 'XGBoost (Tuned)':
            best_preds_per_level[0] = y_pred

    # -------------------------------------------------------------
    # GIAI ĐOẠN B: LÀM PHONG PHÚ DỮ LIỆU BẰNG GAN (200 - 2000 MẪU)
    # -------------------------------------------------------------
    augmented_dfs = {}

    for n_samples in gan_levels:
        print(f"\n==================================================================")
        print(f"✨ [B] TĂNG CƯỜNG DỮ LIỆU: +{n_samples} MẪU GAN TRÊN MỖI LỚP THIỂU SỐ")
        print(f"==================================================================")

        # Xác định số lượng sinh: bổ sung n_samples cho các lớp mã độc (hoặc tất cả các lớp trừ lớp đa số)
        class_counts = y_train.value_counts()
        max_class_count = class_counts.max()
        gen_dict = {}

        for c_idx in range(num_classes):
            # Nếu là lớp thiểu số so với Benign, sinh n_samples
            if class_counts.get(c_idx, 0) < max_class_count:
                gen_dict[c_idx] = n_samples
            else:
                # Nếu là lớp lớn nhất, có thể sinh nhẹ hoặc giữ nguyên
                gen_dict[c_idx] = min(n_samples // 4, 100)

        total_gen = sum(gen_dict.values())
        print(f"  🎯 Số lượng sinh nhân tạo: {total_gen:,} mẫu ({gen_dict})")

        t_gen0 = time.time()
        df_synth_x, s_synth_y = gan.generate(gen_dict)
        print(f"  ⚡ Sinh {len(df_synth_x):,} mẫu bằng GAN trong {time.time() - t_gen0:.2f}s!")

        # Hợp nhất tập Train thực và tập GAN sinh
        X_train_aug = pd.concat([X_train, df_synth_x], axis=0, ignore_index=True)
        y_train_aug = pd.concat([y_train, s_synth_y], axis=0, ignore_index=True)

        print(f"  ✓ Kích thước tập Train sau tăng cường: {len(X_train):,} -> {len(X_train_aug):,} mẫu (+{len(df_synth_x):,})")

        # Lưu lại tập dữ liệu tăng cường cho mức 1000 mẫu
        if n_samples == 1000:
            df_aug_export = X_train_aug.copy()
            df_aug_export['Category'] = y_train_aug.map(inv_mapping)
            aug_csv_path = os.path.join(output_dir, f"Dataset_22features_Scenario_{scenario}_GAN_augmented_{n_samples}.csv")
            df_aug_export.to_csv(aug_csv_path, index=False)
            print(f"  💾 Đã xuất tập dữ liệu tăng cường ra: {aug_csv_path}")

        # Chuẩn hóa: fit trên tập X_train_aug, transform X_test
        scaler_aug = StandardScaler()
        X_train_aug_scaled = scaler_aug.fit_transform(X_train_aug)
        X_test_aug_scaled = scaler_aug.transform(X_test)

        # Đánh giá các mô hình
        models_aug = get_models()
        for name, model in models_aug.items():
            print(f"  ⏳ Đang chạy {name} (+{n_samples} GAN) ...", end="", flush=True)
            res, y_pred = evaluate_single_model(name, model, X_train_aug_scaled, y_train_aug, X_test_aug_scaled, y_test, is_scaled=True)
            res['Augmentation_Level'] = n_samples
            res['Augmentation_Desc'] = f'+{n_samples} GAN Samples/Class'
            res['Scenario'] = scenario
            all_results.append(res)
            print(f" Xong! -> Acc: {res['Accuracy']*100:.2f}% | Macro-F1: {res['Macro_F1']*100:.2f}% ({res['Train_Time_Sec']}s)")
            if name == 'XGBoost (Tuned)':
                best_preds_per_level[n_samples] = y_pred

    # 4. Lưu toàn bộ kết quả vào CSV
    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(output_dir, f"Metrics_GAN_Scenario_{scenario}.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ ĐÃ LƯU BẢNG KẾT QUẢ THỰC NGHIỆM TẠI: {out_csv}")

    # 5. Vẽ biểu đồ so sánh xu hướng tăng trưởng theo số lượng mẫu GAN
    plt.figure(figsize=(10, 6))
    for m_name in ['XGBoost (Tuned)', 'LightGBM (Tuned)', 'Random Forest', 'Deep Neural Net (MLP)']:
        sub = res_df[res_df['Model'] == m_name]
        plt.plot(sub['Augmentation_Level'], sub['Macro_F1'] * 100, marker='s', linewidth=2, label=m_name)

    plt.title(f'Tác động của việc làm giàu dữ liệu bằng GAN lên Macro-F1 (Kịch bản {scenario})', fontsize=12)
    plt.xlabel('Số lượng mẫu sinh bởi GAN trên mỗi lớp thiểu số')
    plt.ylabel('Macro-F1 Score (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    chart_f1 = os.path.join(output_dir, f"gan_macro_f1_curve_scenario_{scenario}.png")
    plt.savefig(chart_f1, dpi=200)
    plt.close()
    print(f"📈 Đã lưu biểu đồ tăng trưởng F1 tại: {chart_f1}")

    # Vẽ Confusion Matrix so sánh Baseline vs +1000 GAN
    if 0 in best_preds_per_level and 1000 in best_preds_per_level:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cm_base = confusion_matrix(y_test, best_preds_per_level[0])
        cm_gan = confusion_matrix(y_test, best_preds_per_level[1000])

        c_labels = [inv_mapping[i] for i in range(num_classes)]
        if num_classes > 6:
            # Rút gọn tên nếu quá nhiều lớp
            c_labels = [c.replace('Ransomware-', 'R-').replace('Spyware-', 'S-').replace('Trojan-', 'T-') for c in c_labels]

        sns.heatmap(cm_base, annot=num_classes <= 6, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=c_labels, yticklabels=c_labels)
        axes[0].set_title(f'Baseline XGBoost (Không GAN) - Kịch bản {scenario}')
        axes[0].set_xlabel('Dự đoán')
        axes[0].set_ylabel('Thực tế')

        sns.heatmap(cm_gan, annot=num_classes <= 6, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=c_labels, yticklabels=c_labels)
        axes[1].set_title(f'XGBoost + 1000 Mẫu GAN - Kịch bản {scenario}')
        axes[1].set_xlabel('Dự đoán')

        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"confusion_matrix_compare_scenario_{scenario}.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"📈 Đã lưu Confusion Matrix so sánh tại: {cm_path}")

    # 6. In bảng tổng kết ngắn gọn
    print(f"\n🏆 BẢNG TỔNG KẾT KẾT QUẢ SO SÁNH (KỊCH BẢN {scenario}):")
    pivot = res_df.pivot(index='Model', columns='Augmentation_Desc', values='Macro_F1') * 100
    print(pivot.round(2).to_string())

    return res_df, gan
