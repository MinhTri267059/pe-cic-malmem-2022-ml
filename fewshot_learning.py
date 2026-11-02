# -*- coding: utf-8 -*-
"""
Mô-đun Huấn luyện Học vài mẫu (Few-Shot Learning) trên 22 đặc trưng bộ nhớ:
1. Prototypical Network (ProtoNet) - Học không gian metric biểu diễn prototype.
2. Siamese Network với Contrastive Loss - So sánh tương đồng cặp vector bộ nhớ.
3. GAN-Assisted Few-Shot Learning - Tận dụng GAN làm giàu tập support khi chỉ có K mẫu (K=5, 10, 20, 50).
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -----------------------------------------------------------------------------
# 1. MẠNG NƠ-RON EMBEDDING CHO METRIC-BASED FEW-SHOT
# -----------------------------------------------------------------------------
class FeatureEncoder(nn.Module):
    """
    Trích xuất biểu diễn embedding tiềm ẩn (latent embedding) từ 22 đặc trưng bộ nhớ.
    Đầu ra là vector chuẩn hóa L2 có số chiều embed_dim (mặc định 32).
    """
    def __init__(self, in_features=22, embed_dim=32, hidden_dim=128):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        h = self.net(x)
        # Chuẩn hóa L2 để khoảng cách Euclidean tương đương khoảng cách Cosine
        return F.normalize(h, p=2, dim=1)


# -----------------------------------------------------------------------------
# 2. PROTOTYPICAL NETWORK (PROTONET)
# -----------------------------------------------------------------------------
class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network (Snell et al., NeurIPS 2017):
    Mỗi lớp được đại diện bởi vector Prototype là tâm cụm (trung bình) của các mẫu trong Support Set.
    Query samples được phân loại dựa trên khoảng cách Euclidean đến các Prototype.
    """
    def __init__(self, in_features=22, embed_dim=32, hidden_dim=128, device=None):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = FeatureEncoder(in_features, embed_dim, hidden_dim)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, support_x, support_y, query_x, n_way):
        """
        Tính toán log-probabilities cho query_x dựa trên support set.
        support_x: (N_way * K_shot, in_features)
        support_y: (N_way * K_shot,) với giá trị từ 0 đến N_way - 1
        query_x: (N_query, in_features)
        """
        s_emb = self.encoder(support_x)  # (N_way * K_shot, embed_dim)
        q_emb = self.encoder(query_x)    # (N_query, embed_dim)

        # Tính toán Prototype cho từng lớp
        prototypes = []
        for c in range(n_way):
            mask = (support_y == c)
            c_samples = s_emb[mask]
            if len(c_samples) > 0:
                proto = c_samples.mean(dim=0)
            else:
                proto = torch.zeros(s_emb.size(1), device=self.device)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)  # (n_way, embed_dim)

        # Khoảng cách Euclidean bình phương giữa queries và prototypes
        # (N_query, n_way)
        dists = torch.cdist(q_emb, prototypes, p=2) ** 2
        log_probs = F.log_softmax(-dists, dim=1)
        return log_probs

    def fit_episodic(self, X_train, y_train, n_way=4, k_shot=5, n_query=15,
                     n_episodes=400, lr=1e-3, verbose_every=100):
        """
        Huấn luyện ProtoNet bằng chiến lược mô phỏng Episode (Meta-training).
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

        classes = np.unique(y_train)
        class_to_indices = {c: np.where(y_train == c)[0] for c in classes}

        print(f"\n🧠 Huấn luyện Prototypical Network (ProtoNet) trên {self.device.upper()}...")
        print(f"   • Episodic Setup: {n_way}-way {k_shot}-shot ({n_query} queries/class) | {n_episodes} episodes")

        t0 = time.time()
        history = []
        for ep in range(1, n_episodes + 1):
            optimizer.zero_grad()
            # Chọn ngẫu nhiên n_way lớp
            selected_classes = np.random.choice(classes, size=n_way, replace=False)

            support_x_list, support_y_list = [], []
            query_x_list, query_y_list = [], []

            for new_c_idx, c in enumerate(selected_classes):
                idx = class_to_indices[c]
                chosen = np.random.choice(idx, size=k_shot + n_query, replace=(len(idx) < k_shot + n_query))
                supp_idx = chosen[:k_shot]
                q_idx = chosen[k_shot:]

                support_x_list.append(X_train[supp_idx])
                support_y_list.extend([new_c_idx] * k_shot)

                query_x_list.append(X_train[q_idx])
                query_y_list.extend([new_c_idx] * n_query)

            support_x = torch.tensor(np.vstack(support_x_list), dtype=torch.float32, device=self.device)
            support_y = torch.tensor(support_y_list, dtype=torch.long, device=self.device)
            query_x = torch.tensor(np.vstack(query_x_list), dtype=torch.float32, device=self.device)
            query_y = torch.tensor(query_y_list, dtype=torch.long, device=self.device)

            log_probs = self.forward(support_x, support_y, query_x, n_way)
            loss = F.nll_loss(log_probs, query_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_y = log_probs.argmax(dim=1)
            acc = (pred_y == query_y).float().mean().item()
            history.append((loss.item(), acc))

            if ep % verbose_every == 0 or ep == 1:
                recent_loss = np.mean([h[0] for h in history[-verbose_every:]])
                recent_acc = np.mean([h[1] for h in history[-verbose_every:]])
                print(f"   [Episode {ep:3d}/{n_episodes}] Loss: {recent_loss:.4f} | Accuracy: {recent_acc*100:.2f}%")

        print(f"✅ Hoàn tất huấn luyện ProtoNet trong {time.time() - t0:.1f}s!")

    def evaluate_fewshot(self, support_x, support_y, query_x, query_y, n_way):
        """Đánh giá 1 tập few-shot cố định."""
        self.eval()
        with torch.no_grad():
            sx = torch.tensor(support_x, dtype=torch.float32, device=self.device)
            sy = torch.tensor(support_y, dtype=torch.long, device=self.device)
            qx = torch.tensor(query_x, dtype=torch.float32, device=self.device)
            qy = np.array(query_y)

            log_probs = self.forward(sx, sy, qx, n_way)
            pred = log_probs.argmax(dim=1).cpu().numpy()

            acc = accuracy_score(qy, pred)
            macro_f1 = f1_score(qy, pred, average='macro', zero_division=0)
            macro_rec = recall_score(qy, pred, average='macro', zero_division=0)
        return acc, macro_f1, macro_rec


# -----------------------------------------------------------------------------
# 3. SIAMESE NETWORK VỚI CONTRASTIVE LOSS
# -----------------------------------------------------------------------------
class SiameseNetwork(nn.Module):
    """
    Mạng Siamese (Koch et al., ICML 2015; Bromme et al.):
    Hai nhánh dùng chung trọng số (weight-sharing) ánh xạ các mẫu vào không gian metric.
    Huấn luyện với Contrastive Loss để kéo các mẫu cùng lớp lại gần và đẩy khác lớp ra xa.
    """
    def __init__(self, in_features=22, embed_dim=32, hidden_dim=128, device=None):
        super(SiameseNetwork, self).__init__()
        self.encoder = FeatureEncoder(in_features, embed_dim, hidden_dim)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dist = F.pairwise_distance(out1, out2)
        return dist

    def fit_pairs(self, X_train, y_train, n_pairs=10000, margin=1.0, epochs=20,
                  batch_size=256, lr=1e-3, verbose_every=5):
        """
        Huấn luyện Siamese Network bằng các cặp mẫu (Pairwise Contrastive Training).
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        # Tạo tập các cặp (x1, x2, label: 1=cùng lớp, 0=khác lớp)
        print(f"\n🧬 Tạo {n_pairs:,} cặp mẫu huấn luyện cho Siamese Network...")
        classes = np.unique(y_train)
        class_to_indices = {c: np.where(y_train == c)[0] for c in classes}

        pairs_x1, pairs_x2, pair_labels = [], [], []
        for _ in range(n_pairs):
            is_same = np.random.rand() > 0.5
            if is_same:
                c = np.random.choice(classes)
                idx = np.random.choice(class_to_indices[c], size=2, replace=True)
                pairs_x1.append(X_train[idx[0]])
                pairs_x2.append(X_train[idx[1]])
                pair_labels.append(1.0)
            else:
                c1, c2 = np.random.choice(classes, size=2, replace=False)
                idx1 = np.random.choice(class_to_indices[c1])
                idx2 = np.random.choice(class_to_indices[c2])
                pairs_x1.append(X_train[idx1])
                pairs_x2.append(X_train[idx2])
                pair_labels.append(0.0)

        t_x1 = torch.tensor(np.array(pairs_x1), dtype=torch.float32)
        t_x2 = torch.tensor(np.array(pairs_x2), dtype=torch.float32)
        t_y = torch.tensor(pair_labels, dtype=torch.float32)

        dataset = TensorDataset(t_x1, t_x2, t_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"🚀 Huấn luyện Siamese Network với Contrastive Loss (Margin={margin})...")
        t0 = time.time()
        for ep in range(1, epochs + 1):
            ep_loss = []
            for b_x1, b_x2, b_y in loader:
                b_x1, b_x2, b_y = b_x1.to(self.device), b_x2.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                dists = self.forward(b_x1, b_x2)
                # Contrastive Loss: y * d^2 + (1-y) * max(0, m - d)^2
                loss = torch.mean(b_y * torch.pow(dists, 2) +
                                  (1.0 - b_y) * torch.pow(torch.clamp(margin - dists, min=0.0), 2))
                loss.backward()
                optimizer.step()
                ep_loss.append(loss.item())

            if ep % verbose_every == 0 or ep == 1:
                print(f"   [Epoch {ep:2d}/{epochs:2d}] Contrastive Loss: {np.mean(ep_loss):.4f}")

        print(f"✅ Hoàn tất huấn luyện Siamese Network trong {time.time() - t0:.1f}s!")

    def classify_by_nearest_neighbor(self, support_x, support_y, query_x, query_y):
        """Phân loại 1-NN trong không gian embedding của Siamese Network."""
        self.eval()
        with torch.no_grad():
            sx = torch.tensor(support_x, dtype=torch.float32, device=self.device)
            qx = torch.tensor(query_x, dtype=torch.float32, device=self.device)

            s_emb = self.forward_one(sx)  # (N_supp, embed_dim)
            q_emb = self.forward_one(qx)  # (N_query, embed_dim)

            dists = torch.cdist(q_emb, s_emb, p=2)  # (N_query, N_supp)
            nearest_idx = dists.argmin(dim=1).cpu().numpy()
            pred = support_y[nearest_idx]

            acc = accuracy_score(query_y, pred)
            macro_f1 = f1_score(query_y, pred, average='macro', zero_division=0)
            macro_rec = recall_score(query_y, pred, average='macro', zero_division=0)
        return acc, macro_f1, macro_rec


# -----------------------------------------------------------------------------
# 4. GAN-ASSISTED FEW-SHOT BENCHMARK
# -----------------------------------------------------------------------------
def benchmark_fewshot_with_gan(X_train_df, y_train_s, X_test_df, y_test_s,
                              gan_model, k_shots=[5, 10, 20, 50],
                              n_gan_augment_per_class=100, n_runs=5,
                              output_dir="new feature/results"):
    """
    So sánh toàn diện giữa:
    1. Baseline Few-Shot (chỉ dùng K mẫu thật).
    2. GAN-Enriched Few-Shot (K mẫu thật + mẫu sinh bởi GAN).
    3. Prototypical Network.
    4. Siamese Network (1-NN Metric).
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = np.unique(y_train_s)
    num_classes = len(classes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # 1. Tiền huấn luyện ProtoNet và Siamese trên tập train tổng thể
    proto_net = PrototypicalNetwork(in_features=X_train_df.shape[1], embed_dim=32, device=gan_model.device)
    proto_net.fit_episodic(X_train_scaled, y_train_s.values, n_way=min(4, num_classes),
                           k_shot=5, n_query=15, n_episodes=150, lr=1e-3, verbose_every=50)

    siamese_net = SiameseNetwork(in_features=X_train_df.shape[1], embed_dim=32, device=gan_model.device)
    siamese_net.fit_pairs(X_train_scaled, y_train_s.values, n_pairs=6000, margin=1.0, epochs=10, verbose_every=5)

    results = []

    print(f"\n==================================================================")
    print(f"🧪 THỰC HIỆN ĐỐI SÁNH FEW-SHOT LEARNING VỚI K = {k_shots}")
    print(f"==================================================================")

    class_indices = {c: np.where(y_train_s.values == c)[0] for c in classes}

    for k in k_shots:
        print(f"\n---> Khảo sát kịch bản K = {k} mẫu/lớp ({k*num_classes} mẫu tổng cộng)...")

        acc_base_rf, f1_base_rf = [], []
        acc_base_xgb, f1_base_xgb = [], []
        acc_gan_rf, f1_gan_rf = [], []
        acc_gan_xgb, f1_gan_xgb = [], []
        acc_proto, f1_proto = [], []
        acc_siamese, f1_siamese = [], []

        for run in range(n_runs):
            np.random.seed(42 + run * 13)
            # Chọn K mẫu thật cho mỗi lớp từ train set
            supp_indices = []
            for c in classes:
                avail = class_indices[c]
                chosen = np.random.choice(avail, size=min(k, len(avail)), replace=False)
                supp_indices.extend(chosen)

            supp_indices = np.array(supp_indices)
            X_supp_real = X_train_scaled[supp_indices]
            y_supp_real = y_train_s.values[supp_indices]

            # ---------------------------------------------------------
            # A. Baseline Models (Chỉ K mẫu thật)
            # ---------------------------------------------------------
            rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_base.fit(X_supp_real, y_supp_real)
            p_rf = rf_base.predict(X_test_scaled)
            acc_base_rf.append(accuracy_score(y_test_s, p_rf))
            f1_base_rf.append(f1_score(y_test_s, p_rf, average='macro', zero_division=0))

            xgb_base = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1)
            xgb_base.fit(X_supp_real, y_supp_real)
            p_xgb = xgb_base.predict(X_test_scaled)
            acc_base_xgb.append(accuracy_score(y_test_s, p_xgb))
            f1_base_xgb.append(f1_score(y_test_s, p_xgb, average='macro', zero_division=0))

            # ---------------------------------------------------------
            # B. GAN-Augmented Few-Shot (K mẫu thật + n_gan mẫu sinh)
            # ---------------------------------------------------------
            gan_dict = {c: n_gan_augment_per_class for c in classes}
            df_synth_x, s_synth_y = gan_model.generate(gan_dict)
            X_synth_scaled = scaler.transform(df_synth_x)

            X_supp_aug = np.vstack([X_supp_real, X_synth_scaled])
            y_supp_aug = np.hstack([y_supp_real, s_synth_y.values])

            rf_gan = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_gan.fit(X_supp_aug, y_supp_aug)
            p_rf_gan = rf_gan.predict(X_test_scaled)
            acc_gan_rf.append(accuracy_score(y_test_s, p_rf_gan))
            f1_gan_rf.append(f1_score(y_test_s, p_rf_gan, average='macro', zero_division=0))

            xgb_gan = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1)
            xgb_gan.fit(X_supp_aug, y_supp_aug)
            p_xgb_gan = xgb_gan.predict(X_test_scaled)
            acc_gan_xgb.append(accuracy_score(y_test_s, p_xgb_gan))
            f1_gan_xgb.append(f1_score(y_test_s, p_xgb_gan, average='macro', zero_division=0))

            # ---------------------------------------------------------
            # C. Prototypical Network (Evaluation trên Support Set)
            # ---------------------------------------------------------
            a_proto, f_proto, _ = proto_net.evaluate_fewshot(
                X_supp_real, y_supp_real, X_test_scaled, y_test_s.values, num_classes
            )
            acc_proto.append(a_proto)
            f1_proto.append(f_proto)

            # ---------------------------------------------------------
            # D. Siamese Network 1-NN Metric
            # ---------------------------------------------------------
            a_siam, f_siam, _ = siamese_net.classify_by_nearest_neighbor(
                X_supp_real, y_supp_real, X_test_scaled, y_test_s.values
            )
            acc_siamese.append(a_siam)
            f1_siamese.append(f_siam)

        # Ghi nhận kết quả trung bình qua n_runs
        results.append({
            'K_Shot': k, 'Model': 'Random Forest (Baseline)',
            'Accuracy_Mean': np.mean(acc_base_rf), 'Accuracy_Std': np.std(acc_base_rf),
            'Macro_F1_Mean': np.mean(f1_base_rf), 'Macro_F1_Std': np.std(f1_base_rf),
            'Augmentation': 'None'
        })
        results.append({
            'K_Shot': k, 'Model': 'Random Forest (+GAN Augment)',
            'Accuracy_Mean': np.mean(acc_gan_rf), 'Accuracy_Std': np.std(acc_gan_rf),
            'Macro_F1_Mean': np.mean(f1_gan_rf), 'Macro_F1_Std': np.std(f1_gan_rf),
            'Augmentation': f'+{n_gan_augment_per_class} GAN'
        })
        results.append({
            'K_Shot': k, 'Model': 'XGBoost (Baseline)',
            'Accuracy_Mean': np.mean(acc_base_xgb), 'Accuracy_Std': np.std(acc_base_xgb),
            'Macro_F1_Mean': np.mean(f1_base_xgb), 'Macro_F1_Std': np.std(f1_base_xgb),
            'Augmentation': 'None'
        })
        results.append({
            'K_Shot': k, 'Model': 'XGBoost (+GAN Augment)',
            'Accuracy_Mean': np.mean(acc_gan_xgb), 'Accuracy_Std': np.std(acc_gan_xgb),
            'Macro_F1_Mean': np.mean(f1_gan_xgb), 'Macro_F1_Std': np.std(f1_gan_xgb),
            'Augmentation': f'+{n_gan_augment_per_class} GAN'
        })
        results.append({
            'K_Shot': k, 'Model': 'Prototypical Network (ProtoNet)',
            'Accuracy_Mean': np.mean(acc_proto), 'Accuracy_Std': np.std(acc_proto),
            'Macro_F1_Mean': np.mean(f1_proto), 'Macro_F1_Std': np.std(f1_proto),
            'Augmentation': 'Metric Embed'
        })
        results.append({
            'K_Shot': k, 'Model': 'Siamese Network (1-NN)',
            'Accuracy_Mean': np.mean(acc_siamese), 'Accuracy_Std': np.std(acc_siamese),
            'Macro_F1_Mean': np.mean(f1_siamese), 'Macro_F1_Std': np.std(f1_siamese),
            'Augmentation': 'Contrastive Metric'
        })

        delta_f1_rf = (np.mean(f1_gan_rf) - np.mean(f1_base_rf)) * 100
        delta_f1_xgb = (np.mean(f1_gan_xgb) - np.mean(f1_base_xgb)) * 100
        print(f"   ✓ K={k:2d} | RF: {np.mean(f1_base_rf)*100:.2f}% -> +GAN: {np.mean(f1_gan_rf)*100:.2f}% (Delta: {delta_f1_rf:+.2f}%)")
        print(f"   ✓ K={k:2d} | XGB: {np.mean(f1_base_xgb)*100:.2f}% -> +GAN: {np.mean(f1_gan_xgb)*100:.2f}% (Delta: {delta_f1_xgb:+.2f}%)")
        print(f"   ✓ K={k:2d} | ProtoNet F1: {np.mean(f1_proto)*100:.2f}% | Siamese F1: {np.mean(f1_siamese)*100:.2f}%")

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, "fewshot_benchmark_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Đã lưu kết quả thực nghiệm Few-Shot vào: {out_csv}")

    # Vẽ biểu đồ đường xu hướng K-shot
    plt.figure(figsize=(9, 5))
    for m in ['Random Forest (Baseline)', 'Random Forest (+GAN Augment)',
              'XGBoost (Baseline)', 'XGBoost (+GAN Augment)', 'Prototypical Network (ProtoNet)']:
        sub = res_df[res_df['Model'] == m]
        plt.plot(sub['K_Shot'], sub['Macro_F1_Mean'] * 100, marker='o', label=m)

    plt.title('Hiệu quả Few-Shot Learning theo số mẫu K (Macro-F1)', fontsize=12)
    plt.xlabel('Số mẫu K trên mỗi lớp (K-Shot)')
    plt.ylabel('Macro-F1 Score (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "fewshot_learning_curve.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"📈 Đã lưu biểu đồ đường Few-Shot tại: {chart_path}")

    return res_df
