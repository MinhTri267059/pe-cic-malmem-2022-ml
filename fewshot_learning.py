# -*- coding: utf-8 -*-
"""
Mô-đun Huấn luyện Học vài mẫu (Few-Shot Learning) trên 22 đặc trưng bộ nhớ:
1. Prototypical Network (ProtoNet) - Học không gian metric biểu diễn prototype.
2. Siamese Network với Contrastive Loss - So sánh tương đồng cặp vector bộ nhớ.
3. GAN-Assisted Few-Shot Learning - Tận dụng GAN làm giàu tập support khi chỉ có K mẫu (K=5, 10, 20, 50).
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


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
                     n_episodes=400, lr=1e-3, verbose_every=100, verbose=True):
        """
        Huấn luyện ProtoNet bằng chiến lược mô phỏng Episode (Meta-training).
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

        classes = np.unique(y_train)
        class_to_indices = {c: np.where(y_train == c)[0] for c in classes}

        _print = print if verbose else (lambda *a, **k: None)
        n_way = min(n_way, len(classes))
        _print(f"\n🧠 Huấn luyện Prototypical Network (ProtoNet) trên {self.device.upper()}...")
        _print(f"   • Episodic Setup: {n_way}-way {k_shot}-shot ({n_query} queries/class) | {n_episodes} episodes")

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
                perm = np.random.permutation(idx)
                if len(perm) >= k_shot + n_query:
                    supp_idx, q_idx = perm[:k_shot], perm[k_shot:k_shot + n_query]
                else:  # lớp quá ít mẫu: chia đôi, query không trùng support
                    cut = max(1, min(k_shot, len(perm) - 1))
                    supp_idx = np.random.choice(perm[:cut], size=k_shot, replace=True)
                    q_pool = perm[cut:] if len(perm) > cut else perm[:cut]
                    q_idx = np.random.choice(q_pool, size=n_query, replace=True)

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
                _print(f"   [Episode {ep:3d}/{n_episodes}] Loss: {recent_loss:.4f} | Accuracy: {recent_acc*100:.2f}%")

        _print(f"✅ Hoàn tất huấn luyện ProtoNet trong {time.time() - t0:.1f}s!")

    def predict(self, support_x, support_y, query_x, n_way):
        self.eval()
        with torch.no_grad():
            sx = torch.tensor(np.asarray(support_x), dtype=torch.float32, device=self.device)
            sy = torch.tensor(np.asarray(support_y), dtype=torch.long, device=self.device)
            qx = torch.tensor(np.asarray(query_x), dtype=torch.float32, device=self.device)
            return self.forward(sx, sy, qx, n_way).argmax(dim=1).cpu().numpy()

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
                  batch_size=256, lr=1e-3, verbose_every=5, verbose=True):
        """
        Huấn luyện Siamese Network bằng các cặp mẫu (Pairwise Contrastive Training).
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        y_train = np.asarray(y_train)

        # Tạo tập các cặp (x1, x2, label: 1=cùng lớp, 0=khác lớp)
        _print = print if verbose else (lambda *a, **k: None)
        _print(f"\n🧬 Tạo {n_pairs:,} cặp mẫu huấn luyện cho Siamese Network...")
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

        _print(f"🚀 Huấn luyện Siamese Network với Contrastive Loss (Margin={margin})...")
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
                _print(f"   [Epoch {ep:2d}/{epochs:2d}] Contrastive Loss: {np.mean(ep_loss):.4f}")

        _print(f"✅ Hoàn tất huấn luyện Siamese Network trong {time.time() - t0:.1f}s!")

    def predict_nn(self, support_x, support_y, query_x):
        self.eval()
        with torch.no_grad():
            s_emb = self.forward_one(torch.tensor(np.asarray(support_x), dtype=torch.float32, device=self.device))
            q_emb = self.forward_one(torch.tensor(np.asarray(query_x), dtype=torch.float32, device=self.device))
            return np.asarray(support_y)[torch.cdist(q_emb, s_emb).argmin(dim=1).cpu().numpy()]

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


# 4. FEW-SHOT BENCHMARK KHÔNG RÒ RỈ (MỌI PHƯƠNG PHÁP CHỈ THẤY K MẪU/LỚP)
# -----------------------------------------------------------------------------
def _sample_support(y, groups, classes, k, rng):
    """Chọn K mẫu/lớp, mỗi mẫu đến từ một SampleID KHÁC NHAU (đa dạng, không lặp dump)."""
    idx = []
    for c in classes:
        cand = np.where(y == c)[0]
        g = groups[cand]
        ug = rng.permutation(np.unique(g))[:k]
        for gg in ug:
            idx.append(rng.choice(cand[g == gg]))
    return np.array(idx)


def _metrics(y_true, pred):
    return accuracy_score(y_true, pred), f1_score(y_true, pred, average='macro', zero_division=0)


def benchmark_fewshot_with_gan(X_train_df, y_train_s, X_test_df, y_test_s,
                               gan_model=None, k_shots=(5, 10, 20, 50),
                               n_gan_augment_per_class=100, n_runs=5,
                               output_dir="new feature/results", groups_train=None,
                               gan_epochs=150, tag=None):
    """
    Đối sánh Few-Shot KHÔNG RÒ RỈ (sửa lỗi phiên bản cũ):
      - Phiên bản cũ tiền huấn luyện ProtoNet/Siamese/GAN trên TOÀN BỘ tập train
        (hàng chục nghìn mẫu có nhãn) rồi gọi là "few-shot" -> rò rỉ tiền tri thức.
      - Ở đây MỌI phương pháp (RF, XGB, GAN, ProtoNet, Siamese) chỉ thấy đúng K mẫu/lớp.
        GAN (nếu bật) được huấn luyện lại từ đầu CHỈ trên support set của mỗi lượt.
      - Test set được chia theo nhóm SampleID nên không chung dump với support.
    Tham số gan_model giữ để tương thích: nếu khác None thì bật nhánh GAN-assisted.
    """
    from gan_augmentation import TabularCWGAN_GP
    os.makedirs(output_dir, exist_ok=True)
    y_train = np.asarray(y_train_s).astype(int)
    y_test = np.asarray(y_test_s).astype(int)
    classes = np.unique(y_train)
    num_classes = len(classes)
    groups_train = (np.asarray(groups_train) if groups_train is not None
                    else np.arange(len(y_train)).astype(str))
    X_train_np = np.asarray(X_train_df, dtype=np.float64)
    X_test_np = np.asarray(X_test_df, dtype=np.float64)
    use_gan = gan_model is not None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = []
    print(f"\n==================================================================")
    print(f"🧪 FEW-SHOT KHÔNG RÒ RỈ: K = {list(k_shots)} | {n_runs} lượt | GAN-assisted: {use_gan}")
    print(f"==================================================================")

    for k in k_shots:
        scores = {}
        for run in range(n_runs):
            seed = 42 + run * 13 + k
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            supp = _sample_support(y_train, groups_train, classes, k, rng)
            Xs_raw, ys = X_train_np[supp], y_train[supp]
            # Scaler fit CHỈ trên support (không dùng thống kê của tập train đầy đủ)
            sc = StandardScaler().fit(Xs_raw)
            Xs, Xq = sc.transform(Xs_raw), sc.transform(X_test_np)

            def add(name, pred):
                scores.setdefault(name, []).append(_metrics(y_test, pred))

            add('Random Forest (Baseline)',
                RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xs, ys).predict(Xq))
            add('XGBoost (Baseline)',
                XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=seed,
                              n_jobs=-1, verbosity=0).fit(Xs, ys).predict(Xq))
            add('Logistic Regression',
                LogisticRegression(max_iter=2000, C=1.0).fit(Xs, ys).predict(Xq))

            if use_gan:
                gan = TabularCWGAN_GP(num_features=Xs_raw.shape[1], num_classes=num_classes,
                                      z_dim=32, embed_dim=8, hidden_dim=128, lr=2e-4, device=device)
                gan.fit(pd.DataFrame(Xs_raw), ys, epochs=gan_epochs, batch_size=128, n_critic=5,
                        verbose=False)
                Xg, yg = gan.generate({c: n_gan_augment_per_class for c in classes}, seed=seed)
                Xa = np.vstack([Xs, sc.transform(Xg.values)])
                ya = np.r_[ys, yg.values]
                add('Random Forest (+GAN Augment)',
                    RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xa, ya).predict(Xq))
                add('XGBoost (+GAN Augment)',
                    XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=seed,
                                  n_jobs=-1, verbosity=0).fit(Xa, ya).predict(Xq))

            # ProtoNet: meta-train episodic CHỈ trên support set
            proto = PrototypicalNetwork(in_features=Xs.shape[1], embed_dim=32, device=device)
            k_s = max(1, k // 2)
            proto.fit_episodic(Xs, ys, n_way=min(num_classes, 4), k_shot=k_s, n_query=max(1, k - k_s),
                               n_episodes=150, lr=1e-3, verbose_every=10 ** 9, verbose=False)
            add('Prototypical Network (ProtoNet)', proto.predict(Xs, ys, Xq, num_classes))

            siam = SiameseNetwork(in_features=Xs.shape[1], embed_dim=32, device=device)
            siam.fit_pairs(Xs, ys, n_pairs=min(4000, 40 * len(ys)), margin=1.0, epochs=10, verbose=False)
            add('Siamese Network (1-NN)', siam.predict_nn(Xs, ys, Xq))

        for name, vals in scores.items():
            accs, f1s = np.array(vals).T
            results.append({'K_Shot': k, 'Model': name,
                            'Accuracy_Mean': accs.mean(), 'Accuracy_Std': accs.std(),
                            'Macro_F1_Mean': f1s.mean(), 'Macro_F1_Std': f1s.std(),
                            'Augmentation': (f'+{n_gan_augment_per_class} GAN (train trên support)'
                                             if 'GAN' in name else 'None'),
                            'Protocol': 'support-only, group-disjoint test'})
            print(f"   K={k:3d} | {name:34s} Acc {accs.mean()*100:6.2f}±{accs.std()*100:4.2f}% "
                  f"| F1 {f1s.mean()*100:6.2f}±{f1s.std()*100:4.2f}%", flush=True)

    res_df = pd.DataFrame(results)
    suffix = f"_{tag}" if tag else ""
    res_df.to_csv(os.path.join(output_dir, f"fewshot_benchmark_results{suffix}.csv"), index=False)

    plt.figure(figsize=(9, 5))
    for m in res_df['Model'].unique():
        sub = res_df[res_df['Model'] == m]
        plt.errorbar(sub['K_Shot'], sub['Macro_F1_Mean'] * 100, yerr=sub['Macro_F1_Std'] * 100,
                     marker='o', capsize=3, label=m)
    plt.title(f'Few-Shot (không rò rỉ) theo K – Macro-F1 {tag or ""}', fontsize=12)
    plt.xlabel('K mẫu / lớp')
    plt.ylabel('Macro-F1 (%)')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fewshot_learning_curve{suffix}.png"), dpi=150)
    plt.close()
    return res_df
