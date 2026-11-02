# -*- coding: utf-8 -*-
"""
Học ít mẫu Zero-Day KHÔNG RÒ RỈ (Class-Disjoint + Group-Disjoint)
=================================================================

  • Base classes  : Benign + 12 họ mã độc -> dùng huấn luyện bộ mã hóa (encoder).
  • Novel classes : 3 họ mới (1 Ransomware + 1 Spyware + 1 Trojan) -> chỉ cấp K mẫu support.

SỬA LỖI so với phiên bản trước:
  1. Support/query được chọn theo SampleID: K mẫu support đến từ K tệp mã độc khác nhau và
     TOÀN BỘ dump của các tệp đó bị loại khỏi query. (Cũ: chọn ngẫu nhiên theo dòng -> query
     chứa các dump khác của chính tệp trong support => rò rỉ snapshot.)
  2. Chỉ một bộ novel cố định (Conti/TIBS/Zeus) -> kết quả phụ thuộc may rủi. Nay xoay vòng
     5 bộ novel (mỗi họ mã độc xuất hiện ở vai trò novel đúng 1 lần) và báo cáo mean ± std.
  3. Đường dẫn tương đối, seed cố định.

CẢI TIẾN độ chính xác (so sánh):
  • Nearest-Centroid trên đặc trưng thô (chuẩn hóa)           – baseline không học.
  • Logistic Regression trên K mẫu support                       – baseline tuyến tính.
  • ProtoNet (episodic meta-training trên base)                  – phương pháp cũ.
  • SimpleShot: encoder huấn luyện CE có giám sát trên base,
    embedding được centering + L2-normalize, phân loại prototype – thường mạnh hơn ProtoNet.
  • Trên cả 22 đặc trưng tàng hình và 55 đặc trưng đầy đủ.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import (load_raw, raw_feature_columns, STEALTH_22_FEATURES, set_seed, RESULTS_DIR)
from fewshot_learning import PrototypicalNetwork

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

GROUP_FAMILIES = {
    'Ransomware': ['Ransomware-Conti', 'Ransomware-Ako', 'Ransomware-Maze', 'Ransomware-Pysa', 'Ransomware-Shade'],
    'Spyware': ['Spyware-TIBS', 'Spyware-180solutions', 'Spyware-CWS', 'Spyware-Gator', 'Spyware-Transponder'],
    'Trojan': ['Trojan-Zeus', 'Trojan-Emotet', 'Trojan-Reconyc', 'Trojan-Refroso', 'Trojan-Scar'],
}
# Fold 0 = bộ novel gốc (Conti, TIBS, Zeus) để so sánh trực tiếp với báo cáo cũ
NOVEL_FOLDS = [[GROUP_FAMILIES[g][i] for g in ('Ransomware', 'Spyware', 'Trojan')] for i in range(5)]


class SupervisedEncoder(nn.Module):
    def __init__(self, in_features, n_classes, embed_dim=64, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, embed_dim))
        self.head = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(F.relu(self.body(x)))


def train_supervised_encoder(X, y, n_classes, epochs=30, device='cpu', seed=42):
    torch.manual_seed(seed)
    model = SupervisedEncoder(X.shape[1], n_classes).to(device)
    # Cân bằng lớp (Benign chiếm ~50% base) để encoder học phân biệt các họ mã độc
    w = torch.tensor(len(y) / (n_classes * np.bincount(y, minlength=n_classes)), dtype=torch.float32,
                     device=device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
                        batch_size=256, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(loader))
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
            sched.step()
    model.eval()
    return model


def embed(model, X, device):
    with torch.no_grad():
        return model.embed(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def simpleshot_predict(Es, ys, Eq, base_mean):
    """SimpleShot (Wang et al. 2019): CL2N = centering theo mean base + L2-normalize, rồi nearest-prototype."""
    def cl2n(E):
        E = E - base_mean
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    Es, Eq = cl2n(Es), cl2n(Eq)
    classes = np.unique(ys)
    protos = np.stack([Es[ys == c].mean(0) for c in classes])
    return classes[((Eq[:, None, :] - protos[None]) ** 2).sum(-1).argmin(1)]


def centroid_predict(Xs, ys, Xq):
    classes = np.unique(ys)
    protos = np.stack([Xs[ys == c].mean(0) for c in classes])
    return classes[((Xq[:, None, :] - protos[None]) ** 2).sum(-1).argmin(1)]


def sample_episode(y_novel, groups_novel, n_way, k, rng):
    """Support: K tệp khác nhau / lớp (1 dump mỗi tệp). Query: mọi dump của các tệp CÒN LẠI."""
    supp, query = [], []
    for c in range(n_way):
        idx = np.where(y_novel == c)[0]
        ug = rng.permutation(np.unique(groups_novel[idx]))
        sg = set(ug[:k])
        for g in ug[:k]:
            supp.append(rng.choice(idx[groups_novel[idx] == g]))
        query.extend(idx[~np.isin(groups_novel[idx], list(sg))])
    return np.array(supp), np.array(query)


def run_zeroday_fewshot(k_list=(1, 5, 10, 20), n_episodes=20, feature_sets=('22_stealth', '55_full'),
                        folds=None, output_dir=RESULTS_DIR):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = load_raw()
    folds = NOVEL_FOLDS if folds is None else folds
    fsets = {'22_stealth': STEALTH_22_FEATURES, '55_full': raw_feature_columns(df)}
    rows = []

    print("=" * 80)
    print("🛡️  ZERO-DAY FEW-SHOT KHÔNG RÒ RỈ (CLASS-DISJOINT + GROUP-DISJOINT)")
    print("=" * 80)
    for fs in feature_sets:
        cols = fsets[fs]
        for fi, novel in enumerate(folds):
            base_classes = sorted(c for c in df['Family'].unique() if c not in novel)
            df_b = df[df['Family'].isin(base_classes)]
            df_n = df[df['Family'].isin(novel)]
            sc = StandardScaler().fit(df_b[cols])            # chỉ fit trên base
            Xb = sc.transform(df_b[cols])
            Xn = sc.transform(df_n[cols])
            yb = df_b['Family'].map({c: i for i, c in enumerate(base_classes)}).values
            yn = df_n['Family'].map({c: i for i, c in enumerate(novel)}).values
            gn = df_n['SampleID'].values

            enc = train_supervised_encoder(Xb, yb, len(base_classes), device=device, seed=42 + fi)
            Eb, En = embed(enc, Xb, device), embed(enc, Xn, device)
            base_mean = Eb.mean(0)

            proto = PrototypicalNetwork(in_features=len(cols), embed_dim=32, device=device)
            torch.manual_seed(42 + fi)
            np.random.seed(42 + fi)
            proto.fit_episodic(Xb, yb, n_way=5, k_shot=5, n_query=15, n_episodes=600, lr=1e-3,
                               verbose=False)

            print(f"\n[{fs}] Fold {fi}: novel = {novel}")
            for k in k_list:
                res = {}
                for ep in range(n_episodes):
                    rng = np.random.default_rng(1000 * fi + 10 * k + ep)
                    s, q = sample_episode(yn, gn, len(novel), k, rng)
                    preds = {
                        'NearestCentroid (raw)': centroid_predict(Xn[s], yn[s], Xn[q]),
                        'ProtoNet (episodic)': proto.predict(Xn[s], yn[s], Xn[q], len(novel)),
                        'SimpleShot (supervised enc. + CL2N)': simpleshot_predict(En[s], yn[s], En[q], base_mean),
                    }
                    if k > 1:
                        preds['LogReg (raw)'] = LogisticRegression(max_iter=2000).fit(Xn[s], yn[s]).predict(Xn[q])
                    for m, p in preds.items():
                        res.setdefault(m, []).append((accuracy_score(yn[q], p),
                                                      f1_score(yn[q], p, average='macro', zero_division=0)))
                for m, v in res.items():
                    a, f = np.array(v).T
                    rows.append({'Feature_Set': fs, 'Fold': fi, 'Novel_Classes': '|'.join(novel), 'K': k,
                                 'Method': m, 'Accuracy': a.mean(), 'Accuracy_Std': a.std(),
                                 'Macro_F1': f.mean(), 'Macro_F1_Std': f.std()})
                best = max(res, key=lambda m: np.mean([x[0] for x in res[m]]))
                print(f"   K={k:2d}: " + " | ".join(f"{m.split(' ')[0]} {np.mean([x[0] for x in v])*100:.1f}%"
                                                  for m, v in res.items()) + f"  (tốt nhất: {best.split(' ')[0]})")

    out = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, 'zeroday_fewshot_leakfree_detail.csv'), index=False)
    summ = (out.groupby(['Feature_Set', 'K', 'Method'])
               .agg(Accuracy_Mean=('Accuracy', 'mean'), Accuracy_Std_Folds=('Accuracy', 'std'),
                    Macro_F1_Mean=('Macro_F1', 'mean'), Macro_F1_Std_Folds=('Macro_F1', 'std'))
               .reset_index())
    summ.to_csv(os.path.join(output_dir, 'zeroday_fewshot_leakfree_summary.csv'), index=False)
    print("\n📊 TỔNG HỢP (trung bình 5 bộ novel, 3-way, ngẫu nhiên = 33.3%):")
    print((summ.pivot_table(index=['Feature_Set', 'Method'], columns='K', values='Accuracy_Mean') * 100)
          .round(2).to_string())
    return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20)
    args = ap.parse_args()
    run_zeroday_fewshot(n_episodes=args.episodes)
