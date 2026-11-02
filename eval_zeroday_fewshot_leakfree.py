# -*- coding: utf-8 -*-
"""
Thực nghiệm Học ít mẫu Không Rò Rỉ (Leak-Free Class-Disjoint Zero-Day Few-Shot):
Chia tập lớp thành:
- 13 Base Classes (Benign + 12 họ mã độc): dùng huấn luyện ProtoNet và SiameseNet
- 3 Novel Classes (Hoàn toàn mới: Ransomware-Conti, Spyware-TIBS, Trojan-Zeus):
  CHỈ được cấp K mẫu support (K=5, 10), tuyệt đối không xuất hiện trong tập train!
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from fewshot_learning import PrototypicalNetwork, SiameseNetwork

def run_zeroday_fewshot():
    csv_path = r"D:\Tai Lieu Hoc Tap Cao Hoc\Chuyên đề nghiên cứu về Phân loại mã độc\pe-cic-malmem-2022-ml\new feature\Dataset_22features_Scenario_3.csv"
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in ['Class', 'Category', 'Label', 'Filename']]
    
    # 3 họ mã độc Novel (Zero-day)
    novel_classes = ['Ransomware-Conti', 'Spyware-TIBS', 'Trojan-Zeus']
    base_classes = [c for c in df['Category'].unique() if c not in novel_classes]
    
    print("=" * 80)
    print("🛡️  THỰC NGHIỆM ZERO-DAY FEW-SHOT KHÔNG RÒ RỈ (CLASS-DISJOINT)")
    print("=" * 80)
    print(f"• 13 Lớp Cơ sở (Base Classes - Huấn luyện Representation): {len(base_classes)} lớp")
    print(f"• 3 Lớp Mới (Novel / Zero-Day Classes - Chỉ cấp K mẫu): {novel_classes}")

    df_base = df[df['Category'].isin(base_classes)].copy()
    df_novel = df[df['Category'].isin(novel_classes)].copy()

    # Chuẩn hóa fit CHỈ trên Base Classes
    scaler = StandardScaler()
    X_base_sc = scaler.fit_transform(df_base[feat_cols])
    X_novel_sc = scaler.transform(df_novel[feat_cols])

    base_map = {c: i for i, c in enumerate(base_classes)}
    novel_map = {c: i for i, c in enumerate(novel_classes)}
    y_base = df_base['Category'].map(base_map).values
    y_novel = df_novel['Category'].map(novel_map).values

    # Huấn luyện ProtoNet trên Base Classes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proto = PrototypicalNetwork(in_features=len(feat_cols), embed_dim=32, device=device)
    proto.fit_episodic(X_base_sc, y_base, n_way=min(4, len(base_classes)), k_shot=5, n_query=15, n_episodes=200, lr=1e-3, verbose_every=100)

    # Đánh giá Few-Shot trên Novel Classes với K = 5 và K = 10
    novel_indices = {c: np.where(y_novel == c)[0] for c in range(len(novel_classes))}

    for k in [5, 10, 20]:
        accs, f1s = [], []
        for run in range(10):
            np.random.seed(42 + run * 7)
            supp_idx, query_idx = [], []
            for c in range(len(novel_classes)):
                avail = novel_indices[c]
                perm = np.random.permutation(avail)
                supp_idx.extend(perm[:k])
                query_idx.extend(perm[k:])

            s_x = torch.tensor(X_novel_sc[supp_idx], dtype=torch.float32, device=device)
            s_y = torch.tensor(y_novel[supp_idx], dtype=torch.long, device=device)
            q_x = torch.tensor(X_novel_sc[query_idx], dtype=torch.float32, device=device)
            q_y = y_novel[query_idx]

            proto.eval()
            with torch.no_grad():
                log_probs = proto(s_x, s_y, q_x, n_way=len(novel_classes))
                preds = log_probs.argmax(dim=1).cpu().numpy()

            accs.append(accuracy_score(q_y, preds))
            f1s.append(f1_score(q_y, preds, average='macro', zero_division=0))

        print(f"\n---> Đánh giá trên 3 Novel Classes với K = {k} mẫu/lớp:")
        print(f"     ✓ ProtoNet Zero-Day Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        print(f"     ✓ ProtoNet Zero-Day Macro-F1: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")

if __name__ == '__main__':
    run_zeroday_fewshot()
