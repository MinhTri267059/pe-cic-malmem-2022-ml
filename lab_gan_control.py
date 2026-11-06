# -*- coding: utf-8 -*-
"""Đối chứng cho lab_gan_experiment.py: GAN có thật sự sinh tri thức mới, hay chỉ làm rộng vùng Malware?

S5 nhiễu Gaussian quanh mẫu Malware thật (sigma x độ lệch chuẩn đặc trưng của train)
S6 IsolationForest chỉ học từ Benign train (không dùng Malware nào) -> điểm bất thường
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, check_no_group_overlap  # noqa: E402
from lab_generalization_experiment import fit_predict, load  # noqa: E402

SEEDS = (42, 7)
N = 200


def noisy_copies(Xtr, ytr, sigma, seed):
    rng = np.random.RandomState(seed)
    sd = Xtr.std().to_numpy()
    upper, intm = Xtr.max().to_numpy(), np.all(np.isclose(Xtr.values, np.round(Xtr.values)), axis=0)
    parts, labels = [], []
    for c in (0, 1):
        base = Xtr[ytr == c].to_numpy()
        x = base[rng.randint(0, len(base), N)] + rng.randn(N, base.shape[1]) * sd * sigma
        x = np.clip(x, 0, upper)
        x[:, intm] = np.round(x[:, intm])
        parts.append(pd.DataFrame(x, columns=Xtr.columns)); labels += [c] * N
    return pd.concat(parts), np.array(labels)


def main():
    _, lab, feats = load()
    sub = lab[lab.machine == "x64"].reset_index(drop=True)
    res = {}
    for seed in SEEDS:
        folds = list(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
                     .split(sub, sub["y"], sub["Category"]))
        preds = {k: np.zeros(len(sub), dtype=int) for k in ("S5_noise_s0.5", "S5_noise_s1.0", "S6_isoforest_benign")}
        for tr, te in folds:
            check_no_group_overlap(sub["Category"], tr, te)
            Xtr, ytr = sub.iloc[tr][feats].reset_index(drop=True), sub.iloc[tr]["y"].to_numpy()
            Xte = sub.iloc[te][feats]
            for sigma in (0.5, 1.0):
                Xs, ys = noisy_copies(Xtr, ytr, sigma, seed)
                preds[f"S5_noise_s{sigma}"][te] = fit_predict(pd.concat([Xtr, Xs]), np.r_[ytr, ys], None, Xte)
            iso = IsolationForest(n_estimators=300, random_state=seed).fit(Xtr[ytr == 0])
            preds["S6_isoforest_benign"][te] = (iso.predict(Xte) == -1).astype(int)
        y = sub["y"].to_numpy()
        for k, p in preds.items():
            res.setdefault(k, []).append(((p[y == 0] == 0).mean(), (p[y == 1] == 1).mean()))
    out = pd.DataFrame([{"scenario": k, "benign_recall": np.mean(v, 0)[0], "malware_recall": np.mean(v, 0)[1],
                         "balanced_acc": np.mean(v, 0).mean()} for k, v in res.items()])
    print(out.round(3).to_string(index=False))
    out.to_csv(os.path.join(RESULTS_DIR, "lab_gan_control.csv"), index=False)


if __name__ == "__main__":
    main()
