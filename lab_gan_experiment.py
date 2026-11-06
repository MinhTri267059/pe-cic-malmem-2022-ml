# -*- coding: utf-8 -*-
"""Dùng CWGAN-GP sinh thêm dữ liệu miền lab rồi huấn luyện, có kiểm soát rò rỉ.

Chốt chống rò rỉ:
  L1 chia fold theo nhóm Category TRƯỚC mọi bước (assert không trùng nhóm train/test)
  L2 GAN và scaler của nó chỉ fit trên lab-train của fold đó
  L3 mẫu sinh chỉ vào train; test luôn là dòng thật chưa từng thấy
  L4 kiểm tra ghi nhớ: khoảng cách láng giềng gần nhất synth->train so với synth->test
  L5 đối chứng: nhân bản mẫu thật (không GAN) cùng số lượng, để biết GAN có hơn copy không
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, check_no_group_overlap, set_seed  # noqa: E402
from gan_augmentation import TabularCWGAN_GP  # noqa: E402
from lab_generalization_experiment import fit_predict, load  # noqa: E402

N_PER_CLASS = 200   # số mẫu sinh/nhân bản cho mỗi lớp
EPOCHS = 600
LAB_W = 10.0        # trọng số dòng lab (thật + sinh) khi trộn với MalMem
SEEDS = (42, 7)


def gan_synth(Xtr, ytr, seed):
    set_seed(seed)
    g = TabularCWGAN_GP(num_features=Xtr.shape[1], num_classes=2)
    g.fit(Xtr, ytr, epochs=EPOCHS, batch_size=64, verbose=False)
    xs, ys = g.generate({0: N_PER_CLASS, 1: N_PER_CLASS}, seed=seed)
    return xs, ys.to_numpy()


def dup_synth(Xtr, ytr, seed):
    rng = np.random.RandomState(seed)
    parts, labels = [], []
    for c in (0, 1):
        idx = np.where(ytr == c)[0]
        pick = rng.choice(idx, N_PER_CLASS, replace=True)
        parts.append(Xtr.iloc[pick])
        labels += [c] * N_PER_CLASS
    return pd.concat(parts).reset_index(drop=True), np.array(labels)


def memorization_audit(Xtr, Xte, Xs):
    sc = StandardScaler().fit(Xtr)
    a, b, s = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xs)
    d_tr = NearestNeighbors(n_neighbors=1).fit(a).kneighbors(s)[0].mean()
    d_te = NearestNeighbors(n_neighbors=1).fit(b).kneighbors(s)[0].mean()
    d_ref = NearestNeighbors(n_neighbors=1).fit(a).kneighbors(b)[0].mean()
    keys = set(map(tuple, np.round(Xtr.values, 3)))
    dup = float(np.mean([tuple(r) in keys for r in np.round(Xs.values, 3)]))
    return d_tr, d_te, d_ref, dup


def evaluate(feats, mal, lab, k, seed):
    Xm, ym = mal[feats], mal["y"]
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                 .split(lab, lab["y"], lab["Category"]))
    preds = {s: np.zeros(len(lab), dtype=int) for s in ("S0_lab", "S1_lab+dup", "S2_lab+GAN",
                                                       "S3_malmem+lab", "S4_malmem+lab+GAN")}
    audits = []
    for tr, te in folds:
        check_no_group_overlap(lab["Category"], tr, te)                       # L1
        Xtr, ytr = lab.iloc[tr][feats].reset_index(drop=True), lab.iloc[tr]["y"].to_numpy()
        Xte = lab.iloc[te][feats]
        if len(set(ytr)) < 2:
            continue
        Xg, yg = gan_synth(Xtr, ytr, seed)                                    # L2
        Xd, yd = dup_synth(Xtr, ytr, seed)                                    # L5
        audits.append(memorization_audit(Xtr, Xte, Xg))                       # L4

        def cat(*parts):
            return pd.concat([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

        sets = {
            "S0_lab": (Xtr, ytr, None),
            "S1_lab+dup": (*cat((Xtr, ytr), (Xd, yd)), None),
            "S2_lab+GAN": (*cat((Xtr, ytr), (Xg, yg)), None),
        }
        Xa, ya = cat((Xm, ym.to_numpy()), (Xtr, ytr))
        sets["S3_malmem+lab"] = (Xa, ya, np.r_[np.ones(len(Xm)), np.full(len(Xtr), LAB_W)])
        Xb, yb = cat((Xm, ym.to_numpy()), (Xtr, ytr), (Xg, yg))
        sets["S4_malmem+lab+GAN"] = (Xb, yb, np.r_[np.ones(len(Xm)), np.full(len(Xtr) + len(Xg), LAB_W)])
        for name, (X, y, w) in sets.items():
            preds[name][te] = fit_predict(X, y, w, Xte)                       # L3
    y = lab["y"].to_numpy()
    rows = []
    for name, p in preds.items():
        rows.append({"scenario": name, "benign_recall": float((p[y == 0] == 0).mean()),
                     "malware_recall": float((p[y == 1] == 1).mean())})
    return rows, np.array(audits)


def main():
    mal, lab, feats = load()
    subsets = {"same_machine_x64": (lab[lab.machine == "x64"].reset_index(drop=True), 2),
               "all_lab(confound)": (lab, 5)}
    out = []
    for sname, (sub, k) in subsets.items():
        print(f"\n######## {sname}: {len(sub)} dòng (Benign {int((sub.y == 0).sum())}, "
              f"Malware {int((sub.y == 1).sum())})", flush=True)
        agg, aud = {}, []
        for seed in SEEDS:
            t0 = time.time()
            rows, a = evaluate(feats, mal, sub, k, seed)
            aud.append(a)
            for r in rows:
                agg.setdefault(r["scenario"], []).append((r["benign_recall"], r["malware_recall"]))
            print(f"  seed {seed}: {time.time() - t0:.0f}s", flush=True)
        for name, v in agg.items():
            b, m = np.mean(v, axis=0)
            out.append({"subset": sname, "scenario": name, "benign_recall": b, "malware_recall": m,
                        "balanced_acc": (b + m) / 2})
        a = np.vstack(aud)
        print(f"  Audit ghi nhớ (chuẩn hóa): synth->train NN={a[:, 0].mean():.2f} | "
              f"synth->test NN={a[:, 1].mean():.2f} | test->train NN (tham chiếu)={a[:, 2].mean():.2f} | "
              f"tỉ lệ synth trùng dòng train={a[:, 3].mean():.1%}")
        print(pd.DataFrame([r for r in out if r["subset"] == sname]).drop(columns="subset").round(3)
              .to_string(index=False), flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(RESULTS_DIR, "lab_gan_experiment.csv"), index=False)


if __name__ == "__main__":
    main()
