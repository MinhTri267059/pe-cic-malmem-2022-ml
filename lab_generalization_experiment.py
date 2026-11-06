# -*- coding: utf-8 -*-
"""Đo khả năng tổng quát hóa sang dữ liệu lab thật của nhóm (Win10 19041).

Dữ liệu lab = Benign tự thu (lab/vms/*/features.csv) + Malware demo (demo test/*.csv).
Chia theo nhóm Category (mọi dump của cùng một mẫu ở cùng một phía) để không rò rỉ.

A  MalMem2022 only            -> baseline, test toàn bộ lab
B  MalMem + lab (có trọng số) -> group-CV trên lab
C  lab only                   -> group-CV trên lab
D  như B nhưng chuẩn hóa z-score theo từng máy (tâm = trung bình Benign của máy đó)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT_DIR, RESULTS_DIR, RAW_CSV  # noqa: E402

LAB_BENIGN = sorted(glob.glob(os.path.join(ROOT_DIR, "lab", "vms", "*", "features.csv")))
LAB_MALWARE = sorted(glob.glob(os.path.join(ROOT_DIR, "demo test", "*.csv")))
EXCLUDE_CATEGORY = {"cleanx86"}  # mẫu "clean" nhưng bị gán nhãn Malware trong file demo: nhãn mơ hồ
LAB_WEIGHTS = (1, 50)
SEED = 42


def load():
    mal = pd.read_csv(RAW_CSV)
    mal["domain"] = "malmem"
    parts = []
    for p in LAB_BENIGN + LAB_MALWARE:
        d = pd.read_csv(p)
        # Máy x86 (nservices=510) khác máy x64 (633): nhãn máy lấy từ tên file nguồn.
        d["machine"] = "x86" if "x86_candidates" in os.path.basename(p) else "x64"
        parts.append(d)
    lab = pd.concat(parts, ignore_index=True)
    lab = lab[~lab["Category"].isin(EXCLUDE_CATEGORY)]
    lab["domain"] = "lab"
    feats = [c for c in mal.columns if c not in ("Category", "Class", "Filename", "domain")]
    # XGBoost nuốt NaN âm thầm; dump có plugin lỗi (handles) thì bỏ hẳn.
    lab = lab.dropna(subset=feats).reset_index(drop=True)
    for d in (mal, lab):
        d["y"] = (d["Class"] == "Malware").astype(int)
    return mal, lab, feats


def fit_predict(Xtr, ytr, wtr, Xte):
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED,
                      verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xte)


def zscore_by_benign(df_dom, benign_ref):
    mu = benign_ref.mean()
    sd = benign_ref.std().replace(0, 1.0).fillna(1.0)
    return (df_dom - mu) / sd


def summarize(name, lab, pred):
    y = lab["y"].to_numpy()
    tnr = float((pred[y == 0] == 0).mean())
    tpr = float((pred[y == 1] == 1).mean())
    row = {"scenario": name, "benign_recall": tnr, "malware_recall": tpr,
           "balanced_acc": (tnr + tpr) / 2}
    per = pd.DataFrame({"Category": lab["Category"], "y": y, "ok": pred == y}).groupby("Category").agg(
        y=("y", "first"), n=("ok", "size"), acc=("ok", "mean"))
    return row, per


def run(feats, mal, lab, tag, n_splits=5):
    rows, details = [], {}
    Xm, ym = mal[feats], mal["y"]

    pred = fit_predict(Xm, ym, None, lab[feats])
    r, p = summarize("A_malmem_only", lab, pred); rows.append(r); details[r["scenario"]] = p

    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                 .split(lab, lab["y"], lab["Category"]))
    for w in LAB_WEIGHTS:
        for scen in ("B", "C", "D"):
            if scen == "C" and w != LAB_WEIGHTS[0]:
                continue
            pred = np.zeros(len(lab), dtype=int)
            for tr, te in folds:
                ltr, lte = lab.iloc[tr], lab.iloc[te]
                if scen == "D":
                    ref = ltr.loc[ltr["y"] == 0, feats]
                    Xl_tr = zscore_by_benign(ltr[feats], ref)
                    Xl_te = zscore_by_benign(lte[feats], ref)
                    Xm_use = zscore_by_benign(Xm, Xm[ym == 0])
                else:
                    Xl_tr, Xl_te, Xm_use = ltr[feats], lte[feats], Xm
                if scen == "C":
                    Xtr, ytr, wtr = Xl_tr, ltr["y"], None
                else:
                    Xtr = pd.concat([Xm_use, Xl_tr])
                    ytr = pd.concat([ym, ltr["y"]])
                    wtr = np.r_[np.ones(len(Xm_use)), np.full(len(Xl_tr), float(w))]
                pred[te] = fit_predict(Xtr, ytr, wtr, Xl_te)
            name = f"{scen}_w{w}" if scen != "C" else "C_lab_only"
            r, p = summarize(name, lab, pred); rows.append(r); details[name] = p
    out = pd.DataFrame(rows); out.insert(0, "features", tag)
    return out, details


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mal, lab, feats = load()
    print(f"MalMem: {len(mal)} dòng | lab: {len(lab)} dòng "
          f"(Benign {int((lab.y == 0).sum())}, Malware {int((lab.y == 1).sum())}, "
          f"{lab['Category'].nunique()} nhóm)")
    variants = {"all": feats, "no_svcscan": [c for c in feats if not c.startswith("svcscan.")]}
    # "all_lab" bị nhiễm confound máy (20/31 Malware ở máy x86); "same_machine" mới đáng tin.
    subsets = {"all_lab": (lab, 5), "same_machine_x64": (lab[lab.machine == "x64"].reset_index(drop=True), 2)}
    results = []
    for sname, (sub, k) in subsets.items():
        print(f"\n######## {sname}: {len(sub)} dòng (Benign {int((sub.y == 0).sum())}, "
              f"Malware {int((sub.y == 1).sum())}, {sub['Category'].nunique()} nhóm)")
        for tag, fs in variants.items():
            out, det = run(fs, mal, sub, f"{sname}/{tag}", n_splits=k)
            results.append(out)
            print(f"\n== {sname}/{tag} ({len(fs)} đặc trưng) ==")
            print(out.drop(columns="features").round(3).to_string(index=False))
            if sname == "same_machine_x64":
                print("Độ chính xác theo nhóm ở B_w50:")
                print(det["B_w50"].round(2).to_string())
    pd.concat(results).to_csv(os.path.join(RESULTS_DIR, "lab_generalization.csv"), index=False)


if __name__ == "__main__":
    main()
