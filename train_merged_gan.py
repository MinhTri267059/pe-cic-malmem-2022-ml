"""
Huấn luyện lại từ đầu trên dữ liệu gộp: MalMem2022 (22 đặc trưng) + 1000 mẫu GAN x64.
Đánh giá trên:
  (1) 20% test của MalMem2022 (cùng split random_state=42 như trước) -> kiểm tra không "quên" miền cũ
  (2) 100 mẫu test x64 thật (x64_test_100.csv) -> miền mới
So sánh với baseline: Models_22features (chỉ train trên MalMem2022).

Biến thể dữ liệu train:
  - Old+GAN        : MalMem2022 train + 1000 mẫu GAN
  - Old+GAN+Real   : MalMem2022 train + 1000 mẫu GAN + 100 mẫu thật x64 train
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from train_and_evaluate_22features import FEATURES_22, calculate_metrics

MODELS = {
    "XGBoost": lambda: XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=-1),
    "Random_Forest": lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
    "J_48": lambda: DecisionTreeClassifier(criterion="entropy", random_state=42),
    "Random_Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Naive_Bayes": lambda: GaussianNB(),
}


def label(df, sc):
    if sc == 1:
        return df["Class"]
    if sc == 2:
        return df["Category"].str.split("-").str[0]
    # KB3: bỏ hậu tố biến thể (Ransomware-Conti-1 -> Ransomware-Conti)
    return df["Category"].str.split("-").str[:2].str.join("-")


def metrics_row(variant, sc, model, eval_set, y_true, y_pred, labels):
    acc, rec, prec, f1, fpr = calculate_metrics(y_true, y_pred, labels)
    return {"Train_Data": variant, "Scenario": sc, "Model": model, "Eval_Set": eval_set,
            "Accuracy": round(acc * 100, 2), "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2), "F1": round(f1 * 100, 2), "FPR": round(fpr * 100, 2)}


def main():
    old = pd.read_csv("MalMem2022.csv")[FEATURES_22 + ["Class", "Category"]]
    gan = pd.read_csv("x64_gan_1000.csv")
    x64_tr = pd.read_csv("x64_train_100.csv")
    x64_te = pd.read_csv("x64_test_100.csv")

    rows, preds = [], x64_te[["Filename", "Class", "Category"]].copy()

    for sc in [1, 2, 3]:
        print(f"\n{'=' * 70}\n🚀 KỊCH BẢN {sc}\n{'=' * 70}")
        o = old.copy()
        o["y"] = label(o, sc)
        o = o[FEATURES_22 + ["y"]].drop_duplicates()
        o_tr, o_te = train_test_split(o, test_size=0.2, random_state=42, stratify=o["y"])

        y_te_x64 = label(x64_te, sc)
        X_te_x64 = x64_te[FEATURES_22].fillna(0)
        X_te_old = o_te[FEATURES_22].fillna(0)

        # ---- Baseline: model cũ (chỉ MalMem2022) trên 100 test x64
        bdir = f"Models_22features/Scenario_{sc}"
        bmap = json.load(open(f"{bdir}/label_mapping.json"))
        binv = {v: k for k, v in bmap.items()}
        blabels = sorted(set(bmap) | set(y_te_x64))
        for m in MODELS:
            mdl = joblib.load(f"{bdir}/{m}.pkl")
            yp = [binv[i] for i in mdl.predict(X_te_x64)]
            rows.append(metrics_row("Old only (baseline)", sc, m, "x64 test 100", y_te_x64, yp, blabels))

        # ---- Train lại từ đầu trên dữ liệu gộp
        variants = {
            "Old+GAN": [gan],
            "Old+GAN+Real": [gan, x64_tr],
        }
        for vname, extras in variants.items():
            new = pd.concat(extras, ignore_index=True)
            new["y"] = label(new, sc)
            tr = pd.concat([o_tr, new[FEATURES_22 + ["y"]]], ignore_index=True)

            classes = sorted(tr["y"].unique())
            lmap = {c: i for i, c in enumerate(classes)}
            inv = {i: c for c, i in lmap.items()}
            X_tr, y_tr = tr[FEATURES_22].fillna(0), tr["y"].map(lmap)
            eval_labels = sorted(set(classes) | set(y_te_x64))

            mdir = f"Models_GAN/{vname.replace('+', '_')}/Scenario_{sc}"
            os.makedirs(mdir, exist_ok=True)
            json.dump(lmap, open(f"{mdir}/label_mapping.json", "w"), indent=4)
            print(f"\n--- {vname}: {len(o_tr)} cũ + {len(new)} mới = {len(tr)} mẫu train, {len(classes)} lớp")

            for m, make in MODELS.items():
                mdl = make().fit(X_tr, y_tr)
                joblib.dump(mdl, f"{mdir}/{m}.pkl")

                yp_old = [inv[i] for i in mdl.predict(X_te_old)]
                r_old = metrics_row(vname, sc, m, "MalMem2022 test 20%", o_te["y"], yp_old, classes)
                yp_new = [inv[i] for i in mdl.predict(X_te_x64)]
                r_new = metrics_row(vname, sc, m, "x64 test 100", y_te_x64, yp_new, eval_labels)
                rows += [r_old, r_new]
                preds[f"{vname}_KB{sc}_{m}"] = yp_new
                print(f"[{m:14s}] MalMem test Acc {r_old['Accuracy']:6.2f}% | "
                      f"x64 test Acc {r_new['Accuracy']:6.2f}%  F1 {r_new['F1']:6.2f}%  FPR {r_new['FPR']:5.2f}%")

    res = pd.DataFrame(rows)
    res.to_csv("Metrics_GAN_Retrain.csv", index=False)
    preds.to_csv("Predictions_x64_test_GAN_Retrain.csv", index=False)

    # Bảng so sánh gọn: Accuracy trên x64 test
    pv = res[res["Eval_Set"] == "x64 test 100"].pivot_table(
        index=["Scenario", "Model"], columns="Train_Data", values="Accuracy")
    pv = pv[["Old only (baseline)", "Old+GAN", "Old+GAN+Real"]]
    print("\n📊 ACCURACY TRÊN 100 MẪU TEST x64:\n", pv.to_string())
    pv.to_csv("Summary_Accuracy_x64_GAN_Retrain.csv")
    print("\n✅ Lưu: Metrics_GAN_Retrain.csv, Predictions_x64_test_GAN_Retrain.csv, Summary_Accuracy_x64_GAN_Retrain.csv")


if __name__ == "__main__":
    main()

