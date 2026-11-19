"""
Huấn luyện & Đánh giá toàn diện trên bộ Test Chuẩn Kép (100 mẫu x32 + 100 mẫu x64 = 200 mẫu)
qua 3 Kịch bản (KB1: Nhị phân, KB2: 4 Nhóm hành vi, KB3: 24 Họ mã độc chi tiết).

So sánh 4 phương pháp:
  - E0 (Baseline 22f): Mô hình 22 đặc trưng gốc chỉ train trên MalMem2022
  - E1 (GAN 22f cũ)  : MalMem2022 + 1000 GAN trên 22 đặc trưng (chưa chuẩn hóa/cân bằng)
  - E2 (Unified 70f) : Mô hình hợp nhất train trên bộ gộp đã làm sạch (x32_clean + x64_GAN_1000 + x64_real_100)
                       với 70 đặc trưng (49 gốc + 20 tỷ lệ hành vi + cờ is_x64), chuẩn hóa theo kiến trúc & cân bằng trọng số
  - E3 (Arch-Adaptive 70f): Mô hình thích ứng kiến trúc (70 đặc trưng + chuẩn hóa theo kiến trúc + gộp GAN + Real)
                            tối ưu hóa đồng thời độ chính xác trên cả x32 và x64 cho cả 5 thuật toán.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train_and_evaluate_22features import FEATURES_22

SEED = 42
MODEL_LIST = ["XGBoost", "Random_Forest", "J_48", "Random_Tree", "Naive_Bayes"]


def make_estimator(name):
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            eval_metric="mlogloss",
            n_jobs=4,
        )
    if name == "Random_Forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=4
        )
    if name == "J_48":
        return DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=SEED)
    if name == "Random_Tree":
        return DecisionTreeClassifier(splitter="random", class_weight="balanced", random_state=SEED)
    if name == "Naive_Bayes":
        return GaussianNB(var_smoothing=1e-1)
    raise ValueError(name)


def get_labels(df, sc):
    if sc == 1:
        return df["Class"]
    if sc == 2:
        return df["Category"].str.split("-").str[0]
    return df["Category"].str.split("-").str[:2].str.join("-")


def build_70_features(df, avail_cols, is_x64_val):
    d = df[avail_cols].copy().fillna(0.0).astype(np.float64)
    eps = 1e-6
    d["ratio_ppid_nproc"] = d["pslist.nppid"] / (d["pslist.nproc"] + eps)
    d["ratio_64bit_nproc"] = d["pslist.nprocs64bit"] / (d["pslist.nproc"] + eps)
    d["ratio_file_nhandles"] = d["handles.nfile"] / (d["handles.nhandles"] + eps)
    d["ratio_event_nhandles"] = d["handles.nevent"] / (d["handles.nhandles"] + eps)
    d["ratio_desktop_nhandles"] = d["handles.ndesktop"] / (d["handles.nhandles"] + eps)
    d["ratio_key_nhandles"] = d["handles.nkey"] / (d["handles.nhandles"] + eps)
    d["ratio_thread_nhandles"] = d["handles.nthread"] / (d["handles.nhandles"] + eps)
    d["ratio_dir_nhandles"] = d["handles.ndirectory"] / (d["handles.nhandles"] + eps)
    d["ratio_sem_nhandles"] = d["handles.nsemaphore"] / (d["handles.nhandles"] + eps)
    d["ratio_timer_nhandles"] = d["handles.ntimer"] / (d["handles.nhandles"] + eps)
    d["ratio_sec_nhandles"] = d["handles.nsection"] / (d["handles.nhandles"] + eps)
    d["ratio_mut_nhandles"] = d["handles.nmutant"] / (d["handles.nhandles"] + eps)
    d["ratio_ldr_load_dlls"] = d["ldrmodules.not_in_load"] / (d["dlllist.ndlls"] + eps)
    d["ratio_ldr_init_dlls"] = d["ldrmodules.not_in_init"] / (d["dlllist.ndlls"] + eps)
    d["ratio_inj_nproc"] = d["malfind.ninjections"] / (d["pslist.nproc"] + eps)
    d["ratio_uniq_inj"] = d["malfind.uniqueInjections"] / (d["malfind.ninjections"] + eps)
    d["ratio_commit_inj"] = d["malfind.commitCharge"] / (d["malfind.ninjections"] + eps)
    d["ratio_active_svc"] = d["svcscan.nactive"] / (d["svcscan.nservices"] + eps)
    d["ratio_kdrv_svc"] = d["svcscan.kernel_drivers"] / (d["svcscan.nservices"] + eps)
    d["ratio_shared_svc"] = d["svcscan.shared_process_services"] / (d["svcscan.nservices"] + eps)
    d["is_x64"] = float(is_x64_val)
    return d


class ArchAdaptiveModel(BaseEstimator, ClassifierMixin):
    """
    Mô hình hợp nhất đa kiến trúc:
    - Tự động nhận diện kiến trúc thông qua cột `is_x64` (tương đương pslist.nprocs64bit > 0)
    - Nhánh x32 học trên bộ x32_train_clean đã khử nhiễu (70 đặc trưng chuẩn hóa)
    - Nhánh x64 học trên bộ gộp (x64_gan_1000_full + x64_train_100 với trọng số ưu tiên mẫu thật ×3 + mẫu x32 đồng nhãn đã chuẩn hóa)
    """

    def __init__(self, model_name):
        self.model_name = model_name

    def fit(self, X32, y32, X64, y64, sw64=None):
        self.classes32_ = sorted(np.unique(y32))
        self.map32_ = {c: i for i, c in enumerate(self.classes32_)}
        self.inv32_ = {i: c for c, i in self.map32_.items()}

        self.classes64_ = sorted(np.unique(y64))
        self.map64_ = {c: i for i, c in enumerate(self.classes64_)}
        self.inv64_ = {i: c for c, i in self.map64_.items()}

        self.est32_ = make_estimator(self.model_name)
        sw32 = compute_sample_weight("balanced", y32)
        if self.model_name == "Naive_Bayes":
            self.est32_.fit(X32, np.array([self.map32_[c] for c in y32]))
        else:
            self.est32_.fit(X32, np.array([self.map32_[c] for c in y32]), sample_weight=sw32)

        self.est64_ = make_estimator(self.model_name)
        y64_idx = np.array([self.map64_[c] for c in y64])
        if sw64 is None:
            sw64 = compute_sample_weight("balanced", y64_idx)
        if self.model_name == "Naive_Bayes":
            self.est64_.fit(X64, y64_idx)
        else:
            self.est64_.fit(X64, y64_idx, sample_weight=sw64)
        return self

    def predict(self, X):
        is_64 = X["is_x64"].values > 0.5
        preds = np.empty(len(X), dtype=object)
        if (~is_64).any():
            p32 = self.est32_.predict(X.loc[~is_64])
            preds[~is_64] = [self.inv32_[int(i)] for i in p32]
        if is_64.any():
            p64 = self.est64_.predict(X.loc[is_64])
            preds[is_64] = [self.inv64_[int(i)] for i in p64]
        return preds


def eval_metrics(method, sc, model_name, eval_subset, y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    fpr_arr = np.divide(FP, (FP + TN), out=np.zeros_like(FP, dtype=float), where=(FP + TN) != 0)
    fpr = float(np.mean(fpr_arr))

    return {
        "Method": method,
        "Scenario": f"KB{sc}",
        "Model": model_name,
        "Test_Subset": eval_subset,
        "Accuracy": round(acc * 100, 2),
        "Balanced_Acc": round(bacc * 100, 2),
        "Macro_F1": round(f1_m * 100, 2),
        "Weighted_F1": round(f1_w * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "FPR": round(fpr * 100, 2),
    }


def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN & ĐÁNH GIÁ ĐA KIẾN TRÚC (100 x32 + 100 x64 = 200 MẪU TEST)")
    print("=" * 80)

    x32_tr = pd.read_csv("x32_train_clean.csv")
    x32_te = pd.read_csv("x32_test_100.csv")
    x64_tr = pd.read_csv("x64_train_100.csv")
    x64_te = pd.read_csv("x64_test_100.csv")
    x64_gan = pd.read_csv("x64_gan_1000_full.csv")

    # Xuất bộ Test chuẩn 200 mẫu (100 x32 + 100 x64) và bộ Train gộp
    x32_te_out = x32_te.copy()
    x32_te_out["Arch"] = "x32"
    x64_te_out = x64_te.copy()
    x64_te_out["Arch"] = "x64"
    test_200_df = pd.concat([x32_te_out, x64_te_out], ignore_index=True)
    test_200_df.to_csv("Test_200_x32_x64.csv", index=False)

    merged_train_df = pd.concat(
        [
            x32_tr.assign(Source="x32_Clean"),
            x64_gan.assign(Source="x64_GAN_1000"),
            x64_tr.assign(Source="x64_Real_100"),
        ],
        ignore_index=True,
    )
    merged_train_df.to_csv("Dataset_Merged_Clean_x32_x64_GAN.csv", index=False)
    print(f"✅ Đã xuất bộ Train gộp đồng bộ: Dataset_Merged_Clean_x32_x64_GAN.csv ({len(merged_train_df)} mẫu)")
    print(f"✅ Đã xuất bộ Test kép chuẩn: Test_200_x32_x64.csv ({len(test_200_df)} mẫu: 100 x32 + 100 x64)")

    raw_cols = [c for c in x64_tr.columns if c not in ["Class", "Category", "Filename"]]
    avail_cols = [c for c in raw_cols if x64_tr[c].notna().any()]

    # Xây dựng 70 đặc trưng (49 gốc + 20 tỷ lệ hành vi + cờ is_x64)
    X32_tr_raw = build_70_features(x32_tr, avail_cols, 0)
    X32_te_raw = build_70_features(x32_te, avail_cols, 0)
    X64_real_raw = build_70_features(x64_tr, avail_cols, 1)
    X64_gan_raw = build_70_features(x64_gan, avail_cols, 1)
    X64_te_raw = build_70_features(x64_te, avail_cols, 1)

    # Chuẩn hóa Z-score theo từng kiến trúc (Domain Alignment: giúp mô hình Unified chuyển giao tri thức từ 57.908 mẫu x32 sang x64)
    num_cols = [c for c in X32_tr_raw.columns if c != "is_x64"]
    scaler32 = StandardScaler().fit(X32_tr_raw[num_cols])
    scaler64 = StandardScaler().fit(X64_real_raw[num_cols])

    def apply_scale(df_feat, scaler):
        out = df_feat.copy()
        out[num_cols] = scaler.transform(df_feat[num_cols])
        return out

    X32_tr = apply_scale(X32_tr_raw, scaler32)
    X32_te = apply_scale(X32_te_raw, scaler32)
    X64_real = apply_scale(X64_real_raw, scaler64)
    X64_gan = apply_scale(X64_gan_raw, scaler64)
    X64_te = apply_scale(X64_te_raw, scaler64)

    X64_comb_tr = pd.concat([X64_gan, X64_real], ignore_index=True)
    X_all_tr = pd.concat([X32_tr, X64_comb_tr], ignore_index=True)
    X_200_te = pd.concat([X32_te, X64_te], ignore_index=True)

    # Đặc trưng 22 cột cho E0 & E1
    X32_te_22 = x32_te[FEATURES_22].fillna(0)
    X64_te_22 = x64_te[FEATURES_22].fillna(0)
    X200_te_22 = pd.concat([X32_te_22, X64_te_22], ignore_index=True)

    all_metrics = []
    pred_records = test_200_df[["Filename", "Arch", "Class", "Category"]].copy()

    for sc in [1, 2, 3]:
        print(f"\n{'=' * 80}\n🔥 ĐÁNH GIÁ KỊCH BẢN {sc}\n{'=' * 80}")
        y32_tr = get_labels(x32_tr, sc).values
        y64_gan = get_labels(x64_gan, sc).values
        y64_real = get_labels(x64_tr, sc).values
        y64_comb = np.concatenate([y64_gan, y64_real])
        y_all_tr = np.concatenate([y32_tr, y64_comb])

        y32_te = get_labels(x32_te, sc).values
        y64_te = get_labels(x64_te, sc).values
        y200_te = np.concatenate([y32_te, y64_te])

        all_classes = sorted(set(y_all_tr) | set(y200_te))
        lmap_all = {c: i for i, c in enumerate(all_classes)}
        inv_all = {i: c for c, i in lmap_all.items()}

        # Trọng số mẫu cho Unified & Arch-Adaptive
        sw32 = compute_sample_weight("balanced", y32_tr)
        sw32_norm = sw32 / sw32.sum() * 10000.0

        sw64 = compute_sample_weight("balanced", y64_comb)
        real_boost = np.array([1.0] * len(y64_gan) + [4.0] * len(y64_real))
        sw64 = sw64 * real_boost
        sw64_norm = sw64 / sw64.sum() * 10000.0
        sw_unified = np.concatenate([sw32_norm, sw64_norm])

        # Thư mục lưu mô hình tốt nhất
        save_dir_u = f"Models_DualArch_Unified/Scenario_{sc}"
        save_dir_a = f"Models_DualArch_Best/Scenario_{sc}"
        os.makedirs(save_dir_u, exist_ok=True)
        os.makedirs(save_dir_a, exist_ok=True)
        joblib.dump({"scaler32": scaler32, "scaler64": scaler64, "avail_cols": avail_cols}, f"{save_dir_a}/preprocessors.pkl")

        # 1. Đánh giá E0 (Baseline 22f) & E1 (Old+GAN 22f)
        for m_code, m_dir in [
            ("E0: Baseline (22f Old)", f"Models_22features/Scenario_{sc}"),
            ("E1: Old+GAN (22f)", f"Models_GAN/Old_GAN/Scenario_{sc}"),
        ]:
            with open(f"{m_dir}/label_mapping.json") as f:
                bmap = json.load(f)
            binv = {v: k for k, v in bmap.items()}
            for m_name in MODEL_LIST:
                mdl = joblib.load(f"{m_dir}/{m_name}.pkl")
                p32 = np.array([binv[int(i)] for i in mdl.predict(X32_te_22)])
                p64 = np.array([binv[int(i)] for i in mdl.predict(X64_te_22)])
                p200 = np.concatenate([p32, p64])
                all_metrics.append(eval_metrics(m_code, sc, m_name, "x32 (100)", y32_te, p32, sorted(set(y32_te) | set(p32))))
                all_metrics.append(eval_metrics(m_code, sc, m_name, "x64 (100)", y64_te, p64, sorted(set(y64_te) | set(p64))))
                all_metrics.append(eval_metrics(m_code, sc, m_name, "Combined (200)", y200_te, p200, all_classes))

        # 2. Huấn luyện & Đánh giá E2 (Unified 70f) và E3 (Arch-Adaptive 70f)
        for m_name in MODEL_LIST:
            # --- E2: Unified 70f ---
            est_u = make_estimator(m_name)
            if m_name == "Naive_Bayes":
                est_u.fit(X_all_tr, np.array([lmap_all[c] for c in y_all_tr]))
            else:
                est_u.fit(X_all_tr, np.array([lmap_all[c] for c in y_all_tr]), sample_weight=sw_unified)
            joblib.dump(est_u, f"{save_dir_u}/{m_name}.pkl")

            p32_u = np.array([inv_all[int(i)] for i in est_u.predict(X32_te)])
            p64_u = np.array([inv_all[int(i)] for i in est_u.predict(X64_te)])
            p200_u = np.concatenate([p32_u, p64_u])

            all_metrics.append(eval_metrics("E2: Unified (Clean+FullGAN 70f)", sc, m_name, "x32 (100)", y32_te, p32_u, sorted(set(y32_te) | set(p32_u))))
            all_metrics.append(eval_metrics("E2: Unified (Clean+FullGAN 70f)", sc, m_name, "x64 (100)", y64_te, p64_u, sorted(set(y64_te) | set(p64_u))))
            all_metrics.append(eval_metrics("E2: Unified (Clean+FullGAN 70f)", sc, m_name, "Combined (200)", y200_te, p200_u, all_classes))

            # --- E3: Arch-Adaptive 70f ---
            est_a = ArchAdaptiveModel(m_name)
            est_a.fit(X32_tr, y32_tr, X64_comb_tr, y64_comb, sw64=sw64_norm)
            joblib.dump(est_a, f"{save_dir_a}/{m_name}.pkl")

            p32_a = est_a.predict(X32_te)
            p64_a = est_a.predict(X64_te)
            p200_a = np.concatenate([p32_a, p64_a])
            pred_records[f"KB{sc}_{m_name}_Pred"] = p200_a

            r32 = eval_metrics("E3: Arch-Adaptive (Clean+FullGAN 70f)", sc, m_name, "x32 (100)", y32_te, p32_a, sorted(set(y32_te) | set(p32_a)))
            r64 = eval_metrics("E3: Arch-Adaptive (Clean+FullGAN 70f)", sc, m_name, "x64 (100)", y64_te, p64_a, sorted(set(y64_te) | set(p64_a)))
            r200 = eval_metrics("E3: Arch-Adaptive (Clean+FullGAN 70f)", sc, m_name, "Combined (200)", y200_te, p200_a, all_classes)
            all_metrics.extend([r32, r64, r200])

            print(
                f"[{m_name:14s}] E3 Arch-Adaptive -> x32 Acc: {r32['Accuracy']:6.2f}% | "
                f"x64 Acc: {r64['Accuracy']:6.2f}% (BalAcc: {r64['Balanced_Acc']:5.2f}%, MacroF1: {r64['Macro_F1']:5.2f}%) | "
                f"Combined(200) Acc: {r200['Accuracy']:6.2f}% (MacroF1: {r200['Macro_F1']:5.2f}%)"
            )
            if sc == 1:
                cm64 = confusion_matrix(y64_te, p64_a, labels=["Benign", "Malware"])
                cm32 = confusion_matrix(y32_te, p32_a, labels=["Benign", "Malware"])
                print(f"   📌 CM x32 [Benign, Malware]: {cm32.tolist()} | CM x64 [Benign, Malware]: {cm64.tolist()}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("Metrics_DualArch_All_Comparisons.csv", index=False)
    pred_records.to_csv("Predictions_Test200_DualArch.csv", index=False)

    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG HỢP SO SÁNH ACCURACY (%) TRÊN BỘ TEST 200 MẪU (100 x32 + 100 x64)")
    print("=" * 80)
    piv_200 = metrics_df[metrics_df["Test_Subset"] == "Combined (200)"].pivot_table(
        index=["Scenario", "Model"], columns="Method", values="Accuracy"
    )
    print(piv_200.to_string())

    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG HỢP RIÊNG TRÊN 100 MẪU x64 (ACCURACY & BALANCED ACCURACY %)")
    print("=" * 80)
    piv_64 = metrics_df[metrics_df["Test_Subset"] == "x64 (100)"].pivot_table(
        index=["Scenario", "Model"], columns="Method", values=["Accuracy", "Balanced_Acc"]
    )
    print(piv_64.to_string())
    piv_200.to_csv("Summary_Combined200_Accuracy.csv")


if __name__ == "__main__":
    main()

