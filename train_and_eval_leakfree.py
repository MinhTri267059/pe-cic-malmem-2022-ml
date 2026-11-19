"""
Bước 3: Huấn luyện & Đánh giá trên Kịch bản Chống Rò Rỉ Tuyệt Đối (Leak-Free Benchmark):
  - Tập Test x32 Cô lập (80 dòng: 40 Benign + 40 Malware từ 40 mã Hash hoàn toàn chưa từng xuất hiện trong Train x32)
  - Tập Test x64 Cô lập (80 dòng: 10 con/phiên chạy hoàn toàn chưa từng xuất hiện trong Train x64 hay GAN)
  - Tổng Bộ Test Kép Không Rò Rỉ: 160 mẫu (80 x32 + 80 x64, 0% overlap của Instance_ID ở cả hai kiến trúc).
  - Bổ sung đánh giá 5-Fold GroupKFold Cross-Validation trên toàn bộ 20 con/phiên chạy của features_all_x64_19041_Cleaned.csv.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
MODEL_LIST = ["XGBoost", "Random_Forest", "Ensemble_XGB_RF", "J_48", "Random_Tree", "Naive_Bayes"]


def build_49_clean_features(df, clean_32_cols, is_x64_val):
    """
    Từ 32 đặc trưng động đã làm sạch (không có cột NaN, không có cột hằng số = 0,
    không có cột trùng lặp, không có cột cấu hình tĩnh định danh phiên VM như svcscan/callbacks/modules),
    chế tác thêm 16 đặc trưng tỷ lệ hành vi (Behavioral Ratios) + 1 cờ is_x64 = 49 đặc trưng sạch.
    """
    d = df[clean_32_cols].copy().fillna(0.0).astype(np.float64)
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
    d["is_x64"] = float(is_x64_val)
    return d


def make_reg_estimator(name):
    """Sử dụng cây có điều chuẩn độ sâu (Regularized Depth) để chống học thuộc lòng phiên chạy và tổng quát hóa sang mẫu mới."""
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            min_child_weight=2,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=SEED,
            eval_metric="mlogloss",
            n_jobs=4,
        )
    if name == "Random_Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=4,
        )
    if name == "Ensemble_XGB_RF":
        return VotingClassifier(
            estimators=[
                ("xgb", make_reg_estimator("XGBoost")),
                ("rf", make_reg_estimator("Random_Forest")),
            ],
            voting="soft",
        )
    if name == "J_48":
        return DecisionTreeClassifier(
            criterion="entropy", max_depth=7, min_samples_leaf=3, class_weight="balanced", random_state=SEED
        )
    if name == "Random_Tree":
        return DecisionTreeClassifier(
            splitter="random", max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=SEED
        )
    if name == "Naive_Bayes":
        return GaussianNB(var_smoothing=1e-1)
    raise ValueError(name)


def get_labels(df, sc):
    if sc == 1:
        return df["Class"]
    if sc == 2:
        return df["Category"].str.split("-").str[0]
    return df["Category"].str.split("-").str[:2].str.join("-")


def calc_row(method, sc, model_name, subset, y_true, y_pred, labels):
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
        "Test_Subset": subset,
        "Accuracy": round(acc * 100, 2),
        "Balanced_Acc": round(bacc * 100, 2),
        "Macro_F1": round(f1_m * 100, 2),
        "Weighted_F1": round(f1_w * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "FPR": round(fpr * 100, 2),
    }


def main():
    print("=" * 85)
    print("🚀 ĐÁNH GIÁ KỊCH BẢN CHỐNG RÒ RỈ TUYỆT ĐỐI (0% INSTANCE/HASH OVERLAP: 80 x32 + 80 x64)")
    print("=" * 85)

    x32_tr = pd.read_csv("x32_train_leakfree.csv")
    x32_te = pd.read_csv("x32_test_leakfree.csv")
    x64_tr = pd.read_csv("x64_train_leakfree.csv")
    x64_te = pd.read_csv("x64_test_leakfree.csv")
    x64_gan = pd.read_csv("x64_gan_1000_leakfree.csv")
    x64_all_clean = pd.read_csv("features_all_x64_19041_Cleaned.csv")

    clean_32_cols = [
        c for c in x64_tr.columns if c not in ["Class", "Category", "Filename", "Instance_ID", "Dump_Num"]
    ]

    X32_tr_r = build_49_clean_features(x32_tr, clean_32_cols, 0)
    X32_te_r = build_49_clean_features(x32_te, clean_32_cols, 0)
    X64_real_r = build_49_clean_features(x64_tr, clean_32_cols, 1)
    X64_gan_r = build_49_clean_features(x64_gan, clean_32_cols, 1)
    X64_te_r = build_49_clean_features(x64_te, clean_32_cols, 1)

    num_cols = [c for c in X32_tr_r.columns if c != "is_x64"]
    sc32 = StandardScaler().fit(X32_tr_r[num_cols])
    sc64 = StandardScaler().fit(X64_real_r[num_cols])

    def scale_df(d, sc_obj):
        out = d.copy()
        out[num_cols] = sc_obj.transform(d[num_cols])
        return out

    X32_tr = scale_df(X32_tr_r, sc32)
    X32_te = scale_df(X32_te_r, sc32)
    X64_real = scale_df(X64_real_r, sc64)
    X64_gan = scale_df(X64_gan_r, sc64)
    X64_te = scale_df(X64_te_r, sc64)

    X64_comb_tr = pd.concat([X64_gan, X64_real], ignore_index=True)
    X_all_tr = pd.concat([X32_tr, X64_comb_tr], ignore_index=True)

    rows = []
    os.makedirs("Models_LeakFree", exist_ok=True)

    for sc in [1, 2, 3]:
        print(f"\n{'=' * 85}\n🔥 KỊCH BẢN {sc} (CHỐNG RÒ RỈ 100% - MẪU TEST LÀ CON MÃ ĐỘC CHƯA TỪNG THẤY)\n{'=' * 85}")
        y32_tr = get_labels(x32_tr, sc).values
        y64_gan = get_labels(x64_gan, sc).values
        y64_real = get_labels(x64_tr, sc).values
        y64_comb = np.concatenate([y64_gan, y64_real])
        y_all_tr = np.concatenate([y32_tr, y64_comb])

        y32_te = get_labels(x32_te, sc).values
        y64_te = get_labels(x64_te, sc).values
        y160_te = np.concatenate([y32_te, y64_te])

        # Ở KB3 trên x64, tách riêng đánh giá trên 72/80 mẫu thuộc các họ đã có đại diện khác trong Train (Closed-set Unseen Variants: Benign, Conti, LockBit, Pysa, AgentTesla, Formbook, QuasarRAT, Remcos) vs toàn bộ 80 mẫu (bao gồm cả họ đơn TrickBot-2)
        seen_fams = set(y_all_tr)
        known_mask_64 = np.array([c in seen_fams for c in y64_te])

        classes_all = sorted(set(y_all_tr))
        lmap_all = {c: i for i, c in enumerate(classes_all)}
        inv_all = {i: c for c, i in lmap_all.items()}

        sw32 = compute_sample_weight("balanced", y32_tr)
        sw32 = sw32 / sw32.sum() * 10000.0
        sw64 = compute_sample_weight("balanced", y64_comb)
        boost = np.array([1.0] * len(y64_gan) + [4.0] * len(y64_real))
        sw64 = (sw64 * boost) / (sw64 * boost).sum() * 10000.0
        sw_all = np.concatenate([sw32, sw64])

        eval_lbls = sorted(set(classes_all) | set(y160_te))

        for m_name in MODEL_LIST:
            # --- Unified Leak-Free (49 Clean Features + Leak-Free GAN) ---
            clf_u = make_reg_estimator(m_name)
            y_idx_all = np.array([lmap_all[c] for c in y_all_tr])
            if m_name == "Naive_Bayes":
                clf_u.fit(X_all_tr.drop(columns=["is_x64"]), y_idx_all)
                p32 = np.array([inv_all[int(i)] for i in clf_u.predict(X32_te.drop(columns=["is_x64"]))])
                p64 = np.array([inv_all[int(i)] for i in clf_u.predict(X64_te.drop(columns=["is_x64"]))])
            else:
                clf_u.fit(X_all_tr, y_idx_all, sample_weight=sw_all)
                p32 = np.array([inv_all[int(i)] for i in clf_u.predict(X32_te)])
                p64 = np.array([inv_all[int(i)] for i in clf_u.predict(X64_te)])

            p160 = np.concatenate([p32, p64])
            r32 = calc_row("LeakFree Unified (49f+GAN)", sc, m_name, "x32 Unseen Hashes (80)", y32_te, p32, sorted(set(y32_te) | set(p32)))
            r64 = calc_row("LeakFree Unified (49f+GAN)", sc, m_name, "x64 Unseen Instances (80)", y64_te, p64, sorted(set(y64_te) | set(p64)))
            r160 = calc_row("LeakFree Unified (49f+GAN)", sc, m_name, "Combined LeakFree (160)", y160_te, p160, eval_lbls)
            rows.extend([r32, r64, r160])

            extra_kb3 = ""
            if sc == 3:
                acc_known = accuracy_score(y64_te[known_mask_64], p64[known_mask_64]) * 100.0
                extra_kb3 = f" | x64 Known-Family Unseen-Variant (72 mẫu): {acc_known:5.2f}%"

            print(
                f"[{m_name:15s}] x32 (80 Hash mới): {r32['Accuracy']:6.2f}% | "
                f"x64 (10 Con mới): {r64['Accuracy']:6.2f}% (BalAcc: {r64['Balanced_Acc']:5.2f}%) | "
                f"Combined(160): {r160['Accuracy']:6.2f}% (MacroF1: {r160['Macro_F1']:5.2f}%){extra_kb3}"
            )
            if sc == 1 and m_name in ["Random_Forest", "Ensemble_XGB_RF", "XGBoost"]:
                cm32 = confusion_matrix(y32_te, p32, labels=["Benign", "Malware"]).tolist()
                cm64 = confusion_matrix(y64_te, p64, labels=["Benign", "Malware"]).tolist()
                print(f"   📌 CM x32 [Benign(40), Malware(40)]: {cm32} | CM x64 [Benign(8), Malware(72)]: {cm64}")

    # =========================================================================
    # 4. ĐÁNH GIÁ 5-FOLD GROUPKFOLD TRÊN TOÀN BỘ 20 CON CỦA TẬP x64 SẠCH (160 DÒNG)
    # =========================================================================
    print("\n" + "=" * 85)
    print("🔬 KIỂM ĐỊNH CHÉO 5-FOLD GROUPKFOLD TRÊN TOÀN BỘ 20 CON MÃ ĐỘC/PHIÊN CHẠY x64 (160 DÒNG SẠCH)")
    print("=" * 85)
    X64_all_f = build_49_clean_features(x64_all_clean, clean_32_cols, 1).drop(columns=["is_x64"])
    gkf = GroupKFold(n_splits=5)
    groups_64 = x64_all_clean["Instance_ID"].values

    gkf_rows = []
    for sc, col_fn in [(1, lambda d: d["Class"]), (2, lambda d: d["Category"].str.split("-").str[0])]:
        y_g = col_fn(x64_all_clean).values
        classes_g = sorted(np.unique(y_g))
        lmap_g = {c: i for i, c in enumerate(classes_g)}
        inv_g = {i: c for c, i in lmap_g.items()}
        y_g_idx = np.array([lmap_g[c] for c in y_g])

        for m_name in ["Random_Forest", "Ensemble_XGB_RF", "XGBoost", "J_48", "Random_Tree"]:
            clf_g = make_reg_estimator(m_name)
            p_g_idx = cross_val_predict(clf_g, X64_all_f, y_g_idx, cv=gkf, groups=groups_64, n_jobs=1)
            p_g = np.array([inv_g[int(i)] for i in p_g_idx])
            acc_g = accuracy_score(y_g, p_g) * 100.0
            bacc_g = balanced_accuracy_score(y_g, p_g) * 100.0
            mf1_g = f1_score(y_g, p_g, average="macro") * 100.0
            gkf_rows.append({
                "Scenario": f"KB{sc}", "Model": m_name,
                "GroupKFold_Acc (%)": round(acc_g, 2),
                "GroupKFold_BalAcc (%)": round(bacc_g, 2),
                "GroupKFold_MacroF1 (%)": round(mf1_g, 2),
            })
            print(f"[5-Fold GroupKFold - KB{sc}] {m_name:15s} -> Acc: {acc_g:6.2f}% | BalAcc: {bacc_g:6.2f}% | MacroF1: {mf1_g:6.2f}%")

    df_res = pd.DataFrame(rows)
    df_res.to_csv("Metrics_LeakFree_Benchmark.csv", index=False)
    pd.DataFrame(gkf_rows).to_csv("Metrics_x64_5Fold_GroupKFold.csv", index=False)
    print("\n✅ Đã xuất kết quả chống rò rỉ ra: Metrics_LeakFree_Benchmark.csv & Metrics_x64_5Fold_GroupKFold.csv")


if __name__ == "__main__":
    main()
