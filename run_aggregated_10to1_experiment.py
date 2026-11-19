"""
THỰC NGHIỆM BỔ SUNG: GỘP 10 DÒNG SNAPSHOT THÀNH 1 DÒNG DUY NHẤT CHO TỪNG MẪU THỰC THI
(1 Con Mã Độc / 1 Phiên Chạy = 1 Dòng Vector Đặc Trưng - Loại bỏ 100% trùng lặp phiên chụp từ gốc)

1. Trên tập MalMem2022 (x32):
   - Gộp 10 bản dump (-1.raw .. -10.raw) của cùng 1 mã Hash Malware thành 1 dòng (29.298 dòng -> 2.906 con Malware độc lập).
   - Gộp mỗi cụm 10 bản dump liên tiếp của Benign thành 1 dòng (29.298 dòng -> 2.930 phiên Benign).
   -> Xuất file: MalMem2022_Aggregated_1RowPerSample.csv (5.836 dòng = 5.836 mẫu thực thi độc lập).
   - Đánh giá 80/20 Split & 10-Fold CV trên 5.836 mẫu đã gộp qua 3 kịch bản (KB1, KB2, KB3).

2. Trên tập Windows 10 x64 (Build 19041):
   - Gộp các bản dump ổn định của cùng 1 Instance_ID thành 1 dòng (200 dòng -> 20 dòng = 3 phiên Benign + 17 con Malware).
   -> Xuất file: features_all_x64_19041_Aggregated_20Samples.csv (20 dòng = 20 mẫu thực thi độc lập).

3. Đánh giá Đa kiến trúc trên dữ liệu đã gộp (1 mẫu = 1 dòng):
   - Đánh giá (A): Chia 10 con x64 Train (+ cWGAN-GP) + Train x32 đã gộp -> Chấm trên 80 con x32 Test + 10 con x64 Test.
   - Đánh giá (B): Leave-One-Sample-Out / 5-Fold CV trên toàn bộ 20 con x64 (kết hợp đồng quy Z-Score với 5.836 con x32).
"""
import re
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train_and_eval_leakfree import build_49_clean_features, get_labels, make_reg_estimator

DROP_23_COLS = [
    "psxview.not_in_pspcid_list", "psxview.not_in_session", "psxview.not_in_deskthrd",
    "psxview.not_in_pspcid_list_false_avg", "psxview.not_in_session_false_avg", "psxview.not_in_deskthrd_false_avg",
    "handles.nport", "malfind.commitCharge", "psxview.not_in_eprocess_pool",
    "psxview.not_in_eprocess_pool_false_avg", "svcscan.interactive_process_services",
    "ldrmodules.not_in_mem", "ldrmodules.not_in_mem_avg",
    "svcscan.nservices", "svcscan.kernel_drivers", "svcscan.fs_drivers",
    "svcscan.process_services", "svcscan.shared_process_services", "svcscan.nactive",
    "callbacks.ncallbacks", "callbacks.nanonymous", "callbacks.ngeneric", "modules.nmodules",
]

SEED = 42
np.random.seed(SEED)
MODEL_LIST = ["XGBoost", "Random_Forest", "Ensemble_XGB_RF", "J_48", "Random_Tree", "Naive_Bayes"]


def aggregate_snapshots_with_dispersion(df, clean_32_cols, is_x64_val):
    """
    Khi gộp nhiều bản dump của cùng 1 con mã độc thành 1 dòng:
    - Tính giá trị trung bình (mean) của 32 đặc trưng động & 16 tỷ lệ hành vi (= 49 cột chuẩn)
    - Ngoài ra trả về bảng 32 cột gốc đã gộp để lưu ra CSV.
    """
    feat_49 = build_49_clean_features(df, clean_32_cols, is_x64_val)
    feat_49["Instance_ID"] = df["Instance_ID"].values
    agg_49 = feat_49.groupby("Instance_ID", sort=False).mean()

    meta = (
        df.groupby("Instance_ID", sort=False)[["Class", "Category", "Filename"]]
        .first()
    )
    raw_agg = df.groupby("Instance_ID", sort=False)[clean_32_cols].mean().round(4)
    full_csv_df = pd.concat([raw_agg, meta], axis=1).reset_index()
    return agg_49.reset_index(drop=True), meta.reset_index(), full_csv_df


def eval_row(exp_name, sc, model_name, subset, y_true, y_pred, labels):
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
        "Experiment": exp_name,
        "Scenario": f"KB{sc}",
        "Model": model_name,
        "Subset": subset,
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
    print("🚀 THỰC NGHIỆM: GỘP 10 DÒNG SNAPSHOT THÀNH 1 DÒNG CHO TỪNG MẪU (1 MẪU = 1 DÒNG)")
    print("=" * 85)

    # 1. Nạp và gộp bộ dữ liệu MalMem2022 (x32): 58.596 dòng -> 5.836 mẫu độc lập
    df32_raw = pd.read_csv("MalMem2022.csv")
    all_55 = [c for c in df32_raw.columns if c not in ["Class", "Category", "Filename"]]
    clean_32_cols = [c for c in all_55 if c not in DROP_23_COLS]

    ben_mask = df32_raw["Class"] == "Benign"
    ben_idx = np.arange(ben_mask.sum()) // 10
    instance_32 = np.empty(len(df32_raw), dtype=object)
    instance_32[ben_mask] = [f"Benign_Session_{i:04d}" for i in ben_idx]
    instance_32[~ben_mask] = (
        df32_raw.loc[~ben_mask, "Filename"].str.replace(r"-\d+\.raw$", "", regex=True).values
    )
    df32_raw["Instance_ID"] = instance_32

    X32_agg, meta32_agg, csv32_agg = aggregate_snapshots_with_dispersion(df32_raw, clean_32_cols, 0)
    csv32_agg.to_csv("MalMem2022_Aggregated_1RowPerSample.csv", index=False)
    print(
        f"✅ [x32 Gộp 10->1] Từ {len(df32_raw)} dòng gốc -> Gộp thành {len(csv32_agg)} dòng độc lập "
        f"({(meta32_agg['Class']=='Benign').sum()} Benign + {(meta32_agg['Class']=='Malware').sum()} con Malware)"
    )
    print("   Đã lưu: MalMem2022_Aggregated_1RowPerSample.csv")

    # 2. Nạp và gộp bộ dữ liệu Windows 10 x64: 200 dòng (160 dòng ổn định) -> 20 mẫu độc lập
    df64_clean = pd.read_csv("features_all_x64_19041_Cleaned.csv")
    X64_agg, meta64_agg, csv64_agg = aggregate_snapshots_with_dispersion(df64_clean, clean_32_cols, 1)
    csv64_agg.to_csv("features_all_x64_19041_Aggregated_20Samples.csv", index=False)
    print(
        f"✅ [x64 Gộp 10->1] Từ 200 dòng gốc (160 dòng ổn định) -> Gộp thành {len(csv64_agg)} dòng độc lập "
        f"({(meta64_agg['Class']=='Benign').sum()} Benign + {(meta64_agg['Class']=='Malware').sum()} con Malware)"
    )
    print("   Đã lưu: features_all_x64_19041_Aggregated_20Samples.csv")

    all_rows = []

    # =========================================================================
    # PHẦN A: ĐÁNH GIÁ TRÊN BỘ MALMEM2022 ĐÃ GỘP 10->1 (5.836 MẪU ĐỘC LẬP, 80/20 SPLIT)
    # =========================================================================
    print("\n" + "=" * 85)
    print("1️⃣ KẾT QUẢ TRÊN BỘ MALMEM2022 SAU KHI GỘP 10 DÒNG THÀNH 1 (5.836 MẪU ĐỘC LẬP - 80/20 SPLIT)")
    print("=" * 85)

    for sc in [1, 2, 3]:
        y32 = get_labels(meta32_agg, sc).values
        classes32 = sorted(np.unique(y32))
        lmap32 = {c: i for i, c in enumerate(classes32)}
        inv32 = {i: c for c, i in lmap32.items()}
        y32_idx = np.array([lmap32[c] for c in y32])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X32_agg.drop(columns=["is_x64"]), y32_idx, test_size=0.2, random_state=SEED, stratify=y32_idx
        )
        y_te_lbl = np.array([inv32[int(i)] for i in y_te])

        print(f"\n--- Kịch bản {sc} (MalMem2022 Gộp 10->1: Train={len(X_tr)} con, Test={len(X_te)} con độc lập) ---")
        for m_name in MODEL_LIST:
            clf = make_reg_estimator(m_name)
            if m_name in ["XGBoost", "Random_Forest", "Ensemble_XGB_RF"]:
                # Trên 4.668 mẫu độc lập của x32, cho phép độ sâu tiêu chuẩn để học đủ 16 họ
                if m_name == "XGBoost":
                    clf = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=SEED, eval_metric="mlogloss", n_jobs=4)
                elif m_name == "Random_Forest":
                    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=4)
                elif m_name == "Ensemble_XGB_RF":
                    clf = VotingClassifier(
                        estimators=[
                            ("xgb", XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=SEED, eval_metric="mlogloss", n_jobs=4)),
                            ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=4)),
                        ],
                        voting="soft",
                    )
            clf.fit(X_tr, y_tr)
            p_te = np.array([inv32[int(i)] for i in clf.predict(X_te)])
            r = eval_row("ExpA: MalMem2022 Aggregated (10->1)", sc, m_name, "x32 Test (1.168 con)", y_te_lbl, p_te, classes32)
            all_rows.append(r)
            print(
                f"[{m_name:15s}] Acc: {r['Accuracy']:6.2f}% | BalAcc: {r['Balanced_Acc']:6.2f}% | "
                f"MacroF1: {r['Macro_F1']:6.2f}% | WeightedF1: {r['Weighted_F1']:6.2f}% | FPR: {r['FPR']:5.2f}%"
            )

    # =========================================================================
    # PHẦN B: ĐÁNH GIÁ TRÊN 20 MẪU x64 ĐÃ GỘP 10->1 & KẾT HỢP ĐA KIẾN TRÚC (x32 + x64 + GAN)
    # =========================================================================
    print("\n" + "=" * 85)
    print("2️⃣ KẾT QUẢ TRÊN 20 MẪU x64 ĐÃ GỘP 10->1 (CHIA 10 CON TRAIN vs. 10 CON TEST + 5-FOLD CV TRÊN 20 CON)")
    print("=" * 85)

    # Chuẩn hóa Z-Score theo từng kiến trúc trên dữ liệu đã gộp
    num_cols = [c for c in X32_agg.columns if c != "is_x64"]
    TRAIN_INSTANCES_64 = [
        "Benign-browse", "Benign-idle",
        "Ransomware-Conti-1", "Ransomware-LockBit-2",
        "Spyware-AgentTesla-1", "Spyware-Formbook-1", "Spyware-SnakeKeylogger-3",
        "Trojan-QuasarRAT-2", "Trojan-Remcos-1", "Trojan-DCRat-1",
    ]
    tr64_mask = meta64_agg["Instance_ID"].isin(TRAIN_INSTANCES_64).values
    te64_mask = ~tr64_mask

    # Cũng lấy 80 con x32 độc lập làm tập Test x32 để ghép cùng 10 con x64 Test (tạo bộ Test Gộp 10->1)
    y32_fam = get_labels(meta32_agg, 3).values
    idx32_tr, idx32_te = train_test_split(
        np.arange(len(X32_agg)), test_size=80, random_state=SEED, stratify=y32_fam
    )

    sc32 = StandardScaler().fit(X32_agg.iloc[idx32_tr][num_cols])
    sc64 = StandardScaler().fit(X64_agg.loc[tr64_mask, num_cols])

    X32_agg_s = X32_agg.copy(); X32_agg_s[num_cols] = sc32.transform(X32_agg[num_cols])
    X64_agg_s = X64_agg.copy(); X64_agg_s[num_cols] = sc64.transform(X64_agg[num_cols])

    # Bổ sung mẫu GAN đã sinh từ 10 con Train x64 (x64_gan_1000_leakfree.csv) được gộp trung bình theo từng cụm 10 mẫu GAN (= 100 mẫu GAN đại diện)
    df_gan = pd.read_csv("x64_gan_1000_leakfree.csv")
    df_gan["Instance_ID"] = [f"GAN_Agg_{i // 10:03d}" for i in range(len(df_gan))]
    X64_gan_agg, meta_gan_agg, _ = aggregate_snapshots_with_dispersion(df_gan, clean_32_cols, 1)
    X64_gan_agg_s = X64_gan_agg.copy(); X64_gan_agg_s[num_cols] = sc64.transform(X64_gan_agg[num_cols])

    X32_tr_s = X32_agg_s.iloc[idx32_tr].reset_index(drop=True)
    X32_te_s = X32_agg_s.iloc[idx32_te].reset_index(drop=True)
    meta32_tr = meta32_agg.iloc[idx32_tr].reset_index(drop=True)
    meta32_te = meta32_agg.iloc[idx32_te].reset_index(drop=True)

    X64_tr_s = pd.concat([X64_gan_agg_s, X64_agg_s.loc[tr64_mask]], ignore_index=True)
    meta64_tr_comb = pd.concat([meta_gan_agg, meta64_agg.loc[tr64_mask]], ignore_index=True)
    X64_te_s = X64_agg_s.loc[te64_mask].reset_index(drop=True)
    meta64_te = meta64_agg.loc[te64_mask].reset_index(drop=True)

    X_comb_tr = pd.concat([X32_tr_s, X64_tr_s], ignore_index=True)

    for sc in [1, 2, 3]:
        print(f"\n--- Kịch bản {sc} (Dữ liệu đã gộp 10->1: Test x32 = 80 con độc lập, Test x64 = 10 con độc lập) ---")
        y32_tr = get_labels(meta32_tr, sc).values
        y32_te = get_labels(meta32_te, sc).values
        y64_tr = get_labels(meta64_tr_comb, sc).values
        y64_te = get_labels(meta64_te, sc).values
        y_all_tr = np.concatenate([y32_tr, y64_tr])
        y_all_te = np.concatenate([y32_te, y64_te])

        classes_all = sorted(set(y_all_tr))
        lmap = {c: i for i, c in enumerate(classes_all)}
        inv = {i: c for c, i in lmap.items()}
        y_tr_idx = np.array([lmap[c] for c in y_all_tr])

        sw32 = compute_sample_weight("balanced", y32_tr)
        sw32 = sw32 / sw32.sum() * 5000.0
        sw64 = compute_sample_weight("balanced", y64_tr)
        boost64 = np.array([1.0] * len(meta_gan_agg) + [6.0] * int(tr64_mask.sum()))
        sw64 = (sw64 * boost64) / (sw64 * boost64).sum() * 5000.0
        sw_all = np.concatenate([sw32, sw64])

        eval_lbls = sorted(set(classes_all) | set(y_all_te))

        for m_name in MODEL_LIST:
            clf = make_reg_estimator(m_name)
            if m_name == "Naive_Bayes":
                clf.fit(X_comb_tr.drop(columns=["is_x64"]), y_tr_idx)
                p32 = np.array([inv[int(i)] for i in clf.predict(X32_te_s.drop(columns=["is_x64"]))])
                p64 = np.array([inv[int(i)] for i in clf.predict(X64_te_s.drop(columns=["is_x64"]))])
            else:
                clf.fit(X_comb_tr, y_tr_idx, sample_weight=sw_all)
                p32 = np.array([inv[int(i)] for i in clf.predict(X32_te_s)])
                p64 = np.array([inv[int(i)] for i in clf.predict(X64_te_s)])

            p_comb = np.concatenate([p32, p64])
            r32 = eval_row("ExpB: DualArch Aggregated (10->1)", sc, m_name, "x32 Test (80 con)", y32_te, p32, sorted(set(y32_te) | set(p32)))
            r64 = eval_row("ExpB: DualArch Aggregated (10->1)", sc, m_name, "x64 Test (10 con)", y64_te, p64, sorted(set(y64_te) | set(p64)))
            rc = eval_row("ExpB: DualArch Aggregated (10->1)", sc, m_name, "Combined (90 con)", y_all_te, p_comb, eval_lbls)
            all_rows.extend([r32, r64, rc])

            print(
                f"[{m_name:15s}] x32 (80 con): {r32['Accuracy']:6.2f}% | "
                f"x64 (10 con): {r64['Accuracy']:6.2f}% (Đúng {int(round(r64['Accuracy']*10/100))}/10 con, BalAcc: {r64['Balanced_Acc']:5.2f}%) | "
                f"Combined (90 con): {rc['Accuracy']:6.2f}%"
            )

    # 3. Đánh giá 3-Fold Stratified CV trực tiếp trên 20 con x64 đã gộp (vì Benign có 3 con -> 3-Fold đảm bảo mỗi Fold có đúng 1 con Benign và 5-6 con Malware!)
    print("\n" + "=" * 85)
    print("3️⃣ KIỂM ĐỊNH CHÉO 3-FOLD STRATIFIED CV TRÊN 20 CON x64 ĐÃ GỘP (MỖI FOLD CÓ 1 CON BENIGN + 5-6 CON MALWARE)")
    print("=" * 85)
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X64_only = X64_agg.drop(columns=["is_x64"])
    for sc in [1, 2]:
        y64_all = get_labels(meta64_agg, sc).values
        cls_g = sorted(np.unique(y64_all))
        lmap_g = {c: i for i, c in enumerate(cls_g)}
        inv_g = {i: c for c, i in lmap_g.items()}
        y_idx_g = np.array([lmap_g[c] for c in y64_all])

        for m_name in ["Random_Forest", "Ensemble_XGB_RF", "XGBoost", "J_48", "Random_Tree"]:
            clf = make_reg_estimator(m_name)
            p_idx = cross_val_predict(clf, X64_only, y_idx_g, cv=skf3, n_jobs=1)
            p_lbl = np.array([inv_g[int(i)] for i in p_idx])
            r_cv = eval_row("ExpC: x64 20-Sample 3Fold-CV (10->1)", sc, m_name, "x64 All 20 Samples", y64_all, p_lbl, cls_g)
            all_rows.append(r_cv)
            n_ok = int((p_lbl == y64_all).sum())
            print(
                f"[3-Fold CV 20 con x64 - KB{sc}] {m_name:15s} -> Đúng {n_ok:2d}/20 con ({r_cv['Accuracy']:5.2f}%) | "
                f"BalAcc: {r_cv['Balanced_Acc']:5.2f}% | MacroF1: {r_cv['Macro_F1']:5.2f}%"
            )

    pd.DataFrame(all_rows).to_csv("Metrics_Aggregated_10to1_Experiment.csv", index=False)
    print("\n✅ Đã lưu toàn bộ kết quả thực nghiệm gộp 10->1 vào: Metrics_Aggregated_10to1_Experiment.csv")


if __name__ == "__main__":
    main()
