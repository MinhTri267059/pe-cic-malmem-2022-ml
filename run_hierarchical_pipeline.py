"""
Pipeline Phân loại Mã độc Phân cấp 3 Tầng (3-Stage Hierarchical Cascade Pipeline)
trên bộ dữ liệu đa kiến trúc (x32 + x64 + Full-Feature cWGAN-GP, 70 đặc trưng):

  - TẦNG 1 (Stage 1 - Binary Filter):
      Xác định Nhị phân [Benign vs. Malware].
      + Nếu dự đoán là Benign -> Dừng ngay: trả về (KB1=Benign, KB2=Benign, KB3=Benign).
      + Nếu dự đoán là Malware -> Chuyển mẫu xuống Tầng 2.

  - TẦNG 2 (Stage 2 - Malware Group Classifier):
      Chỉ huấn luyện trên các mẫu Malware để phân biệt 3 nhóm độc hại:
      [Ransomware, Spyware, Trojan] (loại bỏ nhiễu từ lớp Benign).
      Sau khi xác định nhóm G in {Ransomware, Spyware, Trojan} -> Chuyển xuống chuyên gia Tầng 3 của nhóm G.

  - TẦNG 3 (Stage 3 - Group-Specific Family Specialists):
       gồm 3 bộ phân loại chuyên biệt (Specialists) chỉ học các họ trong đúng nhóm đó:
      + Chuyên gia Ransomware (6 họ): Ako, Conti, LockBit, Maze, Pysa, Shade
      + Chuyên gia Spyware    (8 họ): 180solutions, AgentTesla, CWS, Formbook, Gator, SnakeKeylogger, TIBS, Transponder
      + Chuyên gia Trojan     (9 họ): DCRat, Emotet, QuasarRAT, Reconyc, Refroso, Remcos, Scar, TrickBot, Zeus

Hỗ trợ cả 2 chế độ:
  1. Pipeline Thuần nhất theo từng thuật toán (XGBoost, Random_Forest, J_48, Random_Tree, Naive_Bayes)
     ở cả cấu hình Unified (Hợp nhất) và Arch-Adaptive (Thích ứng kiến trúc)
  2. Pipeline Kết hợp Tối ưu (Best-Stage Ensemble Cascade): kết hợp xác suất giữa XGBoost + Random Forest ở từng tầng.
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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train_and_eval_dual_arch import build_70_features, eval_metrics, get_labels

SEED = 42
MODEL_LIST = ["XGBoost", "Random_Forest", "J_48", "Random_Tree", "Naive_Bayes"]
GROUPS = ["Ransomware", "Spyware", "Trojan"]


def make_clf(name):
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
    if name == "Ensemble_XGB_RF":
        return VotingClassifier(
            estimators=[
                ("xgb", make_clf("XGBoost")),
                ("rf", make_clf("Random_Forest")),
            ],
            voting="soft",
        )
    raise ValueError(name)


def fit_encoded(clf_name, X, y, sample_weight=None):
    classes = sorted(np.unique(y))
    lmap = {c: i for i, c in enumerate(classes)}
    inv = {i: c for c, i in lmap.items()}
    y_idx = np.array([lmap[c] for c in y])
    clf = make_clf(clf_name)
    if clf_name == "Naive_Bayes":
        X_nb = X.drop(columns=["is_x64"], errors="ignore")
        clf.fit(X_nb, y_idx)
    else:
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y_idx)
        clf.fit(X, y_idx, sample_weight=sample_weight)
    return {"clf": clf, "inv": inv, "lmap": lmap, "classes": classes, "is_nb": clf_name == "Naive_Bayes"}


def predict_encoded(bundle, X):
    X_in = X.drop(columns=["is_x64"], errors="ignore") if bundle.get("is_nb") else X
    idx = bundle["clf"].predict(X_in)
    return np.array([bundle["inv"][int(i)] for i in idx], dtype=object)


class Hierarchical3StagePipeline:
    """
    Pipeline 3 Tầng (Tầng 1: Nhị phân -> Tầng 2: 3 Nhóm Malware -> Tầng 3: Chuyên gia Họ theo Nhóm).
    Hỗ trợ chế độ:
      - mode='unified'       : 1 bộ phân loại hợp nhất cho mỗi nút trong cây phân cấp
      - mode='arch_adaptive' : tự động định tuyến theo cờ is_x64 ở mỗi nút trong cây phân cấp
    """

    def __init__(self, model_name, mode="arch_adaptive"):
        self.model_name = model_name
        self.mode = mode

    def _compute_unified_weights(self, y32, y64_gan, y64_real):
        sw32 = compute_sample_weight("balanced", y32)
        sw32 = sw32 / sw32.sum() * 10000.0
        y64 = np.concatenate([y64_gan, y64_real])
        sw64 = compute_sample_weight("balanced", y64)
        boost = np.array([1.0] * len(y64_gan) + [4.0] * len(y64_real))
        sw64 = sw64 * boost
        sw64 = sw64 / sw64.sum() * 10000.0
        return np.concatenate([sw32, sw64]), sw64

    def fit(self, X32, df32, X64_gan, df64_gan, X64_real, df64_real):
        y32_s1 = get_labels(df32, 1).values
        y64g_s1 = get_labels(df64_gan, 1).values
        y64r_s1 = get_labels(df64_real, 1).values

        y32_s2 = get_labels(df32, 2).values
        y64g_s2 = get_labels(df64_gan, 2).values
        y64r_s2 = get_labels(df64_real, 2).values

        y32_s3 = get_labels(df32, 3).values
        y64g_s3 = get_labels(df64_gan, 3).values
        y64r_s3 = get_labels(df64_real, 3).values

        X64_comb = pd.concat([X64_gan, X64_real], ignore_index=True)
        X_all = pd.concat([X32, X64_comb], ignore_index=True)

        # ---------------- TẦNG 1: Benign vs Malware ----------------
        sw_u1, sw64_1 = self._compute_unified_weights(y32_s1, y64g_s1, y64r_s1)
        if self.mode == "unified":
            self.stage1_ = fit_encoded(
                self.model_name, X_all, np.concatenate([y32_s1, y64g_s1, y64r_s1]), sw_u1
            )
        else:
            self.stage1_32_ = fit_encoded(self.model_name, X32, y32_s1)
            self.stage1_64_ = fit_encoded(
                self.model_name, X64_comb, np.concatenate([y64g_s1, y64r_s1]), sw64_1
            )

        # ---------------- TẦNG 2: Chỉ mẫu Malware -> 3 Nhóm (Ransomware, Spyware, Trojan) ----------------
        m32 = y32_s1 == "Malware"
        m64g = y64g_s1 == "Malware"
        m64r = y64r_s1 == "Malware"

        X32_mal = X32.loc[m32].reset_index(drop=True)
        X64g_mal = X64_gan.loc[m64g].reset_index(drop=True)
        X64r_mal = X64_real.loc[m64r].reset_index(drop=True)
        X64_mal = pd.concat([X64g_mal, X64r_mal], ignore_index=True)
        X_all_mal = pd.concat([X32_mal, X64_mal], ignore_index=True)

        sw_u2, sw64_2 = self._compute_unified_weights(y32_s2[m32], y64g_s2[m64g], y64r_s2[m64r])
        if self.mode == "unified":
            self.stage2_ = fit_encoded(
                self.model_name,
                X_all_mal,
                np.concatenate([y32_s2[m32], y64g_s2[m64g], y64r_s2[m64r]]),
                sw_u2,
            )
        else:
            self.stage2_32_ = fit_encoded(self.model_name, X32_mal, y32_s2[m32])
            self.stage2_64_ = fit_encoded(
                self.model_name, X64_mal, np.concatenate([y64g_s2[m64g], y64r_s2[m64r]]), sw64_2
            )

        # ---------------- TẦNG 3: 3 Chuyên gia Họ mã độc cho từng Nhóm ----------------
        self.stage3_ = {}
        self.stage3_32_ = {}
        self.stage3_64_ = {}
        for grp in GROUPS:
            g32 = y32_s2 == grp
            g64g = y64g_s2 == grp
            g64r = y64r_s2 == grp

            X32_g = X32.loc[g32].reset_index(drop=True)
            X64g_g = X64_gan.loc[g64g].reset_index(drop=True)
            X64r_g = X64_real.loc[g64r].reset_index(drop=True)
            X64_g = pd.concat([X64g_g, X64r_g], ignore_index=True)
            X_all_g = pd.concat([X32_g, X64_g], ignore_index=True)

            sw_ug, sw64_g = self._compute_unified_weights(y32_s3[g32], y64g_s3[g64g], y64r_s3[g64r])
            if self.mode == "unified":
                self.stage3_[grp] = fit_encoded(
                    self.model_name,
                    X_all_g,
                    np.concatenate([y32_s3[g32], y64g_s3[g64g], y64r_s3[g64r]]),
                    sw_ug,
                )
            else:
                self.stage3_32_[grp] = fit_encoded(self.model_name, X32_g, y32_s3[g32])
                self.stage3_64_[grp] = fit_encoded(
                    self.model_name, X64_g, np.concatenate([y64g_s3[g64g], y64r_s3[g64r]]), sw64_g
                )
        return self

    def _pred_node(self, X, u_bundle, b32, b64):
        if self.mode == "unified":
            return predict_encoded(u_bundle, X)
        is_64 = X["is_x64"].values > 0.5
        out = np.empty(len(X), dtype=object)
        if (~is_64).any():
            out[~is_64] = predict_encoded(b32, X.loc[~is_64])
        if is_64.any():
            out[is_64] = predict_encoded(b64, X.loc[is_64])
        return out

    def predict_hierarchy(self, X):
        n = len(X)
        pred_s1 = self._pred_node(
            X,
            getattr(self, "stage1_", None),
            getattr(self, "stage1_32_", None),
            getattr(self, "stage1_64_", None),
        )
        pred_s2 = np.full(n, "Benign", dtype=object)
        pred_s3 = np.full(n, "Benign", dtype=object)

        mal_mask = pred_s1 == "Malware"
        if mal_mask.any():
            X_mal = X.loc[mal_mask]
            p2 = self._pred_node(
                X_mal,
                getattr(self, "stage2_", None),
                getattr(self, "stage2_32_", None),
                getattr(self, "stage2_64_", None),
            )
            pred_s2[mal_mask] = p2

            for grp in GROUPS:
                grp_mask = pred_s2 == grp
                if grp_mask.any():
                    X_grp = X.loc[grp_mask]
                    p3 = self._pred_node(
                        X_grp,
                        self.stage3_.get(grp),
                        self.stage3_32_.get(grp),
                        self.stage3_64_.get(grp),
                    )
                    pred_s3[grp_mask] = p3

        return pred_s1, pred_s2, pred_s3


def main():
    print("=" * 85)
    print("🚀 KHỞI CHẠY PIPELINE PHÂN CẤP 3 TẦNG (TẦNG 1: 2 NHÃN -> TẦNG 2: 3 NHÓM -> TẦNG 3: HỌ MÃ ĐỘC)")
    print("=" * 85)

    x32_tr = pd.read_csv("x32_train_clean.csv")
    x32_te = pd.read_csv("x32_test_100.csv")
    x64_tr = pd.read_csv("x64_train_100.csv")
    x64_te = pd.read_csv("x64_test_100.csv")
    x64_gan = pd.read_csv("x64_gan_1000_full.csv")

    raw_cols = [c for c in x64_tr.columns if c not in ["Class", "Category", "Filename"]]
    avail_cols = [c for c in raw_cols if x64_tr[c].notna().any()]

    X32_tr_raw = build_70_features(x32_tr, avail_cols, 0)
    X32_te_raw = build_70_features(x32_te, avail_cols, 0)
    X64_real_raw = build_70_features(x64_tr, avail_cols, 1)
    X64_gan_raw = build_70_features(x64_gan, avail_cols, 1)
    X64_te_raw = build_70_features(x64_te, avail_cols, 1)

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
    X200_te = pd.concat([X32_te, X64_te], ignore_index=True)

    # Nhãn thực tế trên bộ Test
    y_true = {}
    for sc in [1, 2, 3]:
        y32 = get_labels(x32_te, sc).values
        y64 = get_labels(x64_te, sc).values
        y_true[sc] = {"x32": y32, "x64": y64, "200": np.concatenate([y32, y64])}

    os.makedirs("Models_Hierarchical_Pipeline", exist_ok=True)
    pipeline_metrics = []
    stage_conditional_rows = []
    pred_df = pd.concat(
        [x32_te.assign(Arch="x32"), x64_te.assign(Arch="x64")], ignore_index=True
    )[["Filename", "Arch", "Class", "Category"]].copy()
    pred_df["True_KB1"] = y_true[1]["200"]
    pred_df["True_KB2"] = y_true[2]["200"]
    pred_df["True_KB3"] = y_true[3]["200"]

    all_algos = MODEL_LIST + ["Ensemble_XGB_RF"]

    for mode_key, mode_label in [
        ("unified", "Pipeline 3 Tầng (Unified 70f)"),
        ("arch_adaptive", "Pipeline 3 Tầng (Arch-Adaptive 70f)"),
    ]:
        print(f"\n{'=' * 85}\n🔥 CHẾ ĐỘ: {mode_label}\n{'=' * 85}")
        for m_name in all_algos:
            pipe = Hierarchical3StagePipeline(m_name, mode=mode_key)
            pipe.fit(X32_tr, x32_tr, X64_gan, x64_gan, X64_real, x64_tr)
            joblib.dump(pipe, f"Models_Hierarchical_Pipeline/{mode_key}_{m_name}.pkl")

            p32_s1, p32_s2, p32_s3 = pipe.predict_hierarchy(X32_te)
            p64_s1, p64_s2, p64_s3 = pipe.predict_hierarchy(X64_te)

            preds_by_sc = {
                1: (p32_s1, p64_s1, np.concatenate([p32_s1, p64_s1])),
                2: (p32_s2, p64_s2, np.concatenate([p32_s2, p64_s2])),
                3: (p32_s3, p64_s3, np.concatenate([p32_s3, p64_s3])),
            }

            if mode_key == "arch_adaptive" and m_name in ["XGBoost", "Random_Forest", "Ensemble_XGB_RF"]:
                pred_df[f"Pipe_{m_name}_T1"] = preds_by_sc[1][2]
                pred_df[f"Pipe_{m_name}_T2"] = preds_by_sc[2][2]
                pred_df[f"Pipe_{m_name}_T3"] = preds_by_sc[3][2]

            for sc in [1, 2, 3]:
                p32, p64, p200 = preds_by_sc[sc]
                r32 = eval_metrics(mode_label, sc, m_name, "x32 (100)", y_true[sc]["x32"], p32, sorted(set(y_true[sc]["x32"]) | set(p32)))
                r64 = eval_metrics(mode_label, sc, m_name, "x64 (100)", y_true[sc]["x64"], p64, sorted(set(y_true[sc]["x64"]) | set(p64)))
                r200 = eval_metrics(mode_label, sc, m_name, "Combined (200)", y_true[sc]["200"], p200, sorted(set(y_true[sc]["200"]) | set(p200)))
                pipeline_metrics.extend([r32, r64, r200])

            # Đánh giá độ chính xác có điều kiện tại từng tầng (Stage-wise Oracle / Conditional Accuracy)
            mal_200 = y_true[1]["200"] == "Malware"
            m_t1 = mal_200 & (preds_by_sc[1][2] == "Malware")
            s2_cond_acc = (
                accuracy_score(y_true[2]["200"][m_t1], preds_by_sc[2][2][m_t1])
                if m_t1.any()
                else 0.0
            )
            s2_matched = (y_true[2]["200"] == preds_by_sc[2][2]) & mal_200
            s3_cond_acc = (
                accuracy_score(y_true[3]["200"][s2_matched], preds_by_sc[3][2][s2_matched])
                if s2_matched.any()
                else 0.0
            )
            # Tính độ chính xác ranh giới nhóm: số mẫu dự đoán ở Tầng 3 không bao giờ bị nhảy sang nhóm khác với Tầng 2 (100% nhất quán phân cấp!)
            s2_from_s3 = pd.Series(preds_by_sc[3][2]).str.split("-").str[0].values
            hierarchical_consistency = np.mean(s2_from_s3 == preds_by_sc[2][2]) * 100.0

            stage_conditional_rows.append(
                {
                    "Mode": mode_label,
                    "Model": m_name,
                    "T1_Binary_Acc_200": round(accuracy_score(y_true[1]["200"], preds_by_sc[1][2]) * 100, 2),
                    "T2_EndToEnd_Acc_200": round(accuracy_score(y_true[2]["200"], preds_by_sc[2][2]) * 100, 2),
                    "T3_EndToEnd_Acc_200": round(accuracy_score(y_true[3]["200"], preds_by_sc[3][2]) * 100, 2),
                    "T2_Given_T1_Correct (%)": round(s2_cond_acc * 100, 2),
                    "T3_Given_T2_Correct (%)": round(s3_cond_acc * 100, 2),
                    "Consistency_T1_T2_T3 (%)": round(hierarchical_consistency, 2),
                }
            )

            print(
                f"[{m_name:15s}] "
                f"Tầng 1 (2 nhãn): {accuracy_score(y_true[1]['200'], preds_by_sc[1][2])*100:5.2f}% (x32 {accuracy_score(y_true[1]['x32'], p32_s1)*100:5.1f}%, x64 {accuracy_score(y_true[1]['x64'], p64_s1)*100:5.1f}%) | "
                f"Tầng 2 (Nhóm): {accuracy_score(y_true[2]['200'], preds_by_sc[2][2])*100:5.2f}% (x32 {accuracy_score(y_true[2]['x32'], p32_s2)*100:5.1f}%, x64 {accuracy_score(y_true[2]['x64'], p64_s2)*100:5.1f}%) | "
                f"Tầng 3 (Họ): {accuracy_score(y_true[3]['200'], preds_by_sc[3][2])*100:5.2f}% (x32 {accuracy_score(y_true[3]['x32'], p32_s3)*100:5.1f}%, x64 {accuracy_score(y_true[3]['x64'], p64_s3)*100:5.1f}%)"
            )

    pipe_df = pd.DataFrame(pipeline_metrics)
    pipe_df.to_csv("Metrics_Hierarchical_Pipeline_3Stages.csv", index=False)

    cond_df = pd.DataFrame(stage_conditional_rows)
    cond_df.to_csv("Metrics_Stage_Conditional_Analysis.csv", index=False)

    pred_df.to_csv("Predictions_Hierarchical_Pipeline_200.csv", index=False)

    print("\n" + "=" * 85)
    print("📊 PHÂN TÍCH ĐỘ CHÍNH XÁC QUA TỪNG TẦNG & ĐỘ NHẤT QUÁN PHÂN CẤP (TRÊN 200 MẪU TEST)")
    print("=" * 85)
    print(cond_df.to_string(index=False))

    # So sánh trực tiếp Flat vs Hierarchical Pipeline trên Combined (200)
    flat_df = pd.read_csv("Metrics_DualArch_All_Comparisons.csv")
    comb_all = pd.concat([flat_df, pipe_df], ignore_index=True)
    piv = comb_all[comb_all["Test_Subset"] == "Combined (200)"].pivot_table(
        index=["Scenario", "Model"],
        columns="Method",
        values="Accuracy",
    )
    print("\n" + "=" * 85)
    print("📊 SO SÁNH ACCURACY (%): MÔ HÌNH ĐƠN LẺ (FLAT) vs. PIPELINE PHÂN CẤP 3 TẦNG (200 MẪU TEST)")
    print("=" * 85)
    print(piv.to_string())
    piv.to_csv("Summary_Flat_vs_Hierarchical_Pipeline.csv")


if __name__ == "__main__":
    main()
