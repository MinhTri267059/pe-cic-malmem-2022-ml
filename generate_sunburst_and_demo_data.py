"""
1. Vẽ biểu đồ Sunburst (2-Ring Concentric Donut Chart) phân bổ các họ mã độc thực tế
   theo đúng phong cách mẫu của người dùng (Benign màu xanh dương bên phải; Trojan xanh lá,
   Ransomware đỏ, Spyware vàng bên trái với nhãn họ xoay hướng tâm).
2. Trích xuất toàn bộ 200 mẫu từ bộ Test Chuẩn Kép (Test_200_x32_x64.csv) kèm kết quả dự đoán,
   độ tự tin (Confidence %) và đường đi phân tích của cả 2 Pipeline (Pipeline 1: Flat vs. Pipeline 2: 3 Tầng)
   để nạp vào ứng dụng Demo tương tác.
"""
import json
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from train_and_eval_dual_arch import build_70_features, get_labels, ArchAdaptiveModel
from run_hierarchical_pipeline import Hierarchical3StagePipeline

OUT_DIR = "eda_figures/DualArch_GAN"
ARTIFACT_DIR = "/Users/minhtri86/.gemini/antigravity/brain/06867a2e-b8f9-49bc-8bc8-06823bfb357c"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Bảng màu chuẩn theo đúng hình mẫu của người dùng
COLOR_MAP = {
    "Benign": "#3b82f6",      # Google Blue
    "Trojan": "#34a853",      # Google Green
    "Ransomware": "#ea4335",  # Google Red
    "Spyware": "#fbbc05",     # Google Yellow/Gold
}
TEXT_COLOR_MAP = {
    "Benign": "white",
    "Trojan": "white",
    "Ransomware": "white",
    "Spyware": "#222222",
}


def draw_sunburst(ax, df_source, title, normalize_malware_half=False):
    """
    Vẽ biểu đồ 2 vòng đồng tâm (Sunburst / Concentric Donut):
      - Vòng trong: Benign (bên phải) và 3 nhóm Trojan, Ransomware, Spyware (bên trái)
      - Vòng ngoài: Benign (bên phải) và từng họ mã độc con xoay chữ hướng tâm (bên trái)
    """
    df = df_source.copy()
    df["Group"] = df["Category"].str.split("-").str[0]
    df["Family"] = np.where(
        df["Group"] == "Benign",
        "Benign",
        df["Category"].str.split("-").str[1],
    )

    grp_order = ["Benign", "Trojan", "Ransomware", "Spyware"]
    inner_sizes = []
    inner_colors = []
    inner_labels = []

    outer_sizes = []
    outer_colors = []
    outer_labels = []
    outer_groups = []

    total_n = len(df)
    benign_n = int((df["Group"] == "Benign").sum())
    mal_n = total_n - benign_n

    # Nếu normalize_malware_half=True (cho tập x64 có 15% Benign), vẫn giữ tỷ lệ thực hoặc tùy chỉnh
    for g in grp_order:
        sub_g = df[df["Group"] == g]
        if len(sub_g) == 0:
            continue
        inner_sizes.append(len(sub_g))
        inner_colors.append(COLOR_MAP[g])
        inner_labels.append("Ransom-\nware" if g == "Ransomware" else g)

        if g == "Benign":
            outer_sizes.append(len(sub_g))
            outer_colors.append(COLOR_MAP[g])
            outer_labels.append("Benign")
            outer_groups.append(g)
        else:
            fam_counts = sub_g["Family"].value_counts()
            for fam_name, cnt in fam_counts.items():
                outer_sizes.append(int(cnt))
                outer_colors.append(COLOR_MAP[g])
                outer_labels.append(fam_name)
                outer_groups.append(g)

    # Để Benign nằm bên phải (từ góc -90 độ đến +90 độ), tính góc bắt đầu sao cho Benign căn giữa ở góc 0 độ (bên phải)
    benign_frac = inner_sizes[0] / sum(inner_sizes)
    start_angle = -benign_frac * 180.0

    # Vẽ vòng ngoài (r = 0.68 -> 1.00)
    wedges_out, _ = ax.pie(
        outer_sizes,
        radius=1.0,
        colors=outer_colors,
        startangle=start_angle,
        counterclock=True,
        wedgeprops=dict(width=0.32, edgecolor="white", linewidth=1.8),
    )

    # Vẽ vòng trong (r = 0.36 -> 0.68)
    wedges_in, _ = ax.pie(
        inner_sizes,
        radius=0.68,
        colors=inner_colors,
        startangle=start_angle,
        counterclock=True,
        wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2.2),
    )

    # Ghi nhãn vòng trong
    for w, lbl, g_name in zip(wedges_in, inner_labels, grp_order):
        ang = 0.5 * (w.theta1 + w.theta2)
        rad = np.deg2rad(ang)
        r_mid = 0.52
        x, y = r_mid * np.cos(rad), r_mid * np.sin(rad)
        rot = ang if -90 <= (ang % 360) <= 90 or (ang % 360) >= 270 else ang + 180
        if g_name in ["Benign", "Ransomware"]:
            rot = 0
        else:
            ang_norm = ((ang + 180) % 360) - 180
            rot = ang_norm - 90 if ang_norm > 0 else ang_norm + 90
        ax.text(
            x, y, lbl,
            ha="center", va="center",
            color=TEXT_COLOR_MAP[g_name],
            fontweight="bold",
            fontsize=10.5,
            rotation=rot,
        )

    # Ghi nhãn vòng ngoài (xoay hướng tâm căn giữa đúng ô giống hệt mẫu)
    for w, lbl, g_name, sz in zip(wedges_out, outer_labels, outer_groups, outer_sizes):
        ang = 0.5 * (w.theta1 + w.theta2)
        rad = np.deg2rad(ang)
        r_mid = 0.84
        x, y = r_mid * np.cos(rad), r_mid * np.sin(rad)
        if g_name == "Benign":
            rot = 0
            disp_lbl = "Benign"
            fsize = 10.5
        else:
            ang_norm = ((ang + 180) % 360) - 180
            rot = ang_norm if -90 <= ang_norm <= 90 else ang_norm + 180
            disp_lbl = lbl
            fsize = 7.8 if len(outer_labels) > 18 else 9.2
        ax.text(
            x, y, disp_lbl,
            ha="center", va="center",
            color=TEXT_COLOR_MAP[g_name],
            fontweight="bold",
            fontsize=fsize,
            rotation=rot,
        )

    ax.set_title(title, fontweight="bold", fontsize=12.5, pad=12)
    ax.set_aspect("equal")


# Load datasets
df_x32_clean = pd.read_csv("x32_train_clean.csv")
df_x32_test = pd.read_csv("x32_test_100.csv")
df_x32_all = pd.concat([df_x32_clean, df_x32_test], ignore_index=True)

df_x64_all = pd.read_csv("features_all_x64_19041.csv")
df_gan_1000 = pd.read_csv("x64_gan_1000_full.csv")
df_x64_upgraded = pd.concat([df_x64_all, df_gan_1000], ignore_index=True)

# 1. Vẽ Hình 14: So sánh 3 biểu đồ Sunburst (x32 Gốc vs x64 Thực tế 200 mẫu vs Tập hợp 24 Họ Mã độc Đa kiến trúc)
fig, axes = plt.subplots(1, 3, figsize=(22, 7.8))

draw_sunburst(
    axes[0],
    df_x32_all,
    "(A) Bộ dữ liệu Gốc CIC-MalMem-2022 (x32)\n(58.008 mẫu sạch: 50% Benign + 15 Họ mã độc x32)",
)
draw_sunburst(
    axes[1],
    df_x64_all,
    "(B) Bộ dữ liệu Thực tế Windows 10 x64 (Build 19041)\n(200 mẫu thật: 15% Benign + 10 Họ mã độc [8 Họ mới])",
)

# Để hiển thị cân đối toàn bộ 23 họ mã độc (15 họ x32 + 8 họ mới x64) + Benign trên biểu đồ tổng hợp:
# Lấy toàn bộ mẫu Malware của x32 + (x64 thật + x64 GAN nhân trọng số hiển thị miền) và Benign cân bằng 50%
df_viz_24 = pd.concat([
    df_x32_all,
    pd.concat([df_x64_upgraded[df_x64_upgraded["Class"] == "Malware"]] * 12, ignore_index=True),
], ignore_index=True)
draw_sunburst(
    axes[2],
    df_viz_24,
    "(C) Hệ sinh thái 24 Lớp sau Nâng cấp (x32 + x64 + FullGAN)\n(Benign + 23 Họ mã độc thuộc 3 nhóm Trojan, Ransomware, Spyware)",
)

fig.suptitle(
    "BIỂU ĐỒ PHÂN BỔ CÁC NHÓM VÀ HỌ MÃ ĐỘC THỰC TẾ: BỘ DỮ LIỆU GỐC (x32) vs. DỮ LIỆU MỚI (x64) & HỆ SINH THÁI NÂNG CẤP (24 LỚP)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
p14 = os.path.join(OUT_DIR, "14_sunburst_malware_families_comparison.png")
fig.savefig(p14, dpi=250, bbox_inches="tight", facecolor="white")
if os.path.exists(ARTIFACT_DIR):
    shutil.copy(p14, os.path.join(ARTIFACT_DIR, "14_sunburst_malware_families_comparison.png"))
plt.close(fig)
print(f"✅ Đã xuất: {p14}")


# =========================================================================
# 2. XUẤT DỮ LIỆU JSON CHO DEMO TƯƠNG TÁC 2 PIPELINE TRÊN 200 MẪU TEST
# =========================================================================
x32_tr = pd.read_csv("x32_train_clean.csv")
x32_te = pd.read_csv("x32_test_100.csv")
x64_tr = pd.read_csv("x64_train_100.csv")
x64_te = pd.read_csv("x64_test_100.csv")

raw_cols = [c for c in x64_tr.columns if c not in ["Class", "Category", "Filename"]]
avail_cols = [c for c in raw_cols if x64_tr[c].notna().any()]

X32_tr_r = build_70_features(x32_tr, avail_cols, 0)
X32_te_r = build_70_features(x32_te, avail_cols, 0)
X64_tr_r = build_70_features(x64_tr, avail_cols, 1)
X64_te_r = build_70_features(x64_te, avail_cols, 1)

num_cols = [c for c in X32_tr_r.columns if c != "is_x64"]
sc32 = StandardScaler().fit(X32_tr_r[num_cols])
sc64 = StandardScaler().fit(X64_tr_r[num_cols])

X32_te = X32_te_r.copy(); X32_te[num_cols] = sc32.transform(X32_te_r[num_cols])
X64_te = X64_te_r.copy(); X64_te[num_cols] = sc64.transform(X64_te_r[num_cols])
X200_te = pd.concat([X32_te, X64_te], ignore_index=True)
X200_raw = pd.concat([X32_te_r, X64_te_r], ignore_index=True)
df200_meta = pd.concat([x32_te.assign(Arch="x32 (Windows 7)"), x64_te.assign(Arch="x64 (Windows 10 Build 19041)")], ignore_index=True)

# Load best Flat models (Pipeline 1):
# KB1: Unified XGBoost (97.0%), KB2: Unified RF (81.5%) & Adaptive XGB (81.0%), KB3: Adaptive Ensemble XGB+RF (67.5%) & Unified XGB (65.0%)
flat_kb1_xgb = joblib.load("Models_DualArch_Unified/Scenario_1/XGBoost.pkl")
flat_kb2_rf = joblib.load("Models_DualArch_Unified/Scenario_2/Random_Forest.pkl")
flat_kb2_xgb = joblib.load("Models_DualArch_Unified/Scenario_2/XGBoost.pkl")
flat_kb3_xgb = joblib.load("Models_DualArch_Unified/Scenario_3/XGBoost.pkl")
flat_kb3_rf = joblib.load("Models_DualArch_Unified/Scenario_3/Random_Forest.pkl")

axgb3 = joblib.load("Models_DualArch_Best/Scenario_3/XGBoost.pkl")
arf3 = joblib.load("Models_DualArch_Best/Scenario_3/Random_Forest.pkl")
all_cls3 = sorted(set(axgb3.classes32_) | set(axgb3.classes64_))

lbl1 = ["Benign", "Malware"]
lbl2 = ["Benign", "Ransomware", "Spyware", "Trojan"]

# Pipeline 1 predictions & probabilities (for XGBoost Unified and Best-per-Scenario Ensemble)
p1_kb1_prob = flat_kb1_xgb.predict_proba(X200_te)
p1_kb1_idx = np.argmax(p1_kb1_prob, axis=1)
p1_kb1_pred = [lbl1[int(i)] for i in p1_kb1_idx]
p1_kb1_conf = [round(float(p1_kb1_prob[k, idx]) * 100, 1) for k, idx in enumerate(p1_kb1_idx)]

p1_kb2_prob = flat_kb2_rf.predict_proba(X200_te)
p1_kb2_idx = np.argmax(p1_kb2_prob, axis=1)
p1_kb2_pred = [lbl2[int(i)] for i in p1_kb2_idx]
p1_kb2_conf = [round(float(p1_kb2_prob[k, idx]) * 100, 1) for k, idx in enumerate(p1_kb2_idx)]

# Also compute single-algorithm Flat XGBoost KB2 & KB3 so we can show how Flat can have cross-scenario conflicts!
p1_kb2_xgb_prob = flat_kb2_xgb.predict_proba(X200_te)
p1_kb2_xgb_idx = np.argmax(p1_kb2_xgb_prob, axis=1)
p1_kb2_xgb_pred = [lbl2[int(i)] for i in p1_kb2_xgb_idx]
p1_kb2_xgb_conf = [round(float(p1_kb2_xgb_prob[k, idx]) * 100, 1) for k, idx in enumerate(p1_kb2_xgb_idx)]

p1_kb3_xgb_prob = flat_kb3_xgb.predict_proba(X200_te)
p1_kb3_xgb_idx = np.argmax(p1_kb3_xgb_prob, axis=1)
p1_kb3_xgb_pred = [all_cls3[int(i)] for i in p1_kb3_xgb_idx]
p1_kb3_xgb_conf = [round(float(p1_kb3_xgb_prob[k, idx]) * 100, 1) for k, idx in enumerate(p1_kb3_xgb_idx)]

# Pipeline 2: 3-Stage Hierarchical Cascade (Unified XGBoost & Ensemble XGB+RF)
pipe_u_xgb = joblib.load("Models_Hierarchical_Pipeline/unified_XGBoost.pkl")
pipe_u_ens = joblib.load("Models_Hierarchical_Pipeline/unified_Ensemble_XGB_RF.pkl")


def get_pipe_details(pipe_obj, X_df):
    results = []
    for idx in range(len(X_df)):
        row_x = X_df.iloc[[idx]]
        # Stage 1
        b1 = pipe_obj.stage1_
        prob1 = b1["clf"].predict_proba(row_x)[0]
        i1 = int(np.argmax(prob1))
        s1_lbl = b1["inv"][i1]
        s1_conf = round(float(prob1[i1]) * 100, 1)

        if s1_lbl == "Benign":
            results.append({
                "s1_pred": "Benign", "s1_conf": s1_conf,
                "s2_pred": "Benign (Dừng ở Tầng 1)", "s2_conf": s1_conf,
                "s3_pred": "Benign (Dừng ở Tầng 1)", "s3_conf": s1_conf,
                "s2_clean": "Benign", "s3_clean": "Benign",
                "stopped_at_s1": True,
                "specialist_used": "Không cần (Đã lọc ở Tầng 1)",
            })
        else:
            b2 = pipe_obj.stage2_
            prob2 = b2["clf"].predict_proba(row_x)[0]
            i2 = int(np.argmax(prob2))
            s2_lbl = b2["inv"][i2]
            s2_conf = round(float(prob2[i2]) * 100, 1)

            b3 = pipe_obj.stage3_[s2_lbl]
            prob3 = b3["clf"].predict_proba(row_x)[0]
            i3 = int(np.argmax(prob3))
            s3_lbl = b3["inv"][i3]
            s3_conf = round(float(prob3[i3]) * 100, 1)

            n_fams = len(b3["classes"])
            results.append({
                "s1_pred": "Malware", "s1_conf": s1_conf,
                "s2_pred": s2_lbl, "s2_conf": s2_conf,
                "s3_pred": s3_lbl, "s3_conf": s3_conf,
                "s2_clean": s2_lbl, "s3_clean": s3_lbl,
                "stopped_at_s1": False,
                "specialist_used": f"Chuyên gia {s2_lbl} ({n_fams} họ)",
            })
    return results


pipe_xgb_details = get_pipe_details(pipe_u_xgb, X200_te)
pipe_ens_details = get_pipe_details(pipe_u_ens, X200_te)

y_true_1 = get_labels(df200_meta, 1).values
y_true_2 = get_labels(df200_meta, 2).values
y_true_3 = get_labels(df200_meta, 3).values

samples_json = []
for i in range(len(df200_meta)):
    r_raw = X200_raw.iloc[i]
    r_z = X200_te.iloc[i]
    px = pipe_xgb_details[i]
    pe = pipe_ens_details[i]

    # Check if Flat models have any internal contradiction (e.g. KB2 group != prefix of KB3 family)
    f_kb1 = p1_kb1_pred[i]
    f_kb2 = p1_kb2_xgb_pred[i]
    f_kb3 = p1_kb3_xgb_pred[i]
    f_kb3_grp = f_kb3.split("-")[0]
    f_kb1_from_kb2 = "Benign" if f_kb2 == "Benign" else "Malware"
    flat_consistent = (f_kb1 == f_kb1_from_kb2) and (f_kb2 == f_kb3_grp)

    samples_json.append({
        "id": i + 1,
        "filename": str(df200_meta["Filename"].iloc[i]),
        "arch": "x64" if i >= 100 else "x32",
        "arch_label": str(df200_meta["Arch"].iloc[i]),
        "true_kb1": str(y_true_1[i]),
        "true_kb2": str(y_true_2[i]),
        "true_kb3": str(y_true_3[i]),
        "features": {
            "nproc": int(r_raw["pslist.nproc"]),
            "nprocs64bit": int(r_raw["pslist.nprocs64bit"]),
            "ndlls": int(r_raw["dlllist.ndlls"]),
            "nhandles": int(r_raw["handles.nhandles"]),
            "nmutant": int(r_raw["handles.nmutant"]),
            "not_in_load": int(r_raw["ldrmodules.not_in_load"]),
            "ninjections": int(r_raw["malfind.ninjections"]),
            "uniqueInjections": round(float(r_raw["malfind.uniqueInjections"]), 2),
            "nservices": int(r_raw["svcscan.nservices"]),
            "ncallbacks": int(r_raw["callbacks.ncallbacks"]),
            "z_not_in_load": round(float(r_z["ldrmodules.not_in_load"]), 2),
            "z_ninjections": round(float(r_z["malfind.ninjections"]), 2),
            "z_ratio_file": round(float(r_z["ratio_file_nhandles"]), 2),
            "z_ratio_ldr": round(float(r_z["ratio_ldr_load_dlls"]), 2),
        },
        "flat": {
            "kb1_pred": f_kb1, "kb1_conf": p1_kb1_conf[i],
            "kb2_pred": f_kb2, "kb2_conf": p1_kb2_xgb_conf[i],
            "kb2_rf_pred": p1_kb2_pred[i], "kb2_rf_conf": p1_kb2_conf[i],
            "kb3_pred": f_kb3, "kb3_conf": p1_kb3_xgb_conf[i],
            "consistent": bool(flat_consistent),
        },
        "pipe3": px,
        "pipe3_ens": pe,
    })

with open("/tmp/demo_200_samples.json", "w", encoding="utf-8") as f:
    json.dump(samples_json, f, ensure_ascii=False)

n_conflicts = sum(1 for s in samples_json if not s["flat"]["consistent"])
print(f"✅ Đã xuất /tmp/demo_200_samples.json ({len(samples_json)} mẫu | Số mẫu Flat bị mâu thuẫn nhãn giữa 3 kịch bản: {n_conflicts}/200 mẫu)")
