"""
Kịch bản vẽ bộ 8 biểu đồ Phân tích Khám phá Dữ liệu (EDA) & Đánh giá Thực nghiệm
cho toàn bộ dữ liệu đa kiến trúc (x32 MalMem2022 + x64 Build 19041 + 1.000 mẫu cWGAN-GP + 70 Đặc trưng + Pipeline 3 Tầng).
Lưu ảnh độ phân giải cao (300 DPI) vào:
  - eda_figures/DualArch_GAN/
"""
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

from train_and_eval_dual_arch import build_70_features, get_labels

OUT_DIR = "eda_figures/DualArch_GAN"
ARTIFACT_DIR = "/Users/minhtri86/.gemini/antigravity/brain/06867a2e-b8f9-49bc-8bc8-06823bfb357c"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", font="DejaVu Sans")

print("=" * 80)
print("🎨 BẮT ĐẦU VẼ BỘ 8 BIỂU ĐỒ EDA & KẾT QUẢ THỰC NGHIỆM ĐA KIẾN TRÚC (x32 + x64 + GAN)")
print("=" * 80)

# Load datasets
x32_tr = pd.read_csv("x32_train_clean.csv")
x32_te = pd.read_csv("x32_test_100.csv")
x64_tr = pd.read_csv("x64_train_100.csv")
x64_te = pd.read_csv("x64_test_100.csv")
x64_gan = pd.read_csv("x64_gan_1000_full.csv")
metrics_all = pd.read_csv("Metrics_DualArch_All_Comparisons.csv")
metrics_pipe = pd.read_csv("Metrics_Hierarchical_Pipeline_3Stages.csv")
cond_pipe = pd.read_csv("Metrics_Stage_Conditional_Analysis.csv")
preds_200 = pd.read_csv("Predictions_Test200_DualArch.csv")

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

X32_tr_s = X32_tr_raw.copy(); X32_tr_s[num_cols] = scaler32.transform(X32_tr_raw[num_cols])
X64_real_s = X64_real_raw.copy(); X64_real_s[num_cols] = scaler64.transform(X64_real_raw[num_cols])
X64_gan_s = X64_gan_raw.copy(); X64_gan_s[num_cols] = scaler64.transform(X64_gan_raw[num_cols])
X64_te_s = X64_te_raw.copy(); X64_te_s[num_cols] = scaler64.transform(X64_te_raw[num_cols])


def save_fig(fig, filename):
    p1 = os.path.join(OUT_DIR, filename)
    fig.savefig(p1, dpi=250, bbox_inches="tight", facecolor="white")
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy(p1, os.path.join(ARTIFACT_DIR, filename))
    plt.close(fig)
    print(f"✅ Đã xuất biểu đồ: {p1}")


# =========================================================================
# HÌNH 1: TỔNG QUAN CẤU TRÚC DỮ LIỆU TRAIN / GAN / TEST ĐA KIẾN TRÚC
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1.1, 1.4, 1.1]})

# Panel 1: Cơ cấu Benign vs Malware trên x64 trước và sau GAN
ax = axes[0]
categories_p1 = ["x64 Train Thật\n(100 mẫu)", "x64 Sinh từ GAN\n(1.000 mẫu)", "x64 Train Gộp\n(1.100 mẫu)"]
benign_counts = [15, 184, 199]
malware_counts = [85, 816, 901]
x_p1 = np.arange(len(categories_p1))
b1 = ax.bar(x_p1 - 0.18, benign_counts, width=0.35, label="Benign (Mẫu sạch)", color="#2ecc71", edgecolor="black", alpha=0.9)
b2 = ax.bar(x_p1 + 0.18, malware_counts, width=0.35, label="Malware (Mã độc)", color="#e74c3c", edgecolor="black", alpha=0.9)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{int(h)}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_xticks(x_p1)
ax.set_xticklabels(categories_p1, fontweight="bold")
ax.set_ylabel("Số lượng mẫu", fontweight="bold")
ax.set_ylim(0, 1050)
ax.set_title("(A) Mở rộng dữ liệu x64 bằng mạng cWGAN-GP\n(Tăng mẫu Benign x64 gấp 13,3 lần: 15 -> 199 mẫu)", fontweight="bold", fontsize=12)
ax.legend(frameon=True)

# Panel 2: Phân bố 11 Họ trên tập x64 (Train Real vs GAN vs Test)
ax = axes[1]
fam_tr = x64_tr["Category"].str.split("-").str[:2].str.join("-").value_counts()
fam_gan = x64_gan["Category"].str.split("-").str[:2].str.join("-").value_counts()
fam_te = x64_te["Category"].str.split("-").str[:2].str.join("-").value_counts()
fams = sorted(fam_tr.index)
y_pos = np.arange(len(fams))
ax.barh(y_pos - 0.25, [fam_tr[f] for f in fams], height=0.25, label="x64 Train Thật (100)", color="#3498db")
ax.barh(y_pos, [fam_te[f] for f in fams], height=0.25, label="x64 Test Độc lập (100)", color="#f39c12")
ax.barh(y_pos + 0.25, [fam_gan[f] / 5.0 for f in fams], height=0.25, label="x64 GAN (÷5 để cùng thang đo)", color="#9b59b6")
ax.set_yticks(y_pos)
ax.set_yticklabels(fams, fontsize=9.5, fontweight="bold")
ax.set_xlabel("Số mẫu (Riêng cột GAN chia 5)", fontweight="bold")
ax.set_title("(B) Phân bố 11 Họ trên Windows 10 x64\n(8 họ mới Zero-day + Conti, Pysa, Benign)", fontweight="bold", fontsize=12)
ax.legend(loc="lower right", frameon=True, fontsize=9)

# Panel 3: Cấu trúc Bộ Test Chuẩn Kép 200 mẫu (100 x32 + 100 x64)
ax = axes[2]
grp_order = ["Benign", "Ransomware", "Spyware", "Trojan"]
g32 = x32_te["Category"].str.split("-").str[0].value_counts().reindex(grp_order, fill_value=0)
g64 = x64_te["Category"].str.split("-").str[0].value_counts().reindex(grp_order, fill_value=0)
xg = np.arange(len(grp_order))
p_32 = ax.bar(xg - 0.18, g32.values, width=0.35, label="Test x32 (100 mẫu)", color="#1f77b4", edgecolor="black")
p_64 = ax.bar(xg + 0.18, g64.values, width=0.35, label="Test x64 (100 mẫu)", color="#ff7f0e", edgecolor="black")
for bar in list(p_32) + list(p_64):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{int(h)}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_xticks(xg)
ax.set_xticklabels(grp_order, fontweight="bold")
ax.set_ylim(0, 60)
ax.set_ylabel("Số mẫu kiểm thử", fontweight="bold")
ax.set_title("(C) Cấu trúc Bộ Test Chuẩn Kép (200 mẫu)\n(100 mẫu x32 + 100 mẫu x64 khóa cố định)", fontweight="bold", fontsize=12)
ax.legend(frameon=True)

fig.suptitle("TỔNG QUAN PHÂN BỐ DỮ LIỆU ĐA KIẾN TRÚC (x32 MALMEM2022 + x64 BUILD 19041 + cWGAN-GP)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save_fig(fig, "01_dataset_composition_x32_x64_gan.png")


# =========================================================================
# HÌNH 2: HIỆN TƯỢNG LỆCH MIỀN (DOMAIN SHIFT) x32 vs x64 & ĐỒNG QUY Z-SCORE
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

sample_x32 = x32_tr.sample(n=1000, random_state=42)
df_shift_raw = pd.concat([
    sample_x32[["ldrmodules.not_in_load", "dlllist.ndlls", "handles.nhandles", "Class"]].assign(Domain="x32 (Win7)"),
    x64_tr[["ldrmodules.not_in_load", "dlllist.ndlls", "handles.nhandles", "Class"]].assign(Domain="x64 (Win10)"),
], ignore_index=True)
df_shift_raw["Group_Label"] = df_shift_raw["Domain"] + " - " + df_shift_raw["Class"]

palette_shift = {
    "x32 (Win7) - Benign": "#2ecc71",
    "x32 (Win7) - Malware": "#e74c3c",
    "x64 (Win10) - Benign": "#27ae60",
    "x64 (Win10) - Malware": "#c0392b",
}

raw_compare_cols = [
    ("ldrmodules.not_in_load", "Số DLL ẩn khỏi InLoadOrder (ldrmodules.not_in_load)"),
    ("dlllist.ndlls", "Tổng số DLL toàn hệ thống (dlllist.ndlls)"),
    ("handles.nhandles", "Tổng số Handles mở (handles.nhandles)"),
]

for idx, (col, title_col) in enumerate(raw_compare_cols):
    ax = axes[0, idx]
    sns.boxplot(
        data=df_shift_raw, x="Domain", y=col, hue="Class",
        palette={"Benign": "#2ecc71", "Malware": "#e74c3c"}, ax=ax, showfliers=False, width=0.55
    )
    ax.set_title(f"[TRƯỚC ĐỒNG QUY - Thang đo gốc]\n{title_col}", fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Giá trị tuyệt đối (Raw Value)", fontweight="bold")
    ax.legend(title="Nhãn", loc="upper left")

# Hàng dưới: Sau khi tính Tỷ lệ & Chuẩn hóa Z-Score theo Kiến trúc
s32_sub = X32_tr_s.loc[sample_x32.index, ["ldrmodules.not_in_load", "ratio_ldr_load_dlls", "ratio_file_nhandles"]].copy()
s32_sub["Class"] = sample_x32["Class"].values
s32_sub["Domain"] = "x32 (Win7)"

s64_sub = X64_real_s[["ldrmodules.not_in_load", "ratio_ldr_load_dlls", "ratio_file_nhandles"]].copy()
s64_sub["Class"] = x64_tr["Class"].values
s64_sub["Domain"] = "x64 (Win10)"

df_shift_aligned = pd.concat([s32_sub, s64_sub], ignore_index=True)

aligned_compare_cols = [
    ("ldrmodules.not_in_load", "Z-Score: ldrmodules.not_in_load (Đã đồng quy miền)"),
    ("ratio_ldr_load_dlls", "Z-Score Tỷ lệ: not_in_load / dlllist.ndlls"),
    ("ratio_file_nhandles", "Z-Score Tỷ lệ: handles.nfile / handles.nhandles"),
]

for idx, (col, title_col) in enumerate(aligned_compare_cols):
    ax = axes[1, idx]
    sns.boxplot(
        data=df_shift_aligned, x="Domain", y=col, hue="Class",
        palette={"Benign": "#2ecc71", "Malware": "#e74c3c"}, ax=ax, showfliers=False, width=0.55
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"[SAU ĐỒNG QUY - Tỷ lệ + Z-Score Kiến trúc]\n{title_col}", fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Độ lệch chuẩn (Z-Score)", fontweight="bold")
    ax.legend(title="Nhãn", loc="upper right")

fig.suptitle(
    "PHÂN TÍCH HIỆN TƯỢNG LỆCH MIỀN (DOMAIN SHIFT) GIỮA WINDOWS 7 x32 VÀ WINDOWS 10 x64\n"
    "Hàng trên: Giá trị gốc bị lệch gấp 5–8 lần khiến mô hình cũ sụp đổ | Hàng dưới: Đưa về cùng mặt bằng thống kê sau khi chuẩn hóa",
    fontsize=13.5, fontweight="bold", y=1.03
)
plt.tight_layout()
save_fig(fig, "02_domain_shift_and_zscore_alignment.png")


# =========================================================================
# HÌNH 3: KIỂM ĐỊNH CHẤT LƯỢNG MẠNG SINH cWGAN-GP (PCA/t-SNE + KS + DCR)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

const_cols = [c for c in avail_cols if x64_tr[c].nunique() == 1]
dup_of = {}
for i, c in enumerate(avail_cols):
    if c in const_cols or c in dup_of: continue
    for c2 in avail_cols[i + 1:]:
        if c2 not in const_cols and c2 not in dup_of and x64_tr[c].equals(x64_tr[c2]):
            dup_of[c2] = c
gen_cols = [c for c in avail_cols if c not in const_cols and c not in dup_of]

X_real_g = np.log1p(np.maximum(0, x64_tr[gen_cols].values.astype(float)))
X_syn_g = np.log1p(np.maximum(0, x64_gan[gen_cols].values.astype(float)))
sc_g = StandardScaler().fit(X_real_g)
X_comb_g = np.vstack([sc_g.transform(X_syn_g), sc_g.transform(X_real_g)])

pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_comb_g)
n_syn = len(X_syn_g)

grp_syn = x64_gan["Category"].str.split("-").str[0].values
grp_real = x64_tr["Category"].str.split("-").str[0].values
grp_colors = {"Benign": "#2ecc71", "Ransomware": "#e74c3c", "Spyware": "#9b59b6", "Trojan": "#3498db"}

# Panel 1: PCA 2D Real vs GAN
ax = axes[0]
for grp, col_hex in grp_colors.items():
    m_s = grp_syn == grp
    m_r = grp_real == grp
    ax.scatter(pca_2d[:n_syn][m_s, 0], pca_2d[:n_syn][m_s, 1], c=col_hex, alpha=0.28, s=28, label=f"GAN - {grp}")
    ax.scatter(pca_2d[n_syn:][m_r, 0], pca_2d[n_syn:][m_r, 1], c=col_hex, edgecolors="black", linewidth=1.1, marker="*", s=150, label=f"Real - {grp}")
ax.set_xlabel("Thành phần chính 1 (PC1)", fontweight="bold")
ax.set_ylabel("Thành phần chính 2 (PC2)", fontweight="bold")
ax.set_title("(A) Không gian PCA 2D (41 chiều biến thiên)\nNgôi sao (*): 100 mẫu thật | Chấm tròn: 1.000 mẫu GAN", fontweight="bold", fontsize=11.5)
ax.legend(ncol=2, fontsize=8.2, frameon=True, loc="best")

# Panel 2: Phân phối khoảng cách tới mẫu thật gần nhất (DCR)
ax = axes[1]
nn = NearestNeighbors(n_neighbors=2).fit(sc_g.transform(X_real_g))
dcr_syn = nn.kneighbors(sc_g.transform(X_syn_g), n_neighbors=1)[0].ravel()
dcr_real = nn.kneighbors(sc_g.transform(X_real_g), n_neighbors=2)[0][:, 1]
sns.histplot(dcr_syn, bins=30, kde=True, color="#8e44ad", alpha=0.55, label=f"DCR (GAN -> Real) | Trung vị = {np.median(dcr_syn):.2f}", ax=ax)
ax.axvline(np.median(dcr_real), color="#d35400", linestyle="--", linewidth=2, label=f"Khoảng cách nội bộ Real-Real = {np.median(dcr_real):.2f}")
ax.axvline(0.0, color="red", linestyle=":", linewidth=1.8, label="Ngưỡng sao chép (Exact Copy = 0 mẫu)")
ax.set_xlabel("Khoảng cách Euclid chuẩn hóa tới mẫu thật gần nhất", fontweight="bold")
ax.set_ylabel("Tần suất (Số mẫu GAN)", fontweight="bold")
ax.set_title("(B) Kiểm định chống sao chép (DCR Histogram)\n1.000/1.000 mẫu duy nhất, 0 mẫu sao chép y hệt", fontweight="bold", fontsize=11.5)
ax.legend(frameon=True, fontsize=9)

# Panel 3: Phân phối thống kê KS trên 41 đặc trưng biến thiên
ax = axes[2]
ks_stats = pd.Series({c: ks_2samp(x64_tr[c], x64_gan[c]).statistic for c in gen_cols}).sort_values()
top_best_ks = ks_stats.head(15)
ax.barh(range(len(top_best_ks)), top_best_ks.values, color="#16a085", edgecolor="black", alpha=0.85)
ax.axvline(ks_stats.mean(), color="#c0392b", linestyle="--", linewidth=2, label=f"Trung bình toàn bộ 41 cột: KS = {ks_stats.mean():.3f}")
ax.set_yticks(range(len(top_best_ks)))
ax.set_yticklabels(top_best_ks.index, fontsize=8.8, fontweight="bold")
ax.set_xlabel("Thống kê KS (Càng nhỏ càng sát phân phối thật)", fontweight="bold")
ax.set_title("(C) Độ khớp phân phối biên (KS Statistic)\nTop 15 đặc trưng có phân phối GAN sát mẫu thật nhất", fontweight="bold", fontsize=11.5)
ax.legend(loc="lower right", frameon=True, fontsize=9)

fig.suptitle("ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU TỔNG HỢP TỪ MẠNG PROTOTYPE-ANCHORED cWGAN-GP (1.000 MẪU x64)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save_fig(fig, "03_gan_quality_pca_tsne_and_ks.png")


# =========================================================================
# HÌNH 4: PHÂN BỐ CÁC NHÓM ĐẶC TRƯNG & TOP 20 ĐẶC TRƯNG QUAN TRỌNG NHẤT
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), gridspec_kw={"width_ratios": [1.0, 1.35]})

# Panel 1: Số lượng đặc trưng theo từng nhóm trong bộ 70 đặc trưng
ax = axes[0]
grp_counts = [
    ("handles (Gốc)", 13, "#2980b9"),
    ("handles (Tỷ lệ mới)", 10, "#f39c12"),
    ("psxview (Gốc khả dụng)", 8, "#2980b9"),
    ("svcscan (Gốc)", 7, "#2980b9"),
    ("ldrmodules (Gốc)", 6, "#2980b9"),
    ("pslist (Gốc)", 5, "#2980b9"),
    ("malfind (Gốc)", 4, "#2980b9"),
    ("callbacks (Gốc)", 3, "#2980b9"),
    ("malfind (Tỷ lệ mới)", 3, "#f39c12"),
    ("svcscan (Tỷ lệ mới)", 3, "#f39c12"),
    ("dlllist (Gốc)", 2, "#2980b9"),
    ("pslist (Tỷ lệ mới)", 2, "#f39c12"),
    ("ldrmodules (Tỷ lệ mới)", 2, "#f39c12"),
    ("modules (Gốc)", 1, "#2980b9"),
    ("is_x64 (Cờ kiến trúc)", 1, "#27ae60"),
]
grp_counts = grp_counts[::-1]
y_g = np.arange(len(grp_counts))
bars_g = ax.barh(y_g, [x[1] for x in grp_counts], color=[x[2] for x in grp_counts], edgecolor="black", alpha=0.9)
for b in bars_g:
    w = b.get_width()
    ax.text(w + 0.2, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center", fontweight="bold", fontsize=9.5)
ax.set_yticks(y_g)
ax.set_yticklabels([x[0] for x in grp_counts], fontweight="bold", fontsize=9.5)
ax.set_xlim(0, 15)
ax.set_xlabel("Số lượng đặc trưng", fontweight="bold")
ax.set_title("(A) Cơ cấu 70 Đặc trưng đầu vào theo Nhóm\n(Xanh dương: 49 Gốc | Cam: 20 Tỷ lệ hành vi | Xanh lá: 1 Cờ kiến trúc)", fontweight="bold", fontsize=11.5)

# Panel 2: Top 20 đặc trưng có độ quan trọng (Feature Importance) cao nhất
ax = axes[1]
rf_s1 = joblib.load("Models_DualArch_Unified/Scenario_1/Random_Forest.pkl")
rf_s2 = joblib.load("Models_DualArch_Unified/Scenario_2/Random_Forest.pkl")
rf_s3 = joblib.load("Models_DualArch_Unified/Scenario_3/Random_Forest.pkl")
feat_names = list(X32_tr_s.columns)

imp_df = pd.DataFrame({
    "Feature": feat_names,
    "KB1 (Nhị phân)": rf_s1.feature_importances_,
    "KB2 (4 Nhóm)": rf_s2.feature_importances_,
    "KB3 (24 Họ)": rf_s3.feature_importances_,
})
imp_df["Mean_Imp"] = imp_df[["KB1 (Nhị phân)", "KB2 (4 Nhóm)", "KB3 (24 Họ)"]].mean(axis=1)
top20 = imp_df.sort_values("Mean_Imp", ascending=True).tail(20)

y20 = np.arange(len(top20))
ax.barh(y20 - 0.25, top20["KB1 (Nhị phân)"], height=0.25, label="KB1 (Nhị phân)", color="#2ecc71")
ax.barh(y20, top20["KB2 (4 Nhóm)"], height=0.25, label="KB2 (4 Nhóm)", color="#3498db")
ax.barh(y20 + 0.25, top20["KB3 (24 Họ)"], height=0.25, label="KB3 (24 Họ)", color="#e74c3c")
ax.set_yticks(y20)
labels_20 = [f"★ {f}" if f.startswith("ratio_") else f for f in top20["Feature"]]
ax.set_yticklabels(labels_20, fontweight="bold", fontsize=9.5)
ax.set_xlabel("Độ quan trọng đặc trưng (Gini Importance)", fontweight="bold")
ax.set_title("(B) Top 20 Đặc trưng quan trọng nhất qua 3 Kịch bản\n(Ký hiệu ★ là các Đặc trưng Tỷ lệ mới chế tác đóng góp rất lớn vào KB2 & KB3)", fontweight="bold", fontsize=11.5)
ax.legend(loc="lower right", frameon=True)

fig.suptitle("PHÂN TÍCH NHÓM ĐẶC TRƯNG & MỨC ĐỘ ĐÓNG GÓP VÀO QUYẾT ĐỊNH CỦA MÔ HÌNH (70 ĐẶC TRƯNG)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "04_70features_group_breakdown_and_importance.png")


# =========================================================================
# HÌNH 5: BẢN ĐỒ NHIỆT DẤU VẾT HÀNH VI (BEHAVIORAL FINGERPRINT HEATMAP)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)
top16_feats = top20["Feature"].tail(16).tolist()[::-1]

X32_WithGrp = X32_tr_s[top16_feats].copy()
X32_WithGrp["Group"] = x32_tr["Category"].str.split("-").str[0].values
mean32 = X32_WithGrp.groupby("Group")[top16_feats].mean().T[["Benign", "Ransomware", "Spyware", "Trojan"]]

X64_WithGrp = X64_real_s[top16_feats].copy()
X64_WithGrp["Group"] = x64_tr["Category"].str.split("-").str[0].values
mean64 = X64_WithGrp.groupby("Group")[top16_feats].mean().T[["Benign", "Ransomware", "Spyware", "Trojan"]]

sns.heatmap(mean32, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5, ax=axes[0], cbar=False, linewidths=0.5)
axes[0].set_title("(A) Dấu vết hành vi (Z-Score trung bình) trên Windows 7 x32\n(Tập huấn luyện sạch: 57.908 mẫu)", fontweight="bold", fontsize=12)
axes[0].set_xlabel("Nhóm hành vi", fontweight="bold")
axes[0].set_ylabel("Top 16 Đặc trưng chủ chốt", fontweight="bold")

sns.heatmap(mean64, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5, ax=axes[1], cbar=True, linewidths=0.5)
axes[1].set_title("(B) Dấu vết hành vi (Z-Score trung bình) trên Windows 10 x64\n(Sau khi chuẩn hóa đồng quy kiến trúc)", fontweight="bold", fontsize=12)
axes[1].set_xlabel("Nhóm hành vi", fontweight="bold")
axes[1].set_ylabel("")

fig.suptitle("BẢN ĐỒ NHIỆT DẤU VẾT HÀNH VI (BEHAVIORAL FINGERPRINT) CỦA 4 NHÓM TRÊN x32 VÀ x64", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "05_behavioral_fingerprint_heatmap_x32_x64.png")


# =========================================================================
# HÌNH 6: BIỂU ĐỒ CỘT SO SÁNH 4 PHƯƠNG PHÁP (E0 -> E1 -> E2 -> E3) QUA 3 KỊCH BẢN
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), sharey=True)
method_order = [
    "E0: Baseline (22f Old)",
    "E1: Old+GAN (22f)",
    "E2: Unified (Clean+FullGAN 70f)",
    "E3: Arch-Adaptive (Clean+FullGAN 70f)",
]
method_short = ["E0: Baseline\n(22f cũ)", "E1: Old+GAN\n(22f)", "E2: Unified\n(70f + FullGAN)", "E3: Arch-Adaptive\n(70f + FullGAN)"]
model_order = ["XGBoost", "Random_Forest", "J_48", "Random_Tree", "Naive_Bayes"]
m_colors = ["#95a5a6", "#f39c12", "#2ecc71", "#2980b9"]

for idx, (sc_code, sc_title) in enumerate([
    ("KB1", "Kịch bản 1: Nhị phân (2 nhãn)"),
    ("KB2", "Kịch bản 2: Đa lớp (4 nhóm)"),
    ("KB3", "Kịch bản 3: Chi tiết (24 họ)"),
]):
    ax = axes[idx]
    sub = metrics_all[(metrics_all["Scenario"] == sc_code) & (metrics_all["Test_Subset"] == "Combined (200)")]
    piv = sub.pivot(index="Model", columns="Method", values="Accuracy").reindex(model_order)[method_order]
    x = np.arange(len(model_order))
    w = 0.2
    for m_i, m_col in enumerate(method_order):
        vals = piv[m_col].values
        bars = ax.bar(x + (m_i - 1.5) * w, vals, width=w, label=method_short[m_i], color=m_colors[m_i], edgecolor="black")
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.0f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["XGBoost", "Random\nForest", "J-48\n(C4.5)", "Random\nTree", "Naive\nBayes"], fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_title(sc_title, fontweight="bold", fontsize=12.5)
    if idx == 0:
        ax.set_ylabel("Accuracy (%) trên Bộ Test Kép 200 mẫu (100 x32 + 100 x64)", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=8.8)

fig.suptitle("SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY %) TRÊN 200 MẪU TEST (100 x32 + 100 x64) QUA 4 PHƯƠNG PHÁP", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save_fig(fig, "06_methods_progression_bar_comparison.png")


# =========================================================================
# HÌNH 7: MA TRẬN NHẦM LẪN (CONFUSION MATRICES) ĐỐI CHIẾU TRÊN x64 & 200 MẪU
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

# Panel 1: KB1 trên x64 - E1 (GAN 22f cũ)
bmap1 = joblib.load("Models_GAN/Old_GAN/Scenario_1/XGBoost.pkl")
from train_and_evaluate_22features import FEATURES_22
p_old_64 = ["Malware" if i == 1 else "Benign" for i in bmap1.predict(x64_te[FEATURES_22].fillna(0))]
cm_old = confusion_matrix(x64_te["Class"], p_old_64, labels=["Benign", "Malware"])
sns.heatmap(cm_old, annot=True, fmt="d", cmap="Oranges", xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"], ax=axes[0], cbar=False, annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title("(A) KB1 trên x64: GAN 22f cũ (XGBoost)\nAcc: 86% nhưng đoán nhầm 13/15 Benign (BalAcc: 56,1%)", fontweight="bold", fontsize=11)
axes[0].set_xlabel("Nhãn dự đoán", fontweight="bold")
axes[0].set_ylabel("Nhãn thực tế", fontweight="bold")

# Panel 2: KB1 trên x64 - E2 Unified (70f + FullGAN)
u_xgb1 = joblib.load("Models_DualArch_Unified/Scenario_1/XGBoost.pkl")
p_new_64 = ["Malware" if i == 1 else "Benign" for i in u_xgb1.predict(X64_te_s)]
cm_new = confusion_matrix(x64_te["Class"], p_new_64, labels=["Benign", "Malware"])
sns.heatmap(cm_new, annot=True, fmt="d", cmap="Greens", xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"], ax=axes[1], cbar=False, annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title("(B) KB1 trên x64: Unified 70f + FullGAN (XGBoost)\nAcc: 94,0% | Nhận đúng 10/15 Benign & 84/85 Malware (BalAcc: 82,8%)", fontweight="bold", fontsize=11)
axes[1].set_xlabel("Nhãn dự đoán", fontweight="bold")
axes[1].set_ylabel("Nhãn thực tế", fontweight="bold")

# Panel 3: KB2 trên 200 mẫu (100 x32 + 100 x64) - E3 Arch-Adaptive XGBoost
y200_kb2 = np.concatenate([get_labels(x32_te, 2).values, get_labels(x64_te, 2).values])
p200_kb2 = preds_200["KB2_XGBoost_Pred"].values
grp_lbls = ["Benign", "Ransomware", "Spyware", "Trojan"]
cm_kb2 = confusion_matrix(y200_kb2, p200_kb2, labels=grp_lbls)
sns.heatmap(cm_kb2, annot=True, fmt="d", cmap="Blues", xticklabels=grp_lbls, yticklabels=grp_lbls, ax=axes[2], cbar=False, annot_kws={"size": 12, "weight": "bold"})
axes[2].set_title("(C) KB2 trên 200 mẫu (100 x32 + 100 x64)\nXGBoost 70f + FullGAN (Acc: 81,0% | Macro-F1: 79,5%)", fontweight="bold", fontsize=11)
axes[2].set_xlabel("Nhãn dự đoán", fontweight="bold")
axes[2].set_ylabel("Nhãn thực tế", fontweight="bold")

fig.suptitle("MA TRẬN NHẦM LẪN (CONFUSION MATRIX): KHẮC PHỤC HIỆN TƯỢNG ĐOÁN NHẦM BENIGN x64 & PHÂN LOẠI 4 NHÓM", fontsize=13.5, fontweight="bold", y=1.03)
plt.tight_layout()
save_fig(fig, "07_confusion_matrices_kb1_kb2.png")


# =========================================================================
# HÌNH 8: SO SÁNH PIPELINE ĐƠN LẺ (FLAT) vs. PIPELINE PHÂN CẤP 3 TẦNG
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(19, 6.2))

# Panel 1: So sánh Flat Unified vs Pipeline 3 Tầng Unified trên KB2 và KB3
ax = axes[0]
algos_p8 = ["XGBoost", "Random_Forest", "J_48", "Random_Tree"]
flat_kb2 = [metrics_all[(metrics_all["Method"] == "E2: Unified (Clean+FullGAN 70f)") & (metrics_all["Scenario"] == "KB2") & (metrics_all["Model"] == m) & (metrics_all["Test_Subset"] == "x64 (100)")]["Accuracy"].values[0] for m in algos_p8]
pipe_kb2 = [metrics_pipe[(metrics_pipe["Method"] == "Pipeline 3 Tầng (Unified 70f)") & (metrics_pipe["Scenario"] == "KB2") & (metrics_pipe["Model"] == m) & (metrics_pipe["Test_Subset"] == "x64 (100)")]["Accuracy"].values[0] for m in algos_p8]
flat_kb3 = [metrics_all[(metrics_all["Method"] == "E2: Unified (Clean+FullGAN 70f)") & (metrics_all["Scenario"] == "KB3") & (metrics_all["Model"] == m) & (metrics_all["Test_Subset"] == "Combined (200)")]["Accuracy"].values[0] for m in algos_p8]
pipe_kb3 = [metrics_pipe[(metrics_pipe["Method"] == "Pipeline 3 Tầng (Unified 70f)") & (metrics_pipe["Scenario"] == "KB3") & (metrics_pipe["Model"] == m) & (metrics_pipe["Test_Subset"] == "Combined (200)")]["Accuracy"].values[0] for m in algos_p8]

xp8 = np.arange(len(algos_p8))
w8 = 0.2
b_f2 = ax.bar(xp8 - 1.5 * w8, flat_kb2, width=w8, label="Flat KB2 (x64)", color="#85c1e9", edgecolor="black")
b_p2 = ax.bar(xp8 - 0.5 * w8, pipe_kb2, width=w8, label="Pipeline Tầng 2 (x64)", color="#2874a6", edgecolor="black")
b_f3 = ax.bar(xp8 + 0.5 * w8, flat_kb3, width=w8, label="Flat KB3 (200 mẫu)", color="#f5b7b1", edgecolor="black")
b_p3 = ax.bar(xp8 + 1.5 * w8, pipe_kb3, width=w8, label="Pipeline Tầng 3 (200 mẫu)", color="#cb4335", edgecolor="black")

for b in list(b_f2) + list(b_p2) + list(b_f3) + list(b_p3):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(xp8)
ax.set_xticklabels(["XGBoost", "Random Forest", "J-48 (C4.5)", "Random Tree"], fontweight="bold")
ax.set_ylim(0, 90)
ax.set_ylabel("Accuracy (%)", fontweight="bold")
ax.set_title("(A) Cải thiện của Pipeline Phân cấp 3 Tầng so với Mô hình Phẳng (Flat)\n(XGBoost Tầng 2 x64 tăng 68% -> 75% | Random Tree Tầng 3 tăng 56% -> 63%)", fontweight="bold", fontsize=11.5)
ax.legend(ncol=2, frameon=True, loc="upper right")

# Panel 2: Hiệu suất có điều kiện tại từng tầng (Stage-wise Conditional Accuracy)
ax = axes[1]
cond_u = cond_pipe[cond_pipe["Mode"] == "Pipeline 3 Tầng (Unified 70f)"].set_index("Model").reindex(["XGBoost", "Ensemble_XGB_RF", "Random_Forest", "J_48", "Random_Tree"])
xc = np.arange(len(cond_u))
wc = 0.22
c1 = ax.bar(xc - wc, cond_u["T1_Binary_Acc_200"], width=wc, label="Tầng 1: Phát hiện Nhị phân (2 nhãn)", color="#27ae60", edgecolor="black")
c2 = ax.bar(xc, cond_u["T2_Given_T1_Correct (%)"], width=wc, label="Tầng 2 | Khi Tầng 1 đúng (3 nhóm Malware)", color="#2980b9", edgecolor="black")
c3 = ax.bar(xc + wc, cond_u["T3_Given_T2_Correct (%)"], width=wc, label="Chuyên gia Tầng 3 | Khi Tầng 2 đúng nhóm", color="#8e44ad", edgecolor="black")

for b in list(c1) + list(c2) + list(c3):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(xc)
ax.set_xticklabels(["XGBoost", "Ensemble\n(XGB+RF)", "Random\nForest", "J-48\n(C4.5)", "Random\nTree"], fontweight="bold")
ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy tại từng tầng (%)", fontweight="bold")
ax.set_title("(B) Độ chính xác chuyên biệt của Chuyên gia từng Tầng (200 mẫu)\n(Độ nhất quán nhãn giữa Tầng 1 <-> Tầng 2 <-> Tầng 3 đạt tuyệt đối 100%)", fontweight="bold", fontsize=11.5)
ax.legend(frameon=True, loc="lower right", fontsize=9)

fig.suptitle("PHÂN TÍCH HIỆU QUẢ CỦA PIPELINE PHÂN CẤP 3 TẦNG (TẦNG 1: 2 NHÃN -> TẦNG 2: NHÓM -> TẦNG 3: CHUYÊN GIA HỌ)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save_fig(fig, "08_flat_vs_hierarchical_3stage_pipeline.png")

print("\n🎉 HOÀN TẤT VẼ 8 BIỂU ĐỒ EDA & KẾT QUẢ THỰC NGHIỆM!")
