"""
Tạo 2 bộ biểu đồ độ phân giải cao (300 DPI) tổng hợp kết quả của 2 Thực nghiệm Chống Rò Rỉ Dữ Liệu:
- Thực nghiệm 1: Chia Cô lập Tuyệt đối theo Mã Hash / Phiên chạy trên các bản dump (Leak-Free Snapshot Split - 0% Overlap)
- Thực nghiệm 2: Gộp 10 dòng snapshot thành 1 dòng duy nhất cho từng mẫu thực thi (10-to-1 Instance Aggregation - 1 mẫu = 1 dòng)
và Minh chứng quy trình Cô lập Tuyệt đối của mạng Leak-Free Prototype-Anchored cWGAN-GP.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "eda_figures/DualArch_GAN"
ARTIFACT_DIR = "/Users/minhtri86/.gemini/antigravity/brain/06867a2e-b8f9-49bc-8bc8-06823bfb357c"
os.makedirs(OUT_DIR, exist_ok=True)

df_lf = pd.read_csv("Metrics_LeakFree_Benchmark.csv")
df_agg = pd.read_csv("Metrics_Aggregated_10to1_Experiment.csv")
df_gkf = pd.read_csv("Metrics_x64_5Fold_GroupKFold.csv")

MODELS = ["XGBoost", "Random_Forest", "Ensemble_XGB_RF", "J_48", "Random_Tree", "Naive_Bayes"]
MODEL_LABELS = ["XGBoost", "Random\nForest", "Ensemble\n(XGB+RF)", "J48\n(C4.5)", "Random\nTree", "Naive\nBayes"]

# =============================================================================
# HÌNH 15: TỔNG HỢP 2 THỰC NGHIỆM CHỐNG RÒ RỈ (4 PANELS)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
fig.suptitle(
    "KẾT QUẢ 2 THỰC NGHIỆM CHỐNG RÒ RỈ DỮ LIỆU (LEAK-FREE BENCHMARKS)\n"
    "TN1: Chia Cô Lập Theo Mã Hash Trên Bản Dump (0% Overlap)  vs.  TN2: Gộp 10 Dòng Snapshot Thành 1 Cho Từng Mẫu (1 Mẫu = 1 Dòng)",
    fontsize=15, fontweight="bold", y=0.98, color="#1e293b"
)

colors_sc = {"KB1": "#2563eb", "KB2": "#f59e0b", "KB3": "#10b981"}
x = np.arange(len(MODELS))
w = 0.25

# --- PANEL A: Thực nghiệm 1 (Leak-Free Snapshot Split - 160 dòng Test cô lập: 80 x32 + 80 x64) ---
ax = axes[0, 0]
for idx, (sc, lbl) in enumerate([("KB1", "KB1: 2 Nhãn (Phát hiện)"), ("KB2", "KB2: 4 Nhóm hành vi"), ("KB3", "KB3: 24 Họ mã độc")]):
    sub = df_lf[(df_lf["Scenario"] == sc) & (df_lf["Test_Subset"] == "Combined LeakFree (160)")]
    vals = [float(sub[sub["Model"] == m]["Accuracy"].values[0]) for m in MODELS]
    bars = ax.bar(x + (idx - 1) * w, vals, width=w, label=lbl, color=colors_sc[sc], edgecolor="white", alpha=0.92)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_title("A. Thực nghiệm 1: Chia Cô Lập Mã Hash Trên Dòng Dump\n(Test = 160 Dòng Cô Lập: 80 x32 + 80 x64 từ các con mã độc chưa từng thấy)", fontsize=11.5, fontweight="bold", color="#0f172a")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 112)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# --- PANEL B: Thực nghiệm 2A (MalMem2022 Gộp 10->1: 5.836 mẫu độc lập vs. Chia theo dòng 58.596 dòng) ---
ax = axes[0, 1]
orig_x32 = {
    "KB1": [100.0, 100.0, 100.0, 99.98, 99.95, 99.45],
    "KB2": [88.31, 87.97, 88.30, 86.00, 85.80, 56.12],
    "KB3": [76.72, 76.20, 76.80, 74.20, 73.10, 41.20],
}
agg_x32_kb2 = [float(df_agg[(df_agg["Experiment"].str.startswith("ExpA")) & (df_agg["Scenario"] == "KB2") & (df_agg["Model"] == m)]["Accuracy"].values[0]) for m in MODELS]
agg_x32_kb3 = [float(df_agg[(df_agg["Experiment"].str.startswith("ExpA")) & (df_agg["Scenario"] == "KB3") & (df_agg["Model"] == m)]["Accuracy"].values[0]) for m in MODELS]

b1 = ax.bar(x - 1.5 * 0.2, orig_x32["KB2"], width=0.2, label="KB2 - Chia theo dòng gốc (58.596 dòng)", color="#fcd34d", edgecolor="#b45309")
b2 = ax.bar(x - 0.5 * 0.2, agg_x32_kb2, width=0.2, label="KB2 - Gộp 10->1 (5.836 con độc lập)", color="#d97706", edgecolor="white")
b3 = ax.bar(x + 0.5 * 0.2, orig_x32["KB3"], width=0.2, label="KB3 - Chia theo dòng gốc (58.596 dòng)", color="#6ee7b7", edgecolor="#047857")
b4 = ax.bar(x + 1.5 * 0.2, agg_x32_kb3, width=0.2, label="KB3 - Gộp 10->1 (5.836 con độc lập)", color="#059669", edgecolor="white")

for bars in [b1, b2, b3, b4]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_title("B. Thực nghiệm 2A: Bộ MalMem2022 Gộp 10 Dòng Thành 1 (5.836 Con)\n(Đánh giá độ sụt giảm thực tế khi loại bỏ 100% bản dump anh em cách 15s)", fontsize=11.5, fontweight="bold", color="#0f172a")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 108)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

# --- PANEL C: Thực nghiệm 2B (Đa kiến trúc x32 + x64 + GAN sau khi Gộp 10->1: 90 mẫu Test độc lập = 80 con x32 + 10 con x64) ---
ax = axes[1, 0]
for idx, (sc, lbl) in enumerate([("KB1", "KB1: 2 Nhãn (Phát hiện)"), ("KB2", "KB2: 4 Nhóm hành vi"), ("KB3", "KB3: 24 Họ mã độc")]):
    sub = df_agg[(df_agg["Experiment"].str.startswith("ExpB")) & (df_agg["Scenario"] == sc) & (df_agg["Subset"] == "Combined (90 con)")]
    vals = [float(sub[sub["Model"] == m]["Accuracy"].values[0]) for m in MODELS]
    bars = ax.bar(x + (idx - 1) * w, vals, width=w, label=lbl, color=colors_sc[sc], edgecolor="white", alpha=0.92)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_title("C. Thực nghiệm 2B: Đa Kiến Trúc (x32 + x64 + GAN) Sau Khi Gộp 10 -> 1\n(Test = 90 Con Độc Lập: 80 Con x32 + 10 Con x64; 1 Con Mã Độc = 1 Dòng)", fontsize=11.5, fontweight="bold", color="#0f172a")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 112)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# --- PANEL D: Đối sánh Trực tiếp 3 Phương pháp Thực nghiệm (Mô hình Tốt nhất ở mỗi Kịch bản) ---
ax = axes[1, 1]
sc_labels = ["Kịch bản 1\n(2 Nhãn: Benign/Malware)", "Kịch bản 2\n(4 Nhóm Hành vi)", "Kịch bản 3\n(24 Họ Mã độc)"]
xs = np.arange(3)
p1_snap = [97.00, 81.50, 67.50]       # PP1: Chia theo dòng snapshot (chuẩn bài báo gốc)
p2_lf_dump = [89.38, 69.38, 46.25]    # PP2: Chia cô lập mã hash trên dòng dump (0% overlap)
p3_lf_agg = [100.00, 81.11, 65.56]    # PP3: Gộp 10 dòng thành 1 cho từng mẫu (0% overlap + khử nhiễu tạm thời)

w_p = 0.25
r1 = ax.bar(xs - w_p, p1_snap, width=w_p, label="PP1: Chia theo dòng Snapshot (Bài báo gốc - Có trùng phiên)", color="#64748b", edgecolor="white")
r2 = ax.bar(xs, p2_lf_dump, width=w_p, label="PP2 (TN1): Chia Cô lập Mã Hash trên Dòng Dump (0% rò rỉ)", color="#ef4444", edgecolor="white")
r3 = ax.bar(xs + w_p, p3_lf_agg, width=w_p, label="PP3 (TN2): Gộp 10 Dòng Thành 1 Cho Từng Mẫu (0% rò rỉ + Khử nhiễu)", color="#16a34a", edgecolor="white")

for bars in [r1, r2, r3]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Vẽ mũi tên thể hiện mức tăng nhờ Gộp 10->1 so với Dòng Dump đơn lẻ
for i in range(3):
    diff = p3_lf_agg[i] - p2_lf_dump[i]
    ax.annotate(
        f"+{diff:.2f}%\n(Khử nhiễu)",
        xy=(xs[i] + w_p, p3_lf_agg[i] + 1),
        xytext=(xs[i], p2_lf_dump[i] + 11),
        arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.8),
        fontsize=8.5, fontweight="bold", color="#15803d", ha="center"
    )

ax.set_title("D. Tổng Kết So Sánh 3 Phương Pháp Thực Nghiệm Đa Kiến Trúc\n(Gộp 10->1 vừa triệt tiêu 100% rò rỉ vừa khử nhiễu RAM giúp khôi phục độ chính xác)", fontsize=11.5, fontweight="bold", color="#0f172a")
ax.set_xticks(xs)
ax.set_xticklabels(sc_labels, fontsize=10, fontweight="bold")
ax.set_ylabel("Best Combined Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 122)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p15 = os.path.join(OUT_DIR, "15_leakfree_snapshot_vs_aggregated_10to1.png")
plt.savefig(p15, dpi=300)
shutil.copy(p15, os.path.join(ARTIFACT_DIR, "15_leakfree_snapshot_vs_aggregated_10to1.png"))
plt.close(fig)
print(f"✅ Đã lưu biểu đồ 15: {p15}")

# =============================================================================
# HÌNH 16: KIỂM CHỨNG QUY TRÌNH LEAK-FREE GAN & ĐỐI SÁNH RIÊNG TRÊN TẬP TEST x64
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), dpi=300)
fig.suptitle(
    "KIỂM CHỨNG QUY TRÌNH LEAK-FREE cWGAN-GP (CHỈ HỌC TỪ 10 CON TRAIN x64) & KẾT QUẢ RIÊNG TRÊN KIẾN TRÚC WINDOWS 10 x64",
    fontsize=14, fontweight="bold", y=0.98, color="#1e293b"
)

# --- PANEL A: Không gian PCA 2D chứng minh GAN chỉ phủ quanh 10 con Train x64 và tách biệt 10 con Test x64 ---
ax = axes[0]
tr64 = pd.read_csv("x64_train_leakfree.csv")
te64 = pd.read_csv("x64_test_leakfree.csv")
gan64 = pd.read_csv("x64_gan_1000_leakfree.csv")
feat_cols = [c for c in tr64.columns if c not in ["Class", "Category", "Filename", "Instance_ID", "Dump_Num"]]

sc_pca = StandardScaler().fit(np.log1p(np.maximum(0, tr64[feat_cols].values)))
X_tr_p = sc_pca.transform(np.log1p(np.maximum(0, tr64[feat_cols].values)))
X_te_p = sc_pca.transform(np.log1p(np.maximum(0, te64[feat_cols].values)))
X_gan_p = sc_pca.transform(np.log1p(np.maximum(0, gan64[feat_cols].values)))

pca = PCA(n_components=2, random_state=42).fit(X_tr_p)
p_tr = pca.transform(X_tr_p)
p_te = pca.transform(X_te_p)
p_gan = pca.transform(X_gan_p)

ax.scatter(p_gan[:, 0], p_gan[:, 1], c="#93c5fd", s=16, alpha=0.45, label="Leak-Free GAN (1.000 mẫu - Sinh từ 10 con Train)")
ax.scatter(p_tr[:, 0], p_tr[:, 1], c="#1d4ed8", s=55, marker="o", edgecolors="white", linewidth=0.8, label="x64 Train Thật (10 con = 80 dòng)")
ax.scatter(p_te[:, 0], p_te[:, 1], c="#dc2626", s=60, marker="^", edgecolors="white", linewidth=0.8, label="x64 Test Cô Lập (10 con khác = 80 dòng, 0% Overlap)")

ax.set_title("A. Không Gian PCA 2D Của Leak-Free cWGAN-GP\n(GAN chỉ neo trên 10 con Train x64, tuyệt đối không nhìn thấy 10 con Test)", fontsize=10.5, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% phương sai)", fontsize=9.5)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% phương sai)", fontsize=9.5)
ax.set_xlim(-8, 12)
ax.set_ylim(-6, 10)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

# --- PANEL B: So sánh Độ chính xác riêng trên Tập Test x64 Cô lập (80 Dòng Dump vs. 10 Con Gộp 10->1) ---
ax = axes[1]
top_models = ["XGBoost", "Random_Forest", "Ensemble_XGB_RF", "J_48", "Random_Tree"]
top_labels = ["XGBoost", "Random\nForest", "Ensemble\n(XGB+RF)", "J48", "Random\nTree"]
xm = np.arange(len(top_models))

kb1_dump = [float(df_lf[(df_lf["Scenario"] == "KB1") & (df_lf["Test_Subset"] == "x64 Unseen Instances (80)") & (df_lf["Model"] == m)]["Accuracy"].values[0]) for m in top_models]
kb1_agg = [float(df_agg[(df_agg["Experiment"].str.startswith("ExpB")) & (df_agg["Scenario"] == "KB1") & (df_agg["Subset"] == "x64 Test (10 con)") & (df_agg["Model"] == m)]["Accuracy"].values[0]) for m in top_models]

kb3_dump = [float(df_lf[(df_lf["Scenario"] == "KB3") & (df_lf["Test_Subset"] == "x64 Unseen Instances (80)") & (df_lf["Model"] == m)]["Accuracy"].values[0]) for m in top_models]
kb3_agg = [float(df_agg[(df_agg["Experiment"].str.startswith("ExpB")) & (df_agg["Scenario"] == "KB3") & (df_agg["Subset"] == "x64 Test (10 con)") & (df_agg["Model"] == m)]["Accuracy"].values[0]) for m in top_models]

w2 = 0.2
b_d1 = ax.bar(xm - 1.5 * w2, kb1_dump, width=w2, label="KB1 - Từng dòng dump (80 dòng)", color="#93c5fd", edgecolor="#1d4ed8")
b_a1 = ax.bar(xm - 0.5 * w2, kb1_agg, width=w2, label="KB1 - Gộp 10->1 (10 con độc lập)", color="#1d4ed8", edgecolor="white")
b_d3 = ax.bar(xm + 0.5 * w2, kb3_dump, width=w2, label="KB3 - Từng dòng dump (80 dòng)", color="#6ee7b7", edgecolor="#047857")
b_a3 = ax.bar(xm + 1.5 * w2, kb3_agg, width=w2, label="KB3 - Gộp 10->1 (10 con độc lập)", color="#059669", edgecolor="white")

for bars in [b_d1, b_a1, b_d3, b_a3]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_title("B. Độ Chính Xác Riêng Trên 10 Con Test x64 Cô Lập\n(Chấm từng dòng dump rời rạc vs. Gộp 10 dòng thành 1 đại diện)", fontsize=10.5, fontweight="bold")
ax.set_xticks(xm)
ax.set_xticklabels(top_labels, fontsize=9)
ax.set_ylabel("x64 Unseen Test Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 126)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

# --- PANEL C: Kiểm định chéo trên Toàn bộ 20 Con x64 (5-Fold GroupKFold 160 dòng vs. 3-Fold CV 20 con gộp) ---
ax = axes[2]
cv_models = ["Random_Forest", "Ensemble_XGB_RF", "XGBoost", "J_48", "Random_Tree"]
cv_labels = ["Random\nForest", "Ensemble\n(XGB+RF)", "XGBoost", "J48", "Random\nTree"]
xc = np.arange(len(cv_models))

gkf_kb1 = [float(df_gkf[(df_gkf["Scenario"] == "KB1") & (df_gkf["Model"] == m)]["GroupKFold_Acc (%)"].values[0]) for m in cv_models]
cv3_kb1 = [float(df_agg[(df_agg["Experiment"].str.startswith("ExpC")) & (df_agg["Scenario"] == "KB1") & (df_agg["Model"] == m)]["Accuracy"].values[0]) for m in cv_models]
gkf_kb2 = [float(df_gkf[(df_gkf["Scenario"] == "KB2") & (df_gkf["Model"] == m)]["GroupKFold_Acc (%)"].values[0]) for m in cv_models]
cv3_kb2 = [float(df_agg[(df_agg["Experiment"].str.startswith("ExpC")) & (df_agg["Scenario"] == "KB2") & (df_agg["Model"] == m)]["Accuracy"].values[0]) for m in cv_models]

c1 = ax.bar(xc - 1.5 * w2, gkf_kb1, width=w2, label="KB1 - 5-Fold GroupKFold (160 dòng)", color="#bfdbfe", edgecolor="#1e40af")
c2 = ax.bar(xc - 0.5 * w2, cv3_kb1, width=w2, label="KB1 - 3-Fold CV Gộp 10->1 (20 con)", color="#2563eb", edgecolor="white")
c3 = ax.bar(xc + 0.5 * w2, gkf_kb2, width=w2, label="KB2 - 5-Fold GroupKFold (160 dòng)", color="#fde68a", edgecolor="#b45309")
c4 = ax.bar(xc + 1.5 * w2, cv3_kb2, width=w2, label="KB2 - 3-Fold CV Gộp 10->1 (20 con)", color="#d97706", edgecolor="white")

for bars in [c1, c2, c3, c4]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_title("C. Kiểm Định Chéo Cô Lập Trên Toàn Bộ 20 Con Mã Độc x64\n(GroupKFold trên 160 dòng ổn định vs. 3-Fold CV trên 20 dòng gộp)", fontsize=10.5, fontweight="bold")
ax.set_xticks(xc)
ax.set_xticklabels(cv_labels, fontsize=9)
ax.set_ylabel("Cross-Validation Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 118)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p16 = os.path.join(OUT_DIR, "16_leakfree_gan_isolation_and_x64_comparison.png")
plt.savefig(p16, dpi=300)
shutil.copy(p16, os.path.join(ARTIFACT_DIR, "16_leakfree_gan_isolation_and_x64_comparison.png"))
plt.close(fig)
print(f"✅ Đã lưu biểu đồ 16: {p16}")
