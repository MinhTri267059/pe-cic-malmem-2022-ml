"""
Vẽ bộ Ma trận nhầm lẫn (Confusion Matrix) chi tiết của các Mô hình Tốt nhất ở từng Kịch bản (KB1, KB2, KB3)
trên cả 3 tập đánh giá: Tập Test x32 (100 mẫu), Tập Test x64 (100 mẫu) và Bộ Test Gộp (200 mẫu).
"""
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from train_and_eval_dual_arch import build_70_features, get_labels, ArchAdaptiveModel

OUT_DIR = "eda_figures/DualArch_GAN"
ARTIFACT_DIR = "/Users/minhtri86/.gemini/antigravity/brain/06867a2e-b8f9-49bc-8bc8-06823bfb357c"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
sns.set_theme(style="white", font="DejaVu Sans")

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


def save_fig(fig, filename):
    p1 = os.path.join(OUT_DIR, filename)
    fig.savefig(p1, dpi=250, bbox_inches="tight", facecolor="white")
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy(p1, os.path.join(ARTIFACT_DIR, filename))
    plt.close(fig)
    print(f"✅ Đã xuất: {p1}")


def plot_cm_with_pct(cm, labels, ax, title, cmap="Blues", annot_size=11):
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                annot[i, j] = f"{cm[i, j]}\n({pct[i, j]:.1f}%)"
            else:
                annot[i, j] = "0"
    sns.heatmap(
        pct,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=100,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        linewidths=0.8,
        linecolor="#dddddd",
        annot_kws={"size": annot_size, "weight": "bold"},
    )
    ax.set_title(title, fontweight="bold", fontsize=11.5, pad=10)
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Nhãn thực tế (True)", fontweight="bold", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0)


# =========================================================================
# HÌNH 9: MA TRẬN NHẦM LẪN KỊCH BẢN 1 (MÔ HÌNH TỐT NHẤT: XGBOOST E2 & RANDOM TREE E2)
# =========================================================================
lbl_s1 = ["Benign", "Malware"]
y32_1 = get_labels(x32_te, 1).values
y64_1 = get_labels(x64_te, 1).values
y200_1 = np.concatenate([y32_1, y64_1])

u_xgb1 = joblib.load("Models_DualArch_Unified/Scenario_1/XGBoost.pkl")
u_rt1 = joblib.load("Models_DualArch_Unified/Scenario_1/Random_Tree.pkl")

p32_xgb1 = np.array([lbl_s1[int(i)] for i in u_xgb1.predict(X32_te)])
p64_xgb1 = np.array([lbl_s1[int(i)] for i in u_xgb1.predict(X64_te)])
p200_xgb1 = np.concatenate([p32_xgb1, p64_xgb1])

p32_rt1 = np.array([lbl_s1[int(i)] for i in u_rt1.predict(X32_te)])
p64_rt1 = np.array([lbl_s1[int(i)] for i in u_rt1.predict(X64_te)])
p200_rt1 = np.concatenate([p32_rt1, p64_rt1])

fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

# Hàng 1: XGBoost E2 Unified (Quán quân Accuracy: 97.0%)
plot_cm_with_pct(
    confusion_matrix(y32_1, p32_xgb1, labels=lbl_s1), lbl_s1, axes[0, 0],
    "[Hạng 1 Accuracy] XGBoost (E2 Unified)\nTập Test x32 (100 mẫu) — Acc: 100,00%", cmap="Greens", annot_size=13
)
plot_cm_with_pct(
    confusion_matrix(y64_1, p64_xgb1, labels=lbl_s1), lbl_s1, axes[0, 1],
    "[Hạng 1 Accuracy] XGBoost (E2 Unified)\nTập Test x64 (100 mẫu) — Acc: 94,00% | BalAcc: 82,75%", cmap="Greens", annot_size=13
)
plot_cm_with_pct(
    confusion_matrix(y200_1, p200_xgb1, labels=lbl_s1), lbl_s1, axes[0, 2],
    "[Hạng 1 Accuracy] XGBoost (E2 Unified)\nTổng gộp 200 mẫu (x32+x64) — Acc: 97,00% | Macro-F1: 96,52%", cmap="Greens", annot_size=13
)

# Hàng 2: Random Tree E2 Unified (Quán quân Balanced Accuracy: 96.21% / x64 BalAcc: 87.65%)
plot_cm_with_pct(
    confusion_matrix(y32_1, p32_rt1, labels=lbl_s1), lbl_s1, axes[1, 0],
    "[Hạng 1 Balanced Acc] Random Tree (E2 Unified)\nTập Test x32 (100 mẫu) — Acc: 100,00%", cmap="Blues", annot_size=13
)
plot_cm_with_pct(
    confusion_matrix(y64_1, p64_rt1, labels=lbl_s1), lbl_s1, axes[1, 1],
    "[Hạng 1 Balanced Acc] Random Tree (E2 Unified)\nTập Test x64 (100 mẫu) — Acc: 93,00% | BalAcc: 87,65%", cmap="Blues", annot_size=13
)
plot_cm_with_pct(
    confusion_matrix(y200_1, p200_rt1, labels=lbl_s1), lbl_s1, axes[1, 2],
    "[Hạng 1 Balanced Acc] Random Tree (E2 Unified)\nTổng gộp 200 mẫu (x32+x64) — Acc: 96,50% | BalAcc: 96,21%", cmap="Blues", annot_size=13
)

fig.suptitle(
    "MA TRẬN NHẦM LẪN KỊCH BẢN 1 (NHỊ PHÂN: BENIGN vs. MALWARE) CỦA 2 MÔ HÌNH TỐT NHẤT\n"
    "Hàng trên: XGBoost (E2 Unified — Đỉnh Accuracy 97,00%) | Hàng dưới: Random Tree (E2 Unified — Đỉnh Balanced Acc 96,21%)",
    fontsize=14, fontweight="bold", y=1.03
)
plt.tight_layout()
save_fig(fig, "09_cm_scenario1_best_models.png")


# =========================================================================
# HÌNH 10: MA TRẬN NHẦM LẪN KỊCH BẢN 2 (4 NHÓM: RANDOM FOREST E2 & XGBOOST E3)
# =========================================================================
lbl_s2 = ["Benign", "Ransomware", "Spyware", "Trojan"]
y32_2 = get_labels(x32_te, 2).values
y64_2 = get_labels(x64_te, 2).values
y200_2 = np.concatenate([y32_2, y64_2])

u_rf2 = joblib.load("Models_DualArch_Unified/Scenario_2/Random_Forest.pkl")
a_xgb2 = joblib.load("Models_DualArch_Best/Scenario_2/XGBoost.pkl")

p32_rf2 = np.array([lbl_s2[int(i)] for i in u_rf2.predict(X32_te)])
p64_rf2 = np.array([lbl_s2[int(i)] for i in u_rf2.predict(X64_te)])
p200_rf2 = np.concatenate([p32_rf2, p64_rf2])

p32_xgb2 = a_xgb2.predict(X32_te)
p64_xgb2 = a_xgb2.predict(X64_te)
p200_xgb2 = np.concatenate([p32_xgb2, p64_xgb2])

fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.5))

# Hàng 1: Random Forest E2 Unified (81.50%)
plot_cm_with_pct(
    confusion_matrix(y32_2, p32_rf2, labels=lbl_s2), lbl_s2, axes[0, 0],
    "[Hạng 1 KB2] Random Forest (E2 Unified)\nTập Test x32 (100 mẫu) — Acc: 89,00%", cmap="Greens", annot_size=10.5
)
plot_cm_with_pct(
    confusion_matrix(y64_2, p64_rf2, labels=lbl_s2), lbl_s2, axes[0, 1],
    "[Hạng 1 KB2] Random Forest (E2 Unified)\nTập Test x64 (100 mẫu) — Acc: 74,00% | BalAcc: 72,00%", cmap="Greens", annot_size=10.5
)
plot_cm_with_pct(
    confusion_matrix(y200_2, p200_rf2, labels=lbl_s2), lbl_s2, axes[0, 2],
    "[Hạng 1 KB2] Random Forest (E2 Unified)\nTổng gộp 200 mẫu — Acc: 81,50% | Macro-F1: 80,10%", cmap="Greens", annot_size=10.5
)

# Hàng 2: XGBoost E3 Arch-Adaptive (81.00%)
plot_cm_with_pct(
    confusion_matrix(y32_2, p32_xgb2, labels=lbl_s2), lbl_s2, axes[1, 0],
    "[Hạng 2 KB2] XGBoost (E3 Arch-Adaptive)\nTập Test x32 (100 mẫu) — Acc: 89,00%", cmap="Blues", annot_size=10.5
)
plot_cm_with_pct(
    confusion_matrix(y64_2, p64_xgb2, labels=lbl_s2), lbl_s2, axes[1, 1],
    "[Hạng 2 KB2] XGBoost (E3 Arch-Adaptive)\nTập Test x64 (100 mẫu) — Acc: 73,00% | BalAcc: 72,52%", cmap="Blues", annot_size=10.5
)
plot_cm_with_pct(
    confusion_matrix(y200_2, p200_xgb2, labels=lbl_s2), lbl_s2, axes[1, 2],
    "[Hạng 2 KB2] XGBoost (E3 Arch-Adaptive)\nTổng gộp 200 mẫu — Acc: 81,00% | Macro-F1: 79,50%", cmap="Blues", annot_size=10.5
)

fig.suptitle(
    "MA TRẬN NHẦM LẪN KỊCH BẢN 2 (4 NHÓM HÀNH VI: BENIGN, RANSOMWARE, SPYWARE, TROJAN)\n"
    "Hàng trên: Random Forest (E2 Unified — 81,50%) | Hàng dưới: XGBoost (E3 Arch-Adaptive — 81,00%)",
    fontsize=14, fontweight="bold", y=1.03
)
plt.tight_layout()
save_fig(fig, "10_cm_scenario2_best_models.png")


# =========================================================================
# HÌNH 11: MA TRẬN NHẦM LẪN KỊCH BẢN 3 TÁCH RIÊNG x32 (16 HỌ) & x64 (11 HỌ)
# CỦA MÔ HÌNH QUÁN QUÂN: ENSEMBLE (XGB+RF) — E3 ARCH-ADAPTIVE (67,50%)
# =========================================================================
y32_3 = get_labels(x32_te, 3).values
y64_3 = get_labels(x64_te, 3).values
y200_3 = np.concatenate([y32_3, y64_3])

axgb3 = joblib.load("Models_DualArch_Best/Scenario_3/XGBoost.pkl")
arf3 = joblib.load("Models_DualArch_Best/Scenario_3/Random_Forest.pkl")

prob32 = 0.5 * axgb3.est32_.predict_proba(X32_te) + 0.5 * arf3.est32_.predict_proba(X32_te)
prob64 = 0.5 * axgb3.est64_.predict_proba(X64_te) + 0.5 * arf3.est64_.predict_proba(X64_te)
p32_ens3 = np.array([axgb3.inv32_[int(i)] for i in np.argmax(prob32, axis=1)], dtype=object)
p64_ens3 = np.array([axgb3.inv64_[int(i)] for i in np.argmax(prob64, axis=1)], dtype=object)
p200_ens3 = np.concatenate([p32_ens3, p64_ens3])

fig, axes = plt.subplots(1, 2, figsize=(21, 9.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

lbls_32 = sorted(np.unique(y32_3))
cm32_3 = confusion_matrix(y32_3, p32_ens3, labels=lbls_32)
r32_sums = cm32_3.sum(axis=1, keepdims=True)
pct32_3 = np.divide(cm32_3 * 100.0, r32_sums, out=np.zeros_like(cm32_3, dtype=float), where=r32_sums != 0)
sns.heatmap(
    pct32_3, annot=cm32_3, fmt="d", cmap="YlGnBu", vmin=0, vmax=100, xticklabels=lbls_32, yticklabels=lbls_32,
    cbar=False, ax=axes[0], linewidths=0.5, linecolor="#dddddd", annot_kws={"size": 9.5, "weight": "bold"}
)
axes[0].set_title(
    "(A) Tập Test Windows 7 x32 (100 mẫu — 16 lớp)\nEnsemble (XGB+RF) E3 — Accuracy: 72,00% (72/100 mẫu đúng)",
    fontweight="bold", fontsize=12
)
axes[0].set_xlabel("Họ mã độc dự đoán (Predicted Family)", fontweight="bold")
axes[0].set_ylabel("Họ mã độc thực tế (True Family)", fontweight="bold")
plt.setp(axes[0].get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
plt.setp(axes[0].get_yticklabels(), rotation=0)

lbls_64 = sorted(np.unique(y64_3))
cm64_3 = confusion_matrix(y64_3, p64_ens3, labels=lbls_64)
plot_cm_with_pct(
    cm64_3, lbls_64, axes[1],
    "(B) Tập Test Windows 10 x64 (100 mẫu — 11 lớp)\nEnsemble (XGB+RF) E3 — Accuracy: 63,00% | BalAcc: 57,88%",
    cmap="Purples", annot_size=8.8
)
plt.setp(axes[1].get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

fig.suptitle(
    "MA TRẬN NHẦM LẪN KỊCH BẢN 3 (CHI TIẾT HỌ MÃ ĐỘC) CỦA MÔ HÌNH TỐT NHẤT: ENSEMBLE (XGB+RF) — E3 ARCH-ADAPTIVE\n"
    "Tổng độ chính xác trên 200 mẫu: 67,50% (135/200 mẫu đúng: 72/100 mẫu x32 + 63/100 mẫu x64)",
    fontsize=14, fontweight="bold", y=1.03
)
plt.tight_layout()
save_fig(fig, "11_cm_scenario3_x32_and_x64_best.png")


# =========================================================================
# HÌNH 12: MA TRẬN NHẦM LẪN TỔNG HỢP 24x24 HỌ MÃ ĐỘC TRÊN 200 MẪU (KB3)
# =========================================================================
fig, ax = plt.subplots(figsize=(16.5, 13.5))
lbls_24 = sorted(set(y200_3) | set(p200_ens3))
cm200_3 = confusion_matrix(y200_3, p200_ens3, labels=lbls_24)
r200_sums = cm200_3.sum(axis=1, keepdims=True)
pct200_3 = np.divide(cm200_3 * 100.0, r200_sums, out=np.zeros_like(cm200_3, dtype=float), where=r200_sums != 0)
sns.heatmap(
    pct200_3, annot=cm200_3, fmt="d", cmap="Blues", vmin=0, vmax=100, xticklabels=lbls_24, yticklabels=lbls_24,
    cbar=True, cbar_kws={"label": "Tỷ lệ nhận diện đúng theo hàng (%)"},
    ax=ax, linewidths=0.5, linecolor="#dddddd", annot_kws={"size": 9.5, "weight": "bold"}
)
ax.set_title(
    "MA TRẬN NHẦM LẪN TOÀN CỤC 24 HỌ MÃ ĐỘC TRÊN BỘ TEST KÉP 200 MẪU (100 x32 + 100 x64)\n"
    "Mô hình Quán quân Kịch bản 3: Ensemble (XGB+RF) — E3 Arch-Adaptive (70f + FullGAN) | Accuracy: 67,50% (135/200 mẫu đúng)",
    fontweight="bold", fontsize=13, pad=14
)
ax.set_xlabel("Họ mã độc dự đoán (24 lớp)", fontweight="bold", fontsize=11)
ax.set_ylabel("Họ mã độc thực tế (24 lớp)", fontweight="bold", fontsize=11)
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
save_fig(fig, "12_cm_scenario3_combined200_24classes.png")

print("🎉 HOÀN TẤT VẼ 4 HÌNH MA TRẬN NHẦM LẪN CỦA CÁC MÔ HÌNH TỐT NHẤT Ở 3 KỊCH BẢN!")
