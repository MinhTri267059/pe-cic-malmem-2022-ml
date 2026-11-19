"""
Vẽ biểu đồ so sánh toàn diện giữa:
  - PIPELINE 1: Mô hình Độc lập Song song (Flat Multi-Scenario Pipeline)
  - PIPELINE 2: Mô hình Phân cấp 3 Tầng (3-Stage Hierarchical Cascade Pipeline)
qua cả 3 Kịch bản (KB1: 2 nhãn, KB2: 4 nhóm, KB3: 24 họ) trên:
  - Hàng 1: Bộ Test Chuẩn Kép 200 mẫu (100 x32 + 100 x64)
  - Hàng 2: Riêng trên Tập Test Windows 10 x64 (100 mẫu)
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "eda_figures/DualArch_GAN"
ARTIFACT_DIR = "/Users/minhtri86/.gemini/antigravity/brain/06867a2e-b8f9-49bc-8bc8-06823bfb357c"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
sns.set_theme(style="whitegrid", font="DejaVu Sans")

df_flat = pd.read_csv("Metrics_DualArch_All_Comparisons.csv")
df_pipe = pd.read_csv("Metrics_Hierarchical_Pipeline_3Stages.csv")
df_all = pd.concat([df_flat, df_pipe], ignore_index=True)

models_order = ["XGBoost", "Ensemble_XGB_RF", "Random_Forest", "J_48", "Random_Tree", "Naive_Bayes"]
models_labels = ["XGBoost", "Ensemble\n(XGB+RF)", "Random\nForest", "J-48\n(C4.5)", "Random\nTree", "Naive\nBayes"]

methods_compare = [
    ("E2: Unified (Clean+FullGAN 70f)", "Pipeline 1: Flat (Unified)", "#7fb3d5"),
    ("Pipeline 3 Tầng (Unified 70f)", "Pipeline 2: 3 Tầng (Unified)", "#1f618d"),
    ("E3: Arch-Adaptive (Clean+FullGAN 70f)", "Pipeline 1: Flat (Arch-Adaptive)", "#f5b041"),
    ("Pipeline 3 Tầng (Arch-Adaptive 70f)", "Pipeline 2: 3 Tầng (Arch-Adaptive)", "#b9770e"),
]

scenarios_info = [
    ("KB1", "Kịch bản 1 / Tầng 1: Nhị phân (2 nhãn)"),
    ("KB2", "Kịch bản 2 / Tầng 2: Đa lớp (4 nhóm)"),
    ("KB3", "Kịch bản 3 / Tầng 3: Chi tiết (24 họ)"),
]

fig, axes = plt.subplots(2, 3, figsize=(22, 12.5), sharey="row")

for row_idx, (subset_key, subset_title) in enumerate([
    ("Combined (200)", "TRÊN BỘ TEST CHUẨN KÉP 200 MẪU (100 MẪU x32 + 100 MẪU x64)"),
    ("x64 (100)", "RIÊNG TRÊN TẬP TEST WINDOWS 10 x64 (100 MẪU)"),
]):
    for col_idx, (sc_code, sc_title) in enumerate(scenarios_info):
        ax = axes[row_idx, col_idx]
        sub = df_all[(df_all["Scenario"] == sc_code) & (df_all["Test_Subset"] == subset_key)]
        piv = sub.pivot(index="Model", columns="Method", values="Accuracy").reindex(models_order)

        x = np.arange(len(models_order))
        w = 0.185
        step = 0.20

        for m_i, (m_key, m_label, m_color) in enumerate(methods_compare):
            vals = piv[m_key].values
            bars = ax.bar(
                x + (m_i - 1.5) * step,
                vals,
                width=w,
                label=m_label if (row_idx == 0 and col_idx == 0) else "_nolegend_",
                color=m_color,
                edgecolor="black",
                linewidth=0.7,
            )
            for b in bars:
                h = b.get_height()
                if not np.isnan(h) and h > 0:
                    lbl_txt = f"{int(h)}" if abs(h - round(h)) < 0.05 else f"{h:.1f}"
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        h + 1.0,
                        lbl_txt,
                        ha="center",
                        va="bottom",
                        fontsize=7.6,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(models_labels, fontweight="bold", fontsize=9.5)
        ax.set_ylim(0, 109)
        prefix = "(A" if row_idx == 0 else "(B"
        ax.set_title(f"{prefix}{col_idx+1}) {sc_title}\n[{subset_title.split('(')[0].strip()}]", fontweight="bold", fontsize=11.5)
        if col_idx == 0:
            ax.set_ylabel(f"Accuracy (%) — {subset_key}", fontweight="bold", fontsize=11)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=True, fontsize=11, edgecolor="#cccccc")

fig.suptitle(
    "SO SÁNH TOÀN DIỆN PIPELINE 1 (PHẲNG / ĐỘC LẬP) vs. PIPELINE 2 (PHÂN CẤP 3 TẦNG) QUA CẢ 3 KỊCH BẢN\n"
    "Hàng trên: Trên toàn bộ 200 mẫu Test (100 x32 + 100 x64) | Hàng dưới: Riêng trên 100 mẫu Test Windows 10 x64",
    fontsize=14.5,
    fontweight="bold",
    y=1.055,
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "13_pipeline_flat_vs_3stage_across_3scenarios.png")
fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
if os.path.exists(ARTIFACT_DIR):
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, "13_pipeline_flat_vs_3stage_across_3scenarios.png"))
plt.close(fig)
print(f"✅ Đã xuất: {out_path}")
