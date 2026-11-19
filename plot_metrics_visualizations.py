import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory exists
out_dir = 'eda_figures/Overview'
os.makedirs(out_dir, exist_ok=True)

# Set global aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Load metrics for 80% split
scenarios_data = []
for sc in [1, 2, 3]:
    df = pd.read_csv(f'Metrics_Scenario_{sc}.csv')
    df_split = df[df['Test_Method'] == '80% Split'].copy()
    df_split['Scenario'] = f'Kịch bản {sc}'
    df_split['Scenario_Title'] = (
        'Kịch bản 1: Nhị phân (Binary)' if sc == 1 else
        ('Kịch bản 2: Đa lớp 4 nhóm (Multi-4)' if sc == 2 else 'Kịch bản 3: Đa lớp 16 họ (Multi-16)')
    )
    # Ensure numeric
    for col in ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'FPR']:
        df_split[col] = pd.to_numeric(df_split[col], errors='coerce')
    scenarios_data.append(df_split)

df_all = pd.concat(scenarios_data, ignore_index=True)

# -------------------------------------------------------------
# DẠNG 1: GROUPED BAR CHART (Biểu đồ cột nhóm so sánh đa chỉ số)
# -------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharey=True)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

models_order = ['XGBoost', 'Random Forest (RF)', 'J-48 (Decision Tree)', 'Random Tree (RT)', 'Naive Bayes (NB)']

for idx, sc_df in enumerate(scenarios_data):
    ax = axes[idx]
    sc_df = sc_df.set_index('Model').reindex(models_order).reset_index()
    
    x = np.arange(len(models_order))
    width = 0.18
    
    for m_idx, metric in enumerate(metrics):
        offset = (m_idx - 1.5) * width
        rects = ax.bar(x + offset, sc_df[metric] * 100, width, label=metric, color=colors[m_idx], alpha=0.9, edgecolor='white')
        # Add value label
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)

    ax.set_title(f"{sc_df['Scenario_Title'].iloc[0]} - So sánh Accuracy, Precision, Recall, F1-Score (80/20 Split)", 
                 fontsize=13, fontweight='bold', pad=12, color='#1a252f')
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=11, fontweight='semibold')
    ax.set_ylabel('Tỷ lệ (%)', fontsize=11)
    ax.set_ylim(40, 108)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if idx == 0:
        ax.legend(loc='lower left', ncol=4, frameon=True, fontsize=10, facecolor='#ffffff', edgecolor='#e0e0e0')

plt.suptitle('DẠNG 1: BIỂU ĐỒ CỘT NHÓM HIỆU NĂNG CÁC MÔ HÌNH QUA 3 KỊCH BẢN', fontsize=16, fontweight='heavy', y=0.995, color='#0d3b66')
plt.tight_layout()
p1_path = os.path.join(out_dir, '04_metrics_grouped_bar_comparison.png')
plt.savefig(p1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved Chart 1: {p1_path}')


# -------------------------------------------------------------
# DẠNG 2: COMPREHENSIVE HEATMAP (Bản đồ nhiệt ma trận toàn diện kèm FPR)
# -------------------------------------------------------------
fig, (ax_main, ax_fpr) = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={'width_ratios': [4, 1.2]})

# Prepare data for heatmap
df_heatmap = df_all.copy()
df_heatmap['Label'] = df_heatmap['Scenario'] + ' | ' + df_heatmap['Model'].str.replace(' (Decision Tree)', '').str.replace(' (RF)', '').str.replace(' (RT)', '').str.replace(' (NB)', '')

# Sort properly
heatmap_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
main_matrix = df_heatmap.set_index('Label')[heatmap_metrics] * 100
fpr_matrix = df_heatmap.set_index('Label')[['FPR']] * 100

# Plot main metrics heatmap
sns.heatmap(main_matrix, ax=ax_main, annot=True, fmt='.2f', cmap='YlGnBu', vmin=50, vmax=100, 
            cbar_kws={'label': 'Tỷ lệ % (Càng cao càng tốt)'}, linewidths=0.8, linecolor='white')
ax_main.set_title('Ma trận Chỉ số Hiệu năng Chính (Accuracy, Precision, Recall, F1)', fontsize=13, fontweight='bold', pad=10)
ax_main.set_ylabel('Kịch bản & Thuật toán', fontsize=11, fontweight='bold')
ax_main.set_xlabel('Chỉ số Đánh giá', fontsize=11, fontweight='bold')

# Plot FPR heatmap (separate color palette where lower is better)
sns.heatmap(fpr_matrix, ax=ax_fpr, annot=True, fmt='.3f', cmap='Reds', vmin=0, vmax=10, 
            cbar_kws={'label': 'FPR % (Càng thấp càng tốt)'}, linewidths=0.8, linecolor='white')
ax_fpr.set_title('Tỷ lệ Báo động Giả (FPR)', fontsize=13, fontweight='bold', pad=10)
ax_fpr.set_ylabel('')
ax_fpr.set_yticks([])
ax_fpr.set_xlabel('FPR (%)', fontsize=11, fontweight='bold')

plt.suptitle('DẠNG 2: BẢN ĐỒ NHIỆT (HEATMAP) TỔNG HỢP HIỆU NĂNG & TỶ LỆ BÁO ĐỘNG GIẢ', fontsize=15, fontweight='heavy', y=0.98, color='#0d3b66')
plt.tight_layout()
p2_path = os.path.join(out_dir, '05_metrics_comprehensive_heatmap.png')
plt.savefig(p2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved Chart 2: {p2_path}')


# -------------------------------------------------------------
# DẠNG 3: RADAR CHART & TRADE-OFF SCATTER PLOT
# -------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))

# Subplot 1: Radar Chart for Scenario 2 (Multi-4)
ax_radar = fig.add_subplot(1, 2, 1, polar=True)

# Select Scenario 2 models
sc2_df = df_all[df_all['Scenario'] == 'Kịch bản 2'].set_index('Model').reindex(models_order).reset_index()

radar_categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', '100 - FPR*10']
num_vars = len(radar_categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

model_palette = {'XGBoost': '#1f77b4', 'Random Forest (RF)': '#2ca02c', 'J-48 (Decision Tree)': '#ff7f0e', 
                 'Random Tree (RT)': '#9467bd', 'Naive Bayes (NB)': '#d62728'}

for _, row in sc2_df.iterrows():
    m_name = row['Model']
    values = [
        row['Accuracy'] * 100,
        row['Precision'] * 100,
        row['Recall'] * 100,
        row['F1-Score'] * 100,
        max(0, 100 - row['FPR'] * 100 * 2) # Inverse scaled FPR for radar
    ]
    values += values[:1]
    color = model_palette.get(m_name, '#333333')
    ax_radar.plot(angles, values, label=m_name.split(' (')[0], linewidth=2.2, color=color)
    ax_radar.fill(angles, values, color=color, alpha=0.1)

ax_radar.set_xticks(angles[:-1])
radar_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'An toàn (Thấp FPR)']
ax_radar.set_xticklabels(radar_labels, fontsize=11, fontweight='bold')
ax_radar.set_ylim(50, 100)
ax_radar.set_title('A. Radar Profile Đa Tiêu Chí (Kịch bản 2)', fontsize=13, fontweight='bold', pad=15)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True)

# Subplot 2: Trade-off Scatter Plot (Accuracy vs FPR)
ax_scatter = fig.add_subplot(1, 2, 2)

scenario_markers = {'Kịch bản 1': 'o', 'Kịch bản 2': 's', 'Kịch bản 3': '^'}

# Custom offsets to avoid text overlapping in Scenario 1 cluster
offsets = {
    ('Kịch bản 1', 'XGBoost'): (10, 6),
    ('Kịch bản 1', 'Random Forest (RF)'): (10, -8),
    ('Kịch bản 1', 'J-48 (Decision Tree)'): (-95, 6),
    ('Kịch bản 1', 'Random Tree (RT)'): (-95, -8),
    ('Kịch bản 1', 'Naive Bayes (NB)'): (8, -12),
    ('Kịch bản 2', 'XGBoost'): (8, 6),
    ('Kịch bản 2', 'Random Forest (RF)'): (8, -6),
    ('Kịch bản 2', 'J-48 (Decision Tree)'): (8, -8),
    ('Kịch bản 2', 'Random Tree (RT)'): (8, 6),
    ('Kịch bản 2', 'Naive Bayes (NB)'): (-105, 4),
    ('Kịch bản 3', 'XGBoost'): (8, 4),
    ('Kịch bản 3', 'Random Forest (RF)'): (-110, 4),
    ('Kịch bản 3', 'J-48 (Decision Tree)'): (8, 4),
    ('Kịch bản 3', 'Random Tree (RT)'): (8, -8),
    ('Kịch bản 3', 'Naive Bayes (NB)'): (8, 4),
}

for sc_name in ['Kịch bản 1', 'Kịch bản 2', 'Kịch bản 3']:
    sc_sub = df_all[df_all['Scenario'] == sc_name]
    for _, r in sc_sub.iterrows():
        m_label = r['Model'].split(' (')[0]
        c = model_palette.get(r['Model'], '#333333')
        ax_scatter.scatter(r['FPR'] * 100, r['Accuracy'] * 100, color=c, 
                           marker=scenario_markers[sc_name], s=130, alpha=0.9, edgecolors='black', linewidth=0.8)
        
        # Get tailored offset
        ox, oy = offsets.get((sc_name, r['Model']), (6, 2))
        sc_short = sc_name.replace('Kịch bản ', 'KB')
        ax_scatter.annotate(f"{m_label} ({sc_short})", 
                            (r['FPR'] * 100, r['Accuracy'] * 100),
                            xytext=(ox, oy), textcoords="offset points", fontsize=8.5, fontweight='bold', color='#1a252f')

# Highlight optimal region
ax_scatter.axhspan(85, 101, xmin=0, xmax=0.45, color='#2ca02c', alpha=0.08)
ax_scatter.text(0.2, 97, 'VÙNG TỐI ƯU\n(Accuracy Cao, FPR Thấp)', fontsize=10, fontweight='bold', color='#2ca02c',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8f0', edgecolor='#2ca02c', alpha=0.8))

ax_scatter.set_xlabel('Tỷ lệ Báo động Giả - FPR (%) [Càng thấp càng tốt]', fontsize=11, fontweight='bold')
ax_scatter.set_ylabel('Độ chính xác - Accuracy (%) [Càng cao càng tốt]', fontsize=11, fontweight='bold')
ax_scatter.set_title('B. Đánh đổi Hiệu năng (Accuracy vs FPR) qua 3 Kịch bản', fontsize=13, fontweight='bold', pad=12)
ax_scatter.grid(True, linestyle='--', alpha=0.6)
ax_scatter.set_xlim(-0.5, 10.5)
ax_scatter.set_ylim(50, 103)

# Add custom legend for shapes
from matplotlib.lines import Line2D
shape_legends = [
    Line2D([0], [0], marker='o', color='w', label='Kịch bản 1 (Nhị phân)', markerfacecolor='#555', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Kịch bản 2 (4 nhóm)', markerfacecolor='#555', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='Kịch bản 3 (16 họ)', markerfacecolor='#555', markersize=10),
]
ax_scatter.legend(handles=shape_legends, loc='lower left', fontsize=9.5, frameon=True)

plt.suptitle('DẠNG 3: PHÂN TÍCH ĐA CHIỀU (RADAR CHART) VÀ ĐÁNH ĐỔI ACCURACY VS FPR', fontsize=15, fontweight='heavy', y=0.98, color='#0d3b66')
plt.tight_layout()
p3_path = os.path.join(out_dir, '06_metrics_tradeoff_radar_analysis.png')
plt.savefig(p3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved Chart 3: {p3_path}')
