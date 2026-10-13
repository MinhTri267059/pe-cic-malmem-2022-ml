import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Cấu hình thẩm mỹ đồ thị chuẩn khoa học & hỗ trợ tiếng Việt
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '--'

OUTPUT_DIR = "eda_figures"
os.makedirs(f"{OUTPUT_DIR}/Overview", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Scenario_1", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Scenario_2", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Scenario_3", exist_ok=True)

# -------------------------------------------------------------
# 0. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# -------------------------------------------------------------
print("🚀 Đang tải và tiền xử lý tập dữ liệu MalMem2022.csv...")
df_raw = pd.read_csv("MalMem2022.csv")
df = df_raw.drop_duplicates().copy()

# Xác định nhãn cho từng kịch bản
# Kịch bản 1: Nhị phân
df['Label_S1'] = df['Class']

# Kịch bản 2: 4 nhóm
def get_s2_label(cat):
    if 'Benign' in cat:
        return 'Benign'
    elif 'Ransomware' in cat:
        return 'Ransomware'
    elif 'Spyware' in cat:
        return 'Spyware'
    elif 'Trojan' in cat:
        return 'Trojan'
    return 'Unknown'

df['Label_S2'] = df['Category'].apply(get_s2_label)

# Kịch bản 3: 16 lớp (1 Benign + 15 họ mã độc)
def get_s3_label(cat):
    if cat == 'Benign':
        return 'Benign'
    parts = str(cat).split('-')
    return parts[1] if len(parts) > 1 else cat

df['Label_S3'] = df['Category'].apply(get_s3_label)

# Danh sách đặc trưng số học
meta_cols = ['Class', 'Category', 'Filename', 'Label_S1', 'Label_S2', 'Label_S3']
feature_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]
print(f"Tổng số mẫu: {len(df):,} | Số đặc trưng trích xuất: {len(feature_cols)}")

# Bảng màu chuẩn
PALETTE_S1 = {'Benign': '#2b5c8f', 'Malware': '#d9534f'}
PALETTE_S2 = {
    'Benign': '#2b5c8f',
    'Ransomware': '#d9534f',
    'Spyware': '#f0ad4e',
    'Trojan': '#5cb85c'
}

# -------------------------------------------------------------
# PHẦN 1: EDA TỔNG QUAN HỆ THỐNG ĐẶC TRƯNG (OVERVIEW)
# -------------------------------------------------------------
print("📊 [Overview] Đang vẽ biểu đồ phân loại nhóm đặc trưng & tổng hợp...")

def plot_feature_categories():
    categories = {
        'Tiến trình (pslist)': len([c for c in feature_cols if c.startswith('pslist')]),
        'Thư viện liên kết (dlllist)': len([c for c in feature_cols if c.startswith('dlllist')]),
        'Mô tả xử lý (handles)': len([c for c in feature_cols if c.startswith('handles')]),
        'Mô-đun nạp (ldrmodules)': len([c for c in feature_cols if c.startswith('ldrmodules')]),
        'Vùng nhớ ẩn/chèn (malfind)': len([c for c in feature_cols if c.startswith('malfind')]),
        'Đối chiếu bảng (psxview)': len([c for c in feature_cols if c.startswith('psxview')]),
        'Mô-đun hạt nhân (modules)': len([c for c in feature_cols if c.startswith('modules')]),
        'Dịch vụ & Trình điều khiển (svcscan)': len([c for c in feature_cols if c.startswith('svcscan')]),
        'Hàm phản hồi (callbacks)': len([c for c in feature_cols if c.startswith('callbacks')])
    }
    
    cat_df = pd.DataFrame(list(categories.items()), columns=['Category', 'Count']).sort_values('Count', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(cat_df['Category'], cat_df['Count'], color='#34495e', edgecolor='#2c3e50', alpha=0.85, height=0.65)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{int(w)} đặc trưng",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold', color='#2c3e50')
                    
    ax.set_title("Phân Bố 55 Đặc Trưng Bộ Nhớ Volatility Trong Bộ Dữ Liệu CIC-MalMem-2022", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Số lượng đặc trưng", fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(cat_df['Count']) + 3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Overview/01_dataset_feature_categories.png", dpi=300)
    plt.close()

plot_feature_categories()

# -------------------------------------------------------------
# PHẦN 2: EDA KỊCH BẢN 1 - PHÂN LOẠI NHỊ PHÂN (SCENARIO 1)
# -------------------------------------------------------------
print("📊 [Scenario 1] Đang vẽ các biểu đồ phân tích thực nghiệm nhị phân (Benign vs Malware)...")

# 1.1 Phân phối nhãn nhị phân
def plot_s1_distribution():
    counts = df['Label_S1'].value_counts()
    total = len(df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # Bar chart
    bars = ax1.bar(counts.index, counts.values, color=[PALETTE_S1[c] for c in counts.index], width=0.5, edgecolor='#333333', linewidth=1)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({h/total*100:.2f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                     ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax1.set_title("Số lượng mẫu theo lớp (Benign vs Malware)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Số lượng mẫu", fontsize=11)
    ax1.set_ylim(0, max(counts.values) * 1.15)
    
    # Donut chart
    ax2.pie(counts.values, labels=counts.index, autopct='%1.2f%%', startangle=90,
            colors=[PALETTE_S1[c] for c in counts.index], explode=(0.04, 0.04),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.55},
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title("Tỷ lệ phân bố nhãn (Cân bằng lý tưởng ~50:50)", fontsize=12, fontweight='bold', pad=12)
    
    plt.suptitle("Kịch Bản 1: Phân Phối Dữ Liệu Phân Loại Nhị Phân (Binary Classification)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_1/01_binary_class_distribution.png", dpi=300)
    plt.close()

# 1.2 Tương quan 2 chiều với nhãn độc hại (Top positive & negative correlations)
def plot_s1_correlation():
    y_bin = (df['Label_S1'] == 'Malware').astype(int)
    valid_cols = [c for c in feature_cols if df[c].std() > 0]
    corrs = df[valid_cols].apply(lambda x: x.corr(y_bin)).sort_values()
    
    # Top 7 positive correlations và Top 8 negative correlations
    top_pos = corrs.tail(7)
    top_neg = corrs.head(8)
    combined = pd.concat([top_neg, top_pos])
    
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    colors = ['#d9534f' if v > 0 else '#2b5c8f' for v in combined.values]
    bars = ax.barh(combined.index, combined.values, color=colors, height=0.65, edgecolor='#333333', alpha=0.88)
    
    for bar in bars:
        w = bar.get_width()
        offset = 5 if w > 0 else -38
        ax.annotate(f"{w:+.3f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold',
                    color='#222222')
                    
    ax.axvline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
    ax.set_title("Kịch Bản 1: Hệ Số Tương Quan Tuyến Tính Pearson Với Nhãn Mã Độc (Malware)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Hệ số tương quan r (Đỏ: Tỷ lệ thuận với Malware, Xanh: Tỷ lệ thuận với Benign)", fontsize=10.5, fontweight='bold')
    ax.set_xlim(-1.05, 0.65)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_1/02_binary_top_correlation.png", dpi=300)
    plt.close()

# 1.3 So sánh phân phối các đặc trưng then chốt giữa Benign vs Malware
def plot_s1_feature_distributions():
    key_features = [
        'dlllist.avg_dlls_per_proc',
        'handles.nevent',
        'handles.nthread',
        'malfind.ninjections',
        'ldrmodules.not_in_load',
        'pslist.nppid'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=300)
    axes = axes.flatten()
    
    for idx, feat in enumerate(key_features):
        ax = axes[idx]
        sns.boxplot(data=df, x='Label_S1', y=feat, hue='Label_S1', palette=PALETTE_S1, ax=ax,
                    width=0.45, fliersize=1.5, boxprops=dict(alpha=0.85), legend=False)
        ax.set_title(f"{feat}", fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Giá trị đo lường", fontsize=9.5)
        
    plt.suptitle("Kịch Bản 1: So Sánh Giá Trị Các Đặc Trưng Bộ Nhớ Phân Tách Giữa Benign Và Malware", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_1/03_binary_feature_distributions.png", dpi=300)
    plt.close()

# 1.4 Không gian biểu diễn 2D PCA phân tách nhị phân
def plot_s1_pca():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Label'] = df['Label_S1'].values
    
    sample_pca = pca_df.sample(n=min(5000, len(pca_df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=300)
    for label, color in PALETTE_S1.items():
        subset = sample_pca[sample_pca['Label'] == label]
        ax.scatter(subset['PC1'], subset['PC2'], label=label, color=color, alpha=0.55, s=18, edgecolors='none')
        
    var_exp = pca.explained_variance_ratio_
    ax.set_title("Kịch Bản 1: Chiếu Không Gian 2D PCA (Phân Tách Lành Tính vs Độc Hại)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"Thành phần chính 1 (PC1 - {var_exp[0]*100:.1f}% phương sai)", fontsize=11)
    ax.set_ylabel(f"Thành phần chính 2 (PC2 - {var_exp[1]*100:.1f}% phương sai)", fontsize=11)
    ax.legend(title="Phân lớp", frameon=True, fontsize=10.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_1/04_binary_pca_2d.png", dpi=300)
    plt.close()

plot_s1_distribution()
plot_s1_correlation()
plot_s1_feature_distributions()
plot_s1_pca()

# -------------------------------------------------------------
# PHẦN 3: EDA KỊCH BẢN 2 - PHÂN LOẠI 4 NHÓM (SCENARIO 2)
# -------------------------------------------------------------
print("📊 [Scenario 2] Đang vẽ các biểu đồ phân tích 4 nhóm tấn công (Benign, Ransomware, Spyware, Trojan)...")

# 2.1 Phân phối 4 nhóm
def plot_s2_distribution():
    counts = df['Label_S2'].value_counts()
    total = len(df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    order = ['Benign', 'Ransomware', 'Spyware', 'Trojan']
    ordered_counts = counts.loc[order]
    
    bars = ax1.bar(ordered_counts.index, ordered_counts.values,
                   color=[PALETTE_S2[c] for c in ordered_counts.index],
                   width=0.55, edgecolor='#333333', linewidth=1)
                   
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({h/total*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                     ha='center', va='center', fontsize=10.5, fontweight='bold', color='white')
                     
    ax1.set_title("Số lượng mẫu theo 4 nhóm tấn công", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Số lượng mẫu", fontsize=11)
    ax1.set_ylim(0, max(ordered_counts.values) * 1.15)
    
    ax2.pie(ordered_counts.values, labels=ordered_counts.index, autopct='%1.1f%%', startangle=140,
            colors=[PALETTE_S2[c] for c in ordered_counts.index],
            explode=(0.04, 0.03, 0.03, 0.03),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.55},
            textprops={'fontsize': 10.5, 'fontweight': 'bold'})
    ax2.set_title("Tỷ lệ % phân bổ giữa Benign và 3 nhóm mã độc", fontsize=12, fontweight='bold', pad=12)
    
    plt.suptitle("Kịch Bản 2: Phân Phối Dữ Liệu 4 Nhóm (Multi-Class 4 Categories)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_2/01_multi4_class_distribution.png", dpi=300)
    plt.close()

# 2.2 Heatmap "Dấu vết bộ nhớ" trung bình chuẩn hóa của 4 nhóm
def plot_s2_fingerprint_heatmap():
    salient_features = [
        'pslist.avg_threads',
        'pslist.avg_handlers',
        'dlllist.avg_dlls_per_proc',
        'handles.nhandles',
        'handles.nfile',
        'handles.nkey',
        'handles.nthread',
        'handles.nmutant',
        'ldrmodules.not_in_load',
        'ldrmodules.not_in_init',
        'malfind.ninjections',
        'malfind.commitCharge',
        'svcscan.nservices',
        'svcscan.process_services',
        'callbacks.nanonymous'
    ]
    
    scaler = StandardScaler()
    scaled_feats = pd.DataFrame(scaler.fit_transform(df[salient_features]), columns=salient_features)
    scaled_feats['Label'] = df['Label_S2'].values
    
    mean_profiles = scaled_feats.groupby('Label').mean().loc[['Benign', 'Ransomware', 'Spyware', 'Trojan']]
    
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    sns.heatmap(mean_profiles, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Z-Score (Độ lệch chuẩn)'},
                linewidths=0.5, ax=ax, vmin=-1.5, vmax=1.5)
                
    ax.set_title("Kịch Bản 2: 'Dấu Vết Hành Vi Bộ Nhớ' Trung Bình (Normalized Behavioral Fingerprint)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Đặc trưng phân tích Volatility", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nhóm phân loại", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_2/02_multi4_behavioral_fingerprint_heatmap.png", dpi=300)
    plt.close()

# 2.3 Boxplot so sánh các chỉ số then chốt qua 4 nhóm
def plot_s2_key_features():
    features = [
        'handles.nfile',
        'handles.nkey',
        'malfind.ninjections',
        'ldrmodules.not_in_init_avg'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=300)
    axes = axes.flatten()
    order = ['Benign', 'Ransomware', 'Spyware', 'Trojan']
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        sns.boxplot(data=df, x='Label_S2', y=feat, order=order, hue='Label_S2', palette=PALETTE_S2, ax=ax,
                    width=0.45, fliersize=1.5, boxprops=dict(alpha=0.85), legend=False)
        ax.set_title(f"Phân phối {feat}", fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Giá trị", fontsize=10)
        
    plt.suptitle("Kịch Bản 2: So Sánh Các Đặc Trưng Bộ Nhớ Nhạy Cảm Giữa 4 Nhóm Mã Độc", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_2/03_multi4_feature_comparisons.png", dpi=300)
    plt.close()

# 2.4 Không gian PCA 2D cho 4 nhóm
def plot_s2_pca():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Label'] = df['Label_S2'].values
    
    sample_pca = pca_df.sample(n=min(5000, len(pca_df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=300)
    for label in ['Benign', 'Ransomware', 'Spyware', 'Trojan']:
        subset = sample_pca[sample_pca['Label'] == label]
        ax.scatter(subset['PC1'], subset['PC2'], label=label, color=PALETTE_S2[label],
                   alpha=0.55, s=18, edgecolors='none')
                   
    var_exp = pca.explained_variance_ratio_
    ax.set_title("Kịch Bản 2: Chiếu Không Gian 2D PCA của 4 Nhóm Phân Loại", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"Thành phần chính 1 (PC1 - {var_exp[0]*100:.1f}% phương sai)", fontsize=11)
    ax.set_ylabel(f"Thành phần chính 2 (PC2 - {var_exp[1]*100:.1f}% phương sai)", fontsize=11)
    ax.legend(title="Nhóm", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_2/04_multi4_pca_2d.png", dpi=300)
    plt.close()

plot_s2_distribution()
plot_s2_fingerprint_heatmap()
plot_s2_key_features()
plot_s2_pca()

# -------------------------------------------------------------
# PHẦN 4: EDA KỊCH BẢN 3 - PHÂN LOẠI 16 LỚP / 15 HỌ MÃ ĐỘC (SCENARIO 3)
# -------------------------------------------------------------
print("📊 [Scenario 3] Đang vẽ các biểu đồ phân tích 16 lớp (1 Benign + 15 Họ mã độc)...")

# 3.1 Phân phối 16 lớp với màu sắc theo nhóm cha
def plot_s3_distribution():
    category_mapping = {}
    for idx, row in df[['Label_S3', 'Label_S2']].drop_duplicates().iterrows():
        category_mapping[row['Label_S3']] = row['Label_S2']
        
    counts = df['Label_S3'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    
    colors = [PALETTE_S2[category_mapping.get(label, 'Benign')] for label in counts.index]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], height=0.68, edgecolor='#333333', alpha=0.9)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:,}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold')
                    
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=PALETTE_S2['Benign'], label='Lành tính (Benign)'),
        mpatches.Patch(color=PALETTE_S2['Ransomware'], label='Tống tiền (Ransomware families)'),
        mpatches.Patch(color=PALETTE_S2['Spyware'], label='Phần mềm gián điệp (Spyware families)'),
        mpatches.Patch(color=PALETTE_S2['Trojan'], label='Mã độc Trojan (Trojan families)')
    ]
    ax.legend(handles=legend_patches, loc='lower right', frameon=True, fontsize=10.5)
    
    ax.set_title("Kịch Bản 3: Phân Phối Số Lượng Mẫu Của 16 Lớp (Benign + 15 Họ Mã Độc)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Số lượng mẫu (Instances)", fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(counts.values) * 1.12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_3/01_multi16_family_distribution.png", dpi=300)
    plt.close()

# 3.2 Tỷ trọng và sự cân bằng giữa các họ con trong từng phân nhóm cha
def plot_s3_family_breakdown():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    cats = ['Ransomware', 'Spyware', 'Trojan']
    for idx, cat in enumerate(cats):
        ax = axes[idx]
        sub_df = df[df['Label_S2'] == cat]
        family_counts = sub_df['Label_S3'].value_counts()
        
        bars = ax.bar(family_counts.index, family_counts.values,
                      color=PALETTE_S2[cat], width=0.55, edgecolor='#333333', alpha=0.85)
                      
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:,}",
                        xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                        ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
                        
        ax.set_title(f"Các họ thuộc nhóm {cat}\n(Tổng: {len(sub_df):,} mẫu)", fontsize=11.5, fontweight='bold', pad=10)
        ax.set_ylabel("Số lượng mẫu" if idx == 0 else "", fontsize=10)
        ax.set_ylim(0, max(family_counts.values) * 1.2)
        ax.tick_params(axis='x', rotation=30)
        
    plt.suptitle("Kịch Bản 3: Phân Bố Mẫu Của 15 Họ Mã Độc Theo Từng Nhóm Cha Tương Ứng", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_3/02_multi16_family_breakdown_by_category.png", dpi=300)
    plt.close()

# 3.3 Gom cụm phân cấp (Hierarchical Clustermap) giữa 15 họ mã độc
def plot_s3_clustermap():
    malware_df = df[df['Label_S3'] != 'Benign'].copy()
    
    top_features = [
        'dlllist.avg_dlls_per_proc', 'handles.nevent', 'handles.nthread', 'handles.nmutant',
        'ldrmodules.not_in_load', 'malfind.ninjections', 'svcscan.process_services',
        'pslist.avg_threads', 'handles.nfile', 'handles.nkey', 'handles.nsection', 'ldrmodules.not_in_init'
    ]
    
    scaler = StandardScaler()
    scaled_feats = pd.DataFrame(scaler.fit_transform(malware_df[top_features]), columns=top_features)
    scaled_feats['Family'] = malware_df['Label_S3'].values
    
    family_profiles = scaled_feats.groupby('Family').mean()
    
    g = sns.clustermap(family_profiles, cmap='vlag', figsize=(12, 8), dendrogram_ratio=(0.15, 0.15),
                       cbar_pos=(0.02, 0.8, 0.03, 0.15), linewidths=0.5, annot=True, fmt='.2f')
    g.fig.suptitle("Kịch Bản 3: Gom Cụm Phân Cấp (Hierarchical Clustermap) Hành Vi Bộ Nhớ Của 15 Họ Mã Độc",
                   fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f"{OUTPUT_DIR}/Scenario_3/03_multi16_hierarchical_clustering.png", dpi=300, bbox_inches='tight')
    plt.close()

# 3.4 Chiếu PCA 2D phân biệt các nhóm trong không gian mã độc
def plot_s3_pca():
    # Chiếu PCA trên tập mã độc (loại bỏ Benign để zoom cận cảnh cấu trúc 15 họ mã độc)
    malware_df = df[df['Label_S3'] != 'Benign'].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(malware_df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Category'] = malware_df['Label_S2'].values
    pca_df['Family'] = malware_df['Label_S3'].values
    
    sample_pca = pca_df.sample(n=min(5000, len(pca_df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    for cat in ['Ransomware', 'Spyware', 'Trojan']:
        subset = sample_pca[sample_pca['Category'] == cat]
        ax.scatter(subset['PC1'], subset['PC2'], label=f"{cat} (5 họ)",
                   color=PALETTE_S2[cat], alpha=0.55, s=20, edgecolors='none')
                   
    var_exp = pca.explained_variance_ratio_
    ax.set_title("Kịch Bản 3: Không Gian 2D PCA Phân Tách Nội Bộ 15 Họ Mã Độc (Loại Bỏ Benign)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"Thành phần chính 1 (PC1 - {var_exp[0]*100:.1f}% phương sai)", fontsize=11)
    ax.set_ylabel(f"Thành phần chính 2 (PC2 - {var_exp[1]*100:.1f}% phương sai)", fontsize=11)
    ax.legend(title="Nhóm mã độc", frameon=True, fontsize=10.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Scenario_3/04_multi16_pca_2d.png", dpi=300)
    plt.close()

plot_s3_distribution()
plot_s3_family_breakdown()
plot_s3_clustermap()
plot_s3_pca()

# -------------------------------------------------------------
# PHẦN 5: TỔNG HỢP SO SÁNH 3 KỊCH BẢN (SUMMARY)
# -------------------------------------------------------------
print("📊 [Summary] Đang vẽ biểu đồ so sánh độ phức tạp và hiệu suất qua 3 kịch bản...")

def plot_cross_scenario_comparison():
    scenarios = ['Kịch bản 1\n(Nhị phân)', 'Kịch bản 2\n(4 Nhóm)', 'Kịch bản 3\n(16 Lớp họ)']
    num_classes = [2, 4, 16]
    acc_xgboost = [100.0, 87.71, 76.60]
    acc_rf = [100.0, 87.78, 75.68]
    acc_nb = [99.25, 68.70, 56.83]
    
    x = np.arange(len(scenarios))
    width = 0.22
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    bars1 = ax1.bar(x - width, acc_xgboost, width, label='XGBoost', color='#2b5c8f', edgecolor='#333333')
    bars2 = ax1.bar(x, acc_rf, width, label='Random Forest', color='#5cb85c', edgecolor='#333333')
    bars3 = ax1.bar(x + width, acc_nb, width, label='Naive Bayes', color='#d9534f', edgecolor='#333333')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                         
    ax1.set_ylabel("Độ chính xác (Accuracy - %)", fontsize=11, fontweight='bold')
    ax1.set_ylim(40, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True, fontsize=10)
    
    ax2 = ax1.twinx()
    ax2.plot(x, num_classes, color='#f0ad4e', marker='o', linewidth=2.5, markersize=8, label='Số lượng lớp')
    for i, txt in enumerate(num_classes):
        ax2.annotate(f"{txt} lớp", (x[i], num_classes[i] + 0.8), ha='center', fontsize=10, fontweight='bold', color='#c07d00')
    ax2.set_ylabel("Số lượng lớp bài toán (Classes)", fontsize=11, fontweight='bold', color='#c07d00')
    ax2.set_ylim(0, 20)
    ax2.grid(False)
    
    plt.title("Mối Quan Hệ Giữa Độ Phức Tạp Bài Toán (Số Lớp) Và Hiệu Suất Mô Hình", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Overview/02_cross_scenario_complexity_summary.png", dpi=300)
    plt.close()

plot_cross_scenario_comparison()

print("\n🎉 ĐÃ VẼ VÀ LƯU TOÀN BỘ CÁC HÌNH ẢNH EDA THEO TỪNG THỰC NGHIỆM THÀNH CÔNG!")
