"""
Bước 1 & 2:
1. Làm sạch bộ dữ liệu MalMem2022 (x32):
   - Khử trùng lặp hoàn toàn trên 55 đặc trưng
   - Loại bỏ các mẫu có nhãn mâu thuẫn (label conflict)
   - Tách 100 mẫu test x32 phân tầng theo họ mã độc (50 Benign + 50 Malware trải đều 15 họ) -> x32_test_100.csv
   - Phần còn lại (57.908 mẫu sạch) -> x32_train_clean.csv
2. Huấn luyện mạng Prototype-Anchored Conditional WGAN-GP trên toàn bộ 49 đặc trưng khả dụng của x64_train_100.csv:
   - Sinh 1.000 mẫu tổng hợp chất lượng cao (tăng cường tỷ lệ Benign x64 lên 184 mẫu + 48 mẫu × 17 biến thể malware = 1.000 mẫu)
   - Đảm bảo ràng buộc vật lý (miền giá trị, số nguyên, cột hằng, cột trùng lặp)
   - Lưu -> x64_gan_1000_full.csv và báo cáo kiểm định GAN_Full_Quality_Report.csv
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

SEED = 42
N_SYNTH = 1000
Z_DIM = 32
EPOCHS = 2500
N_CRITIC = 4
LAMBDA_GP = 10.0

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================================================================
# 1. LÀM SẠCH BỘ DỮ LIỆU x32 (MalMem2022) & TÁCH 100 MẪU TEST x32
# =========================================================================
print("=" * 75)
print("1️⃣ LÀM SẠCH DỮ LIỆU x32 (MalMem2022) & TÁCH BỘ TEST (100 x32 + 100 x64)")
print("=" * 75)

df_x32 = pd.read_csv("MalMem2022.csv")
raw_cols = [c for c in df_x32.columns if c not in ["Class", "Category", "Filename"]]
df_x32["Family"] = df_x32["Category"].str.split("-").str[:2].str.join("-")

# Loại bỏ dòng trùng lặp cùng nhãn, sau đó loại bỏ các vector đặc trưng có nhãn mâu thuẫn
df_dedup = df_x32.drop_duplicates(subset=raw_cols + ["Family"]).copy()
conflict_mask = df_dedup.duplicated(subset=raw_cols, keep=False)
df_x32_clean = df_dedup[~conflict_mask].drop(columns=["Family"]).copy()

print(f"📌 MalMem2022 gốc: {len(df_x32)} mẫu")
print(f"📌 Sau khi khử trùng lặp & loại bỏ xung đột nhãn: {len(df_x32_clean)} mẫu sạch (-{len(df_x32) - len(df_x32_clean)} mẫu nhiễu/trùng)")

# Tách 100 mẫu x32 test phân tầng theo họ (50 Benign + 50 Malware)
fam_series = df_x32_clean["Category"].str.split("-").str[:2].str.join("-")
x32_tr, x32_te = train_test_split(
    df_x32_clean, test_size=100, random_state=SEED, stratify=fam_series
)
x32_tr.to_csv("x32_train_clean.csv", index=False)
x32_te.to_csv("x32_test_100.csv", index=False)

print(f"✅ Đã lưu x32_train_clean.csv ({len(x32_tr)} mẫu) và x32_test_100.csv ({len(x32_te)} mẫu)")
print(f"   Phân bố x32_test_100 Class: {x32_te['Class'].value_counts().to_dict()}")
print(f"   Phân bố x32_test_100 Group: {x32_te['Category'].str.split('-').str[0].value_counts().to_dict()}")

# =========================================================================
# 2. HUẤN LUYỆN CONDITIONAL WGAN-GP ĐẦY ĐỦ ĐẶC TRƯNG TRÊN x64_train_100.csv
# =========================================================================
print("\n" + "=" * 75)
print("2️⃣ HUẤN LUYỆN FULL-FEATURE PROTOTYPE-ANCHORED cWGAN-GP (100 -> 1000 MẪU)")
print("=" * 75)

real = pd.read_csv("x64_train_100.csv")
test_x64 = pd.read_csv("x64_test_100.csv")

avail = [c for c in raw_cols if real[c].notna().any()]
nan_cols = [c for c in raw_cols if c not in avail]
const_cols = [c for c in avail if real[c].nunique() == 1]

dup_of = {}
for i, c in enumerate(avail):
    if c in const_cols or c in dup_of:
        continue
    for c2 in avail[i + 1:]:
        if c2 not in const_cols and c2 not in dup_of and real[c].equals(real[c2]):
            dup_of[c2] = c

gen_cols = [c for c in avail if c not in const_cols and c not in dup_of]
int_cols = [c for c in gen_cols if (real[c] % 1 == 0).all()]

print(f"📌 Tổng đặc trưng gốc: {len(raw_cols)} | Khả dụng trên x64: {len(avail)} | Biến thiên độc lập cho GAN: {len(gen_cols)}")

classes = sorted(real["Category"].unique())
cls_idx = {c: i for i, c in enumerate(classes)}
n_cls = len(classes)
d = len(gen_cols)

X_log = np.log1p(np.maximum(0.0, real[gen_cols].values.astype(np.float64)))
x_min, x_max = X_log.min(0), X_log.max(0)
span = np.where(x_max - x_min > 1e-8, x_max - x_min, 1.0)
Xs = 2.0 * (X_log - x_min) / span - 1.0
y = real["Category"].map(cls_idx).values

cls_rows = {i: Xs[y == i] for i in range(n_cls)}
cls_std = {i: np.maximum(Xs[y == i].std(0), 0.02) for i in range(n_cls)}


class AnchoredGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM + n_cls + d, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, d), nn.Tanh(),
        )

    def forward(self, z, c_oh, anchor, scale):
        delta = self.net(torch.cat([z, c_oh, anchor], dim=1))
        return torch.clamp(anchor + 0.30 * scale * delta, -1.0, 1.0)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + n_cls, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, c_oh):
        return self.net(torch.cat([x, c_oh], dim=1))


def sample_anchors(labels_np):
    anchors, scales = [], []
    for c in labels_np:
        rows = cls_rows[c]
        w = np.random.dirichlet(np.ones(len(rows)) * 1.5)
        anchors.append(w @ rows)
        scales.append(cls_std[c])
    return (
        torch.tensor(np.array(anchors), dtype=torch.float32),
        torch.tensor(np.array(scales), dtype=torch.float32),
    )


G, D = AnchoredGenerator(), Critic()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))

Xt = torch.tensor(Xs, dtype=torch.float32)
Yt = torch.nn.functional.one_hot(torch.tensor(y), n_cls).float()

for ep in range(1, EPOCHS + 1):
    for _ in range(N_CRITIC):
        z = torch.randn(len(Xt), Z_DIM)
        anc, scl = sample_anchors(y)
        fake = G(z, Yt, anc, scl).detach()
        a = torch.rand(len(Xt), 1)
        inter = (a * Xt + (1.0 - a) * fake).requires_grad_(True)
        grad = torch.autograd.grad(D(inter, Yt).sum(), inter, create_graph=True)[0]
        gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
        loss_d = D(fake, Yt).mean() - D(Xt, Yt).mean() + LAMBDA_GP * gp
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

    z = torch.randn(len(Xt), Z_DIM)
    anc, scl = sample_anchors(y)
    fake = G(z, Yt, anc, scl)
    loss_g = -D(fake, Yt).mean() + 0.4 * ((fake - anc) ** 2).mean()
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

    if ep % 500 == 0:
        print(f"   Epoch {ep:4d}/{EPOCHS} | Critic Loss: {loss_d.item():.4f} | Generator Loss: {loss_g.item():.4f}")

# Sinh 1.000 mẫu (Tăng cường Benign: 184 mẫu Benign + 48 mẫu × 17 biến thể Malware = 1.000 mẫu)
lab = []
for i, c in enumerate(classes):
    n_c = 184 if c == "Benign" else 48
    lab.extend([i] * n_c)
lab = np.array(lab)

G.eval()
with torch.no_grad():
    anc, scl = sample_anchors(lab)
    c_oh = torch.nn.functional.one_hot(torch.tensor(lab), n_cls).float()
    out = G(torch.randn(N_SYNTH, Z_DIM), c_oh, anc, scl).numpy()

Xg = np.expm1((out + 1.0) / 2.0 * span + x_min)
syn = pd.DataFrame(Xg, columns=gen_cols)
for c in gen_cols:
    syn[c] = syn[c].clip(real[c].min(), real[c].max())
for c in int_cols:
    syn[c] = syn[c].round()
for c in const_cols:
    syn[c] = real[c].iloc[0]
for c, src in dup_of.items():
    syn[c] = syn[src]
for c in nan_cols:
    syn[c] = np.nan

syn["Category"] = [classes[i] for i in lab]
syn["Class"] = np.where(syn["Category"] == "Benign", "Benign", "Malware")
syn["Filename"] = [f"GAN_FULL_{classes[i]}_{k}" for k, i in enumerate(lab)]
syn = syn[raw_cols + ["Class", "Category", "Filename"]]
syn.to_csv("x64_gan_1000_full.csv", index=False)
print(f"\n✅ Đã lưu 1.000 mẫu tổng hợp đầy đủ đặc trưng -> x64_gan_1000_full.csv")

# Kiểm định chất lượng GAN trên tập test độc lập x64_test_100.csv (TSTR)
nn_real = NearestNeighbors(n_neighbors=2).fit(Xs)
syn_s = 2.0 * (np.log1p(np.maximum(0.0, syn[gen_cols].values)) - x_min) / span - 1.0
dcr_syn = nn_real.kneighbors(syn_s, n_neighbors=1)[0].ravel()
n_exact = int((dcr_syn < 1e-6).sum())
n_unique = int(syn[gen_cols].drop_duplicates().shape[0])

ks_vals = [ks_2samp(real[c], syn[c]).statistic for c in gen_cols]
q_rows = [
    {"Metric": "Mean KS Statistic (41 features)", "Value": round(float(np.mean(ks_vals)), 4)},
    {"Metric": "Median DCR (Synthetic -> Real)", "Value": round(float(np.median(dcr_syn)), 4)},
    {"Metric": "Exact Memorized Copies", "Value": n_exact},
    {"Metric": "Unique Synthetic Samples / 1000", "Value": n_unique},
]

lbl_fns = {
    "KB1 (Binary)": lambda s: s["Class"],
    "KB2 (4 Groups)": lambda s: s["Category"].str.split("-").str[0],
    "KB3 (Families)": lambda s: s["Category"].str.split("-").str[:2].str.join("-"),
}
for k, fn in lbl_fns.items():
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(syn[avail], fn(syn))
    pred_te = rf.predict(test_x64[avail])
    acc = accuracy_score(fn(test_x64), pred_te)
    bacc = balanced_accuracy_score(fn(test_x64), pred_te)
    mf1 = f1_score(fn(test_x64), pred_te, average="macro")
    q_rows.append({"Metric": f"TSTR Held-out Test {k} Accuracy", "Value": round(acc * 100, 2)})
    q_rows.append({"Metric": f"TSTR Held-out Test {k} Balanced Acc", "Value": round(bacc * 100, 2)})
    q_rows.append({"Metric": f"TSTR Held-out Test {k} Macro-F1", "Value": round(mf1 * 100, 2)})

q_df = pd.DataFrame(q_rows)
q_df.to_csv("GAN_Full_Quality_Report.csv", index=False)
print("\n📊 BÁO CÁO CHẤT LƯỢNG MẪU GAN (49 ĐẶC TRƯNG):")
print(q_df.to_string(index=False))

