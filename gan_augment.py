"""
Conditional WGAN-GP sinh 1000 mẫu tổng hợp từ 100 mẫu train x64 (build 19041).

- Điều kiện: Category (18 biến thể: Benign + 17 biến thể malware)
- Đặc trưng: 22 đặc trưng bất biến; 6 cột psxview không khả dụng trên x64 giữ NaN (như dữ liệu thật)
- Cột hằng số và cột trùng lặp (vd: not_in_load == not_in_mem) được xử lý riêng để giữ ràng buộc
- Đầu ra: x64_gan_1000.csv + GAN_Quality_Report.csv
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

from train_and_evaluate_22features import FEATURES_22

SEED = 42
N_SYNTH = 1000
Z_DIM = 32
EPOCHS = 4000
N_CRITIC = 5
LAMBDA_GP = 10.0

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- dữ liệu
real = pd.read_csv("x64_train_100.csv")
avail = [c for c in FEATURES_22 if real[c].notna().any()]
nan_cols = [c for c in FEATURES_22 if c not in avail]

const_cols = [c for c in avail if real[c].nunique() == 1]
dup_of = {}  # cột trùng lặp -> cột gốc
for i, c in enumerate(avail):
    if c in const_cols or c in dup_of:
        continue
    for c2 in avail[i + 1:]:
        if c2 not in const_cols and c2 not in dup_of and real[c].equals(real[c2]):
            dup_of[c2] = c
gen_cols = [c for c in avail if c not in const_cols and c not in dup_of]
int_cols = [c for c in gen_cols if (real[c] % 1 == 0).all()]

print(f"Cột sinh bởi GAN ({len(gen_cols)}): {gen_cols}")
print(f"Cột hằng: {const_cols} | Cột trùng: {dup_of} | Cột NaN: {len(nan_cols)}")

classes = sorted(real["Category"].unique())
cls_idx = {c: i for i, c in enumerate(classes)}
n_cls = len(classes)

X = np.log1p(real[gen_cols].values.astype(np.float64))
x_min, x_max = X.min(0), X.max(0)
Xs = 2 * (X - x_min) / (x_max - x_min) - 1  # [-1, 1]
y = real["Category"].map(cls_idx).values

Xt = torch.tensor(Xs, dtype=torch.float32)
Yt = torch.nn.functional.one_hot(torch.tensor(y), n_cls).float()
d = len(gen_cols)


# ---------------------------------------------------------------- mô hình
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM + n_cls, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, d), nn.Tanh())

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + n_cls, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], 1))


G, D = Generator(), Critic()
opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))


def gradient_penalty(real_x, fake_x, c):
    a = torch.rand(real_x.size(0), 1)
    inter = (a * real_x + (1 - a) * fake_x).requires_grad_(True)
    grad = torch.autograd.grad(D(inter, c).sum(), inter, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# ---------------------------------------------------------------- huấn luyện (full-batch, 100 mẫu)
for ep in range(1, EPOCHS + 1):
    for _ in range(N_CRITIC):
        z = torch.randn(len(Xt), Z_DIM)
        fake = G(z, Yt).detach()
        loss_d = D(fake, Yt).mean() - D(Xt, Yt).mean() + LAMBDA_GP * gradient_penalty(Xt, fake, Yt)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    z = torch.randn(len(Xt), Z_DIM)
    loss_g = -D(G(z, Yt), Yt).mean()
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    if ep % 500 == 0:
        print(f"Epoch {ep:5d} | W-dist ≈ {-loss_d.item() + 0:.4f} | G loss {loss_g.item():.4f}")

# ---------------------------------------------------------------- sinh 1000 mẫu cân bằng theo lớp
per_cls = [N_SYNTH // n_cls + (1 if i < N_SYNTH % n_cls else 0) for i in range(n_cls)]
lab = np.concatenate([[i] * k for i, k in enumerate(per_cls)])
G.eval()
with torch.no_grad():
    out = G(torch.randn(N_SYNTH, Z_DIM),
            torch.nn.functional.one_hot(torch.tensor(lab), n_cls).float()).numpy()

Xg = np.expm1((out + 1) / 2 * (x_max - x_min) + x_min)
syn = pd.DataFrame(Xg, columns=gen_cols)
for c in gen_cols:  # cắt về miền giá trị thật
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
syn["Filename"] = [f"GAN_{classes[i]}_{k}" for k, i in enumerate(lab)]
syn = syn[FEATURES_22 + ["Class", "Category", "Filename"]]
syn.to_csv("x64_gan_1000.csv", index=False)
print(f"\n✅ Đã lưu {len(syn)} mẫu tổng hợp -> x64_gan_1000.csv")

# ---------------------------------------------------------------- đánh giá chất lượng
report = []
for c in gen_cols:
    ks = ks_2samp(real[c], syn[c])
    report.append({"Metric": f"KS[{c}]", "Value": round(ks.statistic, 4), "p_value": round(ks.pvalue, 4)})

# Khoảng cách tới mẫu thật gần nhất (DCR) – phát hiện việc chép lại mẫu thật
nn_real = NearestNeighbors(n_neighbors=2).fit(Xs)
syn_s = 2 * (np.log1p(syn[gen_cols].values) - x_min) / (x_max - x_min) - 1
dcr_syn = nn_real.kneighbors(syn_s, n_neighbors=1)[0].ravel()
dcr_real = nn_real.kneighbors(Xs, n_neighbors=2)[0][:, 1]
n_exact = int((dcr_syn < 1e-6).sum())
report += [
    {"Metric": "DCR median (synthetic -> real)", "Value": round(float(np.median(dcr_syn)), 4)},
    {"Metric": "NN median (real -> real khác)", "Value": round(float(np.median(dcr_real)), 4)},
    {"Metric": "Số mẫu trùng khớp tuyệt đối", "Value": n_exact},
    {"Metric": "Số mẫu duy nhất / 1000", "Value": int(syn[gen_cols].drop_duplicates().shape[0])},
]

# TSTR: train trên dữ liệu giả, test trên 100 mẫu thật
lbl = {
    "KB1": lambda s: s["Class"],
    "KB2": lambda s: s["Category"].str.split("-").str[0],
    "KB3": lambda s: s["Category"].str.split("-").str[:2].str.join("-"),
}
for k, f in lbl.items():
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(syn[gen_cols], f(syn))
    acc = accuracy_score(f(real), rf.predict(real[gen_cols]))
    report.append({"Metric": f"TSTR accuracy {k} (train giả -> test thật)", "Value": round(acc, 4)})

rep = pd.DataFrame(report)
rep.to_csv("GAN_Quality_Report.csv", index=False)
print(rep.to_string(index=False))

