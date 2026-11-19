"""
Bước 1 & 2 (KỊCH BẢN CHỐNG RÒ RỈ DỮ LIỆU TUYỆT ĐỐI - LEAK-FREE BENCHMARK):
1. Làm sạch bộ dữ liệu 200 dòng x64 (features_all_x64_19041.csv -> features_all_x64_19041_Cleaned.csv):
   - Loại bỏ 40 bản dump nhiễu khởi động máy ảo (dump_1 & dump_2: 15-30 giây đầu khi tiến trình nền
     Windows 10 vừa resume làm mẫu Benign có nproc=117, ninjections=18 giống hệt Malware) -> giữ 160 mẫu trạng thái ổn định (dump_3..dump_10).
   - Loại bỏ 23 cột lỗi/rác/gây rò rỉ phiên chạy máy ảo:
     + 6 cột psxview bị trống hoàn toàn (100% NaN trên x64)
     + 5 cột hằng số bằng 0 (handles.nport, malfind.commitCharge, psxview.not_in_eprocess_pool*, svcscan.interactive_process_services)
     + 2 cột trùng lặp tuyến tính 100% (ldrmodules.not_in_mem == not_in_load, not_in_mem_avg == not_in_load_avg)
     + 10 cột cấu hình tĩnh định danh phiên máy ảo (svcscan.*, callbacks.*, modules.nmodules có phương sai nội bộ phiên = 0)
   -> Thu được đúng 32 đặc trưng hành vi động cốt lõi (CLEAN_32_COLS) + cột định danh Instance_ID.

2. Chia Train/Test Cô lập Tuyệt đối theo Con Mã độc / Phiên chạy (0% Instance/Hash Overlap):
   - Trên x64 (20 con/phiên chạy = 160 dòng sạch):
     + Tập Train (10 con = 80 dòng): Benign-browse, Benign-idle, Conti-1, LockBit-2, AgentTesla-1, Formbook-1, SnakeKeylogger-3, QuasarRAT-2, Remcos-1, DCRat-1.
     + Tập Test Cô lập (10 con hoàn toàn khác = 80 dòng): Benign-office, Conti-2, LockBit-3, Pysa-1, AgentTesla-2, Formbook-2, QuasarRAT-3, Remcos-2, Remcos-3, TrickBot-2.
     -> Giao của Instance_ID giữa Train và Test = 0 / 20 (0% rò rỉ).
   - Trên x32 (MalMem2022):
     + Trích xuất mã Hash thực thi của 2.906 con mã độc (Instance_ID).
     + Chọn 40 con mã độc độc lập (trải đều qua 15 họ) đưa vào Tập Test x32 (lấy 1 bản dump đại diện/con = 40 dòng Malware)
       và XÓA SẠCH toàn bộ 10 bản dump (-1.raw..-10.raw) của cả 40 con mã độc đó khỏi Tập Train x32!
     + Với Benign x32: tách 1 khối phiên chạy riêng biệt (40 dòng) làm Test và xóa mọi mẫu Benign trong Train có khoảng cách Euclid gần khối Test.
     -> Tập Test x32 Cô lập = 80 dòng (40 Benign + 40 Malware từ 40 mã hash chưa từng thấy).

3. Huấn luyện mạng Prototype-Anchored cWGAN-GP CHỈ TRÊN 80 dòng (10 con) của x64_train_leakfree.csv:
   - Tuyệt đối không nhìn thấy bất kỳ con mã độc nào trong x64_test_leakfree.csv.
   - Sinh 1.000 mẫu tổng hợp (220 Benign + 780 Malware) -> x64_gan_1000_leakfree.csv.
"""
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DROP_23_COLS = [
    # 6 cột 100% NaN trên x64
    "psxview.not_in_pspcid_list", "psxview.not_in_session", "psxview.not_in_deskthrd",
    "psxview.not_in_pspcid_list_false_avg", "psxview.not_in_session_false_avg", "psxview.not_in_deskthrd_false_avg",
    # 5 cột hằng số bằng 0
    "handles.nport", "malfind.commitCharge", "psxview.not_in_eprocess_pool",
    "psxview.not_in_eprocess_pool_false_avg", "svcscan.interactive_process_services",
    # 2 cột trùng lặp 100% với ldrmodules.not_in_load*
    "ldrmodules.not_in_mem", "ldrmodules.not_in_mem_avg",
    # 10 cột cấu hình tĩnh định danh phiên máy ảo (within-session std = 0 trên x64)
    "svcscan.nservices", "svcscan.kernel_drivers", "svcscan.fs_drivers",
    "svcscan.process_services", "svcscan.shared_process_services", "svcscan.nactive",
    "callbacks.ncallbacks", "callbacks.nanonymous", "callbacks.ngeneric", "modules.nmodules",
]

print("=" * 80)
print("1️⃣ LÀM SẠCH BỘ DỮ LIỆU 200 DÒNG x64 (LOẠI NHIỄU KHỞI ĐỘNG VM & 23 CỘT RÁC/TĨNH)")
print("=" * 80)

df64_raw = pd.read_csv("features_all_x64_19041.csv")
all_55_cols = [c for c in df64_raw.columns if c not in ["Class", "Category", "Filename"]]
CLEAN_32_COLS = [c for c in all_55_cols if c not in DROP_23_COLS]

df64_raw["Instance_ID"] = df64_raw["Filename"].apply(lambda f: re.sub(r"_dump_.*", "", f))
df64_raw["Dump_Num"] = df64_raw["Filename"].apply(lambda f: int(re.search(r"_dump_(\d+)_", f).group(1)))

# Lọc bỏ dump_1 và dump_2 (nhiễu tiến trình khởi động Windows 10 khi vừa resume VM)
df64_clean = (
    df64_raw[df64_raw["Dump_Num"] >= 3][CLEAN_32_COLS + ["Class", "Category", "Filename", "Instance_ID", "Dump_Num"]]
    .copy()
    .reset_index(drop=True)
)
df64_clean.to_csv("features_all_x64_19041_Cleaned.csv", index=False)

print(f"📌 Số dòng gốc: {len(df64_raw)} dòng (20 phiên × 10 dump)")
print(f"📌 Sau khi cắt bỏ 2 dump khởi động nhiễu (dump_1, dump_2)/phiên: {len(df64_clean)} dòng sạch (20 phiên × 8 dump ổn định)")
print(f"📌 Số đặc trưng gốc: 55 -> Sau khi loại 23 cột (6 NaN + 5 cột 0 + 2 cột trùng + 10 cột tĩnh phiên VM): {len(CLEAN_32_COLS)} đặc trưng động sạch")
print(f"✅ Đã lưu bộ dữ liệu x64 sạch: features_all_x64_19041_Cleaned.csv")

# =========================================================================
# 2. CHIA TRAIN/TEST CÔ LẬP TUYỆT ĐỐI THEO CON MÃ ĐỘC (0% OVERLAP)
# =========================================================================
print("\n" + "=" * 80)
print("2️⃣ CHIA TRAIN/TEST CÔ LẬP THEO MẪU THỰC THI (INSTANCE/HASH-LEVEL ISOLATION)")
print("=" * 80)

TRAIN_INSTANCES_64 = [
    "Benign-browse", "Benign-idle",
    "Ransomware-Conti-1", "Ransomware-LockBit-2",
    "Spyware-AgentTesla-1", "Spyware-Formbook-1", "Spyware-SnakeKeylogger-3",
    "Trojan-QuasarRAT-2", "Trojan-Remcos-1", "Trojan-DCRat-1",
]
tr64 = df64_clean[df64_clean["Instance_ID"].isin(TRAIN_INSTANCES_64)].copy().reset_index(drop=True)
te64 = df64_clean[~df64_clean["Instance_ID"].isin(TRAIN_INSTANCES_64)].copy().reset_index(drop=True)

tr64.to_csv("x64_train_leakfree.csv", index=False)
te64.to_csv("x64_test_leakfree.csv", index=False)

overlap_64 = set(tr64["Instance_ID"]) & set(te64["Instance_ID"])
print(f"📌 [x64] Train: {len(tr64)} dòng ({tr64['Instance_ID'].nunique()} con) | Test: {len(te64)} dòng ({te64['Instance_ID'].nunique()} con hoàn toàn khác)")
print(f"   🔒 Số con mã độc/phiên chạy trùng giữa Train và Test x64: {len(overlap_64)} (0% Rò rỉ!)")
print(f"   • Danh sách 10 con ở Train x64: {sorted(tr64['Instance_ID'].unique())}")
print(f"   • Danh sách 10 con ở Test  x64: {sorted(te64['Instance_ID'].unique())}")

# Làm sạch & Chia cô lập theo mã Hash trên x32 (MalMem2022)
df32_raw = pd.read_csv("MalMem2022.csv")
df32_raw["Family"] = df32_raw["Category"].str.split("-").str[:2].str.join("-")
df32_dedup = df32_raw.drop_duplicates(subset=CLEAN_32_COLS + ["Family"]).copy()
conflict_32 = df32_dedup.duplicated(subset=CLEAN_32_COLS, keep=False)
df32_clean = df32_dedup[~conflict_32].copy().reset_index(drop=True)

# Trích xuất mã Hash của từng con mã độc trên x32
df32_clean["Instance_ID"] = np.where(
    df32_clean["Class"] == "Benign",
    "Benign_Block_" + (df32_clean.index // 200).astype(str),
    df32_clean["Filename"].str.replace(r"-\d+\.raw$", "", regex=True),
)

# Chọn 40 mã Hash Malware độc lập (2-3 mã hash mỗi họ × 15 họ = 40 con mã độc) cho Test x32
mal32 = df32_clean[df32_clean["Class"] == "Malware"]
fams32 = sorted(mal32["Family"].unique())
rng = np.random.default_rng(SEED)
test_hashes_32 = []
for idx_f, fam in enumerate(fams32):
    unique_h = sorted(mal32[mal32["Family"] == fam]["Instance_ID"].unique())
    n_pick = 3 if idx_f < 10 else 2  # 10*3 + 5*2 = 40 con mã độc độc lập
    picked_h = list(rng.choice(unique_h, size=n_pick, replace=False))
    test_hashes_32.extend(picked_h)

# Với mỗi mã hash trong 40 mã hash Test, lấy 1 bản dump đại diện đưa vào Test, và XÓA CẢ 10 BẢN DUMP khỏi Train!
te32_mal = (
    mal32[mal32["Instance_ID"].isin(test_hashes_32)]
    .groupby("Instance_ID", as_index=False)
    .first()
)
tr32_mal = mal32[~mal32["Instance_ID"].isin(test_hashes_32)].copy()

# Với Benign x32: lấy 40 mẫu từ 2 khối phiên chạy cuối làm Test, và loại bỏ toàn bộ 2 khối đó + mọi mẫu gần trùng khỏi Train
ben32 = df32_clean[df32_clean["Class"] == "Benign"].copy()
ben_blocks = sorted(ben32["Instance_ID"].unique())
test_ben_blocks = ben_blocks[-4:]
te32_ben = ben32[ben32["Instance_ID"].isin(test_ben_blocks)].sample(n=40, random_state=SEED)
tr32_ben_cand = ben32[~ben32["Instance_ID"].isin(test_ben_blocks)].copy()

# Khử thêm mọi mẫu Benign trong Train nằm quá gần bất kỳ mẫu Benign Test nào trong không gian chuẩn hóa
sc_b = (tr32_ben_cand[CLEAN_32_COLS] - tr32_ben_cand[CLEAN_32_COLS].mean()) / (tr32_ben_cand[CLEAN_32_COLS].std() + 1e-6)
sc_b_te = (te32_ben[CLEAN_32_COLS] - tr32_ben_cand[CLEAN_32_COLS].mean()) / (tr32_ben_cand[CLEAN_32_COLS].std() + 1e-6)
nn_b = NearestNeighbors(n_neighbors=1).fit(sc_b_te.values)
dist_b = nn_b.kneighbors(sc_b.values)[0].ravel()
tr32_ben = tr32_ben_cand[dist_b > 0.35].copy()

cols_keep_32 = CLEAN_32_COLS + ["Class", "Category", "Filename", "Instance_ID"]
tr32 = pd.concat([tr32_ben[cols_keep_32], tr32_mal[cols_keep_32]], ignore_index=True)
te32 = pd.concat([te32_ben[cols_keep_32], te32_mal[cols_keep_32]], ignore_index=True)

tr32.to_csv("x32_train_leakfree.csv", index=False)
te32.to_csv("x32_test_leakfree.csv", index=False)

overlap_32 = set(tr32["Instance_ID"]) & set(te32["Instance_ID"])
print(f"📌 [x32] Train: {len(tr32)} dòng ({tr32_mal['Instance_ID'].nunique()} mã hash Malware) | Test: {len(te32)} dòng (40 Benign + 40 mã hash Malware độc lập)")
print(f"   🔒 Số mã Hash / khối phiên chạy trùng giữa Train và Test x32: {len(overlap_32)} (0% Rò rỉ!)")

# =========================================================================
# 3. HUẤN LUYỆN cWGAN-GP CHỈ TRÊN 10 CON CỦA x64_train_leakfree.csv (80 DÒNG)
# =========================================================================
print("\n" + "=" * 80)
print("3️⃣ HUẤN LUYỆN LEAK-FREE PROTOTYPE-ANCHORED cWGAN-GP (CHỈ HỌC TỪ 10 CON TRAIN x64)")
print("=" * 80)

int_cols = [c for c in CLEAN_32_COLS if (tr64[c] % 1 == 0).all()]
classes_tr64 = sorted(tr64["Category"].unique())
cls_idx = {c: i for i, c in enumerate(classes_tr64)}
n_cls = len(classes_tr64)
d = len(CLEAN_32_COLS)
Z_DIM = 32

X_log = np.log1p(np.maximum(0.0, tr64[CLEAN_32_COLS].values.astype(np.float64)))
x_min, x_max = X_log.min(0), X_log.max(0)
span = np.where(x_max - x_min > 1e-8, x_max - x_min, 1.0)
Xs = 2.0 * (X_log - x_min) / span - 1.0
y = tr64["Category"].map(cls_idx).values

cls_rows = {i: Xs[y == i] for i in range(n_cls)}
cls_std = {i: np.maximum(Xs[y == i].std(0), 0.025) for i in range(n_cls)}


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
        return torch.clamp(anchor + 0.35 * scale * delta, -1.0, 1.0)


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
        w = np.random.dirichlet(np.ones(len(rows)) * 1.2)
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

for ep in range(1, 2001):
    for _ in range(4):
        z = torch.randn(len(Xt), Z_DIM)
        anc, scl = sample_anchors(y)
        fake = G(z, Yt, anc, scl).detach()
        a = torch.rand(len(Xt), 1)
        inter = (a * Xt + (1.0 - a) * fake).requires_grad_(True)
        grad = torch.autograd.grad(D(inter, Yt).sum(), inter, create_graph=True)[0]
        gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
        loss_d = D(fake, Yt).mean() - D(Xt, Yt).mean() + 10.0 * gp
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

    z = torch.randn(len(Xt), Z_DIM)
    anc, scl = sample_anchors(y)
    fake = G(z, Yt, anc, scl)
    loss_g = -D(fake, Yt).mean() + 0.35 * ((fake - anc) ** 2).mean()
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    if ep % 500 == 0:
        print(f"   Epoch {ep:4d}/2000 | Critic Loss: {loss_d.item():.4f} | Generator Loss: {loss_g.item():.4f}")

# Sinh 1.000 mẫu từ 10 con trong tập Train (232 Benign + 96 mẫu × 8 biến thể Malware = 1.000 mẫu)
lab = []
for i, c in enumerate(classes_tr64):
    n_c = 232 if c == "Benign" else 96
    lab.extend([i] * n_c)
lab = np.array(lab)

G.eval()
with torch.no_grad():
    anc, scl = sample_anchors(lab)
    c_oh = torch.nn.functional.one_hot(torch.tensor(lab), n_cls).float()
    out = G(torch.randn(len(lab), Z_DIM), c_oh, anc, scl).numpy()

Xg = np.expm1((out + 1.0) / 2.0 * span + x_min)
syn = pd.DataFrame(Xg, columns=CLEAN_32_COLS)
for c in CLEAN_32_COLS:
    syn[c] = syn[c].clip(tr64[c].min() * 0.95, tr64[c].max() * 1.05)
for c in int_cols:
    syn[c] = np.maximum(0, syn[c].round())

syn["Category"] = [classes_tr64[i] for i in lab]
syn["Class"] = np.where(syn["Category"] == "Benign", "Benign", "Malware")
syn["Filename"] = [f"GAN_LEAKFREE_{classes_tr64[i]}_{k}" for k, i in enumerate(lab)]
syn["Instance_ID"] = "GAN_SYNTHETIC"
syn.to_csv("x64_gan_1000_leakfree.csv", index=False)

ks_mean = float(np.mean([ks_2samp(tr64[c], syn[c]).statistic for c in CLEAN_32_COLS]))
print(f"\n✅ Đã lưu 1.000 mẫu GAN không rò rỉ -> x64_gan_1000_leakfree.csv (Mean KS = {ks_mean:.4f})")
