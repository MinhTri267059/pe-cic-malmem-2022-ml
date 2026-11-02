# -*- coding: utf-8 -*-
"""
Mô-đun Conditional WGAN-GP (Wasserstein GAN với Gradient Penalty)
dành cho dữ liệu bảng 22 đặc trưng trích xuất từ bộ nhớ (CIC-MalMem-2022).
Hỗ trợ sinh dữ liệu tăng cường (Data Augmentation) từ 200 - 2000 mẫu theo từng lớp.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -----------------------------------------------------------------------------
# 1. KIẾN TRÚC MÔ HÌNH CONDITIONAL WGAN-GP
# -----------------------------------------------------------------------------
class TabularGenerator(nn.Module):
    """
    Generator cho dữ liệu bảng:
    Nhận vector nhiễu z (Gaussian) + Class Embedding -> Sinh vector đặc trưng 22D.
    Sử dụng Residual Block và LayerNorm để ổn định đạo hàm.
    """
    def __init__(self, num_features=22, num_classes=4, z_dim=64, embed_dim=16, hidden_dim=256):
        super(TabularGenerator, self).__init__()
        self.z_dim = z_dim
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        in_dim = z_dim + embed_dim
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Residual Block
        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_ln1 = nn.LayerNorm(hidden_dim)
        self.res_act = nn.LeakyReLU(0.2, inplace=True)
        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res_ln2 = nn.LayerNorm(hidden_dim)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, num_features)
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], dim=1)
        h = self.fc1(x)
        h = self.fc2(h)
        # Skip connection
        residual = h
        res = self.res_act(self.res_ln1(self.res_fc1(h)))
        res = self.res_ln2(self.res_fc2(res))
        h = self.res_act(h + res)
        out = self.out(h)
        return out


class TabularCritic(nn.Module):
    """
    Critic (Discriminator) cho WGAN-GP:
    Nhận vector đặc trưng x + Class Embedding -> Đánh giá điểm số Wasserstein không bị chặn.
    Tuân thủ điều kiện 1-Lipschitz thông qua Gradient Penalty (KHÔNG dùng Batch Normalization).
    """
    def __init__(self, num_features=22, num_classes=4, embed_dim=16, hidden_dim=256):
        super(TabularCritic, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        in_dim = num_features + embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        inp = torch.cat([x, c], dim=1)
        score = self.net(inp)
        return score


# -----------------------------------------------------------------------------
# 2. KHUNG HUẤN LUYỆN CONDITIONAL WGAN-GP
# -----------------------------------------------------------------------------
class TabularCWGAN_GP:
    """
    Bộ khung huấn luyện và sinh dữ liệu WGAN-GP có điều kiện (Conditional WGAN-GP).
    """
    def __init__(self, num_features=22, num_classes=4, z_dim=64, embed_dim=16,
                 hidden_dim=256, lr=1.5e-4, lambda_gp=10.0, device=None):
        self.num_features = num_features
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = TabularGenerator(
            num_features, num_classes, z_dim, embed_dim, hidden_dim
        ).to(self.device)
        self.critic = TabularCritic(
            num_features, num_classes, embed_dim, hidden_dim
        ).to(self.device)

        # Adam với beta1=0.5, beta2=0.9 chuẩn cho WGAN
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr, betas=(0.5, 0.9))

        self.scaler = None
        self.feature_names = None
        self.class_prior = None
        self.int_mask = None      # cột đếm (số nguyên) -> làm tròn khi sinh
        self.upper = None         # giới hạn trên quan sát được trên train
        self.training_history = {'critic_loss': [], 'gen_loss': [], 'w_dist': []}

    def compute_gradient_penalty(self, real_samples, fake_samples, labels):
        """Tính toán Gradient Penalty trên các điểm nội suy ngẫu nhiên."""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device).expand_as(real_samples)
        interpolates = (alpha * real_samples + (1.0 - alpha) * fake_samples).requires_grad_(True)

        d_interpolates = self.critic(interpolates, labels)
        grad_outputs = torch.ones_like(d_interpolates, device=self.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
        return gradient_penalty

    def fit(self, X_train, y_train, epochs=40, batch_size=512, n_critic=5, verbose_every=10,
            verbose=True):
        """
        Huấn luyện CWGAN-GP trên tập X_train, y_train.
        Scaler được fit CHỈ trên X_train để tránh rò rỉ dữ liệu.
        """
        X_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(np.asarray(X_train))
        self.feature_names = list(X_df.columns)
        X_np = X_df.values.astype(np.float64)
        y_np = np.asarray(y_train).astype(int)
        self.int_mask = np.all(np.isclose(X_np, np.round(X_np)), axis=0)
        self.upper = X_np.max(axis=0)
        counts = np.bincount(y_np, minlength=self.num_classes).astype(float)
        self.class_prior = counts / counts.sum()

        # Chuẩn hóa StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_np)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        # BUG CŨ: drop_last=True với tập nhỏ (< batch_size, vd. few-shot) -> 0 batch, GAN không học.
        batch_size = int(min(batch_size, len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                drop_last=len(dataset) > batch_size)
        _print = print if verbose else (lambda *a, **k: None)

        _print(f"\n🚀 Khởi động huấn luyện CWGAN-GP trên {self.device.upper()}...", flush=True)
        _print(f"   • Số mẫu train: {len(X_np):,} | Số đặc trưng: {self.num_features} | Số lớp: {self.num_classes}", flush=True)
        _print(f"   • Số Epochs: {epochs} | Batch size: {batch_size} | Critic ratio: {n_critic}:1", flush=True)

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            epoch_c_loss = []
            epoch_g_loss = []
            epoch_w_dist = []

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                cur_batch_size = batch_x.size(0)

                # ---------------------
                # 1. Huấn luyện CRITIC
                # ---------------------
                for _ in range(n_critic):
                    self.opt_c.zero_grad()
                    z = torch.randn(cur_batch_size, self.z_dim, device=self.device)
                    fake_x = self.generator(z, batch_y)

                    real_scores = self.critic(batch_x, batch_y)
                    fake_scores = self.critic(fake_x.detach(), batch_y)

                    gp = self.compute_gradient_penalty(batch_x, fake_x, batch_y)
                    # Wasserstein loss: min -(E[D(real)] - E[D(fake)]) + lambda*GP
                    loss_c = -(torch.mean(real_scores) - torch.mean(fake_scores)) + self.lambda_gp * gp
                    loss_c.backward()
                    self.opt_c.step()

                w_dist = (torch.mean(real_scores) - torch.mean(fake_scores)).item()
                epoch_c_loss.append(loss_c.item())
                epoch_w_dist.append(w_dist)

                # ------------------------
                # 2. Huấn luyện GENERATOR
                # ------------------------
                self.opt_g.zero_grad()
                z = torch.randn(cur_batch_size, self.z_dim, device=self.device)
                # Dùng nhãn của batch thật để generator và critic được huấn luyện trên cùng phân phối điều kiện
                gen_labels = batch_y
                fake_x = self.generator(z, gen_labels)
                fake_scores = self.critic(fake_x, gen_labels)

                loss_g = -torch.mean(fake_scores)
                loss_g.backward()
                self.opt_g.step()
                epoch_g_loss.append(loss_g.item())

            mean_c = np.mean(epoch_c_loss)
            mean_g = np.mean(epoch_g_loss)
            mean_w = np.mean(epoch_w_dist)

            self.training_history['critic_loss'].append(mean_c)
            self.training_history['gen_loss'].append(mean_g)
            self.training_history['w_dist'].append(mean_w)

            if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
                elapsed = time.time() - t0
                _print(f"   [Epoch {epoch:3d}/{epochs:3d}] ({elapsed:.1f}s) -> "
                      f"Critic Loss: {mean_c:+.4f} | Gen Loss: {mean_g:+.4f} | W-Dist: {mean_w:+.4f}", flush=True)

        total_time = time.time() - t0
        _print(f"✅ Hoàn tất huấn luyện CWGAN-GP trong {total_time:.1f}s ({total_time/60:.2f} phút)!", flush=True)

    def generate(self, class_counts_dict, seed=None):
        """
        Sinh mẫu dữ liệu nhân tạo theo số lượng chỉ định cho từng lớp.
        class_counts_dict: dict {class_idx: num_samples_to_generate}
        Trả về: DataFrame (X_synth) và Series (y_synth) ở tỷ lệ gốc ban đầu.
        """
        self.generator.eval()
        if seed is not None:
            torch.manual_seed(seed)
        all_synth_x = []
        all_synth_y = []

        with torch.no_grad():
            for c_idx, count in class_counts_dict.items():
                if count <= 0:
                    continue
                # Sinh theo batch để tránh tràn GPU
                b_size = 512
                remaining = count
                c_samples = []

                while remaining > 0:
                    cur_b = min(remaining, b_size)
                    z = torch.randn(cur_b, self.z_dim, device=self.device)
                    y = torch.full((cur_b,), c_idx, dtype=torch.long, device=self.device)
                    fake_scaled = self.generator(z, y).cpu().numpy()
                    c_samples.append(fake_scaled)
                    remaining -= cur_b

                c_samples = np.vstack(c_samples)
                all_synth_x.append(c_samples)
                all_synth_y.extend([c_idx] * count)

        if not all_synth_x:
            return pd.DataFrame(columns=self.feature_names), pd.Series(dtype=int, name='Label')

        all_synth_x = np.vstack(all_synth_x)
        # Nghịch đảo chuẩn hóa về không gian đặc trưng thực tế
        real_scale_x = self.scaler.inverse_transform(all_synth_x)

        # Ràng buộc miền giá trị hợp lệ: không âm, không vượt max quan sát trên train,
        # cột tỷ lệ '*_avg' nằm trong [0, 1], cột đếm được làm tròn về số nguyên.
        real_scale_x = np.clip(real_scale_x, 0.0, self.upper)
        if self.feature_names is not None:
            for j, name in enumerate(self.feature_names):
                if str(name).endswith('_avg'):
                    real_scale_x[:, j] = np.clip(real_scale_x[:, j], 0.0, 1.0)
        if self.int_mask is not None:
            real_scale_x[:, self.int_mask] = np.round(real_scale_x[:, self.int_mask])

        df_synth_x = pd.DataFrame(real_scale_x, columns=self.feature_names)
        s_synth_y = pd.Series(all_synth_y, name='Label')

        return df_synth_x, s_synth_y

    def save_checkpoint(self, path):
        """Lưu trọng số mô hình và scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'generator_state': self.generator.state_dict(),
            'critic_state': self.critic.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'z_dim': self.z_dim,
            'int_mask': self.int_mask,
            'upper': self.upper,
            'class_prior': self.class_prior,
            'training_history': self.training_history
        }, path)
        print(f"💾 Đã lưu checkpoint GAN tại: {path}")

    def load_checkpoint(self, path):
        """Nạp trọng số mô hình."""
        # weights_only=False: checkpoint chứa StandardScaler (torch>=2.6 mặc định weights_only=True -> lỗi)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt['generator_state'])
        self.critic.load_state_dict(ckpt['critic_state'])
        self.scaler = ckpt['scaler']
        self.feature_names = ckpt['feature_names']
        self.num_features = ckpt['num_features']
        self.num_classes = ckpt['num_classes']
        self.z_dim = ckpt['z_dim']
        self.int_mask = ckpt.get('int_mask')
        self.upper = ckpt.get('upper')
        self.class_prior = ckpt.get('class_prior')
        self.training_history = ckpt.get('training_history', {})
        print(f"📖 Đã nạp checkpoint GAN từ: {path}")


# -----------------------------------------------------------------------------
# 3. HÀM ĐÁNH GIÁ CHẤT LƯỢNG MẪU SINH BẰNG GAN (FID / PCA / CORRELATION)
# -----------------------------------------------------------------------------
def evaluate_gan_quality(X_real, y_real, X_synth, y_synth, label_mapping, output_dir="new feature/results",
                         tag=None):
    """
    Đánh giá chất lượng dữ liệu sinh bằng GAN:
    1. So sánh ma trận tương quan giữa dữ liệu thật và dữ liệu sinh (Frobenius norm diff).
    2. Trực quan hóa phép chiếu PCA 2D Real vs Synthetic.
    3. Thống kê sai lệch trung bình và độ lệch chuẩn.
    """
    os.makedirs(output_dir, exist_ok=True)
    inv_mapping = {v: k for k, v in label_mapping.items()}

    # 1. So sánh sai lệch tương quan
    corr_real = np.corrcoef(X_real.values, rowvar=False)
    corr_synth = np.corrcoef(X_synth.values, rowvar=False)
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_synth = np.nan_to_num(corr_synth, nan=0.0)
    corr_diff_norm = np.linalg.norm(corr_real - corr_synth, 'fro')

    # 2. Thống kê sai lệch giá trị trung bình giữa dữ liệu thật và sinh
    mean_real = X_real.mean(axis=0)
    mean_synth = X_synth.mean(axis=0)
    std_real = X_real.std(axis=0)
    std_synth = X_synth.std(axis=0)
    mean_mae = np.mean(np.abs(mean_real - mean_synth))

    print(f"\n📊 ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG DỮ LIỆU SINH BỞI GAN:")
    print(f"   • Sai khác ma trận tương quan (Frobenius Norm): {corr_diff_norm:.4f}")
    print(f"   • Sai số tuyệt đối trung bình giữa các thuộc tính (Mean MAE): {mean_mae:.4f}")

    # 3. Trực quan hóa PCA 2D
    # PCA trên dữ liệu đã chuẩn hóa (dữ liệu thô bị chi phối bởi cột có biên độ lớn)
    sc = StandardScaler().fit(X_real)
    X_real = pd.DataFrame(sc.transform(X_real), columns=X_real.columns)
    X_synth = pd.DataFrame(sc.transform(X_synth[X_real.columns]), columns=X_real.columns)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_real)
    pca_real = pca.transform(X_real.sample(min(3000, len(X_real)), random_state=42))
    pca_synth = pca.transform(X_synth.sample(min(3000, len(X_synth)), random_state=42))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_real[:, 0], pca_real[:, 1], c='dodgerblue', alpha=0.3, s=15, label='Dữ liệu Thật (Real)')
    plt.title('Dữ liệu Thật trong không gian PCA 2D')
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(pca_synth[:, 0], pca_synth[:, 1], c='crimson', alpha=0.3, s=15, label='Dữ liệu Sinh (GAN Synthetic)')
    plt.title('Dữ liệu GAN Sinh trong không gian PCA 2D')
    plt.xlabel('Thành phần chính 1')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, f"gan_pca_real_vs_synth{'_' + tag if tag else ''}.png")
    plt.savefig(pca_plot_path, dpi=200)
    plt.close()
    print(f"   📈 Đã lưu biểu đồ PCA so sánh tại: {pca_plot_path}")

    return {
        'corr_diff_norm': float(corr_diff_norm),
        'mean_mae': float(mean_mae),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()
    }
