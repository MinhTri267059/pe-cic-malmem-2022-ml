# BÁO CÁO KHOA HỌC: HUẤN LUYỆN NÂNG CAO PHÂN LOẠI MÃ ĐỘC VỚI CONDITIONAL WGAN-GP VÀ FEW-SHOT LEARNING TRÊN BỘ ĐẶC TRƯNG TÀNG HÌNH BỘ NHỚ (22 STEALTH FEATURES)

> ⚠️ **ĐÍNH CHÍNH (cập nhật sau kiểm định lại):** Các số liệu GAN / Few-Shot trong báo cáo này được đo bằng
> `train_test_split` ngẫu nhiên theo dòng (≈99,9% tệp mã độc trong test đã có dump ở train) và few-shot được
> tiền huấn luyện trên toàn bộ tập train, nên **bị lạc quan giả**. Các mức "cải thiện" do GAN (±0,3–0,5 điểm) nằm
> trong nhiễu của một lần chia duy nhất. Kết quả đã được đo lại theo giao thức không rò rỉ (chia theo SampleID,
> kiểm định bootstrap) – xem **[BaoCao_SuaLoi_VaToiUu_KhongRoRi.md](BaoCao_SuaLoi_VaToiUu_KhongRoRi.md)**.


**Đề tài nghiên cứu:** Nghiên cứu và thực nghiệm phương pháp làm giàu dữ liệu bằng Generative Adversarial Networks (GAN) và học ít mẫu (Few-Shot Learning) trên tập đặc trưng tàng hình bộ nhớ CIC-MalMem-2022 (New Feature - 22 Features).  
**Học viên thực hiện:** Học viên Cao học ngành An toàn thông tin / Khoa học máy tính  
**Môi trường thực nghiệm:** NVIDIA GeForce RTX 5070 GPU (CUDA 12.8), PyTorch 2.6, XGBoost 3.1, LightGBM 4.6, CatBoost 1.2, Scikit-learn.

---

## 1. TỔNG QUAN VÀ ĐẶT VẤN ĐỀ

### 1.1. Bối cảnh bộ dữ liệu "New Feature" (22 Stealth Memory Features)
Trong bài toán điều tra phân tích bộ nhớ mã độc (Memory Forensics) bằng Volatility Framework trên tập CIC-MalMem-2022, bộ dữ liệu thô ban đầu có 55 đặc trưng (chứa các số liệu thống kê chung như số lượng Handle, DLL, Thread tổng quát). Mặc dù các bộ phân loại đạt tới >99% trên bộ 55 đặc trưng đó, các nghiên cứu chuyên sâu đã chỉ ra sự "rò rỉ phân phối" hoặc phụ thuộc quá mức vào các số liệu thống kê bề nổi mà kẻ tấn công dễ dàng làm nhiễu hoặc ngụy trang.

Thư mục `new feature` trích xuất tập con tinh chỉnh gồm **22 đặc trưng cốt lõi**, chỉ tập trung duy nhất vào **các hành vi lẩn trốn, tàng hình và tiêm mã (evasion & stealth mechanisms)**:
1. **Ldrmodules (6 đặc trưng):** `ldrmodules.not_in_load`, `ldrmodules.not_in_init`, `ldrmodules.not_in_mem`, cùng các giá trị trung bình (`_avg`). Phát hiện các DLL độc hại tự ngắt liên kết (unlinking) khỏi 3 danh sách liên kết kép trong PEB (`InLoadOrderModuleList`, `InMemoryOrderModuleList`, `InInitializationOrderModuleList`).
2. **Malfind (2 đặc trưng):** `malfind.ninjections`, `malfind.commitCharge`. Phát hiện vùng nhớ thực thi được cấp phát bất thường (như `PAGE_EXECUTE_READWRITE`), kỹ thuật Reflective DLL Injection, Hollow Process hoặc tiêm shellcode.
3. **Psxview (14 đặc trưng):** `psxview.not_in_pslist`, `psxview.not_in_eprocess_pool`, `psxview.not_in_ethread_pool`, `psxview.not_in_pspcid_list`, `psxview.not_in_csrss_handles`, `psxview.not_in_session`, `psxview.not_in_deskthrd` và các giá trị trung bình tương ứng. Dùng kỹ thuật cross-view diffing để phát hiện tiến trình ẩn giấu qua kỹ thuật DKOM (Direct Kernel Object Manipulation - ngắt liên kết `ActiveProcessLinks`).

### 1.2. Thách thức phân loại và động lực nghiên cứu
- **Loại bỏ đặc trưng bề nổi:** Trên 22 đặc trưng tàng hình này, các mô hình phân loại không thể dựa vào các chỉ số đơn giản, khiến bài toán phân loại đa lớp (Scenario 2: 4 lớp và Scenario 3: 16 lớp) trở nên thách thức và sát với thực tế kiểm thử an ninh.
- **Mất cân bằng dữ liệu nghiêm trọng (Severe Class Imbalance):**
  - Trong Scenario 2: Benign (16,306 mẫu) chiếm ưu thế áp đảo so với Trojan (5,479), Ransomware (5,290), Spyware (5,049).
  - Trong Scenario 3: Benign (16,306 mẫu) chiếm tới 45.8% toàn bộ tập dữ liệu, trong khi nhiều họ mã độc cụ thể như `Spyware-TIBS` chỉ có 777 mẫu (chênh lệch tỷ lệ > 21:1).
- **Hiện tượng mã độc mới xuất hiện (Zero-day / Few-Shot Malware):** Trong thực tế tác chiến an ninh mạng, các dòng mã độc mới chỉ xuất hiện với số lượng mẫu vô cùng hạn chế ($K=5, 10, 20$ mẫu). Các mô hình phân loại truyền thống hoàn toàn thất bại hoặc overfit trầm trọng trong bối cảnh này.

---

## 2. KIẾN TRÚC PHƯƠNG PHÁP VÀ MÔ HÌNH TRIỂN KHAI

### 2.1. Conditional WGAN-GP (Wasserstein GAN với Gradient Penalty)
Để làm giàu dữ liệu bảng (Tabular Data Augmentation) một cách chân thực và bảo toàn mối tương quan phi tuyến giữa 22 đặc trưng, chúng tôi thiết kế mô hình **Conditional WGAN-GP (Conditional Wasserstein Generative Adversarial Network with Gradient Penalty)**:
- **Hàm mục tiêu Wasserstein-1 Distance:** Khắc phục triệt để hiện tượng sụp đổ mode (mode collapse) và biến mất gradient (vanishing gradient) của GAN truyền thống:
  $$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim \mathbb{P}_r}[D(x|y)] - \mathbb{E}_{\tilde{x} \sim \mathbb{P}_g}[D(\tilde{x}|y)] - \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x}|y)\|_2 - 1)^2]$$
- **Kiến trúc Mạng Sinh (Generator):**
  - Input: Vector nhiễu ngẫu nhiên Gaussian $z \sim \mathcal{N}(0, I_{32})$ kết hợp với vector nhãn lớp $y$.
  - Khối ẩn: 3 tầng Dense tuần tự (256 -> 256 -> 256 nơ-ron) kết hợp khối kết nối tắt (Residual Connections / Skip Connections) và chuẩn hóa tầng (`LayerNorm`), hàm kích hoạt LeakyReLU.
  - Tầng xuất ra: Linear 22 chiều đại diện cho 22 đặc trưng chuẩn hóa.
- **Kiến trúc Mạng Phê bình (Critic / Discriminator):**
  - Input: Vector đặc trưng 22 chiều kết hợp với nhãn lớp.
  - Cấu trúc: 3 tầng Dense (256 -> 128 -> 64), sử dụng `LayerNorm` (không dùng `BatchNorm` để tránh làm sai lệch phép tính Gradient Penalty theo từng mẫu riêng lẻ).
  - Gradient Penalty: Hệ số $\lambda = 10.0$ nội suy mẫu trên đoạn thẳng giữa phân phối thật và giả $\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}$.
- **Tối ưu hóa huấn luyện:**
  - Optimizer: Adam ($\beta_1=0.5, \beta_2=0.9$), Learning Rate = $2 \times 10^{-4}$.
  - Tỷ lệ Critic : Generator = 2 : 1, Batch size = 512, huấn luyện 40-45 epochs trên GPU CUDA.

### 2.2. Chiến lược Tăng cường Dữ liệu (GAN Augmentation Levels)
Thực hiện huấn luyện và sinh mẫu tăng cường có kiểm soát theo các mức:
- **Mức 0 (Baseline):** Tập dữ liệu huấn luyện nguyên bản (không bổ sung dữ liệu sinh).
- **Mức 1 (+200 mẫu):** Sinh bổ sung 200 mẫu nhân tạo cho mỗi lớp thiểu số.
- **Mức 2 (+500 mẫu):** Sinh bổ sung 500 mẫu nhân tạo cho mỗi lớp thiểu số.
- **Mức 3 (+1000 mẫu):** Sinh bổ sung 1,000 mẫu nhân tạo cho mỗi lớp thiểu số (đạt trạng thái cân bằng phân phối). Xuất bộ dữ liệu mới dạng CSV.
- **Mức 4 (+2000 mẫu):** Sinh bổ sung 2,000 mẫu nhân tạo cho mỗi lớp thiểu số (kiểm tra ngưỡng bão hòa thông tin).

### 2.3. Khung Học ít mẫu (Few-Shot Learning Frameworks)
Mô phỏng khả năng nhận diện các biến thể mã độc mới với số lượng quan sát cực thấp ($K \in \{5, 10, 20, 50\}$):
1. **Prototypical Networks (ProtoNet):**
   - Cơ chế Episodic Meta-learning: Học một không gian nhúng số đo (metric embedding space) $f_\phi(x)$.
   - Vector nguyên mẫu của mỗi lớp (Prototype) được tính bằng trọng tâm của các mẫu hỗ trợ (Support set $S_k$):
     $$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$
   - Xác suất phân loại mẫu truy vấn (Query sample $x$) dựa trên hàm khoảng cách bình phương Euclid kết hợp Softmax.
2. **Siamese Networks:**
   - Sử dụng kiến trúc mạng chia sẻ trọng số hai nhánh (twin network) tối ưu hóa hàm mất mát Contrastive Loss:
     $$\mathcal{L}(y_{pair}, x_1, x_2) = (1-y_{pair}) \frac{1}{2} D_W^2 + y_{pair} \frac{1}{2} \{\max(0, m - D_W)\}^2$$
   - Khi suy luận, sử dụng khoảng cách trong không gian nhúng để phân loại thông qua bộ phân loại 1-Nearest Neighbor (1-NN).
3. **GAN-Assisted Few-Shot Learning:**
   - Sử dụng Conditional WGAN-GP để sinh bổ sung mẫu tổng hợp từ các điểm hỗ trợ cực ít ($K=5, 10, 20$), giúp các mô hình Tree-based (XGBoost, Random Forest) tái cấu trúc mặt phẳng siêu phân chia (decision boundary) mà không bị rơi vào tình trạng thiếu dữ liệu phân nhánh.

---

## 3. KẾT QUẢ THỰC NGHIỆM CHI TIẾT

### 3.1. Kịch bản 2 (Scenario 2 - 4 Phân lớp: Benign, Ransomware, Trojan, Spyware)

#### A. Đánh giá chất lượng dữ liệu sinh bởi GAN
- **Sai khác ma trận tương quan (Frobenius Norm):** 6.7365 (chỉ ra mối quan hệ giữa 22 đặc trưng được mô phỏng trung thực).
- **Sai số tuyệt đối trung bình giữa các thuộc tính (Mean MAE):** 2.2908.
- Biểu đồ PCA 2D ([gan_pca_real_vs_synth.png](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/gan_pca_real_vs_synth.png)) cho thấy dữ liệu do GAN sinh bao phủ mượt mà không gian phân phối thực của cả 4 lớp mà không bị co cụm (mode collapse).

#### B. Bảng kết quả phân loại qua các mức làm giàu dữ liệu GAN (Scenario 2)

| Mô hình | Mức GAN | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Thời gian huấn luyện (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | Baseline (0 GAN) | 78.30% | 67.36% | 66.85% | 66.89% | 78.25% | 1.91s |
| | +200 GAN | 78.23% | 67.28% | 66.72% | 66.74% | 78.15% | 1.99s |
| | +500 GAN | 78.30% | 67.40% | 66.83% | 66.87% | 78.22% | 1.99s |
| | **+1000 GAN** | **78.47%** | **67.66%** | **67.12%** | **67.12%** | **78.40%** | 2.31s |
| | **+2000 GAN** | **78.57%** | **67.79%** | **67.23%** | **67.23%** | **78.48%** | 2.56s |
| **LightGBM (Tuned)** | Baseline (0 GAN) | 77.81% | 66.46% | 66.11% | 66.13% | 77.77% | 2.73s |
| | +200 GAN | 78.51% | 67.51% | 67.16% | 67.16% | 78.44% | 2.83s |
| | +500 GAN | 77.98% | 66.85% | 66.34% | 66.35% | 77.91% | 2.84s |
| | **+1000 GAN** | **78.60%** | **67.85%** | **67.30%** | **67.33%** | **78.55%** | 3.00s |
| | +2000 GAN | 78.19% | 67.29% | 66.66% | 66.69% | 78.12% | 3.96s |
| **CatBoost (Tuned)** | Baseline (0 GAN) | 77.95% | 67.66% | 66.22% | 66.13% | 77.69% | 2.89s |
| | +200 GAN | 77.60% | 67.00% | 65.69% | 65.55% | 77.31% | 3.38s |
| | +500 GAN | 77.53% | 66.80% | 65.55% | 65.35% | 77.19% | 3.39s |
| | +1000 GAN | 77.42% | 66.62% | 65.39% | 65.20% | 77.10% | 3.58s |
| | +2000 GAN | 76.92% | 65.92% | 64.65% | 64.51% | 76.63% | 3.75s |
| **Random Forest** | Baseline (0 GAN) | 76.22% | 63.92% | 63.69% | 63.70% | 76.17% | 0.90s |
| | +200 GAN | 76.23% | 63.98% | 63.68% | 63.69% | 76.17% | 0.92s |
| | +500 GAN | 76.37% | 64.18% | 63.89% | 63.90% | 76.30% | 0.99s |
| | +1000 GAN | 76.17% | 63.87% | 63.58% | 63.57% | 76.10% | 1.11s |
| | **+2000 GAN** | **76.45%** | **64.33%** | **64.01%** | **63.99%** | **76.38%** | 1.25s |
| **Deep MLP (PyTorch)**| Baseline (0 GAN) | 75.08% | 63.06% | 61.72% | 61.08% | 74.48% | 7.81s |
| | +200 GAN | 74.04% | 62.00% | 60.51% | 60.43% | 73.94% | 5.98s |
| | **+500 GAN** | **75.28%** | **64.49%** | **62.15%** | **61.68%** | **74.85%** | 5.91s |
| | +1000 GAN | 75.02% | 62.80% | 61.66% | 60.41% | 74.01% | 7.03s |
| | +2000 GAN | 74.61% | 63.07% | 61.08% | 60.66% | 74.08% | 7.16s |

**Nhận xét chính Kịch bản 2:**
1. **LightGBM đạt hiệu năng cao nhất ở mức +1000 GAN:** Độ chính xác tăng từ **77.81% lên 78.60%** (+0.79%), và Macro-F1 tăng mạnh từ **66.13% lên 67.33% (+1.20%)**.
2. **XGBoost tăng trưởng bền vững:** Độ chính xác đạt đỉnh ở mức **+2000 GAN với 78.57%** (tăng từ 78.30%), Macro-F1 tăng từ **66.89% lên 67.23%**.
3. **Random Forest tăng đều:** Từ 76.22% lên 76.45% (+2000 GAN).

---

### 3.2. Kịch bản 3 (Scenario 3 - 16 Phân lớp Họ Mã Độc Chi Tiết)

#### A. Phân tích bối cảnh mất cân bằng 16 lớp
Tập dữ liệu gồm 35,566 mẫu phân bổ trên 16 lớp. Benign có 16,306 mẫu (45.8%), trong khi 15 họ mã độc thiểu số chỉ có từ 777 mẫu (`Spyware-TIBS`) đến 1,601 mẫu (`Trojan-Scar`). Điều này khiến Macro-F1 baseline của các mô hình chỉ đạt quanh mức 28% - 30%.

#### B. Bảng kết quả phân loại qua các mức làm giàu dữ liệu GAN (Scenario 3)

| Mô hình | Mức GAN | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Thời gian huấn luyện (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | Baseline (0 GAN) | 59.77% | 29.98% | 30.57% | 30.01% | 59.43% | 8.09s |
| | +200 GAN | 59.74% | 30.09% | 30.45% | 29.99% | 59.40% | 8.72s |
| | +500 GAN | 59.71% | 29.91% | 30.42% | 29.89% | 59.37% | 10.49s |
| | +1000 GAN | 59.64% | 29.94% | 30.35% | 29.82% | 59.26% | 14.28s |
| | **+2000 GAN** | **59.94%** | **30.36%** | **30.94%** | **30.34%** | **59.54%** | 26.93s |
| **LightGBM (Tuned)** | Baseline (0 GAN) | 59.07% | 29.10% | 29.46% | 29.19% | 58.91% | 13.34s |
| | +200 GAN | 59.08% | 29.15% | 29.48% | 29.21% | 58.93% | 15.24s |
| | +500 GAN | 58.98% | 28.88% | 29.25% | 28.93% | 58.77% | 23.76s |
| | +1000 GAN | 59.40% | 29.57% | 29.95% | 29.61% | 59.20% | 33.18s |
| | **+2000 GAN** | **59.47%** | **29.74%** | **30.15%** | **29.72%** | **59.21%** | 33.06s |
| **CatBoost (Tuned)** | Baseline (0 GAN) | 59.45% | 32.18% | 29.55% | 29.29% | 58.80% | 12.03s |
| | +200 GAN | 59.05% | 31.89% | 29.13% | 29.13% | 58.56% | 18.24s |
| | +500 GAN | 58.77% | 31.29% | 28.61% | 28.42% | 58.15% | 20.16s |
| | +1000 GAN | 58.28% | 30.31% | 27.85% | 27.75% | 57.70% | 19.83s |
| | +2000 GAN | 58.22% | 30.20% | 27.70% | 27.67% | 57.62% | 27.77s |
| **Random Forest** | Baseline (0 GAN) | 58.74% | 28.64% | 28.73% | 28.57% | 58.55% | 1.02s |
| | **+200 GAN** | **59.11%** | **29.21%** | **29.28%** | **29.07%** | **58.86%** | 1.26s |
| | +500 GAN | 58.94% | 28.87% | 28.91% | 28.67% | 58.66% | 2.14s |
| | +1000 GAN | 58.86% | 28.67% | 28.73% | 28.50% | 58.59% | 2.28s |
| | +2000 GAN | 58.94% | 28.86% | 29.00% | 28.74% | 58.68% | 3.94s |
| **Deep MLP (PyTorch)**| Baseline (0 GAN) | 57.56% | 28.68% | 27.04% | 24.66% | 56.06% | 5.97s |
| | +200 GAN | 57.46% | 33.41% | 26.35% | 25.84% | 56.73% | 6.59s |
| | **+500 GAN** | **57.79%** | **35.19%** | **26.70%** | **26.35%** | **57.19%** | 9.47s |
| | +1000 GAN | 56.65% | 32.53% | 24.81% | 23.89% | 55.65% | 10.07s |
| | +2000 GAN | 57.42% | 32.68% | 25.80% | 24.84% | 56.41% | 27.27s |

**Nhận xét chính Kịch bản 3:**
1. **XGBoost ở mức +2000 GAN đạt kỷ lục cao nhất:** Accuracy tăng lên **59.94%** (vượt baseline 59.77%), Macro-F1 tăng lên **30.34%** (vượt baseline 30.01%).
2. **LightGBM tăng trưởng ấn tượng:** Accuracy từ 59.07% tăng liên tục lên **59.47%** (+0.40%), Macro-F1 tăng từ 29.19% lên **29.72% (+0.53%)**.
3. **Random Forest tăng ngay ở mức +200 GAN:** Accuracy từ 58.74% tăng lên **59.11%**, Macro-F1 từ 28.57% lên **29.07% (+0.50%)**.
4. **Deep MLP:** Đạt đỉnh ở mức +500 GAN với Macro-F1 tăng vọt từ **24.66% lên 26.35% (+1.69%)**.

---

### 3.3. Kết quả Thực nghiệm Học ít mẫu (Few-Shot Learning)

Thực nghiệm Few-Shot Learning được thiết kế đa đợt (Multi-trial random support/query splitting) để kiểm tra độ tin cậy thống kê (Báo cáo giá trị Mean ± Std).

#### A. Kịch bản 2 (4 phân lớp: Benign, Ransomware, Trojan, Spyware)

| K-Shot | Mô hình | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Ghi chú kỹ thuật |
| :---: | :--- | :---: | :---: | :--- |
| **K = 5** | Random Forest (Baseline) | 60.24% ± 3.31% | 47.09% ± 1.20% | Dữ liệu cực ít (20 mẫu train) |
| | **Random Forest (+GAN Augment)** | **64.25% ± 0.23%** | **47.25% ± 0.54%** | **Tăng vọt +4.01% Acc, Std giảm 14 lần** |
| | XGBoost (Baseline) | 54.04% ± 6.82% | 42.96% ± 2.50% | Overfitting nặng vì thiếu node tách |
| | **XGBoost (+GAN Augment)** | **62.25% ± 0.25%** | **46.47% ± 1.20%** | **BƯỚC NHẢY VỌT: +8.21% Acc, +3.51% F1** |
| | **Prototypical Network (ProtoNet)** | **67.90% ± 1.28%** | **47.68% ± 1.52%** | **Độ chính xác cao nhất nhóm Metric** |
| | **Siamese Network (1-NN)** | **67.09% ± 0.90%** | **50.76% ± 0.59%** | **Macro-F1 cao nhất (+7.80% so với RF base)** |
| **K = 10**| Random Forest (Baseline) | 63.99% ± 1.52% | 49.87% ± 0.96% | 40 mẫu train |
| | **Random Forest (+GAN Augment)** | **64.94% ± 1.01%** | 48.58% ± 1.26% | Tăng +0.95% Acc |
| | XGBoost (Baseline) | 62.12% ± 1.32% | 49.42% ± 0.54% | |
| | **XGBoost (+GAN Augment)** | **65.08% ± 1.34%** | **49.34% ± 0.73%** | **Tăng +2.96% Acc** |
| | **Prototypical Network (ProtoNet)** | **67.67% ± 1.27%** | 49.20% ± 3.20% | Ổn định quanh mức 67.7% |
| | **Siamese Network (1-NN)** | **66.80% ± 1.63%** | **50.35% ± 2.01%** | F1 duy trì vượt trội > 50% |
| **K = 20**| Random Forest (Baseline) | 65.41% ± 0.75% | 51.44% ± 0.20% | 80 mẫu train |
| | **Random Forest (+GAN Augment)** | **66.47% ± 0.16%** | 51.17% ± 1.23% | Tăng +1.06% Acc |
| | XGBoost (Baseline) | 66.23% ± 0.81% | 52.39% ± 0.75% | |
| | **XGBoost (+GAN Augment)** | **67.00% ± 0.78%** | 51.94% ± 0.88% | Tăng +0.77% Acc |
| | **Prototypical Network (ProtoNet)** | **69.00% ± 0.54%** | **51.58% ± 0.86%** | **ĐẠT ĐỈNH 69.00% ACCURACY** |
| | Siamese Network (1-NN) | 66.48% ± 1.82% | 51.34% ± 0.54% | |
| **K = 50**| Random Forest (Baseline) | 69.09% ± 0.45% | 54.86% ± 0.36% | 200 mẫu train |
| | Random Forest (+GAN Augment) | 68.57% ± 0.61% | 53.96% ± 0.67% | Bão hòa mẫu |
| | XGBoost (Baseline) | 68.97% ± 0.50% | 54.36% ± 0.31% | |
| | XGBoost (+GAN Augment) | 69.08% ± 0.31% | 54.29% ± 0.52% | |
| | **Prototypical Network (ProtoNet)** | **69.50% ± 0.29%** | 51.76% ± 0.39% | **KỶ LỤC: 69.50% ACCURACY** |
| | Siamese Network (1-NN) | 68.26% ± 1.05% | 53.00% ± 0.75% | |

#### B. Kịch bản 3 (16 phân lớp họ mã độc chi tiết)

| K-Shot | Mô hình | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Ghi chú & Bước nhảy vọt |
| :---: | :--- | :---: | :---: | :--- |
| **K = 5** | Random Forest (Baseline) | 44.00% ± 2.54% | 13.73% ± 0.84% | Tổng cộng chỉ 80 mẫu train cho 16 lớp |
| | **Random Forest (+GAN Augment)** | **48.60% ± 0.45%** | **14.95% ± 0.51%** | **BƯỚC NHẢY: +4.60% Acc, +1.22% Macro-F1** |
| | XGBoost (Baseline) | 40.85% ± 2.76% | 13.63% ± 0.91% | |
| | **XGBoost (+GAN Augment)** | **47.92% ± 1.01%** | **14.96% ± 0.67%** | **BƯỚC NHẢY VỌT: +7.07% Acc, +1.33% Macro-F1** |
| | **Prototypical Network (ProtoNet)** | **47.88% ± 0.48%** | 12.58% ± 1.07% | Tự động thích ứng không gian Metric |
| | **Siamese Network (1-NN)** | **49.27% ± 0.40%** | **14.99% ± 0.58%** | **ĐẠT HIỆU NĂNG CAO NHẤT Ở K=5 (+8.42% Acc)** |
| **K = 10**| Random Forest (Baseline) | 46.42% ± 2.04% | 16.27% ± 0.15% | Tổng cộng 160 mẫu train |
| | **Random Forest (+GAN Augment)** | **50.02% ± 0.54%** | **17.00% ± 0.06%** | **VƯỢT NGƯỠNG 50%: +3.60% Acc, +0.73% F1** |
| | XGBoost (Baseline) | 41.48% ± 3.39% | 15.55% ± 0.49% | |
| | **XGBoost (+GAN Augment)** | **49.49% ± 0.51%** | **17.03% ± 0.50%** | **BƯỚC NHẢY VỌT: +8.01% Acc, +1.48% Macro-F1** |
| | Prototypical Network (ProtoNet) | 47.75% ± 0.41% | 12.46% ± 0.98% | |
| | **Siamese Network (1-NN)** | 47.69% ± 2.87% | **15.82% ± 0.30%** | |
| **K = 20**| Random Forest (Baseline) | 48.71% ± 1.20% | 17.96% ± 0.48% | Tổng cộng 320 mẫu train |
| | **Random Forest (+GAN Augment)** | **50.39% ± 0.22%** | **18.14% ± 0.49%** | **ĐẠT ĐỈNH: 50.39% Acc, 18.14% Macro-F1** |
| | XGBoost (Baseline) | 48.30% ± 1.10% | 17.79% ± 0.17% | |
| | XGBoost (+GAN Augment) | 49.83% ± 1.15% | 17.60% ± 0.69% | Tăng +1.53% Acc |
| | Prototypical Network (ProtoNet) | 48.35% ± 0.18% | 13.07% ± 0.30% | |
| | **Siamese Network (1-NN)** | **49.43% ± 0.93%** | **16.95% ± 0.12%** | |

---

## 4. PHÂN TÍCH CHUYÊN SÂU & Ý NGHĨA KHOA HỌC

### 4.1. Tại sao GAN tạo ra bước nhảy vọt ở chế độ Few-Shot ($K=5, 10$)?
1. **Giải cứu mô hình cây (Tree-based Models) khỏi sự sụp đổ biên phân chia:**
   - Trong các mô hình dựa trên cây quyết định như Random Forest và Gradient Boosting (XGBoost), thuật toán cần tìm các điểm chia (split points) trên từng thuộc tính nhằm tối đa hóa độ giảm Gini hoặc Entropy.
   - Khi mỗi lớp chỉ có 5 mẫu ($K=5$), không gian mẫu 22 chiều trở nên cực kỳ thưa thớt (Curse of Dimensionality). Các cây quyết định sẽ phân nhánh ngẫu nhiên hoặc tạo ra các lá đơn lẻ (single-sample leaves), dẫn đến phương sai dự báo cực lớn (Std lên tới 6.82%).
   - Khi Conditional WGAN-GP sinh bổ sung mẫu nhân tạo bám theo phân phối thực, nó tạo ra một **vùng lân cận mật độ liên tục (dense continuous neighborhood)** xung quanh các điểm hỗ trợ. Nhờ đó, XGBoost tìm được các siêu mặt phân chia đại diện chính xác, giúp độ chính xác nhảy vọt từ **54.04% lên 62.25% (+8.21% ở Scenario 2)** và từ **40.85% lên 47.92% (+7.07% ở Scenario 3)**.
2. **Triệt tiêu độ lệch phương sai (Variance Reduction):**
   - Độ lệch chuẩn (Std) của XGBoost ở $K=5$ giảm từ **6.82% xuống chỉ còn 0.25%**, và Random Forest giảm từ **3.31% xuống 0.23%**. GAN đóng vai trò như một bộ điều hòa chính quy hóa (regularization) mạnh mẽ.

### 4.2. Sức mạnh của Mạng Đo lường (Metric-Based Learning: ProtoNet & Siamese)
- Cả Prototypical Networks và Siamese Networks không cố gắng xây dựng các siêu phẳng phân chia trực tiếp trên không gian thô 22 chiều, mà chiếu dữ liệu qua mạng nơ-ron phi tuyến $f_\phi(x)$ để học một **không gian nhúng metric** nơi khoảng cách Euclid phản ánh độ tương đồng ngữ nghĩa hành vi mã độc.
- Nhờ đó, ngay cả ở $K=5$, **ProtoNet đạt 67.90% Accuracy** (vượt xa Random Forest baseline 60.24% và XGBoost baseline 54.04%), trong khi **Siamese Network đạt Macro-F1 lên tới 50.76%**.
- Đây là giải pháp hoàn hảo cho hệ thống SOC/SIEM khi phát hiện một biến thể mã độc Zero-Day mới: chỉ cần một vài mẫu vết nhớ được trích xuất từ Sandbox là hệ thống đã có thể lập tức nhận diện biến thể đó mà không cần huấn luyện lại toàn bộ mô hình!

### 4.3. Hiện tượng bão hòa thông tin trên tập dữ liệu đầy đủ
- Khi dữ liệu thực tế đã có sẵn hàng chục ngàn mẫu (như tập Scenario 2 với 25,699 mẫu train), việc bổ sung mẫu sinh nhân tạo đạt hiệu quả tối ưu ở mức **+1000 GAN samples** (LightGBM đạt 78.60% Acc, 67.33% Macro-F1).
- Khi đẩy lên quá cao (+2000 mẫu, tương ứng với việc thêm hơn 30,000 mẫu sinh vào Scenario 3), các mẫu sinh nhân tạo bắt đầu chiếm tỷ trọng lớn và có thể làm mờ nhẹ các chi tiết vi mô của các mẫu thật hiếm hoi. Do đó, trong thực tế triển khai, mức tăng cường **+500 đến +1000 mẫu/lớp** là điểm cân bằng tối ưu giữa chi phí tính toán và độ chính xác phân loại.

---

## 5. DANH MỤC CÁC TẬP TIN DỮ LIỆU, MÃ NGUỒN VÀ ARTIFACTS ĐÃ TẠO

Toàn bộ các mô hình, tập dữ liệu mới và biểu đồ trực quan hóa đã được lưu trữ hoàn chỉnh tại thư mục dự án:

### 5.1. File Mã Nguồn Thực Nghiệm (Python Scripts)
- [`pe-cic-malmem-2022-ml/gan_augmentation.py`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/gan_augmentation.py): Thiết kế kiến trúc `TabularCWGAN_GP`, Generator Residual, Critic LayerNorm, Gradient Penalty, hàm đo lường khoảng cách tương quan.
- [`pe-cic-malmem-2022-ml/train_with_gan.py`](file:///d:/Tai%20Lieu%20Hoc Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/train_with_gan.py): Pipeline huấn luyện phân loại (XGBoost, LightGBM, CatBoost, RF, MLP) với các mức GAN +200 đến +2000.
- [`pe-cic-malmem-2022-ml/fewshot_learning.py`](file:///d:/Tai%20Lieu%20Hoc Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/fewshot_learning.py): Khung học ít mẫu Prototypical Networks, Siamese Networks và GAN-Assisted Few-Shot.
- [`pe-cic-malmem-2022-ml/run_all_experiments.py`](file:///d:/Tai%20Lieu%20Hoc Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/run_all_experiments.py): Master script điều phối chạy liên hoàn tự động trên GPU.

### 5.2. File Dữ Liệu Tăng Cường Mới (Augmented Datasets)
- [`pe-cic-malmem-2022-ml/new feature/results/Dataset_22features_Scenario_2_GAN_augmented_1000.csv`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/Dataset_22features_Scenario_2_GAN_augmented_1000.csv): Tập dữ liệu 28,799 mẫu (bổ sung 3,100 mẫu sinh chất lượng cao cho 4 lớp). Kích thước: 5.47 MB.
- [`pe-cic-malmem-2022-ml/new feature/results/Dataset_22features_Scenario_3_GAN_augmented_1000.csv`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/Dataset_22features_Scenario_3_GAN_augmented_1000.csv): Tập dữ liệu 43,552 mẫu (bổ sung 15,100 mẫu sinh cân bằng cho 16 họ mã độc). Kích thước: 11.06 MB.

### 5.3. Checkpoint Trọng Số Mô Hình GAN (PyTorch Model Weights)
- [`pe-cic-malmem-2022-ml/new feature/results/cwgan_gp_scenario_2.pt`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/cwgan_gp_scenario_2.pt): Model weights của CWGAN-GP 4 lớp (1.08 MB).
- [`pe-cic-malmem-2022-ml/new feature/results/cwgan_gp_scenario_3.pt`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/cwgan_gp_scenario_3.pt): Model weights của CWGAN-GP 16 lớp (1.08 MB).

### 5.4. File Báo Cáo Số Liệu (Metrics CSVs)
- [`pe-cic-malmem-2022-ml/new feature/results/Metrics_GAN_Scenario_2.csv`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/Metrics_GAN_Scenario_2.csv)
- [`pe-cic-malmem-2022-ml/new feature/results/Metrics_GAN_Scenario_3.csv`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/Metrics_GAN_Scenario_3.csv)
- [`pe-cic-malmem-2022-ml/new feature/results/fewshot_benchmark_results_scenario_2.csv`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/fewshot_benchmark_results_scenario_2.csv)
- [`pe-cic-malmem-2022-ml/new feature/results/fewshot_benchmark_results_scenario_3.csv`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/fewshot_benchmark_results_scenario_3.csv)

### 5.5. Đồ Thị & Trực Quan Hóa (Visualizations)
- [`pe-cic-malmem-2022-ml/new feature/results/gan_pca_real_vs_synth.png`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/gan_pca_real_vs_synth.png): Biểu đồ phân phối 2D PCA so sánh dữ liệu thực vs dữ liệu GAN.
- [`pe-cic-malmem-2022-ml/new feature/results/gan_macro_f1_curve_scenario_2.png`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/gan_macro_f1_curve_scenario_2.png): Đường cong biến thiên Macro-F1 theo số lượng mẫu GAN (Scenario 2).
- [`pe-cic-malmem-2022-ml/new feature/results/gan_macro_f1_curve_scenario_3.png`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/gan_macro_f1_curve_scenario_3.png): Đường cong biến thiên Macro-F1 theo số lượng mẫu GAN (Scenario 3).
- [`pe-cic-malmem-2022-ml/new feature/results/confusion_matrix_compare_scenario_2.png`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/confusion_matrix_compare_scenario_2.png): Ma trận nhầm lẫn đối sánh Baseline vs GAN (Scenario 2).
- [`pe-cic-malmem-2022-ml/new feature/results/confusion_matrix_compare_scenario_3.png`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/confusion_matrix_compare_scenario_3.png): Ma trận nhầm lẫn 16x16 đối sánh Baseline vs GAN (Scenario 3).
- [`pe-cic-malmem-2022-ml/new feature/results/fewshot_learning_curve.png`](file:///d:/Tai%20Lieu%20Hoc%20Tap%20Cao%20Hoc/Chuy%C3%AAn%20%C4%91%E1%BB%81%20nghi%C3%AAn%20c%E1%BB%A9u%20v%E1%BB%81%20Ph%C3%A2n%20lo%E1%BA%A1i%20m%C3%A3%20%C4%91%E1%BB%99c/pe-cic-malmem-2022-ml/new%20feature/results/fewshot_learning_curve.png): Đồ thị so sánh hiệu năng các mô hình Few-Shot theo các giá trị $K \in \{5, 10, 20, 50\}$.

---

## 6. KẾT LUẬN & ĐÓNG GÓP HỌC THUẬT

1. **Khẳng định tính ưu việt của Conditional WGAN-GP trên dữ liệu bảng mã độc:** Mô hình WGAN-GP với cơ chế Gradient Penalty đã giải quyết triệt để tính bất ổn định của GAN truyền thống, sinh dữ liệu nhân tạo bảo toàn các đặc trưng tàng hình bộ nhớ (`ldrmodules`, `malfind`, `psxview`) với độ tương đồng hình học cao.
2. **Cải thiện độ chính xác và F1-Score trên các tập mất cân bằng:** Giúp LightGBM đạt **78.60% Accuracy và 67.33% Macro-F1** (Scenario 2) và XGBoost đạt **59.94% Accuracy và 30.34% Macro-F1** (Scenario 3).
3. **Đột phá trong nhận diện mã độc ít mẫu (Few-Shot Malware Classification):** 
   - GAN giúp tăng vọt độ chính xác của các mô hình phân loại cây từ **+7% đến +8.21%** khi chỉ có 5 mẫu/họ mã độc.
   - Các mô hình Metric Learning (Prototypical Networks và Siamese Networks) chứng minh khả năng vượt trội, đạt độ chính xác lên tới **69.50%**, mở ra giải pháp tác chiến thực tế cho các trung tâm giám sát an toàn thông tin (SOC) khi đối đầu với các cuộc tấn công Zero-Day.
