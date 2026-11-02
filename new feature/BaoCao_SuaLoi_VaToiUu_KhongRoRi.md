# BÁO CÁO SỬA LỖI, KIỂM ĐỊNH LẠI & TỐI ƯU ĐỘ CHÍNH XÁC (GIAO THỨC KHÔNG RÒ RỈ)

**Dữ liệu:** CIC-MalMem-2022 – 58.596 dòng (29.298 Benign, 29.298 Malware từ **2.887 tệp mã độc**, ~10 dump bộ nhớ/tệp).
**Giao thức chung:** mọi phép chia train/test dùng `StratifiedGroupKFold` theo **SampleID** (hash tệp mã độc) → 10 dump của
cùng một tệp chỉ nằm ở train **hoặc** test. Scaler/GAN/encoder chỉ fit trên train (few-shot: chỉ trên support).
Toàn bộ số liệu dưới đây được tái tạo bằng `python3 run_all_experiments.py` + `python3 leakfree_benchmark.py --quick`.

---

## 1. Các lỗi đã phát hiện và sửa

| # | File | Lỗi | Hậu quả | Cách sửa |
|---|------|-----|---------|----------|
| 1 | `train_with_gan.py` | Chia `train_test_split` ngẫu nhiên theo dòng – **giải pháp group-split trong báo cáo rò rỉ chưa được áp dụng vào pipeline chính** | 99,9% tệp mã độc trong test đã có dump ở train → mọi số liệu GAN bị lạc quan giả | Chia theo SampleID (`common.group_train_test_split`), assert không trùng nhóm |
| 2 | `fewshot_learning.py` | ProtoNet/Siamese được tiền huấn luyện trên **toàn bộ** tập train (hàng chục nghìn mẫu có nhãn của chính các lớp cần phân loại); GAN cũng học trên toàn bộ train | "Few-shot" thực chất là học đầy đủ → kết quả ProtoNet 67–69% không phản ánh K mẫu | Mọi phương pháp chỉ thấy đúng K mẫu/lớp; GAN huấn luyện lại trên support mỗi lượt; K mẫu lấy từ K tệp khác nhau |
| 3 | `eval_zeroday_fewshot_leakfree.py` | Support/query chọn ngẫu nhiên theo dòng trong lớp novel | Query chứa dump khác của **chính tệp** trong support → rò rỉ snapshot | Support = K tệp khác nhau; loại toàn bộ dump của các tệp đó khỏi query |
| 4 | `eval_zeroday_fewshot_leakfree.py` | Chỉ một bộ novel cố định (Conti/TIBS/Zeus) | Kết luận phụ thuộc may rủi | Xoay vòng 5 bộ novel (mỗi họ làm novel đúng 1 lần) |
| 5 | Tất cả script mới | Đường dẫn tuyệt đối `D:\Tai Lieu Hoc Tap Cao Hoc\...`; `generate_summary_pdf.py` dùng `C:/Windows/Fonts` | Không chạy được trên máy khác / Linux / macOS | `common.py` dùng đường dẫn tương đối; tự dò font |
| 6 | (thiếu) | Không có script nào sinh `Dataset_22features*.csv` | Pipeline không tái lập được | `common.build_22feature_datasets()` tự sinh từ `MalMem2022.csv` |
| 7 | `gan_augmentation.py` | `torch.load` mặc định `weights_only=True` (torch ≥ 2.6) → lỗi khi nạp checkpoint chứa `StandardScaler` | `load_checkpoint` không dùng được | `weights_only=False` |
| 8 | `gan_augmentation.py` | `DataLoader(drop_last=True)` với tập < batch_size → 0 batch | GAN không học gì trên support few-shot, loss = NaN | `batch_size = min(batch, n)` |
| 9 | `gan_augmentation.py` | Mẫu sinh chỉ chặn dưới 0: cột tỷ lệ `*_avg` có thể > 1, cột đếm có giá trị thập phân | Mẫu GAN phi vật lý | Chặn `[0, max_train]`, `*_avg` ∈ [0,1], làm tròn cột đếm |
| 10 | `gan_augmentation.py` | Critic có Dropout (sai với WGAN-GP: GP không còn ước lượng đúng ràng buộc Lipschitz); generator lấy nhãn đều trong khi critic thấy nhãn thật | Huấn luyện kém ổn định | Bỏ Dropout ở critic, generator dùng nhãn của batch thật, `n_critic=5` |
| 11 | `fewshot_learning.py` | Episode sampling `replace=True` → support và query có thể trùng **cùng một mẫu** | Accuracy episode bị thổi phồng | Hoán vị không lặp; lớp nhỏ thì chia đôi support/query |
| 12 | Tất cả | Không đặt seed torch; GAN sinh mẫu không tái lập | Kết quả dao động mỗi lần chạy | `common.set_seed`, `generate(seed=...)` |
| 13 | `train_with_gan.py` | Code không ghi `Is_Synthetic`/`Split_Role` nhưng CSV đã commit lại có | Code và kết quả không khớp | Ghi đủ provenance, kèm cả phần test (`Split_Role=test`) |
| 14 | `train_with_gan.py` | Chỉ một lần chia, không kiểm định thống kê; chênh lệch ±0,3 điểm được báo là "bước nhảy vọt" | Kết luận sai | Bootstrap 95% CI theo nhóm SampleID cho ΔMacro-F1 + metric TSTR |
| 15 | `compare_leak_vs_leakfree.py` | `GroupShuffleSplit` không phân tầng, 1 lần chia | Tỷ lệ lớp test lệch, phương sai lớn | `StratifiedGroupKFold` 5-fold, báo cáo mean ± std |

---

## 2. Kiểm định rò rỉ (`leakage_analysis.py`)

| Chỉ số | Giá trị |
|---|---|
| Dòng trùng lặp trên 22 đặc trưng | 32.505 / 58.596 (55,47%) – chỉ 29.935 vector duy nhất |
| Vector có > 1 nhãn (16 lớp) | 2.506 |
| Tệp mã độc test đã có dump ở train – **Naive split** | **99,92%** (2.522/2.524) |
| Tệp mã độc test đã có dump ở train – **Group split** | **0,00%** (0/578) |
| **Trần độ chính xác lý thuyết (Bayes ceiling) với 22 đặc trưng** | KB1 100% · **KB2 91,56%** · **KB3 84,08%** |
| Trần lý thuyết với 52 đặc trưng đầy đủ | KB2 99,98% · KB3 99,97% |

**Ý nghĩa:** với 22 đặc trưng tàng hình, rất nhiều dump có vector toàn 0 hoặc giống hệt nhau giữa các họ mã độc ⇒ dù mô
hình hoàn hảo cũng không vượt 84% ở KB3. **Nút thắt là thông tin của tập đặc trưng, không phải mô hình hay lượng dữ liệu** —
đây là lý do GAN không thể giúp (GAN không tạo thêm thông tin phân biệt).

### Naive vs Group split (22 đặc trưng, 5-fold)

| Kịch bản | Mô hình | Naive (rò rỉ) Acc / F1 | Group (không rò rỉ) Acc / F1 | Độ lạc quan giả |
|---|---|---|---|---|
| KB2 | XGBoost | 82,94% / 74,41% | 80,88% / 71,32% | +2,1 / +3,1 điểm |
| KB2 | Random Forest | 82,66% / 73,95% | 80,36% / 70,51% | +2,3 / +3,4 |
| KB3 | XGBoost | 69,50% / 42,27% | 66,43% / 36,29% | +3,1 / +6,0 |
| KB3 | Random Forest | 68,91% / 41,54% | 65,55% / 35,00% | +3,4 / +6,5 |

> Ghi chú: số tuyệt đối khác báo cáo cũ (KB3 cũ 59,8%) vì bản cũ `drop_duplicates()` trên 22 đặc trưng trước khi chia
> (bỏ phần lớn dòng Benign trùng → thay đổi phân phối lớp). Bản mới giữ toàn bộ dòng để chia được theo SampleID.

---

## 3. Phương hướng tăng độ chính xác & kết quả (`leakfree_benchmark.py`, 5-fold Group CV)

Các hướng được thử theo thứ tự tác động:

1. **Mở rộng tập đặc trưng 22 → 52** (bỏ 3 cột hằng số trong 55 cột gốc) – tác động lớn nhất.
2. **Đặc trưng kỹ nghệ (30 tỷ lệ):** handles theo loại / tổng handles, dll/handles/injection theo tiến trình, commitCharge/
   protection theo injection, tỷ lệ dịch vụ, tổng ẩn ldrmodules/psxview, callbacks ẩn danh – cải thiện nhỏ nhưng ổn định.
3. **Mô hình:** RF, ExtraTrees, XGBoost, LightGBM, Soft-Voting ensemble – boosting tốt nhất; ensemble không vượt boosting đơn.
4. **Suy luận theo mẫu (sample-level):** trung bình xác suất các dump cùng SampleID (khi phân tích một tệp nghi vấn ta có
   thể chụp RAM nhiều lần) – tăng thêm 2,5–4 điểm Acc, 4–6 điểm Macro-F1.
5. *(Đã thử, không giúp)* tinh chỉnh LightGBM regularization mạnh (1.200 cây, 31 lá, colsample 0,5, λ=5): KB3 71,00% /
   F1 45,87% – không vượt cấu hình mặc định ⇒ mô hình đã bão hòa với đặc trưng hiện có.
6. *(Đã thử, không giúp)* GAN augmentation – xem mục 4.

### Kết quả tốt nhất mỗi cấu hình (Accuracy / Macro-F1, %)

| Kịch bản | 22 tàng hình | 52 đầy đủ | 52 + kỹ nghệ | **+ suy luận theo mẫu** | Naive (rò rỉ, tham chiếu) |
|---|---|---|---|---|---|
| KB1 – 2 lớp | 99,94 / 99,94 | 99,98 / 99,98 | 99,98 / 99,98 | **100,00 / 100,00** (ExtraTrees) | 99,99 |
| KB2 – 4 lớp | 81,01 / 71,51 | 85,43 / 78,07 | 85,54 / 78,25 | **88,24 / 82,31** (RF) · 87,86 / 81,70 (LightGBM) | 88,14 / 82,15 |
| KB3 – 16 lớp | 66,34 / 36,15 | 71,31 / 46,13 | 71,44 / 46,42 | **74,41 / 51,97** (LightGBM) | 76,81 / 56,75 |

So với pipeline 22 đặc trưng ban đầu (cùng giao thức không rò rỉ): **KB2 +7,2 điểm Acc / +10,8 điểm Macro-F1**, **KB3
+8,1 điểm Acc / +15,8 điểm Macro-F1**. Ở KB2, độ chính xác theo mẫu (không rò rỉ) đã ngang mức naive có rò rỉ.

> ⚠️ Lưu ý khi dùng suy luận theo mẫu: Benign trong dataset không có hash nên mỗi dòng Benign là một nhóm riêng.
> **Không** được đưa "số dump trong nhóm" hay thống kê nội nhóm vào làm đặc trưng – mô hình sẽ học được "nhóm 1 dòng =
> Benign" (một tạo tác của cách đặt tên file, không phải hành vi). Cách trung bình xác suất ở đây không dùng thông tin đó.

---

## 4. GAN augmentation dưới giao thức không rò rỉ (`train_with_gan.py`)

Chất lượng GAN: **TSTR** (huấn luyện chỉ trên mẫu GAN, kiểm thử trên dữ liệu thật) = 55,7% vs 71,8% (KB2, tỷ lệ 0,78) và
18,2% vs 36,2% (KB3, tỷ lệ 0,50) ⇒ mẫu sinh chưa tái tạo được phân phối có điều kiện theo họ mã độc.

Macro-F1 (%) theo số mẫu GAN/lớp thiểu số (CI 95% bootstrap theo SampleID cho Δ so với baseline):

| KB | Mô hình | 0 | +200 | +500 | +1000 | +2000 | Kết luận |
|---|---|---|---|---|---|---|---|
| 2 | XGBoost | 72,20 | 72,21 | 72,37 | 72,15 | 72,00 | không khác biệt có ý nghĩa |
| 2 | LightGBM | 71,72 | 71,55 | 71,77 | 72,17 | 71,81 | không có ý nghĩa (+1000: CI [−0,11; +0,92]) |
| 2 | MLP | 66,22 | 66,04 | 65,73 | 65,61 | 65,48 | **giảm có ý nghĩa** từ +500 |
| 3 | XGBoost | 36,20 | 35,82 | 36,15 | 35,93 | 36,18 | không có ý nghĩa |
| 3 | LightGBM | 35,53 | 35,54 | 35,45 | 35,95 | 36,27 | không có ý nghĩa |
| 3 | CatBoost | 33,67 | 32,87 | 32,12 | 31,74 | 31,40 | **giảm có ý nghĩa** ở mọi mức |

**Kết luận:** các "cải thiện" GAN trong báo cáo cũ (±0,3–0,5 điểm trên một lần chia rò rỉ) là nhiễu. GAN không phải
hướng tăng độ chính xác cho bài toán này; mở rộng đặc trưng mang lại gấp ~20 lần.

---

## 5. Few-shot thật sự (mọi phương pháp chỉ thấy K mẫu/lớp, 3 lượt)

| KB | K | RF | XGB | LogReg | RF+GAN | XGB+GAN | ProtoNet | Siamese |
|---|---|---|---|---|---|---|---|---|
| 2 (Acc %) | 5 | 63,2 | 58,2 | 63,0 | 62,6 | 61,4 | **64,2** | 62,4 |
| 2 | 10 | 66,9 | 66,1 | 65,8 | 65,1 | 66,1 | **67,8** | 65,4 |
| 2 | 20 | 65,4 | 65,3 | 65,7 | 66,0 | 66,6 | 66,9 | **67,3** |
| 2 | 50 | 70,2 | 70,4 | 67,5 | 70,9 | **71,4** | 69,7 | 69,1 |
| 3 (Acc %) | 5 | **47,6** | 36,1 | 44,7 | 43,5 | 42,3 | 47,3 | 47,3 |
| 3 | 10 | **54,6** | 53,9 | 54,0 | 54,0 | 51,9 | 53,8 | 54,1 |
| 3 | 20 | **55,8** | 54,1 | 52,8 | 54,4 | 55,5 | 51,6 | 49,7 |

Ghi chú: Macro-F1 KB3 chỉ 14–21% (16 lớp, trần 22 đặc trưng thấp). GAN trên support chỉ giúp XGBoost ở K=5 của KB2
(ổn định cây khi quá ít mẫu); ở KB3 thường làm giảm. ProtoNet/Siamese chỉ nhỉnh hơn RF ở K nhỏ của KB2.

---

## 6. Zero-day (class-disjoint + group-disjoint), 3-way, trung bình 5 bộ novel (ngẫu nhiên = 33,3%)

| Đặc trưng | Phương pháp | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| 22 tàng hình | Nearest-Centroid (thô) | 36,7 | 39,9 | 40,7 | 42,6 |
| 22 tàng hình | ProtoNet (cũ) | **39,1** | 44,0 | 45,4 | 46,4 |
| 22 tàng hình | LogReg (thô) | – | 43,1 | 45,6 | 47,7 |
| 22 tàng hình | **SimpleShot** (encoder CE + CL2N) | 38,9 | **46,5** | **49,3** | **50,5** |
| 52 đầy đủ | ProtoNet | 37,1 | 42,3 | 43,3 | 44,3 |
| 52 đầy đủ | **SimpleShot** | 38,6 | **48,5** | **51,9** | **53,8** |

**Cải tiến:** thay ProtoNet bằng SimpleShot (tiền huấn luyện có giám sát + centering/L2-normalize) và dùng 52 đặc trưng:
K=20 tăng từ 46,4% lên **53,8%** (+7,4 điểm). Bộ novel gốc (Conti/TIBS/Zeus) dễ hơn trung bình, nên số liệu một bộ
cũ không đại diện.

---

## 7. Khuyến nghị tiếp theo

1. **Dùng 52 đặc trưng + đặc trưng kỹ nghệ + suy luận theo mẫu** làm cấu hình chính (KB2 88,2%, KB3 74,4% không rò rỉ).
2. Nếu bắt buộc dùng 22 đặc trưng tàng hình: báo cáo kèm trần Bayes (KB2 91,6%, KB3 84,1%) để đánh giá đúng.
3. Bổ sung đặc trưng mới từ Volatility (vd. `malfind` theo vùng nhớ, `handles` theo loại đối tượng chi tiết, entropy vùng
   inject, chuỗi API) – đây là đòn bẩy duy nhất còn lại cho KB3, vì các mô hình đã bão hòa.
4. Gom nhóm Benign theo phiên/máy khi thu thập dữ liệu mới, để chia nhóm được cho cả Benign.
5. Bỏ GAN khỏi pipeline chính; nếu giữ để nghiên cứu, luôn báo cáo TSTR và CI.
