# BÁO CÁO CHUYÊN SÂU: PHÂN TÍCH, KIỂM ĐỊNH VÀ KHẮC PHỤC RÒ RỈ DỮ LIỆU (DATA LEAKAGE AUDIT & REMEDIATION) TRÊN BỘ 22 ĐẶC TRƯNG CIC-MALMEM-2022

> ⚠️ **ĐÍNH CHÍNH (cập nhật sau kiểm định lại):** Các số liệu GAN / Few-Shot trong báo cáo này được đo bằng
> `train_test_split` ngẫu nhiên theo dòng (≈99,9% tệp mã độc trong test đã có dump ở train) và few-shot được
> tiền huấn luyện trên toàn bộ tập train, nên **bị lạc quan giả**. Các mức "cải thiện" do GAN (±0,3–0,5 điểm) nằm
> trong nhiễu của một lần chia duy nhất. Kết quả đã được đo lại theo giao thức không rò rỉ (chia theo SampleID,
> kiểm định bootstrap) – xem **[BaoCao_SuaLoi_VaToiUu_KhongRoRi.md](BaoCao_SuaLoi_VaToiUu_KhongRoRi.md)**.


**Dự án:** Phân loại mã độc trên bộ nhớ RAM (Volatility Memory Forensics) - Tập 22 đặc trưng tàng hình (`ldrmodules`, `malfind`, `psxview`).  
**Tác giả:** Nhóm nghiên cứu Cao học An toàn thông tin.  
**Ngày thực hiện:** 02/11/2026.

---

## 1. TỔNG QUAN VẤN ĐỀ RÒ RỈ DỮ LIỆU TRONG NGHIÊN CỨU MÃ ĐỘC

Trong các nghiên cứu ứng dụng Trí tuệ nhân tạo (AI/ML) vào An toàn thông tin, hiện tượng **Rò rỉ dữ liệu (Data Leakage)** là một trong những nguyên nhân hàng đầu khiến các mô hình đạt độ chính xác ảo (over-optimistic performance) trong phòng thí nghiệm (>95-99%) nhưng suy giảm nghiêm trọng khi triển khai thực tế.

Qua kiểm định toàn diện trên mã nguồn và bộ dữ liệu `new feature` (CIC-MalMem-2022 với 22 đặc trưng tàng hình), chúng tôi đã phát hiện và định lượng **5 điểm rò rỉ dữ liệu tiềm ẩn**, đồng thời thiết kế và kiểm thử thành công các giải pháp khắc phục.

---

## 2. NĂM ĐIỂM RÒ RỈ DỮ LIỆU ĐƯỢC PHÁT HIỆN & ĐỊNH LƯỢNG

```
                             +-------------------------------------------------------------+
                             |           CÁC ĐIỂM RÒ RỈ DỮ LIỆU TIỀM ẨN                   |
                             +-------------------------------------------------------------+
                                      |
         +----------------------------+----------------------------+-----------------------------+
         |                            |                            |                             |
         v                            v                            v                             v
[Điểm 1: Group Leakage]      [Điểm 2: Exact Duplicate]    [Điểm 3: Few-Shot Leakage]   [Điểm 4: Contradiction]
2,671 mẫu mã độc             32,505 hàng trùng nhau       GAN thấy hàng nghìn mẫu      2,506 vector đặc trưng
chụp 10 dump thời gian.      10.5% - 21.8% Test set       nhưng lại dùng để hỗ trợ     giống hệt nhau nhưng
99.92% Test bị rò rỉ!        là bản sao của Train!        cho bài toán 5-shot!         khác Category!
```

---

### 2.1. ĐIỂM RÒ RỈ 1: Rò rỉ Nhóm / Bản chụp bộ nhớ định kỳ (Temporal Snapshot / Group Leakage)
- **Mức độ nghiêm trọng:** 🔴 **RẤT CAO (Critical)**.
- **Bản chất kỹ thuật:**
  Trong quá trình xây dựng tập CIC-MalMem-2022, tác giả thu thập 10 bản chụp bộ nhớ RAM định kỳ (`-1.raw` đến `-10.raw`) cho mỗi lần thực thi mẫu mã độc.
  - Tổng số bản ghi Malware: **29,298 mẫu**.
  - Tổng số mẫu mã độc nhị phân thực tế (Unique SHA256 / Binary ID): **2,906 mẫu**.
  - Có tới **2,671 mẫu mã độc sở hữu đúng 10 bản dump liên tiếp**.
- **Cơ chế rò rỉ:**
  Khi sử dụng hàm chia ngẫu nhiên `train_test_split(df, test_size=0.2)` thông thường:
  - Bản dump `-1` đến `-8` rơi vào tập **Train**, còn `-9`, `-10` rơi vào tập **Test**.
  - Vì các bản dump này xuất phát từ **cùng một tiến trình mã độc chạy trong cùng một máy ảo**, các địa chỉ ngắt DLL (`ldrmodules`), vùng nhớ tiêm mã (`malfind`) và PID ẩn (`psxview`) là hoàn toàn đồng nhất.
  - **Đo đạc thực tế:** **99.92% mẫu mã độc trong Test set đã từng xuất hiện ở tập Train!** Mô hình không học được đặc trưng khái quát của dòng họ mã độc mà chỉ "nhớ mặt" chính tiến trình đó.
- **Thực nghiệm đối chứng định lượng (Scenario 2 - 4 lớp):**

| Thuật toán | Naive Random Split (Có rò rỉ 100%) | Leak-Free Group Split (0% rò rỉ) | Độ lệch do rò rỉ (Optimistic Bias) |
| :--- | :---: | :---: | :---: |
| **Tỷ lệ rò rỉ mã độc trong Test** | **100.00% (2,578/2,578)** | **0.00% (0/582)** | -100% |
| **Random Forest Accuracy** | 82.30% | 80.43% | **+1.87% (Ảo)** |
| **Random Forest Macro-F1** | 73.37% | 70.62% | **+2.75% (Ảo)** |
| **XGBoost Accuracy** | 81.80% | 80.83% | **+0.97% (Ảo)** |
| **XGBoost Macro-F1** | 72.71% | 71.24% | **+1.47% (Ảo)** |

> **Kết luận:** Naive Split tạo ra độ chênh lệch lạc quan từ 1.5% đến 2.8% Macro-F1. Tuy nhiên, khi chuyển sang **Leak-Free Group Split**, mô hình vẫn đạt trên **80.8% Accuracy và 71.2% Macro-F1**, chứng minh 22 đặc trưng tàng hình bộ nhớ có khả năng khái quát hóa thực sự trên các file mã độc hoàn toàn mới.

---

### 2.2. ĐIỂM RÒ RỈ 2: Rò rỉ Hàng Trùng lặp Chính xác (Exact Duplicate Rows Leakage)
- **Mức độ nghiêm trọng:** 🟠 **CAO (High)**.
- **Bản chất kỹ thuật:**
  Do 22 đặc trưng chỉ ghi nhận trạng thái ẩn giấu (không chứa các biến động nhỏ như số handle/dll tăng dần), nhiều bản dump kế tiếp của cùng một tiến trình có giá trị 22 đặc trưng hoàn toàn giống nhau 100%.
  - Tổng số hàng bị trùng lặp trên 22 đặc trưng trong toàn bộ dataset: **32,505 hàng (chiếm 55.47%)**.
  - Khi chia dữ liệu ngẫu nhiên:
    - **Scenario 2:** Có **673 mẫu trong Test (10.47%)** là bản sao y hệt của mẫu trong Train.
    - **Scenario 3:** Có **1,550 mẫu trong Test (21.79%)** là bản sao y hệt của mẫu trong Train.
- **Hậu quả:** 1/5 tập test bị "học vẹt" (exact memorization).
- **Giải pháp:** Cần lọc trùng (deduplication) hoặc đảm bảo các cụm mẫu trùng nhau chỉ nằm ở một phía của phân vùng dữ liệu.

---

### 2.3. ĐIỂM RÒ RỈ 3: Rò rỉ Xung đột Nhãn (Contradictory Label Ambiguity)
- **Mức độ nghiêm trọng:** 🟡 **TRUNG BÌNH (Medium)**.
- **Bản chất kỹ thuật:**
  Kiểm định phát hiện **2,506 vector đặc trưng độc nhất** có giá trị 22 thuộc tính giống hệt nhau nhưng lại mang nhãn khác nhau (ví dụ: cùng một dấu vết bộ nhớ nhưng hàng A là `Trojan-Refroso`, hàng B là `Ransomware-Conti`, thậm chí có hàng là `Benign`).
- **Nguyên nhân:** Các kỹ thuật stealth cơ bản (tiêm mã hoặc ngắt DLL) là kỹ thuật dùng chung của nhiều chủng mã độc, khiến 22 đặc trưng này đôi khi không đủ phân biệt chi tiết tên dòng họ nếu không kết hợp thêm ngữ cảnh hành vi mạng hoặc hệ thống tệp.
- **Giải pháp:** Trong bài toán 16 lớp họ mã độc, các mô hình cần xuất xác suất phân loại (confidence / soft probability) thay vì dự đoán cứng (hard classification).

---

### 2.4. ĐIỂM RÒ RỈ 4: Rò rỉ Phân phối Tiên nghiệm trong Few-Shot Learning (Transductive Prior-Knowledge Leakage)
- **Mức độ nghiêm trọng:** 🔴 **RẤT CAO (Critical)**.
- **Bản chất kỹ thuật:**
  Trong quy trình Few-Shot mô phỏng ban đầu:
  - Conditional WGAN-GP và mạng nhúng Metric (ProtoNet/Siamese) được tiền huấn luyện trên toàn bộ `X_train` (gồm hàng nghìn mẫu của tất cả các lớp).
  - Sau đó, khi mô phỏng $K=5$ mẫu/lớp, chúng ta dùng GAN đã học từ trước để sinh thêm 100 mẫu/lớp.
  - **Lỗ hổng:** Trong thực tế, khi một họ mã độc Zero-Day mới xuất hiện với chỉ 5 mẫu, **chúng ta không thể có một mô hình GAN đã học từ 5,000 mẫu của chính họ đó**. Sử dụng GAN như vậy là gián tiếp làm rò rỉ tri thức phân phối tổng thể vào bài toán ít mẫu.
- **Khắc phục triệt để bằng Kịch bản Zero-Day Class-Disjoint Few-Shot:**
  - Chia tập dữ liệu thành:
    - **13 Lớp Cơ sở (Base Classes):** Benign + 12 họ mã độc quen thuộc (dùng huấn luyện biểu diễn mạng ProtoNet).
    - **3 Lớp Mới (Novel / Zero-Day Classes):** `Ransomware-Conti`, `Spyware-TIBS`, `Trojan-Zeus`. Tuyệt đối không xuất hiện trong quá trình huấn luyện mạng.
  - Đánh giá khả năng thích ứng của ProtoNet khi chỉ cấp duy nhất $K$ mẫu hỗ trợ của 3 lớp mới này:

| Số lượng mẫu hỗ trợ ($K$) | ProtoNet Zero-Day Accuracy (Mean ± Std) | ProtoNet Zero-Day Macro-F1 (Mean ± Std) | Đánh giá khoa học |
| :---: | :---: | :---: | :--- |
| **K = 5** | 37.50% ± 6.42% | 33.90% ± 6.86% | Vượt qua mức đoán ngẫu nhiên (33.33%) |
| **K = 10** | 42.49% ± 4.98% | 38.20% ± 4.74% | Cải thiện rõ rệt (+4.99% Acc) |
| **K = 20** | **44.27% ± 1.99%** | **40.71% ± 3.25%** | **Độ ổn định cao, phương sai giảm 3 lần** |

---

### 2.5. ĐIỂM RÒ RỈ 5: Rò rỉ khi Tái sử dụng Tập Dữ Liệu Tăng Cường (Downstream Artifact Leakage)
- **Mức độ nghiêm trọng:** 🟡 **TRUNG BÌNH (Medium)**.
- **Bản chất kỹ thuật:**
  Các file CSV tăng cường (`Dataset_22features_Scenario_2_GAN_augmented_1000.csv`) nếu không có siêu dữ liệu rõ ràng có thể bị các nhà nghiên cứu khác hiểu nhầm là dataset đầy đủ, dẫn đến việc chia ngẫu nhiên và đưa mẫu GAN vào tập Test.
- **Khắc phục đã thực hiện:**
  Đã bổ sung 2 cột siêu dữ liệu tường minh vào cả 2 file kết quả:
  - `Is_Synthetic`: `0` (Mẫu dữ liệu thực tế), `1` (Mẫu nhân tạo sinh bởi GAN).
  - `Split_Role`: `'train'` (Chỉ định rõ ràng chỉ dùng cho tập huấn luyện).

---

## 3. TỔNG KẾT BÀI HỌC KHOA HỌC & ĐÓNG GÓP

1. **Khẳng định tính liêm chính học thuật (Scientific Integrity):** Việc chủ động rà soát, phát hiện và định lượng các điểm rò rỉ dữ liệu giúp báo cáo nghiên cứu có độ tin cậy khoa học cao nhất, sẵn sàng bảo vệ trước hội đồng khoa học.
2. **Xác lập ranh giới hiệu năng thực tế:**
   - Khi loại bỏ 100% rò rỉ snapshot thời gian, độ chính xác trên 22 đặc trưng tàng hình bộ nhớ vẫn đạt **>80.8%** và Macro-F1 **>71.2%**, chứng minh giá trị thực tiễn vượt trội của nhóm đặc trưng `ldrmodules`, `malfind`, `psxview`.
   - Kết quả Zero-Day Few-Shot phân lập nhãn cung cấp một baseline chuẩn tắc cho các nghiên cứu tiếp theo về nhận diện biến thể mã độc mới.
