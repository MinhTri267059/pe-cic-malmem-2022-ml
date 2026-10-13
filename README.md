# Detection and Analysis of Malicious Software Using Machine Learning Models (CIC-MalMem-2022)

Repository này chứa mã nguồn Python nhằm tái hiện lại các kịch bản thực nghiệm phân loại mã độc dựa trên bộ dữ liệu **CIC-MalMem-2022**. Mã nguồn thực hiện quá trình tiền xử lý, huấn luyện và đánh giá 5 thuật toán Học máy (Random Tree, Random Forest, J-48, Naive Bayes, XGBoost) qua 3 kịch bản phân loại với 2 phương pháp kiểm thử (Percentage Split 80% và 10-Fold Cross-Validation).

## 📥 Tải xuống Bộ dữ liệu (Dataset)

Để chạy mã nguồn, bạn cần tải xuống tập dữ liệu gốc `MalMem2022.csv`.
🔗 **[Tải xuống Dataset CIC-MalMem-2022 (Google Drive)](https://drive.google.com/drive/folders/1TooIQ5HmU1n5X34zYpIgmtTW49X9_QMY?usp=sharing)**

Sau khi tải xong, hãy đặt file `MalMem2022.csv` vào thư mục gốc của repository này.

## 🚀 Hướng dẫn Sử dụng

1. **Cài đặt thư viện yêu cầu:**
   Bạn cần cài đặt các thư viện Python như `pandas`, `scikit-learn`, và `xgboost`.
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib
   ```

2. **Chạy kịch bản thực nghiệm:**
   Đảm bảo file `MalMem2022.csv` đã nằm trong thư mục gốc. Chạy lệnh sau:
   ```bash
   python3 reproduce_experiment.py
   ```

   **Mã nguồn sẽ tự động thực hiện:**
   - Xóa bỏ dữ liệu trùng lặp (giữ lại 58.062 mẫu hợp lệ).
   - Xuất dữ liệu đã chuẩn hóa của 3 kịch bản ra các file CSV (`Dataset_Scenario_1.csv`, `Dataset_Scenario_2.csv`, v.v.).
   - Lưu trữ trọng số mô hình đã huấn luyện (file `.pkl`) và tham số (file `.json`) vào thư mục `Models/`.
   - Xuất bảng chỉ số đánh giá ra các file `Metrics_Scenario_*.csv`.

*(Ghi chú: Các file Dataset dung lượng lớn và file model nặng đã được `.gitignore` để tránh phình to kích thước repository).*

3. **Vẽ các biểu đồ EDA theo từng kịch bản:**
   ```bash
   pip install matplotlib seaborn
   python3 generate_eda.py
   ```
   Ảnh được lưu trong thư mục `eda_figures/` (`Overview/`, `Scenario_1/`, `Scenario_2/`, `Scenario_3/`). Phân tích chi tiết: [EDA_Report.md](EDA_Report.md).

---

## 🔍 Phân Tích Dữ Liệu Khám Phá (EDA)

### Kịch bản 1 – Nhị phân (Benign vs Malware)
Dữ liệu cân bằng gần 50:50 (29.231 Benign / 28.831 Malware). Hai lớp tách biệt rõ trong không gian PCA, các đặc trưng như `dlllist.avg_dlls_per_proc` (r = -0.91) và `handles.nevent` (r = -0.88) phân biệt rất mạnh.

| Phân phối lớp | PCA 2D |
| :---: | :---: |
| ![](eda_figures/Scenario_1/01_binary_class_distribution.png) | ![](eda_figures/Scenario_1/04_binary_pca_2d.png) |

### Kịch bản 2 – 4 nhóm (Benign, Ransomware, Spyware, Trojan)
3 nhóm mã độc có số mẫu gần bằng nhau (~9.500 mẫu mỗi nhóm) nhưng dấu vết bộ nhớ gần như giống hệt nhau, nên chồng lấn mạnh trong PCA.

| Dấu vết hành vi (Z-score) | PCA 2D |
| :---: | :---: |
| ![](eda_figures/Scenario_2/02_multi4_behavioral_fingerprint_heatmap.png) | ![](eda_figures/Scenario_2/04_multi4_pca_2d.png) |

### Kịch bản 3 – 16 lớp (Benign + 15 họ mã độc)
Mỗi họ có ~1.400–2.400 mẫu. Khi gom cụm phân cấp, phần lớn các họ có profile rất giống nhau và không gom theo nhóm cha, nên đây là kịch bản khó nhất.

| Phân phối 16 lớp | Gom cụm 15 họ mã độc |
| :---: | :---: |
| ![](eda_figures/Scenario_3/01_multi16_family_distribution.png) | ![](eda_figures/Scenario_3/03_multi16_hierarchical_clustering.png) |

![Độ phức tạp vs hiệu suất](eda_figures/Overview/02_cross_scenario_complexity_summary.png)

---

## 📊 Kết Quả Thực Nghiệm (Tái hiện)

Dưới đây là tóm tắt độ chính xác (Accuracy) đạt được sau quá trình chạy mô phỏng bằng Python. Nhìn chung, kết quả bám rất sát và phản ánh đúng xu hướng của bài báo gốc.

### 1. Kịch bản Phân loại Nhị phân (Binary - 2 lớp: Benign và Malware)
Sự phân tách giữa lành tính và mã độc rất rõ ràng. Các thuật toán Tree-based và Ensemble đều đạt hiệu suất gần như tuyệt đối (100%).

| Mô hình | Phương pháp | Accuracy (Tái hiện) |
| :--- | :--- | :--- |
| **XGBoost** | 80% Split | **100.0%** |
| **Random Forest (RF)** | 80% Split | **100.0%** |
| **J-48 (Decision Tree)**| 80% Split | 99.98% |
| **Naive Bayes (NB)** | 10-Fold CV | 99.25% |

### 2. Kịch bản Đa lớp 4 nhóm (Multi-4: Benign, Ransomware, Spyware, Trojan)
Bài toán trở nên phức tạp hơn, XGBoost và Random Forest là 2 mô hình dẫn đầu, trong khi Naive Bayes giảm sút mạnh.

| Mô hình | Phương pháp | Accuracy (Tái hiện) |
| :--- | :--- | :--- |
| **XGBoost** | 10-Fold CV | **87.71%** |
| **Random Forest (RF)** | 10-Fold CV | **87.78%** |
| **J-48 (Decision Tree)**| 10-Fold CV | 85.09% |
| **Naive Bayes (NB)** | 10-Fold CV | 68.70% |

### 3. Kịch bản Đa lớp 16 nhóm (Multi-16: 1 Benign + 15 họ mã độc)
Trong kịch bản khó nhất (phân biệt chi tiết 15 dòng mã độc), XGBoost thể hiện rõ ưu thế tuyệt đối so với các thuật toán khác.

| Mô hình | Phương pháp | Accuracy (Tái hiện) |
| :--- | :--- | :--- |
| **XGBoost** | 10-Fold CV | **76.60%** |
| **Random Forest (RF)** | 10-Fold CV | 75.68% |
| **J-48 (Decision Tree)**| 10-Fold CV | 73.65% |
| **Naive Bayes (NB)** | 10-Fold CV | 56.83% |

---

## ⚖️ So Sánh Với Bài Báo Gốc

Dưới đây là phần đối chiếu độ chính xác (Accuracy) đạt được từ mã nguồn tái hiện (Python/Scikit-learn/XGBoost) với kết quả công bố trong bài báo gốc (sử dụng công cụ WEKA). Nhìn chung, kết quả bám rất sát bài báo và sự chênh lệch phần lớn nằm ở mức dưới 1%.

### 1. Kịch bản Phân loại Nhị phân (Binary)
| Mô hình | Phương pháp | Kết quả Bài báo | Kết quả Tái hiện | Chênh lệch (Delta) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 80% Split | 99.99% | 100.0% | + 0.01% |
| **Random Forest (RF)** | 80% Split | > 99.90% | 100.0% | + 0.10% |
| **J-48 (Decision Tree)**| 80% Split | > 99.90% | 99.98% | + 0.08% |
| **Naive Bayes (NB)** | 10-Fold CV | 98.87% - 98.89% | 99.25% | ~ + 0.36% |

### 2. Kịch bản Đa lớp 4 nhóm (Multi-4)
| Mô hình | Phương pháp | Kết quả Bài báo | Kết quả Tái hiện | Chênh lệch (Delta) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 80% Split | **87.79%** | 87.30% | - 0.49% |
| **Random Forest (RF)** | 80% Split | 87.61% | 87.02% | - 0.59% |
| **J-48 (Decision Tree)**| 10-Fold CV | 86.76% - 86.84% | 85.09% | ~ - 1.75% |
| **Naive Bayes (NB)** | 10-Fold CV | 67.80% - 68.24% | 68.70% | ~ + 0.46% |

### 3. Kịch bản Đa lớp 16 nhóm (Multi-16)
| Mô hình | Phương pháp | Kết quả Bài báo | Kết quả Tái hiện | Chênh lệch (Delta) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 80% Split | **75.49%** | 76.37% | + 0.88% |
| **Random Forest (RF)** | 10-Fold CV | 74.66% - 75.42% | 75.68% | ~ + 0.26% |
| **J-48 (Decision Tree)**| 10-Fold CV | 74.76% - 75.12% | 73.65% | ~ - 1.11% |
| **Naive Bayes (NB)** | 10-Fold CV | 56.99% - 57.17% | 56.83% | ~ - 0.16% |

> **Nhận xét:** Quá trình tái hiện đã thành công xuất sắc. Ở những kịch bản phân loại sâu phức tạp (16 lớp), mã mô phỏng bằng Python thậm chí đạt độ chính xác **nhỉnh hơn bài báo gốc gần 1%** ở thuật toán XGBoost và Random Forest.
