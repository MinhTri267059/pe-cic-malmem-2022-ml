# 🧪 Phòng Thí Nghiệm Ảo Hóa Phân Tích Mã Độc Bộ Nhớ (Malware Memory Analysis Lab)

Môi trường phòng thí nghiệm ảo hóa này được xây dựng bằng **Vagrant** và **Oracle VirtualBox** nhằm tái hiện quy trình thu thập dữ liệu bộ nhớ của tập dữ liệu **CIC-MalMem-2022** (*MalMemAnalysis-2022*). Phòng lab cung cấp một sandbox Windows 10 cô lập an toàn để thực thi mẫu mã độc PE, trích xuất RAM dump định kỳ và trích xuất đặc trưng bằng **Volatility / VolMemLyzer**.

---

## 📑 Mục Lục
- [1. Kiến Trúc Môi Trường Lab](#1-kiến-trúc-môi-trường-lab)
- [2. Kịch Bản Thu Thập & Xử Lý Dữ Liệu](#2-kịch-bản-thu-thập--xử-lý-dữ-liệu)
  - [2.1 Danh mục mẫu mã độc kiểm thử](#21-danh-mục-mẫu-mã-độc-kiểm-thử)
  - [2.2 Chu trình 4 bước thu thập](#22-chu-trình-4-bước-thu-thập)
  - [2.3 Chụp bộ nhớ theo chuỗi thời gian (10 lần / 15 giây)](#23-chụp-bộ-nhớ-theo-chuỗi-thời-gian-10-lần--15-giây)
  - [2.4 Giả lập hoạt động người dùng & Cân bằng dữ liệu (SMOTE)](#24-giả-lập-hoạt-động-người-dùng--cân-bằng-dữ-liệu-smote)
- [3. Hệ Thống Đặc Trưng Bộ Nhớ (VolMemLyzer-V2)](#3-hệ-thống-đặc-trưng-bộ-nhớ-volmemlyzer-v2)
- [4. Cơ Chế Cô Lập & Bảo Mật Sandbox](#4-cơ-chế-cô-lập--bảo-mật-sandbox)
- [5. Hướng Dẫn Vận Hành Máy Ảo Với Vagrant](#5-hướng-dẫn-vận-hành-máy-ảo-với-vagrant)
- [6. Quản Lý Snapshot & Khôi Phục Trạng Thái Sạch](#6-quản-lý-snapshot--khôi-phục-trạng-thái-sạch)
- [7. Kịch Bản Tự Động Hóa Thu Thập (Automation Script)](#7-kịch-bản-tự-động-hóa-thu-thập-automation-script)
- [8. Khắc Phục Sự Cố Thường Gặp (Troubleshooting)](#8-khắc-phục-sự-cố-thường-gặp-troubleshooting)
- [9. Tài Liệu Tham Khảo (Citation)](#9-tài-liệu-tham-khảo-citation)

---

## 1. Kiến Trúc Môi Trường Lab

```
+-----------------------------------------------------------------------------------------+
|                                  MÁY THẬT (HOST MACHINE)                                |
|  - Trình quản lý ảo hóa: Oracle VirtualBox                                             |
|  - Công cụ quản lý tự động: HashiCorp Vagrant                                          |
|  - Script điều khiển: Tự động kích hoạt mẫu, chụp RAM và dọn dẹp môi trường            |
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   |                  MÁY ẢO CÔ LẬP: Win10_Malware_Mem_Demo                          |   |
|   |  - Hệ điều hành: 64-bit Windows 10                                              |   |
|   |  - Cấu hình: 2 vCPUs, 2048 MB RAM (2 GB)                                        |   |
|   |  - Chế độ bảo mật: Cách ly hoàn toàn (Không Shared Folder, Clipboard, USB)      |   |
|   |                                                                                 |   |
|   |      [Ứng dụng thường nhật chạy nền]   +   [Kích hoạt mẫu mã độc PE]            |   |
|   |       (Trình duyệt, Office, Editor)         (Trojan, Spyware, Ransomware)       |   |
|   +---------------------------------------------------------------------------------+   |
|                                            │                                            |
|                                            ▼ (Lệnh chụp ngoài VM bằng VBoxManage)       |
|                 VBoxManage debugvm "Win10_Malware_Mem_Demo" dumpvmcore                  |
|                                            │                                            |
|                                            ▼                                            |
|                     Ảnh chụp bộ nhớ RAM thô (.raw - 2 GB / file)                        |
|                                            │                                            |
|                                            ▼ (Trích xuất bằng Volatility / VolMemLyzer) |
|               58 Đặc trưng bộ nhớ (Malfind, Handles, Ldrmodules, Hooks)                |
|                                            │                                            |
|                                            ▼                                            |
|                Tập dữ liệu CSV (~2 KB / mẫu) -> Đưa vào Huấn luyện AI/ML                |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Kịch Bản Thu Thập & Xử Lý Dữ Liệu

### 2.1 Danh mục mẫu mã độc kiểm thử

Bộ dữ liệu sử dụng **2.916 mẫu mã độc PE** thực tế (thu thập từ VirusTotal), thuộc 3 nhóm chính và phân vào 15 họ mã độc:

| Nhóm mã độc (Category) | Họ mã độc (Family) | Số lượng mẫu | Hành vi điển hình |
| :--- | :--- | :---: | :--- |
| **Trojan Horse** | Zeus, Emotet, Refroso, Scar, Reconyc | 948 mẫu | Đánh cắp thông tin, mở cửa sau (backdoor), tiêm mã DLL vào tiến trình sạch |
| **Spyware** | 180Solutions, Coolwebsearch, Gator, Transponder, TIBS | 982 mẫu | Chiếm quyền trình duyệt, lén theo dõi bàn phím và gửi dữ liệu người dùng |
| **Ransomware** | Conti, MAZE, Pysa, Ako, Shade | 986 mẫu | Quét tệp tin, mã hóa dữ liệu hàng loạt và xóa bản sao phục hồi (Shadow Copies) |

---

### 2.2 Chu trình 4 bước thu thập

1. **Tuyển chọn mẫu:** Lựa chọn các mẫu mã độc nhắm vào nền tảng Windows 10 x64 hiện đại có sử dụng các kỹ thuật che giấu (obfuscation, packing, unlinking).
2. **Kích hoạt & Chụp RAM ngoài máy ảo:** Chạy mẫu trong máy ảo Windows 10 (2 GB RAM). Lệnh chụp bộ nhớ được kích hoạt **từ máy thật** qua VirtualBox để không để lại bất kỳ tiến trình lạ nào bên trong máy ảo.
3. **Chuyển giao dữ liệu:** Gom các file RAM dump (.raw) chuyển sang môi trường phân tích pháp y số.
4. **Trích xuất đặc trưng:** Dùng công cụ VolMemLyzer phân tích các bảng quản lý của hệ điều hành và xuất ra dòng dữ liệu CSV.

---

### 2.3 Chụp bộ nhớ theo chuỗi thời gian (10 lần / 15 giây)

* **Cách thực hiện:** Với mỗi mẫu mã độc, hệ thống không chỉ chụp 1 lần mà thực hiện **chụp liên tiếp 10 lần**, mỗi lần cách nhau **15 giây** ($0s, 15s, 30s, \dots, 135s$).
* **Mục đích:** Mã độc hiện đại thường trì hoãn thực thi (sleep/delay), giải nén payload trong bộ nhớ, sau đó mới tiêm mã độc vào tiến trình khác hoặc tiến hành mã hóa. Việc chụp định kỳ 10 lần giúp bắt trọn toàn bộ các giai đoạn biến đổi hành vi trong RAM.
* **Quy mô:** $2.916\text{ mẫu} \times 10\text{ lần} \approx 29.298\text{ bản ghi RAM mã độc}$.

---

### 2.4 Giả lập hoạt động người dùng & Cân bằng dữ liệu (SMOTE)

Phần này xử lý 2 thách thức lớn nhất trong việc tạo dữ liệu bộ nhớ:

#### A. Giả lập hoạt động người dùng thường nhật (Benign Noise Injection)
* **Vấn đề:** Nếu chạy mã độc trong một máy ảo "trắng tinh" (chỉ có hệ điều hành vừa bật, không có ứng dụng nào mở), mô hình AI sẽ rất dễ "học vẹt": cứ thấy có tiến trình hoặc kết nối bất kỳ là phán đoán ngay là mã độc.
* **Giải pháp:** Trong suốt quá trình chạy mẫu mã độc, máy ảo luôn được mở sẵn các ứng dụng người dùng thông thường (trình duyệt web, ứng dụng văn phòng, trình nghe nhạc, dịch vụ nền). Nhờ vậy, môi trường RAM phản ánh đúng máy tính đang làm việc thực tế, buộc mã độc phải lẩn trốn giữa các tiến trình bình thường.

#### B. Cân bằng dữ liệu bằng thuật toán SMOTE
* **Vấn đề:** Do mỗi mẫu mã độc được chụp 10 lần nên số lượng bản ghi mã độc tăng vọt lên gần **29.298 bản ghi**. Trong khi đó, số lượng ảnh chụp máy sạch (Benign) ít hơn nhiều. Nếu dữ liệu bị lệch (quá nhiều mẫu độc hại, quá ít mẫu sạch), mô hình học máy sẽ bị thiên lệch (bias) và dự đoán kém chính xác trên thực tế.
* **Giải pháp:** Thuật toán **SMOTE** (*Synthetic Minority Over-sampling Technique*) được sử dụng. Thay vì sao chép y nguyên các mẫu sạch (dễ gây overfitting), SMOTE tính toán khoảng cách giữa các điểm dữ liệu lành tính hiện có và tự động sinh ra các mẫu sạch nhân tạo mới nhưng hoàn toàn hợp lý. Kết quả tạo nên tập dữ liệu cân bằng hoàn hảo **50:50** ($29.298$ mẫu sạch và $29.298$ mẫu mã độc = $58.596$ bản ghi tổng thể).

---

## 3. Hệ Thống Đặc Trưng Bộ Nhớ (VolMemLyzer-V2)

Bộ công cụ trích xuất **58 đặc trưng** từ RAM (bao gồm 26 đặc trưng chuyên sâu phát hiện mã độc ẩn) chia thành 5 nhóm:

| Nhóm đặc trưng | Nguồn phân tích | Ý nghĩa thực tiễn |
| :--- | :--- | :--- |
| **Malfind** | Vùng nhớ ảo (VAD) | Phát hiện vùng nhớ được cấp quyền thực thi bất thường (`PAGE_EXECUTE_READWRITE`) và hành vi tiêm mã (Code Injection). |
| **Ldrmodule** | Danh sách DLL liên kết | So sánh các bảng danh sách DLL nạp vào tiến trình để tìm các DLL bị mã độc cố tình gỡ liên kết nhằm ẩn thân. |
| **Handles** | Bảng quản lý tài nguyên | Đếm số lượng tệp mở, registry key, mutex/mutant (dấu hiệu mã độc kiểm tra tránh chạy đúp) và cổng mạng. |
| **Process View** | Bảng danh sách tiến trình | Đối chiếu chéo giữa danh sách tiến trình hiển thị chuẩn (`pslist`) và quét vết bộ nhớ thô (`psscan`) để lật tẩy tiến trình bị rootkit giấu đi. |
| **Apihooks** | Bảng hook hàm hệ thống | Thống kê số lượng API Windows bị hook (chuyển hướng gọi hàm) ở cả tầng người dùng (user mode) lẫn tầng lõi. |

---

## 4. Cơ Chế Cô Lập & Bảo Mật Sandbox

Tệp cấu hình [Vagrantfile](Vagrantfile) được thiết kế theo nguyên tắc an toàn nghiêm ngặt để mã độc không thể thoát ra máy thật:

* **Tắt thư mục chia sẻ (`synced_folder disabled: true`):** Chặn đứng ransomware mã hóa hoặc can thiệp tệp tin trên máy thật.
* **Tắt Clipboard & Kéo thả tệp:** Ngăn rò rỉ dữ liệu hoặc vô tình kích hoạt file qua thao tác chuột.
* **Tắt toàn bộ cổng ngoại vi:** Vô hiệu hóa USB, cổng nối tiếp (UART) và cổng song song (LPT).
* **Cấu hình phần cứng:** Cấp phát 2 vCPUs và đúng **2048 MB RAM** (2 GB) theo chuẩn thử nghiệm.

---

## 5. Hướng Dẫn Vận Hành Máy Ảo Với Vagrant

Di chuyển vào thư mục `lab`:
```bash
cd lab
```

### 5.1 Khởi động máy ảo
```bash
vagrant up
```
Cửa sổ giao diện đồ họa (GUI) của Windows 10 sẽ hiển thị trực tiếp trên màn hình.

### 5.2 Kiểm tra trạng thái máy ảo
```bash
vagrant status
```

### 5.3 Tạm dừng và tiếp tục
* **Tạm dừng (lưu trạng thái RAM vào ổ cứng):**
  ```bash
  vagrant suspend
  ```
* **Tiếp tục hoạt động:**
  ```bash
  vagrant resume
  ```

### 5.4 Tắt máy và dọn dẹp
* **Tắt máy an toàn:**
  ```bash
  vagrant halt
  ```
* **Buộc dừng ngay lập tức:**
  ```bash
  vagrant halt --force
  ```
* **Hủy hoàn toàn máy ảo để giải phóng ổ cứng:**
  ```bash
  vagrant destroy -f
  ```

---

## 6. Quản Lý Snapshot & Khôi Phục Trạng Thái Sạch

Trước khi thả bất kỳ tệp mã độc nào vào máy ảo, **bắt buộc** phải tạo một điểm lưu trạng thái máy sạch (Clean Snapshot):

```bash
# 1. Lưu snapshot máy ảo sạch (khi Windows đã khởi động ổn định)
vagrant snapshot save clean_base_state

# 2. Xem danh sách snapshot
vagrant snapshot list

# 3. Khôi phục máy về trạng thái sạch ban đầu sau khi kết thúc thử nghiệm
vagrant snapshot restore clean_base_state
```

---

## 7. Kịch Bản Tự Động Hóa Thu Thập (Automation Script)

Dưới đây là script Bash mô phỏng chu trình tự động: khôi phục máy sạch $\rightarrow$ nạp mẫu & kích hoạt chạy $\rightarrow$ chụp 10 lần cách nhau 15 giây $\rightarrow$ phục hồi lại máy sạch:

```bash
#!/usr/bin/env bash
set -e

VM_NAME="Win10_Malware_Mem_Demo"
OUTPUT_DIR="./dumps"
SAMPLE_NAME="test_malware_sample"

mkdir -p "$OUTPUT_DIR"

echo "[1] Khôi phục máy ảo về trạng thái sạch ban đầu..."
vagrant snapshot restore clean_base_state

echo "[2] Nạp và kích hoạt mẫu mã độc: $SAMPLE_NAME trong máy ảo..."
# Nạp mẫu mã độc vào thư mục tạm qua kênh hypervisor:
VBoxManage guestcontrol "$VM_NAME" copyto \
    --username "vagrant" --password "vagrant" \
    --target-directory "C:\\Users\\vagrant\\AppData\\Local\\Temp\\sample.exe" \
    "./samples/${SAMPLE_NAME}.exe"

# Kích hoạt thực thi ngầm:
VBoxManage guestcontrol "$VM_NAME" run \
    --username "vagrant" --password "vagrant" \
    --exe "cmd.exe" -- "/c" "start" "" "C:\\Users\\vagrant\\AppData\\Local\\Temp\\sample.exe"

echo "[3] Bắt đầu chu kỳ chụp 10 ảnh bộ nhớ RAM (cách nhau 15 giây)..."
for i in {1..10}; do
    TIMESTAMP=$(date +%s)
    DUMP_FILE="${OUTPUT_DIR}/${SAMPLE_NAME}_dump_${i}_${TIMESTAMP}.raw"

    echo " -> Chụp lần $i/10: $DUMP_FILE"
    VBoxManage debugvm "$VM_NAME" dumpvmcore --filename "$DUMP_FILE"

    if [ "$i" -lt 10 ]; then
        echo " -> Tạm dừng 15 giây để mã độc tiếp tục thực thi..."
        sleep 15
    fi
done

echo "[4] Đã hoàn tất 10 lần chụp. Khôi phục lại máy sạch..."
vagrant snapshot restore clean_base_state
echo "Hoàn thành! Các file .raw sẵn sàng để đưa vào Volatility / VolMemLyzer."
```

---

## 8. Khắc Phục Sự Cố Thường Gặp (Troubleshooting)

### Lỗi 1: Quá thời gian chờ nạp Windows (`WinRM Timeout`)
* **Nguyên nhân:** Windows 10 khởi động lần đầu hoặc chạy cấu hình sysprep mất nhiều thời gian hơn dự kiến.
* **Cách xử lý:** Tham số `config.vm.boot_timeout = 700` đã được cấu hình trong [Vagrantfile](Vagrantfile). Bạn có thể mở trực tiếp cửa sổ giao diện VirtualBox để theo dõi tiến trình khởi động của hệ thống.

### Lỗi 2: Đầy dung lượng ổ cứng
* **Nguyên nhân:** Mỗi bản ghi dump RAM đầy đủ chiếm 2 GB dung lượng. Một chuỗi 10 lần chụp cho một mẫu sẽ tiêu tốn ~20 GB lưu trữ.
* **Cách xử lý:** Sau khi dùng VolMemLyzer trích xuất xong đặc trưng sang file CSV (~2 KB), hãy chủ động xóa hoặc nén các file `.raw` thô để giải phóng không gian đĩa cứng.

---

## 9. Tài Liệu Tham Khảo (Citation)

Nếu bạn sử dụng môi trường phòng thí nghiệm, quy trình thu thập dữ liệu bộ nhớ hoặc bộ dữ liệu này trong các nghiên cứu và dự án, vui lòng trích dẫn tài liệu gốc:

```bibtex
@inproceedings{carrier2022detecting,
  author       = {Tristan Carrier and Princy Victor and Ali Tekeoglu and Arash Habibi Lashkari},
  title        = {Detecting Obfuscated Malware using Memory Feature Engineering},
  booktitle    = {Proceedings of the 8th International Conference on Information Systems Security and Privacy (ICISSP 2022)},
  pages        = {177--188},
  year         = {2022},
  publisher    = {SCITEPRESS},
  doi          = {10.5220/0010908200003120}
}
```
