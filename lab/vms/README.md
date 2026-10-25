# VM đa dạng (Multi-OS) cho thu thập Benign

Mỗi thư mục con là một Vagrant instance riêng (state `.vagrant/` tách biệt),
dùng chung [`../Vagrantfile`](../Vagrantfile) qua `load`. Mục đích: phá vỡ sự
nhầm lẫn giữa nhãn Benign/Malware và "chữ ký máy" (xem `EDA_Report.md` mục 5)
bằng cách thu thập dữ liệu từ nhiều OS/build khác nhau thay vì 1 snapshot
duy nhất như CIC-MalMem-2022.

| Thư mục | Box | VM_NAME | Ghi chú |
| :--- | :--- | :--- | :--- |
| `win10_legacy/` | `StefanScherer/windows_10` (2020.05.27) | `Win10Legacy_Mem` | Giống cấu hình gốc của repo — dùng để đối chiếu |
| `win10_22h2/` | `gusztavvargadr/windows-10-22h2-enterprise` | `Win10_22H2_Mem` | Build mới hơn hẳn |
| `win11/` | `gusztavvargadr/windows-11` | `Win11_Mem` | Windows 11 |
| `win7/` | `ferventcoder/win7pro-x64-nocm-lite` | `Win7_Mem` | Windows 7, nhẹ, không config management |

## Cách chạy

```bash
cd lab/vms/win10_22h2
export WINDOWS_BOX=gusztavvargadr/windows-10-22h2-enterprise
export VM_NAME=Win10_22H2_Mem
export VAGRANT_EXPERIMENTAL=none_communicator
vagrant up
```

Sau khi Windows khởi động ổn định trong GUI, lưu 1 (hoặc nhiều) snapshot sạch
— xem mục 6 của `../README.md`. Mỗi VM nên có **ít nhất 2 snapshot khác nhau**
(ví dụ: "mới cài" và "đã mở sẵn Office + browser") để tăng đa dạng ngay trong
cùng 1 OS.

Thu thập Benign bằng `run_sample.py --sample` với `--vm` trỏ đúng `VM_NAME`
và chạy từ thư mục instance tương ứng (để Vagrant/VBoxManage thấy đúng VM).

## Tiến độ (2026-10-25)

- **win10_legacy: XONG.** 50 dump Benign thu bằng `collect_benign.py` (5 profile
  idle/browser_tabs/notepad_explorer/many_cmd_windows/powershell_busy × 10),
  đã trích xuất bằng `extract_features.py` → `win10_legacy/features.csv`
  (50 dòng, 0 lỗi, tất cả Class=Benign đúng). File `.raw` (106GB) đã xóa.
- **win10_22h2, win11, win7: CHƯA bắt đầu.** Chưa tải box nào.

### Bẫy đã gặp khi chạy extract_features.py (Python 2.7 / Volatility 2)

1. **Đường dẫn dự án có dấu tiếng Việt** (`Chuyên đề...`) khiến
   `vol.py` báo "The requested file doesn't exist" dù file tồn tại thật.
   Fix: tạo junction ASCII trỏ tới thư mục dump trước khi chạy, ví dụ:
   ```powershell
   cmd /c mklink /J C:\vmwork "<đường dẫn đầy đủ tới dumps/>"
   ```
   rồi trỏ `extract_features.py` vào `C:\vmwork` (và dùng `--output`/`--cache-dir`
   cũng ở ngoài đường dẫn Unicode, ví dụ `C:\vmout`, `C:\vmcache`).
   `C:\` root bị chặn viết trực tiếp — dùng thư mục con như `C:\vmout\`.
2. **vol.py nằm trực tiếp ở** `C:\Users\ADMIN\tools\volatility\vol.py`
   (không phải `...\volatility\volatility\vol.py`).
3. **Nhãn "Benign" bị gán sai "Malware":** `collect_benign.py` đặt label
   `Benign-{profile}-r{rep}` (ví dụ `Benign-idle-r0`), nhưng
   `extract_features.py` gốc so khớp đúng string `"benign"`. Đã sửa
   `meta_of()` dùng `label.lower().startswith("benign")`.
4. Lệnh đã dùng thành công (từ thư mục dự án gốc, biến `VOL2_PYTHON`/`VOL2_PY`
   đã set env):
   ```powershell
   $env:VOL2_PYTHON = "C:\Users\ADMIN\miniconda3\envs\vol2\python.exe"
   $env:VOL2_PY = "C:\Users\ADMIN\tools\volatility\vol.py"
   python lab\extract_features.py "C:\vmwork" --output "C:\vmout\<vm>_features.csv" --cache-dir "C:\vmcache" --resume
   ```
   Tốc độ thực tế: ~1 phút/dump (9 plugin, 4 jobs song song).

### Bước tiếp theo

Lặp lại cho `win10_22h2`, `win11`, `win7`: `vagrant up` (tải box lần đầu
có thể mất nhiều thời gian/băng thông) → tạo snapshot `clean_base_state`
→ `collect_benign.py --vm <VM_NAME> --profiles idle browser_tabs
notepad_explorer many_cmd_windows powershell_busy --count 10 --interval 15
--settle 20 --auto --output dumps` (chạy từ thư mục instance) →
`extract_features.py` qua junction ASCII như trên → copy CSV vào
`lab/vms/<vm>/features.csv` → xóa `.raw`. Sau đủ 4 VM, hợp nhất 4
`features.csv` + nguồn gốc MalMem2022.csv, huấn luyện lại, rồi kiểm tra
lại trên `demo test/` xem model đã tổng quát hóa tốt hơn chưa.
