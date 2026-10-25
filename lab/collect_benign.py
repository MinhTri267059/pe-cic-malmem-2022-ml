#!/usr/bin/env python3
"""Thu thập RAM dump Benign đa dạng: mở các ứng dụng khác nhau qua guestcontrol
(tự động, không cần thao tác tay trong GUI), rồi chụp RAM N lần như run_sample.py.

Không chạy mẫu mã độc nào — chỉ mở ứng dụng Windows có sẵn trong box
(Edge, Notepad, Explorer, PowerShell...) với nội dung khác nhau mỗi lần, để
tạo "chữ ký hệ thống" (process list, handle count...) đa dạng hơn một baseline
tĩnh duy nhất. Xem EDA_Report.md mục 5 để biết lý do.

Dùng chung snapshot/VM với run_sample.py; restore/verify-isolation/dump/revert
giống hệt, chỉ khác bước "deliver" (ở đây là mở app thay vì chạy zip).
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))
import run_sample  # noqa: E402
from run_sample import Lab, Aborted, POWERSHELL  # noqa: E402

# ponytail: run_sample.Lab.vagrant() chạy `vagrant` với cwd=LAB_DIR (thư mục
# lab/); nhưng mỗi VM instance có Vagrantfile riêng trong lab/vms/<name>/, nên
# phải chạy vagrant từ cwd hiện tại (nơi script này được gọi) để đúng VM.
run_sample.LAB_DIR = Path.cwd()

# Mỗi "profile" mở một tổ hợp ứng dụng khác nhau qua guestcontrol, để mỗi lần
# chụp Benign có process list / handle count khác nhau thay vì luôn giống hệt.
PROFILES = {
    "idle": r"""
        Start-Sleep -Seconds 2
    """,
    "browser_tabs": r"""
        Start-Process msedge.exe -ArgumentList '--new-window','about:blank'
        Start-Sleep -Seconds 8
    """,
    "notepad_explorer": r"""
        Start-Process notepad.exe
        Start-Process explorer.exe -ArgumentList 'C:\Windows'
        Start-Sleep -Seconds 5
    """,
    "many_cmd_windows": r"""
        1..5 | ForEach-Object { Start-Process cmd.exe -ArgumentList '/k','echo window',$_ }
        Start-Sleep -Seconds 5
    """,
    "powershell_busy": r"""
        Start-Process powershell.exe -ArgumentList '-NoExit','-Command','Get-Process | Out-Null; Start-Sleep -Seconds 30'
        Start-Sleep -Seconds 5
    """,
}


def open_apps(lab: Lab, profile: str):
    """Mở ứng dụng trong guest theo 1 profile, không chờ thao tác tay."""
    a = lab.a
    script = PROFILES[profile]
    with tempfile.TemporaryDirectory() as tmp:
        pw_file = Path(tmp) / "guest.pw"
        pw_file.write_text(a.guest_password)

        def gc(sub, *argv, check=True):
            return lab.vbox("guestcontrol", a.vm, sub, "--username", a.guest_user,
                             "--passwordfile", str(pw_file), *argv, check=check,
                             capture=True if not check else False)

        print(f"Waiting for Guest Additions (up to {a.boot_wait}s)...")
        deadline = time.monotonic() + a.boot_wait
        while gc("mkdir", "--parents", "C:\\Sandbox", check=False).returncode != 0:
            if time.monotonic() > deadline:
                raise RuntimeError("guest did not answer guestcontrol")
            time.sleep(5)

        ps_file = Path(tmp) / "open_apps.ps1"
        ps_file.write_text(script)
        gc("copyto", "--target-directory", "C:\\Sandbox\\", str(ps_file), check=False)
        gc("run", "--exe", POWERSHELL, "--wait-stdout", "--wait-stderr",
           "--timeout", "90000", "--", "powershell.exe",
           "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\\Sandbox\\open_apps.ps1")


def collect_one(lab: Lab, label: str, profile: str):
    """restore -> verify-isolation -> mở app theo profile -> dump N lần -> revert."""
    a = lab.a
    lab.pause(f"[{label}] restore the clean snapshot '{a.snapshot}'")
    lab.restore()
    aborted = False
    try:
        lab.pause(f"[{label}] verify the VM has no network path")
        lab.verify_isolation()

        print(f"Opening apps (profile: {profile})...")
        open_apps(lab, profile)
        print(f"Letting the system settle for {a.settle}s before the first dump...")
        time.sleep(a.settle)

        lab.pause(f"[{label}] capture {a.count} memory dumps")
        lab.dump(label)
    except (KeyboardInterrupt, Aborted):
        aborted = True
        raise
    finally:
        if not aborted:
            lab.pause(f"[{label}] revert to the clean snapshot")
        print("Reverting to the clean snapshot...")
        try:
            lab.restore()
        except subprocess.CalledProcessError:
            print(f"WARNING: snapshot restore failed; restore '{a.snapshot}' manually.",
                  file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles", nargs="+", choices=list(PROFILES) + ["random"],
                    default=["idle", "browser_tabs", "notepad_explorer"],
                    help="profile(s) ứng dụng để mở trước khi chụp (mặc định: idle, browser_tabs, notepad_explorer)")
    p.add_argument("--repeat", type=int, default=1, help="số lần lặp lại toàn bộ danh sách profile")
    p.add_argument("--auto", action="store_true", help="no pauses between stages")
    p.add_argument("--count", type=int, default=10, help="dumps per profile run (default: 10)")
    p.add_argument("--interval", type=int, default=15, help="seconds between dumps (default: 15)")
    p.add_argument("--settle", type=int, default=20, help="seconds to let apps settle (default: 20)")
    p.add_argument("--pause", action="store_true", help="pause VM during each dump")
    p.add_argument("--output", default=str(LAB_DIR / "dumps"), help="dump directory")
    p.add_argument("--vm", required=True, help="VM name (e.g. Win10Legacy_Mem)")
    p.add_argument("--snapshot", default="clean_base_state", help="clean snapshot name")
    p.add_argument("--guest-user", default="vagrant")
    p.add_argument("--boot-wait", type=int, default=300)
    a = p.parse_args()
    a.guest_password = os.environ.get("VBOX_GUEST_PASSWORD", "vagrant")
    if min(a.count, a.interval, a.settle, a.boot_wait) < 1:
        p.error("--count, --interval, --settle and --boot-wait must be positive")
    return a


def main():
    a = parse_args()
    lab = Lab(a)
    profiles = list(PROFILES) if "random" in a.profiles else a.profiles
    jobs = []
    for rep in range(a.repeat):
        order = profiles[:]
        if "random" in a.profiles:
            random.shuffle(order)
        for profile in order:
            label = f"Benign-{profile}-r{rep}"
            jobs.append((label, profile))

    failed = []
    try:
        for label, profile in jobs:
            try:
                collect_one(lab, label, profile)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                print(f"FAILED {label}: {e}", file=sys.stderr)
                failed.append(label)
    except (KeyboardInterrupt, Aborted) as e:
        print(f"\nAborted ({e or 'interrupted'}); the snapshot restore was attempted.")
        lab.poweroff()
        return 130
    lab.poweroff()
    print(f"\nFinished: {len(jobs) - len(failed)} ok, {len(failed)} failed. Dumps in {a.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
