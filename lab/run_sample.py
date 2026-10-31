#!/usr/bin/env python3
"""Run malware samples in the offline sandbox VM and collect memory dumps.

Per sample: restore clean snapshot -> verify isolation -> copy the still-locked
zip into the guest (VBoxManage guestcontrol), unzip and launch it there ->
dump memory N times -> revert to the clean snapshot. The host never extracts or
executes the zip.

Default is step by step (Enter before each stage); --auto runs unattended.
Without any zip, use --sample NAME and start the sample by hand in the VM GUI.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
GUEST_DIR = "C:\\Sandbox"
SANDBOX_NET = "sbxnet"
CMD = r"C:\Windows\System32\cmd.exe"
PIECE = 48 * 1024  # guestcontrol copyto fails on larger writes with some Guest Additions
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PASSWORD_RE = re.compile(r"^[A-Za-z0-9._@%+=-]*$")  # reaches a guest command line

# Runs INSIDE the guest: unzip with 7-Zip, then start the sample through a
# one-shot scheduled task so it lives in the interactive session and is not a
# child of VBoxService.
LAUNCH_PS1 = r"""
param(
    [Parameter(Mandatory)] [string] $Zip,
    [string] $Password = '',
    [string] $Run = '',
    [Parameter(Mandatory)] [string] $User
)
$ErrorActionPreference = 'Stop'
$out = 'C:\Sandbox\out'
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$sevenZip = @('C:\Sandbox\7z\x64\7za.exe', 'C:\Sandbox\7-ZipPortable\App\7-Zip\7z.exe', "$env:ProgramFiles\7-Zip\7z.exe", "${env:ProgramFiles(x86)}\7-Zip\7z.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if ($sevenZip) {
    $a = @('x', '-y', "-o$out")
    if ($Password) { $a += "-p$Password" }
    $a += $Zip
    & $sevenZip @a | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed with exit code $LASTEXITCODE" }
} elseif (-not $Password) {
    Expand-Archive -Path $Zip -DestinationPath $out -Force
} else {
    throw 'Password-protected zip needs 7-Zip installed in the guest'
}

if ($Run) {
    $target = Get-ChildItem -Path $out -Recurse -File -Filter $Run | Select-Object -First 1
} else {
    $c = @(Get-ChildItem -Path $out -Recurse -File -Include *.exe, *.scr, *.com, *.bat, *.cmd, *.ps1, *.vbs, *.js)
    if ($c.Count -ne 1) { throw "Expected exactly 1 launchable file, found $($c.Count); pass --run" }
    $target = $c[0]
}
if (-not $target) { throw 'Launch target not found in archive' }

$task = 'SbxRun'
& schtasks.exe /Create /F /TN $task /SC ONCE /ST 23:59 /RU $User /IT /TR "`"$($target.FullName)`""
if ($LASTEXITCODE -ne 0) { throw 'schtasks /Create failed' }
& schtasks.exe /Run /TN $task
if ($LASTEXITCODE -ne 0) { throw 'schtasks /Run failed' }
& schtasks.exe /Delete /F /TN $task | Out-Null
Write-Output "Launched $($target.FullName)"
"""


# Benign profiles: guest-native programs started in the interactive session (no sample).
BENIGN_PROFILES = {
    "idle": [],
    "office": [r"C:\Windows\System32\notepad.exe", r"C:\Windows\System32\calc.exe",
               r"C:\Windows\System32\mspaint.exe", r"C:\Program Files\Windows NT\Accessories\wordpad.exe"],
    "browse": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe http://example.com",
               r"C:\Windows\System32\notepad.exe"],
}


class Aborted(Exception):
    pass


class Lab:
    def __init__(self, args):
        self.a = args
        self.env = dict(os.environ)
        flags = [f for f in self.env.get("VAGRANT_EXPERIMENTAL", "").split(",") if f]
        if "none_communicator" not in flags:
            flags.append("none_communicator")
        self.env["VAGRANT_EXPERIMENTAL"] = ",".join(flags)

    # ---- helpers -------------------------------------------------------
    def pause(self, what):
        """Wait for Enter before a stage (skipped with --auto)."""
        if self.a.auto:
            print(f"\n== {what}")
            return
        try:
            input(f"\n>> NEXT: {what} -- press Enter to continue (Ctrl-C aborts): ")
        except EOFError:
            raise Aborted("stdin closed; use --auto for unattended runs")

    def vagrant(self, *argv):
        """Run vagrant in lab/ with the offline communicator enabled.

        $ cd lab && VAGRANT_EXPERIMENTAL=none_communicator vagrant snapshot restore NAME
        """
        subprocess.run(["vagrant", *argv], cwd=LAB_DIR, env=self.env, check=True)

    def vbox(self, *argv, capture=False, check=True):
        """Run VBoxManage.

        $ VBoxManage showvminfo Win10_Malware_Mem_Demo --machinereadable
        """
        return subprocess.run(["VBoxManage", *argv], check=check,
                              capture_output=capture, text=True)

    # ---- stages --------------------------------------------------------
    def restore(self):
        """Stage 1 (and the final revert): back to the clean snapshot; boots the VM.

        $ VAGRANT_EXPERIMENTAL=none_communicator vagrant snapshot restore clean_base_state_7zip
        """
        self.vagrant("snapshot", "restore", self.a.snapshot)

    def verify_isolation(self):
        """Stage 2: refuse to go on if the VM has any NIC or NAT port-forward.

        $ VBoxManage showvminfo Win10_Malware_Mem_Demo --machinereadable | grep -E '^(nic[1-8]|natpf)'
        # expected: nic1="none" ... nic8="none", and no natpf lines
        """
        info = self.vbox("showvminfo", self.a.vm, "--machinereadable", capture=True).stdout
        lines = set(info.splitlines())
        if self.a.network == "none":
            for n in range(1, 9):
                if f'nic{n}="none"' not in lines:
                    raise RuntimeError(f"adapter {n} is not disabled")
            if re.search(r"^natpf[1-8]=", info, re.M):
                raise RuntimeError("a NAT port-forward rule is configured")
            print("OK: all 8 adapters disabled, no NAT forwards.")
            return

        # Fake internet: the guest's only adapter is on the internal network, and
        # the gateway VM (INetSim) has no adapter other than that network.
        self.check_only_sandbox_net(self.a.vm, lines, info)
        gw_info = self.vbox("showvminfo", self.a.gateway, "--machinereadable", capture=True).stdout
        if 'VMState="running"' not in gw_info:
            raise RuntimeError(f"gateway VM {self.a.gateway} is not running")
        self.check_only_sandbox_net(self.a.gateway, set(gw_info.splitlines()), gw_info)
        print(f"OK: {self.a.vm} and {self.a.gateway} only on internal network "
              f"'{SANDBOX_NET}', no NAT/bridged/host-only adapters.")

    def check_only_sandbox_net(self, vm, lines, info):
        """Exactly one adapter, type intnet on SANDBOX_NET; every other adapter none."""
        in_use = [n for n in range(1, 9) if f'nic{n}="none"' not in lines]
        if len(in_use) != 1:
            raise RuntimeError(f"{vm}: expected exactly one adapter, found {in_use}")
        n = in_use[0]
        if f'nic{n}="intnet"' not in lines or f'intnet{n}="{SANDBOX_NET}"' not in lines:
            raise RuntimeError(f"{vm}: adapter {n} is not internal network '{SANDBOX_NET}'")
        if re.search(r"^natpf[1-8]=", info, re.M):
            raise RuntimeError(f"{vm}: a NAT port-forward rule is configured")

    def deliver(self, zip_path):
        """Stage 3: copy the still-locked zip into the guest, unzip and launch it there.

        $ VBoxManage guestcontrol VM mkdir --parents 'C:\\Sandbox' --username vagrant --passwordfile PW
        $ VBoxManage guestcontrol VM copyto --target-directory 'C:\\Sandbox\\' sample.zip launch_sample.ps1 \\
              --username vagrant --passwordfile PW
        $ VBoxManage guestcontrol VM run --username vagrant --passwordfile PW \\
              --exe powershell.exe --wait-stdout --wait-stderr -- powershell.exe \\
              -File C:\\Sandbox\\launch_sample.ps1 -Zip C:\\Sandbox\\sample.zip -User vagrant -Password infected
        """
        a = self.a
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pw_file = tmp / "guest.pw"
            pw_file.write_text(a.guest_password)
            pw_file.chmod(0o600)
            (tmp / "sample.zip").write_bytes(zip_path.read_bytes())  # normalised name
            (tmp / "launch_sample.ps1").write_text(LAUNCH_PS1)

            def gc(sub, *argv, check=True):
                return self.vbox("guestcontrol", a.vm, sub, "--username", a.guest_user,
                                 "--passwordfile", str(pw_file), *argv, check=check,
                                 capture=True if not check else False)

            print(f"Waiting for Guest Additions (up to {a.boot_wait}s)...")
            deadline = time.monotonic() + a.boot_wait
            while gc("mkdir", "--parents", GUEST_DIR, check=False).returncode != 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("guest did not answer guestcontrol: check Guest "
                                       "Additions, credentials and that the user is logged in")
                time.sleep(5)

            self.pause("copy the still-locked zip into the guest")

            def guest_cmd(line):
                r = gc("run", "--exe", CMD, "--wait-stdout", "--timeout", "60000", "--",
                       "cmd.exe", "/c", line, check=False)
                return (r.stdout or "") if r.returncode == 0 else ""

            def guest_sha256(path):
                m = re.search(r"\b[0-9a-f]{64}\b", guest_cmd(f"certutil -hashfile {path} SHA256").lower())
                return m.group(0) if m else None

            def put(src):
                dest = GUEST_DIR + "\\" + src.name
                want = hashlib.sha256(src.read_bytes()).hexdigest()
                gc("copyto", "--target-directory", GUEST_DIR + "\\", str(src), check=False)
                if guest_sha256(dest) == want:
                    return
                print(f"copyto of {src.name} corrupt or failed, resending in {PIECE // 1024} KiB pieces")
                parts = GUEST_DIR + "\\parts"
                guest_cmd(f"rmdir /s /q {parts} & mkdir {parts}")
                data = src.read_bytes()
                for n, i in enumerate(range(0, len(data), PIECE)):
                    piece = tmp / f"p{n:05d}"
                    piece.write_bytes(data[i:i + PIECE])
                    gc("copyto", "--target-directory", parts + "\\", str(piece), check=False)
                    piece.unlink()
                guest_cmd(f"cd /d {parts} && copy /b /y p* {dest}")
                if guest_sha256(dest) != want:
                    raise RuntimeError(f"{src.name} still differs from the host copy after resending")

            put(tmp / "sample.zip")
            put(tmp / "launch_sample.ps1")

            self.pause("unzip inside the guest and launch the sample")
            ps_args = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       GUEST_DIR + "\\launch_sample.ps1", "-Zip", GUEST_DIR + "\\sample.zip",
                       "-User", a.guest_user]
            if a.password:
                ps_args += ["-Password", a.password]
            if a.run:
                ps_args += ["-Run", a.run]
            gc("run", "--exe", POWERSHELL, "--wait-stdout", "--wait-stderr",
               "--timeout", "120000", "--", "powershell.exe", *ps_args)

    def deliver_benign(self, profile):
        """Wait for the guest, then start the profile's programs as the interactive user."""
        a = self.a
        with tempfile.TemporaryDirectory() as tmp:
            pw_file = Path(tmp) / "guest.pw"
            pw_file.write_text(a.guest_password)
            pw_file.chmod(0o600)

            def gc(sub, *argv, check=True):
                return self.vbox("guestcontrol", a.vm, sub, "--username", a.guest_user,
                                 "--passwordfile", str(pw_file), *argv, check=check,
                                 capture=True if not check else False)

            print(f"Waiting for Guest Additions (up to {a.boot_wait}s)...")
            deadline = time.monotonic() + a.boot_wait
            while gc("mkdir", "--parents", GUEST_DIR, check=False).returncode != 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("guest did not answer guestcontrol")
                time.sleep(5)
            for n, cmdline in enumerate(BENIGN_PROFILES[profile]):
                exe = cmdline.split(".exe")[0] + ".exe"
                task = f"SbxBenign{n}"
                line = (f'if exist "{exe}" (schtasks /Create /F /TN {task} /SC ONCE /ST 23:59 '
                        f'/RU {a.guest_user} /IT /TR "{cmdline}" && schtasks /Run /TN {task} '
                        f'&& schtasks /Delete /F /TN {task})')
                gc("run", "--exe", CMD, "--wait-stdout", "--timeout", "60000", "--",
                   "cmd.exe", "/c", line, check=False)
                print(f"Started {cmdline}")
                time.sleep(3)

    def poweroff(self):
        """Last step of the whole run: power the (already reverted) VM off.

        $ VBoxManage controlvm Win10_Malware_Mem_Demo poweroff
        """
        print("Powering off the VM...")
        self.vbox("controlvm", self.a.vm, "poweroff", check=False)

    def save_tasklist(self, sample, when):
        """With --tasklist: save the guest's process list next to the dumps.

        $ VBoxManage guestcontrol Win10_Malware_Mem_Demo run --username vagrant --passwordfile PW \\
              --exe cmd.exe --wait-stdout -- cmd.exe /c "tasklist /v"
        """
        a = self.a
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        reports = {
            f"{sample}_tasklist_{when}.txt": "tasklist /v",
            # crashes and .NET errors from the sample show up here
            f"{sample}_events_{when}.txt": "wevtutil qe Application /c:20 /rd:true /f:text",
        }
        with tempfile.TemporaryDirectory() as tmp:
            pw_file = Path(tmp) / "guest.pw"
            pw_file.write_text(a.guest_password)
            pw_file.chmod(0o600)
            for name, command in reports.items():
                r = self.vbox("guestcontrol", a.vm, "run", "--username", a.guest_user,
                              "--passwordfile", str(pw_file), "--exe", CMD, "--wait-stdout",
                              "--timeout", "60000", "--", "cmd.exe", "/c", command,
                              capture=True, check=False)
                (out / name).write_text(r.stdout or "")

    def dump(self, sample):
        """Stage 4: take --count memory dumps of the running VM, --interval seconds apart.

        $ VBoxManage debugvm Win10_Malware_Mem_Demo dumpvmcore --filename dumps/SAMPLE_dump_1_TIMESTAMP.raw
        With --pause each dump is wrapped in `controlvm pause` ... `controlvm resume`, so the VM
        window shows the dimmed "Paused" state while memory is captured.
        """
        a = self.a
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(1, a.count + 1):
            target = out / f"{sample}_dump_{i}_{int(time.time())}.raw"
            print(f" -> Dump {i}/{a.count}: {target}")
            if a.pause:
                self.vbox("controlvm", a.vm, "pause")
            try:
                self.vbox("debugvm", a.vm, "dumpvmcore", "--filename", str(target))
            finally:
                if a.pause:
                    self.vbox("controlvm", a.vm, "resume")
            if i < a.count:
                time.sleep(a.interval)

    # ---- one sample ----------------------------------------------------
    def process(self, sample, zip_path):
        """One sample end to end: restore -> verify -> deliver -> dump -> revert (always).

        $ ./run_sample.py foo.zip            # pauses for Enter before each stage
        $ ./run_sample.py --auto foo.zip     # no pauses
        (main() then powers the VM off once all samples are done.)
        """
        self.pause(f"[{sample}] restore the clean snapshot '{self.a.snapshot}'")
        self.restore()
        aborted = False
        try:
            self.pause(f"[{sample}] verify the VM has no network path")
            self.verify_isolation()

            if self.a.benign:
                self.deliver_benign(self.a.benign)
                print(f"Letting the benign '{self.a.benign}' profile run for {self.a.settle}s before the first dump...")
                time.sleep(self.a.settle)
            elif zip_path:
                self.deliver(zip_path)
                print(f"Letting the sample run for {self.a.settle}s before the first dump...")
                time.sleep(self.a.settle)
            else:
                input(f"\n>> Start '{sample}' by hand in the VM GUI, then press Enter: ")

            if self.a.tasklist:
                self.save_tasklist(sample, "start")
            self.pause(f"[{sample}] capture {self.a.count} memory dumps")
            self.dump(sample)
            if self.a.tasklist:
                self.save_tasklist(sample, "end")
        except (KeyboardInterrupt, Aborted):
            aborted = True
            raise
        finally:
            if not aborted:
                self.pause(f"[{sample}] revert to the clean snapshot (wipes the sample)")
            print("Reverting to the clean snapshot...")
            try:
                self.restore()
            except subprocess.CalledProcessError:
                print(f"WARNING: snapshot restore failed; restore '{self.a.snapshot}' manually.",
                      file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zips", nargs="*", type=Path, help="password-protected sample zip(s)")
    p.add_argument("--auto", action="store_true", help="no pauses between stages")
    p.add_argument("--sample", help="dump label (default: zip name); required without zips")
    p.add_argument("--password", default="infected", help="zip password (default: infected)")
    p.add_argument("--run", help="file inside the zip to launch (default: the only launchable file)")
    p.add_argument("--count", type=int, default=10, help="dumps per sample (default: 10)")
    p.add_argument("--interval", type=int, default=15, help="seconds between dumps (default: 15)")
    p.add_argument("--settle", type=int, default=20, help="seconds to run before first dump (default: 20)")
    p.add_argument("--pause", action="store_true",
                   help="pause the VM during each dump (screen dims) and resume afterwards")
    p.add_argument("--output", default=str(LAB_DIR / "dumps"), help="dump directory (default: ./dumps)")
    p.add_argument("--vm", default="Win10_Malware_Mem_Demo")
    p.add_argument("--snapshot", default="clean_base_state_7zip")
    p.add_argument("--network", choices=["none", "fake"], default="none",
                   help="none: guest has no adapter (default); fake: guest on internal "
                        "network sbxnet with the INetSim gateway VM")
    p.add_argument("--gateway", default="Sandbox_GW", help="INetSim gateway VM (--network fake)")
    p.add_argument("--tasklist", action="store_true",
                   help="save the guest process list before the first and after the last dump")
    p.add_argument("--benign", choices=sorted(BENIGN_PROFILES),
                   help="no sample: run a benign profile; label the dumps with --sample Benign-<profile>")
    p.add_argument("--guest-user", default="vagrant")
    p.add_argument("--boot-wait", type=int, default=300, help="seconds to wait for Guest Additions")
    a = p.parse_args()
    a.guest_password = os.environ.get("VBOX_GUEST_PASSWORD", "vagrant")  # not on argv

    if a.benign and (a.zips or not a.sample):
        p.error("--benign takes no zip and needs --sample NAME")
    if not a.zips and not a.sample:
        p.error("give at least one zip, or --sample NAME to start a sample by hand")
    if a.sample and len(a.zips) > 1:
        p.error("--sample labels a single sample; omit it to name dumps after each zip")
    if min(a.count, a.interval, a.settle, a.boot_wait) < 1:
        p.error("--count, --interval, --settle and --boot-wait must be positive")
    if not PASSWORD_RE.match(a.password):
        p.error("password may only contain letters, digits and . _ @ % + = -")
    if a.run and not NAME_RE.match(a.run):
        p.error("--run must be a bare file name (letters, digits, . _ -)")
    if not NAME_RE.match(a.guest_user):
        p.error("--guest-user has unsupported characters")
    if a.sample and not NAME_RE.match(a.sample):
        p.error("--sample may only contain letters, digits, '.', '_' and '-'")
    for z in a.zips:
        if not z.is_file() or z.read_bytes()[:2] != b"PK":
            p.error(f"not a zip archive: {z}")
    for tool in ("VBoxManage", "vagrant"):
        if not shutil.which(tool):
            p.error(f"{tool} was not found in PATH")
    return a


WARNING_BANNER = """\
============================== WARNING ==============================
This runs REAL malware. No sandbox is 100% safe: VM escape bugs exist,
and guestcontrol keeps a host-guest channel open while the sample runs.
Use a dedicated, patched host that is off the LAN and holds no
sensitive data. For personal, learning and research use only.
Provided "as is", without warranty. You are fully responsible for
what you run. Read run_sample.py and the lab README first.
====================================================================="""


def main():
    a = parse_args()
    print(WARNING_BANNER)
    lab = Lab(a)
    jobs = [(a.sample or re.sub(r"[^A-Za-z0-9._-]", "_", z.stem), z) for z in a.zips] \
        or [(a.sample, None)]
    failed = []
    try:
        for sample, z in jobs:
            try:
                lab.process(sample, z)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                print(f"FAILED {sample}: {e}", file=sys.stderr)
                failed.append(sample)
    except (KeyboardInterrupt, Aborted) as e:
        print(f"\nAborted ({e or 'interrupted'}); the snapshot restore was attempted.")
        lab.poweroff()
        return 130
    lab.poweroff()
    print(f"\nFinished: {len(jobs) - len(failed)} ok, {len(failed)} failed. Dumps in {a.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
