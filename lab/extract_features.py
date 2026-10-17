#!/usr/bin/env python3
"""Extract the 55 CIC-MalMem-2022 features from raw memory dumps with Volatility 2.

Pipeline per dump: run 9 Volatility 2 plugins (JSON output, cached) -> compute
features -> append one row to a CSV with the same layout as MalMem2022.csv:
55 feature columns, then Class, Category, Filename.

The feature formulas are ported from VolMemLyzer tag V1.0.0
(VolatilityFeatureExtractor.py, https://github.com/ahlashkari/VolMemLyzer, GPLv3),
the Volatility 2 extractor that defines the dataset's column names. Quirks of the
original are kept on purpose so values stay comparable with MalMem2022.csv.

Use Volatility 2 master (not the 2.6.1 release): the guest is Windows 10 2004
(build 19041), which 2.6.1 cannot read. Volatility 2 needs Python 2.7; point --vol-python / --vol-py (or the env vars
VOL2_PYTHON / VOL2_PY) at an interpreter and its vol.py. The label of a dump is
the part of the file name before "_dump_" (see run_sample.py).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = "Win10x64_19041"
META_COLUMNS = ["Class", "Category", "Filename"]


def rc2kv(rc):
    keys = rc["columns"]
    return [dict(zip(keys, r)) for r in rc["rows"]]


def count(rows, key, value):
    return sum(1 for r in rows if r[key] == value)


def avg(total, n):
    if n == 0:
        raise ValueError("no rows")
    return total / n


def extract_pslist(rc):
    procs = rc2kv(rc)
    n = len(procs)
    return {
        "pslist.nproc": n,
        "pslist.nppid": len({p["PPID"] for p in procs}),
        "pslist.avg_threads": avg(sum(p["Thds"] for p in procs), n),
        "pslist.nprocs64bit": sum(p["Wow64"] for p in procs),
        # Vol2 reports Hnds as 0/-1 on newer builds; None here triggers the fallback in extract_dump.
        "pslist.avg_handlers": avg(sum(h), len(h)) if (h := [p["Hnds"] for p in procs if p["Hnds"] > 0]) else None,
    }


def extract_dlllist(rc):
    dlls = rc2kv(rc)
    return {
        "dlllist.ndlls": len(dlls),
        "dlllist.avg_dlls_per_proc": avg(len(dlls), len({d["Pid"] for d in dlls})),
    }


HANDLE_TYPES = [
    ("nport", "Port"), ("nfile", "File"), ("nevent", "Event"), ("ndesktop", "Desktop"),
    ("nkey", "Key"), ("nthread", "Thread"), ("ndirectory", "Directory"),
    ("nsemaphore", "Semaphore"), ("ntimer", "Timer"), ("nsection", "Section"),
    ("nmutant", "Mutant"),
]


def extract_handles(rc):
    handles = rc2kv(rc)
    feats = {
        "handles.nhandles": len(handles),
        "handles.avg_handles_per_proc": avg(len(handles), len({h["Pid"] for h in handles})),
    }
    for name, htype in HANDLE_TYPES:
        feats["handles." + name] = count(handles, "Type", htype)
    return feats


def extract_ldrmodules(rc):
    mods = rc2kv(rc)
    feats = {}
    for name, key in (("load", "InLoad"), ("init", "InInit"), ("mem", "InMem")):
        feats["ldrmodules.not_in_" + name] = count(mods, key, "False")
    for name, key in (("load", "InLoad"), ("init", "InInit"), ("mem", "InMem")):
        # `x / n or 1` as in the original: an average of exactly 0 becomes 1.
        feats["ldrmodules.not_in_%s_avg" % name] = avg(count(mods, key, "False"), len(mods)) or 1
    return feats


def malfind_flags(row):
    flags = {}
    for entry in row["Flags"].split(", "):
        if ": " in entry:
            k, v = entry.split(": ", 1)
            flags[k] = v
    return flags


def extract_malfind(rc):
    finds = rc2kv(rc)
    flags = [malfind_flags(f) for f in finds]
    n_pids = len({f["Pid"] for f in finds})
    return {
        "malfind.ninjections": len(finds),
        "malfind.commitCharge": sum(int(f.get("CommitCharge", 0)) for f in flags),
        "malfind.protection": sum(int(f.get("Protection", 0)) for f in flags),
        # the original divides by zero when malfind finds nothing; report 0 instead
        "malfind.uniqueInjections": len(finds) / n_pids if n_pids else 0,
    }


PSXVIEW_VIEWS = [
    ("pslist", "pslist"), ("eprocess_pool", "psscan"), ("ethread_pool", "thrdproc"),
    ("pspcid_list", "pspcid"), ("csrss_handles", "csrss"), ("session", "session"),
    ("deskthrd", "deskthrd"),
]


def extract_psxview(rc):
    procs = rc2kv(rc)
    feats = {}
    for name, key in PSXVIEW_VIEWS:
        feats["psxview.not_in_" + name] = count(procs, key, "False")
    for name, key in PSXVIEW_VIEWS:
        feats["psxview.not_in_%s_false_avg" % name] = avg(count(procs, key, "False"), len(procs))
    return feats


def extract_modules(rc):
    return {"modules.nmodules": len(rc["rows"])}


def extract_svcscan(rc):
    svcs = rc2kv(rc)
    return {
        "svcscan.nservices": len(svcs),
        "svcscan.kernel_drivers": count(svcs, "ServiceType", "SERVICE_KERNEL_DRIVER"),
        "svcscan.fs_drivers": count(svcs, "ServiceType", "SERVICE_FILE_SYSTEM_DRIVER"),
        "svcscan.process_services": count(svcs, "ServiceType", "SERVICE_WIN32_OWN_PROCESS"),
        "svcscan.shared_process_services": count(svcs, "ServiceType", "SERVICE_WIN32_SHARE_PROCESS"),
        "svcscan.interactive_process_services": count(svcs, "ServiceType", "SERVICE_INTERACTIVE_PROCESS"),
        "svcscan.nactive": count(svcs, "State", "SERVICE_RUNNING"),
    }


def extract_callbacks(rc):
    cbs = rc2kv(rc)
    return {
        "callbacks.ncallbacks": len(cbs),
        "callbacks.nanonymous": count(cbs, "Module", "UNKNOWN"),
        "callbacks.ngeneric": count(cbs, "Type", "GenericKernelCallback"),
    }


# Order matters: it is the column order of MalMem2022.csv.
PLUGINS = {
    "pslist": extract_pslist,
    "dlllist": extract_dlllist,
    "handles": extract_handles,
    "ldrmodules": extract_ldrmodules,
    "malfind": extract_malfind,
    "psxview": extract_psxview,
    "modules": extract_modules,
    "svcscan": extract_svcscan,
    "callbacks": extract_callbacks,
}


def run_plugin(args, dump, plugin, cache_dir):
    """Return the parsed Volatility 2 JSON for one plugin, using the cache."""
    cached = cache_dir / (plugin + ".json")
    if cached.is_file() and cached.stat().st_size > 0:
        return json.loads(cached.read_text())
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        out = Path(tmp) / (plugin + ".json")
        cmd = [args.vol_python, args.vol_py, "-f", str(dump), "--profile=" + args.profile,
               "--output=json", "--output-file=" + str(out), plugin]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            rc = json.loads(out.read_text())
            rc["columns"], rc["rows"]
        except (OSError, ValueError, KeyError):
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            raise RuntimeError("%s failed (exit %d): %s" % (plugin, proc.returncode, " | ".join(tail)))
        out.replace(cached)
    return rc


def extract_dump(args, dump):
    cache_dir = args.cache_dir / dump.stem
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {p: pool.submit(run_plugin, args, dump, p, cache_dir) for p in PLUGINS}
    feats = {}
    failed = []
    for plugin, extractor in PLUGINS.items():
        try:
            feats.update(extractor(futures[plugin].result()))
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            # An unusable plugin leaves its columns blank; they are dropped at the end
            # if no dump produced a value.
            print("  WARNING: %s: %s: %s" % (plugin, type(e).__name__, e), file=sys.stderr)
            failed.append(plugin)
    if len(failed) == len(PLUGINS):
        raise RuntimeError("no plugin produced usable output")
    if feats.get("pslist.avg_handlers") is None and "handles.nhandles" in feats and "pslist.nproc" in feats:
        # pslist's Hnds is unusable on newer builds; the handles plugin gives the real count.
        feats["pslist.avg_handlers"] = feats["handles.nhandles"] / feats["pslist.nproc"]
        print("  note: pslist.avg_handlers computed from the handles plugin", file=sys.stderr)
    return feats


def label_of(dump):
    return dump.stem.split("_dump_")[0]


def meta_of(dump):
    label = label_of(dump)
    return {
        "Class": "Benign" if label.lower() == "benign" else "Malware",
        "Category": label,
        "Filename": dump.name,
    }


def feature_columns():
    cols = []
    for plugin in PLUGINS:
        cols.extend(COLUMNS_BY_PLUGIN[plugin])
    return cols


COLUMNS_BY_PLUGIN = {
    "pslist": ["pslist." + c for c in ("nproc", "nppid", "avg_threads", "nprocs64bit", "avg_handlers")],
    "dlllist": ["dlllist." + c for c in ("ndlls", "avg_dlls_per_proc")],
    "handles": ["handles." + c for c in ["nhandles", "avg_handles_per_proc"] + [n for n, _ in HANDLE_TYPES]],
    "ldrmodules": ["ldrmodules." + c for c in (
        "not_in_load", "not_in_init", "not_in_mem", "not_in_load_avg", "not_in_init_avg", "not_in_mem_avg")],
    "malfind": ["malfind." + c for c in ("ninjections", "commitCharge", "protection", "uniqueInjections")],
    "psxview": (["psxview.not_in_" + n for n, _ in PSXVIEW_VIEWS]
                + ["psxview.not_in_%s_false_avg" % n for n, _ in PSXVIEW_VIEWS]),
    "modules": ["modules.nmodules"],
    "svcscan": ["svcscan." + c for c in (
        "nservices", "kernel_drivers", "fs_drivers", "process_services",
        "shared_process_services", "interactive_process_services", "nactive")],
    "callbacks": ["callbacks." + c for c in ("ncallbacks", "nanonymous", "ngeneric")],
}


def check_reference(columns, reference):
    with open(reference, newline="") as f:
        ref = next(csv.reader(f))
    if ref != columns:
        missing = [c for c in ref if c not in columns]
        extra = [c for c in columns if c not in ref]
        sys.exit("Header differs from %s (missing: %s, extra: %s, same set but other order: %s)"
                 % (reference, missing, extra, not missing and not extra))


def find_dumps(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".raw")


def done_filenames(output):
    if not output.is_file():
        return set()
    with open(output, newline="") as f:
        return {row["Filename"] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dumps", type=Path, nargs="?", default=LAB_DIR / "dumps",
                    help="dump file or directory of *.raw (default: lab/dumps)")
    ap.add_argument("--output", type=Path, default=LAB_DIR / "features.csv")
    ap.add_argument("--vol-python", default=os.environ.get("VOL2_PYTHON"),
                    help="Python 2.7 interpreter that runs Volatility 2 (env VOL2_PYTHON)")
    ap.add_argument("--vol-py", default=os.environ.get("VOL2_PY"),
                    help="path to Volatility 2 vol.py (env VOL2_PY)")
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    help="Volatility profile (default: %(default)s)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="raw plugin JSON cache (default: <dumps dir>/.vol_cache)")
    ap.add_argument("--reference-csv", type=Path, default=None,
                    help="MalMem2022.csv; abort if our 55 feature columns do not match its header")
    ap.add_argument("--resume", action="store_true", help="skip dumps already in --output")
    ap.add_argument("--jobs", type=int, default=4, help="plugins run in parallel per dump")
    args = ap.parse_args()

    if not args.vol_python or not args.vol_py:
        ap.error("set --vol-python and --vol-py (or VOL2_PYTHON / VOL2_PY)")
    dumps = find_dumps(args.dumps)
    if not dumps:
        ap.error("no .raw dumps found in %s" % args.dumps)
    if args.cache_dir is None:
        args.cache_dir = (args.dumps if args.dumps.is_dir() else args.dumps.parent) / ".vol_cache"

    columns = feature_columns() + META_COLUMNS
    if args.reference_csv:
        check_reference(columns, args.reference_csv)

    skip = done_filenames(args.output) if args.resume else set()
    new_file = not args.output.is_file() or not args.resume
    mode = "w" if new_file else "a"
    failed = []
    with open(args.output, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        for i, dump in enumerate(dumps, 1):
            tag = "[%d/%d] %s" % (i, len(dumps), dump.name)
            if dump.name in skip:
                print(tag, "- already done, skipping")
                continue
            print(tag, flush=True)
            try:
                row = extract_dump(args, dump)
            except RuntimeError as e:
                print("  FAILED:", e, file=sys.stderr)
                failed.append(dump.name)
                continue
            row.update(meta_of(dump))
            writer.writerow(row)
            f.flush()
    print("Wrote %s (%d failed)" % (args.output, len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
