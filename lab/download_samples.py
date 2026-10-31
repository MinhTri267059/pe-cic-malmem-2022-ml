#!/usr/bin/env python3
"""Download candidate samples from MalwareBazaar and keep only 64-bit PE files.

For each ``candidates_<family>.csv`` written by find_samples.py, fetch the
password-protected zip (``get_file``), open it in memory with the standard
password, check that the content hash matches the requested SHA256 and that the
PE header says AMD64 (PE32+), then save the original zip as
``lab/samples/<Family>-<n>.zip``.  The sample is never written to disk
unencrypted and never executed.  Requires ``pyzipper``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import struct
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pyzipper


API_URL = "https://mb-api.abuse.ch/api/v1/"
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
ZIP_PASSWORD = b"infected"
MACHINE_AMD64 = 0x8664


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", help="Family such as Ransomware-Conti (repeatable; default: all candidates files)")
    parser.add_argument("--per-family", type=int, default=3, help="64-bit samples to keep per family (default: 3)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between API requests (default: 2)")
    return parser.parse_args()


def fetch_zip(api_key: str, sha256: str) -> bytes | None:
    body = urlencode({"query": "get_file", "sha256_hash": sha256}).encode()
    request = Request(
        API_URL,
        data=body,
        headers={"Auth-Key": api_key, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        data = response.read()
    # Errors come back as JSON text instead of a zip.
    return data if data.startswith(b"PK") else None


def pe_machine(content: bytes) -> int | None:
    """Return the PE COFF machine type, or None if this is not a PE file."""
    if len(content) < 0x40 or content[:2] != b"MZ":
        return None
    (offset,) = struct.unpack_from("<I", content, 0x3C)
    if offset + 6 > len(content) or content[offset : offset + 4] != b"PE\0\0":
        return None
    (machine,) = struct.unpack_from("<H", content, offset + 4)
    return machine


def check_zip(data: bytes, sha256: str) -> str:
    """Return 'ok' or the reason the zip does not hold a matching 64-bit PE."""
    try:
        with pyzipper.AESZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                return f"unexpected_members:{len(names)}"
            content = archive.read(names[0], pwd=ZIP_PASSWORD)
    except (pyzipper.BadZipFile, RuntimeError, ValueError) as error:
        return f"unreadable:{error}"
    if hashlib.sha256(content).hexdigest() != sha256.casefold():
        return "hash_mismatch"
    machine = pe_machine(content)
    if machine is None:
        return "not_pe"
    return "ok" if machine == MACHINE_AMD64 else f"not_x64:0x{machine:04x}"


def next_index(family: str) -> int:
    taken = [int(m.group(1)) for path in SAMPLES_DIR.glob(f"{family}-*.zip") if (m := re.fullmatch(rf"{re.escape(family)}-(\d+)\.zip", path.name))]
    return max(taken, default=0) + 1


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("MALWAREBAZAAR_API_KEY")
    if not api_key:
        raise SystemExit("Set MALWAREBAZAAR_API_KEY in .env or the process environment.")

    families = args.family or sorted(p.stem.removeprefix("candidates_") for p in SAMPLES_DIR.glob("candidates_*.csv"))
    for family in families:
        candidates_path = SAMPLES_DIR / f"candidates_{family}.csv"
        if not candidates_path.is_file():
            print(f"{family}: no candidates file")
            continue
        with candidates_path.open(newline="", encoding="utf-8") as file:
            candidates = list(csv.DictReader(file))

        kept = 0
        for row in candidates:
            if kept >= args.per_family:
                break
            sha256 = row["sha256_hash"]
            try:
                data = fetch_zip(api_key, sha256)
            except (HTTPError, URLError, TimeoutError) as error:
                print(f"{family} {sha256[:12]}: request_error {error}")
                continue
            finally:
                time.sleep(args.delay)
            result = check_zip(data, sha256) if data else "no_zip_returned"
            if result == "ok":
                kept += 1
                target = SAMPLES_DIR / f"{family}-{next_index(family)}.zip"
                target.write_bytes(data)
                print(f"{family} {sha256[:12]}: ok -> {target.name}")
            else:
                print(f"{family} {sha256[:12]}: skipped ({result})")
        print(f"== {family}: kept {kept}/{args.per_family}")


if __name__ == "__main__":
    main()
