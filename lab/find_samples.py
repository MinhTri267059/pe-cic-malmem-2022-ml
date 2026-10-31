#!/usr/bin/env python3
"""Search MalwareBazaar for candidate samples per malware family (metadata only).

Calls only the ``get_siginfo`` query and writes one candidates CSV per family
under ``lab/samples/``.  There is no download code path; fetching a sample is a
separate, explicit step.  Candidates are ranked so that samples known to be
64-bit come first, then samples whose architecture the API does not report
(verify those with ``file`` after download -- they must say PE32+).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://mb-api.abuse.ch/api/v1/"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "samples"

# Category-Family -> MalwareBazaar signature names to try, in order.
FAMILIES = {
    "Ransomware-Conti": ["Conti"],
    "Ransomware-Maze": ["Maze"],
    "Ransomware-Pysa": ["Pysa", "Mespinoza"],
    "Ransomware-Ako": ["Ako", "AkoCrypt"],
    "Ransomware-Shade": ["Shade", "Troldesh"],
    "Trojan-Zeus": ["ZeuS", "Zbot"],
    "Trojan-Emotet": ["Emotet"],
    "Trojan-Refroso": ["Refroso"],
    "Trojan-Scar": ["Scar"],
    "Trojan-Reconyc": ["Reconyc"],
    "Spyware-180solutions": ["180solutions"],
    "Spyware-Coolwebsearch": ["CoolWebSearch"],
    "Spyware-Gator": ["Gator"],
    "Spyware-Transponder": ["Transponder"],
    "Spyware-TIBS": ["TIBS"],
    # Modern substitutes: the original spyware families are not MalwareBazaar signatures.
    "Spyware-AgentTesla": ["AgentTesla"],
    "Spyware-Formbook": ["Formbook"],
    "Spyware-RedLine": ["RedLineStealer"],
    "Spyware-SnakeKeylogger": ["SnakeKeylogger"],
    "Spyware-LokiBot": ["Loki"],
    # Modern 64-bit substitutes for ransomware/trojan families with no 64-bit samples.
    "Ransomware-LockBit": ["LockBit"],
    "Ransomware-BlackCat": ["BlackCat"],
    "Ransomware-Akira": ["Akira"],
    "Trojan-QakBot": ["QakBot"],
    "Trojan-TrickBot": ["TrickBot"],
    "Trojan-IcedID": ["IcedID"],
    "Trojan-Dridex": ["Dridex"],
    # Remote-access trojans, as extra 64-bit trojan candidates.
    "Trojan-AsyncRAT": ["AsyncRAT"],
    "Trojan-QuasarRAT": ["QuasarRAT"],
    "Trojan-Remcos": ["RemcosRAT"],
    "Trojan-NjRAT": ["njrat"],
    "Trojan-DCRat": ["DCRat"],
}
COLUMNS = (
    "family",
    "sha256_hash",
    "signature",
    "first_seen",
    "file_name",
    "file_size",
    "file_type",
    "file_arch",
    "arch_guess",
    "tags",
)


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
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES),
                        help="Family to search (repeatable; default: all)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=200, help="Max API results per signature (default: 200)")
    parser.add_argument("--max-size-mb", type=float, default=10.0, help="Skip larger files (default: 10)")
    parser.add_argument("--keep", type=int, default=10, help="Candidates kept per family (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API requests (default: 1)")
    return parser.parse_args()


def query_signature(api_key: str, signature: str, limit: int) -> dict:
    body = urlencode({"query": "get_siginfo", "signature": signature, "limit": limit}).encode()
    request = Request(
        API_URL,
        data=body,
        headers={"Auth-Key": api_key, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def guess_arch(item: dict) -> str:
    """Return x64, x86 or unknown from the API's arch field and tags."""
    text = " ".join([str(item.get("file_arch") or "")] + [str(tag) for tag in item.get("tags") or []]).casefold()
    if re.search(r"x64|x86-64|amd64|64-?bit|pe32\+", text):
        return "x64"
    if re.search(r"x86|i386|32-?bit|pe32", text):
        return "x86"
    return "unknown"


def candidate_row(family: str, item: dict, arch: str) -> dict:
    row = {column: item.get(column, "") for column in COLUMNS if column in item}
    row.update(family=family, arch_guess=arch)
    tags = item.get("tags") or []
    row["tags"] = ";".join(map(str, tags)) if isinstance(tags, list) else str(tags)
    return {column: row.get(column, "") for column in COLUMNS}


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("MALWAREBAZAAR_API_KEY")
    if not api_key:
        raise SystemExit("Set MALWAREBAZAAR_API_KEY in .env or the process environment.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_size = int(args.max_size_mb * 1024 * 1024)
    for family in args.family or list(FAMILIES):
        seen: set[str] = set()
        rows: list[dict] = []
        status = "no_signature_matched"
        for signature in FAMILIES[family]:
            try:
                response = query_signature(api_key, signature, args.limit)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                status = f"request_error: {error}"
                continue
            finally:
                time.sleep(args.delay)
            if response.get("query_status") != "ok":
                status = response.get("query_status", "invalid_response")
                continue
            status = "ok"
            for item in response.get("data") or []:
                sha256 = item.get("sha256_hash", "")
                size = item.get("file_size") or 0
                if sha256 in seen or item.get("file_type") != "exe" or not 0 < size <= max_size:
                    continue
                arch = guess_arch(item)
                if arch == "x86":
                    continue
                seen.add(sha256)
                rows.append(candidate_row(family, item, arch))

        rows.sort(key=lambda row: row["arch_guess"] != "x64")  # stable: known x64 first
        rows = rows[: args.keep]
        output = args.output_dir / f"candidates_{family}.csv"
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        known = sum(row["arch_guess"] == "x64" for row in rows)
        print(f"{family}: {status}, {len(rows)} candidates ({known} known x64) -> {output}")


if __name__ == "__main__":
    main()
