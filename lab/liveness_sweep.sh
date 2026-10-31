#!/usr/bin/env bash
# Launch each sample for ~60 s in the sandbox (one throw-away dump) and record whether
# its process is still alive.  Results: lab/liveness/summary.tsv
#
# Usage: lab/liveness_sweep.sh [zip ...]   (default: every <Category>-<Family>-<n>.zip in lab/samples)
# Resumable: a sample that already has a row in summary.tsv is skipped.
set -u

LAB="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$LAB")"
OUT="$LAB/liveness"
SUMMARY="$OUT/summary.tsv"

mkdir -p "$OUT"
cd "$ROOT" || exit 1
[ -f "$SUMMARY" ] || printf 'sample\tlaunched_exe\talive_at_end\n' > "$SUMMARY"

if [ $# -gt 0 ]; then
    zips=("$@")
else
    zips=()
    for zip in "$LAB"/samples/*-[0-9]*.zip; do
        case "$(basename "$zip")" in Trojan-Reconyc-*|Trojan-Refroso-*) continue ;; esac
        zips+=("$zip")
    done
fi

for zip in "${zips[@]}"; do
    name="$(basename "$zip" .zip)"
    if cut -f1 "$SUMMARY" | grep -qx "$name"; then
        echo "[$name] already done, skipping"; continue
    fi
    echo "[$name] launching"
    python3 -I "$LAB/run_sample.py" --auto --network fake --snapshot clean_base_state_net \
        --count 1 --settle 60 --tasklist --output "$OUT" "$zip" > "$OUT/$name.run.log" 2>&1
    rm -f "$OUT"/*.raw
    exe="$(grep -o 'Launched C:.*' "$OUT/$name.run.log" | sed 's/.*\\//' | tr -d '\r')"
    # tasklist truncates image names to 25 characters
    if [ -n "$exe" ] && grep -qi -F "${exe:0:25}" "$OUT/${name}_tasklist_end.txt" 2>/dev/null; then
        alive=yes
    else
        alive=no
    fi
    printf '%s\t%s\t%s\n' "$name" "${exe:-none}" "$alive" >> "$SUMMARY"
    echo "[$name] alive_at_end=$alive"
done
echo "Done. See $SUMMARY"
