#!/usr/bin/env bash
# Collect dumps, extract features and delete the raw dumps, one sample at a time.
#
# Usage: lab/collect_all.sh zip [zip ...]  
# Per sample: run_sample.py -> extract_features.py -> rm raw dumps + cache.
# Resumable: a sample with lab/features_online/<name>.csv is skipped.
# Needs ~25 GB free; stops if less.  Logs go to lab/batch_logs_online/.
set -u

LAB="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$LAB")"
DUMPS="$LAB/dumps_online_batch"
FEATURES="$LAB/features_online"
LOGS="$LAB/batch_logs_online"
# Guest on the fake-internet network (INetSim gateway Sandbox_GW must be running).
RUN_ARGS=(--network fake --snapshot clean_base_state_net)
MIN_FREE_GB=25
export VOL2_PYTHON="${VOL2_PYTHON:-$HOME/miniforge3/envs/vol2/bin/python}"
export VOL2_PY="${VOL2_PY:-$HOME/tools/volatility/vol.py}"
export VOL3="${VOL3:-$HOME/miniforge3/envs/vol3/bin/vol}"

mkdir -p "$DUMPS" "$FEATURES" "$LOGS"
cd "$ROOT" || exit 1

if [ $# -gt 0 ]; then
    zips=("$@")
else
    zips=()
    for zip in "$LAB"/samples/*-[0-9]*.zip; do
        case "$(basename "$zip")" in Trojan-Reconyc-*|Trojan-Refroso-*) continue ;; esac
        zips+=("$zip")
    done
fi

ok=0; failed=0; skipped=0
for zip in "${zips[@]}"; do
    name="$(basename "$zip" .zip)"
    if [ -s "$FEATURES/$name.csv" ]; then
        echo "[$name] already done, skipping"; skipped=$((skipped + 1)); continue
    fi
    free_gb=$(df --output=avail -BG "$LAB" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "[$name] only ${free_gb} GB free (< $MIN_FREE_GB), stopping"; break
    fi

    echo "[$name] collecting dumps"
    if python3 -I "$LAB/run_sample.py" --auto "${RUN_ARGS[@]}" --count 10 --interval 15 --settle 20 \
            --output "$DUMPS" "$zip" > "$LOGS/$name.run.log" 2>&1 \
        && [ "$(ls "$DUMPS/${name}_dump_"*.raw 2>/dev/null | wc -l)" -eq 10 ]; then
        echo "[$name] extracting features"
        if python3 -I "$LAB/extract_features.py" "$DUMPS" --output "$FEATURES/$name.csv" \
                --reference-csv "$ROOT/MalMem2022.csv" --jobs 3 > "$LOGS/$name.extract.log" 2>&1 \
            && [ -s "$FEATURES/$name.csv" ]; then
            ok=$((ok + 1)); echo "[$name] ok"
        else
            rm -f "$FEATURES/$name.csv"; failed=$((failed + 1)); echo "[$name] FAILED (extract), see $LOGS/$name.extract.log"
        fi
    else
        failed=$((failed + 1)); echo "[$name] FAILED (run), see $LOGS/$name.run.log"
    fi
    # Always free the disk before the next sample.
    rm -f "$DUMPS/${name}_dump_"*.raw
    rm -rf "$DUMPS/.vol_cache"
done

echo "Done: $ok ok, $failed failed, $skipped skipped"
echo "Merge: python3 -I -c \"import pandas as pd,glob; pd.concat(map(pd.read_csv,sorted(glob.glob('$FEATURES/*.csv')))).to_csv('$LAB/features_all.csv',index=False)\""
