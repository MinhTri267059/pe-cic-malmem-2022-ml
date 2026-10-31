#!/usr/bin/env bash
# Benign dumps: for each profile, run guest-native programs, dump, extract features, delete raw dumps.
# Usage: lab/collect_benign.sh [profile ...]   (default: office browse idle)
set -u
LAB="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$LAB")"
DUMPS="$LAB/dumps_online_batch"; FEATURES="$LAB/features_online"; LOGS="$LAB/batch_logs_online"
export VOL2_PYTHON="${VOL2_PYTHON:-$HOME/miniforge3/envs/vol2/bin/python}"
export VOL2_PY="${VOL2_PY:-$HOME/tools/volatility/vol.py}"
export VOL3="${VOL3:-$HOME/miniforge3/envs/vol3/bin/vol}"
mkdir -p "$DUMPS" "$FEATURES" "$LOGS"; cd "$ROOT" || exit 1
profiles=("$@"); [ ${#profiles[@]} -gt 0 ] || profiles=(office browse idle)
for profile in "${profiles[@]}"; do
    name="Benign-$profile"
    [ -s "$FEATURES/$name.csv" ] && { echo "[$name] already done"; continue; }
    echo "[$name] collecting dumps"
    if python3 -I "$LAB/run_sample.py" --auto --network fake --snapshot clean_base_state_net --benign "$profile" \
            --sample "$name" --count 10 --interval 15 --settle 20 --output "$DUMPS" > "$LOGS/$name.run.log" 2>&1 \
        && [ "$(ls "$DUMPS/${name}_dump_"*.raw 2>/dev/null | wc -l)" -eq 10 ]; then
        python3 -I "$LAB/extract_features.py" "$DUMPS" --output "$FEATURES/$name.csv" \
            --reference-csv "$ROOT/MalMem2022.csv" --jobs 3 > "$LOGS/$name.extract.log" 2>&1 \
            && [ -s "$FEATURES/$name.csv" ] && echo "[$name] ok" || { rm -f "$FEATURES/$name.csv"; echo "[$name] FAILED (extract)"; }
    else
        echo "[$name] FAILED (run), see $LOGS/$name.run.log"
    fi
    rm -f "$DUMPS/${name}_dump_"*.raw; rm -rf "$DUMPS/.vol_cache"
done
echo done
