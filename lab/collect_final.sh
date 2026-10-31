#!/usr/bin/env bash
# Full re-collection with Volatility 2 + 3 features: 17 active samples, then 3 benign profiles, then merge.
cd "$(dirname "$0")/.." || exit 1
Z=lab/samples
lab/collect_all.sh $Z/Ransomware-{Conti-1,Conti-2,LockBit-2,LockBit-3,Pysa-1}.zip \
    $Z/Spyware-{AgentTesla-1,AgentTesla-2,Formbook-1,Formbook-2,SnakeKeylogger-3}.zip \
    $Z/Trojan-{TrickBot-2,QuasarRAT-2,QuasarRAT-3,Remcos-1,Remcos-2,Remcos-3,DCRat-1}.zip
lab/collect_benign.sh office browse idle
python3 -I -c "
import csv,glob
rows=[];hdr=None
for f in sorted(glob.glob('lab/features_online/*.csv')):
    r=list(csv.DictReader(open(f))); hdr=hdr or list(r[0]); rows+=r
w=csv.DictWriter(open('lab/features_all.csv','w',newline=''),fieldnames=hdr); w.writeheader(); w.writerows(rows)
print('merged',len(rows))"
echo ALL-DONE
