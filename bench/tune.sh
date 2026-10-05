#!/bin/bash
# usage: tune.sh <label>  — freezes target/release/augur, sizes every file in
# data/tune in parallel, writes work/<label>.tsv, and diffs against work/base.tsv.
set -e
cd "$(dirname "$0")"
L=${1:-run}
cp ../target/release/augur work/augur-$L
ls data/tune/* | xargs -P ${P:-4} -n 1 ./work/augur-$L size | sed 's#data/tune/##' | sort > work/$L.tsv
python3 - "$L" <<'PY'
import sys
L=sys.argv[1]
def rd(n):
    try: return dict((a,int(b)) for a,b in (l.split() for l in open(f'work/{n}.tsv')))
    except FileNotFoundError: return {}
cur=rd(L); base=rd('base'); zp=rd('zpaq')
tc=tb=0; pct=[]
for k in sorted(cur):
    b=base.get(k); z=zp.get(k)
    d=f"{(b-cur[k])/b*100:+.3f}%" if b else ""
    vz=f"{(z-cur[k])/z*100:+.2f}%" if z else ""
    print(f"{k:12} {cur[k]:>10} {d:>9}   vs zpaq {vz}")
    tc+=cur[k]; tb+=b or 0
    if b: pct.append((b-cur[k])/b*100)
if tb: print(f"{'TOTAL':12} {tc:>10} {(tb-tc)/tb*100:+.3f}%   mean/file {sum(pct)/len(pct):+.3f}%")
PY
