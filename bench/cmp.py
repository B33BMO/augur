#!/usr/bin/env python3
# cmp.py A B — per-file % change from run A to run B (positive = B smaller)
import sys
def rd(n): return dict((a,int(b)) for a,b in (l.split() for l in open(f'work/{n}.tsv')))
a,b=rd(sys.argv[1]),rd(sys.argv[2]); p=[]
for k in sorted(a):
    d=(a[k]-b[k])/a[k]*100; p.append(d); print(f"{k:12} {a[k]:>9} {b[k]:>9} {d:+.3f}%")
ta,tb=sum(a.values()),sum(b.values())
print(f"{'TOTAL':12} {ta:>9} {tb:>9} {(ta-tb)/ta*100:+.3f}%   mean/file {sum(p)/len(p):+.3f}%")
