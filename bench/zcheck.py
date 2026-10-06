#!/usr/bin/env python3
"""Compare augur's zlib-clone deflate with the system zlib, byte for byte.

usage: zcheck.py [quick]   — runs every (level, memLevel, wbits, strategy)
combination over a set of inputs and reports mismatches.
"""
import zlib, subprocess, sys, os, random, itertools
AUGUR = os.path.join(os.path.dirname(__file__), '../target/release/augur')
W = os.path.join(os.path.dirname(__file__), 'work/z')
os.makedirs(W, exist_ok=True)

def inputs():
    rnd = random.Random(7)
    d = os.path.join(os.path.dirname(__file__), 'data')
    yield 'empty', b''
    yield 'one', b'a'
    yield 'abc', b'abcabcabcabcabc' * 3
    yield 'zeros', bytes(300000)
    yield 'random', bytes(rnd.getrandbits(8) for _ in range(200000))
    txt = open(os.path.join(d, 'silesia/dickens'), 'rb').read()
    yield 'dickens', txt[:700000]
    yield 'xml', open(os.path.join(d, 'silesia/xml'), 'rb').read()[:400000]
    yield 'mozilla', open(os.path.join(d, 'silesia/mozilla'), 'rb').read()[3000000:3600000]
    yield 'img', open(os.path.join(d, 'img/k07g.pgm'), 'rb').read()
    # runs, short repeats, far matches
    yield 'mixed', b''.join(bytes([rnd.randrange(4)]) * rnd.randrange(1, 300) + bytes(rnd.getrandbits(8) for _ in range(rnd.randrange(0, 40))) for _ in range(3000))

def combos(quick):
    levels = range(1, 10)
    mems = [8] if quick else [1, 4, 8, 9]
    wbits = [15] if quick else [9, 12, 15]
    strats = [0, 1, 2, 3, 4]
    for l, m, w, s in itertools.product(levels, mems, wbits, strats):
        if s in (2, 3) and l != 6:
            continue  # huffman-only and RLE ignore the level's match settings
        yield l, m, w, s

quick = len(sys.argv) > 1 and sys.argv[1] == 'quick'
bad = 0; total = 0
for name, data in inputs():
    src = os.path.join(W, name); open(src, 'wb').write(data)
    for l, m, w, s in combos(quick):
        c = zlib.compressobj(l, zlib.DEFLATED, -w, m, s)
        ref = c.compress(data) + c.flush()
        out = os.path.join(W, 'out')
        subprocess.run([AUGUR, 'zdeflate', str(l), str(m), str(w), str(s), src, out], check=True)
        got = open(out, 'rb').read()
        total += 1
        if got != ref:
            bad += 1
            i = next((k for k in range(min(len(got), len(ref))) if got[k] != ref[k]), min(len(got), len(ref)))
            print(f'MISMATCH {name} L{l} M{m} W{w} S{s}: ref {len(ref)} got {len(got)} first diff at {i}')
print(f'{total - bad}/{total} identical')
