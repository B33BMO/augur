#!/usr/bin/env python3
"""Run augur against the honest peer set and emit a results table.

Every codec here is run at its strongest practical setting, because augur is
slow by design and comparing it to a fast setting of anything else is not a
comparison. Sizes are absolute compressed bytes; ratio is against the named
input file.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
AUGUR = ROOT.parent / "target" / "release" / "augur"
WORK = ROOT / "work"

# name -> (argv template producing stdout, argv template restoring stdout)
CODECS = {
    "zstd-19":   (["zstd", "-19", "-T0", "-c"],                     ["zstd", "-d", "-c"]),
    "zstd-22":   (["zstd", "--ultra", "-22", "--long=31", "-T0", "-c"],
                                                                    ["zstd", "-d", "--long=31", "-c"]),
    "xz-9e":     (["xz", "-9e", "-T0", "-c"],                       ["xz", "-d", "-c"]),
    "brotli-11": (["brotli", "-q", "11", "--large_window=24", "-c"],
                                                                    ["brotli", "-d", "--large_window=24", "-c"]),
}


def run_stream(argv, src: Path, dst: Path) -> float:
    t0 = time.time()
    with src.open("rb") as fin, dst.open("wb") as fout:
        subprocess.run(argv, stdin=fin, stdout=fout, check=True)
    return time.time() - t0


def bench_codec(name: str, src: Path) -> dict:
    comp_argv, decomp_argv = CODECS[name]
    WORK.mkdir(exist_ok=True)
    packed = WORK / f"{src.name}.{name}"
    restored = WORK / f"{src.name}.{name}.out"
    enc = run_stream(comp_argv, src, packed)
    dec = run_stream(decomp_argv, packed, restored)
    ok = restored.stat().st_size == src.stat().st_size and \
        subprocess.run(["cmp", "-s", str(src), str(restored)]).returncode == 0
    size = packed.stat().st_size
    packed.unlink()
    restored.unlink()
    return {"codec": name, "bytes": size, "enc_s": enc, "dec_s": dec, "verified": ok}


def bench_zpaq(src: Path) -> dict:
    """zpaq is archive-oriented; it stores a path, so run it on a copy named
    identically for every input to keep the filename overhead constant."""
    WORK.mkdir(exist_ok=True)
    arc = WORK / "z.zpaq"
    if arc.exists():
        arc.unlink()
    t0 = time.time()
    subprocess.run(["zpaq", "a", str(arc), str(src), "-m5"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    enc = time.time() - t0
    size = arc.stat().st_size
    arc.unlink()
    return {"codec": "zpaq-m5", "bytes": size, "enc_s": enc, "dec_s": None, "verified": None}


def bench_augur(src: Path) -> dict:
    t0 = time.time()
    out = subprocess.run([str(AUGUR), "bench", str(src)],
                         capture_output=True, text=True, check=True).stdout
    total = time.time() - t0
    # "  6991577 -> 123918 bytes   ratio=56.42x   enc=12.7s (...) dec=12.9s (...) roundtrip=OK"
    line = [l for l in out.splitlines() if "->" in l][0]
    size = int(line.split("->")[1].split("bytes")[0].strip())
    enc = float(line.split("enc=")[1].split("s")[0])
    dec = float(line.split("dec=")[1].split("s")[0])
    return {"codec": "augur", "bytes": size, "enc_s": enc, "dec_s": dec,
            "verified": "roundtrip=OK" in out, "wall_s": total}


def main() -> None:
    targets = sys.argv[1:]
    if not targets:
        print("usage: bench.py <file> [file...]")
        sys.exit(2)

    results = {}
    for t in targets:
        src = Path(t) if Path(t).exists() else DATA / t
        orig = src.stat().st_size
        rows = []
        for name in CODECS:
            r = bench_codec(name, src)
            rows.append(r)
            print(f"  {r['codec']:<10} {r['bytes']:>12,}  {orig/r['bytes']:>7.2f}x  "
                  f"enc={r['enc_s']:.1f}s  verified={r['verified']}", flush=True)
        r = bench_zpaq(src)
        rows.append(r)
        print(f"  {r['codec']:<10} {r['bytes']:>12,}  {orig/r['bytes']:>7.2f}x  "
              f"enc={r['enc_s']:.1f}s", flush=True)
        r = bench_augur(src)
        rows.append(r)
        print(f"  {r['codec']:<10} {r['bytes']:>12,}  {orig/r['bytes']:>7.2f}x  "
              f"enc={r['enc_s']:.1f}s  verified={r['verified']}", flush=True)
        results[src.name] = {"original_bytes": orig, "codecs": rows}

    out = ROOT / "results.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
