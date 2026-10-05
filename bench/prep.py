#!/usr/bin/env python3
"""Build the benchmark datasets from canonical public sources.

The point of this script is that CSV, NDJSON and Parquet are all emitted from
the *same rows*, so the encodings can be compared on absolute compressed bytes
rather than on ratio-against-whichever-text-encoding-you-happened-to-pick.
"""
import gzip
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv

DATA = Path(__file__).parent / "data"

# Row count chosen to land taxi.csv near 22 MB, matching the README's dataset.
TAXI_ROWS = 180_000
# Truncate gh_events to roughly the README's 61 MB.
GH_TARGET_BYTES = 61 * 1024 * 1024


def prep_taxi() -> None:
    src = DATA / "yellow_tripdata_2024-01.parquet"
    table = pq.read_table(src).slice(0, TAXI_ROWS)

    # CSV
    csv_path = DATA / "taxi.csv"
    pacsv.write_csv(table, csv_path)

    # NDJSON, from the identical rows. Timestamps are stringified in ISO form,
    # which is what a real JSON feed of this table would carry.
    ndjson_path = DATA / "taxi.ndjson"
    cols = table.column_names
    with ndjson_path.open("w") as fh:
        for batch in table.to_batches(max_chunksize=10_000):
            pylist = batch.to_pylist()
            for row in pylist:
                out = {}
                for c in cols:
                    v = row[c]
                    out[c] = v.isoformat(sep=" ") if hasattr(v, "isoformat") else v
                fh.write(json.dumps(out, separators=(",", ":")) + "\n")

    # Parquet, same rows, at each codec's strongest setting.
    for codec, level, name in [
        ("zstd", 22, "taxi.zstd22.parquet"),
        ("brotli", 11, "taxi.brotli11.parquet"),
        ("gzip", 9, "taxi.gzip9.parquet"),
    ]:
        pq.write_table(
            table,
            DATA / name,
            compression=codec,
            compression_level=level,
            use_dictionary=True,
            write_statistics=True,
        )

    print(f"taxi: {TAXI_ROWS} rows, {len(cols)} cols")


def prep_gh() -> None:
    src = DATA / "gh_2024-01-01-15.json.gz"
    out = DATA / "gh_events.ndjson"
    written = 0
    with gzip.open(src, "rb") as fh, out.open("wb") as w:
        for line in fh:
            if written + len(line) > GH_TARGET_BYTES:
                break
            w.write(line)
            written += len(line)
    print(f"gh_events.ndjson: {written} bytes")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "taxi"):
        prep_taxi()
    if which in ("all", "gh"):
        prep_gh()
