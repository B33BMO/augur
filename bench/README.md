# benchmarks

Reproducible harness for augur's published numbers. Everything here fetches
from canonical public sources, so anyone can check the claims in the top-level
README rather than taking them on faith.

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install pyarrow
./fetch.sh                       # download canonical sources
./.venv/bin/python prep.py       # derive CSV / NDJSON / Parquet from the same rows
cargo build --release --manifest-path ../Cargo.toml
./.venv/bin/python bench.py taxi.csv nginx_logs taxi.ndjson gh_events.ndjson
```

Results accumulate in `results.json`.

## Methodology

**The peer set is every codec at its strongest practical setting.** augur is
slow by design; comparing it against a fast setting of anything else is not a
comparison. `zstd -22 --ultra --long=31`, `xz -9e`, `brotli -q11 -w24`, and
`zpaq -m5` — the last being the one that actually matters, since it is a mature
context-mixing archiver and therefore augur's real opponent. Beating `xz` is a
much weaker claim than beating `zpaq`.

**Ratios are reported against the named input file, but the number that decides
anything is absolute compressed bytes.** A ratio is partly a measurement of how
verbose the encoding you started from was: "13.9x on CSV" and "53x on NDJSON"
can describe the same information stored equally well, because the NDJSON
repeats every field name on every row. When comparing storage strategies for
the *same data*, compare bytes.

**`prep.py` emits CSV, NDJSON and Parquet from identical rows.** This is what
makes the Parquet comparison meaningful. Parquet+zstd is what people actually
store tabular data in, so it — not `xz` on a CSV — is the honest baseline for
the structured-data claim.

**Every general-purpose codec is verified by a full roundtrip and `cmp`.** augur
is verified by its own `bench` subcommand, which does the same. A ratio from an
unverified decompress is not a result.

## Caveat on the Parquet comparison

augur wins on bytes, but Parquet and augur are not interchangeable. Parquet
offers random access, column pruning, row-group statistics and predicate
pushdown, and decodes at hundreds of MB/s. augur produces an opaque blob that
must be decoded serially at ~0.5 MB/s. The comparison establishes that augur
stores the same information in fewer bytes; it does not establish that augur
should replace Parquet in a query path. It is an argument for cold storage,
where bytes-at-rest are the cost and reads are rare.

## Datasets

| file | source |
|---|---|
| `nginx_logs` | [elastic/examples](https://github.com/elastic/examples) — Common Data Formats |
| `yellow_tripdata_2024-01.parquet` | NYC TLC trip records (CloudFront) |
| `taxi.csv` / `taxi.ndjson` / `taxi.*.parquet` | derived by `prep.py` from the same 180,000 TLC rows |
| `gh_2024-01-01-15.json.gz` | [GH Archive](https://www.gharchive.org/) |
| `gh_events.ndjson` | first ~61 MB of the above, decompressed |
| `enwik8` | first 10^8 bytes of a Wikipedia XML dump |

Silesia is not yet wired in here; add it to `fetch.sh` before trusting the
Silesia table in the top-level README.
