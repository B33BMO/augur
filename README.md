# augur

**A structure-aware, lossless compressor that beats `xz -9e` by understanding your data instead of just packing bytes.**

augur is a from-scratch context-mixing compressor built on one idea: **compression is prediction.** Predict the next bit, code only the surprise. A logistic mixer blends a portfolio of predictors — local context, long-range hash-chain matches, *structure-aware* models that understand JSON fields, CSV columns, SQL-dump tuples and XML elements, and a numeric model that learns both sequential and **cross-column** relationships (`lastSeen = firstSeen`, `id = seq + 100000`) — feeding a single arithmetic coder. The encoder and decoder run the identical predict→code→update loop, so they can never desync.

It has **zero dependencies** (not even for the CLI) and is a single Rust file.

## Results

Compression ratio (higher is better), full files, byte-exact lossless:

| dataset | zstd-19 | xz-9e | parquet+zstd | **augur** | vs xz |
|---|---|---|---|---|---|
| gh_events.ndjson | 16.84x | 19.89x | — | **22.31x** | +12% |
| nginx_logs | 26.54x | 29.32x | 28.29x | **41.89x** | +43% |
| taxi.csv | 8.12x | 8.45x | 6.18x | **11.62x** | +37% |
| taxi.ndjson | 27.98x | 31.84x | 33.50x | **42.67x** | +34% |
| threats.ndjson (real DB export) | 11.71x | 12.76x | — | **16.09x** | +26% |
| threats.sql (real DB dump) | 6.72x | 7.25x | — | **9.05x** | +25% |
| threats.xml (real DB as XML) | 13.73x | 15.37x | — | **19.28x** | +25% |
| seq.ndjson (sequential IDs/timestamps) | 11.37x | 15.19x | — | **32.41x** | +113% |
| xcol.csv (cross-column relations) | 3.31x | 4.10x | — | **6.74x** | +64% |

augur beats `xz -9e` on every dataset tested, and beats `parquet+zstd` (the columnar specialist) on every tabular case where it applies.

### Standard corpora

On the benchmarks the field reports, vs `zstd -19` / `xz -9e` (byte-exact lossless):

- **enwik8** (100 MB, English Wikipedia): augur **4.14x** — beats both xz (4.03x) and zstd. (A heavy context-mixer like cmix goes further on pure text; augur is a lightweight CM that trades peak ratio for speed and a single small file.)
- **Silesia** (212 MB, 12 mixed files): augur wins **9 of 13 files** — every text file (dickens, webster, reymont), plus xml, osdb, samba, and *both* medical images (mr +18%, x-ray +15% vs xz). It trails xz on the binaries (mozilla, ooffice, sao) and on nci (extremely repetitive data where LZMA's long-match parsing wins — augur's hash-chain match models narrow it to 20.8x vs 23.2x), so xz still narrowly takes the byte-weighted aggregate, which is dominated by those large binary files.

Read: augur is a **text/structured specialist** — it wins most files, decisively on text and structured data, and trails general compressors on binaries and on data with very long exact repeats.

## Build

```bash
cargo build --release
```

## Usage

```bash
# compress (writes <file>.augur; format is auto-detected)
augur compress data.ndjson
augur compress data.csv -o data.csv.augur

# decompress (restores the original)
augur decompress data.ndjson.augur
augur decompress data.csv.augur -o restored.csv

# benchmark a file in memory (compress + verify roundtrip + timings)
augur bench data.ndjson
augur bench data.ndjson 8388608   # only the first 8 MB
```

## How it works

Every model is a `predict()` returning P(next bit = 1). A logistic mixer combines them (online-learned weights, integer fixed-point), and one binary arithmetic coder turns the blend into bits.

The portfolio:

- **Order 0–4 context models** — local byte statistics.
- **Match models (hash chains)** — long-range repeats, the redundancy a local model structurally cannot see. On a miss, each model walks a chain of recent positions sharing the current context and picks the one whose *preceding* bytes match longest — locking onto genuine long repeats instead of the most-recent coincidence.
- **Structure models** — a streaming, format-aware parser exposes *semantic position*: which JSON field's value, which CSV column, which SQL `INSERT ... VALUES` tuple column, or which XML element you're currently inside. Byte-level coders can't condition on "I'm reading the value of `created_at`"; augur can. The format (JSON / CSV / SQL / XML / generic) is sniffed at compress time and recorded in a one-byte header, so the decoder configures the same parser.
- **Numeric model** — predicts the digits of a value *before reading them*, choosing the more confident of two hypotheses: cross-row extrapolation (`last + delta` — auto-increment IDs, timestamps, counters) or a **cross-column** relation within the same row (a copy like `lastSeen = firstSeen`, or a constant offset like `id = seq + 100000`). When neither is predictable, the mixer simply learns to ignore it.

### Container format

```
"AUGR" | version (1) | mode (1) | original_length (8, little-endian) | arithmetic stream
```

## Honest caveats

- **Encode is fast (~5 MB/s, faster than `xz -9e`); decode is the gap.** Context mixing is symmetric and serial — each bit must be decoded before the next can be predicted, so decode (~3–4 MB/s) can't pipeline like encode, and it's far slower than zstd/xz's GB/s decode. augur is ideal for **write-once / read-rarely** data (archival, cold feeds, backups). Closing the decode gap needs a fundamentally cheaper decode-side model — open work.
- **It's a structure/text specialist, not a universal ratio king.** augur wins on structured data (JSON, NDJSON, CSV, logs, DB exports) and is also ahead on plain text. It is **behind xz on binary/executable data**, and on already-compressed or random data there's nothing to model — it correctly punts to ~1.0x (a few bytes of container overhead). Reach for it where the data has structure to understand.

## Testing

```bash
cargo test          # roundtrip + robustness suite (empty, 1-byte, random,
                    # all-byte-values, malformed JSON, quoted CSV, garbage rejection, …)
```

## License

[Apache-2.0](LICENSE).
