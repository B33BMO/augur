# augur

**A structure-aware, lossless compressor that beats `xz -9e` by understanding your data instead of just packing bytes.**

augur is a from-scratch context-mixing compressor built on one idea: **compression is prediction.** Predict the next bit, code only the surprise. A two-layer logistic mixer blends a portfolio of predictors — local context, word models, long-range hash-chain matches, *structure-aware* models that understand JSON fields, CSV columns, SQL-dump tuples, XML elements and log columns, a **record-history** model that replays the previous record's value for the same field, a numeric model that learns sequential and **cross-column** relationships (`lastSeen = firstSeen`, `id = seq + 100000`), stride models that find the record period of binary tables, and an adaptive-filter front-end for PCM audio — feeding a single arithmetic coder. The encoder and decoder run the identical predict→code→update loop, so they can never desync.

It beats `flac -8` on lossless audio, and `xz -9e` on **all 20 other datasets tested** — by 22% on enwik8, 21% on the full
Silesia corpus (winning every one of its 12 files), and 42–92% on real structured
data. It is also **slow** — see the caveats.

It has **zero dependencies** (not even for the CLI) and is a single Rust file.

## Results

Compression ratio (higher is better), byte-exact lossless. Every row was verified
by a full compress → decompress → compare cycle.

### Real-world data, whole files

| dataset | size | zstd-19 | xz-9e | **augur** | vs xz |
|---|---|---|---|---|---|
| nginx_logs | 7.0 MB | 26.54x | 29.31x | **56.42x** | **+92%** |
| taxi.ndjson | 91 MB | 27.98x | 31.84x | **53.58x** | **+68%** |
| taxi.csv | 22 MB | 8.12x | 8.45x | **13.96x** | **+65%** |
| gh_events.ndjson | 61 MB | 16.84x | 19.89x | **28.28x** | **+42%** |
| enwik8 | 100 MB | 3.71x | 4.03x | **4.91x** | **+22%** |

### Synthetic, to isolate the structural models

These are generated to exercise one mechanism each, so treat them as an upper
bound on what the structure and numeric models can do — not as typical data.

| dataset | zstd-19 | xz-9e | **augur** | vs xz |
|---|---|---|---|---|
| seq.ndjson — sequential IDs + timestamps | 14.96x | 19.14x | **87.75x** | **+358%** |
| dump.sql — `INSERT INTO … VALUES` batches | 10.92x | 14.47x | **45.43x** | **+214%** |
| xcol.csv — cross-column relations | 3.13x | 3.94x | **7.07x** | **+79%** |

### Lossless audio, vs FLAC

16-bit PCM WAV, whole tracks. FLAC is the reference lossless audio coder; `xz`
and `zstd` are included to show that general-purpose compressors are simply not
in this game.

| track | size | zstd-19 | xz-9e | flac -8 | **augur** | vs flac |
|---|---|---|---|---|---|---|
| creed_higher.wav | 47.5 MB | 1.13x | 1.17x | 1.76x | **1.89x** | **+7.6%** |
| waiting_for_the_end.wav | 38.9 MB | 1.10x | 1.14x | 1.67x | **1.78x** | **+6.1%** |

### Silesia

The standard mixed corpus (212 MB, 12 files), and the hard case: half of it is
binary, which is where a structure-aware compressor has the least to say. Whole
files, not slices:

| file | zstd-19 | xz-9e | **augur** | vs xz |
|---|---|---|---|---|
| dickens | 3.58x | 3.60x | **4.63x** | +28.7% |
| mozilla | 3.40x | 3.83x | **4.08x** | +6.6% |
| mr | 3.21x | 3.62x | **4.56x** | +25.9% |
| nci | 20.15x | 23.15x | **29.25x** | +26.3% |
| ooffice | 2.37x | 2.53x | **3.24x** | +27.8% |
| osdb | 3.25x | 3.55x | **4.39x** | +23.9% |
| reymont | 4.91x | 5.04x | **6.67x** | +32.4% |
| samba | 5.55x | 5.78x | **6.88x** | +19.1% |
| sao | 1.45x | 1.64x | **1.88x** | +14.6% |
| webster | 4.78x | 4.95x | **7.18x** | +44.9% |
| x-ray | 1.65x | 1.89x | **2.28x** | +20.8% |
| xml | 11.79x | 12.29x | **17.88x** | +45.5% |
| **aggregate** | **4.01x** | **4.37x** | **5.29x** | **+20.9%** |

**augur wins all 12, and the byte-weighted aggregate by 21%** — including the
binaries a structure-aware design has no business winning. `sao` (a star
catalogue) and `x-ray` are carried by the record-stride detector finding a period
nothing in the file declares; `ooffice` and `mozilla` by the E8E9 pass. `nci` is
the interesting one: it is extremely repetitive data where LZMA's long-match
parsing traditionally wins, and where a previous version of augur lost outright
at 20.8x against xz's 23.2x. Hash-chain match models with backward-context
candidate selection now take it at 29.2x.

Across all 20 datasets above, augur beats `xz -9e` on **20 of 20**.

Read: augur is strongest on structured and textual data, where it wins by wide
margins, and is ahead everywhere else too — by a slimmer margin on binaries.

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

Every model is a `predict()` returning P(next bit = 1). A two-layer logistic mixer combines them (online-learned weights, integer fixed-point), an SSE stage calibrates the result, and one binary arithmetic coder turns it into bits.

The portfolio:

- **Order 0,1,2,3,4,6,8 context models** — local byte statistics. Counters track a hit count and adapt at `1/(n+1.5)`, so a context seen once jumps most of the way to what it saw, while a well-established one barely moves.
- **Word models** — the alphanumeric token being typed, and that token in the company of the one before it. Byte orders see `tio`; this sees `informatio`, and `the informatio`.
- **Match models (hash chains)** — long-range repeats, the redundancy a local model structurally cannot see. On a miss, each model walks a chain of recent positions sharing the current context and picks the one whose *preceding* bytes match longest. The two models search differently on purpose: the short-context one reacquires after a break, where the most *recent* occurrence is usually right and searching deeper actively hurts; the long-context one locks onto genuine long repeats, where depth pays enormously. A shared search depth costs one job or the other.
- **Structure models** — a streaming, format-aware parser exposes *semantic position*: which JSON field's value, which CSV column, which SQL `INSERT ... VALUES` tuple column, which XML element, or which whitespace-delimited column of a log line you are currently inside. Byte-level coders can't condition on "I'm reading the value of `created_at`"; augur can. Three contexts are formed from it — field with position-in-value, field with a secondary axis, and field crossed with the last two bytes, which is far more specific than either half alone. The format is sniffed at compress time and recorded in the header, so the decoder configures the same parser.
- **Record-history model** — replays the previous record's value *for this field*, from its very first byte. This is redundancy the match model is structurally blind to: it needs six matching bytes of context before it can speak, and the bytes immediately before a field's value belong to a *different* field. `"city":"Springfield"` follows `"city":"Springfield"` even when the id and timestamp before them share nothing.
- **Numeric model** — predicts the digits of a value *before reading them*, choosing the more confident of two hypotheses: cross-row extrapolation (`last + delta` — auto-increment IDs, timestamps, counters) or a **cross-column** relation within the same row (a copy like `lastSeen = firstSeen`, or a constant offset like `id = seq + 100000`).
- **Stride and sparse models** — binary tables (a star catalogue, a database page, a struct array) repeat with a period nothing in the file declares. augur watches how far apart four-byte patterns recur and lets the winning distance vote itself into being the record length, then models each value against the one a *record* above it. Text never produces a sharp peak, so these stay silent.

The match, record and numeric models are **oracles**: each names the byte it thinks comes next, but none asserts how sure it is. A `TrustMap` learns that empirically, per (agreement length, bit position, predicted bit) — so a match 200 bytes into a repeat and one that just reacquired are trusted differently, by measurement rather than by a tuned constant.

A **lossless audio front-end** handles 16-bit PCM WAV, where the rest of the portfolio is helpless: sample 44100 has almost nothing byte-wise in common with sample 44099 even though it is nearly *numerically* equal to it, so the context and match models see noise. Instead augur decorrelates the stereo pair, runs a cascade of sign-sign LMS filters, and hands the residual to the mixer. The filters adapt rather than storing per-block coefficients the way FLAC does — the decoder runs the identical integer update over samples it has already reconstructed, so there is no side channel at all. Every stage works in wrapped 16-bit arithmetic, which is what keeps the transform exactly invertible when a prediction overshoots.

An **E8E9 pass** rewrites x86 `CALL` offsets from relative to absolute, so the same function called from a hundred sites produces a hundred identical byte sequences. augur decides whether to apply it by *trying* it on a sample and comparing — which sidesteps the unanswerable question of whether an archive that merely *contains* an executable "is" one.

### Container format

```
"AUGR" | version (1) | mode (1) | mem_bits (1) | flags (1) | original_length (8, LE) | stream
```

Table size is chosen from the input length and recorded in the header, so a small file doesn't pay for a large file's tables, and the decoder rebuilds them byte-identically.

## Honest caveats

- **It is slow: roughly 0.5–0.6 MB/s each way.** Context mixing is symmetric and serial — every bit must be predicted before the next can be coded, and augur consults nineteen models, four mixers and four SSE stages per bit. Encode and decode cost about the same, and both are orders of magnitude below zstd/xz. This buys the ratios above; it is the wrong tool for anything latency-sensitive, and the right one for **write-once, read-rarely** data: archival, cold feeds, backups, long-tail object storage.
- **Memory is ~360 MB** for inputs above a couple of megabytes, scaled down for smaller ones and recorded in the header so the decoder matches. Halving it costs about 0.4% ratio; doubling it buys about 0.2%.
- **On already-compressed or random data there is nothing to model** — augur correctly punts to ~1.0x plus a 16-byte header rather than expanding meaningfully.
- **The ratio is not the theoretical ceiling.** Heavyweight mixers (cmix, paq8) go substantially further on text by spending thousands of times more compute. augur aims to be the best compressor you'd actually run on structured data, not the winner of an unconstrained ratio contest.

## Where the speed went

augur used to encode at ~5 MB/s; the models added since cost roughly 10x that.
Some of it has been clawed back **without giving up a single byte of ratio** —
the optimisations below leave the compressed output bit-identical:

- **Prefetching the next bit's table lines** (+23%). The fifteen scattered loads
  per bit dominate the inner loop, and their addresses are known one bit early:
  the next partial byte can only be `c0<<1` or `c0<<1|1`. Both are requested
  before the mixer and coder run, which covers most of the latency.
- **Hoisting the `OnceLock` table derefs** out of the per-bit path (+9%). An
  atomic load and a branch, paid ~28 times per bit, for pointers that never change.
- **A density pre-check before the E8E9 trial.** Deciding whether the transform
  pays costs two sample encodes; prose contains essentially no `0xE8` bytes, so
  it can be rejected in one linear scan instead.

Two things were measured and **rejected**:

- **Nibble-bucketed counter tables**, the textbook cache fix: +55% speed for
  −6.6% ratio, because at fixed memory the buckets spend four bits of context
  resolution to buy locality. Wrong trade for this project.
- **`i16` mixer weights for SIMD.** Removing three of the four layer-1 mixers
  entirely only saved 11% wall-clock, so the whole mixer is ~15% of runtime and
  vectorising it caps out around 8% — not worth the retuning risk. The bottleneck
  is memory, not arithmetic.

## Testing

```bash
cargo test          # roundtrip + robustness suite
```

The suite covers the boundaries that break compressors: empty and 1-byte inputs,
incompressible random data, every byte value, malformed JSON, quoted CSV, and
garbage/corrupt-header rejection (including a `mem_bits` field that would
otherwise size an absurd allocation). Two tests exist specifically because the
E8E9 pass is easy to get subtly wrong — it is only reversible if both directions
agree on which bytes are instructions, so one test hammers it with adversarial
`0xE8` placements and another drives executable-like data through the full
container.

## License

[Apache-2.0](LICENSE).
