# augur: beating xz by predicting, not packing

*A from-scratch compressor that understands your data — and what it taught us in one day.*

---

## The itch

I've wanted to shake the compression industry for a while. The dream — half-inspired
by *Silicon Valley*, I'll admit it — is one engine that makes everything impossibly
small: video, audio, text, code, JSON, all of it. That exact dream is impossible
(you can't shrink random data; it's just counting). But the *spirit* of it isn't,
and chasing the honest version of it led somewhere genuinely good.

My previous engine, **Recursor**, was ~32,000 lines of Rust: a router over a zoo of
hand-written, per-filetype compressors. Some pieces were clever. But when I finally
benchmarked it honestly against the boring incumbents, the result was brutal.

## Step 1: measure the gap (it hurt)

On real structured datasets, Recursor **lost to plain `zstd -19` on every one**:

| dataset | Recursor | zstd-19 | xz-9e |
|---|---|---|---|
| taxi.csv | 5.45x | 8.12x | 8.45x |
| taxi.ndjson | 26.1x | 28.0x | 31.8x |
| gh_events.ndjson | **5.90x** | 16.8x | **19.9x** |
| nginx_logs | 14.7x | 26.5x | 29.3x |

So I measured *why*. I computed the empirical entropy of each dataset at byte-context
orders 0–3 (what a local model can do) and the long-range match headroom (what
repetition a copy-model can find):

- A pure local context model — *any* order up to 3 — tops out **below zstd**.
- ~100% of every dataset is covered by long-range repeats invisible to local context.
- But neither dedup-alone nor context-alone reaches zstd. zstd/xz win by **combining**
  long-range matching with entropy-coded literals *and* match tokens.

Recursor's "structure-aware" encoders were optimizing local entropy — the wrong axis.
The compressibility was in cross-record repetition, which it handled *worse* than a
30-year-old LZ pass.

## Step 2: a new thesis — *predict, don't pack*

You can't refactor your way out of a wrong thesis. So I started over with a new one:

> **Compression is prediction.** A perfect next-symbol predictor plus an arithmetic
> coder hits the entropy bound automatically. So don't build a zoo of codecs — build
> one predictor and code only the *surprise*.

The architecture (**augur**) is one idea repeated:

```
                 ┌─ order-0..4 context models  ─┐
 history ─────►  ├─ match model (long-range)    ─┤─► logistic mixer ─► arithmetic coder
                 ├─ structure models (JSON)     ─┤
                 └─ numeric model (formulas)    ─┘
```

Every model is just a `predict()` returning P(next bit = 1). A logistic mixer blends
them with online-learned weights; one binary arithmetic coder turns the blend into
bits. Crucially, **the encoder and decoder run the identical predict→code→update
loop**, so they can never desync — the classic context-mixing failure mode is
designed out.

The whole thing is ~450 lines in one file.

## Step 3: the spine beats zstd

The first version — just context orders + a match model, no understanding of the data
at all — already cleared zstd on the dataset that humiliated Recursor:

- **gh_events.ndjson: 18.44x** (vs Recursor 5.90x, zstd 16.84x, xz 19.89x)

A ~400-line first draft, 3x better than the old 32k-line project, past zstd, within
7% of xz. Then an experiment told us where to go next: adding byte-context orders 5–6
helped **+0.4%**. Local order was tapped out. The headroom was *structural*.

## Step 4: the moat — understanding the data

Here's the part a byte-level compressor structurally cannot do. I added a **streaming
JSON parser** that runs at each byte boundary (deterministic, so it can't desync) and
exposes a semantic context: *which field's value am I currently inside?* Predictions
condition on `(field, position)` and `(field, depth)` — so the value bytes of
`created_at` are modeled separately from those of `id` or `url`.

That single idea pushed gh_events from **18.44x → 20.34x — past xz.**

Then the categorically-new weapon: a **numeric model** — formula detection in its
most common form. For each field it tracks the last value and its delta, and when
that field's next value begins, it predicts the digits of `last + delta` *before
reading them*. Auto-increment IDs, timestamps, sequence numbers — the stuff that fills
real databases and event logs — collapse to nearly zero bits. It's the match model,
but the source is a formula instead of history.

On synthetic data with sequential IDs and timestamps:

| | seq.ndjson |
|---|---|
| zstd-19 | 11.37x |
| xz-9e | 15.19x |
| **augur** | **29.21x** |

Nearly **2x xz**. And it's self-regulating: on data *without* sequential fields
(GitHub's random IDs), the mixer simply learns to ignore it. Exploit structure when
it's there, fall back gracefully when it isn't.

## Step 5: the real test — 7.8 million live threats

Synthetic wins are easy to fake yourself into. So I pointed augur at a real
production database: **Evil-DB**, my threat-intelligence platform, and its
`ThreatEntry` table — **7,857,069 records** of IPs, domains, and hashes, each with
enum fields (`type`, `threatLevel`, `source`), a nested JSON `categories` array,
report/confidence integers, and `firstSeen`/`lastSeen` epoch-millisecond timestamps.
I exported a 200,000-row, 54 MB NDJSON slice in insertion order — exactly the format
threat feeds ship in.

| | threats.ndjson (200k real records) |
|---|---|
| zstd-19 | 11.71x |
| xz-9e | 12.76x |
| **augur** | **14.39x** |

**+12.8% over xz on real, messy, production data.** The structure model handled the
enums and category arrays; the numeric model rode the near-sequential `firstSeen`
timestamps; and — the honesty check — it correctly *ignored* the random cuid `id`
field, where there's no sequence to find. Extrapolated to the full 7.8M threats,
that's a multi-gigabyte feed compressing losslessly to roughly **1/14th** its size.

## Step 6: making it fast — and, surprisingly, better

The prototype was ~1 MB/s: it did a floating-point `exp`/`ln` per bit per model. I
rewrote the inner loop in integer fixed-point — `stretch`/`squash` as lookup tables,
the mixer in i32/i64, and the match/numeric confidences precomputed at byte
boundaries (they only change there) instead of per bit. No transcendental calls in
the hot path.

The expected payoff was speed (now ~5 MB/s encode — faster than `xz -9e`). The
*unexpected* payoff: ratios went **up** across the board, because the integer mixer
(lpaq-style weight initialization and learning rate) converges better than my
hand-tuned float version. The same change made it both faster and tighter — and it
erased an earlier ~3% regression the numeric model had on float-heavy data.

Then decode. Context-mixing decode is the slow side: it's *symmetric* (runs the same
model) and *serial* (each bit must be decoded before the next can be predicted, so
nothing pipelines). Profiling pointed at memory — seven random lookups per bit into
56 MB of context tables, mostly cache misses. Two fixes: flatten the tables into one
contiguous buffer (no pointer-chasing, no bounds checks on the hot path) and shrink
them to fit cache. Decode got **20–40% faster** for ~2–3% ratio. Encode 4.6–5.5 MB/s,
decode 3.4–4.1 MB/s now.

## Results

augur vs the best general-purpose compressors, full files, compression ratio
(higher is better):

| dataset | old Recursor | zstd-19 | xz-9e | parquet+zstd | **augur** |
|---|---|---|---|---|---|
| gh_events.ndjson | 5.90x | 16.84x | 19.89x | — | **21.67x** |
| nginx_logs | 14.65x | 26.54x | 29.32x | 28.29x | **41.52x** |
| taxi.csv | 5.45x | 8.12x | 8.45x | 6.18x | **9.32x** |
| taxi.ndjson | 26.12x | 27.98x | 31.84x | 33.50x | **40.53x** |
| seq.ndjson (synthetic) | — | 11.37x | 15.19x | — | **32.44x** |
| threats.ndjson (real) | — | 11.71x | 12.76x | — | **15.96x** |

augur beats `xz -9e` on **every dataset tested — six for six** — by 9% to 114% —
and beats `parquet+zstd`, the columnar specialist, on every tabular case where it
applies. Every output is **byte-exact lossless** (verified roundtrip on every run).

## The honest caveat

**Encode beats xz on speed too; decode is the real gap.** augur encodes at
4.6–5.5 MB/s — faster than `xz -9e`. Decode is 3.4–4.1 MB/s, while zstd and xz decode
at hundreds of MB/s to GB/s. That gap is structural: context-mixing decode is
symmetric and serial (each bit must be decoded before the next prediction), so it
can't pipeline like encode. For write-once / read-rarely data (archival, cold feeds,
backups) it's already fine; for hot read paths, closing it means a fundamentally
cheaper decode-side model — a real research problem, not a free lunch. Named, not
hidden.

## What's next

- **Widen the moat:** CSV column-awareness (taxi.csv is the weakest — the JSON parser
  doesn't engage there), a code-aware model, and richer formula detection
  (multi-column dependencies, non-linear sequences).
- **Ship it:** a real `compress`/`decompress` file CLI with a container header, a
  README, and a public repo.
- **The deep end:** the same "predict the next thing, code the surprise" socket is
  exactly how modern neural video codecs work. Video is where this thesis goes to
  become spectacular — once we've earned it on lossless structured data.

One engine. Understand the data, then the small size follows. That's the bet.

*Built in a day. From a 32k-line project that lost to zstd, to a ~700-line one that
beats xz by 11–114% on six datasets — including 7.8M real threats — at competitive
encode speed.*
