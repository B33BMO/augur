# augur

**A lossless compressor that understands what it is compressing.**

augur is a from-scratch context-mixing compressor built on one idea: **compression is prediction.** Predict the next bit, code only the surprise. A two-layer logistic mixer blends a portfolio of predictors into a single arithmetic coder, and the encoder and decoder run the identical predict→code→update loop, so they can never desync.

What sets it apart is how much of a file it can *see into*:

- **Structured text** — JSON fields, CSV columns, SQL tuples, XML elements and log columns are parsed as they stream past, with record-history and cross-column numeric models.
- **Audio** — 8/16/24/32-bit PCM WAV, any channel count, coded through least-squares and LMS predictors running in lockstep with the mixer.
- **Images** — PPM/PGM, BMP, TGA, raw 16-bit rasters, and the pixels inside PNGs, through a 24-predictor image model with a colour cache.
- **Anything deflated** — PNG, ZIP, gzip, PDF, DOCX/XLSX, JAR, EPUB. Streams are unpacked, modelled, and re-deflated **bit-exactly** on decompression: by a clean-room clone of zlib when zlib made them, and by a recorded description of the encoder's choices when something else did.
- **JPEG** — baseline JPEG scans are decoded bit by bit as they are coded, and every Huffman bit is predicted from the neighbouring blocks' DCT coefficients.
- **GIF** — LZW is undone and the palette indices go to the image model; re-encoding replays the encoder's clear codes and the rare places it cut a string short, bit for bit.
- **x86 code** — instructions are decoded as they stream past, so each byte is modelled as the opcode, ModRM, displacement or immediate it is.

It has **zero dependencies** (not even for the CLI).

Where it stands, plainly:

- **Against `zpaq -m5`**, the strongest widely-packaged context mixer, it takes the Silesia corpus by **9.6%** and wins 11 of its 12 files; on deflate-packed files (PDFs, JARs, archives) it is **26–66%** smaller, because zpaq cannot see inside them.
- **Against format specialists** it wins too: **8–31% smaller than `flac -8`** on audio, **~26% smaller than JPEG XL** (`cjxl -d 0 -e 9`) on photographic images, and **20%** off baseline JPEGs.
- **Against paq8px**, the research-grade champion, it is mostly behind: within ~0.5% on most 16-bit audio (and far ahead on 24-bit, which paq8px does not model), ~3.5 points behind on JPEG, ~10% on colour photographs, and 10–35% on prose, source code and executables. paq8px also unpacks zlib streams, so on zlib-made containers (JARs, PDFs, DOCX) it is 30–35% smaller than augur; augur wins only where another encoder made the stream (GNU gzip: 2x smaller; Info-ZIP: 6% smaller), which paq8px cannot unpack.
- It is **slow** — tens of KB/s on audio, images and JPEG. See the caveats.

## Results

Byte-exact lossless. Every augur number below was verified by a full compress → decompress → compare cycle, all with the same build.

### Silesia

The standard mixed corpus, 212 MB, whole files.

| file | zstd-19 | xz-9e | zpaq -m5 | **augur** | augur vs zpaq |
|---|---|---|---|---|---|
| dickens | 3.58x | 3.60x | 4.87x | **4.83x** | −0.7% |
| mozilla | 3.40x | 3.83x | 4.25x | **5.18x** | **+17.9%** |
| mr | 3.21x | 3.62x | 4.57x | **5.53x** | **+17.4%** |
| nci | 20.15x | 23.15x | 26.82x | **34.19x** | **+21.6%** |
| ooffice | 2.37x | 2.53x | 3.48x | **3.53x** | **+1.4%** |
| osdb | 3.25x | 3.55x | 4.57x | **4.62x** | **+1.0%** |
| reymont | 4.91x | 5.04x | 6.93x | **7.60x** | **+8.8%** |
| samba | 5.55x | 5.78x | 7.07x | **8.27x** | **+14.4%** |
| sao | 1.45x | 1.64x | 1.86x | **1.90x** | **+1.9%** |
| webster | 4.78x | 4.95x | 7.32x | **7.53x** | **+2.9%** |
| x-ray | 1.65x | 1.89x | 2.31x | **2.38x** | **+3.2%** |
| xml | 11.79x | 12.29x | 16.34x | **19.27x** | **+15.2%** |
| **aggregate** | **4.01x** | **4.37x** | **5.42x** | **5.99x** | **+9.6%** |

`mozilla` and `samba` gain from recompression: both tarballs carry zlib streams inside them, which augur now unpacks.

### Deflate-packed files

| file | original | xz-9e | zpaq -m5 | **augur** | vs zpaq |
|---|---|---|---|---|---|
| guava.jar | 3,047,503 | 2,709,424 | 2,678,693 | **918,949** | **−65.7%** |
| attention.pdf (arXiv) | 2,215,244 | 1,033,444 | 1,022,294 | **454,734** | **−55.5%** |
| f1040.pdf (IRS form) | 220,237 | 150,856 | 149,967 | **73,025** | **−51.3%** |
| samba.tar.gz (GNU gzip) | 5,408,338 | 5,301,088 | 5,299,091 | **2,629,128** | **−50.4%** |
| silesia_xml.zip (Info-ZIP) | 4,514,357 | 4,514,644 | 4,515,568 | **2,411,288** | **−46.6%** |
| requests.whl | 64,928 | 63,584 | 64,137 | **37,187** | **−42.0%** |
| demo.docx | 1,311,881 | 1,304,780 | 1,300,616 | **957,669** | **−26.4%** |
| pride.epub (Gutenberg, JPEG-heavy) | 24,836,548 | 24,226,444 | 23,046,695 | **18,939,996** | **−17.8%** |

GNU gzip, Info-ZIP and most PNG optimisers are *not* zlib, so no parameter search can reproduce their output — these go through reflate (below).

paq8px -8 on the same files, for honesty: it also unpacks zlib streams and then models their contents better than augur does. It is smaller on guava.jar (624,847), attention.pdf (336,592), f1040.pdf (46,670), demo.docx (634,912), requests.whl (31,901) and pride.epub (17,482,977). augur is smaller on samba.tar.gz (2,629,128 vs 5,254,194) and silesia_xml.zip (2,411,288 vs 2,557,898), whose GNU gzip and Info-ZIP streams paq8px cannot reproduce.

### Lossless audio

16-bit stereo unless named, whole tracks from the EBU SQAM test disc and public 24-bit samples.

| track | size | flac -8 | wavpack -hhx6 | **augur** | vs flac |
|---|---|---|---|---|---|
| sqam44 (soprano) | 4.9 MB | 1,623,197 | 1,564,030 | **1,285,994** | **+20.8%** |
| sqam49 (speech) | 4.1 MB | 1,143,061 | 1,095,084 | **956,040** | **+16.4%** |
| sqam60 (piano) | 16.2 MB | 3,476,410 | 3,489,386 | **2,951,791** | **+15.1%** |
| sqam65 (orchestra) | 19.8 MB | 7,175,496 | 7,064,824 | **6,355,390** | **+11.4%** |
| sqam69 (ABBA) | 5.8 MB | 2,363,622 | 2,302,286 | **2,134,418** | **+9.7%** |
| music, 96 kHz 24-bit | 17.3 MB | 5,368,880 | 5,354,310 | **4,627,446** | **+13.8%** |
| instrumental, 48 kHz 24-bit | 13.0 MB | 6,913,483 | 6,831,914 | **6,244,651** | **+9.7%** |
| speech, mono | 2.0 MB | 775,395 | 749,730 | **713,210** | **+8.0%** |
| ABBA, 8-bit | 2.9 MB | 611,990 | 653,588 | **419,053** | **+31.5%** |

Against paq8px -8 on 1.5 MB excerpts: within 0.1–0.6% on 16-bit stereo and speech, 3% behind on solo piano, 7% behind on 8-bit — and 29% *ahead* on 24-bit, which paq8px has no model for.

### Images

Kodak test images as PPM/PGM/BMP (same pixels for every codec):

| | WebP lossless -z 9 | JPEG XL -e 9 | **augur** | paq8px -8 |
|---|---|---|---|---|
| 12 images, total | 5,152,832 | 4,684,382 | **3,484,909** | 3,171,165 |

All 24 Kodak **PNGs** as distributed (15.4 MB): **augur 9,095,522 (−40.9%)** with the previous image model (the current one is ~12% smaller on the same pixels; not yet re-measured on the PNGs), JPEG XL 10,156,780, WebP 11,252,092. Half of these PNGs were not written by zlib; reflate costs them under 1%.

augur is 25.6% smaller than JPEG XL here; paq8px is still ~10% smaller than augur on colour and 3–6% on greyscale.

### JPEG

| | original | **augur** | paq8px -8 |
|---|---|---|---|
| 8 Kodak JPEGs (cjpeg: plain, optimised, restart markers, greyscale) | 951,266 | **761,518 (−19.9%)** | 729,007 (−23.4%) |
| Nikon D80 photo, 46 MP | 14,679,474 | **10,220,623 (−30.4%)** | — |

### Structured data (earlier release)

Measured with v0.1-era builds and not re-run here; the models involved are unchanged.

| dataset | size | zstd-19 | xz-9e | **augur** | vs xz |
|---|---|---|---|---|---|
| nginx_logs | 7.0 MB | 26.54x | 29.31x | **56.42x** | **+92%** |
| taxi.ndjson | 91 MB | 27.98x | 31.84x | **53.58x** | **+68%** |
| taxi.csv | 22 MB | 8.12x | 8.45x | **13.96x** | **+65%** |
| gh_events.ndjson | 61 MB | 16.84x | 19.89x | **28.28x** | **+42%** |

## Build

```bash
cargo build --release
```

## Usage

```bash
augur compress data.ndjson            # writes data.ndjson.augur; format is auto-detected
augur compress photo.png -o p.augur
augur decompress p.augur -o photo.png # restores the original, byte for byte
augur bench data.ndjson               # compress + verify + timings, in memory
```

## How it works

### The context mixer

Every model is a `predict()` returning P(next bit = 1). Several context-selected logistic mixers blend them, a second layer blends the mixers, an SSE chain calibrates the result, and one binary arithmetic coder turns it into bits.

- **Order 1–24 context models** in checksummed 64-byte slots holding bit histories and direct counters, plus run, indirect, text-column and byte-class contexts.
- **Word models** — the token being typed, with bigram, trigram and skip-gram company.
- **Match models (hash chains)** — long-range repeats, with backward-context candidate selection.
- **Structure models** — a streaming parser names the semantic position (which JSON field, CSV column, SQL tuple column, XML element, log column) and the models condition on it.
- **Record-history and numeric models** — replay the previous record's value for this field; extrapolate counters and timestamps; detect cross-column relations.
- **Stride models** — find the record period of binary tables nothing declares.

The match, record and numeric models are oracles: a `TrustMap` learns how often each one is right in each situation, rather than trusting a tuned constant.

### Sample front-ends (`front.rs`)

Audio and raster data are numbers, not bytes. A front-end runs a numeric model in lockstep with the mixer: before each sample it makes a main prediction, and the mixer codes the *residual*. The model also keeps a portfolio of other predictors, and at every bit each contributes a context of the form "my guess, minus what is already known of this residual" — so the mixer learns per bit which predictor to believe here. Parametric (Laplace) inputs give it a calibrated prior from the first sample.

- **Audio:** forgetting least-squares solved by Cholesky (64 taps own channel + 32 cross-channel, including the *current* sample of the channel coded first), a sparse-lag least-squares fit reaching 1024 samples back for pitch periodicity, an RMS-normalised LMS cascade on its residual, and polynomial predictors. Wasted low bits are detected and dropped.
- **Images:** LOCO-I, CALIC and planar predictors, cross-component predictors borrowing the previous colour's local slope, five spatial least-squares fits over neighbourhoods of up to 32 taps, bias cancellation, and a **colour cache** — given the components of this pixel already coded, what the next one was the last two times. On the Kodak set that cache alone is worth 10%.

Floating point is used throughout and is safe: Rust never contracts `a*b+c` into a fused multiply-add, IEEE 754 correctly rounds `+ − × ÷ √`, and the one exponential is computed from those operations. The decoder reproduces every prediction bit for bit.

### Recompression (`deflate.rs`, `reflate.rs`, `recomp.rs`)

A deflate stream is noise to a context mixer — its redundancy was already squeezed out by a weaker model. augur finds deflate streams (zlib wrappers, gzip members, ZIP entries, PNG IDAT chunks, PNGs and JPEGs embedded anywhere), unpacks them, and recurses: a gzip of a tar of PNGs unpacks all the way to pixels. A recipe records how to put everything back.

- **zlib streams** are regenerated by a clean-room reimplementation of zlib's compressor — hash chains, lazy matching, window sliding, block-split points, and Huffman tree construction including its tie-breaking and length limiting. It matches the real zlib on all 3,480 parameter/input combinations in `bench/zcheck.py`. A parameter search with early abort finds the level, memLevel, window and strategy.
- **Everything else** — GNU gzip, Info-ZIP, zopfli, zlib-ng, 7-Zip, Microsoft Office — goes through **reflate**. The stream is parsed into tokens, and each token is compared with what a lazy matcher *would* emit given the data; the matcher learns the encoder's policy as it goes (one encoder never defers a 3-byte match; another ignores zlib's distance rule). Only disagreements are written down: on real PNGs 99.8% of tokens agree, and the description costs 0.3–1% of the image. When an unusual encoder makes the description too costly, a trial encode decides whether to unpack at all.
- **PNG** scanlines are unfiltered back into pixels, which go to the image model.

### JPEG (`jpeg.rs`)

The scan bytes pass through unchanged — nothing needs reconstructing — but the model decodes them bit by bit as they are coded: Huffman codes, extra bits, zero runs, MCU layout, restart markers and byte stuffing. Each Huffman bit is predicted from the same coefficient in the blocks above and to the left, from in-block neighbours rescaled across quantisers, and from paq8px's cross-block *edge prediction*: the neighbouring blocks' coefficients projected onto the shared boundary through the cosine basis, which predicts this block's coefficients before they are read. Inside a scan the byte models are skipped entirely.

### Container

```
"AUGR" | version (1) = 5 | mode (1) | mem_bits (1) | flags (1) | original_length (8) | virtual_length (8) | stream
```

The coded stream begins with the recipe (expansion tree and sample layouts), so the decoder knows where every sample region begins before it reaches it.

## Honest caveats

- **It is slow.** Generic data runs around 0.25 MB/s each way; audio, images and JPEG run at tens of KB/s, because every sample drives least-squares solves or every bit drives two dozen hashed lookups. Encode and decode cost about the same. This is a write-once, read-rarely compressor: archival, backups, cold storage.
- **Memory is up to ~1.3 GB** for large inputs.
- **paq8px is stronger** on prose, source code and executables (10–38% on Silesia's text and binary files), colour photographs (~10%), JPEG (~3.5 points of saving), and some audio. augur is ahead of every *practical* codec measured, not of every research one.
- **Not yet handled:** progressive JPEG (coded as ordinary bytes), MP3/AAC, FLAC, xz/bzip2/zstd payloads. These pass through the generic models and gain little.
- **Containers from v0.1 and earlier versions cannot be decoded** by this release; the format changed twice (versions 4 and 5).

## Testing

```bash
cargo test --release
python3 bench/zcheck.py   # the zlib clone against the system zlib, byte for byte
```

The suite covers the boundaries that break compressors: empty and 1-byte inputs, random data, malformed JSON, every PCM shape (8–32 bits, 1–6 channels, wasted bits, ragged frames, trailing chunks), images with padded rows and 16-bit noise, PNGs of every colour type, bit depth and filter (including interlaced and palette), a deliberately odd deflate encoder that forces reflate, JPEGs with restart markers, subsampling, greyscale and corrupted scans — with a check that the JPEG model follows each scan to its last MCU — and hundreds of bit-flipped and truncated containers, which must fail cleanly rather than panic.

## License

[Apache-2.0](LICENSE).
