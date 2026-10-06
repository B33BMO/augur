#!/usr/bin/env bash
# Download the canonical public sources for the benchmark datasets.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data

get() { [ -f "$2" ] || curl -sSL --fail -o "$2" "$1"; echo "  $2 ($(wc -c <"$2") bytes)"; }

echo "fetching:"
get "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_logs/nginx_logs" nginx_logs
get "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet" yellow_tripdata_2024-01.parquet
get "https://data.gharchive.org/2024-01-01-15.json.gz" gh_2024-01-01-15.json.gz

# enwik8 ships inside a zip of the Wikipedia dump snapshot.
if [ ! -f enwik8 ]; then
  curl -sSL --fail -o enwik8.zip "https://mattmahoney.net/dc/enwik8.zip"
  unzip -qo enwik8.zip && rm -f enwik8.zip
fi
echo "  enwik8 ($(wc -c <enwik8) bytes)"

# Silesia: the standard mixed corpus, 12 files, unpacked into data/silesia/.
if [ ! -d silesia ]; then
  curl -sSL --fail -o silesia.zip "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
  mkdir -p silesia && unzip -qo silesia.zip -d silesia && rm -f silesia.zip
fi
echo "  silesia/ ($(ls silesia | wc -l) files)"
