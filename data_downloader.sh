#!/bin/bash
set -euo pipefail

destination="${1:-data}"
archive="$destination/fruits.zip"

mkdir -p "$destination"
curl --fail --location --retry 3 \
  --output "$archive" \
  "https://www.kaggle.com/api/v1/datasets/download/moltean/fruits"
unzip -q "$archive" -d "$destination"
rm "$archive"

printf 'Dataset extracted under %s\n' "$destination"