#!/bin/bash

set -euo pipefail

echo "Downloading dataset..."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"

mkdir -p "$DATA_DIR"
curl \
    --fail \
    --location \
    --remote-name-all \
    --output-dir "$DATA_DIR" \
    'https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/paisa.raw.utf8.gz'

# unpack the dataset
gzip -df "$DATA_DIR/paisa.raw.utf8.gz"

echo "Download now the wikipedia italian version..."

hf download wikimedia/wikipedia --repo-type dataset --include "20231101.it/*.parquet" --local-dir $DATA_DIR

echo "Download completed in $DATA_DIR"
