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
    'https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3{/paisa.raw.utf8.gz,/paisa.annotated.CoNLL.utf8.gz,/lemma-WITHOUTnumberssymbols-frequencies-paisa.txt.gz,/lemma-frequencies-paisa.txt.gz}'

# unpack the dataset
gzip -df "$DATA_DIR/paisa.raw.utf8.gz"
# gzip -df "$DATA_DIR/paisa.annotated.CoNLL.utf8.gz"
# gzip -df "$DATA_DIR/lemma-WITHOUTnumberssymbols-frequencies-paisa.txt.gz"
# gzip -df "$DATA_DIR/lemma-frequencies-paisa.txt.gz"

echo "Download completed in $DATA_DIR"
