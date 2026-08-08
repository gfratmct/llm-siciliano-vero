#!/bin/bash

echo "Downloading dataset..."

cd .. && mkdir data/ && curl --remote-name-all https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3{/paisa.raw.utf8.gz,/paisa.annotated.CoNLL.utf8.gz,/lemma-WITHOUTnumberssymbols-frequencies-paisa.txt.gz,/lemma-frequencies-paisa.txt.gz}

# unpack the dataset
gzip -d paisa.raw.utf8.gz
# gzip -d paisa.annotated.CoNLL.utf8.gz
# gzip -d lemma-WITHOUTnumberssymbols-frequencies-paisa.txt.gz
# gzip -d lemma-frequencies-paisa.txt.gz

echo "Download completed"