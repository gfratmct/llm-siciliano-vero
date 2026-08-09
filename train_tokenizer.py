"""Train a byte-level BPE tokenizer on the Italian corpus and save it for reuse.

Usage:
    python train_tokenizer.py [--vocab-size 50257] [--output models/tokenizer.json]

The trained tokenizer is then picked up automatically by lib.tokenizer.Tokenizer
whenever it is instantiated, replacing the GPT-2 fallback.
"""

import argparse
import os
import sys

from tokenizers import Tokenizer as BaseTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.dataset import DatasetReader
from lib.tokenizer import SPECIAL_TOKENS

DEFAULT_VOCAB_SIZE = 50257
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tokenizer.json")
DEFAULT_DATA_DIR = "data"


def train_tokenizer(data_dir: str, vocab_size: int, output_path: str) -> None:
    reader = DatasetReader(data_dir)

    tokenizer = BaseTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
    )

    tokenizer.train_from_iterator(reader.iter_text_chunks(), trainer=trainer)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    print(f"Saved tokenizer with vocab size {tokenizer.get_vocab_size()} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on the Italian corpus.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to the data directory.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Target vocabulary size.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to save the trained tokenizer.")
    args = parser.parse_args()

    train_tokenizer(args.data_dir, args.vocab_size, args.output)
