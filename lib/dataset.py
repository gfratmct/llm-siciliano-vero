import glob
import hashlib
import os
import re

import torch

from torch.utils.data import DataLoader, Dataset, random_split

from tokenizers import Tokenizer

# Encode the corpus in chunks of this many characters so peak memory stays
# bounded. Encoding the whole corpus at once materializes a Python list of
# hundreds of millions of ints (tens of GB) and crashes machines with large
# datasets. Chunking deviates only at chunk boundaries (a handful of tokens
# across the whole corpus), which is the standard LLM sharding trade-off.
_TOKENIZE_CHUNK_SIZE = 20_000_000


def _encode_corpus(corpus: str, tokenizer: Tokenizer) -> torch.Tensor:
    """Encode a large corpus in chunks, returning a flat int32 token tensor."""
    chunk_tensors = []
    total = 0
    for start in range(0, len(corpus), _TOKENIZE_CHUNK_SIZE):
        piece = corpus[start : start + _TOKENIZE_CHUNK_SIZE]
        encoding = tokenizer.encode(piece)
        chunk_tensors.append(torch.tensor(encoding.ids, dtype=torch.int32))
        total += chunk_tensors[-1].numel()
    print(f"Tokenized corpus: {total / 1e6:.0f}M tokens")
    return torch.cat(chunk_tensors)


class DatasetReader:
    whitelist_extensions = [".utf8"]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_paths(self) -> list[str]:
        """Return all supported text file paths from the configured data directory."""
        patterns = [os.path.join(self.data_dir, f"*{ext}") for ext in self.whitelist_extensions]
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(pattern))
        return sorted(file_paths)

    def fingerprint(self) -> str:
        """Return a short hash of the data files, used to key the token cache on disk."""
        hasher = hashlib.sha256()
        for file_path in self.get_paths():
            if not os.path.exists(file_path):
                continue
            stat = os.stat(file_path)
            hasher.update(file_path.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
        return hasher.hexdigest()[:16]

    def read(self) -> str:
        """Read raw files and return the cleaned corpus as a single string."""
        file_paths = self.get_paths()
        parts = []

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                parts.append(f.read())

        print(f"Corpus length: {len(parts)} files")

        full_raw_text = "\n".join(parts)

        cleaned_text = re.sub(r"<[^>]+>", "", full_raw_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_text = re.sub(r"##.*?##", "", cleaned_text, flags=re.DOTALL)

        return cleaned_text


class TextDataset(Dataset):
    def __init__(self, corpus: str | None, tokenizer: Tokenizer, block_size: int, cache_path: str | None = None):
        self.block_size = block_size
        self.tokens = self._load_tokens(corpus, tokenizer, cache_path)

        if self.tokens.numel() < block_size + 1:
            raise ValueError("Corpus is too short for the requested block_size.")

    def _load_tokens(self, corpus: str | None, tokenizer: Tokenizer, cache_path: str | None) -> torch.Tensor:
        """Return the flat token ids, reading from cache when available or tokenizing once."""
        if cache_path is not None and os.path.exists(cache_path):
            print(f"Loaded {os.path.getsize(cache_path) / 1e6:.0f} MB token cache from {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        if corpus is None:
            raise ValueError("No cached tokens found and no corpus provided.")

        tokens = _encode_corpus(corpus, tokenizer)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save(tokens, cache_path)
            print(f"Cached tokens to {cache_path}")

        return tokens

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) where each token predicts the next one."""
        start = idx * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1].long(), chunk[1:].long()


def load_text_datasets(
    corpus: str | None,
    tokenizer: Tokenizer,
    block_size: int,
    test_ratio: float = 0.1,
    seed: int = 42,
    cache_path: str | None = None,
) -> tuple[Dataset, Dataset]:
    """Create train/test split datasets from the full dataset."""
    full_dataset = TextDataset(corpus, tokenizer, block_size, cache_path=cache_path)
    test_size = max(1, int(len(full_dataset) * test_ratio))
    train_size = len(full_dataset) - test_size

    if train_size < 1:
        raise ValueError("Not enough examples to create a train/test split.")

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Wrap datasets with PyTorch DataLoader objects."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    reader = DatasetReader("data/")
    corpus = reader.read()[:1000]

    if not corpus:
        print("No data files found in data/. Using a small sample corpus for test execution.")
        corpus = "This is a sample sentence for dataset loader testing. " * 20

    tokenizer = Tokenizer.from_pretrained("gpt2")
    block_size = 32
    train_dataset, test_dataset = load_text_datasets(corpus, tokenizer, block_size, test_ratio=0.2)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=2)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        print(f"Train batch {batch_idx} x shape: {x.shape}")
        print(f"Train batch {batch_idx} y shape: {y.shape}")
        print("x:", x)
        print("y:", y)
        break

    for batch_idx, batch in enumerate(test_loader):
        x, y = batch
        print(f"Test batch {batch_idx} x shape: {x.shape}")
        break
