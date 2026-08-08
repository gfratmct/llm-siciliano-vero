import glob
import os
import re

import torch

from torch.utils.data import DataLoader, Dataset, random_split

from tokenizers import Tokenizer


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

    def read(self) -> list[str]:
        """Read raw files, clean their text, and return sentences."""
        file_paths = self.get_paths()
        corpus = []

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                corpus.append(f.read())

        print(f"Corpus length: {len(corpus)} files")

        full_raw_text = "\n".join(corpus)

        cleaned_text = re.sub(r"<[^>]+>", "", full_raw_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_text = re.sub(r"##.*?##", "", cleaned_text, flags=re.DOTALL)

        split_sentences = re.split(r"(?<=[.\!?]) +", cleaned_text)
        return [sentence for sentence in split_sentences if sentence]


class TextDataset(Dataset):
    def __init__(self, corpus: str, tokenizer: Tokenizer, block_size: int):
        self.block_size = block_size
        self.examples = []

        encoding = tokenizer.encode(corpus)
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)

        if len(input_ids) < block_size:
            raise ValueError("Corpus is too short for the requested block_size.")

        for i in range(0, len(input_ids) - block_size + 1, block_size):
            self.examples.append(input_ids[i : i + block_size])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


def load_text_datasets(
    corpus: str,
    tokenizer: Tokenizer,
    block_size: int,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Create train/test split datasets from the full dataset."""
    full_dataset = TextDataset(corpus, tokenizer, block_size)
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
) -> tuple[DataLoader, DataLoader]:
    """Wrap datasets with PyTorch DataLoader objects."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == "__main__":
    reader = DatasetReader("data/")
    sentences = reader.read()
    corpus = " ".join(sentences)[0:1000]

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
        print(f"Train batch {batch_idx} shape: {batch.shape}")
        print(batch)
        break

    for batch_idx, batch in enumerate(test_loader):
        print(f"Test batch {batch_idx} shape: {batch.shape}")
        print(batch)
        break
