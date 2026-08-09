import os
from typing import List

from tokenizers import Tokenizer as BaseTokenizer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TURN_TOKEN = "<|end|>"
SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TURN_TOKEN,
]

# Default location for a tokenizer trained on the project corpus. By default
# outputs (tokenizer.json, config.json, checkpoints) live under ./models.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TOKENIZER_PATH = os.path.join(_PROJECT_ROOT, "models", "tokenizer.json")


class Tokenizer:
    """Wrapper around the GPT-2 / custom Italian tokenizer with chat special tokens."""

    PAD_TOKEN = PAD_TOKEN
    UNK_TOKEN = UNK_TOKEN
    SYSTEM_TOKEN = SYSTEM_TOKEN
    USER_TOKEN = USER_TOKEN
    ASSISTANT_TOKEN = ASSISTANT_TOKEN
    END_TURN_TOKEN = END_TURN_TOKEN

    def __init__(self, model_name: str = "gpt2", tokenizer_path: str | None = None):
        path = tokenizer_path or _DEFAULT_TOKENIZER_PATH
        if os.path.exists(path):
            self._tokenizer = BaseTokenizer.from_file(path)
        else:
            self._tokenizer = BaseTokenizer.from_pretrained(model_name)
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Register any special tokens that are not already in the vocabulary."""
        existing = self._tokenizer.get_vocab()
        missing = [tok for tok in SPECIAL_TOKENS if tok not in existing]
        if missing:
            self._tokenizer.add_special_tokens(missing)

    # Vocabulary / ID helpers
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_vocab_size(self) -> int:
        """Mirror the underlying tokenizer API used in the training script."""
        return self.vocab_size

    def token_to_id(self, token: str) -> int:
        """Return the ID for a token, or the UNK ID if it does not exist."""
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            return self.unk_id
        return token_id

    def id_to_token(self, token_id: int) -> str:
        return self._tokenizer.id_to_token(token_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id(self.PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.token_to_id(self.UNK_TOKEN)

    @property
    def system_id(self) -> int:
        return self.token_to_id(self.SYSTEM_TOKEN)

    @property
    def user_id(self) -> int:
        return self.token_to_id(self.USER_TOKEN)

    @property
    def assistant_id(self) -> int:
        return self.token_to_id(self.ASSISTANT_TOKEN)

    @property
    def end_turn_id(self) -> int:
        return self.token_to_id(self.END_TURN_TOKEN)

    # Encoding / decoding
    def encode(self, text: str, add_special_tokens: bool = True):
        """Return a tokenizers Encoding object (has .ids, .attention_mask, etc.)."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # Chat template helper
    def format_chat(self, messages: List[dict]) -> str:
        """
        Convert a list of chat messages into a single prompt string.

        Example:
            messages = [
                {"role": "system", "content": "Sei un assistente utile."},
                {"role": "user", "content": "Ciao!"},
            ]
        Returns:
            "<|system|>Sei un assistente utile.<|end|><|user|>Ciao!<|end|><|assistant|>"
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>{content}{self.END_TURN_TOKEN}")
        parts.append(self.ASSISTANT_TOKEN)
        return "".join(parts)
