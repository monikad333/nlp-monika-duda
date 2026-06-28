from __future__ import annotations

import numpy as np

from common.nlp_task import clean_text
from lab03.config import MAX_LEN, MAX_VOCAB_SIZE


def clean_texts(texts: list[str]) -> list[str]:
    return [clean_text(text) for text in texts]


def build_tokenizer(texts: list[str], max_vocab_size: int = MAX_VOCAB_SIZE):
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer, texts: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
