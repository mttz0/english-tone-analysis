import re
import unicodedata
from collections import Counter
from typing import List

import torch
from torch.utils.data import Dataset


def _normalize_quotes_and_punct(text: str) -> str:
    # Normalizzazione Unicode
    s = unicodedata.normalize("NFKC", str(text))
    # Normalizza virgolette e trattini
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    s = s.replace("–", "-").replace("—", "-")
    # Normalizza ellipsis di puntini sospensivi
    s = s.replace("…", "...")
    # Riduce spazi eccessivi
    s = re.sub(r"\s+", " ", s)
    return s.strip()


_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "i'll": "i will",
    "you're": "you are",
    "you'd": "you would",
    "you'll": "you will",
    "you've": "you have",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "let's": "let us",
}


def _expand_contractions(text: str) -> str:
    s = text
    # Applica prima i più lunghi per evitare sovrapposizioni parziali
    for k in sorted(_CONTRACTIONS.keys(), key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(k)}\b", _CONTRACTIONS[k], s)
    return s


def normalize_text_for_model(text: str) -> str:
    s = _normalize_quotes_and_punct(text)
    s = s.lower()
    s = _expand_contractions(s)
    # Rimuove problemi di spaziatura della punteggiatura
    s = re.sub(r"\s+", " ", s).strip()
    return s


def signature_for_dedupe(text: str) -> str:
    s = normalize_text_for_model(text)
    # Rimuove tutti i caratteri non alfanumerici
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def simple_tokenize(s: str) -> List[str]:
    s = normalize_text_for_model(s)
    # Tokenizza parole, numeri e separa la punteggiatura
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", s)


def build_vocab(texts: List[str], min_freq: int = 2, specials=("<pad>", "<unk>")):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(str(t)))
    itos = list(specials)
    for w, c in counter.most_common():
        if c >= min_freq and w not in specials:
            itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def numericalize(tokens: List[str], stoi: dict, unk_idx: int) -> List[int]:
    return [stoi.get(tok, unk_idx) for tok in tokens]


def pad_or_truncate(seq: List[int], max_len: int, pad_idx: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len=128, pad_idx=0, unk_idx=1):
        self.texts = [str(t) for t in texts]
        self.labels = [int(label_value) for label_value in labels]
        self.stoi = stoi
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = simple_tokenize(self.texts[idx])
        ids = numericalize(toks, self.stoi, self.unk_idx)
        ids = pad_or_truncate(ids, self.max_len, self.pad_idx)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
