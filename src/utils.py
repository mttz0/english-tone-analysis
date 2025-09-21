import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(preference: str = "auto") -> str:
    """
    Restituisce il device da usare tra "cpu", "cuda" o "mps" in base alla preferenza.
    - preference="auto": priorità MPS (Apple), poi CUDA, infine CPU.
    - preference in {"cpu","cuda","mps"}: ritorna se disponibile, altrimenti fallback a CPU.
    """
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"