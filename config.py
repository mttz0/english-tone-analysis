from dataclasses import dataclass


@dataclass
class Config:
    """Configurazione per l'addestramento del modello"""
    # Dati e addestramento
    epochs: int = 30
    emb_dim: int = 300
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    pooling: str = "mean"  # last | max | mean | attn
    mlp_hidden: int = 256
    max_len: int = 256
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping: bool = True
    es_metric: str = "val_loss"
    es_patience: int = 3
    es_min_delta: float = 1e-3
    min_freq: int = 2
    seed: int = 42
    device: str = "auto"
