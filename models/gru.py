import torch
import torch.nn as nn


class BiGRUClassifier(nn.Module):
    """
    Modello BiGRU per classificazione del sentiment/tono testuale.
    
    Architettura:
    - Embedding layer per trasformare token IDs in vettori
    - BiGRU bidirezionale multi-layer per catturare contesto sequenziale
    - Pooling configurabile per aggregare le rappresentazioni temporali
    - MLP opzionale per proiezione intermedia
    - Linear finale per classificazione multi-classe
    
    Args:
        vocab_size: dimensione del vocabolario
        emb_dim: dimensione vettori embedding
        hidden_dim: dimensione hidden state GRU
        num_classes: numero classi target
        pad_idx: indice del token di padding
        num_layers: numero layer GRU (default: 1)
        bidirectional: se usare GRU bidirezionale (default: True)
        dropout: tasso dropout per regolarizzazione (default: 0.5)
        pooling: strategia pooling ("last", "max", "mean", "attn")
        mlp_hidden: dimensione MLP intermedio (0 = disabilitato)
    """
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_classes,
        pad_idx=0,
        num_layers=1,
        bidirectional=True,
        dropout=0.5,
        pooling="max",  # opzioni: "last", "max", "mean", "attn"
        mlp_hidden: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.pooling = pooling
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        if self.pooling == "attn":
            self.attn = nn.Sequential(
                nn.Linear(out_dim, max(2, out_dim // 2)),
                nn.Tanh(),
                nn.Linear(max(2, out_dim // 2), 1),
            )
        self.mlp_hidden = mlp_hidden
        if mlp_hidden and mlp_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(out_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fc = nn.Linear(mlp_hidden, num_classes)
        else:
            self.proj = None
            self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        E = self.emb(x)
        E = self.emb_dropout(E)
        out, _ = self.gru(E)
        mask = (x != self.pad_idx).float()  # (B, T)

        if self.pooling == "last":
            lengths = mask.sum(dim=1).long().clamp(min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
            pooled = out.gather(1, idx).squeeze(1)
        elif self.pooling == "mean":
            summed = (out * mask.unsqueeze(-1)).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            pooled = summed / denom
        elif self.pooling == "attn":
            scores = self.attn(out).squeeze(-1)  # (B, T)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
            pooled = (weights * out).sum(dim=1)
        else:  # "max"
            # imposta posizioni padding a valori negativi prima del max
            out_masked = out.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            pooled = out_masked.max(dim=1).values

        pooled = self.dropout(pooled)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return self.fc(pooled)


