"""
Script per la predizione del tono usando un modello BiGRU addestrato.

Questo script carica un modello BiGRU pre-addestrato e fornisce funzionalità per
predire il tono di testi inglesi in input. Il modello classifica i testi in 4 categorie:
- 0: neutral
- 1: polite  
- 2: professional
- 3: casual

Funzionalità:
- Caricamento automatico del modello da checkpoint
- Predizione singola o batch di testi
- Calcolo delle probabilità per ogni classe
- Supporto per input da riga di comando

File richiesti:
- checkpoints/sentiment_gru_best.pt: modello addestrato salvato da train.py

Utilizzo:
1. Da riga di comando con testi specifici:
   python predict.py "Hello world" "Thank you very much"

2. Da riga di comando senza argomenti (usa esempi di default):
   python predict.py

Output:
Per ogni testo viene mostrato:
- Il testo originale
- La classe predetta
- La confidence score (probabilità massima)
- Le probabilità per tutte le classi
"""

import sys
import os
import torch
from config import Config
from models.gru import BiGRUClassifier
from src.utils import select_device
from src.text_processing import simple_tokenize, pad_or_truncate, numericalize

# -------- Carica checkpoint --------
project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, "checkpoints")
checkpoint_path = os.path.join(model_dir, "sentiment_gru_best.pt")
_cfg = Config()
device = select_device(_cfg.device)
checkpoint = torch.load(checkpoint_path, map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint["config"]

pad_idx = config["pad_idx"]
unk_idx = config["unk_idx"]
max_len = config["max_len"]
num_classes = config["num_classes"]

# -------- Ricostruisce il modello (BiGRU) --------
# Usa la configurazione dal checkpoint per garantire la compatibilità
_cfg = Config()
model = BiGRUClassifier(
    len(itos),
    config.get("emb_dim", _cfg.emb_dim),
    config.get("hidden_dim", _cfg.hidden_dim),
    num_classes,
    pad_idx=pad_idx,
    num_layers=int(config.get("num_layers", _cfg.num_layers)),
    bidirectional=bool(config.get("bidirectional", True)),
    dropout=float(config.get("dropout", _cfg.dropout)),
    pooling=str(config.get("pooling", _cfg.pooling)),
    mlp_hidden=int(config.get("mlp_hidden", _cfg.mlp_hidden)),
)

model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# -------- Funzione di predizione --------
def predict(texts, show_confidence=True):
    results = []
    with torch.no_grad():
        for t in texts:
            tokens = simple_tokenize(t)
            ids = numericalize(tokens, stoi, unk_idx)
            ids = pad_or_truncate(ids, max_len, pad_idx)
            x = torch.tensor([ids], dtype=torch.long, device=device)  # batch di dimensione 1
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0][pred].item()
            
            if show_confidence:
                results.append((pred, confidence, probs[0].tolist()))
            else:
                results.append(pred)
    return results

# -------- Mappatura delle etichette --------
label_map = {
    0: "neutral",
    1: "polite",
    2: "professional",
    3: "casual"
}

# -------- Main --------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_texts = sys.argv[1:]  # prende le frasi da terminale
    else:
        # Esempi di default
        sample_texts = [
            "I’m waiting to know the deadline for this task.",
            "I would greatly appreciate it if might kindly let me know the deadline for this task.",
            "Can you please confirm the deadline for this assignment?",
            "Hey dude, done?"
        ]

    preds = predict(sample_texts)
    for t, (p, conf, all_probs) in zip(sample_texts, preds):
        print(f"{t}")
        print(f"  --> {label_map.get(p, p)} (confidence: {conf:.3f})")
        print(f"  Tulle le probabilità: {[f'{label_map.get(i, i)}: {prob:.3f}' for i, prob in enumerate(all_probs)]}")
