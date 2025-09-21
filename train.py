
"""
Script di addestramento per il modello BiGRU per classificazione del tono.

Questo script addestra un modello Bidirectional GRU per classificare il tono di testi inglesi
in 4 categorie: neutral, polite, professional, casual. Il modello utilizza embedding,
pooling e un MLP finale per la classificazione.

Funzionalità principali:
- Caricamento automatico dei dataset da CSV
- Divisione train/validation automatica (70/30)
- Costruzione del vocabolario dai dati di training
- Addestramento con early stopping e learning rate scheduling
- Valutazione su test set con metriche complete
- Generazione di grafici e visualizzazioni
- Salvataggio del miglior modello

File CSV richiesti nella cartella datasets/:
- train.csv: colonne 'text', 'label' (ID intero per la classe)
- test.csv: colonne 'text', 'label'
- Opzionale: colonna 'label_name' per nomi delle classi

Output:
- Modello salvato in: checkpoints/sentiment_gru_best.pt
- Grafici salvati in: plots/
  - training_curves.png: curve di loss e accuracy
  - confusion_matrix.png: matrice di confusione
  - per_class_f1.png: F1 score per classe

Configurazione:
Il modello viene configurato tramite il file config.py che contiene:
- Parametri del modello (embedding dim, hidden dim, etc.)
- Iperparametri di training (learning rate, batch size, etc.)
- Configurazioni per early stopping e scheduling

Esempio di utilizzo:
    python train.py

"""

import os
import time

import numpy as np
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from src.utils import set_seed, select_device
from src.text_processing import build_vocab, TextDataset
from src.metrics import per_class_f1, compute_confusion_matrix
from src.train_helper import run_epoch, evaluate, collect_predictions
from models.gru import BiGRUClassifier
from src.plotting import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_f1,
)


# --------------------------
# Main
# --------------------------


def main():
    cfg = Config()

    set_seed(cfg.seed)

    # Verifica che i dataset esistano nella cartella datasets/
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "datasets")
    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        print("Train and tests dataset presenti")
    else:
        print("Dataset non trovati in 'datasets/' (train.csv e/o test.csv). Prepararli con prepare_datasets.py prima dell'addestramento")
        return

    device = select_device(cfg.device)
    print(f"Dispositivo utilizzato: {device}")

    # Salva sempre i grafici nella cartella plots
    plots_dir = "plots"

    # Carica i dati (CSV con header: text,label)
    def load_csv_text_label(path):
        texts = []
        labels = []
        label_names_map = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must contain 'text' and 'label' columns")
            has_label_name = "label_name" in reader.fieldnames
            for row in reader:
                label_int = int(row["label"]) if row["label"] != "" else 0
                texts.append(row.get("text", ""))
                labels.append(label_int)
                if has_label_name:
                    name = row.get("label_name", "").strip()
                    if name != "":
                        label_names_map[label_int] = name
        # Deriva i nomi delle classi se possibile (assume classi contigue da 0)
        if labels:
            num_classes_guess = int(len(set(labels)))
            # Costruisce nomi ordinati 0..C-1
            class_names = []
            for i in range(num_classes_guess):
                class_names.append(label_names_map.get(i, str(i)))
        else:
            class_names = []
        return texts, labels, class_names

    all_train_texts, all_train_labels, train_class_names = load_csv_text_label(train_csv_path)
    test_texts, test_labels, _ = load_csv_text_label(test_csv_path)

    # Divisione train/validation (70/30)
    rng = np.random.RandomState(cfg.seed)
    num_all = len(all_train_texts)
    indices = np.arange(num_all)
    rng.shuffle(indices)
    split_at = int(num_all * 0.7)
    train_idx = indices[:split_at]
    val_idx = indices[split_at:]

    train_texts = [all_train_texts[i] for i in train_idx]
    train_labels = [all_train_labels[i] for i in train_idx]
    val_texts = [all_train_texts[i] for i in val_idx]
    val_labels = [all_train_labels[i] for i in val_idx]

    # Inferisce il numero di classi dai label di training
    num_classes = int(len(set(train_labels)))
    print(f"Rilevate {num_classes} classi.")
    class_names = train_class_names if len(train_class_names) == num_classes else [str(i) for i in range(num_classes)]

    # Costruisce il vocabolario solo sui dati di training
    stoi, itos = build_vocab(train_texts, min_freq=cfg.min_freq)
    pad_idx = stoi["<pad>"]
    unk_idx = stoi["<unk>"]
    vocab_size = len(itos)
    print(f"Dimensione vocabolario: {vocab_size} (min_freq={cfg.min_freq})")

    # Dataset e DataLoader
    train_ds = TextDataset(train_texts, train_labels, stoi, max_len=cfg.max_len, pad_idx=pad_idx, unk_idx=unk_idx)
    val_ds   = TextDataset(val_texts,   val_labels,   stoi, max_len=cfg.max_len, pad_idx=pad_idx, unk_idx=unk_idx)
    test_ds  = TextDataset(test_texts,  test_labels,  stoi, max_len=cfg.max_len, pad_idx=pad_idx, unk_idx=unk_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Modello
    model = BiGRUClassifier(
        vocab_size,
        cfg.emb_dim,
        cfg.hidden_dim,
        num_classes,
        pad_idx=pad_idx,
        num_layers=cfg.num_layers,
        bidirectional=True,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        mlp_hidden=cfg.mlp_hidden,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, cooldown=0, min_lr=1e-5)

    # Monitoraggio dell'addestramento
    best_state = None
    tr_losses, tr_accs, val_losses, val_accs, val_f1_macros = [], [], [], [], []
    best_metric = None
    epochs_without_improve = 0
    monitor = cfg.es_metric
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        v_loss, v_acc, v_f1s = evaluate(model, val_loader, criterion, num_classes, device=device)
        dt = time.time() - t0
        if np.isfinite(v_loss):
            scheduler.step(v_loss)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        v_macro = float(np.mean(v_f1s)) if len(v_f1s) > 0 else float("nan")
        val_f1_macros.append(v_macro)
        print(f"[{epoch:02d}/{cfg.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={v_loss:.4f} acc={v_acc:.4f} f1={v_macro:.4f} | "
              f"{dt:.1f}s")

        # Metrica per early stopping
        if monitor == "val_loss":
            current = -v_loss
        elif monitor == "val_acc":
            current = v_acc
        else:
            current = v_macro
        if (best_metric is None) or (current > best_metric + cfg.es_min_delta):
            best_metric = current
            epochs_without_improve = 0
            best_state = {
                "model_state": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "emb_dim": cfg.emb_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "max_len": cfg.max_len,
                    "pad_idx": pad_idx,
                    "unk_idx": unk_idx,
                    "num_classes": num_classes,
                    # include tutti gli iperparametri per il caricamento
                    "num_layers": cfg.num_layers,
                    "dropout": cfg.dropout,
                    "pooling": cfg.pooling,
                    "mlp_hidden": cfg.mlp_hidden,
                    "bidirectional": True,
                }
            }
        else:
            epochs_without_improve += 1
            if cfg.early_stopping and epochs_without_improve >= cfg.es_patience:
                print("Early stopping attivato.")
                break

    # Valutazione finale su test set con il miglior modello
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    te_loss, te_acc, te_f1s = evaluate(model, test_loader, criterion, num_classes, device=device)
    print(f"Test finale: loss={te_loss:.4f} acc={te_acc:.4f} f1={np.mean(te_f1s):.4f}")

    # Salva il miglior modello
    model_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "sentiment_gru_best.pt")
    torch.save(best_state, out_path)
    print(f"Checkpoint salvato in: {out_path}")

    # Genera sempre i grafici
    plot_training_curves(tr_losses, val_losses, tr_accs, val_accs, val_f1_macros, plots_dir)

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        preds_np, labels_np = collect_predictions(model, test_loader, device=device)
        if preds_np.size > 0:
            cm = compute_confusion_matrix(preds_np, labels_np, num_classes)
            plot_confusion_matrix(cm, os.path.join(plots_dir, "confusion_matrix.png"), class_names=class_names, normalize=True, annotate=True)
            labels_tensor = torch.from_numpy(labels_np)
            preds_tensor_logits = torch.nn.functional.one_hot(torch.from_numpy(preds_np), num_classes=num_classes).float()
            per_class_f1_scores = per_class_f1(preds_tensor_logits, labels_tensor, num_classes)
            plot_per_class_f1(per_class_f1_scores, os.path.join(plots_dir, "per_class_f1.png"), class_names=class_names)


if __name__ == "__main__":
    main()
