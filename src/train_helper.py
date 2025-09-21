import math

import numpy as np
import torch
import torch.nn as nn

from .metrics import accuracy, per_class_f1


def collect_predictions(model, loader, device="cpu"):
    """Raccoglie le predizioni del modello su un dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([])
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Esegue un'epoca di training"""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, loader, criterion, num_classes, device="cpu"):
    """Valuta il modello su un dataset e calcola le metriche"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    if n_batches == 0:
        return math.nan, math.nan, []
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    f1s = per_class_f1(all_logits, all_labels, num_classes)
    return total_loss / n_batches, total_acc / n_batches, f1s
