import numpy as np
import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == labels).float().mean().item()


def per_class_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    preds_cls = preds.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds_cls == c) & (labels_np == c))
        fp = np.sum((preds_cls == c) & (labels_np != c))
        fn = np.sum((preds_cls != c) & (labels_np == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return f1s


def compute_confusion_matrix(preds_cls_np: np.ndarray, labels_np: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(preds_cls_np, labels_np):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            cm[label, pred] += 1
    return cm
