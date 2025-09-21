import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _apply_common_axes_style(ax, title, y_label):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False)


def _maybe_mark_best(ax, x_values, y_values, mode="min", label_prefix="best"):
    if len(x_values) == 0:
        return
    if mode == "min":
        best_idx = int(np.nanargmin(y_values))
    else:
        best_idx = int(np.nanargmax(y_values))
    best_x = x_values[best_idx]
    best_y = y_values[best_idx]
    ax.scatter([best_x], [best_y], s=60, color="#d62728", zorder=5, label=f"{label_prefix} @ {best_x}")
    ax.annotate(
        f"{best_y:.3f}",
        (best_x, best_y),
        textcoords="offset points",
        xytext=(6, -6 if mode == "max" else 8),
        ha="left",
        fontsize=9,
    )


def plot_training_curves(tr_losses, te_losses, tr_accs, te_accs, te_f1_macros, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(tr_losses) + 1)

    # Loss
    fig, ax = plt.subplots(dpi=140)
    ax.plot(epochs, tr_losses, label="train", marker="o", linewidth=2)
    ax.plot(epochs, te_losses, label="val", marker="o", linewidth=2)
    _maybe_mark_best(ax, epochs, te_losses, mode="min", label_prefix="best val")
    _apply_common_axes_style(ax, "Loss", "Loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(dpi=140)
    ax.plot(epochs, tr_accs, label="train", marker="o", linewidth=2)
    ax.plot(epochs, te_accs, label="val", marker="o", linewidth=2)
    _maybe_mark_best(ax, epochs, te_accs, mode="max", label_prefix="best val")
    _apply_common_axes_style(ax, "Accuracy", "Accuracy")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close(fig)

    # Macro-F1 (validazione)
    fig, ax = plt.subplots(dpi=140)
    ax.plot(epochs, te_f1_macros, label="val macro-F1", marker="o", linewidth=2)
    _maybe_mark_best(ax, epochs, te_f1_macros, mode="max", label_prefix="best val")
    _apply_common_axes_style(ax, "Macro-F1", "Macro-F1")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "macro_f1_curve.png"))
    plt.close(fig)

    # Plot training e validazione
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    ax1.plot(epochs, tr_losses, label="train", marker="o", linewidth=2)
    ax1.plot(epochs, te_losses, label="val", marker="o", linewidth=2)
    _maybe_mark_best(ax1, epochs, te_losses, mode="min", label_prefix="best val")
    _apply_common_axes_style(ax1, "Loss", "Loss")

    ax2.plot(epochs, tr_accs, label="train", marker="o", linewidth=2)
    ax2.plot(epochs, te_accs, label="val", marker="o", linewidth=2)
    _maybe_mark_best(ax2, epochs, te_accs, mode="max", label_prefix="best val")
    _apply_common_axes_style(ax2, "Accuracy", "Accuracy")
    ax2.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close(fig)


def plot_confusion_matrix(cm, out_path, class_names=None, normalize=True, annotate=True):
    import numpy as np
    num_classes = cm.shape[0]
    if class_names is None or len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums
        matrix_to_show = cm_norm
    else:
        matrix_to_show = cm

    plt.figure(figsize=(max(6, num_classes * 0.8), max(5, num_classes * 0.8)))
    im = plt.imshow(matrix_to_show, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    if annotate:
        thresh = matrix_to_show.max() / 2.0 if matrix_to_show.size > 0 else 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j]
                if cm.sum(axis=1)[i] == 0:
                    pct = 0.0
                else:
                    pct = cm[i, j] / cm.sum(axis=1)[i]
                text_color = "white" if matrix_to_show[i, j] > thresh else "black"
                plt.text(j, i, f"{count}\n{pct:.0%}", ha="center", va="center", color=text_color, fontsize=9)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_class_f1(f1s, out_path, class_names=None):
    import numpy as np
    plt.figure(figsize=(max(6, len(f1s) * 0.8), 4))
    indices = np.arange(len(f1s))
    plt.bar(indices, f1s)
    plt.xlabel("Class")
    plt.ylabel("F1 score")
    plt.title("Per-class F1")
    plt.ylim(0.0, 1.0)
    if class_names is None or len(class_names) != len(f1s):
        plt.xticks(indices, indices)
    else:
        plt.xticks(indices, class_names, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


