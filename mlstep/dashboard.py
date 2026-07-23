
"""Live Matplotlib dashboard for UKCA timestep-halving training."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class LiveTrainingDashboard:
    """Display and refresh useful validation plots after every epoch."""

    def __init__(self, n_classes, k=100, baseline_metrics=None):
        self.n_classes = n_classes
        self.k = k
        self.baseline_metrics = baseline_metrics
        self.closed = False

        plt.ion()
        self.figure, axes = plt.subplots(3, 3, figsize=(17, 10))
        self.axes = axes.ravel()

        manager = self.figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title("UKCA timestep-halving training")

        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.figure.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def _on_close(self, _event):
        self.closed = True

    @staticmethod
    def _values(history, key):
        return np.asarray([row[key] for row in history], dtype=float)

    def update(self, seed, current_history, completed_histories):
        """Refresh the dashboard using the history available so far."""
        if self.closed or not current_history:
            return

        for axis in self.axes:
            axis.clear()

        epochs = self._values(current_history, "epoch")
        latest = current_history[-1]

        # 1. Training and validation objective.
        axis = self.axes[0]
        axis.plot(epochs,self._values(current_history, "train_weighted_ce"),label="Train",)
        axis.plot(epochs,self._values(current_history, "weighted_ce"),label="Validation",)
        axis.set_title("Weighted cross-entropy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()

        # 2. Average precision.
        axis = self.axes[1]
        axis.plot(epochs, self._values(current_history, "ap"), label="FCNN")
        if self.baseline_metrics is not None:
            axis.axhline(self.baseline_metrics["ap"],linestyle="--",label="Logistic baseline",)
            axis.legend()
        axis.set_title("Detection average precision")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("AP")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

        # 3. ROC-AUC.
        axis = self.axes[2]
        roc_auc = self._values(current_history, "roc_auc")
        axis.plot(epochs, roc_auc, label="FCNN")
        if self.baseline_metrics is not None:
            axis.axhline(self.baseline_metrics["roc_auc"],linestyle="--",label="Logistic baseline")
            axis.legend()
        axis.set_title("Detection ROC-AUC")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("ROC-AUC")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

        # 4. Concrete top-k operating point.
        axis = self.axes[3]
        axis.plot(epochs,self._values(current_history, "precision_at_k"),label=f"Precision@{latest['k']}")
        axis.plot(epochs,self._values(current_history, "recall_at_k"),label=f"Recall@{latest['k']}")
        if self.baseline_metrics is not None:
            axis.axhline(self.baseline_metrics["precision_at_k"],linestyle="--",label="Logistic precision",)
            axis.axhline(self.baseline_metrics["recall_at_k"],linestyle=":",label="Logistic recall",)
        axis.set_title(f"Top-{latest['k']} detection")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric value")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()

        # 5. Severity on boxes that genuinely require a halving.
        axis = self.axes[4]
        axis.plot(epochs,self._values(current_history, "exact_on_pos"),label="Exact on positives",)
        axis.plot(epochs,self._values(current_history, "mae_on_pos"),label="MAE on positives",)
        axis.set_title("Positive-box severity")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric value")
        axis.grid(alpha=0.25)
        axis.legend()

        # 6. True support versus number predicted in each class.
        axis = self.axes[5]
        class_indices = np.arange(self.n_classes)
        supports = np.asarray(
            [
                latest["per_class"][str(c)]["support"]
                if str(c) in latest["per_class"]
                else latest["per_class"][c]["support"]
                for c in class_indices
            ],
            dtype=float,
        )
        predicted = np.asarray(
            [
                latest["per_class"][str(c)]["n_predicted"]
                if str(c) in latest["per_class"]
                else latest["per_class"][c]["n_predicted"]
                for c in class_indices
            ],
            dtype=float,
        )
        width = 0.38
        axis.bar(class_indices - width / 2, supports, width, label="True support")
        axis.bar(class_indices + width / 2, predicted, width, label="Predicted")
        axis.set_title("Class counts")
        axis.set_xlabel("Halving class")
        axis.set_ylabel("Count")
        axis.set_xticks(class_indices)
        axis.set_yscale("symlog", linthresh=1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

        # 7. Latest row-normalised confusion matrix.
        axis = self.axes[6]
        confusion = np.asarray(latest["confusion"], dtype=float)
        row_totals = confusion.sum(axis=1, keepdims=True)
        row_normalised = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion),
            where=row_totals != 0,
        )
        axis.imshow(row_normalised, vmin=0, vmax=1)
        axis.set_title("Latest confusion matrix")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(class_indices)
        axis.set_yticks(class_indices)

        for true_class in class_indices:
            for predicted_class in class_indices:
                if row_totals[true_class, 0] == 0:
                    label = "—"
                else:
                    label = (
                        f"{int(confusion[true_class, predicted_class])}\n"
                        f"{100 * row_normalised[true_class, predicted_class]:.1f}%"
                    )
                axis.text(
                    predicted_class,
                    true_class,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        # 8. Latest per-class recall.
        axis = self.axes[7]
        recalls = []
        for class_index in class_indices:
            class_metrics = (latest["per_class"].get(str(class_index)) or latest["per_class"].get(class_index))
            value = class_metrics["recall"]
            recalls.append(np.nan if value is None else value)

        axis.bar(class_indices, recalls)
        axis.set_title("Latest per-class recall")
        axis.set_xlabel("Halving class")
        axis.set_ylabel("Recall")
        axis.set_xticks(class_indices)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)

        # 9. Final AP of completed seeds plus the current seed so far.
        axis = self.axes[8]
        seed_labels = []
        seed_aps = []

        if self.baseline_metrics is not None:
            seed_labels.append("Logistic")
            seed_aps.append(self.baseline_metrics["ap"])

        for history_index, history in enumerate(completed_histories):
            seed_labels.append(f"Seed {history_index}")
            seed_aps.append(history[-1]["ap"])

        seed_labels.append(f"Seed {seed} current")
        seed_aps.append(latest["ap"])

        axis.bar(seed_labels, seed_aps)
        axis.set_title("Seed comparison")
        axis.set_ylabel("Final/current AP")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)

        self.figure.suptitle(
            f"UKCA FCNN training — seed {seed}, epoch {latest['epoch']}",
            fontsize=14,
        )
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path):
        """Save the dashboard's current state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=200, bbox_inches="tight")

    def finish(self, block=True):
        """Keep the final dashboard visible until the user closes it."""
        if self.closed:
            return

        plt.ioff()
        plt.show(block=block)
