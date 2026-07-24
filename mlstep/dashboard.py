"""Live plots for the timestep-halving training loop."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def is_interactive_backend(backend):
    """Return whether a Matplotlib backend can display a live window."""
    backend = backend.lower()
    return any(
        name in backend
        for name in (
            "gtk",
            "macosx",
            "nbagg",
            "notebook",
            "qt",
            "tk",
            "webagg",
            "widget",
            "wx",
        )
    )


class LiveTrainingDashboard:
    """Show the main validation diagnostics after every epoch."""

    def __init__(self, n_classes, baseline_metrics=None):
        self.n_classes = n_classes
        self.baseline = baseline_metrics
        self.closed = False
        self.backend = plt.get_backend()
        self.interactive = is_interactive_backend(self.backend)

        if self.interactive:
            plt.ion()
        self.figure, axes = plt.subplots(3, 3, figsize=(17, 10))
        self.axes = axes.ravel()
        manager = self.figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title("UKCA timestep-halving training")
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        if self.interactive:
            plt.show(block=False)
            plt.pause(0.001)
            print(f"Live dashboard opened with Matplotlib backend {self.backend}.")
        else:
            print(
                f"Matplotlib backend {self.backend} is non-interactive; "
                "the dashboard will be saved but cannot pop up."
            )

    def _on_close(self, _event):
        self.closed = True

    @staticmethod
    def values(history, key):
        """Extract a metric history, converting missing values to NaN."""
        return np.array(
            [np.nan if record.get(key) is None else record[key] for record in history],
            dtype=float,
        )

    def update(self, seed, history, completed_histories):
        """Redraw all panels using the latest epoch."""
        if self.closed or not history:
            return

        for axis in self.axes:
            axis.clear()

        epochs = self.values(history, "epoch")
        latest = history[-1]
        confusion = np.asarray(latest["confusion"])
        classes = np.arange(self.n_classes)

        axis = self.axes[0]
        axis.plot(
            epochs,
            self.values(history, "train_weighted_ce"),
            label="Train",
        )
        axis.plot(
            epochs,
            self.values(history, "weighted_ce"),
            label="Validation",
        )
        axis.set_title("Weighted cross-entropy")
        axis.set_yscale("log")
        axis.legend()

        axis = self.axes[1]
        axis.plot(epochs, self.values(history, "ap"), label="FCNN")
        if self.baseline:
            axis.axhline(self.baseline["ap"], linestyle="--", label="Baseline")
        axis.set_title("Detection average precision")
        axis.set_ylim(0, 1)
        axis.legend()

        axis = self.axes[2]
        axis.plot(epochs, self.values(history, "roc_auc"), label="FCNN")
        if self.baseline and self.baseline.get("roc_auc") is not None:
            axis.axhline(
                self.baseline["roc_auc"],
                linestyle="--",
                label="Baseline",
            )
        axis.set_title("Detection ROC-AUC")
        axis.set_ylim(0, 1)
        axis.legend()

        axis = self.axes[3]
        axis.plot(
            epochs,
            self.values(history, "precision_at_top"),
            label="Precision",
        )
        axis.plot(
            epochs,
            self.values(history, "recall_at_top"),
            label="Recall",
        )
        if self.baseline:
            axis.axhline(self.baseline["precision_at_top"], linestyle="--")
            axis.axhline(self.baseline["recall_at_top"], linestyle=":")
        axis.set_title(f"Top {100 * latest['top_fraction']:.2g}% detection")
        axis.set_ylim(0, 1)
        axis.legend()

        axis = self.axes[4]
        axis.plot(
            epochs,
            self.values(history, "exact_on_pos"),
            label="Exact",
        )
        axis.plot(
            epochs,
            self.values(history, "mae_on_pos"),
            label="MAE",
        )
        axis.plot(
            epochs,
            self.values(history, "underprediction_rate"),
            label="Under",
        )
        axis.plot(
            epochs,
            self.values(history, "overprediction_rate"),
            label="Over",
        )
        axis.set_title("Severity on positive boxes")
        axis.legend()

        axis = self.axes[5]
        support = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        width = 0.38
        axis.bar(classes - width / 2, support, width, label="True")
        axis.bar(classes + width / 2, predicted, width, label="Predicted")
        axis.set_yscale("symlog", linthresh=1)
        axis.set_title("Class counts")
        axis.set_xticks(classes)
        axis.legend()

        axis = self.axes[6]
        row_totals = confusion.sum(axis=1, keepdims=True)
        normalised = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion, dtype=float),
            where=row_totals != 0,
        )
        axis.imshow(normalised, vmin=0, vmax=1)
        axis.set_title("Row-normalised confusion")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(classes)
        axis.set_yticks(classes)
        for true_class in classes:
            for predicted_class in classes:
                axis.text(
                    predicted_class,
                    true_class,
                    f"{confusion[true_class, predicted_class]}\n"
                    f"{100 * normalised[true_class, predicted_class]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        axis = self.axes[7]
        recall = np.divide(
            np.diag(confusion),
            support,
            out=np.full(self.n_classes, np.nan),
            where=support != 0,
        )
        axis.bar(classes, recall)
        axis.set_title("Per-class recall")
        axis.set_ylim(0, 1)
        axis.set_xticks(classes)

        axis = self.axes[8]
        labels = []
        scores = []
        if self.baseline:
            labels.append("Baseline")
            scores.append(self.baseline["ap"])
        for completed_seed, completed_history in completed_histories:
            labels.append(f"Seed {completed_seed}")
            scores.append(completed_history[-1]["ap"])
        labels.append(f"Seed {seed} current")
        scores.append(latest["ap"])
        axis.bar(labels, scores)
        axis.set_title("Final/current AP by seed")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=30)

        for index, axis in enumerate(self.axes):
            axis.grid(alpha=0.2)
            if index < 5:
                axis.set_xlabel("Epoch")

        self.figure.suptitle(f"UKCA training — seed {seed}, epoch {latest['epoch']}")
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        # Draw synchronously so this epoch is visible before the next starts.
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        if self.interactive:
            plt.pause(0.001)

    def save(self, path):
        """Save the final dashboard."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=200, bbox_inches="tight")

    def finish(self):
        """Keep the completed dashboard open until it is closed."""
        if not self.closed and self.interactive:
            plt.ioff()
            plt.show(block=True)
