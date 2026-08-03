"""Optional live plots for FCNN and XGBoost training."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.registry import BackendFilter, backend_registry

_CONFUSION_TEXT_THRESHOLD = 0.5


class _DashboardFigure:
    """Shared Matplotlib window and plotting helpers."""

    def __init__(self, window_title: str) -> None:
        self.closed = False
        backend = plt.get_backend()
        self.interactive = backend.lower() in backend_registry.list_builtin(BackendFilter.INTERACTIVE)

        if self.interactive:
            plt.ion()

        self.figure, axes = plt.subplots(2, 4, figsize=(18, 9))
        self.axes = axes.ravel()

        manager = self.figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title(window_title)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        if self.interactive:
            plt.show(block=False)
            plt.pause(0.001)
            print(f"Live dashboard opened with Matplotlib backend {backend}.")
        else:
            print(f"Matplotlib backend {backend} is headless; the final dashboard will be saved.")

    def _on_close(self, _event) -> None:
        self.closed = True

    @staticmethod
    def _plot(axis, history: list[dict], key: str, label: str, **kwargs) -> None:
        axis.plot(
            [record["epoch"] for record in history],
            [record[key] for record in history],
            label=label,
            **kwargs,
        )

    @staticmethod
    def _empty_panel(axis, message: str) -> None:
        axis.text(0.5, 0.5, message, ha="center", va="center", color="0.5", transform=axis.transAxes)

    @staticmethod
    def _plot_class_counts(axis, confusion: np.ndarray, labels: list[str], title: str) -> None:
        true_counts = confusion.sum(axis=1)
        predicted_counts = confusion.sum(axis=0)
        positions = np.arange(len(labels))
        width = 0.38

        axis.bar(positions - width / 2, true_counts, width, label="True")
        axis.bar(positions + width / 2, predicted_counts, width, label="Predicted")
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Rows")
        axis.set_yscale("symlog", linthresh=1)
        axis.legend()
        axis.grid(axis="y", alpha=0.2)

    @staticmethod
    def _plot_confusion(axis, confusion: np.ndarray, labels: list[str], title: str) -> None:
        row_totals = confusion.sum(axis=1, keepdims=True)
        normalised = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion, dtype=float),
            where=row_totals != 0,
        )
        axis.imshow(normalised, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_yticks(np.arange(len(labels)), labels)

        for true_class in range(len(labels)):
            for predicted_class in range(len(labels)):
                percentage = normalised[true_class, predicted_class]
                axis.text(
                    predicted_class,
                    true_class,
                    f"{confusion[true_class, predicted_class]:,}\n{percentage:.1%}",
                    ha="center",
                    va="center",
                    color="white" if percentage > _CONFUSION_TEXT_THRESHOLD else "black",
                    fontsize=8,
                )

    def _plot_binary_metrics(self, counts_axis, confusion_axis, metrics: dict | None) -> None:
        if metrics is None:
            self._empty_panel(counts_axis, "Available after threshold selection")
            self._empty_panel(confusion_axis, "Available after threshold selection")
            counts_axis.set_title("Validation: binary class counts")
            confusion_axis.set_title("Validation: binary confusion")
            return

        confusion = np.array(
            [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ]
        )
        labels = ["No halving", "Halving"]
        self._plot_class_counts(counts_axis, confusion, labels, "Validation: binary class counts")
        self._plot_confusion(
            confusion_axis,
            confusion,
            labels,
            f"Validation: binary confusion (threshold {metrics['threshold']:.3g})",
        )

    def _draw(self) -> None:
        if self.interactive and not self.closed:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

    def save(self, path: Path) -> None:
        """Save the latest dashboard state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=160, bbox_inches="tight")

    def close(self) -> None:
        """Close the dashboard without blocking the training command."""
        if not self.closed:
            self.closed = True
            plt.close(self.figure)


class LiveTrainingDashboard(_DashboardFigure):
    """Display the metrics reported by the two FCNN training stages."""

    def __init__(self, window_title: str = "UKCA FCNN training") -> None:
        self.detector_history: list[dict] = []
        self.exact_halvings_history: list[dict] = []
        self.final_metrics: dict | None = None
        super().__init__(window_title)
        self._redraw()

    def update_detector(self, history: list[dict]) -> None:
        """Update the hard-box detector panels."""
        if self.closed:
            return
        self.detector_history = history
        if self.interactive:
            self._redraw()

    def update_exact_halvings(self, history: list[dict]) -> None:
        """Update the exact-halving panels."""
        if self.closed:
            return
        self.exact_halvings_history = history
        if self.interactive:
            self._redraw()

    def update_final_metrics(self, metrics: dict) -> None:
        """Add the threshold-dependent diagnostics from final validation."""
        if self.closed:
            return
        self.final_metrics = metrics
        self._redraw()

    def _plot_detector_history(self, loss_axis, ap_axis) -> None:
        history = self.detector_history
        if history:
            self._plot(loss_axis, history, "training_loss", "Training loss")
            self._plot(ap_axis, history, "validation_ap", "Validation AP", color="tab:green")
            best = max(history, key=lambda record: record["validation_ap"])
            ap_axis.scatter(best["epoch"], best["validation_ap"], color="black", zorder=3, label="Best")
            loss_axis.legend()
            ap_axis.legend()
        else:
            self._empty_panel(loss_axis, "Waiting for detector training")
            self._empty_panel(ap_axis, "Waiting for detector validation")

        loss_axis.set_title("Hard-box detector loss")
        ap_axis.set_title("Hard-box detector average precision")
        ap_axis.set_ylim(0, 1)
        for axis in (loss_axis, ap_axis):
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)

    def _plot_exact_history(self, loss_axis, metrics_axis) -> None:
        history = self.exact_halvings_history
        if history:
            self._plot(loss_axis, history, "training_loss", "Training loss")
            self._plot(metrics_axis, history, "exact", "Exact", color="tab:green")
            self._plot(
                metrics_axis,
                history,
                "mean_supported_class_recall",
                "Mean class recall",
                color="tab:blue",
            )
            loss_axis.legend()
            metrics_axis.legend()
            metrics_axis.set_title(f"Exact-halving validation (latest MAE {history[-1]['mae']:.3f})")
        else:
            self._empty_panel(loss_axis, "Available during multiclass training")
            self._empty_panel(metrics_axis, "Available during multiclass training")
            metrics_axis.set_title("Exact-halving validation")

        loss_axis.set_title("Exact-halving training loss")
        metrics_axis.set_ylim(0, 1)
        for axis in (loss_axis, metrics_axis):
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)

    def _redraw(self) -> None:
        for axis in self.axes:
            axis.clear()

        (
            detector_loss,
            detector_ap,
            binary_counts,
            binary_confusion,
            exact_loss,
            exact_metrics,
            class_counts,
            class_confusion,
        ) = self.axes

        self._plot_detector_history(detector_loss, detector_ap)
        self._plot_binary_metrics(binary_counts, binary_confusion, self.final_metrics)
        self._plot_exact_history(exact_loss, exact_metrics)

        severity = self.final_metrics.get("severity") if self.final_metrics else None
        if severity:
            multiclass = np.asarray(severity["confusion"])
            multiclass_labels = [str(label) for label in range(len(multiclass))]
            self._plot_class_counts(class_counts, multiclass, multiclass_labels, "Validation: halving class counts")
            self._plot_confusion(class_confusion, multiclass, multiclass_labels, "Validation: end-to-end confusion")
        else:
            message = "Available after multiclass evaluation" if self.exact_halvings_history else "Multiclass runs only"
            self._empty_panel(class_counts, message)
            self._empty_panel(class_confusion, message)
            class_counts.set_title("Validation: halving class counts")
            class_confusion.set_title("Validation: end-to-end confusion")

        if self.final_metrics:
            stage = "complete"
        elif self.exact_halvings_history:
            stage = "exact-halving model"
        else:
            stage = "hard-box detector"
        self.figure.suptitle(f"UKCA FCNN training — {stage}")
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self._draw()


class XGBoostTrainingDashboard(_DashboardFigure):
    """Display XGBoost training progress and final validation diagnostics."""

    def __init__(self, task: str, window_title: str = "UKCA XGBoost training") -> None:
        self.task = task
        self.training_history: list[dict] = []
        self.metric_name = "validation metric"
        self.final_metrics: dict | None = None
        self.importance: dict[str, float] = {}
        self.group_importance: dict[str, float] = {}
        self.best_iteration: int | None = None
        self.timing: dict | None = None
        super().__init__(window_title)
        self._redraw()

    def update_training(self, round_number: int, value: float, metric_name: str) -> None:
        """Store one validation metric and occasionally refresh a live window."""
        if self.closed:
            return
        self.metric_name = metric_name
        self.training_history.append({"round": round_number, "validation_metric": value})
        if self.interactive and (round_number == 1 or round_number % 25 == 0):
            self._redraw()

    def set_training_history(self, history: list[dict], metric_name: str) -> None:
        """Replace callback history with the complete XGBoost evaluation log."""
        if not self.closed:
            self.training_history = history
            self.metric_name = metric_name

    def update_final_metrics(
        self,
        metrics: dict,
        importance: dict[str, float],
        group_importance: dict[str, float],
        best_iteration: int,
        timing: dict,
    ) -> None:
        """Add threshold diagnostics, feature importance and timing."""
        if self.closed:
            return
        self.final_metrics = metrics
        self.importance = importance
        self.group_importance = group_importance
        self.best_iteration = best_iteration
        self.timing = timing
        self._redraw()

    @staticmethod
    def _plot_importance(axis, values: dict[str, float], title: str) -> None:
        items = [(name, value) for name, value in values.items() if value > 0][:10]
        if not items:
            XGBoostTrainingDashboard._empty_panel(axis, "No split gain was recorded")
            axis.set_title(title)
            return
        labels, scores = zip(*reversed(items), strict=True)
        axis.barh(labels, scores, color="tab:blue")
        axis.set_title(title)
        axis.set_xlabel("Normalised total gain")
        axis.grid(axis="x", alpha=0.2)

    def _plot_recall_sweep(self, axis) -> None:
        sweep = self.final_metrics.get("validation_recall_sweep", {}) if self.final_metrics else {}
        points = {(point["recall"], point["false_positive"]) for point in sweep.values()}
        if not points:
            self._empty_panel(axis, "Available after threshold selection")
            axis.set_title("False positives across recall targets")
            return

        ordered = sorted(points)
        axis.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            marker="o",
            color="tab:red",
        )
        axis.set_title("False positives across recall targets")
        axis.set_xlabel("Recall")
        axis.set_ylabel("False-positive rows")
        axis.set_xlim(0, 1.01)
        axis.set_yscale("symlog", linthresh=1)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)

    def _plot_summary(self, axis) -> None:
        axis.axis("off")
        if not self.final_metrics:
            self._empty_panel(axis, "Available after validation")
            return
        metrics = self.final_metrics
        total = (
            metrics["true_positive"] + metrics["false_positive"] + metrics["false_negative"] + metrics["true_negative"]
        )
        candidates = metrics["true_positive"] + metrics["false_positive"]
        lines = [
            f"AP: {metrics['ap']:.4f}",
            f"Threshold: {metrics['threshold']:.4g}",
            f"Recall: {metrics['recall']:.2%}",
            f"Precision: {metrics['precision']:.2%}",
            f"TP / FP / FN: {metrics['true_positive']:,} / {metrics['false_positive']:,} / "
            f"{metrics['false_negative']:,}",
            f"Candidates: {candidates:,} ({candidates / total:.2%})",
        ]
        axis.text(0.03, 0.97, "\n".join(lines), va="top", transform=axis.transAxes)
        axis.set_title("Selected operating point")

    def _plot_timing(self, axis) -> None:
        axis.axis("off")
        if not self.timing:
            self._empty_panel(axis, "Available after benchmarking")
            return
        inference = self.timing["inference"]
        lines = [
            f"Data loading: {self.timing['data_loading_seconds']:.2f} s",
            f"Matrix construction: {self.timing['matrix_construction_seconds']:.2f} s",
            f"Training: {self.timing['training_seconds']:.2f} s",
            f"Inference: {inference['median_seconds']:.4f} s",
            f"Throughput: {inference['rows_per_second']:,.0f} rows/s",
        ]
        axis.text(0.03, 0.97, "\n".join(lines), va="top", transform=axis.transAxes)
        axis.set_title("Runtime")

    def _redraw(self) -> None:
        for axis in self.axes:
            axis.clear()

        (
            training_metric,
            recall_sweep,
            binary_counts,
            binary_confusion,
            feature_importance,
            group_importance,
            final_counts,
            final_confusion,
        ) = self.axes

        if self.training_history:
            rounds = [record["round"] for record in self.training_history]
            values = [record["validation_metric"] for record in self.training_history]
            training_metric.plot(rounds, values, color="tab:green", label=self.metric_name)
            if self.best_iteration is not None and self.best_iteration < len(values):
                training_metric.scatter(
                    self.best_iteration + 1,
                    values[self.best_iteration],
                    color="black",
                    zorder=3,
                    label="Selected",
                )
            training_metric.legend()
        else:
            self._empty_panel(training_metric, "Waiting for boosting")
        training_metric.set_title(f"Validation {self.metric_name}")
        training_metric.set_xlabel("Boosting round")
        training_metric.set_ylim(0, 1)
        training_metric.grid(alpha=0.2)

        self._plot_recall_sweep(recall_sweep)

        self._plot_binary_metrics(binary_counts, binary_confusion, self.final_metrics)

        self._plot_importance(feature_importance, self.importance, "Top feature importance")
        self._plot_importance(group_importance, self.group_importance, "Feature-group importance")

        severity = self.final_metrics.get("severity") if self.final_metrics else None
        if severity:
            multiclass = np.asarray(severity["confusion"])
            labels = [str(label) for label in range(len(multiclass))]
            self._plot_class_counts(final_counts, multiclass, labels, "Validation: halving class counts")
            self._plot_confusion(final_confusion, multiclass, labels, "Validation: end-to-end confusion")
        else:
            self._plot_summary(final_counts)
            self._plot_timing(final_confusion)

        stage = "complete" if self.final_metrics else "training"
        title = f"UKCA XGBoost {self.task} — {stage}"
        if self.final_metrics:
            title += f" — AP {self.final_metrics['ap']:.4f}, recall {self.final_metrics['recall']:.2%}"
        self.figure.suptitle(title)
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self._draw()
