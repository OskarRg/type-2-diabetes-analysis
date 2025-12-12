import logging
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.metrics import auc, confusion_matrix, roc_curve

logger: logging.Logger = logging.getLogger(__name__)


class Visualizer:
    """Handles data visualization and model performance plots."""

    @staticmethod
    def plot_confusion_matrix(
        model: ClassifierMixin,
        X_test: np.ndarray,
        y_test: np.ndarray,
        title: str = "Confusion Matrix",
    ) -> None:
        """
        Plot a confusion matrix for a given model.

        :param model: Trained classifier.
        :param X_test: Test feature matrix.
        :param y_test: True labels.
        """
        logger.info("Plotting confusion matrix...")

        y_pred: np.ndarray = model.predict(X_test)
        cm: np.ndarray = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_confusion_matrix(
        model: ClassifierMixin,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Path,
        title: str = "Confusion Matrix",
        normalize: bool = True,
        threshold: float = 0.3,
    ) -> None:
        """
        Saves confusion matrix. If normalize is `True` the result is a Recall of the row.

        :param model: Model used to get predictions.
        :param X_test: Input data.
        :param y_test: True output.
        :param save_path: Path to save confusion matrix under.
        :param title: Title of the plot.
        :param normalize: If the data should be normalized.
        :param threshold: Threshold of illness classification.
        """
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
            else:
                y_pred = model.predict(X_test)

            if normalize:
                cm = confusion_matrix(y_test, y_pred, normalize="true")
                fmt = ".1%"
            else:
                cm = confusion_matrix(y_test, y_pred)
                fmt = "d"

            plt.figure(figsize=(6, 5))

            sns.heatmap(
                cm, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=1 if normalize else None
            )

            plt.title(f"{title}\n(Threshold: {threshold})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual (True)")
            plt.tight_layout()

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}", exc_info=True)
        finally:
            plt.close()

    @staticmethod
    def save_roc_curve(
        model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray, save_path: Path
    ) -> None:
        """
        Generates and saves a ROC from calculated AUC.

        :param model: Model used to get predictions.
        :param X_test: Input data.
        :param y_test: True output.
        :param save_path: Path to save the ROC.
        """
        try:
            if hasattr(model, "predict_proba"):
                y_probs: np.ndarray = model.predict_proba(X_test)[:, 1]
            else:
                logger.warning(
                    f"Model {type(model).__name__} does not support predict_proba. Skipping ROC."
                )
                return

            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc: float = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate (Recall)")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {e}", exc_info=True)
        finally:
            plt.close()
