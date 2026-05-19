import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    axes[0, 0].plot(epochs, history.get("train_loss", []), "b-", label="Train Loss")
    axes[0, 0].plot(epochs, history.get("val_loss", []), "r-", label="Val Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history.get("val_mae", []), "orange", marker=".", label="MAE")
    axes[0, 1].set_title("Validation MAE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE (days)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history.get("val_r2", []), "g-", marker=".", label="R²")
    axes[0, 2].set_title("Validation R² Score")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("R²")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history.get("val_rmse", []), "purple", marker=".", label="RMSE")
    axes[1, 0].set_title("Validation RMSE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("RMSE (days)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history.get("learning_rates", []), "brown", marker=".")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                      y_std: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="k", linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
    axes[0].set_xlabel("Actual Shelf Life (days)")
    axes[0].set_ylabel("Predicted Shelf Life (days)")
    axes[0].set_title("Predictions vs Actual")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    errors = y_pred - y_true
    axes[1].scatter(y_true, errors, alpha=0.6, s=30, edgecolors="k", linewidth=0.5)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Actual Shelf Life (days)")
    axes[1].set_ylabel("Prediction Error (days)")
    axes[1].set_title("Residual Plot")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(errors, bins=30, edgecolor="black", alpha=0.7)
    axes[2].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[2].set_xlabel("Error (days)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Error Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    errors = np.abs(y_true - y_pred)
    bins = np.linspace(0, 10, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(y_true, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    bin_mae = [errors[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else 0
               for i in range(len(bins) - 1)]

    axes[0].bar(bin_centers, bin_mae, width=0.8, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("True Shelf Life Range (days)")
    axes[0].set_ylabel("Mean Absolute Error (days)")
    axes[0].set_title("Error by Shelf Life Range")
    axes[0].grid(True, alpha=0.3)

    cum_errors = np.sort(errors)
    cum_freq = np.arange(1, len(cum_errors) + 1) / len(cum_errors)
    axes[1].plot(cum_errors, cum_freq, "b-", lw=2)
    axes[1].axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95%")
    axes[1].axhline(y=0.9, color="orange", linestyle="--", alpha=0.7, label="90%")
    axes[1].set_xlabel("Absolute Error (days)")
    axes[1].set_ylabel("Cumulative Frequency")
    axes[1].set_title("Error Cumulative Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names: List[str], save_path: Optional[str] = None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return

    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importances[idx], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def overlay_prediction(image: np.ndarray, shelf_life: float, stage_name: str,
                        confidence: Optional[str] = None) -> np.ndarray:
    overlay = image.copy()
    h, w = overlay.shape[:2]

    if shelf_life > 5:
        color = (50, 200, 50)
    elif shelf_life > 2:
        color = (50, 200, 200)
    else:
        color = (50, 50, 200)

    cv2.rectangle(overlay, (5, 5), (w - 5, h - 5), color, 3)
    bg_h = 90 if confidence else 70
    cv2.rectangle(overlay, (5, h - bg_h - 5), (min(420, w - 5), h - 5), (0, 0, 0), -1)
    cv2.putText(overlay, f"Shelf Life: {shelf_life:.1f} days", (15, h - bg_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(overlay, f"Stage: {stage_name}", (15, h - bg_h + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if confidence:
        cv2.putText(overlay, f"95% CI: {confidence}", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return overlay


def create_dashboard(metrics: Dict, save_path: str = "dashboard.png"):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.text(0.5, 0.5, "BANANA SHELF-LIFE MODEL DASHBOARD",
             ha="center", va="center", fontsize=24, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    info_items = [
        f"MAE: {metrics.get('mae', 'N/A'):.4f} days",
        f"RMSE: {metrics.get('rmse', 'N/A'):.4f} days",
        f"R²: {metrics.get('r2', 'N/A'):.4f}",
        f"Median AE: {metrics.get('median_ae', 'N/A'):.4f}",
    ]
    for i, item in enumerate(info_items):
        ax2.text(0.1, 0.85 - i * 0.15, item, fontsize=12, transform=ax2.transAxes)

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    acc_items = [f"Within 0.5d: {metrics.get('accuracy_within_0.5d', 0)*100:.1f}%",
                  f"Within 1d: {metrics.get('accuracy_within_1d', 0)*100:.1f}%",
                  f"Within 2d: {metrics.get('accuracy_within_2d', 0)*100:.1f}%"]
    for i, item in enumerate(acc_items):
        ax3.text(0.1, 0.85 - i * 0.15, item, fontsize=12, transform=ax3.transAxes)

    ax4 = fig.add_subplot(gs[1, :2])
    ax4.text(0.5, 0.5, "Predicted vs Actual", ha="center",
             fontsize=14, fontweight="bold", transform=ax4.transAxes)
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.text(0.5, 0.5, "Residual Distribution", ha="center",
             fontsize=14, fontweight="bold", transform=ax5.transAxes)
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[2, :])
    ax6.text(0.5, 0.5, "Error by Shelf Life Range", ha="center",
             fontsize=14, fontweight="bold", transform=ax6.transAxes)
    ax6.axis("off")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
