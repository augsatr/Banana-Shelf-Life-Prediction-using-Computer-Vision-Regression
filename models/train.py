import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Callable, List
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from config import cfg


class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, current: float) -> bool:
        if self.mode == "min":
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


class SWA:
    def __init__(self, start_epoch: int = 40):
        self.start_epoch = start_epoch
        self.swa_model = None
        self.n_averaged = 0

    def update(self, model: nn.Module, epoch: int):
        if epoch < self.start_epoch:
            return
        if self.swa_model is None:
            self.swa_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            for key in self.swa_model:
                self.swa_model[key] = (
                    self.swa_model[key] * self.n_averaged + model.state_dict()[key].detach()
                ) / (self.n_averaged + 1)
        self.n_averaged += 1

    def apply(self, model: nn.Module):
        if self.swa_model is not None:
            model.load_state_dict(self.swa_model)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict] = None,
        device: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        lr = self.cfg.get("learning_rate", 3e-4)
        wd = self.cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        warmup = self.cfg.get("warmup_epochs", 3)
        n_epochs = self.cfg.get("epochs", 60)
        main_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs - warmup, eta_min=lr * 0.01
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup
                ),
                main_lr,
            ],
            milestones=[warmup],
        )

        ls = self.cfg.get("label_smoothing", 0.05)
        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        self.grad_clip = self.cfg.get("grad_clip_norm", 1.0)
        self.use_amp = self.cfg.get("mixed_precision", True) and device != "cpu"
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        self.early_stopping = EarlyStopping(patience=self.cfg.get("early_stop_patience", 12))
        self.swa = SWA(start_epoch=self.cfg.get("swa_start", 40))

        self.history = {
            "train_loss": [], "val_loss": [], "val_mae": [], "val_r2": [],
            "val_rmse": [], "learning_rates": [],
        }
        self.best_val_loss = float("inf")
        self.models_dir = Path(cfg.get("paths", "models_dir", default="models/saved"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            shelf_life = batch["shelf_life"].to(self.device, non_blocking=True)
            stage = batch.get("stage")
            stage = stage.to(self.device, non_blocking=True) if stage is not None else None

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device):
                    pred_shelf, pred_stage, log_var, _ = self.model(images)
                    loss = self._compute_loss(pred_shelf, pred_stage, log_var, shelf_life, stage)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_shelf, pred_stage, log_var, _ = self.model(images)
                loss = self._compute_loss(pred_shelf, pred_stage, log_var, shelf_life, stage)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / n_batches

    def _compute_loss(self, pred_shelf, pred_stage, log_var, shelf_life, stage):
        reg_loss = self.reg_criterion(pred_shelf, shelf_life)
        uncertainty_loss = (torch.exp(-log_var) * reg_loss + 0.5 * log_var).mean()
        if stage is not None:
            cls_loss = self.cls_criterion(pred_stage, stage)
            return 0.5 * reg_loss + 0.5 * uncertainty_loss + 0.3 * cls_loss
        return 0.5 * reg_loss + 0.5 * uncertainty_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            shelf_life = batch["shelf_life"].to(self.device, non_blocking=True)

            pred_shelf, _, _, _ = self.model(images)
            loss = self.reg_criterion(pred_shelf, shelf_life)
            total_loss += loss.item()
            all_preds.extend(pred_shelf.cpu().numpy())
            all_targets.extend(shelf_life.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        return {
            "loss": total_loss / len(self.val_loader),
            "mae": mean_absolute_error(all_targets, all_preds),
            "rmse": np.sqrt(mean_squared_error(all_targets, all_preds)),
            "r2": r2_score(all_targets, all_preds),
        }

    def fit(self, epochs: int = None, callbacks: Optional[List[Callable]] = None) -> Dict:
        if epochs is None:
            epochs = self.cfg.get("epochs", 60)
        callbacks = callbacks or []

        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(self)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            self.swa.update(self.model, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_r2"].append(val_metrics["r2"])
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch:2d}/{epochs} | Train: {train_loss:.4f} | "
                f"Val: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | "
                f"R²: {val_metrics['r2']:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            improved = self.early_stopping(val_metrics[self.cfg.get("save_best_metric", "val_mae")])
            if improved:
                self._save_checkpoint("best_model", epoch, val_metrics)

            self._save_checkpoint("latest", epoch, val_metrics)

            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(self, epoch, val_metrics)

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        self.swa.apply(self.model)
        self._save_checkpoint("swa_model", epoch, val_metrics)
        return self.history

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict):
        path = self.models_dir / f"{name}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }, path)


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: str = "cpu") -> Dict:
    model.eval()
    model.to(device)
    all_preds, all_targets, all_stages, all_stage_preds = [], [], [], []

    for batch in test_loader:
        images = batch["image"].to(device)
        shelf_life = batch["shelf_life"].to(device)
        stage = batch.get("stage")

        pred_shelf, pred_stage, _, _ = model(images)
        all_preds.extend(pred_shelf.cpu().numpy())
        all_targets.extend(shelf_life.cpu().numpy())

        if stage is not None:
            all_stages.extend(stage.numpy())
            all_stage_preds.extend(torch.argmax(pred_stage, dim=1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    results = {
        "mae": mean_absolute_error(all_targets, all_preds),
        "rmse": np.sqrt(mean_squared_error(all_targets, all_preds)),
        "r2": r2_score(all_targets, all_preds),
    }

    if all_stages:
        from sklearn.metrics import accuracy_score
        results["stage_accuracy"] = accuracy_score(all_stages, all_stage_preds)

    errors = np.abs(all_targets - all_preds)
    results["mae_std"] = float(errors.std())
    results["p25"] = float(np.percentile(errors, 25))
    results["p75"] = float(np.percentile(errors, 75))

    return results
