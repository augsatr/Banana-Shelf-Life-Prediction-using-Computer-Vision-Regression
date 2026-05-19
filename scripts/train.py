import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

from config import cfg
from data.synthetic import RealisticBananaGenerator
from data.dataset import create_dataloaders
from data.preprocessing import BananaPreprocessor
from data.augmentations import RandAugment
from features.extractor import AdvancedFeatureExtractor
from models.factory import ModelFactory
from models.ensemble import AdvancedMLEnsemble
from models.train import Trainer, evaluate
from utils.visualization import plot_training_history, plot_predictions, plot_error_analysis


def main():
    parser = argparse.ArgumentParser(description="Banana Shelf-Life Training")
    parser.add_argument("--model", choices=["cnn", "vit", "ensemble", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    n_samples = args.samples or cfg.get("data", "synthetic_samples", default=2000)
    models_dir = Path(cfg.get("paths", "models_dir", default="models/saved"))

    if args.data_dir:
        data_path = Path(args.data_dir)
        image_paths = sorted(list(data_path.glob("*.jpg")) + list(data_path.glob("*.png")))
        print(f"Loading {len(image_paths)} images from {args.data_dir}")
        shelf_life_labels, stage_labels = [], []
        for p in image_paths:
            name = p.stem
            parts = name.split("_")
            try:
                sl = float([x for x in parts if x.startswith("s")][0][1:])
                st = int([x for x in parts if x.startswith("st")][0][2:])
            except (IndexError, ValueError):
                continue
            shelf_life_labels.append(sl)
            stage_labels.append(st)
    else:
        print(f"Generating {n_samples} synthetic banana images...")
        gen = RealisticBananaGenerator()
        processed_dir = Path(cfg.get("paths", "processed_data", default="data/processed"))
        image_paths, shelf_life_labels, stage_labels = gen.generate_batch(n_samples, processed_dir)
        print(f"Generated {len(image_paths)} images")

    preprocessor = BananaPreprocessor()
    augment_fn = RandAugment(
        magnitude=cfg.get("augmentation", "randaug_magnitude", default=12),
        num_ops=cfg.get("augmentation", "randaug_num_ops", default=3),
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        image_paths, shelf_life_labels, stage_labels,
        preprocessor=preprocessor, augment_fn=augment_fn,
    )

    models_to_train = []
    if args.model in ("cnn", "all"):
        models_to_train.append("cnn")
    if args.model in ("vit", "all"):
        models_to_train.append("vit")
    if args.model in ("ensemble", "all"):
        models_to_train.append("ensemble")

    for mtype in models_to_train:
        print(f"\n{'='*60}")
        print(f"  Training {mtype.upper()} model")
        print(f"{'='*60}")

        if mtype == "ensemble":
            from tqdm import tqdm
            extractor = AdvancedFeatureExtractor()
            features = []
            for path in tqdm(image_paths, desc="Extracting features"):
                with open(str(path), "rb") as f:
                    raw = bytearray(f.read())
                arr = np.asarray(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed = preprocessor(img)
                feat = extractor.extract_all(processed)
                features.append(feat)

            X = np.array(features)
            y = np.array(shelf_life_labels)
            print(f"Feature matrix: {X.shape}")

            model = AdvancedMLEnsemble(use_optimization=True)
            model.fit(X, y)
            model.save("advanced_ensemble")

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            y_pred = model.predict(X_test)
            print(f"Ensemble Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
            print(f"Ensemble Test R²: {r2_score(y_test, y_pred):.4f}")

        else:
            if mtype == "vit":
                model_kwargs = {
                    "model_name": cfg.get("models", "vit", "model_name",
                                          default="google/vit-base-patch16-224-in21k"),
                    "dropout": cfg.get("models", "vit", "dropout", default=0.2),
                    "freeze_backbone": cfg.get("models", "vit", "freeze_backbone", default=False),
                    "embedding_dim": cfg.get("models", "vit", "embedding_dim", default=128),
                }
            else:
                model_kwargs = {
                    "backbone": cfg.get("models", "cnn", "backbone", default="efficientnet_v2_s"),
                    "pretrained": True,
                    "dropout": cfg.get("models", "cnn", "dropout", default=0.3),
                    "embedding_dim": cfg.get("models", "cnn", "embedding_dim", default=128),
                }
            model = ModelFactory.create(mtype, model_kwargs)

            epochs = args.epochs or cfg.get("training", "epochs", default=30)
            trainer = Trainer(model, train_loader, val_loader, config=cfg.get("training", {}))
            history = trainer.fit(epochs=epochs)

            results = evaluate(model, test_loader)
            print(f"\n{mtype.upper()} Test Results:")
            for k, v in results.items():
                print(f"  {k}: {v:.4f}")

            plot_training_history(history, str(models_dir / f"{mtype}_training_history.png"))

            all_preds, all_targets = [], []
            for batch in test_loader:
                with torch.no_grad():
                    pred, _, _, _ = model(batch["image"].to(next(model.parameters()).device))
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch["shelf_life"].numpy())

            plot_predictions(np.array(all_targets), np.array(all_preds),
                             save_path=str(models_dir / f"{mtype}_predictions.png"))
            plot_error_analysis(np.array(all_targets), np.array(all_preds),
                                save_path=str(models_dir / f"{mtype}_error_analysis.png"))

    print("\nTraining complete!")


if __name__ == "__main__":
    import torch
    main()
