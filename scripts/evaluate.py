import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import cv2
import json
import warnings
warnings.filterwarnings("ignore")

from config import cfg
from data.synthetic import RealisticBananaGenerator
from data.preprocessing import BananaPreprocessor
from data.dataset import create_dataloaders
from features.extractor import AdvancedFeatureExtractor
from models.ensemble import AdvancedMLEnsemble
from models.factory import ModelFactory
from models.train import evaluate
from utils.metrics import comprehensive_metrics
from utils.visualization import plot_predictions, plot_error_analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate banana shelf-life models")
    parser.add_argument("--model", choices=["ensemble", "cnn", "vit"], default="ensemble")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--save", type=str, default=None, help="Save metrics JSON")
    args = parser.parse_args()

    gen = RealisticBananaGenerator()
    processed_dir = Path(cfg.get("paths", "processed_data", default="data/processed"))
    image_paths, shelf_life_labels, stage_labels = gen.generate_batch(
        args.samples, processed_dir / "eval"
    )

    if args.model == "ensemble":
        preprocessor = BananaPreprocessor()
        extractor = AdvancedFeatureExtractor()

        features = []
        for path in image_paths:
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

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = AdvancedMLEnsemble.load("advanced_ensemble")
        y_pred = model.predict(X_test)
        metrics = comprehensive_metrics(y_test, y_pred)

    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            image_paths, shelf_life_labels, stage_labels,
            preprocessor=BananaPreprocessor(),
        )
        model = ModelFactory.create(args.model, {
            "backbone": cfg.get("models", "cnn", "backbone", default="efficientnet_v2_s"),
            "dropout": 0.0,
        })
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_dir = Path(cfg.get("paths", "models_dir", default="models/saved"))
        state = torch.load(models_dir / "best_model.pth", map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.to(device)
        model.eval()

        all_preds, all_targets = [], []
        for batch in test_loader:
            with torch.no_grad():
                pred, _, _, _ = model(batch["image"].to(device))
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch["shelf_life"].numpy())
        y_pred = np.array(all_preds)
        y_test = np.array(all_targets)
        metrics = comprehensive_metrics(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  COMPREHENSIVE EVALUATION - {args.model.upper()}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    models_dir = Path(cfg.get("paths", "models_dir", default="models/saved"))
    plot_predictions(y_test, y_pred, save_path=str(models_dir / f"{args.model}_eval_predictions.png"))
    plot_error_analysis(y_test, y_pred, save_path=str(models_dir / f"{args.model}_eval_errors.png"))

    if args.save:
        with open(args.save, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved: {args.save}")


if __name__ == "__main__":
    main()
