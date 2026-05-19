import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from config import cfg


class AdvancedMLEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, model_names: Optional[List[str]] = None, use_optimization: bool = True):
        self.model_names = model_names or ["random_forest", "xgboost", "lightgbm", "ridge", "elastic_net"]
        self.use_optimization = use_optimization
        self.models: Dict[str, Pipeline] = {}
        self.weights: Dict[str, float] = {}
        self.scaler = RobustScaler()
        self.meta_model = Ridge(alpha=0.5)

    def _get_base_model(self, name: str):
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=500, max_depth=20, min_samples_leaf=1,
                max_features="sqrt", n_jobs=-1, random_state=42
            ),
            "xgboost": xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, verbosity=0
            ),
            "lightgbm": lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, verbose=-1
            ),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.01, max_iter=5000),
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42),
            "svr": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
        }
        return models.get(name)

    def _optimize_hyperparams(self, name: str, X: np.ndarray, y: np.ndarray):
        if not self.use_optimization:
            return self._get_base_model(name)

        try:
            import optuna

            def objective(trial):
                if name == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                        "max_depth": trial.suggest_int("max_depth", 4, 12),
                        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                    }
                    model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
                elif name == "lightgbm":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                        "max_depth": trial.suggest_int("max_depth", 4, 12),
                        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                    }
                    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
                elif name == "random_forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                        "max_depth": trial.suggest_int("max_depth", 5, 30),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    }
                    model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
                else:
                    model = self._get_base_model(name)

                pipe = Pipeline([("scaler", RobustScaler()), ("model", model)])
                scores = cross_val_score(pipe, X, y, cv=3, scoring="neg_mean_absolute_error")
                return -scores.mean()

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=5, show_progress_bar=False)

            if name == "xgboost":
                return xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0, n_jobs=-1)
            elif name == "lightgbm":
                return lgb.LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
            elif name == "random_forest":
                return RandomForestRegressor(**study.best_params, n_jobs=-1, random_state=42)
        except ImportError:
            pass

        return self._get_base_model(name)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)

        for name in self.model_names:
            model = self._optimize_hyperparams(name, X_scaled, y)
            pipe = Pipeline([
                ("scaler", RobustScaler()),
                ("model", model),
            ])
            pipe.fit(X, y)
            self.models[name] = pipe

            scores = cross_val_score(pipe, X, y, cv=3, scoring="neg_mean_absolute_error")
            self.weights[name] = -scores.mean()
            print(f"  {name:20s} | CV MAE: {-scores.mean():.4f} ± {scores.std():.4f}")

        total = sum(1 / max(w, 1e-8) for w in self.weights.values())
        for name in self.weights:
            self.weights[name] = (1 / max(self.weights[name], 1e-8)) / total

        oof_preds = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            oof_preds[:, i] = cross_val_predict_wrapper(model, X, y, cv=3)

        self.meta_model.fit(oof_preds, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        weighted = np.zeros(len(X))
        for name, model in self.models.items():
            weighted += self.weights[name] * model.predict(X)
        return weighted

    def predict_with_components(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        components = {}
        for name, model in self.models.items():
            components[name] = model.predict(X)
        ensemble = np.mean(list(components.values()), axis=0)
        return ensemble, components

    def save(self, name: str = "advanced_ensemble"):
        path = Path(cfg.get("paths", "models_dir", default="models/saved")) / f"{name}.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "advanced_ensemble"):
        stem = name.replace(".pkl", "")
        path = Path(cfg.get("paths", "models_dir", default="models/saved")) / f"{stem}.pkl"
        return joblib.load(path)


class ModelEnsemble:
    def __init__(self, models: List[BaseEstimator], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return np.average(preds, axis=1, weights=self.weights)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return preds.std(axis=1)


def cross_val_predict_wrapper(estimator, X, y, cv=3):
    from sklearn.model_selection import cross_val_predict
    try:
        return cross_val_predict(estimator, X, y, cv=cv, method="predict")
    except Exception:
        return np.full(len(X), y.mean())
