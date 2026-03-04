"""Model training, calibration, and betting optimization."""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from scipy.optimize import differential_evolution, minimize
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from .config import AppConfig, show_progress

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class DixonColesModel:
    def __init__(
        self,
        max_goals: int = 8,
        alpha: float = 0.1,
        max_iter: int = 1200,
        rho_bounds: Tuple[float, float] = (-0.2, 0.2),
    ):
        self.max_goals = max_goals
        self.alpha = alpha
        self.max_iter = max_iter
        self.rho_bounds = rho_bounds

        self.home_reg = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.away_reg = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.rho = 0.0
        self.optim_success = False
        self.train_nll = np.nan

    @staticmethod
    def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
        tau = np.ones_like(lam, dtype=float)
        mask_00 = (x == 0) & (y == 0)
        mask_01 = (x == 0) & (y == 1)
        mask_10 = (x == 1) & (y == 0)
        mask_11 = (x == 1) & (y == 1)

        tau[mask_00] = 1.0 - (lam[mask_00] * mu[mask_00] * rho)
        tau[mask_01] = 1.0 + (lam[mask_01] * rho)
        tau[mask_10] = 1.0 + (mu[mask_10] * rho)
        tau[mask_11] = 1.0 - rho

        return np.clip(tau, 1e-8, None)

    @staticmethod
    def _tau_scalar(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
        if hg == 0 and ag == 0:
            return max(1.0 - lam * mu * rho, 1e-8)
        if hg == 0 and ag == 1:
            return max(1.0 + lam * rho, 1e-8)
        if hg == 1 and ag == 0:
            return max(1.0 + mu * rho, 1e-8)
        if hg == 1 and ag == 1:
            return max(1.0 - rho, 1e-8)
        return 1.0

    @staticmethod
    def _clip_lambdas(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr.astype(float), 0.05, 6.0)

    def fit(self, X: np.ndarray, y_home: np.ndarray, y_away: np.ndarray) -> "DixonColesModel":
        self.home_reg.fit(X, y_home)
        self.away_reg.fit(X, y_away)

        lam = self._clip_lambdas(self.home_reg.predict(X))
        mu = self._clip_lambdas(self.away_reg.predict(X))
        y_h = y_home.astype(int)
        y_a = y_away.astype(int)

        def nll(params: np.ndarray) -> float:
            rho = float(params[0])
            base_log = poisson.logpmf(y_h, lam) + poisson.logpmf(y_a, mu)
            tau = self._tau(y_h, y_a, lam, mu, rho)
            return float(-(base_log + np.log(tau)).sum())

        opt = minimize(
            nll,
            x0=np.array([0.0], dtype=float),
            method="L-BFGS-B",
            bounds=[self.rho_bounds],
        )

        if opt.success:
            self.rho = float(np.clip(opt.x[0], self.rho_bounds[0], self.rho_bounds[1]))
            self.optim_success = True
        else:
            self.rho = 0.0
            self.optim_success = False

        self.train_nll = float(nll(np.array([self.rho], dtype=float)))
        return self

    def predict_expected_goals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lam = self._clip_lambdas(self.home_reg.predict(X))
        mu = self._clip_lambdas(self.away_reg.predict(X))
        return lam, mu

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        matrix = np.zeros((self.max_goals, self.max_goals), dtype=float)
        for hg in range(self.max_goals):
            for ag in range(self.max_goals):
                p = poisson.pmf(hg, lam) * poisson.pmf(ag, mu)
                p *= self._tau_scalar(hg, ag, lam, mu, self.rho)
                matrix[hg, ag] = p
        total = matrix.sum()
        if total <= 0:
            matrix[:] = 1.0 / (self.max_goals * self.max_goals)
            return matrix
        return matrix / total

    def predict_proba_hda(self, X: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
        cls_to_idx = {c: i for i, c in enumerate(label_encoder.classes_)}
        idx_h = cls_to_idx["H"]
        idx_d = cls_to_idx["D"]
        idx_a = cls_to_idx["A"]

        lam, mu = self.predict_expected_goals(X)
        out = np.zeros((len(X), len(label_encoder.classes_)), dtype=float)

        for i in range(len(X)):
            mat = self.score_matrix(float(lam[i]), float(mu[i]))
            p_h = float(np.tril(mat, k=-1).sum())
            p_d = float(np.trace(mat))
            p_a = float(np.triu(mat, k=1).sum())
            out[i, idx_h] = p_h
            out[i, idx_d] = p_d
            out[i, idx_a] = p_a
        return out


class BettingOptimizer:
    @staticmethod
    def expected_value(prob: float, decimal_odds: float) -> float:
        if decimal_odds <= 1.0:
            return -1.0
        return float(prob * decimal_odds - 1.0)

    @staticmethod
    def kelly_fraction(prob: float, decimal_odds: float, cap: float = 0.25, scale: float = 1.0) -> float:
        if decimal_odds <= 1.0:
            return 0.0
        b = decimal_odds - 1.0
        q = 1.0 - prob
        frac = (prob * b - q) / b
        if frac <= 0:
            return 0.0
        scaled = max(0.0, float(frac) * max(float(scale), 0.0))
        return float(min(scaled, cap))

    @classmethod
    def evaluate_market(
        cls,
        prob_dict: Dict[str, float],
        odds_dict: Dict[str, float],
        kelly_cap: float = 0.25,
        kelly_scale: float = 1.0,
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for outcome, prob in prob_dict.items():
            odds = float(odds_dict.get(outcome, np.nan))
            ev = cls.expected_value(float(prob), odds) if np.isfinite(odds) else -1.0
            kelly = cls.kelly_fraction(float(prob), odds, cap=kelly_cap, scale=kelly_scale) if np.isfinite(odds) else 0.0
            out[outcome] = {
                "prob": float(prob),
                "odds": float(odds) if np.isfinite(odds) else np.nan,
                "ev": float(ev),
                "kelly": float(kelly),
                "plus_ev": 1.0 if ev > 0 else 0.0,
            }
        return out


class ModelTrainer:
    def __init__(self, config: AppConfig, quiet: bool, interactive_mode: bool, force_retrain: bool):
        self.config = config
        self.quiet = quiet
        self.interactive_mode = interactive_mode
        self.force_retrain = force_retrain

        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.dixon_coles_model: Optional[DixonColesModel] = None

        self.weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        self.thresholds: Optional[np.ndarray] = None
        self.best_params: Dict[str, Any] = {}
        self.cv_best_score: float = 0.0
        self.val_scores: Dict[str, float] = {}
        self.feature_cols: List[str] = []
        self.full_feature_cols: List[str] = []
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.full_feature_count: Optional[int] = None
        self.class_priors: Optional[np.ndarray] = None

    def _time_series_cv_score(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        n_samples = len(X)
        max_splits = min(self.config.tscv_splits, max(2, n_samples // 250))
        splitter = TimeSeriesSplit(n_splits=max_splits)
        scores = []
        for tr_idx, va_idx in splitter.split(X):
            model = lgb.LGBMClassifier(**params)
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[va_idx])
            scores.append(accuracy_score(y[va_idx], preds))

        return float(np.mean(scores)) if scores else 0.0

    def _tune_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        if not self.quiet:
            logger.info(f"\n--- Hyperparameter Tuning with Optuna + TimeSeriesSplit ({self.config.optuna_trials} trials) ---")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 80),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
            }
            return self._time_series_cv_score(params, X, y)

        def _progress(study: optuna.Study, trial: optuna.Trial) -> None:
            show_progress(
                58 + int(15 * (trial.number + 1) / max(self.config.optuna_trials, 1)),
                f"Tuning ({trial.number + 1}/{self.config.optuna_trials})",
                self.interactive_mode,
            )

        study = optuna.create_study(direction="maximize")
        callbacks = [_progress] if self.interactive_mode else []
        study.optimize(objective, n_trials=self.config.optuna_trials, show_progress_bar=not self.quiet, callbacks=callbacks)

        best_params = dict(study.best_params)
        best_params.update(
            {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "verbosity": -1,
                "random_state": 42,
                "n_jobs": -1,
            }
        )

        if not self.quiet:
            logger.info(f"  Best TimeSeries CV Accuracy: {study.best_value:.1%}")
            logger.info(
                f"  Best params: lr={best_params['learning_rate']:.4f}, depth={best_params['max_depth']}, "
                f"leaves={best_params['num_leaves']}, n_estimators={best_params['n_estimators']}"
            )

        return best_params, float(study.best_value)

    def _chronological_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_home_goals: np.ndarray,
        y_away_goals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        val_size = max(80, int(n * self.config.validation_fraction))
        val_size = min(val_size, max(40, n // 3))
        split_idx = n - val_size

        return (
            X[:split_idx],
            X[split_idx:],
            y[:split_idx],
            y[split_idx:],
            y_home_goals[:split_idx],
            y_home_goals[split_idx:],
            y_away_goals[:split_idx],
            y_away_goals[split_idx:],
        )

    def _fit_base_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_home_goals: np.ndarray,
        y_away_goals: np.ndarray,
    ) -> None:
        self.lgb_model = lgb.LGBMClassifier(**self.best_params)
        self.lgb_model.fit(X, y)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.best_params["n_estimators"],
            max_depth=self.best_params["max_depth"],
            learning_rate=self.best_params["learning_rate"],
            subsample=self.best_params["subsample"],
            colsample_bytree=self.best_params["colsample_bytree"],
            reg_alpha=self.best_params["reg_alpha"],
            reg_lambda=self.best_params["reg_lambda"],
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.xgb_model.fit(X, y)

        self.dixon_coles_model = DixonColesModel(
            max_goals=self.config.dixon_coles_max_goals,
            alpha=0.1,
            max_iter=1200,
            rho_bounds=(self.config.dixon_coles_rho_min, self.config.dixon_coles_rho_max),
        )
        self.dixon_coles_model.fit(X, y_home_goals, y_away_goals)

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if self.selected_feature_indices is None:
            return X
        if X.shape[1] == len(self.selected_feature_indices):
            return X
        if self.full_feature_count is not None and X.shape[1] == self.full_feature_count:
            return X[:, self.selected_feature_indices]
        raise ValueError(
            f"Unexpected feature width {X.shape[1]} for model expecting "
            f"{self.full_feature_count} (full) or {len(self.selected_feature_indices)} (selected)."
        )

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        return self._select_features(X)

    def _dixon_coles_proba(self, X: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
        X_model = self._select_features(X)
        return self.dixon_coles_model.predict_proba_hda(X_model, label_encoder)

    def predict_proba(self, X: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
        X_model = self._select_features(X)
        lgb_proba = self.lgb_model.predict_proba(X_model)
        xgb_proba = self.xgb_model.predict_proba(X_model)
        dc_proba = self._dixon_coles_proba(X, label_encoder)
        proba = self.weights[0] * lgb_proba + self.weights[1] * xgb_proba + self.weights[2] * dc_proba
        if self.class_priors is not None:
            lam = float(np.clip(getattr(self.config, "class_prior_blend", 0.0), 0.0, 0.4))
            if lam > 0:
                proba = (1.0 - lam) * proba + lam * self.class_priors.reshape(1, -1)
                row_sum = proba.sum(axis=1, keepdims=True)
                proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
        return proba

    @staticmethod
    def threshold_predict(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        thr = thresholds.reshape(1, -1)
        adjusted = proba / (thr + 1e-6)
        eligible = proba >= thr
        preds = np.empty(proba.shape[0], dtype=int)

        for i in range(proba.shape[0]):
            if eligible[i].any():
                cand = np.where(eligible[i], adjusted[i], -1e9)
                preds[i] = int(np.argmax(cand))
            else:
                preds[i] = int(np.argmax(adjusted[i]))

        return preds

    def _optimize_thresholds(self, proba_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.quiet:
            logger.info("\n--- Optimizing H/D/A Probability Thresholds (validation) ---")

        def objective(thr: np.ndarray) -> float:
            preds = self.threshold_predict(proba_val, thr)
            score = balanced_accuracy_score(y_val, preds)
            # Discourage pushing Draw threshold too high (class index 1),
            # which can suppress draw predictions entirely.
            penalty = 0.0
            if len(thr) > 1 and thr[1] > 0.45:
                penalty = 0.05 * (thr[1] - 0.45)
            # Keep predicted draw rate near observed draw base rate.
            if proba_val.shape[1] > 1:
                draw_true_rate = float(np.mean(y_val == 1))
                draw_pred_rate = float(np.mean(preds == 1))
                penalty += 0.20 * abs(draw_pred_rate - draw_true_rate)
            return -(score - penalty)

        bounds = [(0.10, 0.70)] * proba_val.shape[1]
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=80,
            popsize=16,
            tol=1e-4,
            polish=True,
            updating="deferred",
            workers=1,
        )

        best_thr = np.clip(result.x, 0.05, 0.90)
        best_acc = balanced_accuracy_score(y_val, self.threshold_predict(proba_val, best_thr))

        if not self.quiet:
            thr_str = ", ".join(f"{v:.3f}" for v in best_thr)
            logger.info(f"  Optimal thresholds: [{thr_str}]")
            logger.info(f"  Thresholded validation balanced accuracy: {best_acc:.1%}")

        return best_thr.astype(float), float(best_acc)

    def _build_oof_threshold_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_home_goals: np.ndarray,
        y_away_goals: np.ndarray,
        label_encoder: LabelEncoder,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        n_classes = len(label_encoder.classes_)
        max_splits = min(getattr(self.config, "threshold_oof_splits", 5), max(2, n_samples // 250))
        splitter = TimeSeriesSplit(n_splits=max_splits)
        oof = np.full((n_samples, n_classes), np.nan, dtype=float)

        for tr_idx, va_idx in splitter.split(X):
            if len(np.unique(y[tr_idx])) < n_classes:
                continue
            try:
                fold_lgb = lgb.LGBMClassifier(**self.best_params)
                fold_lgb.fit(X[tr_idx], y[tr_idx])

                fold_xgb = xgb.XGBClassifier(
                    n_estimators=self.best_params["n_estimators"],
                    max_depth=self.best_params["max_depth"],
                    learning_rate=self.best_params["learning_rate"],
                    subsample=self.best_params["subsample"],
                    colsample_bytree=self.best_params["colsample_bytree"],
                    reg_alpha=self.best_params["reg_alpha"],
                    reg_lambda=self.best_params["reg_lambda"],
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                fold_xgb.fit(X[tr_idx], y[tr_idx])

                fold_dc = DixonColesModel(
                    max_goals=self.config.dixon_coles_max_goals,
                    alpha=0.1,
                    max_iter=1200,
                    rho_bounds=(self.config.dixon_coles_rho_min, self.config.dixon_coles_rho_max),
                )
                fold_dc.fit(X[tr_idx], y_home_goals[tr_idx], y_away_goals[tr_idx])

                lgb_p = fold_lgb.predict_proba(X[va_idx])
                xgb_p = fold_xgb.predict_proba(X[va_idx])
                dc_p = fold_dc.predict_proba_hda(X[va_idx], label_encoder)

                # Weight by fold validation accuracy to approximate ensemble reliability.
                lgb_acc = accuracy_score(y[va_idx], np.argmax(lgb_p, axis=1))
                xgb_acc = accuracy_score(y[va_idx], np.argmax(xgb_p, axis=1))
                dc_acc = accuracy_score(y[va_idx], np.argmax(dc_p, axis=1))
                w = np.array([lgb_acc, xgb_acc, dc_acc], dtype=float) + 1e-6
                w = w / w.sum()
                oof[va_idx] = w[0] * lgb_p + w[1] * xgb_p + w[2] * dc_p
            except Exception as exc:
                logger.warning("  OOF threshold fold failed: %s", exc)
                continue

        valid = np.isfinite(oof).all(axis=1)
        return oof[valid], y[valid]

    def _save_cache(self) -> None:
        payload = {
            "lgb_model": self.lgb_model,
            "xgb_model": self.xgb_model,
            "dixon_coles_model": self.dixon_coles_model,
            "weights": self.weights,
            "thresholds": self.thresholds,
            "best_params": self.best_params,
            "cv_best_score": self.cv_best_score,
            "val_scores": self.val_scores,
            "feature_cols": self.feature_cols,
            "full_feature_cols": self.full_feature_cols,
            "selected_feature_indices": self.selected_feature_indices,
            "full_feature_count": self.full_feature_count,
            "class_priors": self.class_priors,
            "cache_meta": {
                "use_bookmaker_features": bool(getattr(self.config, "use_bookmaker_features", False)),
                "use_odds_movement_feature": bool(getattr(self.config, "use_odds_movement_feature", False)),
                "class_prior_blend": float(getattr(self.config, "class_prior_blend", 0.0)),
            },
        }
        joblib.dump(payload, self.config.model_cache_file, compress=3)

    def _load_cache(self) -> bool:
        if not os.path.exists(self.config.model_cache_file) or self.force_retrain:
            return False

        try:
            data = joblib.load(self.config.model_cache_file)
        except Exception as exc:
            if not self.quiet:
                logger.warning("  Cache load failed (%s), retraining from scratch.", exc)
            return False

        # Guard against feature-schema drift between cached model and current runtime.
        current_full_feature_cols = list(self.full_feature_cols if self.full_feature_cols else self.feature_cols)
        current_full_count = self.full_feature_count
        cached_full_count = data.get("full_feature_count")
        cached_full_feature_cols = list(data.get("full_feature_cols", []))
        if current_full_count is not None and cached_full_count is not None:
            if int(current_full_count) != int(cached_full_count):
                return False
        if current_full_feature_cols and cached_full_feature_cols:
            if current_full_feature_cols != cached_full_feature_cols:
                return False

        self.lgb_model = data["lgb_model"]
        self.xgb_model = data["xgb_model"]
        self.dixon_coles_model = data.get("dixon_coles_model")
        if self.dixon_coles_model is None:
            return False
        self.weights = np.array(data.get("weights", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
        self.thresholds = np.array(data.get("thresholds", [0.33, 0.33, 0.33]), dtype=float)
        self.best_params = data.get("best_params", {})
        self.cv_best_score = float(data.get("cv_best_score", 0.0))
        self.val_scores = data.get("val_scores", {})
        self.feature_cols = list(data.get("feature_cols", []))
        self.full_feature_cols = list(data.get("full_feature_cols", []))
        sel = data.get("selected_feature_indices")
        self.selected_feature_indices = np.array(sel, dtype=int) if sel is not None else None
        self.full_feature_count = data.get("full_feature_count")
        cp = data.get("class_priors")
        self.class_priors = np.array(cp, dtype=float) if cp is not None else None
        cache_meta = data.get("cache_meta", {})
        if bool(cache_meta.get("use_bookmaker_features", False)) != bool(
            getattr(self.config, "use_bookmaker_features", False)
        ):
            return False
        if bool(cache_meta.get("use_odds_movement_feature", False)) != bool(
            getattr(self.config, "use_odds_movement_feature", False)
        ):
            return False
        return True

    def fit(
        self,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        y_home_goals_train_full: np.ndarray,
        y_away_goals_train_full: np.ndarray,
        label_encoder: LabelEncoder,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        if feature_cols is None:
            feature_cols = [f"f{i}" for i in range(X_train_full.shape[1])]
        self.full_feature_cols = list(feature_cols)
        self.feature_cols = list(feature_cols)
        self.full_feature_count = len(self.feature_cols)
        n_classes = len(label_encoder.classes_)
        class_counts = np.bincount(y_train_full.astype(int), minlength=n_classes).astype(float)
        class_total = max(class_counts.sum(), 1.0)
        self.class_priors = class_counts / class_total

        if self._load_cache():
            if not self.quiet:
                logger.info("\n--- Found cached v7 model, loading (skip retraining) ---")
            show_progress(93, "Model loaded", self.interactive_mode)
            return

        X_fit, X_val, y_fit, y_val, y_hg_fit, y_hg_val, y_ag_fit, y_ag_val = self._chronological_validation_split(
            X_train_full,
            y_train_full,
            y_home_goals_train_full,
            y_away_goals_train_full,
        )

        show_progress(58, "Tuning hyperparameters...", self.interactive_mode)
        self.best_params, self.cv_best_score = self._tune_lightgbm(X_fit, y_fit)

        # Automated feature pruning: drop the bottom 20% by LightGBM importance.
        selector_model = lgb.LGBMClassifier(**self.best_params)
        selector_model.fit(X_fit, y_fit)
        importances = np.asarray(selector_model.feature_importances_, dtype=float)
        n_features = importances.shape[0]
        drop_n = int(np.floor(0.20 * n_features))
        if n_features > 5 and drop_n > 0:
            drop_idx = np.argsort(importances)[:drop_n]
            keep_mask = np.ones(n_features, dtype=bool)
            keep_mask[drop_idx] = False
            keep_idx = np.where(keep_mask)[0]
            if keep_idx.size == 0:
                keep_idx = np.array([int(np.argmax(importances))], dtype=int)
        else:
            keep_idx = np.arange(n_features, dtype=int)

        self.selected_feature_indices = keep_idx.astype(int)
        self.feature_cols = [self.feature_cols[i] for i in self.selected_feature_indices]

        X_fit_sel = X_fit[:, self.selected_feature_indices]
        X_val_sel = X_val[:, self.selected_feature_indices]
        X_train_full_sel = X_train_full[:, self.selected_feature_indices]

        if not self.quiet:
            removed = n_features - len(self.selected_feature_indices)
            logger.info(
                f"  Feature selection: kept {len(self.selected_feature_indices)}/{n_features} "
                f"features, dropped {removed}."
            )

        show_progress(76, "Training base models...", self.interactive_mode)
        self._fit_base_models(X_fit_sel, y_fit, y_hg_fit, y_ag_fit)

        lgb_val_acc = accuracy_score(y_val, self.lgb_model.predict(X_val_sel))
        xgb_val_acc = accuracy_score(y_val, self.xgb_model.predict(X_val_sel))

        dc_val_proba = self._dixon_coles_proba(X_val_sel, label_encoder)
        dc_val_acc = accuracy_score(y_val, np.argmax(dc_val_proba, axis=1))

        weight_raw = np.array([lgb_val_acc, xgb_val_acc, dc_val_acc], dtype=float) + 1e-6
        self.weights = weight_raw / weight_raw.sum()

        ens_val_proba = self.predict_proba(X_val_sel, label_encoder)
        ens_val_argmax = accuracy_score(y_val, np.argmax(ens_val_proba, axis=1))
        # Leakage-safe threshold calibration on out-of-fold time-series predictions.
        oof_proba, oof_y = self._build_oof_threshold_data(
            X_train_full_sel,
            y_train_full,
            y_home_goals_train_full,
            y_away_goals_train_full,
            label_encoder,
        )
        if len(oof_y) >= max(120, n_classes * 40):
            self.thresholds, ens_val_thr = self._optimize_thresholds(oof_proba, oof_y)
        else:
            self.thresholds, ens_val_thr = self._optimize_thresholds(ens_val_proba, y_val)

        self.val_scores = {
            "lgb_val_acc": float(lgb_val_acc),
            "xgb_val_acc": float(xgb_val_acc),
            "dixon_coles_val_acc": float(dc_val_acc),
            "ensemble_val_argmax_acc": float(ens_val_argmax),
            "ensemble_val_threshold_acc": float(ens_val_thr),
        }

        if not self.quiet:
            logger.info("\nValidation scores:")
            logger.info(f"  LightGBM: {lgb_val_acc:.1%}")
            logger.info(f"  XGBoost:  {xgb_val_acc:.1%}")
            logger.info(f"  Dixon-Coles: {dc_val_acc:.1%}")
            logger.info(f"  Ensemble (argmax):     {ens_val_argmax:.1%}")
            logger.info(f"  Ensemble (threshold):  {ens_val_thr:.1%}")
            logger.info(
                f"  Ensemble weights (LGB/XGB/DC): "
                f"{self.weights[0]:.3f} / {self.weights[1]:.3f} / {self.weights[2]:.3f}"
            )

        # Retrain base models on all train seasons for final deployment
        show_progress(86, "Retraining on full train set...", self.interactive_mode)
        self._fit_base_models(X_train_full_sel, y_train_full, y_home_goals_train_full, y_away_goals_train_full)

        self._save_cache()
        if not self.quiet:
            logger.info(f"\n  Model cached to {self.config.model_cache_file}")

    def predict(self, X: np.ndarray, label_encoder: LabelEncoder, use_thresholds: bool = True) -> np.ndarray:
        proba = self.predict_proba(X, label_encoder)
        if use_thresholds:
            if self.thresholds is None:
                self.thresholds = np.array([0.33] * proba.shape[1], dtype=float)
            return self.threshold_predict(proba, self.thresholds)
        return np.argmax(proba, axis=1)

    def predict_expected_goals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_model = self._select_features(X)
        return self.dixon_coles_model.predict_expected_goals(X_model)
