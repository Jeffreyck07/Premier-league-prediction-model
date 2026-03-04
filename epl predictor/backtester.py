"""Backtesting and calibration utilities for EPL predictor."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .models import BettingOptimizer, ModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class BacktestSummary:
    initial_bankroll: float
    ending_bankroll: float
    total_bets_placed: int
    win_rate: float
    total_profit_loss: float
    roi: float
    max_drawdown: float
    total_staked: float
    brier_score: float
    ece: float
    n_matches: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_bankroll": self.initial_bankroll,
            "ending_bankroll": self.ending_bankroll,
            "total_bets_placed": self.total_bets_placed,
            "win_rate": self.win_rate,
            "total_profit_loss": self.total_profit_loss,
            "roi": self.roi,
            "max_drawdown": self.max_drawdown,
            "total_staked": self.total_staked,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "n_matches": self.n_matches,
        }


class ModelEvaluator:
    """Probability quality metrics for multiclass match outcomes."""

    @staticmethod
    def _encode_targets(y_true: Sequence[Any], label_encoder: LabelEncoder) -> np.ndarray:
        arr = np.asarray(y_true)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        if np.issubdtype(arr.dtype, np.integer):
            y_idx = arr.astype(int)
        elif np.issubdtype(arr.dtype, np.floating) and np.all(np.mod(arr, 1.0) == 0):
            y_idx = arr.astype(int)
        else:
            y_idx = label_encoder.transform(arr.astype(str))

        n_classes = len(label_encoder.classes_)
        if np.any((y_idx < 0) | (y_idx >= n_classes)):
            raise ValueError("y_true contains values outside valid class index range.")
        return y_idx

    @staticmethod
    def _sanitize_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
        probs = np.asarray(proba, dtype=float)
        if probs.ndim != 2:
            raise ValueError("proba must be a 2D array of shape (n_samples, n_classes).")
        if probs.shape[1] != n_classes:
            raise ValueError(f"proba has {probs.shape[1]} columns but expected {n_classes}.")

        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        uniform = np.full_like(probs, 1.0 / n_classes)
        return np.divide(probs, row_sums, out=uniform, where=row_sums > 0)

    def brier_score_multiclass(
        self,
        y_true: Sequence[Any],
        proba: np.ndarray,
        label_encoder: LabelEncoder,
    ) -> float:
        """Compute multiclass Brier score: mean(sum_k (p_k - y_k)^2)."""
        y_idx = self._encode_targets(y_true, label_encoder)
        n_classes = len(label_encoder.classes_)
        probs = self._sanitize_proba(proba, n_classes)
        if probs.shape[0] != y_idx.shape[0]:
            raise ValueError("y_true and proba must have the same number of samples.")

        one_hot = np.eye(n_classes, dtype=float)[y_idx]
        score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        return float(score)

    def reliability_summary(
        self,
        y_true: Sequence[Any],
        proba: np.ndarray,
        label_encoder: LabelEncoder,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Build a confidence reliability table using max-probability bins."""
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2.")

        y_idx = self._encode_targets(y_true, label_encoder)
        probs = self._sanitize_proba(proba, len(label_encoder.classes_))
        if probs.shape[0] != y_idx.shape[0]:
            raise ValueError("y_true and proba must have the same number of samples.")

        preds = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        correct = (preds == y_idx).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)

        rows: List[Dict[str, Any]] = []
        for i in range(n_bins):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == 0:
                mask = (conf >= lo) & (conf <= hi)
            else:
                mask = (conf > lo) & (conf <= hi)

            count = int(mask.sum())
            if count == 0:
                rows.append(
                    {
                        "bin": i,
                        "bin_low": lo,
                        "bin_high": hi,
                        "count": 0,
                        "avg_confidence": np.nan,
                        "accuracy": np.nan,
                        "abs_gap": np.nan,
                    }
                )
                continue

            avg_conf = float(conf[mask].mean())
            acc = float(correct[mask].mean())
            rows.append(
                {
                    "bin": i,
                    "bin_low": lo,
                    "bin_high": hi,
                    "count": count,
                    "avg_confidence": avg_conf,
                    "accuracy": acc,
                    "abs_gap": float(abs(acc - avg_conf)),
                }
            )
        return pd.DataFrame(rows)

    def expected_calibration_error(
        self,
        y_true: Sequence[Any],
        proba: np.ndarray,
        label_encoder: LabelEncoder,
        n_bins: int = 10,
    ) -> float:
        """Compute ECE over confidence bins: sum_w |acc(bin) - conf(bin)|."""
        summary = self.reliability_summary(y_true, proba, label_encoder, n_bins=n_bins)
        used = summary["count"] > 0
        if not bool(used.any()):
            return 0.0

        n_total = float(summary.loc[used, "count"].sum())
        weighted_gap = (summary.loc[used, "count"] / n_total) * summary.loc[used, "abs_gap"]
        return float(weighted_gap.sum())


class BacktestEngine:
    """Simulate betting strategy over historical fixtures."""

    def __init__(
        self,
        trainer: ModelTrainer,
        label_encoder: LabelEncoder,
        evaluator: Optional[ModelEvaluator] = None,
        initial_bankroll: float = 10_000.0,
        kelly_cap: float = 0.25,
        kelly_scale: float = 1.0,
        min_ev: float = 0.0,
        calibration_bins: int = 10,
    ):
        if initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive.")
        if not (0 < kelly_cap <= 1.0):
            raise ValueError("kelly_cap must be in (0, 1].")
        if calibration_bins < 2:
            raise ValueError("calibration_bins must be >= 2.")

        self.trainer = trainer
        self.label_encoder = label_encoder
        self.evaluator = evaluator or ModelEvaluator()
        self.initial_bankroll = float(initial_bankroll)
        self.kelly_cap = float(kelly_cap)
        self.kelly_scale = float(kelly_scale)
        self.min_ev = float(min_ev)
        self.calibration_bins = int(calibration_bins)

    @staticmethod
    def _max_drawdown(equity: Sequence[float]) -> float:
        curve = np.asarray(equity, dtype=float)
        if curve.size == 0:
            return 0.0
        running_peak = np.maximum.accumulate(curve)
        drawdowns = np.divide(
            running_peak - curve,
            running_peak,
            out=np.zeros_like(curve),
            where=running_peak > 0,
        )
        return float(np.max(drawdowns))

    @staticmethod
    def _normalize_outcome(value: Any, label_encoder: LabelEncoder) -> Optional[str]:
        if pd.isna(value):
            return None

        if isinstance(value, (np.integer, int)):
            idx = int(value)
            if 0 <= idx < len(label_encoder.classes_):
                return str(label_encoder.inverse_transform([idx])[0])
            return None

        if isinstance(value, (np.floating, float)) and float(value).is_integer():
            idx = int(value)
            if 0 <= idx < len(label_encoder.classes_):
                return str(label_encoder.inverse_transform([idx])[0])
            return None

        text = str(value).strip().upper()
        aliases = {
            "H": "H",
            "D": "D",
            "A": "A",
            "HOME": "H",
            "DRAW": "D",
            "AWAY": "A",
            "1": "H",
            "X": "D",
            "2": "A",
        }
        return aliases.get(text)

    @staticmethod
    def _validate_inputs(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        odds_cols: Tuple[str, str, str],
    ) -> None:
        if df.empty:
            raise ValueError("Backtest DataFrame is empty.")
        if len(odds_cols) != 3:
            raise ValueError("odds_cols must contain exactly 3 columns: (B365H, B365D, B365A).")

        required = set(feature_cols) | {target_col} | set(odds_cols)
        missing = sorted(col for col in required if col not in df.columns)
        if missing:
            raise ValueError(f"Missing required columns for backtest: {missing}")

    def run_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str = "target",
        odds_cols: Tuple[str, str, str] = ("B365H", "B365D", "B365A"),
        use_thresholds: bool = True,
    ) -> Dict[str, Any]:
        """
        Run bankroll simulation using +EV + Kelly logic.

        Returns:
            dict containing summary metrics, `bet_log` DataFrame, and reliability table.
        """
        self._validate_inputs(df, feature_cols, target_col, odds_cols)
        odds_h_col, odds_d_col, odds_a_col = odds_cols

        bankroll = float(self.initial_bankroll)
        total_staked = 0.0
        total_bets = 0
        winning_bets = 0
        equity_curve: List[float] = [bankroll]

        eval_y: List[str] = []
        eval_proba: List[np.ndarray] = []
        logs: List[Dict[str, Any]] = []

        for seq, (row_idx, row) in enumerate(df.iterrows(), start=1):
            X_row = np.asarray(row.loc[list(feature_cols)], dtype=float).reshape(1, -1)
            proba = self.trainer.predict_proba(X_row, self.label_encoder)[0]
            pred_idx = int(self.trainer.predict(X_row, self.label_encoder, use_thresholds=use_thresholds)[0])
            pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])

            prob_dict = {cls: float(proba[i]) for i, cls in enumerate(self.label_encoder.classes_)}
            actual_outcome = self._normalize_outcome(row[target_col], self.label_encoder)

            if actual_outcome is not None:
                eval_y.append(actual_outcome)
                eval_proba.append(proba)

            odds_dict = {
                "H": float(row[odds_h_col]) if pd.notna(row[odds_h_col]) else np.nan,
                "D": float(row[odds_d_col]) if pd.notna(row[odds_d_col]) else np.nan,
                "A": float(row[odds_a_col]) if pd.notna(row[odds_a_col]) else np.nan,
            }
            market = BettingOptimizer.evaluate_market(
                prob_dict,
                odds_dict,
                kelly_cap=self.kelly_cap,
                kelly_scale=self.kelly_scale,
            )

            candidates = [
                (outcome, data)
                for outcome, data in market.items()
                if data["plus_ev"] > 0 and data["kelly"] > 0 and data["ev"] >= self.min_ev and np.isfinite(data["odds"])
            ]

            selected_outcome: Optional[str] = None
            selected = None
            stake = 0.0
            pnl = 0.0
            won = 0

            if candidates:
                selected_outcome, selected = max(candidates, key=lambda item: (item[1]["ev"], item[1]["kelly"]))
                stake = float(bankroll * selected["kelly"])
                if stake > 0:
                    total_bets += 1
                    total_staked += stake
                    if actual_outcome is not None and selected_outcome == actual_outcome:
                        pnl = stake * (float(selected["odds"]) - 1.0)
                        winning_bets += 1
                        won = 1
                    else:
                        pnl = -stake
                    bankroll += pnl

            equity_curve.append(bankroll)

            rec: Dict[str, Any] = {
                "seq": seq,
                "match_index": row_idx,
                "predicted_outcome": pred_label,
                "actual_outcome": actual_outcome,
                "bet_placed": 1 if selected_outcome is not None and stake > 0 else 0,
                "bet_outcome": selected_outcome,
                "stake": float(stake),
                "pnl": float(pnl),
                "bankroll": float(bankroll),
                "won_bet": won,
                "prob_H": prob_dict.get("H", np.nan),
                "prob_D": prob_dict.get("D", np.nan),
                "prob_A": prob_dict.get("A", np.nan),
                "odds_H": odds_dict["H"],
                "odds_D": odds_dict["D"],
                "odds_A": odds_dict["A"],
            }

            if selected is not None:
                rec["selected_ev"] = float(selected["ev"])
                rec["selected_kelly"] = float(selected["kelly"])
            else:
                rec["selected_ev"] = np.nan
                rec["selected_kelly"] = np.nan

            if "season" in row.index:
                rec["season"] = row["season"]
            if "home_team" in row.index:
                rec["home_team"] = row["home_team"]
            if "away_team" in row.index:
                rec["away_team"] = row["away_team"]
            if "parsed_date" in row.index:
                rec["parsed_date"] = row["parsed_date"]
            elif "Date" in row.index:
                rec["Date"] = row["Date"]

            logs.append(rec)

        win_rate = float(winning_bets / total_bets) if total_bets > 0 else 0.0
        total_profit = float(bankroll - self.initial_bankroll)
        roi = float(total_profit / total_staked) if total_staked > 0 else 0.0
        max_dd = self._max_drawdown(equity_curve)

        if eval_proba:
            eval_proba_arr = np.vstack(eval_proba)
            brier = self.evaluator.brier_score_multiclass(eval_y, eval_proba_arr, self.label_encoder)
            ece = self.evaluator.expected_calibration_error(
                eval_y,
                eval_proba_arr,
                self.label_encoder,
                n_bins=self.calibration_bins,
            )
            reliability = self.evaluator.reliability_summary(
                eval_y,
                eval_proba_arr,
                self.label_encoder,
                n_bins=self.calibration_bins,
            )
        else:
            brier = np.nan
            ece = np.nan
            reliability = pd.DataFrame(
                columns=["bin", "bin_low", "bin_high", "count", "avg_confidence", "accuracy", "abs_gap"]
            )

        summary = BacktestSummary(
            initial_bankroll=self.initial_bankroll,
            ending_bankroll=float(bankroll),
            total_bets_placed=int(total_bets),
            win_rate=win_rate,
            total_profit_loss=total_profit,
            roi=roi,
            max_drawdown=float(max_dd),
            total_staked=float(total_staked),
            brier_score=float(brier) if np.isfinite(brier) else np.nan,
            ece=float(ece) if np.isfinite(ece) else np.nan,
            n_matches=int(len(df)),
        )

        return {
            **summary.to_dict(),
            "bet_log": pd.DataFrame(logs),
            "reliability": reliability,
        }
