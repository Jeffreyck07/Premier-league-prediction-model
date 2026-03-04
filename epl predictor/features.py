"""Feature engineering module."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    AppConfig,
    DEFAULT_POSS,
    DEFAULT_XG,
    EURO_COMPETITION_DATA,
    show_progress,
)

class FeatureEngineer:
    ELO_K = 20
    # Reduced to avoid overweighting weak-home vs strong-away fixtures.
    ELO_HOME = 5
    ELO_START = 1500
    # Retain more carry-over strength across seasons.
    ELO_SEASON_REGRESS = 0.15
    DECAY = 0.5
    # Numeric stat keys that get halved at season boundaries.
    _DECAY_KEYS = (
        "played", "wins", "draws", "losses",
        "goals_for", "goals_against",
        "home_wins", "home_played", "home_goals_for", "home_goals_against",
        "away_wins", "away_played", "away_goals_for", "away_goals_against",
        "shots_for", "shots_against", "sot_for", "sot_against",
        "corners_for", "corners_against",
        "fouls", "fouls_against", "yellows", "reds",
    )

    def __init__(
        self,
        config: AppConfig,
        fbref_xg: Dict[Tuple[str, str], Dict[str, float]],
        fbref_poss: Dict[Tuple[str, str], float],
        match_xg_lookup: Dict[Tuple[str, str, str], Tuple[float, float]],
        fpl_team_data: Dict[str, Dict[str, float]],
        quiet: bool,
        interactive_mode: bool,
    ):
        self.config = config
        self.fbref_xg = fbref_xg
        self.fbref_poss = fbref_poss
        self.match_xg_lookup = match_xg_lookup
        self.fpl_team_data = fpl_team_data
        self.quiet = quiet
        self.interactive_mode = interactive_mode

        self.team_stats: Dict[str, Dict[str, Any]] = {}
        self.elo_ratings: Dict[str, float] = defaultdict(lambda: self.ELO_START)
        self.h2h_records: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "matches": 0}
        )
        self.all_teams: List[str] = []
        self.current_teams: List[str] = []

        self.feature_cols: List[str] = []
        self.euro_feature_names: List[str] = [
            "home_euro_tier",
            "away_euro_tier",
            "euro_tier_diff",
            "home_in_europe",
            "away_in_europe",
            "both_in_europe",
            "euro_fatigue_diff",
        ]
        self.euro_indices: List[int] = []
        self.use_bookmaker_features_runtime: bool = bool(self.config.use_bookmaker_features)
        self.use_odds_movement_feature_runtime: bool = bool(self.config.use_odds_movement_feature)
        self.odds_defaults: Dict[str, float] = {"B365H": 2.5, "B365D": 3.3, "B365A": 3.0}
        self.implied_defaults: Dict[str, float] = {"H": 0.45, "D": 0.27, "A": 0.28}
        self.odds_coverage: float = 0.0
        self.xg_proxy_coef_sot: float = float(self.config.xg_proxy_sot_coef)
        self.xg_proxy_coef_non_sot: float = float(self.config.xg_proxy_shot_coef)
        self.xg_proxy_intercept: float = float(self.config.xg_proxy_intercept)

    @staticmethod
    def elo_expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    @classmethod
    def elo_update(cls, ra: float, rb: float, score_a: float, k: float) -> float:
        ea = cls.elo_expected(ra, rb)
        return ra + k * (score_a - ea)

    @staticmethod
    def safe_div(a: float, b: float, default: float = 0.0) -> float:
        return float(a) / float(b) if b >= 1 else float(default)

    @staticmethod
    def ema(values: List[float], alpha: float, default: float) -> float:
        if not values:
            return default
        ema_value = float(values[0])
        for v in values[1:]:
            ema_value = alpha * float(v) + (1.0 - alpha) * ema_value
        return float(ema_value)

    def recent_window(self, values: List[float]) -> List[float]:
        n = max(int(self.config.ema_recent_window), 1)
        return values[-n:]

    def make_empty_stats(self) -> Dict[str, Any]:
        return {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "home_wins": 0,
            "home_played": 0,
            "home_goals_for": 0,
            "home_goals_against": 0,
            "away_wins": 0,
            "away_played": 0,
            "away_goals_for": 0,
            "away_goals_against": 0,
            "shots_for": 0,
            "shots_against": 0,
            "sot_for": 0,
            "sot_against": 0,
            "corners_for": 0,
            "corners_against": 0,
            "fouls": 0,
            "fouls_against": 0,
            "yellows": 0,
            "reds": 0,
            "recent_results": [],
            "recent_gf": [],
            "recent_ga": [],
            "recent_xg": [],
            "recent_xga": [],
            "last_match_date": None,
        }

    def decay_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        decayed = self.make_empty_stats()
        for key in self._DECAY_KEYS:
            decayed[key] = stats[key] * self.DECAY

        decayed["recent_results"] = stats["recent_results"][-20:]
        decayed["recent_gf"] = stats["recent_gf"][-20:]
        decayed["recent_ga"] = stats["recent_ga"][-20:]
        decayed["recent_xg"] = stats["recent_xg"][-20:]
        decayed["recent_xga"] = stats["recent_xga"][-20:]
        decayed["last_match_date"] = stats["last_match_date"]
        return decayed

    def get_team_xg(self, team_name: str, season: str) -> Dict[str, float]:
        return self.fbref_xg.get((season, team_name), DEFAULT_XG)

    def get_team_poss(self, team_name: str, season: str) -> float:
        return float(self.fbref_poss.get((season, team_name), DEFAULT_POSS))

    def get_team_fpl_data(self, team_name: str) -> Dict[str, float]:
        team_data = self.fpl_team_data.get(team_name, {})
        return {
            "fpl_form_score": float(team_data.get("fpl_form_score", self.config.fpl_default_form)),
            "key_players_missing": float(team_data.get("key_players_missing", 0.0)),
            "injury_pressure": float(team_data.get("injury_pressure", 0.0)),
        }

    @staticmethod
    def _as_float(value: Any, default: float = np.nan) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _placeholder_odds_movement(self, home_name: str, away_name: str, match_date: Any) -> float:
        # Keep placeholder deterministic and neutral to avoid injecting synthetic noise.
        _ = (home_name, away_name, match_date)
        return 0.0

    def compute_odds_movement(self, row_get: Any, home_name: str, away_name: str, match_date: Any) -> float:
        # Opening - Closing odds, averaged across H/D/A if closing lines are available.
        mapping = [
            ("B365H", ["B365CH", "PSCH", "AvgCH", "CloseH"]),
            ("B365D", ["B365CD", "PSCD", "AvgCD", "CloseD"]),
            ("B365A", ["B365CA", "PSCA", "AvgCA", "CloseA"]),
        ]
        moves: List[float] = []
        for open_col, close_candidates in mapping:
            opening = self._as_float(row_get(open_col, np.nan))
            if not np.isfinite(opening) or opening <= 0:
                continue
            closing = np.nan
            for ccol in close_candidates:
                cval = self._as_float(row_get(ccol, np.nan))
                if np.isfinite(cval) and cval > 0:
                    closing = cval
                    break
            if np.isfinite(closing):
                moves.append(opening - closing)

        if moves:
            return float(np.mean(moves))
        return self._placeholder_odds_movement(home_name, away_name, match_date)

    @staticmethod
    def _valid_odds_row(df: pd.DataFrame) -> pd.Series:
        return (
            df["B365H"].notna()
            & df["B365D"].notna()
            & df["B365A"].notna()
            & (df["B365H"] > 1.01)
            & (df["B365D"] > 1.01)
            & (df["B365A"] > 1.01)
        )

    def _fit_odds_priors(self, df: pd.DataFrame) -> None:
        required = {"B365H", "B365D", "B365A"}
        if not required.issubset(df.columns):
            self.use_bookmaker_features_runtime = False
            self.use_odds_movement_feature_runtime = False
            self.odds_coverage = 0.0
            return

        odds_df = df[["B365H", "B365D", "B365A"]].apply(pd.to_numeric, errors="coerce")
        valid = self._valid_odds_row(odds_df)
        self.odds_coverage = float(valid.mean()) if len(odds_df) > 0 else 0.0
        if valid.any():
            med_h = float(odds_df.loc[valid, "B365H"].median())
            med_d = float(odds_df.loc[valid, "B365D"].median())
            med_a = float(odds_df.loc[valid, "B365A"].median())
            self.odds_defaults = {"B365H": med_h, "B365D": med_d, "B365A": med_a}

            total = (1.0 / med_h) + (1.0 / med_d) + (1.0 / med_a)
            self.implied_defaults = {
                "H": float((1.0 / med_h) / total),
                "D": float((1.0 / med_d) / total),
                "A": float((1.0 / med_a) / total),
            }

        min_cov = float(self.config.min_odds_coverage_for_features)
        if self.odds_coverage < min_cov:
            self.use_bookmaker_features_runtime = False
            self.use_odds_movement_feature_runtime = False
        else:
            self.use_bookmaker_features_runtime = bool(self.config.use_bookmaker_features)
            self.use_odds_movement_feature_runtime = bool(self.config.use_odds_movement_feature)

    def _fit_xg_proxy_model(self, df: pd.DataFrame) -> None:
        if not self.match_xg_lookup:
            return

        x_rows: List[List[float]] = []
        y_vals: List[float] = []

        for _, row in df.iterrows():
            season = row.get("season")
            home = row.get("HomeTeam")
            away = row.get("AwayTeam")
            key = (season, home, away)
            if key not in self.match_xg_lookup:
                continue

            h_xg, a_xg = self.match_xg_lookup[key]
            hs = self._as_float(row.get("HS", np.nan))
            hst = self._as_float(row.get("HST", np.nan))
            ars = self._as_float(row.get("AS", np.nan))
            ast = self._as_float(row.get("AST", np.nan))

            if np.isfinite(hs) and np.isfinite(hst):
                h_non = max(hs - hst, 0.0)
                x_rows.append([float(hst), float(h_non), 1.0])
                y_vals.append(float(h_xg))
            if np.isfinite(ars) and np.isfinite(ast):
                a_non = max(ars - ast, 0.0)
                x_rows.append([float(ast), float(a_non), 1.0])
                y_vals.append(float(a_xg))

        if len(x_rows) < int(self.config.xg_proxy_fit_min_matches):
            return

        X = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            coef_sot = float(np.clip(coef[0], 0.05, 0.50))
            coef_non = float(np.clip(coef[1], 0.00, 0.20))
            intercept = float(np.clip(coef[2], 0.0, 0.8))
            self.xg_proxy_coef_sot = coef_sot
            self.xg_proxy_coef_non_sot = coef_non
            self.xg_proxy_intercept = intercept
        except Exception:
            return

    def _estimate_match_xg(self, shots: float, sot: float, team_xg_baseline: float) -> float:
        non_sot = max(float(shots) - float(sot), 0.0)
        proxy = (
            self.xg_proxy_coef_sot * float(sot)
            + self.xg_proxy_coef_non_sot * non_sot
            + self.xg_proxy_intercept
        )
        blend = float(np.clip(self.config.xg_proxy_baseline_weight, 0.0, 1.0))
        est = (1.0 - blend) * proxy + blend * float(team_xg_baseline)
        return float(np.clip(est, 0.05, 5.0))

    @staticmethod
    def get_euro_tier(team: str, season: str) -> int:
        return EURO_COMPETITION_DATA.get((season, team), 0)

    def extract_features(
        self,
        hs: Dict[str, Any],
        aws: Dict[str, Any],
        row: Any,
        home_name: str,
        away_name: str,
        season: str,
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # Elo
        home_elo = float(self.elo_ratings[home_name])
        away_elo = float(self.elo_ratings[away_name])
        f["home_elo"] = home_elo
        f["away_elo"] = away_elo
        f["elo_diff"] = home_elo - away_elo
        f["elo_expected_home"] = self.elo_expected(home_elo + self.ELO_HOME, away_elo)

        # Cumulative home stats
        f["home_win_rate"] = self.safe_div(hs["wins"], hs["played"], 0.33)
        f["home_draw_rate"] = self.safe_div(hs["draws"], hs["played"], 0.33)
        f["home_loss_rate"] = self.safe_div(hs["losses"], hs["played"], 0.33)
        f["home_avg_gf"] = self.safe_div(hs["goals_for"], hs["played"], 1.3)
        f["home_avg_ga"] = self.safe_div(hs["goals_against"], hs["played"], 1.3)
        f["home_gd_avg"] = self.safe_div(hs["goals_for"] - hs["goals_against"], hs["played"], 0.0)
        f["home_avg_shots"] = self.safe_div(hs["shots_for"], hs["played"], 12.0)
        f["home_avg_sot"] = self.safe_div(hs["sot_for"], hs["played"], 4.0)
        f["home_avg_corners"] = self.safe_div(hs["corners_for"], hs["played"], 5.0)
        f["home_avg_fouls"] = self.safe_div(hs["fouls"], hs["played"], 11.0)
        f["home_home_wr"] = self.safe_div(hs["home_wins"], hs["home_played"], 0.45)
        f["home_home_avg_gf"] = self.safe_div(hs["home_goals_for"], hs["home_played"], 1.5)
        f["home_home_avg_ga"] = self.safe_div(hs["home_goals_against"], hs["home_played"], 1.1)
        f["home_shot_accuracy"] = self.safe_div(hs["sot_for"], hs["shots_for"], 0.33)

        # Cumulative away stats
        f["away_win_rate"] = self.safe_div(aws["wins"], aws["played"], 0.33)
        f["away_draw_rate"] = self.safe_div(aws["draws"], aws["played"], 0.33)
        f["away_loss_rate"] = self.safe_div(aws["losses"], aws["played"], 0.33)
        f["away_avg_gf"] = self.safe_div(aws["goals_for"], aws["played"], 1.3)
        f["away_avg_ga"] = self.safe_div(aws["goals_against"], aws["played"], 1.3)
        f["away_gd_avg"] = self.safe_div(aws["goals_for"] - aws["goals_against"], aws["played"], 0.0)
        f["away_avg_shots"] = self.safe_div(aws["shots_for"], aws["played"], 12.0)
        f["away_avg_sot"] = self.safe_div(aws["sot_for"], aws["played"], 4.0)
        f["away_avg_corners"] = self.safe_div(aws["corners_for"], aws["played"], 5.0)
        f["away_avg_fouls"] = self.safe_div(aws["fouls"], aws["played"], 11.0)
        f["away_away_wr"] = self.safe_div(aws["away_wins"], aws["away_played"], 0.30)
        f["away_away_avg_gf"] = self.safe_div(aws["away_goals_for"], aws["away_played"], 1.1)
        f["away_away_avg_ga"] = self.safe_div(aws["away_goals_against"], aws["away_played"], 1.5)
        f["away_shot_accuracy"] = self.safe_div(aws["sot_for"], aws["shots_for"], 0.33)

        # EMA recent form and goals
        home_points = [3 if r == "W" else 1 if r == "D" else 0 for r in self.recent_window(hs["recent_results"])]
        away_points = [3 if r == "W" else 1 if r == "D" else 0 for r in self.recent_window(aws["recent_results"])]

        home_recent_gf = self.recent_window(hs["recent_gf"])
        home_recent_ga = self.recent_window(hs["recent_ga"])
        away_recent_gf = self.recent_window(aws["recent_gf"])
        away_recent_ga = self.recent_window(aws["recent_ga"])

        f["home_form_ema"] = self.ema(home_points, self.config.ema_alpha_form, 1.3)
        f["home_form_ema_short"] = self.ema(home_points, self.config.ema_alpha_short, 1.3)
        f["home_form_momentum"] = f["home_form_ema_short"] - f["home_form_ema"]
        f["home_recent_gf_ema"] = self.ema(home_recent_gf, self.config.ema_alpha_goals, 1.3)
        f["home_recent_ga_ema"] = self.ema(home_recent_ga, self.config.ema_alpha_goals, 1.3)

        f["away_form_ema"] = self.ema(away_points, self.config.ema_alpha_form, 1.3)
        f["away_form_ema_short"] = self.ema(away_points, self.config.ema_alpha_short, 1.3)
        f["away_form_momentum"] = f["away_form_ema_short"] - f["away_form_ema"]
        f["away_recent_gf_ema"] = self.ema(away_recent_gf, self.config.ema_alpha_goals, 1.3)
        f["away_recent_ga_ema"] = self.ema(away_recent_ga, self.config.ema_alpha_goals, 1.3)

        # Relative strength
        f["win_rate_diff"] = f["home_win_rate"] - f["away_win_rate"]
        f["gd_diff"] = f["home_gd_avg"] - f["away_gd_avg"]
        f["gf_diff"] = f["home_avg_gf"] - f["away_avg_gf"]
        f["sot_diff"] = f["home_avg_sot"] - f["away_avg_sot"]
        f["form_diff"] = f["home_form_ema"] - f["away_form_ema"]
        f["corners_diff"] = f["home_avg_corners"] - f["away_avg_corners"]
        f["attack_vs_defense"] = f["home_avg_gf"] - f["away_avg_ga"]
        f["defense_vs_attack"] = f["away_avg_gf"] - f["home_avg_ga"]

        # Head-to-head
        h2h = self.h2h_records[(home_name, away_name)]
        if h2h["matches"] > 0:
            f["h2h_home_wr"] = h2h["wins"] / h2h["matches"]
            f["h2h_draw_rate"] = h2h["draws"] / h2h["matches"]
            f["h2h_home_avg_gf"] = h2h["gf"] / h2h["matches"]
            f["h2h_home_avg_ga"] = h2h["ga"] / h2h["matches"]
        else:
            f["h2h_home_wr"] = 0.4
            f["h2h_draw_rate"] = 0.25
            f["h2h_home_avg_gf"] = 1.3
            f["h2h_home_avg_ga"] = 1.3

        # Rest days
        row_get = row.get if hasattr(row, "get") else lambda k, d=None: d
        match_date = row_get("parsed_date", None)
        home_rest = 7
        away_rest = 7
        if match_date is not None and pd.notna(match_date):
            if hs["last_match_date"] is not None and pd.notna(hs["last_match_date"]):
                home_rest = min((match_date - hs["last_match_date"]).days, 30)
            if aws["last_match_date"] is not None and pd.notna(aws["last_match_date"]):
                away_rest = min((match_date - aws["last_match_date"]).days, 30)
        f["home_rest_days"] = home_rest
        f["away_rest_days"] = away_rest
        f["rest_diff"] = home_rest - away_rest

        # FBref xG and possession
        home_xg = self.get_team_xg(home_name, season)
        away_xg = self.get_team_xg(away_name, season)
        home_poss = self.get_team_poss(home_name, season)
        away_poss = self.get_team_poss(away_name, season)

        f["home_xg_per90"] = home_xg["xg_per90"]
        f["home_xga_per90"] = home_xg["xga_per90"]
        f["home_xgd_per90"] = home_xg["xgd_per90"]
        f["home_xg_overperf"] = home_xg["xg_overperf"]

        f["away_xg_per90"] = away_xg["xg_per90"]
        f["away_xga_per90"] = away_xg["xga_per90"]
        f["away_xgd_per90"] = away_xg["xgd_per90"]
        f["away_xg_overperf"] = away_xg["xg_overperf"]

        f["xg_diff"] = home_xg["xg_per90"] - away_xg["xg_per90"]
        f["xga_diff"] = home_xg["xga_per90"] - away_xg["xga_per90"]
        f["xgd_diff"] = home_xg["xgd_per90"] - away_xg["xgd_per90"]
        f["home_xg_vs_away_xga"] = home_xg["xg_per90"] - away_xg["xga_per90"]
        f["away_xg_vs_home_xga"] = away_xg["xg_per90"] - home_xg["xga_per90"]

        f["home_possession"] = home_poss
        f["away_possession"] = away_poss
        f["possession_diff"] = home_poss - away_poss
        f["tactical_possession_edge"] = (home_poss - away_poss) * (f["home_shot_accuracy"] - f["away_shot_accuracy"])

        # Prevent temporal leakage: apply live FPL data only for current target season
        # (or interactive mode), and use neutral priors for historical seasons.
        def _season_start_year(season_label: Any) -> int:
            try:
                return int(str(season_label).split("/")[0])
            except Exception:
                return -1

        target_start = _season_start_year(self.config.target_test_season)
        season_start = _season_start_year(season)
        use_real_fpl = self.interactive_mode or (str(season) == str(self.config.target_test_season))
        if season_start >= 0 and target_start >= 0 and season_start < target_start:
            use_real_fpl = False

        if use_real_fpl:
            home_fpl = self.get_team_fpl_data(home_name)
            away_fpl = self.get_team_fpl_data(away_name)
        else:
            neutral_form = float(self.config.fpl_default_form)
            home_fpl = {"fpl_form_score": neutral_form, "key_players_missing": 0.0, "injury_pressure": 0.0}
            away_fpl = {"fpl_form_score": neutral_form, "key_players_missing": 0.0, "injury_pressure": 0.0}

        f["home_fpl_form_score"] = home_fpl["fpl_form_score"]
        f["away_fpl_form_score"] = away_fpl["fpl_form_score"]
        f["home_key_players_missing"] = home_fpl["key_players_missing"]
        f["away_key_players_missing"] = away_fpl["key_players_missing"]
        f["home_injury_pressure"] = home_fpl["injury_pressure"]
        f["away_injury_pressure"] = away_fpl["injury_pressure"]
        f["fpl_form_diff"] = home_fpl["fpl_form_score"] - away_fpl["fpl_form_score"]
        f["key_players_missing_diff"] = away_fpl["key_players_missing"] - home_fpl["key_players_missing"]
        f["injury_pressure_diff"] = away_fpl["injury_pressure"] - home_fpl["injury_pressure"]

        # EMA xG form (replaces simple rolling windows)
        home_recent_xg = self.recent_window(hs["recent_xg"])
        home_recent_xga = self.recent_window(hs["recent_xga"])
        away_recent_xg = self.recent_window(aws["recent_xg"])
        away_recent_xga = self.recent_window(aws["recent_xga"])

        home_xg_short = self.ema(home_recent_xg, self.config.ema_alpha_short, home_xg["xg_per90"])
        away_xg_short = self.ema(away_recent_xg, self.config.ema_alpha_short, away_xg["xg_per90"])
        home_xg_long = self.ema(home_recent_xg, self.config.ema_alpha_long, home_xg["xg_per90"])
        away_xg_long = self.ema(away_recent_xg, self.config.ema_alpha_long, away_xg["xg_per90"])

        f["home_xg_ema"] = self.ema(home_recent_xg, self.config.ema_alpha_xg, home_xg["xg_per90"])
        f["home_xga_ema"] = self.ema(home_recent_xga, self.config.ema_alpha_xg, home_xg["xga_per90"])
        f["home_xgd_ema"] = f["home_xg_ema"] - f["home_xga_ema"]

        f["away_xg_ema"] = self.ema(away_recent_xg, self.config.ema_alpha_xg, away_xg["xg_per90"])
        f["away_xga_ema"] = self.ema(away_recent_xga, self.config.ema_alpha_xg, away_xg["xga_per90"])
        f["away_xgd_ema"] = f["away_xg_ema"] - f["away_xga_ema"]

        f["home_xg_ema_trend"] = home_xg_short - home_xg_long
        f["away_xg_ema_trend"] = away_xg_short - away_xg_long
        f["xg_ema_diff"] = f["home_xg_ema"] - f["away_xg_ema"]

        # European features
        home_euro = self.get_euro_tier(home_name, season)
        away_euro = self.get_euro_tier(away_name, season)
        f["home_euro_tier"] = float(home_euro)
        f["away_euro_tier"] = float(away_euro)
        f["euro_tier_diff"] = float(home_euro - away_euro)
        f["home_in_europe"] = 1.0 if home_euro > 0 else 0.0
        f["away_in_europe"] = 1.0 if away_euro > 0 else 0.0
        f["both_in_europe"] = 1.0 if home_euro > 0 and away_euro > 0 else 0.0
        f["euro_fatigue_diff"] = (home_euro * hs["played"] / 38.0) - (away_euro * aws["played"] / 38.0)

        # Betting odds
        b365h = row_get("B365H", np.nan)
        b365d = row_get("B365D", np.nan)
        b365a = row_get("B365A", np.nan)

        if pd.notna(b365h) and pd.notna(b365d) and pd.notna(b365a) and b365h > 0 and b365d > 0 and b365a > 0:
            f["B365H"] = float(b365h)
            f["B365D"] = float(b365d)
            f["B365A"] = float(b365a)
            total = 1 / b365h + 1 / b365d + 1 / b365a
            f["implied_prob_H"] = (1 / b365h) / total
            f["implied_prob_D"] = (1 / b365d) / total
            f["implied_prob_A"] = (1 / b365a) / total
            f["odds_home_fav"] = 1.0 if b365h < b365a else 0.0
        else:
            f["B365H"] = float(self.odds_defaults["B365H"])
            f["B365D"] = float(self.odds_defaults["B365D"])
            f["B365A"] = float(self.odds_defaults["B365A"])
            f["implied_prob_H"] = float(self.implied_defaults["H"])
            f["implied_prob_D"] = float(self.implied_defaults["D"])
            f["implied_prob_A"] = float(self.implied_defaults["A"])
            f["odds_home_fav"] = 1.0 if f["B365H"] < f["B365A"] else 0.0

        # Opening vs closing odds movement (kept neutral unless enabled + real closing lines)
        if self.use_odds_movement_feature_runtime:
            f["odds_movement"] = self.compute_odds_movement(row_get, home_name, away_name, match_date)
        else:
            f["odds_movement"] = 0.0

        return f

    def update_stats(
        self,
        stats: Dict[str, Any],
        goals_for: int,
        goals_against: int,
        result_char: str,
        is_home: bool,
        shots: float,
        sot: float,
        corners: float,
        fouls: float,
        yellows: float,
        reds: float,
        opp_shots: float,
        opp_sot: float,
        opp_corners: float,
        opp_fouls: float,
        match_date: Optional[datetime],
        match_xg: Optional[float],
        match_xga: Optional[float],
    ) -> None:
        stats["played"] += 1
        stats["goals_for"] += goals_for
        stats["goals_against"] += goals_against
        stats["shots_for"] += shots
        stats["shots_against"] += opp_shots
        stats["sot_for"] += sot
        stats["sot_against"] += opp_sot
        stats["corners_for"] += corners
        stats["corners_against"] += opp_corners
        stats["fouls"] += fouls
        stats["fouls_against"] += opp_fouls
        stats["yellows"] += yellows
        stats["reds"] += reds

        if is_home:
            stats["home_played"] += 1
            stats["home_goals_for"] += goals_for
            stats["home_goals_against"] += goals_against
        else:
            stats["away_played"] += 1
            stats["away_goals_for"] += goals_for
            stats["away_goals_against"] += goals_against

        if result_char == "W":
            stats["wins"] += 1
            if is_home:
                stats["home_wins"] += 1
            else:
                stats["away_wins"] += 1
        elif result_char == "D":
            stats["draws"] += 1
        else:
            stats["losses"] += 1

        stats["recent_results"].append(result_char)
        stats["recent_gf"].append(float(goals_for))
        stats["recent_ga"].append(float(goals_against))

        if match_xg is not None:
            stats["recent_xg"].append(float(match_xg))
        if match_xga is not None:
            stats["recent_xga"].append(float(match_xga))

        for key in ["recent_results", "recent_gf", "recent_ga", "recent_xg", "recent_xga"]:
            if len(stats[key]) > 20:
                stats[key] = stats[key][-20:]

        if match_date is not None and pd.notna(match_date):
            stats["last_match_date"] = match_date

    def _build_feature_columns(self) -> List[str]:
        cols = [
            "home_elo", "away_elo", "elo_diff", "elo_expected_home",
            "home_win_rate", "home_draw_rate", "home_loss_rate",
            "home_avg_gf", "home_avg_ga", "home_gd_avg",
            "home_avg_shots", "home_avg_sot", "home_avg_corners", "home_avg_fouls",
            "home_home_wr", "home_home_avg_gf", "home_home_avg_ga", "home_shot_accuracy",
            "away_win_rate", "away_draw_rate", "away_loss_rate",
            "away_avg_gf", "away_avg_ga", "away_gd_avg",
            "away_avg_shots", "away_avg_sot", "away_avg_corners", "away_avg_fouls",
            "away_away_wr", "away_away_avg_gf", "away_away_avg_ga", "away_shot_accuracy",
            "home_form_ema", "home_form_ema_short", "home_form_momentum",
            "home_recent_gf_ema", "home_recent_ga_ema",
            "away_form_ema", "away_form_ema_short", "away_form_momentum",
            "away_recent_gf_ema", "away_recent_ga_ema",
            "win_rate_diff", "gd_diff", "gf_diff", "sot_diff", "form_diff", "corners_diff",
            "attack_vs_defense", "defense_vs_attack",
            "h2h_home_wr", "h2h_draw_rate", "h2h_home_avg_gf", "h2h_home_avg_ga",
            "home_rest_days", "away_rest_days", "rest_diff",
            "home_xg_per90", "home_xga_per90", "home_xgd_per90", "home_xg_overperf",
            "away_xg_per90", "away_xga_per90", "away_xgd_per90", "away_xg_overperf",
            "xg_diff", "xga_diff", "xgd_diff", "home_xg_vs_away_xga", "away_xg_vs_home_xga",
            "home_possession", "away_possession", "possession_diff", "tactical_possession_edge",
            "home_fpl_form_score", "away_fpl_form_score",
            "home_key_players_missing", "away_key_players_missing",
            "home_injury_pressure", "away_injury_pressure",
            "fpl_form_diff", "key_players_missing_diff", "injury_pressure_diff",
            "home_xg_ema", "home_xga_ema", "home_xgd_ema",
            "away_xg_ema", "away_xga_ema", "away_xgd_ema",
            "home_xg_ema_trend", "away_xg_ema_trend", "xg_ema_diff",
            "home_euro_tier", "away_euro_tier", "euro_tier_diff",
            "home_in_europe", "away_in_europe", "both_in_europe", "euro_fatigue_diff",
        ]
        if self.use_bookmaker_features_runtime:
            cols.extend(
                [
                    "B365H",
                    "B365D",
                    "B365A",
                    "implied_prob_H",
                    "implied_prob_D",
                    "implied_prob_A",
                    "odds_home_fav",
                ]
            )
        if self.use_odds_movement_feature_runtime:
            cols.append("odds_movement")
        return cols

    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        self.all_teams = sorted(df["HomeTeam"].unique())
        self.team_stats = {t: self.make_empty_stats() for t in self.all_teams}
        self.elo_ratings = defaultdict(lambda: self.ELO_START)
        self.h2h_records = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "matches": 0})
        self._fit_odds_priors(df)
        self._fit_xg_proxy_model(df)

        feature_rows: List[Dict[str, Any]] = []
        current_season = None
        total_rows = len(df)

        for idx, row in df.iterrows():
            if self.interactive_mode and idx % 500 == 0:
                show_progress(25 + int(25 * idx / max(total_rows, 1)), "Engineering features...", self.interactive_mode)

            season = row["season"]
            if season != current_season:
                if current_season is not None:
                    for team in list(self.team_stats.keys()):
                        self.team_stats[team] = self.decay_stats(self.team_stats[team])
                    mean_elo = np.mean([self.elo_ratings[t] for t in self.elo_ratings]) if self.elo_ratings else self.ELO_START
                    for team in self.all_teams:
                        self.elo_ratings[team] = (
                            self.elo_ratings[team] * (1 - self.ELO_SEASON_REGRESS)
                            + mean_elo * self.ELO_SEASON_REGRESS
                        )
                current_season = season

            home = row["HomeTeam"]
            away = row["AwayTeam"]

            if home not in self.team_stats:
                self.team_stats[home] = self.make_empty_stats()
            if away not in self.team_stats:
                self.team_stats[away] = self.make_empty_stats()

            hs = self.team_stats[home]
            aws = self.team_stats[away]

            features = self.extract_features(hs, aws, row, home, away, season)
            features["target"] = row["FTR"]
            features["home_goals"] = int(row["FTHG"])
            features["away_goals"] = int(row["FTAG"])
            features["home_team"] = home
            features["away_team"] = away
            features["season"] = season
            features["parsed_date"] = row.get("parsed_date", None)
            feature_rows.append(features)

            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])
            ftr = row["FTR"]
            match_date = row.get("parsed_date", None)

            h_shots = float(row.get("HS", 0) if pd.notna(row.get("HS", np.nan)) else 0)
            a_shots = float(row.get("AS", 0) if pd.notna(row.get("AS", np.nan)) else 0)
            h_sot = float(row.get("HST", 0) if pd.notna(row.get("HST", np.nan)) else 0)
            a_sot = float(row.get("AST", 0) if pd.notna(row.get("AST", np.nan)) else 0)
            h_corners = float(row.get("HC", 0) if pd.notna(row.get("HC", np.nan)) else 0)
            a_corners = float(row.get("AC", 0) if pd.notna(row.get("AC", np.nan)) else 0)
            h_fouls = float(row.get("HF", 0) if pd.notna(row.get("HF", np.nan)) else 0)
            a_fouls = float(row.get("AF", 0) if pd.notna(row.get("AF", np.nan)) else 0)
            h_yellows = float(row.get("HY", 0) if pd.notna(row.get("HY", np.nan)) else 0)
            a_yellows = float(row.get("AY", 0) if pd.notna(row.get("AY", np.nan)) else 0)
            h_reds = float(row.get("HR", 0) if pd.notna(row.get("HR", np.nan)) else 0)
            a_reds = float(row.get("AR", 0) if pd.notna(row.get("AR", np.nan)) else 0)

            home_result = "W" if ftr == "H" else "D" if ftr == "D" else "L"
            away_result = "W" if ftr == "A" else "D" if ftr == "D" else "L"

            key = (season, home, away)
            if key in self.match_xg_lookup:
                h_match_xg, a_match_xg = self.match_xg_lookup[key]
            else:
                home_xg_base = self.get_team_xg(home, season)["xg_per90"]
                away_xg_base = self.get_team_xg(away, season)["xg_per90"]
                h_match_xg = self._estimate_match_xg(h_shots, h_sot, home_xg_base)
                a_match_xg = self._estimate_match_xg(a_shots, a_sot, away_xg_base)

            self.update_stats(
                hs,
                fthg,
                ftag,
                home_result,
                True,
                h_shots,
                h_sot,
                h_corners,
                h_fouls,
                h_yellows,
                h_reds,
                a_shots,
                a_sot,
                a_corners,
                a_fouls,
                match_date,
                h_match_xg,
                a_match_xg,
            )
            self.update_stats(
                aws,
                ftag,
                fthg,
                away_result,
                False,
                a_shots,
                a_sot,
                a_corners,
                a_fouls,
                a_yellows,
                a_reds,
                h_shots,
                h_sot,
                h_corners,
                h_fouls,
                match_date,
                a_match_xg,
                h_match_xg,
            )

            home_elo = self.elo_ratings[home]
            away_elo = self.elo_ratings[away]
            if ftr == "H":
                score_h, score_a = 1.0, 0.0
            elif ftr == "D":
                score_h, score_a = 0.5, 0.5
            else:
                score_h, score_a = 0.0, 1.0

            gd = abs(fthg - ftag)
            k_mult = 1.0 + 0.1 * min(gd, 3)
            self.elo_ratings[home] = self.elo_update(home_elo + self.ELO_HOME, away_elo, score_h, self.ELO_K * k_mult) - self.ELO_HOME
            self.elo_ratings[away] = self.elo_update(away_elo, home_elo + self.ELO_HOME, score_a, self.ELO_K * k_mult)

            h2h_key = (home, away)
            self.h2h_records[h2h_key]["matches"] += 1
            self.h2h_records[h2h_key]["gf"] += fthg
            self.h2h_records[h2h_key]["ga"] += ftag
            if ftr == "H":
                self.h2h_records[h2h_key]["wins"] += 1
            elif ftr == "D":
                self.h2h_records[h2h_key]["draws"] += 1
            else:
                self.h2h_records[h2h_key]["losses"] += 1

        feat_df = pd.DataFrame(feature_rows)
        warmup = min(10, max(0, len(feat_df) // 20))
        if warmup > 0:
            feat_df = feat_df.iloc[warmup:].reset_index(drop=True)

        self.feature_cols = self._build_feature_columns()
        self.euro_indices = [self.feature_cols.index(col) for col in self.euro_feature_names]
        self.current_teams = sorted(df[df["season"] == self.config.target_test_season]["HomeTeam"].unique())

        return feat_df
