"""Inference and interactive prediction services."""

from collections import defaultdict
from datetime import datetime
import logging
import shutil
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import AppConfig, EURO_TIER_NAMES
from .features import FeatureEngineer
from .models import BettingOptimizer, ModelTrainer

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(
        self,
        config: AppConfig,
        feature_engineer: FeatureEngineer,
        trainer: ModelTrainer,
        label_encoder: LabelEncoder,
        quiet: bool,
    ):
        self.config = config
        self.fe = feature_engineer
        self.trainer = trainer
        self.le = label_encoder
        self.quiet = quiet

    def find_team(self, name: str) -> Optional[str]:
        name_clean = name.lower().replace(" ", "").replace("'", "")
        for team in self.fe.current_teams:
            if team.lower().replace(" ", "").replace("'", "") == name_clean:
                return team
        for team in self.fe.current_teams:
            t_clean = team.lower().replace(" ", "").replace("'", "")
            if name_clean in t_clean or t_clean in name_clean:
                return team
        for team in self.fe.all_teams:
            t_clean = team.lower().replace(" ", "").replace("'", "")
            if name_clean in t_clean or t_clean in name_clean:
                return team
        return None

    def build_prediction_features(
        self, home_team: str, away_team: str, return_feature_map: bool = False
    ) -> Optional[Any]:
        hs = self.fe.team_stats.get(home_team)
        aws = self.fe.team_stats.get(away_team)
        if hs is None or aws is None:
            return None

        dummy_row = {"B365H": np.nan, "B365D": np.nan, "B365A": np.nan, "parsed_date": datetime.now()}
        features = self.fe.extract_features(hs, aws, dummy_row, home_team, away_team, self.config.target_test_season)
        # Align prediction vector with the trained model schema (cached or freshly trained).
        expected_cols = self.fe.feature_cols
        if self.trainer.full_feature_count is not None and self.trainer.full_feature_count != len(self.fe.feature_cols):
            expected_cols = self.trainer.feature_cols if self.trainer.feature_cols else expected_cols

        X_pred = np.array([[float(features.get(c, 0.0)) for c in expected_cols]], dtype=float)
        euro_indices = [idx for idx, col in enumerate(expected_cols) if col in self.fe.euro_feature_names]
        if euro_indices:
            X_pred[:, euro_indices] *= self.config.euro_weight
        if return_feature_map:
            return X_pred, features
        return X_pred

    @staticmethod
    def _form_str(results: List[str]) -> str:
        if not results:
            return "-----"
        return "".join("W" if r == "W" else "D" if r == "D" else "L" for r in results[-5:])

    @staticmethod
    def _result_points(result_char: str) -> int:
        if result_char == "W":
            return 3
        if result_char == "D":
            return 1
        return 0

    def _champion_reasons(
        self,
        champion: str,
        standings: List[str],
        proj_pts: Dict[str, float],
        champ_prob: Dict[str, float],
        season: str,
        n_sim: int,
    ) -> List[str]:
        runner_up = standings[1] if len(standings) > 1 else None
        if runner_up is not None:
            gap = proj_pts.get(champion, 0.0) - proj_pts.get(runner_up, 0.0)
            reason_1 = (
                f"Highest projected points ({proj_pts.get(champion, 0.0):.1f}), "
                f"{gap:.1f} clear of {runner_up}."
            )
        else:
            reason_1 = f"Highest projected points in the league ({proj_pts.get(champion, 0.0):.1f})."

        if n_sim > 0:
            reason_2 = (
                f"Best title probability in simulation ({champ_prob.get(champion, 0.0):.1%}) "
                f"across {n_sim} Monte Carlo seasons."
            )
        else:
            reason_2 = "Top ranked by expected-points projection from remaining fixtures."

        team_xg = self.fe.get_team_xg(champion, season)
        elo = float(self.fe.elo_ratings.get(champion, self.fe.ELO_START))
        recent = self.fe.team_stats.get(champion, {}).get("recent_results", [])
        form_pts = sum(self._result_points(r) for r in recent[-5:])
        reason_3 = (
            f"Elite strength profile: Elo {elo:.0f}, xGD/90 {team_xg['xgd_per90']:+.2f}, "
            f"recent form {form_pts}/15."
        )

        return [reason_1, reason_2, reason_3]

    @staticmethod
    def _truncate_text(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."

    @staticmethod
    def _log_wrapped_line(prefix: str, text: str, max_width: int) -> None:
        width = max(20, max_width - len(prefix))
        chunks = textwrap.wrap(text, width=width) or [""]
        logger.info(f"{prefix}{chunks[0]}")
        continuation_prefix = " " * len(prefix)
        for chunk in chunks[1:]:
            logger.info(f"{continuation_prefix}{chunk}")

    def predict_match(self, home_team: str, away_team: str) -> None:
        home = self.find_team(home_team)
        away = self.find_team(away_team)

        if home is None:
            logger.warning("\n  Team '%s' not found.", home_team)
            logger.info("  Teams: %s", ", ".join(self.fe.current_teams))
            return
        if away is None:
            logger.warning("\n  Team '%s' not found.", away_team)
            logger.info("  Teams: %s", ", ".join(self.fe.current_teams))
            return
        if home == away:
            logger.warning("\n  A team cannot play against itself!")
            return

        pred_pack = self.build_prediction_features(home, away, return_feature_map=True)
        X_pred = pred_pack[0] if pred_pack is not None else None
        pred_features = pred_pack[1] if pred_pack is not None else None
        if X_pred is None:
            logger.error("  Could not build prediction features.")
            return

        proba = self.trainer.predict_proba(X_pred, self.le)[0]
        pred_idx = self.trainer.predict(X_pred, self.le, use_thresholds=True)[0]
        pred_class = self.le.inverse_transform([pred_idx])[0]
        prob_dict = dict(zip(self.le.classes_, proba))

        exp_home_goals, exp_away_goals = self.trainer.predict_expected_goals(X_pred)
        exp_home_goals = float(exp_home_goals[0])
        exp_away_goals = float(exp_away_goals[0])

        if pred_features is None:
            pred_features = {}
        odds_dict = {
            "H": float(pred_features.get("B365H", np.nan)),
            "D": float(pred_features.get("B365D", np.nan)),
            "A": float(pred_features.get("B365A", np.nan)),
        }
        betting = BettingOptimizer.evaluate_market(
            prob_dict,
            odds_dict,
            kelly_cap=self.config.kelly_cap_fraction,
            kelly_scale=self.config.kelly_fraction_scale,
        )

        result_map = {"H": f"{home} Win", "D": "Draw", "A": f"{away} Win"}

        hs = self.fe.team_stats[home]
        aws = self.fe.team_stats[away]
        home_xg = self.fe.get_team_xg(home, self.config.target_test_season)
        away_xg = self.fe.get_team_xg(away, self.config.target_test_season)

        home_xg_ema = self.fe.ema(self.fe.recent_window(hs["recent_xg"]), self.config.ema_alpha_xg, home_xg["xg_per90"])
        home_xga_ema = self.fe.ema(self.fe.recent_window(hs["recent_xga"]), self.config.ema_alpha_xg, home_xg["xga_per90"])
        away_xg_ema = self.fe.ema(self.fe.recent_window(aws["recent_xg"]), self.config.ema_alpha_xg, away_xg["xg_per90"])
        away_xga_ema = self.fe.ema(self.fe.recent_window(aws["recent_xga"]), self.config.ema_alpha_xg, away_xg["xga_per90"])

        cls_order = ", ".join(f"{c}:{self.trainer.thresholds[i]:.3f}" for i, c in enumerate(self.le.classes_))

        logger.info("")
        logger.info("=" * 64)
        logger.info(f"  {home} (Home)  vs  {away} (Away)")
        logger.info("=" * 64)
        logger.info(f"\n  {'Prediction (thresholded):':<30s}{result_map[pred_class]}")
        logger.info(f"  {'Expected Score (Dixon-Coles):':<30s}{exp_home_goals:.1f} - {exp_away_goals:.1f}")
        logger.info(f"  {'Class thresholds:':<30s}{cls_order}")
        logger.info("")
        logger.info(f"  {'Home Win:':<22s} {prob_dict.get('H', 0):>6.1%}  {'|' * int(prob_dict.get('H', 0) * 40)}")
        logger.info(f"  {'Draw:':<22s} {prob_dict.get('D', 0):>6.1%}  {'|' * int(prob_dict.get('D', 0) * 40)}")
        logger.info(f"  {'Away Win:':<22s} {prob_dict.get('A', 0):>6.1%}  {'|' * int(prob_dict.get('A', 0) * 40)}")

        logger.info("\n  --- Betting Edge (EV + Kelly) ---")
        logger.info(f"  {'Outcome':<10s} {'Odds':>7s} {'ModelP':>8s} {'EV':>8s} {'Kelly':>8s}")
        for outcome in ["H", "D", "A"]:
            item = betting[outcome]
            ev_text = f"{item['ev']:+.3f}"
            kelly_text = f"{item['kelly']*100:>6.2f}%"
            logger.info(
                f"  {outcome:<10s} {item['odds']:>7.2f} {item['prob']:>7.1%} {ev_text:>8s} {kelly_text:>8s}"
            )
        plus_ev = [(k, v) for k, v in betting.items() if v["plus_ev"] > 0]
        if plus_ev:
            best_outcome, best_item = max(plus_ev, key=lambda kv: kv[1]["ev"])
            logger.info(
                f"  Best +EV: {best_outcome} | EV {best_item['ev']:+.3f} | "
                f"Kelly stake {best_item['kelly']*100:.2f}% bankroll"
            )
        else:
            logger.info("  No +EV outcome under current odds.")

        logger.info("\n  --- Elo Ratings ---")
        logger.info(f"  {home:<20s} {self.fe.elo_ratings[home]:>7.1f}")
        logger.info(f"  {away:<20s} {self.fe.elo_ratings[away]:>7.1f}")

        logger.info("\n  --- xG Profile (Season) ---")
        logger.info(
            f"  {home:<20s} xG:{home_xg['xg_per90']:.2f} xGA:{home_xg['xga_per90']:.2f} "
            f"xGD:{home_xg['xgd_per90']:+.2f} Overperf:{home_xg['xg_overperf']:+.2f}"
        )
        logger.info(
            f"  {away:<20s} xG:{away_xg['xg_per90']:.2f} xGA:{away_xg['xga_per90']:.2f} "
            f"xGD:{away_xg['xgd_per90']:+.2f} Overperf:{away_xg['xg_overperf']:+.2f}"
        )

        home_fpl = self.fe.get_team_fpl_data(home)
        away_fpl = self.fe.get_team_fpl_data(away)
        logger.info("\n  --- FPL Squad Health ---")
        logger.info(
            f"  {home:<20s} Form:{home_fpl['fpl_form_score']:.2f} "
            f"KeyMissing:{home_fpl['key_players_missing']:.0f} InjuryPressure:{home_fpl['injury_pressure']:.2f}"
        )
        logger.info(
            f"  {away:<20s} Form:{away_fpl['fpl_form_score']:.2f} "
            f"KeyMissing:{away_fpl['key_players_missing']:.0f} InjuryPressure:{away_fpl['injury_pressure']:.2f}"
        )

        logger.info("\n  --- EMA xG Form ---")
        logger.info(f"  {home:<20s} xG_EMA:{home_xg_ema:.2f} xGA_EMA:{home_xga_ema:.2f} xGD_EMA:{home_xg_ema-home_xga_ema:+.2f}")
        logger.info(f"  {away:<20s} xG_EMA:{away_xg_ema:.2f} xGA_EMA:{away_xga_ema:.2f} xGD_EMA:{away_xg_ema-away_xga_ema:+.2f}")

        home_euro = self.fe.get_euro_tier(home, self.config.target_test_season)
        away_euro = self.fe.get_euro_tier(away, self.config.target_test_season)
        if home_euro > 0 or away_euro > 0:
            logger.info("\n  --- European Competition ---")
            logger.info(f"  {home:<20s} {EURO_TIER_NAMES[home_euro]}")
            logger.info(f"  {away:<20s} {EURO_TIER_NAMES[away_euro]}")

        logger.info("\n  --- Season Form ---")
        logger.info(
            f"  {home:<20s} W:{int(round(hs['wins'])):>2d} D:{int(round(hs['draws'])):>2d} L:{int(round(hs['losses'])):>2d} "
            f"Form:{self._form_str(hs['recent_results'])}"
        )
        logger.info(
            f"  {away:<20s} W:{int(round(aws['wins'])):>2d} D:{int(round(aws['draws'])):>2d} L:{int(round(aws['losses'])):>2d} "
            f"Form:{self._form_str(aws['recent_results'])}"
        )

        h2h = self.fe.h2h_records[(home, away)]
        if h2h["matches"] > 0:
            logger.info(f"\n  --- Head-to-Head at {home} ---")
            logger.info(
                f"  {int(h2h['matches'])} matches: {home} wins {int(h2h['wins'])}, "
                f"Draws {int(h2h['draws'])}, {away} wins {int(h2h['losses'])}"
            )

        logger.info("=" * 64)


    def predict_season_standings(self, feat_df: pd.DataFrame, explain: bool = False) -> None:
        """Project final standings for the current season by simulating remaining fixtures."""
        season = self.config.target_test_season
        teams = sorted(self.fe.current_teams)

        # Accumulate actual results from matches already played this season.
        pts: Dict[str, float] = defaultdict(float)
        gf: Dict[str, float] = defaultdict(float)
        ga: Dict[str, float] = defaultdict(float)
        played: Dict[str, int] = defaultdict(int)
        played_fixtures: set = set()

        for _, row in feat_df[feat_df["season"] == season].iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            ftr = str(row["target"])
            hg, ag = float(row["home_goals"]), float(row["away_goals"])

            played[home] += 1
            played[away] += 1
            gf[home] += hg
            ga[home] += ag
            gf[away] += ag
            ga[away] += hg

            if ftr == "H":
                pts[home] += 3
            elif ftr == "D":
                pts[home] += 1
                pts[away] += 1
            else:
                pts[away] += 3

            played_fixtures.add((home, away))

        # Build remaining fixture probability list once.
        remaining_fixtures: List[Dict[str, Any]] = []
        for home in teams:
            for away in teams:
                if home == away or (home, away) in played_fixtures:
                    continue
                X_pred = self.build_prediction_features(home, away)
                if X_pred is None:
                    continue
                proba = self.trainer.predict_proba(X_pred, self.le)[0]
                prob_dict = dict(zip(self.le.classes_, proba))
                lam, mu = self.trainer.predict_expected_goals(X_pred)
                remaining_fixtures.append(
                    {
                        "home": home,
                        "away": away,
                        "p": np.array([prob_dict.get("H", 0.0), prob_dict.get("D", 0.0), prob_dict.get("A", 0.0)], dtype=float),
                        "lam": float(lam[0]),
                        "mu": float(mu[0]),
                    }
                )

        remaining = len(remaining_fixtures)
        n_sim = max(int(getattr(self.config, "standings_simulations", 0)), 0)
        rng = np.random.default_rng(int(getattr(self.config, "standings_random_seed", 42)))

        if n_sim <= 0 or remaining == 0:
            # Fallback: expected points method.
            proj_pts: Dict[str, float] = dict(pts)
            proj_gf: Dict[str, float] = dict(gf)
            proj_ga: Dict[str, float] = dict(ga)

            for fx in remaining_fixtures:
                ph, pd_prob, pa = fx["p"]
                home = fx["home"]
                away = fx["away"]
                proj_pts[home] = proj_pts.get(home, 0.0) + ph * 3 + pd_prob
                proj_pts[away] = proj_pts.get(away, 0.0) + pa * 3 + pd_prob
                proj_gf[home] = proj_gf.get(home, 0.0) + fx["lam"]
                proj_ga[home] = proj_ga.get(home, 0.0) + fx["mu"]
                proj_gf[away] = proj_gf.get(away, 0.0) + fx["mu"]
                proj_ga[away] = proj_ga.get(away, 0.0) + fx["lam"]

            champ_prob = {t: 0.0 for t in teams}
            standings = sorted(teams, key=lambda t: (-proj_pts.get(t, 0.0), -(proj_gf.get(t, 0.0) - proj_ga.get(t, 0.0))))
            champion = standings[0]
            champ_prob[champion] = 1.0
        else:
            # Monte Carlo season simulation for realistic variance in projections.
            pts_sum = defaultdict(float)
            gf_sum = defaultdict(float)
            ga_sum = defaultdict(float)
            champ_counts = defaultdict(int)

            for _ in range(n_sim):
                sim_pts = defaultdict(float, pts)
                sim_gf = defaultdict(float, gf)
                sim_ga = defaultdict(float, ga)

                for fx in remaining_fixtures:
                    home = fx["home"]
                    away = fx["away"]
                    p = fx["p"]
                    p_sum = float(p.sum())
                    if p_sum <= 0:
                        p = np.array([0.40, 0.25, 0.35], dtype=float)
                        p_sum = 1.0
                    p = p / p_sum

                    outcome = int(rng.choice(3, p=p))
                    if outcome == 0:
                        sim_pts[home] += 3.0
                    elif outcome == 1:
                        sim_pts[home] += 1.0
                        sim_pts[away] += 1.0
                    else:
                        sim_pts[away] += 3.0

                    hg = int(rng.poisson(max(fx["lam"], 0.05)))
                    ag = int(rng.poisson(max(fx["mu"], 0.05)))
                    sim_gf[home] += hg
                    sim_ga[home] += ag
                    sim_gf[away] += ag
                    sim_ga[away] += hg

                sim_standings = sorted(
                    teams,
                    key=lambda t: (-sim_pts[t], -((sim_gf[t] - sim_ga[t])), -sim_gf[t]),
                )
                champ_counts[sim_standings[0]] += 1
                for t in teams:
                    pts_sum[t] += sim_pts[t]
                    gf_sum[t] += sim_gf[t]
                    ga_sum[t] += sim_ga[t]

            proj_pts = {t: pts_sum[t] / n_sim for t in teams}
            proj_gf = {t: gf_sum[t] / n_sim for t in teams}
            proj_ga = {t: ga_sum[t] / n_sim for t in teams}
            champ_prob = {t: champ_counts[t] / n_sim for t in teams}
            standings = sorted(teams, key=lambda t: (-proj_pts[t], -((proj_gf[t] - proj_ga[t]))))
            champion = standings[0]

        matches_played = sum(played.values()) // 2

        term_width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns)
        line_width = min(74, term_width)
        sep = "=" * line_width
        dash = "-" * line_width

        logger.info("")
        logger.info(sep)
        title = self._truncate_text(f"PREDICTED FINAL STANDINGS - {season} Premier League", line_width - 2)
        logger.info(f"  {title}")
        if n_sim > 0:
            meta = f"{matches_played} played | {remaining} remaining | {n_sim} sims"
        else:
            meta = f"{matches_played} played | {remaining} remaining | expected-points mode"
        logger.info(f"  {self._truncate_text(meta, line_width - 2)}")
        logger.info(sep)

        full_mode = line_width >= 64
        compact_mode = 48 <= line_width < 64
        if full_mode:
            logger.info(f"  {'#':>2s}  {'Team':<20s} {'MP':>3s}  {'Pts':>4s}  {'Proj Pts':>8s}  {'Proj GD':>8s}  {'Champ%':>7s}")
        elif compact_mode:
            logger.info(f"  {'#':>2s}  {'Team':<16s} {'ProjPts':>7s}  {'Champ%':>7s}")
        else:
            logger.info(f"  {'#':>2s} {'Team':<10s} {'Pts':>5s} {'C%':>5s}")
        logger.info(dash)
        for rank, team in enumerate(standings, 1):
            cur_pts = int(pts.get(team, 0))
            cur_played = played.get(team, 0)
            p_pts = proj_pts.get(team, 0.0)
            p_gd = proj_gf.get(team, 0.0) - proj_ga.get(team, 0.0)
            c_prob = champ_prob.get(team, 0.0)
            if full_mode:
                team_name = self._truncate_text(team, 20)
                logger.info(
                    f"  {rank:>2d}  {team_name:<20s} {cur_played:>3d}  {cur_pts:>4d}  "
                    f"{p_pts:>8.1f}  {p_gd:>+8.1f}  {c_prob:>6.1%}"
                )
            elif compact_mode:
                team_name = self._truncate_text(team, 16)
                logger.info(f"  {rank:>2d}  {team_name:<16s} {p_pts:>7.1f}  {c_prob:>6.1%}")
            else:
                team_name = self._truncate_text(team, 10)
                logger.info(f"  {rank:>2d} {team_name:<10s} {p_pts:>5.1f} {c_prob:>5.1%}")
        logger.info(sep)
        logger.info("")
        self._log_wrapped_line("  ", f"Predicted {season} Premier League Champion: {champion}", line_width)
        self._log_wrapped_line("  ", f"Projected Points: {proj_pts[champion]:.1f}", line_width)
        if explain:
            logger.info("")
            self._log_wrapped_line("  ", "Why this team is the champion favorite:", line_width)
            reasons = self._champion_reasons(
                champion=champion,
                standings=standings,
                proj_pts=proj_pts,
                champ_prob=champ_prob,
                season=season,
                n_sim=n_sim,
            )
            for idx, reason in enumerate(reasons, 1):
                self._log_wrapped_line(f"  {idx}. ", reason, line_width)
        logger.info(sep)
