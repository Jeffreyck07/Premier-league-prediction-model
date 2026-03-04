"""Data ingestion and external API access layer."""

from collections import defaultdict
import json
import logging
import os
import time
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import AppConfig, map_fbref_name, map_fpl_name, parse_date

logger = logging.getLogger(__name__)


class DataLoader:
    COMMON_COLS = [
        "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A",
        # Optional closing-line columns (pipeline-ready for CLV ingestion)
        "B365CH", "B365CD", "B365CA",
        "PSCH", "PSCD", "PSCA",
        "AvgCH", "AvgCD", "AvgCA",
        "OpenH", "OpenD", "OpenA",
        "CloseH", "CloseD", "CloseA",
    ]

    def __init__(self, config: AppConfig, quiet: bool, force_scrape: bool, interactive_mode: bool):
        self.config = config
        self.quiet = quiet
        self.force_scrape = force_scrape
        self.interactive_mode = interactive_mode
        # Reserved for future paid odds feeds; keep secret in environment only.
        self.odds_api_key = os.environ.get("ODDS_API_KEY")
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        os.makedirs(self.config.fbref_cache_dir, exist_ok=True)

    def scrape_fbref_season(self, url: str, season_label: str) -> Optional[pd.DataFrame]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        if not self.quiet:
            logger.info(f"  Scraping FBref for {season_label}...")

        try:
            response = self.session.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                if not self.quiet:
                    logger.warning("    Failed (HTTP %s)", response.status_code)
                return None

            tables = pd.read_html(StringIO(response.text))
            if not tables:
                return None

            df_standings = tables[0]
            if isinstance(df_standings.columns, pd.MultiIndex):
                df_standings.columns = df_standings.columns.droplevel()

            target_cols = ["Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "xG", "xGA", "xGD"]
            available = [c for c in target_cols if c in df_standings.columns]
            if "xG" not in available:
                if not self.quiet:
                    logger.warning("    No xG data found in standings table")
                return None

            df_clean = df_standings[available].copy()
            df_clean = df_clean.dropna(subset=["Squad", "MP"])
            df_clean = df_clean[df_clean["Squad"].astype(str).str.len() > 2]

            for col in ["MP", "xG", "xGA", "xGD", "GF", "GA"]:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

            df_clean = df_clean[df_clean["MP"] > 0]
            df_clean["xg_per90"] = df_clean["xG"] / df_clean["MP"]
            df_clean["xga_per90"] = df_clean["xGA"] / df_clean["MP"]
            df_clean["xgd_per90"] = df_clean["xGD"] / df_clean["MP"]
            df_clean["gf_per90"] = df_clean["GF"] / df_clean["MP"]
            df_clean["ga_per90"] = df_clean["GA"] / df_clean["MP"]
            df_clean["xg_overperf"] = df_clean["gf_per90"] - df_clean["xg_per90"]
            df_clean["fd_name"] = df_clean["Squad"].apply(map_fbref_name)
            df_clean["season"] = season_label

            if not self.quiet:
                logger.info(f"    Got xG data for {len(df_clean)} teams")

            return df_clean
        except Exception as exc:
            if not self.quiet:
                logger.error("    Error scraping season xG: %s", exc)
            return None

    def scrape_fbref_match_xg(self, season_label: str) -> List[Dict[str, Any]]:
        parts = season_label.split("/")
        y1 = int(parts[0])
        y2 = int(f"{str(y1)[:2]}{parts[1]}")

        cache_file = os.path.join(self.config.fbref_cache_dir, f"match_xg_{season_label.replace('/', '_')}.csv")
        if os.path.exists(cache_file) and not self.force_scrape:
            try:
                df_cached = pd.read_csv(cache_file)
                if len(df_cached) > 0:
                    if not self.quiet:
                        logger.info(f"  Loaded cached match xG for {season_label} ({len(df_cached)} matches)")
                    return df_cached.to_dict("records")
            except Exception:
                pass

        url = f"https://fbref.com/en/comps/9/{y1}-{y2}/schedule/{y1}-{y2}-Premier-League-Scores-and-Fixtures"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        if not self.quiet:
            logger.info(f"  Scraping FBref match xG for {season_label}...")

        try:
            response = self.session.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                if not self.quiet:
                    logger.warning("    Failed (HTTP %s)", response.status_code)
                return []

            tables = pd.read_html(StringIO(response.text))
            if not tables:
                return []

            df_fix = tables[0]
            if isinstance(df_fix.columns, pd.MultiIndex):
                df_fix.columns = df_fix.columns.droplevel()

            cols = list(df_fix.columns)
            xg_indices = [i for i, c in enumerate(cols) if str(c) == "xG"]
            home_col = "Home" if "Home" in cols else None
            away_col = "Away" if "Away" in cols else None

            if home_col is None or away_col is None or len(xg_indices) < 2:
                if not self.quiet:
                    logger.warning("    Could not find Home/Away/xG columns")
                return []

            records = []
            for _, row in df_fix.iterrows():
                home_name = str(row[home_col]).strip()
                away_name = str(row[away_col]).strip()
                if len(home_name) < 3 or len(away_name) < 3:
                    continue

                home_xg_val = pd.to_numeric(row.iloc[xg_indices[0]], errors="coerce")
                away_xg_val = pd.to_numeric(row.iloc[xg_indices[1]], errors="coerce")
                if pd.isna(home_xg_val) or pd.isna(away_xg_val):
                    continue

                records.append({
                    "home": map_fbref_name(home_name),
                    "away": map_fbref_name(away_name),
                    "home_xg": float(home_xg_val),
                    "away_xg": float(away_xg_val),
                })

            if records:
                pd.DataFrame(records).to_csv(cache_file, index=False)
                if not self.quiet:
                    logger.info(f"    Got match xG for {len(records)} matches")

            return records
        except Exception as exc:
            if not self.quiet:
                logger.error("    Error scraping match xG: %s", exc)
            return []

    def load_fbref_data(self) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], float]]:
        all_xg_data: Dict[Tuple[str, str], Dict[str, float]] = {}
        all_poss_data: Dict[Tuple[str, str], float] = {}

        xg_2025_26 = {
            "Man United": (1.76, 1.30, +0.46, 1.78, 1.37, +0.02),
            "Liverpool": (1.74, 1.26, +0.48, 1.56, 1.30, -0.18),
            "Arsenal": (1.72, 0.89, +0.83, 2.00, 0.75, +0.28),
            "Man City": (1.68, 1.16, +0.52, 2.07, 0.93, +0.39),
            "Chelsea": (1.59, 1.28, +0.31, 1.78, 1.15, +0.19),
            "Newcastle": (1.58, 1.36, +0.22, 1.41, 1.44, -0.17),
            "Bournemouth": (1.55, 1.53, +0.02, 1.59, 1.67, +0.04),
            "Nott'm Forest": (1.50, 1.43, +0.07, 0.93, 1.44, -0.57),
            "Brighton": (1.49, 1.40, +0.09, 1.33, 1.26, -0.16),
            "Aston Villa": (1.48, 1.43, +0.05, 1.36, 1.07, -0.12),
            "Leeds": (1.36, 1.45, -0.09, 1.37, 1.70, +0.01),
            "Crystal Palace": (1.35, 1.34, +0.01, 1.07, 1.19, -0.28),
            "Fulham": (1.35, 1.39, -0.04, 1.41, 1.52, +0.06),
            "Tottenham": (1.29, 1.46, -0.17, 1.37, 1.52, +0.08),
            "Everton": (1.29, 1.49, -0.20, 1.07, 1.15, -0.22),
            "Brentford": (1.27, 1.49, -0.22, 1.48, 1.37, +0.21),
            "West Ham": (1.22, 1.75, -0.53, 1.19, 1.81, -0.03),
            "Sunderland": (1.15, 1.63, -0.48, 1.04, 1.22, -0.11),
            "Wolves": (1.10, 1.63, -0.53, 0.69, 1.76, -0.41),
            "Burnley": (1.08, 1.87, -0.79, 1.07, 1.93, -0.01),
        }

        for team, (xg, xga, xgd, gf, ga, overperf) in xg_2025_26.items():
            all_xg_data[("2025/26", team)] = {
                "xg_per90": xg,
                "xga_per90": xga,
                "xgd_per90": xgd,
                "xg_overperf": overperf,
                "gf_per90": gf,
                "ga_per90": ga,
            }

        if not self.quiet:
            logger.info(f"  Loaded 2025/26 xG data for {len(xg_2025_26)} teams (from FBref screenshots)")

        scraping_blocked = False
        for start_year in range(2017, 2025):
            end_year = start_year + 1
            season_label = f"{start_year}/{str(end_year)[-2:]}"
            url = f"https://fbref.com/en/comps/9/{start_year}-{end_year}/{start_year}-{end_year}-Premier-League-Stats"
            cache_file = os.path.join(self.config.fbref_cache_dir, f"xg_{season_label.replace('/', '_')}.csv")

            df_xg = None
            if os.path.exists(cache_file) and not self.force_scrape:
                try:
                    df_xg = pd.read_csv(cache_file)
                    if not self.quiet:
                        logger.info(f"  Loaded cached xG for {season_label} ({len(df_xg)} teams)")
                except Exception:
                    df_xg = None

            if df_xg is None and not scraping_blocked:
                df_xg = self.scrape_fbref_season(url, season_label)
                if df_xg is not None:
                    df_xg.to_csv(cache_file, index=False)
                else:
                    if not self.quiet:
                        logger.warning("  FBref blocked scraping, using defaults for remaining historical seasons")
                    scraping_blocked = True
                time.sleep(4)

            if df_xg is not None:
                for _, row in df_xg.iterrows():
                    fd_name = row.get("fd_name", row.get("Squad", ""))
                    all_xg_data[(season_label, fd_name)] = {
                        "xg_per90": float(row.get("xg_per90", 1.3)),
                        "xga_per90": float(row.get("xga_per90", 1.3)),
                        "xgd_per90": float(row.get("xgd_per90", 0.0)),
                        "xg_overperf": float(row.get("xg_overperf", 0.0)),
                        "gf_per90": float(row.get("gf_per90", 1.3)),
                        "ga_per90": float(row.get("ga_per90", 1.3)),
                    }

        poss_from_screenshots = {
            "Arsenal": 57.9,
            "Aston Villa": 54.1,
            "Bournemouth": 49.4,
            "Brentford": 46.6,
            "Brighton": 52.6,
            "Burnley": 41.3,
            "Chelsea": 58.8,
            "Crystal Palace": 45.1,
            "Everton": 44.1,
            "Fulham": 51.3,
            "Leeds": 45.1,
            "Liverpool": 60.5,
            "Man City": 59.1,
            "Man United": 53.2,
            "Newcastle": 52.8,
            "Nott'm Forest": 48.5,
            "Sunderland": 44.3,
            "Tottenham": 50.7,
            "West Ham": 42.6,
            "Wolves": 43.2,
        }
        for team, poss in poss_from_screenshots.items():
            all_poss_data[("2025/26", team)] = float(poss)

        return all_xg_data, all_poss_data

    def load_match_xg_data(self) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
        match_xg_lookup: Dict[Tuple[str, str, str], Tuple[float, float]] = {}

        for start_year in range(2017, 2025):
            end_year = start_year + 1
            season_label = f"{start_year}/{str(end_year)[-2:]}"
            records = self.scrape_fbref_match_xg(season_label)
            for rec in records:
                match_xg_lookup[(season_label, rec["home"], rec["away"])] = (float(rec["home_xg"]), float(rec["away_xg"]))
            if records:
                time.sleep(4)

        if not self.quiet:
            logger.info(f"  Total per-match xG records: {len(match_xg_lookup)}")

        return match_xg_lookup

    def load_fpl_team_data(self) -> Dict[str, Dict[str, float]]:
        """Load team-level FPL aggregates from the public bootstrap endpoint."""
        endpoint = "https://fantasy.premierleague.com/api/bootstrap-static/"
        cache_file = os.path.join(self.config.fbref_cache_dir, "fpl_bootstrap_cache.json")

        payload: Optional[Dict[str, Any]] = None

        def _read_cache() -> Optional[Dict[str, Any]]:
            if not os.path.exists(cache_file):
                return None
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                ts = float(cached.get("_fetched_at", 0.0))
                age_hours = (time.time() - ts) / 3600.0
                if age_hours <= self.config.fpl_cache_max_age_hours and isinstance(cached.get("payload"), dict):
                    return cached["payload"]
            except Exception:
                return None
            return None

        if not self.force_scrape:
            payload = _read_cache()

        if payload is None:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = self.session.get(endpoint, headers=headers, timeout=20)
                if response.status_code == 200:
                    payload = response.json()
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump({"_fetched_at": time.time(), "payload": payload}, f)
                    except Exception:
                        pass
                    if not self.quiet:
                        logger.info("  Loaded FPL bootstrap data from API")
            except Exception:
                payload = None

        if payload is None:
            # Fallback: use stale cache if available
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                payload = cached.get("payload")
                if not self.quiet and isinstance(payload, dict):
                    logger.warning("  Loaded stale FPL bootstrap cache (API unavailable)")
            except Exception:
                payload = None

        if not isinstance(payload, dict):
            if not self.quiet:
                logger.warning("  FPL data unavailable, using neutral defaults")
            return {}

        teams = payload.get("teams", [])
        elements = payload.get("elements", [])
        if not isinstance(teams, list) or not isinstance(elements, list):
            return {}

        team_id_to_name: Dict[int, str] = {}
        for t in teams:
            team_id = int(t.get("id", -1))
            if team_id > 0:
                team_id_to_name[team_id] = map_fpl_name(t.get("name", ""))

        agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "form_weighted_sum": 0.0,
                "weight_sum": 0.0,
                "key_players_missing": 0.0,
                "unavailable_count": 0.0,
                "squad_count": 0.0,
            }
        )

        for elem in elements:
            try:
                team_name = team_id_to_name.get(int(elem.get("team", -1)))
                if not team_name:
                    continue

                form = float(elem.get("form", 0.0) or 0.0)
                minutes = float(elem.get("minutes", 0.0) or 0.0)
                now_cost = float(elem.get("now_cost", 0.0) or 0.0)
                selected_by = float(elem.get("selected_by_percent", 0.0) or 0.0)
                chance = elem.get("chance_of_playing_next_round")
                chance = 100.0 if chance is None else float(chance)
                status = str(elem.get("status", "a")).lower()

                # Weight form toward likely regulars and high-value players.
                weight = max(1.0, minutes / 180.0) * (1.0 + max(now_cost - 55.0, 0.0) / 30.0)
                agg[team_name]["form_weighted_sum"] += form * weight
                agg[team_name]["weight_sum"] += weight
                agg[team_name]["squad_count"] += 1.0

                unavailable = (status not in ("a", "d")) or (chance < 75.0)
                is_key = now_cost >= self.config.fpl_key_player_cost_threshold or selected_by >= 10.0
                if unavailable:
                    agg[team_name]["unavailable_count"] += 1.0
                    if is_key:
                        agg[team_name]["key_players_missing"] += 1.0
            except Exception:
                continue

        fpl_team_data: Dict[str, Dict[str, float]] = {}
        for team, stats in agg.items():
            weight_sum = max(stats["weight_sum"], 1e-6)
            squad_count = max(stats["squad_count"], 1.0)
            fpl_team_data[team] = {
                "fpl_form_score": float(stats["form_weighted_sum"] / weight_sum),
                "key_players_missing": float(stats["key_players_missing"]),
                "injury_pressure": float(stats["unavailable_count"] / squad_count),
            }

        if not self.quiet:
            logger.info(f"  FPL team profiles loaded: {len(fpl_team_data)}")

        return fpl_team_data

    def load_match_data(self) -> Tuple[pd.DataFrame, List[str]]:
        def season_start_year(season_label: str) -> int:
            try:
                return int(str(season_label).split("/")[0])
            except Exception:
                return -1

        season_files: List[str] = []
        season_labels: List[str] = []
        min_start_year = season_start_year(self.config.min_training_season)

        for i in range(13, 0, -1):
            path = os.path.join(self.config.data_dir, f"E0 ({i}).csv")
            if os.path.exists(path):
                label = f"{2025 - i}/{str(2026 - i)[-2:]}"
                if season_start_year(label) >= min_start_year:
                    season_files.append(path)
                    season_labels.append(label)

        current = os.path.join(self.config.data_dir, "E0.csv")
        if os.path.exists(current):
            label = "2025/26"
            if season_start_year(label) >= min_start_year:
                season_files.append(current)
                season_labels.append(label)

        if not season_files:
            raise FileNotFoundError("No EPL season CSV files found in data directory.")
        if self.config.target_test_season not in season_labels:
            raise ValueError(
                f"Target test season {self.config.target_test_season} is earlier than min_training_season "
                f"{self.config.min_training_season} or missing from available files."
            )

        all_dfs = []
        for path, label in zip(season_files, season_labels):
            tmp = pd.read_csv(path, encoding="utf-8-sig")
            use_cols = [c for c in self.COMMON_COLS if c in tmp.columns]
            tmp = tmp[use_cols].copy()
            tmp["season"] = label
            tmp = tmp.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
            all_dfs.append(tmp)

        df = pd.concat(all_dfs, ignore_index=True)
        df["parsed_date"] = df["Date"].apply(parse_date)

        season_order = {s: i for i, s in enumerate(season_labels)}
        df["_orig_idx"] = np.arange(len(df))
        df["_season_order"] = df["season"].map(season_order)
        df["_date_sort"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["_date_sort"] = df["_date_sort"].fillna(pd.Timestamp("1900-01-01"))

        df = df.sort_values(["_season_order", "_date_sort", "_orig_idx"]).reset_index(drop=True)
        df = df.drop(columns=["_orig_idx", "_season_order", "_date_sort"])

        return df, season_labels
