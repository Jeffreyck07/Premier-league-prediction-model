"""Configuration, constants, and shared helpers for EPL predictor."""

import logging
import os
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - handled gracefully at runtime
    yaml = None


def setup_logger(log_file: str = "logs/epl_predictor.log", console_with_timestamp: bool = True) -> logging.Logger:
    """Configure process-wide logging once.

    Console: INFO and above
    File: DEBUG and above
    """
    root_logger = logging.getLogger()
    if getattr(setup_logger, "_configured", False):
        return root_logger

    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_format = "%(asctime)s - %(levelname)s - %(message)s" if console_with_timestamp else "%(message)s"
    console_formatter = logging.Formatter(console_format)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    expanded_log_file = os.path.expanduser(os.path.expandvars(log_file))
    log_dir = os.path.dirname(expanded_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(expanded_log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    setup_logger._configured = True
    root_logger.debug("Logger initialized. Console=INFO, File=DEBUG, path=%s", expanded_log_file)
    return root_logger


@dataclass
class AppConfig:
    data_dir: str = os.getenv("EPL_DATA_DIR", "./data")
    fbref_cache_dir: str = os.getenv("EPL_CACHE_DIR", "./cache/fbref")
    model_cache_file: str = os.path.join(os.path.dirname(__file__), "epl_models_v7_biasfix.joblib")
    target_test_season: str = "2025/26"
    euro_weight: float = 1.5
    optuna_trials: int = 50
    tscv_splits: int = 5
    validation_fraction: float = 0.15
    # Bias controls
    min_training_season: str = "2019/20"
    # EMA controls
    ema_recent_window: int = 5
    ema_alpha_form: float = 0.55
    ema_alpha_goals: float = 0.52
    ema_alpha_xg: float = 0.56
    ema_alpha_short: float = 0.62
    ema_alpha_long: float = 0.40

    # Bookmaker feature controls (reduce bookmaker-signal copying by default)
    use_bookmaker_features: bool = False
    use_odds_movement_feature: bool = False
    min_odds_coverage_for_features: float = 0.60

    fpl_cache_max_age_hours: float = 24.0
    fpl_default_form: float = 1.0
    fpl_key_player_cost_threshold: float = 70.0
    odds_movement_placeholder_std: float = 0.0

    # Historical match xG proxy calibration
    xg_proxy_fit_min_matches: int = 80
    xg_proxy_sot_coef: float = 0.23
    xg_proxy_shot_coef: float = 0.05
    xg_proxy_intercept: float = 0.20
    xg_proxy_baseline_weight: float = 0.35

    # Probability/threshold calibration
    class_prior_blend: float = 0.08
    threshold_oof_splits: int = 5

    dixon_coles_max_goals: int = 8
    dixon_coles_rho_min: float = -0.20
    dixon_coles_rho_max: float = 0.35

    # Safer bankroll controls
    kelly_cap_fraction: float = 0.15
    kelly_fraction_scale: float = 0.50

    # Standings projection robustness
    standings_simulations: int = 1000
    standings_random_seed: int = 42

    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[str] = None) -> "AppConfig":
        logger = logging.getLogger(__name__)
        config = cls()

        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        yaml_path = os.path.expanduser(os.path.expandvars(str(yaml_path)))

        if not os.path.exists(yaml_path):
            logger.warning("Config YAML not found at %s. Using AppConfig defaults.", yaml_path)
            return config
        if yaml is None:
            logger.warning("PyYAML is not installed. Using AppConfig defaults.")
            return config

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to parse config YAML at %s (%s). Using defaults.", yaml_path, exc)
            return config

        if not isinstance(raw, dict):
            logger.error("Config YAML at %s is not a mapping. Using defaults.", yaml_path)
            return config

        payload = raw.get("app_config", raw)
        if not isinstance(payload, dict):
            logger.error("Config YAML payload at %s is invalid. Using defaults.", yaml_path)
            return config

        valid_fields = {f.name for f in fields(cls)}
        for key, value in payload.items():
            if key not in valid_fields or value is None:
                continue
            if isinstance(value, str):
                value = os.path.expanduser(os.path.expandvars(value))
            setattr(config, key, value)

        logger.info("Loaded configuration from %s", yaml_path)
        return config


# Team name mapping: FBref Squad name -> football-data.co.uk name
FBREF_TO_FD = {
    "Manchester Utd": "Man United",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Newcastle United": "Newcastle",
    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Wolves": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Sheffield Utd": "Sheffield United",
    "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth",
    "Luton Town": "Luton",
    "Norwich City": "Norwich",
    "Watford": "Watford",
    "Huddersfield": "Huddersfield",
    "Cardiff City": "Cardiff",
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Southampton": "Southampton",
    "Sheffield United": "Sheffield United",
}

FPL_TO_FD = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Wolves": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
}

# Tier: 0=none, 1=ECL, 2=EL, 3=CL
EURO_COMPETITION_DATA = {
    ("2012/13", "Chelsea"): 3, ("2012/13", "Man City"): 3,
    ("2012/13", "Man United"): 3, ("2012/13", "Arsenal"): 3,
    ("2012/13", "Tottenham"): 2, ("2012/13", "Newcastle"): 2,
    ("2012/13", "Liverpool"): 2,
    ("2013/14", "Man United"): 3, ("2013/14", "Man City"): 3,
    ("2013/14", "Chelsea"): 3, ("2013/14", "Arsenal"): 3,
    ("2013/14", "Tottenham"): 2, ("2013/14", "Swansea"): 2,
    ("2013/14", "Liverpool"): 0,
    ("2014/15", "Chelsea"): 3, ("2014/15", "Man City"): 3,
    ("2014/15", "Arsenal"): 3, ("2014/15", "Liverpool"): 3,
    ("2014/15", "Everton"): 2, ("2014/15", "Tottenham"): 2,
    ("2014/15", "Hull"): 2,
    ("2015/16", "Chelsea"): 3, ("2015/16", "Arsenal"): 3,
    ("2015/16", "Man City"): 3, ("2015/16", "Man United"): 3,
    ("2015/16", "Liverpool"): 2, ("2015/16", "Tottenham"): 2,
    ("2015/16", "Southampton"): 2, ("2015/16", "Everton"): 2,
    ("2016/17", "Leicester"): 3, ("2016/17", "Arsenal"): 3,
    ("2016/17", "Man City"): 3, ("2016/17", "Tottenham"): 3,
    ("2016/17", "Man United"): 2, ("2016/17", "Liverpool"): 2,
    ("2016/17", "Southampton"): 2, ("2016/17", "West Ham"): 2,
    ("2017/18", "Chelsea"): 3, ("2017/18", "Tottenham"): 3,
    ("2017/18", "Man City"): 3, ("2017/18", "Liverpool"): 3,
    ("2017/18", "Arsenal"): 3, ("2017/18", "Man United"): 2,
    ("2017/18", "Everton"): 2,
    ("2018/19", "Liverpool"): 3, ("2018/19", "Man City"): 3,
    ("2018/19", "Tottenham"): 3, ("2018/19", "Man United"): 3,
    ("2018/19", "Chelsea"): 2, ("2018/19", "Arsenal"): 2,
    ("2018/19", "Burnley"): 2, ("2018/19", "Everton"): 2,
    ("2019/20", "Liverpool"): 3, ("2019/20", "Tottenham"): 3,
    ("2019/20", "Man City"): 3, ("2019/20", "Chelsea"): 3,
    ("2019/20", "Arsenal"): 2, ("2019/20", "Man United"): 2,
    ("2019/20", "Wolves"): 2,
    ("2020/21", "Liverpool"): 3, ("2020/21", "Man City"): 3,
    ("2020/21", "Man United"): 3, ("2020/21", "Chelsea"): 3,
    ("2020/21", "Leicester"): 2, ("2020/21", "Arsenal"): 2,
    ("2020/21", "Tottenham"): 2,
    ("2021/22", "Man City"): 3, ("2021/22", "Liverpool"): 3,
    ("2021/22", "Chelsea"): 3, ("2021/22", "Man United"): 3,
    ("2021/22", "Leicester"): 2, ("2021/22", "West Ham"): 2,
    ("2021/22", "Tottenham"): 1,
    ("2022/23", "Man City"): 3, ("2022/23", "Liverpool"): 3,
    ("2022/23", "Chelsea"): 3, ("2022/23", "Tottenham"): 3,
    ("2022/23", "Arsenal"): 2, ("2022/23", "Man United"): 2,
    ("2022/23", "West Ham"): 1,
    ("2023/24", "Man City"): 3, ("2023/24", "Chelsea"): 3,
    ("2023/24", "Man United"): 3, ("2023/24", "Newcastle"): 3,
    ("2023/24", "Liverpool"): 2, ("2023/24", "Brighton"): 2,
    ("2023/24", "West Ham"): 2, ("2023/24", "Aston Villa"): 1,
    ("2024/25", "Man City"): 3, ("2024/25", "Arsenal"): 3,
    ("2024/25", "Liverpool"): 3, ("2024/25", "Aston Villa"): 3,
    ("2024/25", "Man United"): 2, ("2024/25", "West Ham"): 2,
    ("2024/25", "Tottenham"): 2, ("2024/25", "Chelsea"): 1,
    ("2025/26", "Liverpool"): 3, ("2025/26", "Arsenal"): 3,
    ("2025/26", "Man City"): 3, ("2025/26", "Aston Villa"): 3,
    ("2025/26", "Man United"): 2, ("2025/26", "Tottenham"): 2,
    ("2025/26", "Newcastle"): 2, ("2025/26", "Chelsea"): 1,
}

EURO_TIER_NAMES = {0: "None", 1: "Conference League", 2: "Europa League", 3: "Champions League"}

DEFAULT_XG = {
    "xg_per90": 1.35,
    "xga_per90": 1.35,
    "xgd_per90": 0.0,
    "xg_overperf": 0.0,
    "gf_per90": 1.35,
    "ga_per90": 1.35,
}
DEFAULT_POSS = 50.0


def show_progress(pct: int, stage: str, interactive_mode: bool) -> None:
    if not interactive_mode:
        return
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(f"\r  Loading model... [{bar}] {pct:>3d}%  {stage:<30s}")
    sys.stdout.flush()
    if pct >= 100:
        sys.stdout.write("\r" + " " * 90 + "\r")
        sys.stdout.flush()


def map_fbref_name(fbref_name: Any) -> str:
    name = str(fbref_name).strip()
    if name in FBREF_TO_FD:
        return FBREF_TO_FD[name]
    name_lower = name.lower().replace(" fc", "").replace(" afc", "").strip()
    for fb_key, fd_val in FBREF_TO_FD.items():
        key = fb_key.lower()
        if name_lower in key or key in name_lower:
            return fd_val
    return name


def map_fpl_name(fpl_name: Any) -> str:
    name = str(fpl_name).strip()
    if name in FPL_TO_FD:
        return FPL_TO_FD[name]
    # Try FBREF mapping as secondary lookup
    mapped = map_fbref_name(name)
    if mapped != name:
        return mapped
    # Fuzzy substring match against FPL dict keys
    name_lower = name.lower()
    for key, value in FPL_TO_FD.items():
        key_lower = key.lower()
        if name_lower in key_lower or key_lower in name_lower:
            return value
    return name


def parse_date(value: Any) -> Optional[datetime]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None
