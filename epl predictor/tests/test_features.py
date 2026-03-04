import pytest

from epl_predictor.config import AppConfig
from epl_predictor.features import FeatureEngineer


def _build_feature_engineer() -> FeatureEngineer:
    config = AppConfig()
    config.target_test_season = "2025/26"
    config.fpl_default_form = 1.0

    fbref_xg = {
        ("2021/22", "Home FC"): {
            "xg_per90": 1.40,
            "xga_per90": 1.10,
            "xgd_per90": 0.30,
            "xg_overperf": 0.05,
            "gf_per90": 1.45,
            "ga_per90": 1.15,
        },
        ("2021/22", "Away FC"): {
            "xg_per90": 1.20,
            "xga_per90": 1.30,
            "xgd_per90": -0.10,
            "xg_overperf": -0.03,
            "gf_per90": 1.18,
            "ga_per90": 1.33,
        },
        ("2025/26", "Home FC"): {
            "xg_per90": 1.60,
            "xga_per90": 1.00,
            "xgd_per90": 0.60,
            "xg_overperf": 0.08,
            "gf_per90": 1.70,
            "ga_per90": 1.02,
        },
        ("2025/26", "Away FC"): {
            "xg_per90": 1.30,
            "xga_per90": 1.40,
            "xgd_per90": -0.10,
            "xg_overperf": -0.04,
            "gf_per90": 1.28,
            "ga_per90": 1.42,
        },
    }
    fbref_poss = {
        ("2021/22", "Home FC"): 54.0,
        ("2021/22", "Away FC"): 46.0,
        ("2025/26", "Home FC"): 57.0,
        ("2025/26", "Away FC"): 45.0,
    }
    match_xg_lookup = {}

    fpl_team_data = {
        "Home FC": {"fpl_form_score": 2.5, "key_players_missing": 3.0, "injury_pressure": 1.1},
        "Away FC": {"fpl_form_score": 1.8, "key_players_missing": 1.0, "injury_pressure": 0.4},
    }

    return FeatureEngineer(
        config=config,
        fbref_xg=fbref_xg,
        fbref_poss=fbref_poss,
        match_xg_lookup=match_xg_lookup,
        fpl_team_data=fpl_team_data,
        quiet=True,
        interactive_mode=False,
    )


def _base_row():
    return {"B365H": 2.20, "B365D": 3.30, "B365A": 3.10, "parsed_date": None}


def test_ema_known_values():
    values = [1.0, 2.0, 3.0, 4.0]
    alpha = 0.5
    # Start=1.0 -> 1.5 -> 2.25 -> 3.125
    expected = 3.125
    observed = FeatureEngineer.ema(values, alpha=alpha, default=0.0)
    assert observed == pytest.approx(expected)


def test_elo_update_home_and_away_win_scenarios():
    home_win_rating = FeatureEngineer.elo_update(ra=1500.0, rb=1500.0, score_a=1.0, k=20.0)
    away_win_rating = FeatureEngineer.elo_update(ra=1500.0, rb=1500.0, score_a=0.0, k=20.0)
    assert home_win_rating == pytest.approx(1510.0)
    assert away_win_rating == pytest.approx(1490.0)


def test_extract_features_historical_season_uses_neutral_fpl_defaults():
    fe = _build_feature_engineer()
    hs = fe.make_empty_stats()
    aws = fe.make_empty_stats()

    features = fe.extract_features(
        hs=hs,
        aws=aws,
        row=_base_row(),
        home_name="Home FC",
        away_name="Away FC",
        season="2021/22",
    )

    assert features["home_fpl_form_score"] == pytest.approx(fe.config.fpl_default_form)
    assert features["away_fpl_form_score"] == pytest.approx(fe.config.fpl_default_form)
    assert features["home_key_players_missing"] == pytest.approx(0.0)
    assert features["away_key_players_missing"] == pytest.approx(0.0)
    assert features["home_injury_pressure"] == pytest.approx(0.0)
    assert features["away_injury_pressure"] == pytest.approx(0.0)


def test_extract_features_current_season_uses_real_fpl_data():
    fe = _build_feature_engineer()
    hs = fe.make_empty_stats()
    aws = fe.make_empty_stats()

    features = fe.extract_features(
        hs=hs,
        aws=aws,
        row=_base_row(),
        home_name="Home FC",
        away_name="Away FC",
        season="2025/26",
    )

    assert features["home_fpl_form_score"] == pytest.approx(2.5)
    assert features["away_fpl_form_score"] == pytest.approx(1.8)
    assert features["home_key_players_missing"] == pytest.approx(3.0)
    assert features["away_key_players_missing"] == pytest.approx(1.0)
    assert features["home_injury_pressure"] == pytest.approx(1.1)
    assert features["away_injury_pressure"] == pytest.approx(0.4)
