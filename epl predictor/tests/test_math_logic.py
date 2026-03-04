import pytest

from epl_predictor.features import FeatureEngineer
from epl_predictor.models import BettingOptimizer


def test_expected_value_positive_ev():
    # EV = p * odds - 1
    ev = BettingOptimizer.expected_value(prob=0.55, decimal_odds=2.10)
    assert ev == pytest.approx(0.155)


def test_expected_value_negative_ev():
    ev = BettingOptimizer.expected_value(prob=0.40, decimal_odds=2.20)
    assert ev == pytest.approx(-0.12)


def test_expected_value_invalid_odds_returns_negative_one():
    assert BettingOptimizer.expected_value(prob=0.8, decimal_odds=1.0) == pytest.approx(-1.0)
    assert BettingOptimizer.expected_value(prob=0.8, decimal_odds=0.9) == pytest.approx(-1.0)


def test_kelly_fraction_standard_case():
    # p=0.60, odds=2.0 => b=1.0, q=0.4, f*=(0.6*1-0.4)/1=0.2
    kelly = BettingOptimizer.kelly_fraction(prob=0.60, decimal_odds=2.0, cap=0.25)
    assert kelly == pytest.approx(0.20)


def test_kelly_fraction_respects_cap():
    # p=0.80, odds=3.0 => b=2.0, q=0.2, f*=0.7, but cap at 0.25
    kelly = BettingOptimizer.kelly_fraction(prob=0.80, decimal_odds=3.0, cap=0.25)
    assert kelly == pytest.approx(0.25)


def test_kelly_fraction_negative_ev_returns_zero():
    # p=0.30, odds=2.0 => b=1.0, q=0.7, f*=-0.4 => 0.0
    kelly = BettingOptimizer.kelly_fraction(prob=0.30, decimal_odds=2.0, cap=0.25)
    assert kelly == pytest.approx(0.0)


def test_kelly_fraction_invalid_odds_returns_zero():
    assert BettingOptimizer.kelly_fraction(prob=0.8, decimal_odds=1.0, cap=0.25) == pytest.approx(0.0)
    assert BettingOptimizer.kelly_fraction(prob=0.8, decimal_odds=0.8, cap=0.25) == pytest.approx(0.0)


def test_elo_expected_equal_ratings_is_half():
    p = FeatureEngineer.elo_expected(1500, 1500)
    assert p == pytest.approx(0.5)


def test_elo_update_home_win_standard_case():
    # Equal Elo, team A wins with k=20 => +10 Elo
    updated = FeatureEngineer.elo_update(ra=1500, rb=1500, score_a=1.0, k=20)
    assert updated == pytest.approx(1510.0)


def test_elo_update_away_win_standard_case_for_team_a_loss():
    # Equal Elo, team A loses with k=20 => -10 Elo
    updated = FeatureEngineer.elo_update(ra=1500, rb=1500, score_a=0.0, k=20)
    assert updated == pytest.approx(1490.0)
