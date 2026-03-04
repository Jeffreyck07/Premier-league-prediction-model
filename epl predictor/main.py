"""CLI entry point for EPL predictor package."""

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from .backtester import BacktestEngine, ModelEvaluator
from .config import AppConfig, setup_logger, show_progress
from .data_loader import DataLoader
from .features import FeatureEngineer
from .inference import PredictionService
from .models import BettingOptimizer, ModelTrainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EPL Match Result Predictor v7")
    parser.add_argument("--predict", nargs=2, metavar=("HOME_TEAM", "AWAY_TEAM"), help="Predict one fixture")
    parser.add_argument("--interactive", action="store_true", help="Interactive prediction mode")
    parser.add_argument("--standings", action="store_true", help="Predict final season standings and champion")
    parser.add_argument(
        "--champion",
        action="store_true",
        help="Champion mode: final rankings plus 3 reasons for the top title favorite",
    )
    parser.add_argument("--backtest", action="store_true", help="Run historical backtesting and calibration")
    parser.add_argument("--scrape", action="store_true", help="Force FBref re-scrape")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(console_with_timestamp=not (args.interactive or args.champion))
    predict_mode = args.predict is not None
    interactive_mode = args.interactive
    champion_mode = args.champion and not predict_mode and not interactive_mode
    backtest_mode = args.backtest and not predict_mode and not interactive_mode
    standings_only = args.standings and not predict_mode and not interactive_mode and not backtest_mode and not champion_mode
    quiet = predict_mode or interactive_mode or standings_only or backtest_mode or champion_mode

    config = AppConfig.load_from_yaml()

    if not quiet:
        logger.info("--- Loading FBref xG Data ---")

    loader = DataLoader(config=config, quiet=quiet, force_scrape=args.scrape, interactive_mode=interactive_mode)

    show_progress(0, "Loading xG data...", interactive_mode)
    fbref_xg, fbref_poss = loader.load_fbref_data()
    show_progress(7, "Season xG loaded", interactive_mode)

    if not quiet:
        logger.info(f"  Total xG records: {len(fbref_xg)} team-seasons")
        logger.info(f"  Possession records: {len(fbref_poss)} team-seasons")

    show_progress(8, "Loading match xG...", interactive_mode)
    match_xg_lookup = loader.load_match_xg_data()
    show_progress(10, "All xG data loaded", interactive_mode)

    show_progress(11, "Loading FPL player data...", interactive_mode)
    fpl_team_data = loader.load_fpl_team_data()
    show_progress(13, "FPL data loaded", interactive_mode)
    if not quiet:
        logger.info(f"  FPL team records: {len(fpl_team_data)}")

    show_progress(15, "Loading match data...", interactive_mode)
    df, season_labels = loader.load_match_data()
    show_progress(20, "Match data loaded", interactive_mode)

    if not quiet:
        logger.info("=" * 74)
        logger.info("EPL MATCH RESULT PREDICTOR v7 — FPL + Dixon-Coles + Betting Optimizer")
        logger.info("=" * 74)
        logger.info(f"\nSeasons: {len(season_labels)} ({season_labels[0]} to {season_labels[-1]})")
        logger.info(f"Training floor season: {config.min_training_season}")
        logger.info(f"Matches: {len(df)}")
        logger.info(
            f"\nHome Win: {(df['FTR'] == 'H').mean():.1%}  "
            f"Draw: {(df['FTR'] == 'D').mean():.1%}  "
            f"Away Win: {(df['FTR'] == 'A').mean():.1%}"
        )

    show_progress(25, "Engineering features...", interactive_mode)
    fe = FeatureEngineer(
        config=config,
        fbref_xg=fbref_xg,
        fbref_poss=fbref_poss,
        match_xg_lookup=match_xg_lookup,
        fpl_team_data=fpl_team_data,
        quiet=quiet,
        interactive_mode=interactive_mode,
    )
    feat_df = fe.build_feature_matrix(df)

    X = feat_df[fe.feature_cols].values.astype(float)
    y = feat_df["target"].values
    y_home_goals = feat_df["home_goals"].values.astype(float)
    y_away_goals = feat_df["away_goals"].values.astype(float)

    X[:, fe.euro_indices] *= config.euro_weight

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    show_progress(55, "Feature matrix built", interactive_mode)

    if not quiet:
        logger.info(f"\nSamples: {len(X)}")
        logger.info(f"Features: {len(fe.feature_cols)}")
        logger.info(f"Classes: {list(le.classes_)}")

    test_mask = feat_df["season"] == config.target_test_season
    train_mask = ~test_mask

    X_train = X[train_mask.values]
    y_train = y_encoded[train_mask.values]
    X_test = X[test_mask.values]
    y_test = y_encoded[test_mask.values]

    y_hg_train = y_home_goals[train_mask.values]
    y_ag_train = y_away_goals[train_mask.values]

    if not quiet:
        logger.info(f"\nTrain: {len(X_train)} matches | Test: {len(X_test)} matches ({config.target_test_season})")

    trainer = ModelTrainer(config=config, quiet=quiet, interactive_mode=interactive_mode, force_retrain=args.retrain)
    trainer.fit(X_train, y_train, y_hg_train, y_ag_train, le, feature_cols=fe.feature_cols)

    show_progress(100, "Ready!", interactive_mode)

    predictor = PredictionService(config=config, feature_engineer=fe, trainer=trainer, label_encoder=le, quiet=quiet)

    if predict_mode:
        predictor.predict_match(args.predict[0], args.predict[1])
        return

    if interactive_mode:
        logger.info("=" * 64)
        logger.info("  EPL Match Predictor v7 — Interactive Mode")
        logger.info(f"  Ensemble: LightGBM + XGBoost + Dixon-Coles ({len(fe.feature_cols)} features)")
        logger.info("=" * 64)
        logger.info(f"\n  Teams ({len(fe.current_teams)}):")
        for idx, team in enumerate(fe.current_teams, 1):
            logger.info(f"    {idx:>2d}. {team}")
        logger.info("\n  Type 'quit' to exit.\n")

        while True:
            try:
                home_in = input("  Home team: ").strip()
                if home_in.lower() in ("quit", "exit", "q"):
                    break
                away_in = input("  Away team: ").strip()
                if away_in.lower() in ("quit", "exit", "q"):
                    break
                predictor.predict_match(home_in, away_in)
                logger.info("")
            except (EOFError, KeyboardInterrupt):
                logger.info("\n  Goodbye!")
                break
        return

    if champion_mode:
        predictor.predict_season_standings(feat_df, explain=True)
        return

    if standings_only:
        predictor.predict_season_standings(feat_df, explain=False)
        return

    # Full evaluation
    X_test_model = trainer.transform_features(X_test)
    lgb_test_acc = accuracy_score(y_test, trainer.lgb_model.predict(X_test_model))
    xgb_test_acc = accuracy_score(y_test, trainer.xgb_model.predict(X_test_model))
    dc_test_proba = trainer._dixon_coles_proba(X_test, le)
    dc_test_acc = accuracy_score(y_test, np.argmax(dc_test_proba, axis=1))

    ens_proba = trainer.predict_proba(X_test, le)
    ens_pred_argmax = np.argmax(ens_proba, axis=1)
    ens_pred_thr = trainer.predict(X_test, le, use_thresholds=True)

    ens_acc_argmax = accuracy_score(y_test, ens_pred_argmax)
    ens_acc_thr = accuracy_score(y_test, ens_pred_thr)

    if backtest_mode:
        evaluator = ModelEvaluator()
        brier = evaluator.brier_score_multiclass(y_test, ens_proba, le)
        ece = evaluator.expected_calibration_error(y_test, ens_proba, le, n_bins=10)

        test_df = feat_df.loc[test_mask].copy()
        backtester = BacktestEngine(
            trainer=trainer,
            label_encoder=le,
            evaluator=evaluator,
            initial_bankroll=10_000.0,
            kelly_cap=config.kelly_cap_fraction,
            kelly_scale=config.kelly_fraction_scale,
        )
        bt_result = backtester.run_backtest(
            df=test_df,
            feature_cols=fe.feature_cols,
            target_col="target",
            odds_cols=("B365H", "B365D", "B365A"),
            use_thresholds=True,
        )

        logger.info("\n" + "=" * 74)
        logger.info(f"BACKTEST & CALIBRATION — {config.target_test_season}")
        logger.info("=" * 74)
        logger.info(f"Brier Score:               {brier:.6f}")
        logger.info(f"Expected Calibration Error {ece:.6f}")
        logger.info("-" * 74)
        logger.info(f"Initial Bankroll:          ${bt_result['initial_bankroll']:,.2f}")
        logger.info(f"Ending Bankroll:           ${bt_result['ending_bankroll']:,.2f}")
        logger.info(f"Total Bets Placed:         {bt_result['total_bets_placed']}")
        logger.info(f"Win Rate:                  {bt_result['win_rate']:.2%}")
        logger.info(f"Total Profit/Loss:         ${bt_result['total_profit_loss']:,.2f}")
        logger.info(f"ROI (on staked capital):   {bt_result['roi']:.2%}")
        logger.info(f"Max Drawdown:              {bt_result['max_drawdown']:.2%}")
        logger.info("=" * 74)
        return

    models = {
        "Ensemble (Threshold Optimized)": ens_acc_thr,
        "Ensemble (Argmax)": ens_acc_argmax,
        "LightGBM": lgb_test_acc,
        "XGBoost": xgb_test_acc,
        "Dixon-Coles": dc_test_acc,
    }
    best_name = max(models, key=models.get)

    logger.info("\n" + "=" * 74)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 74)
    for name, acc in sorted(models.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if name == best_name else ""
        logger.info(f"  {name:<36s} {acc:.1%}{marker}")

    logger.info("\n" + "=" * 74)
    logger.info("VALIDATION CALIBRATION")
    logger.info("=" * 74)
    if trainer.val_scores:
        for key, val in trainer.val_scores.items():
            logger.info(f"  {key:<30s} {val:.1%}")
    logger.info(f"  TimeSeries CV best accuracy      {trainer.cv_best_score:.1%}")
    cls_thresholds = ", ".join(f"{c}:{trainer.thresholds[i]:.3f}" for i, c in enumerate(le.classes_))
    logger.info(f"  Decision thresholds               {cls_thresholds}")

    logger.info("\n" + "=" * 74)
    logger.info(f"DETAILED EVALUATION — {best_name}")
    logger.info(f"Trained on {len(X_train)} matches, tested on {len(X_test)} ({config.target_test_season})")
    logger.info("=" * 74)

    logger.info(f"\nAccuracy: {ens_acc_thr:.1%}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, ens_pred_thr, target_names=le.classes_))

    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, ens_pred_thr)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=[f"Pred_{c}" for c in le.classes_])
    logger.info(cm_df)

    logger.info("\n--- Top 20 Feature Importances (LightGBM) ---")
    importances = trainer.lgb_model.feature_importances_
    feat_imp = sorted(zip(trainer.feature_cols, importances), key=lambda x: x[1], reverse=True)
    max_imp = feat_imp[0][1] if feat_imp else 1
    for rank, (feat_name, imp) in enumerate(feat_imp[:20], 1):
        bar = "█" * int((imp / max_imp) * 30) if max_imp > 0 else ""
        marker = " *" if any(k in feat_name.lower() for k in ["xg", "poss", "euro", "ema"]) else ""
        logger.info(f"  {rank:2d}. {feat_name:<28s} {int(imp):>5d}  {bar}{marker}")

    logger.info("\n  (* = xG/EMA/possession/european feature)")

    logger.info("\n--- Predictions on Last 15 Matches (Thresholded Ensemble + Betting) ---")
    logger.info(
        f"{'Home':<18s} {'Away':<18s} {'Act':>4s} {'Pred':>4s}  "
        f"{'P(H)':>6s} {'P(D)':>6s} {'P(A)':>6s} {'ExpG':>7s} {'EV*':>7s} {'Kelly*':>8s}"
    )
    logger.info("-" * 108)

    test_indices = feat_df.index[test_mask].tolist()
    last_n = min(15, len(X_test))
    start = len(X_test) - last_n
    correct = 0
    plus_ev_hits = 0

    for i in range(start, len(X_test)):
        proba = ens_proba[i]
        pred_label = le.inverse_transform([ens_pred_thr[i]])[0]
        actual_label = le.inverse_transform([y_test[i]])[0]
        row_idx = test_indices[i]
        home = feat_df.iloc[row_idx]["home_team"]
        away = feat_df.iloc[row_idx]["away_team"]
        prob_dict = dict(zip(le.classes_, proba))
        exp_hg_arr, exp_ag_arr = trainer.predict_expected_goals(X_test[i : i + 1])
        exp_hg = float(exp_hg_arr[0])
        exp_ag = float(exp_ag_arr[0])
        odds_dict = {
            "H": float(feat_df.iloc[row_idx]["B365H"]),
            "D": float(feat_df.iloc[row_idx]["B365D"]),
            "A": float(feat_df.iloc[row_idx]["B365A"]),
        }
        market = BettingOptimizer.evaluate_market(
            prob_dict,
            odds_dict,
            kelly_cap=config.kelly_cap_fraction,
            kelly_scale=config.kelly_fraction_scale,
        )
        best_edge = max(market.items(), key=lambda kv: kv[1]["ev"])
        best_ev = best_edge[1]["ev"]
        best_kelly = best_edge[1]["kelly"]
        if best_ev > 0:
            plus_ev_hits += 1
        hit = pred_label == actual_label
        if hit:
            correct += 1
        mark = "+" if hit else "-"
        logger.info(
            f"{home:<18s} {away:<18s} {actual_label:>4s} {pred_label:>4s}  "
            f"{prob_dict.get('H', 0):>5.1%} {prob_dict.get('D', 0):>5.1%} {prob_dict.get('A', 0):>5.1%} "
            f"{exp_hg:.1f}-{exp_ag:.1f} {best_ev:+6.3f} {best_kelly*100:>7.2f}%  {mark}"
        )

    logger.info(f"\nLast {last_n}: {correct}/{last_n} = {correct / max(last_n, 1):.1%}")
    logger.info(f"Best available +EV in last {last_n}: {plus_ev_hits}/{last_n}")

    logger.info("\n--- Elo Power Rankings (Current) ---")
    elo_rank = sorted([(t, fe.elo_ratings[t]) for t in fe.current_teams], key=lambda x: -x[1])
    for rank, (team, elo) in enumerate(elo_rank, 1):
        bar = "█" * max(0, int((elo - 1350) / 10))
        logger.info(f"  {rank:>2d}. {team:<18s} {elo:>7.1f}  {bar}")

    logger.info("\n--- xG Power Rankings (2025/26) ---")
    xg_ranks = []
    for team in fe.current_teams:
        xg = fe.get_team_xg(team, config.target_test_season)
        xg_ranks.append((team, xg["xg_per90"], xg["xga_per90"], xg["xgd_per90"], xg["xg_overperf"]))
    xg_ranks.sort(key=lambda x: -x[3])
    logger.info(f"  {'#':>3s}  {'Team':<18s} {'xG':>5s} {'xGA':>5s} {'xGD':>6s} {'Over':>6s}")
    for rank, (team, xg, xga, xgd, over) in enumerate(xg_ranks, 1):
        logger.info(f"  {rank:>3d}  {team:<18s} {xg:>5.2f} {xga:>5.2f} {xgd:>+6.2f} {over:>+6.2f}")

    logger.info("\n" + "=" * 74)
    logger.info("WHAT'S INCLUDED (v7)")
    logger.info("=" * 74)
    logger.info("  [+] OOP architecture (DataLoader / FeatureEngineer / ModelTrainer)")
    logger.info(f"  [+] Training truncated to >= {config.min_training_season}")
    logger.info("  [+] Dynamic Elo with season regression + goal-diff K scaling")
    logger.info("  [+] Head-to-head records and rest days")
    logger.info("  [+] LightGBM + XGBoost + Dixon-Coles ensemble")
    logger.info("  [+] Optuna with TimeSeriesSplit CV (leakage-safe for sequential football data)")
    logger.info("  [+] EMA recent form features (points/goals)")
    logger.info("  [+] EMA recent xG/xGA/xGD features + trend")
    logger.info("  [+] FBref xG and possession integration")
    logger.info("  [+] FPL player-level form/injury aggregation features")
    logger.info("  [+] Tactical possession matchup features")
    logger.info("  [+] European competition fatigue features")
    logger.info("  [+] Odds movement feature (opening-closing, placeholder-ready)")
    logger.info("  [+] Validation-based class threshold optimization for H/D/A")
    logger.info("  [+] Betting optimizer: EV and Kelly criterion")
    logger.info("  [+] Interactive mode and single-match predict mode")
    logger.info(f"  [+] Joblib model cache: {config.model_cache_file}")
    logger.info(f"  Total features: {len(fe.feature_cols)}")
    logger.info("=" * 74)

    predictor.predict_season_standings(feat_df, explain=False)


if __name__ == "__main__":
    main()
