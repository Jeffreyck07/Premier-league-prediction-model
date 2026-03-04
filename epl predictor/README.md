# EPL Predictor (v7)

Production-style English Premier League match prediction pipeline with:
- LightGBM + XGBoost + Dixon-Coles ensemble
- Time-series validation and threshold optimization
- EMA/Elo/xG tactical features
- FPL-derived squad health features (current-season guarded to avoid leakage)
- Betting EV + Kelly sizing
- Standings/champion projection with Monte Carlo simulation

## Project Layout

```text
epl_predictor/
  __init__.py
  config.py
  config.yaml
  data_loader.py
  features.py
  models.py
  inference.py
  backtester.py
  main.py
  tests/
  data/
```

## Requirements

- Python 3.10+
- See `requirements.txt`

Install:

```bash
python3 -m pip install -r requirements.txt
```

## Quick Start

From the parent directory of this package:

```bash
python3 -m epl_predictor.main --retrain
```

Predict one match:

```bash
python3 -m epl_predictor.main --predict Arsenal Chelsea
```

Interactive mode:

```bash
python3 -m epl_predictor.main --interactive
```

Champion mode (rankings + 3 reasons):

```bash
python3 -m epl_predictor.main --champion
```

Backtesting + calibration:

```bash
python3 -m epl_predictor.main --backtest
```

## Optional Environment Variables

```bash
export EPL_DATA_DIR="./data"
export EPL_CACHE_DIR="./cache/fbref"
```

## Tests

```bash
python3 -m pytest -q
```

## Notes

- FBref endpoints may block scraping (HTTP 403). The pipeline falls back to defaults/cached data.
- `config.yaml` controls runtime behavior (EMA, Kelly cap, Monte Carlo sims, etc.).
- Model cache artifacts (`*.joblib`) are intentionally ignored in git.

## License

MIT (see `LICENSE`).
