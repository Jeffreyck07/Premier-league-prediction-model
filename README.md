# EPL Predictor (v7)

EPL match prediction system with:
- LightGBM + XGBoost + Dixon-Coles ensemble
- Time-series CV and threshold optimization
- Elo + EMA + xG + FPL squad-health features
- EV/Kelly betting analytics
- Backtesting and champion projection (Monte Carlo)

## Current Performance (Latest Local Benchmark)

- TimeSeries CV accuracy: **53.5%**
- Holdout accuracy (2025/26 test season, thresholded ensemble): **49.4%** on 271 matches
- Benchmark date: **March 4, 2026**

Note: performance can vary with data updates, retraining seed, and feature toggles in `config.yaml`.

## 1) Requirements

- Python 3.10+
- Terminal (Linux/macOS shell, Windows PowerShell, or Windows CMD)

## 2) Project Structure

The Python package folder must be named `epl_predictor`:

```text
repo-root/
  epl_predictor/
    __init__.py
    main.py
    config.py
    ...
```

If your folder is named `epl predictor` (with a space), rename it to `epl_predictor`.

## 3) Setup (Linux/macOS)

```bash
git clone <YOUR_REPO_URL>
cd <repo-root>

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r epl_predictor/requirements.txt
```

## 4) Setup (Windows PowerShell)

```powershell
git clone <YOUR_REPO_URL>
cd <repo-root>

py -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r epl_predictor\requirements.txt
```

## 5) Setup (Windows CMD)

```bat
git clone <YOUR_REPO_URL>
cd <repo-root>

py -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r epl_predictor\requirements.txt
```

## 6) Data and Environment Variables

Put season CSV files (e.g., `E0.csv`, `E0 (1).csv`, ...) in your data folder.

Linux/macOS:

```bash
export EPL_DATA_DIR="$PWD/epl_predictor/data"
export EPL_CACHE_DIR="$PWD/epl_predictor/cache/fbref"
```

Windows PowerShell:

```powershell
$env:EPL_DATA_DIR = "$PWD\epl_predictor\data"
$env:EPL_CACHE_DIR = "$PWD\epl_predictor\cache\fbref"
```

Windows CMD:

```bat
set EPL_DATA_DIR=%CD%\epl_predictor\data
set EPL_CACHE_DIR=%CD%\epl_predictor\cache\fbref
```

## 7) Run the Model

Run from **repo root** (the parent of `epl_predictor/`).

Train/retrain model:

```bash
python -m epl_predictor.main --retrain
```

Predict one match:

```bash
python -m epl_predictor.main --predict Arsenal Chelsea
```

Interactive mode:

```bash
python -m epl_predictor.main --interactive
```

Champion mode (full ranking + 3 reasons for champion):

```bash
python -m epl_predictor.main --champion
```

Standings mode:

```bash
python -m epl_predictor.main --standings
```

Backtest + calibration:

```bash
python -m epl_predictor.main --backtest
```

## 8) Tests

```bash
python -m pytest -q epl_predictor/tests
```

## 9) Common Troubleshooting

- `ModuleNotFoundError: No module named epl_predictor`
  - Run from repo root (parent of `epl_predictor/`), not inside the package folder.
- `No EPL season CSV files found`
  - Check `EPL_DATA_DIR` and confirm CSVs exist there.
- FBref returns HTTP 403
  - Normal on some networks; model uses fallback/default behavior.
- Wrong Python environment
  - Use the same interpreter for install and run (`python -m pip ...` then `python -m epl_predictor.main ...`).

## 10) Configuration

Main runtime settings are in:
- `epl_predictor/config.yaml`

You can tune EMA, Kelly cap, Optuna trials, simulation count, etc. without changing code.

## License

MIT (see `LICENSE`).
