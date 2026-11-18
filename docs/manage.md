# Pipeline and app commands

Run `manage.py` from the repository root after installing `requirements.txt` and configuring `.env`/`src/config.toml`.

```bash
python manage.py --help
python manage.py --stage all
```

`all` runs ingestion, cleaning, feature engineering, model training, and offline backtesting in order. It completes without launching a long-lived web server.

Individual choices are `ingestion`, `cleaning`, `feature_engineering`, `model_training`, and `trading_logic`. Selecting a later stage assumes the required upstream files exist; missing stages are not run automatically.

Launch the app separately:

```bash
python manage.py --stage app
# Equivalent direct entry point:
python -m streamlit run src/Home.py
```

The app command uses the active Python environment via `sys.executable` and a checked subprocess. It no longer invokes an unqualified shell command. The CLI still imports pipeline dependencies before argument parsing, so even `--help` requires them, including XGBoost's platform OpenMP runtime.
