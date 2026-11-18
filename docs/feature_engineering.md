# Feature engineering

`src/feature_engineering.py` contains the reusable `build_features(lf, include_targets=True)` function and the file-based `main()` pipeline stage.

The stage reads `src/config.toml`, consumes `SILVER_DIR/parquet/{company}_share_prices_cleaned.parquet`, and writes `GOLD_DIR/parquet/{company}_share_prices_modeled.parquet`. Paths and company filenames retain the original pipeline contract.

## Transformations

Features include daily and rolling log returns, rolling volatility, moving-average ratios, momentum, volume ratios/changes, intraday returns, price-range/close-position features, market capitalization, dilution, and interaction terms. Windows are 5, 10, and 20 trading rows.

The moving-average computation now uses its corresponding `window`. Previously all three labels used a five-row mean. Raw/intermediate columns are removed and incomplete rows are dropped. With complete inputs this produces 31 predictors and, for training, two next-day targets.

`include_targets=False` is used by the API wrapper for inference. It avoids requiring a future price, so the most recent usable row is retained. The wrapper preserves dates using a temporary row identifier and removes that identifier from the predictors.

## Model compatibility

`FEATURE_VERSION` identifies this corrected feature definition. Training saves it in model metadata; live inference and offline model loading require it. Do not use historical models trained with the old moving-average behavior against corrected features. Regenerate gold datasets, retrain models, and regenerate backtests.

## Errors and verification

The stage emits one saved-row summary per company. File/configuration/Polars exceptions propagate with their original traceback, instead of repeated print-and-reraise blocks.

Regression checks independently calculate the expected moving-average ratios and check warmup/target behavior, training/live parity, retained dates, and the latest inference row. No feature schema aliases were renamed.

Input rows must be chronological. This builder does not validate/sort dates itself. The live wrapper sorts by date; the original offline input pipeline still assumes order. Rows with non-finite values beyond nulls may need further validation on real data.
