# Script Summary: `data_ingestion.py`

## Purpose
Handles the project’s raw data ingestion step by loading configuration and environment settings, downloading market data from SimFin, and saving the results as parquet files for downstream use.

## Inputs
- `src/config.toml`
- Environment variable: `API_KEY`
- External data source: SimFin (`market='us'`, daily share prices)

## Outputs
- `BRONZE_DIR/parquet/share_prices.parquet`
- `BRONZE_DIR/parquet/companies.parquet`

## Main Logic
1. Load paths from `config.toml`.
2. Load the API key and create the configured bronze/parquet directories if needed.
3. Load environment variables and configure the SimFin client.
4. Download U.S. company metadata.
5. Download daily share-price data.
6. Convert the downloaded pandas DataFrames to Polars DataFrames.
7. Persist both datasets as parquet files.

## Key Dependencies
- `python-dotenv`
- `simfin`
- `polars`
- `tomllib`
- `pathlib`
- `os`

## Important Notes
- Uses a `main()` entrypoint and runs as a standalone script.
- Relies on `config.toml` for directory structure.
- Relies on a valid SimFin API key in the environment.
- Emits concise download/save summaries and lets underlying failures propagate once with their original traceback.
- Rejects a missing or blank `API_KEY` before any network call, without printing its value.

## Pipeline Role
This is the first stage of the project pipeline. It prepares the raw company and share-price datasets that later scripts can clean, transform, and use for feature engineering or model training.

## Caveats
- Assumes the expected keys exist in `config.toml`.
- Validates that `API_KEY` is present; the service determines whether it is valid.
- Downloads only U.S. company data and daily share prices.
- Performs minimal validation beyond exception handling.
