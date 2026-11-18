"""Build market features and next-period targets for each company."""

import tomllib
from pathlib import Path

import polars as pl


FEATURE_VERSION = "moving-average-windows-v2"


def validate_feature_version(metadata: dict) -> None:
    """Reject models trained with the earlier, incorrect moving-average features."""
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            "These models predate the feature-engineering fix. Regenerate features and "
            "retrain with python manage.py --stage feature_engineering, then "
            "python manage.py --stage model_training."
        )


def build_features(lf: pl.LazyFrame, include_targets: bool = True) -> pl.LazyFrame:
    """Transform chronologically ordered prices without reading or writing files."""
    lf = lf.with_columns(
        # Log Return
        (pl.col('Adj. Close') / pl.col('Adj. Close').shift(1)).log().alias(f'Log Return')
    )

    for window in [5, 10, 20]:
        lf = lf.with_columns(
            # Rolling Log Returns
            (pl.col('Adj. Close') / pl.col('Adj. Close').shift(window)).log().alias(f'Log Return {window}d'),

            # Volatility
            pl.col('Log Return').rolling_std(window).log1p().alias(f'Volatility {window}d'),

            # Moving Averages
            pl.col("Adj. Close").rolling_mean(window).alias(f"Moving Average {window}d"),

            # Momentum Pct. Change
            ((pl.col("Adj. Close") / pl.col("Adj. Close").shift(window)) - 1).alias(f"Momentum Pct. {window}d"),

            # Log Volume Ratio
            ((pl.col("Volume") / pl.col("Volume").rolling_mean(window))).log().alias(f"Log Volume Ratio {window}d"),
        )

        lf = lf.with_columns(
            # Log MA Ratio
            (pl.col("Adj. Close") / pl.col(f"Moving Average {window}d")).log().alias(f"Log MA Ratio {window}d")
        )

    lf = lf.with_columns(
        # Intraday Returns
        ((pl.col('Close') / pl.col('Open')) - 1).alias('Intraday Pct. Return'),

        # Ranges
        (pl.col('High') - pl.col('Low')).alias('Range'),
        ((pl.col('High') - pl.col('Low')) / pl.col('Close')).alias('Range Pct.'),

        # Close Position
        (((pl.col('Close') - pl.col('Low')) / (pl.col('High') - pl.col('Low'))) - 0.5).alias('Close Position'),

        # Log Volume Change
        (pl.col("Volume") / pl.col("Volume").shift(1)).log().tanh().alias("Log Volume Change"),

        # Log Market Cap
        (pl.col("Adj. Close") * pl.col("Shares Outstanding")).log().alias("Log Market Cap"),

        # Dilution / Issuance
        (pl.col("Shares Outstanding") / pl.col("Shares Outstanding").shift(1) - 1).alias('Delta Pct. Dilution / Issuance'),

        # Volume Return Interaction
        (pl.col("Log Return") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Return Volume 5d"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Return Volume 10d"),
        (pl.col("Log Return") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Return Volume 20d"),

        # Volume Volatility Interaction
        (pl.col("Volatility 5d") * pl.col("Log Volume Ratio 5d")).tanh().alias("Interaction Volatility Volume 5d"),
        (pl.col("Volatility 10d") * pl.col("Log Volume Ratio 10d")).tanh().alias("Interaction Volatility Volume 10d"),
        (pl.col("Volatility 20d") * pl.col("Log Volume Ratio 20d")).tanh().alias("Interaction Volatility Volume 20d"),

        # Momentum Volatility Interaction
        (pl.col("Momentum Pct. 5d") * pl.col("Volatility 5d")).tanh().alias("Interaction Momentum Volatility 5d"),
        (pl.col("Momentum Pct. 10d") * pl.col("Volatility 10d")).tanh().alias("Interaction Momentum Volume 10d"),
        (pl.col("Momentum Pct. 20d") * pl.col("Volatility 20d")).tanh().alias("Interaction Momentum Volume 20d"),

    )

    if include_targets:
        lf = lf.with_columns(
            (pl.col("Adj. Close").shift(-1) / pl.col("Adj. Close") - 1).alias("Target Return Metric"),
            ((pl.col("Adj. Close").shift(-1) / pl.col("Adj. Close") - 1) > 0).alias("Target Return Class"),
        )

    lf = lf.drop([
        "Open",
        "High",
        "Low",
        "Close",
        "Adj. Close",
        "Moving Average 5d",
        "Moving Average 10d",
        "Moving Average 20d",
        "Volume",
        "Range",
        "Shares Outstanding"
    ])

    return lf.drop_nulls()


def main():
    with Path("src/config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    input_dir = Path(config["SILVER_DIR"]) / "parquet"
    output_dir = Path(config["GOLD_DIR"]) / "parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    for company in config["companies"]:
        input_path = input_dir / f"{company}_share_prices_cleaned.parquet"
        output_path = output_dir / f"{company}_share_prices_modeled.parquet"
        modeled = build_features(pl.scan_parquet(input_path)).collect()
        modeled.write_parquet(output_path)
        print(f"[features] {company}: saved {modeled.height} modeled rows.")


if __name__ == "__main__":
    main()
