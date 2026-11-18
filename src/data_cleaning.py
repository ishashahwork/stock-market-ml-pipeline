"""Select configured companies and write their silver Parquet datasets."""

import tomllib
from pathlib import Path

import polars as pl


def main():
    with Path("src/config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    raw_dir = Path(config["BRONZE_DIR"]) / "parquet"
    output_dir = Path(config["SILVER_DIR"]) / "parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    companies = pl.read_parquet(raw_dir / "companies.parquet")
    share_prices = pl.read_parquet(raw_dir / "share_prices.parquet")

    for company in config["companies"]:
        matches = companies.filter(pl.col("Company Name") == company)
        if matches.height != 1:
            raise ValueError(
                f"Expected one SimFin company named {company!r}; found {matches.height}. "
                "Check the company names in src/config.toml."
            )
        simfin_id = matches["SimFinId"].item()
        cleaned = share_prices.filter(pl.col("SimFinId") == simfin_id).drop("Dividend", "SimFinId")
        if cleaned.is_empty():
            raise ValueError(f"No share-price rows found for {company!r} (SimFinId {simfin_id}).")
        cleaned.write_parquet(output_dir / f"{company}_share_prices_cleaned.parquet")
        print(f"[cleaning] {company}: saved {cleaned.height} price rows.")


if __name__ == "__main__":
    main()
