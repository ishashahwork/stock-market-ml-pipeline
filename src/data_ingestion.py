"""Download SimFin data and persist the bronze Parquet datasets."""

import os
import tomllib
from pathlib import Path

import polars as pl
import simfin as sf
from dotenv import load_dotenv


def main():
    with Path("src/config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("Set API_KEY in your environment or .env before running ingestion.")

    raw_dir = Path(config["BRONZE_DIR"])
    parquet_dir = raw_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    sf.set_api_key(api_key)
    sf.set_data_dir(str(raw_dir))

    print("[ingestion] Downloading U.S. company metadata and daily prices from SimFin.")
    companies = pl.DataFrame(sf.load_companies(market="us"))
    share_prices = pl.DataFrame(sf.load_shareprices(variant="daily"))
    companies.write_parquet(parquet_dir / "companies.parquet")
    share_prices.write_parquet(parquet_dir / "share_prices.parquet")
    print(f"[ingestion] Saved {companies.height} companies and {share_prices.height} price rows.")


if __name__ == "__main__":
    main()
