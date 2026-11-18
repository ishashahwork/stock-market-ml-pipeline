"""Offline regression checks for the ETL feedback fixes."""

import contextlib
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from src import data_cleaning, data_ingestion
from src.feature_engineering import build_features


def prices(count=60):
    # Curved prices make the three moving-average windows observably different.
    close = [100 + i * i / 10 for i in range(count)]
    return pl.DataFrame({
        "Open": close, "High": [v + 2 for v in close],
        "Low": [v - 2 for v in close], "Close": close, "Adj. Close": close,
        "Volume": [1000 + i * 5 for i in range(count)],
        "Shares Outstanding": [1_000_000] * count,
    })


class FeatureRegressionTests(unittest.TestCase):
    def test_ratios_use_their_named_window(self):
        data = prices()
        result = build_features(data.lazy()).collect()
        values = data["Adj. Close"].to_list()
        # First modeled row corresponds to source row 20 (longest return window).
        index = 20
        actual = []
        for window in (5, 10, 20):
            with self.subTest(window=window):
                average = sum(values[index - window + 1:index + 1]) / window
                expected = math.log(values[index] / average)
                ratio = result[f"Log MA Ratio {window}d"][0]
                self.assertAlmostEqual(ratio, expected, places=12)
                actual.append(ratio)
        self.assertEqual(len(set(actual)), 3)

    def test_targets_and_warmup_are_preserved(self):
        data = prices()
        result = build_features(data.lazy()).collect()
        self.assertEqual(result.height, data.height - 21)
        self.assertEqual(result.width, 33)  # 31 predictors and two targets.
        self.assertFalse(any(result.null_count().row(0)))
        expected = data["Adj. Close"][21] / data["Adj. Close"][20] - 1
        self.assertAlmostEqual(result["Target Return Metric"][0], expected)
        self.assertTrue(result["Target Return Class"][0])


class StageValidationTests(unittest.TestCase):
    def setUp(self):
        self.original_dir = Path.cwd()
        self.temp = tempfile.TemporaryDirectory(prefix="etl-regression-")
        os.chdir(self.temp.name)
        Path("src").mkdir()
        Path("src/config.toml").write_text(
            'companies = ["TEST INC"]\nETL_DIR = "ETL"\n'
            'BRONZE_DIR = "ETL/data/bronze"\nSILVER_DIR = "ETL/data/silver"\n'
            'GOLD_DIR = "ETL/data/gold"\n'
        )
        self.bronze = Path("ETL/data/bronze/parquet")
        self.bronze.mkdir(parents=True)

    def tearDown(self):
        os.chdir(self.original_dir)
        self.temp.cleanup()

    def test_missing_key_fails_before_network_access(self):
        with patch.dict(os.environ, {"API_KEY": ""}), patch.object(data_ingestion, "load_dotenv"), patch.object(data_ingestion.sf, "load_companies") as download:
            with self.assertRaisesRegex(ValueError, "Set API_KEY"):
                data_ingestion.main()
            download.assert_not_called()

    def test_missing_or_duplicate_company_has_actionable_error(self):
        for names in (["OTHER INC"], ["TEST INC", "TEST INC"]):
            with self.subTest(names=names):
                pl.DataFrame({"Company Name": names, "SimFinId": list(range(len(names)))}).write_parquet(self.bronze / "companies.parquet")
                pl.DataFrame({"SimFinId": [123], "Dividend": [0.0], "Close": [100.0]}).write_parquet(self.bronze / "share_prices.parquet")
                with self.assertRaisesRegex(ValueError, "Expected one SimFin company"):
                    data_cleaning.main()

    def test_cleaning_keeps_company_files_and_actual_row_counts(self):
        pl.DataFrame({"Company Name": ["TEST INC", "OTHER INC"], "SimFinId": [123, 456]}).write_parquet(self.bronze / "companies.parquet")
        pl.DataFrame({"SimFinId": [123, 123, 456], "Dividend": [0.0] * 3, "Close": [100.0, 101.0, 20.0]}).write_parquet(self.bronze / "share_prices.parquet")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            data_cleaning.main()
        cleaned = pl.read_parquet("ETL/data/silver/parquet/TEST INC_share_prices_cleaned.parquet")
        self.assertEqual(cleaned["Close"].to_list(), [100.0, 101.0])
        self.assertNotIn("SimFinId", cleaned.columns)
        self.assertNotIn("Dividend", cleaned.columns)
        self.assertIn("saved 2 price rows", output.getvalue())

    def test_company_without_prices_fails_clearly(self):
        pl.DataFrame({"Company Name": ["TEST INC"], "SimFinId": [123]}).write_parquet(self.bronze / "companies.parquet")
        pl.DataFrame({"SimFinId": [456], "Dividend": [0.0], "Close": [20.0]}).write_parquet(self.bronze / "share_prices.parquet")
        with self.assertRaisesRegex(ValueError, "No share-price rows"):
            data_cleaning.main()


if __name__ == "__main__":
    unittest.main()
