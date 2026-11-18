"""Verify the demo's data provenance, held-out inputs, and visitor controls."""

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.demo import COMPANIES, build_demo_analysis, generate_demo_prices
from src.PySimFin import backtest_trading_strategy

ROOT = Path(__file__).resolve().parents[1]


class DemoDataTests(unittest.TestCase):
    def test_all_company_datasets_are_reproducible_and_valid(self):
        for ticker in COMPANIES:
            with self.subTest(ticker=ticker):
                data = generate_demo_prices(ticker)
                pd.testing.assert_frame_equal(data, generate_demo_prices(ticker))
                self.assertEqual(len(data), 780)
                self.assertTrue(data.index.is_monotonic_increasing)
                self.assertTrue((data["High"] >= data[["Open", "Close"]].max(axis=1)).all())
                self.assertTrue((data["Low"] <= data[["Open", "Close"]].min(axis=1)).all())
                self.assertTrue((data["Low"] > 0).all())
                self.assertTrue((data["Volume"] > 0).all())

    def test_model_predictions_are_after_training_and_include_latest_row(self):
        analysis = build_demo_analysis("AMZN")
        self.assertEqual(analysis.feature_count, 31)
        self.assertLess(analysis.training_end, analysis.evaluation_start)
        self.assertTrue((analysis.predictions.index > analysis.training_end).all())
        self.assertEqual(analysis.predictions.index[-1], analysis.raw.index[-1])
        self.assertTrue(analysis.predictions["Prediction Confidence"].between(0.5, 1).all())
        self.assertTrue(pd.isna(analysis.predictions["Actual Next Return"].iloc[-1]))


class FeeTests(unittest.TestCase):
    def test_fees_are_charged_only_on_executed_trades(self):
        frame = pd.DataFrame({"Close": [100., 110., 100.], "Action": ["BUY", "SELL", "HOLD"], "Trade Size": [1, 1, 0]})
        result = backtest_trading_strategy(frame, initial_cash=1000., transaction_fee=1.)
        self.assertEqual(result["Cash"].tolist(), [899., 1008., 1008.])
        self.assertEqual(result["Transaction Fee"].tolist(), [1., 1., 0.])

    def test_fee_is_reserved_before_affordability_calculation(self):
        frame = pd.DataFrame({"Close": [100.], "Action": ["BUY"], "Trade Size": [1]})
        result = backtest_trading_strategy(frame, initial_cash=100., transaction_fee=1.)
        self.assertEqual(result["Executed Action"].tolist(), ["HOLD"])
        self.assertEqual(result["Cash"].tolist(), [100.])
        self.assertEqual(result["Transaction Fee"].tolist(), [0.])

    def test_invalid_costs_and_prices_fail_clearly(self):
        frame = pd.DataFrame({"Close": [100.], "Action": ["BUY"], "Trade Size": [1]})
        for fee in (-1., float("nan")):
            with self.assertRaisesRegex(ValueError, "fees"):
                backtest_trading_strategy(frame, transaction_fee=fee)
        frame["Close"] = 0.
        with self.assertRaisesRegex(ValueError, "prices"):
            backtest_trading_strategy(frame)


class DemoAppTests(unittest.TestCase):
    def test_visitor_can_explore_without_credentials_or_network(self):
        with patch("requests.get", side_effect=AssertionError("Demo must not call external services")):
            app = AppTest.from_file(str(ROOT / "src/Home.py")).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("synthetic" in item.value.lower() for item in app.info))
            self.assertEqual(app.metric[4].value, "31")
            app.selectbox[0].select("MSFT").run(timeout=30)
            app.radio[0].set_value("1M").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("MSFT" in item.value for item in app.subheader))
            app.multiselect[0].set_value([]).run(timeout=30)
            app.multiselect[1].set_value([]).run(timeout=30)
            self.assertTrue(app.dataframe[0].value.empty)
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
