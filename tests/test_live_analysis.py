"""Check real live-page controls and shared features without API calls or real models."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.PySimFin import transform_share_prices
from src.feature_engineering import build_features, validate_feature_version, FEATURE_VERSION
from test_etl import prices

PAGE = Path(__file__).resolve().parents[1] / "src/pages/Live_Analysis.py"


def raw_prices():
    data = prices(90).to_pandas()
    data["Dividend"] = 0.0
    data.index = pd.bdate_range("2026-01-01", periods=len(data), name="Date")
    return data


def fake_predict(features, company, model_dir):
    result = features.copy()
    result["Prediction"] = [i % 2 for i in range(len(features))]
    result["Prediction Confidence"] = 0.8
    return result


class SharedFeatureTests(unittest.TestCase):
    def test_training_and_live_features_match_and_latest_row_is_retained(self):
        raw = raw_prices()
        live = transform_share_prices(raw)
        trained = build_features(prices(90).lazy()).collect().to_pandas()
        trained = trained.drop(columns=["Target Return Metric", "Target Return Class"])
        trained.index = raw.index[20:-1]
        pd.testing.assert_frame_equal(live.iloc[:-1], trained, check_freq=False)
        self.assertEqual(live.index[-1], raw.index[-1])
        self.assertNotIn("Target Return Metric", live.columns)

    def test_dropped_rows_keep_their_actual_dates(self):
        raw = raw_prices()
        missing_date = raw.index[40]
        raw.loc[missing_date, "High"] = float("nan")
        live = transform_share_prices(raw)
        self.assertNotIn(missing_date, live.index)
        self.assertIn(raw.index[41], live.index)
        self.assertEqual(live.index[-1], raw.index[-1])

    def test_old_models_are_rejected_with_retraining_instructions(self):
        with self.assertRaisesRegex(ValueError, "Regenerate features and retrain"):
            validate_feature_version({})
        validate_feature_version({"feature_version": FEATURE_VERSION})


class LivePageTests(unittest.TestCase):
    def test_no_request_until_submit_and_controls_reuse_results(self):
        with patch("src.PySimFin.PySimFin") as client, patch("src.PySimFin.predict_company_classification", side_effect=fake_predict) as predict:
            client.return_value.get_share_prices.return_value = raw_prices()
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            client.assert_not_called()
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            client.return_value.get_share_prices.assert_called_once()
            predict.assert_called_once()
            self.assertEqual(len(app.metric), 3)
            app.slider[0].set_value(0.85).run(timeout=30)
            self.assertEqual(app.metric[2].value, "0")
            app.multiselect[0].set_value(["Close"]).run(timeout=30)
            app.multiselect[1].set_value(["BUY"]).run(timeout=30)
            self.assertTrue(app.dataframe[-1].value.empty)
            client.return_value.get_share_prices.assert_called_once()
            predict.assert_called_once()
            self.assertEqual(len(app.exception), 0)

    def test_missing_models_still_allow_market_charts(self):
        with patch("src.PySimFin.PySimFin") as client, patch("src.PySimFin.predict_company_classification", side_effect=FileNotFoundError("No model")):
            client.return_value.get_share_prices.return_value = raw_prices()
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Model signals unavailable", app.warning[0].value)
            self.assertEqual(len(app.dataframe), 1)
            self.assertEqual(len(app.metric), 0)

    def test_invalid_date_range_does_not_request_data(self):
        with patch("src.PySimFin.PySimFin") as client:
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            app.date_input[0].set_value(date.today())
            app.button[0].click().run(timeout=30)
            client.assert_not_called()
            self.assertIn("30 business days", app.warning[0].value)
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
