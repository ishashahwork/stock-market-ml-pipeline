"""Exercise the real dashboard with temporary backtest files and Streamlit AppTest."""

from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dashboard-regression-")
        self.root = Path(self.temp.name)
        (self.root / "src/pages").mkdir(parents=True)
        shutil.copy2(ROOT / "src/pages/Backtest_Explorer.py", self.root / "src/pages/Backtest_Explorer.py")
        (self.root / "src/config.toml").write_text(
            'companies = ["TEST INC", "OTHER INC"]\nGOLD_DIR = "ETL/data/gold"\n'
        )

    def tearDown(self):
        self.temp.cleanup()

    def write_results(self, folder="Test"):
        directory = self.root / "ETL/data/gold/backtests" / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "holdout_backtest.csv"
        pd.DataFrame({
            "Close": [100.0, 102.0, 101.0, 103.0],
            "portfolio_value": [10000.0, 10020.0, 10010.0, 10030.0],
            "predicted_return": [0.01, -0.01, 0.02, 0.0],
            "Target Return Metric": [0.02, -0.01, 0.02, -0.01],
            "executed_action": ["BUY", "SELL", "HOLD", "BUY"],
        }).to_csv(path, index=False)
        return path

    def run_app(self):
        return AppTest.from_file(str(self.root / "src/pages/Backtest_Explorer.py")).run(timeout=30)

    def test_empty_state_does_not_invent_results(self):
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("No backtest results", app.warning[0].value)
        self.assertEqual(len(app.metric), 0)

    def test_company_selection_range_and_action_filter(self):
        self.write_results()
        self.write_results("Other")
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[2].value, "3")
        app.sidebar.selectbox[0].select("OTHER INC").run()
        self.assertEqual(app.subheader[0].value, "OTHER INC")
        app.sidebar.slider[0].set_value((1, 2)).run()
        self.assertEqual(app.metric[2].value, "1")
        app.multiselect[0].set_value(["SELL"]).run()
        self.assertEqual(app.dataframe[0].value["executed_action"].tolist(), ["SELL"])
        app.multiselect[0].set_value([]).run()
        self.assertTrue(app.dataframe[0].value.empty)
        self.assertEqual(len(app.exception), 0)

    def test_malformed_output_has_actionable_error(self):
        path = self.write_results()
        pd.DataFrame({"unrelated": [1]}).to_csv(path, index=False)
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("missing required columns", app.error[0].value)

    def test_invalid_numeric_data_does_not_produce_misleading_charts(self):
        path = self.write_results()
        data = pd.read_csv(path)
        data.loc[1, "portfolio_value"] = float("inf")
        data.to_csv(path, index=False)
        app = self.run_app()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("invalid numeric values", app.error[0].value)
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
