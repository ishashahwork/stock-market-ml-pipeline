from dotenv import load_dotenv
import pandas as pd
import requests
import simfin as sf
import os
import polars as pl
import json
import pickle
import math

from src.feature_engineering import build_features, validate_feature_version


class PySimFinError(Exception):
    pass


class PySimFinAuthError(PySimFinError):
    pass


class PySimFinRateLimitError(PySimFinError):
    pass


class PySimFinRequestError(PySimFinError):
    pass


_PRICE_COLUMNS = {
    "Opening Price": "Open",
    "Highest Price": "High",
    "Lowest Price": "Low",
    "Last Closing Price": "Close",
    "Adjusted Closing Price": "Adj. Close",
    "Trading Volume": "Volume",
    "Dividend Paid": "Dividend",
    "Common Shares Outstanding": "Shares Outstanding",
}

def generate_trading_actions(
    prediction_dataframe: pd.DataFrame,
    buy_confidence_threshold: float = 0.55,
    sell_confidence_threshold: float = 0.55,
    base_trade_size: int = 1,
    high_confidence_trade_size: int = 5,
    high_confidence_threshold: float = 0.70,
) -> pd.DataFrame:
    action_dataframe = prediction_dataframe.copy()

    action_list = []
    trade_size_list = []

    for _, row in action_dataframe.iterrows():
        prediction = row["Prediction"]
        prediction_confidence = float(row["Prediction Confidence"])

        action = "HOLD"
        trade_size = 0

        if prediction in [1, "Bullish"]:
            if prediction_confidence >= high_confidence_threshold:
                action = "BUY"
                trade_size = high_confidence_trade_size
            elif prediction_confidence >= buy_confidence_threshold:
                action = "BUY"
                trade_size = base_trade_size

        elif prediction in [0, "Bearish"]:
            if prediction_confidence >= high_confidence_threshold:
                action = "SELL"
                trade_size = high_confidence_trade_size
            elif prediction_confidence >= sell_confidence_threshold:
                action = "SELL"
                trade_size = base_trade_size

        action_list.append(action)
        trade_size_list.append(trade_size)

    action_dataframe["Action"] = action_list
    action_dataframe["Trade Size"] = trade_size_list

    return action_dataframe


def backtest_trading_strategy(
    action_dataframe: pd.DataFrame,
    price_column: str = "Close",
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.0,
) -> pd.DataFrame:
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Starting cash must be positive and finite.")
    if not math.isfinite(transaction_fee) or transaction_fee < 0:
        raise ValueError("Transaction fees must be nonnegative and finite.")
    backtest_dataframe = action_dataframe.copy()

    cash = float(initial_cash)
    shares_held = 0

    executed_action_list = []
    executed_trade_size_list = []
    trade_value_list = []
    transaction_fee_history = []
    cash_history = []
    shares_held_history = []
    portfolio_value_history = []

    for _, row in backtest_dataframe.iterrows():
        current_price = float(row[price_column])
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError("Simulation prices must be positive and finite.")
        requested_action = row["Action"]
        requested_trade_size = int(row["Trade Size"])

        executed_action = "HOLD"
        executed_trade_size = 0
        trade_value = 0.0
        executed_fee = 0.0

        if requested_action == "BUY" and requested_trade_size > 0:
            affordable_trade_size = int(max(0.0, cash - transaction_fee) // current_price)
            shares_to_buy = min(requested_trade_size, affordable_trade_size)

            if shares_to_buy > 0:
                trade_value = shares_to_buy * current_price
                cash -= trade_value + transaction_fee
                executed_fee = transaction_fee
                shares_held += shares_to_buy
                executed_action = "BUY"
                executed_trade_size = shares_to_buy

        elif requested_action == "SELL" and requested_trade_size > 0:
            shares_to_sell = min(requested_trade_size, shares_held)

            if shares_to_sell > 0 and cash + shares_to_sell * current_price >= transaction_fee:
                trade_value = shares_to_sell * current_price
                cash += trade_value - transaction_fee
                executed_fee = transaction_fee
                shares_held -= shares_to_sell
                executed_action = "SELL"
                executed_trade_size = shares_to_sell

        portfolio_value = cash + shares_held * current_price

        executed_action_list.append(executed_action)
        executed_trade_size_list.append(executed_trade_size)
        trade_value_list.append(trade_value)
        transaction_fee_history.append(executed_fee)
        cash_history.append(cash)
        shares_held_history.append(shares_held)
        portfolio_value_history.append(portfolio_value)

    backtest_dataframe["Executed Action"] = executed_action_list
    backtest_dataframe["Executed Trade Size"] = executed_trade_size_list
    backtest_dataframe["Trade Value"] = trade_value_list
    backtest_dataframe["Transaction Fee"] = transaction_fee_history
    backtest_dataframe["Cash"] = cash_history
    backtest_dataframe["Shares Held"] = shares_held_history
    backtest_dataframe["Portfolio Value"] = portfolio_value_history

    return backtest_dataframe


def summarize_backtest_results(
    backtest_dataframe: pd.DataFrame,
    initial_cash: float = 10000.0,
) -> dict:
    if backtest_dataframe.empty:
        return {
            "initial_cash": initial_cash,
            "final_portfolio_value": initial_cash,
            "total_return": 0.0,
            "buy_trades": 0,
            "sell_trades": 0,
            "hold_days": 0,
            "total_shares_bought": 0,
            "total_shares_sold": 0,
        }

    final_portfolio_value = float(backtest_dataframe["Portfolio Value"].iloc[-1])
    total_return = (final_portfolio_value - initial_cash) / initial_cash

    buy_trades = int((backtest_dataframe["Executed Action"] == "BUY").sum())
    sell_trades = int((backtest_dataframe["Executed Action"] == "SELL").sum())
    hold_days = int((backtest_dataframe["Executed Action"] == "HOLD").sum())

    total_shares_bought = int(
        backtest_dataframe.loc[
            backtest_dataframe["Executed Action"] == "BUY",
            "Executed Trade Size",
        ].sum()
    )

    total_shares_sold = int(
        backtest_dataframe.loc[
            backtest_dataframe["Executed Action"] == "SELL",
            "Executed Trade Size",
        ].sum()
    )

    return {
        "initial_cash": float(initial_cash),
        "final_portfolio_value": final_portfolio_value,
        "total_return": float(total_return),
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "hold_days": hold_days,
        "total_shares_bought": total_shares_bought,
        "total_shares_sold": total_shares_sold,
    }


def run_strategy_backtest(
    prediction_dataframe: pd.DataFrame,
    price_column: str = "Close",
    initial_cash: float = 10000.0,
    buy_confidence_threshold: float = 0.55,
    sell_confidence_threshold: float = 0.55,
    base_trade_size: int = 1,
    high_confidence_trade_size: int = 5,
    high_confidence_threshold: float = 0.70,
    transaction_fee: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    action_dataframe = generate_trading_actions(
        prediction_dataframe=prediction_dataframe,
        buy_confidence_threshold=buy_confidence_threshold,
        sell_confidence_threshold=sell_confidence_threshold,
        base_trade_size=base_trade_size,
        high_confidence_trade_size=high_confidence_trade_size,
        high_confidence_threshold=high_confidence_threshold,
    )

    backtest_dataframe = backtest_trading_strategy(
        action_dataframe=action_dataframe,
        price_column=price_column,
        initial_cash=initial_cash,
        transaction_fee=transaction_fee,
    )

    backtest_summary = summarize_backtest_results(
        backtest_dataframe=backtest_dataframe,
        initial_cash=initial_cash,
    )

    return backtest_dataframe, backtest_summary
    

def transform_share_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the training features while retaining dates and the latest usable row."""
    ordered = df.copy()
    ordered.index = pd.to_datetime(ordered.index)
    ordered = ordered.sort_index()
    prices = pl.from_pandas(ordered.drop(columns=["Dividend"], errors="ignore"))
    transformed = build_features(
        prices.lazy().with_row_index("__row_id"), include_targets=False
    ).collect().to_pandas()
    positions = transformed.pop("__row_id").to_numpy()
    transformed.index = ordered.index.take(positions)
    return transformed


def predict_company_classification(
    input_dataframe: pd.DataFrame,
    company: str,
    model_output_dir: str,
) -> pd.DataFrame:
    company_simple_name = company.split()[0].title()
    company_model_dir = os.path.join(model_output_dir, company_simple_name)

    classification_model_path = os.path.join(company_model_dir, "classification_model.pkl")
    metadata_path = os.path.join(company_model_dir, "metadata.json")

    if not os.path.exists(classification_model_path):
        raise FileNotFoundError(
            f"Classification model not found for {company} at {classification_model_path}"
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found for {company} at {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    validate_feature_version(metadata)

    with open(classification_model_path, "rb") as classification_model_file:
        classification_model = pickle.load(classification_model_file)

    feature_columns = metadata["feature_columns"]

    missing_feature_columns = [
        feature_column
        for feature_column in feature_columns
        if feature_column not in input_dataframe.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            f"Input dataframe is missing required feature columns: {missing_feature_columns}"
        )

    model_input_dataframe = input_dataframe[feature_columns].copy()

    prediction_probabilities = classification_model.predict_proba(model_input_dataframe)
    predicted_classes = classification_model.predict(model_input_dataframe)

    output_dataframe = input_dataframe.copy()
    output_dataframe["Prediction"] = predicted_classes
    output_dataframe["Probability Class 0"] = prediction_probabilities[:, 0]
    output_dataframe["Probability Class 1"] = prediction_probabilities[:, 1]
    output_dataframe["Prediction Confidence"] = prediction_probabilities.max(axis=1)

    return output_dataframe


class PySimFin:
    """Python wrapper for the SimFin REST API v3."""

    BASE_URL = "https://backend.simfin.com/api/v3"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            load_dotenv()
            self.api_key = api_key or os.getenv("API_KEY")
        except Exception as e:
            raise PySimFinAuthError("PySimFin.py: Failed to load API key from environment variables.")

        if not self.api_key or not self.api_key.strip():
            raise PySimFinAuthError("Set API_KEY in your environment or .env before requesting SimFin data.")

        self.headers = {
            "accept": "application/json",
            "Authorization": f"api-key {self.api_key}"
        }

    def get_share_prices(self, ticker: str, start: str = '2026-02-01', end: str = '2026-03-22') -> pd.DataFrame:
        try:
            url = f"{self.BASE_URL}/companies/prices/compact"
            params = {
                "ticker": ticker.upper(),
                "start": start,
                "end": end
            }
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
        except Exception as e:
            print('PySimFin.py: Failed to load share prices. Raising exception.')
            raise e

        if response.status_code in (401, 403):
            print(f"Response: {response.text}")
            raise PySimFinAuthError("PySimFin.py: Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            print(f"Response: {response.text}")
            raise PySimFinRateLimitError("PySimFin.py: Rate limit reached. Wait and try again.")
        if response.status_code >= 400:
            print(f"Response: {response.text}")
            raise PySimFinRequestError(f"PySimFin.py: Request failed ({response.status_code}): {response.text}")

        payload = response.json()
        item = payload[0]
        df = pd.DataFrame(item["data"], columns=item["columns"])
        df = df.rename(columns=_PRICE_COLUMNS)
        df.set_index('Date', inplace=True)
        return df

    def get_financial_statement(self, ticker: str, start: str = '2025-01-01', end: str = '2026-03-22', statement: str = 'pl,bs,cf,derived') -> dict[str, pd.DataFrame]:
        try:
            response = requests.get(
                f"{self.BASE_URL}/companies/statements/compact",
                params={
                    "ticker": ticker.upper(),
                    "start": start,
                    "end": end,
                    "statements": statement
                },
                headers=self.headers,
                timeout=30,
            )
        except Exception as e:
            print('PySimFin.py: Failed to load financial statement. Raising exception.')
            raise e

        if response.status_code in (401, 403):
            raise PySimFinAuthError("PySimFin.py: Authentication failed. Check your SimFin API key.")
        if response.status_code == 429:
            raise PySimFinRateLimitError("PySimFin.py: Rate limit reached. Wait and try again.")
        if response.status_code >= 400:
            raise PySimFinRequestError(f"PySimFin.py: Request failed ({response.status_code}): {response.text}")

        payload = response.json()
        dict_of_dfs = {}
        for item in payload[0]['statements']:
            df = pd.DataFrame(item["data"], columns=item["columns"])
            dict_of_dfs[item['statement']] = df
        return dict_of_dfs