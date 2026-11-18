"""Interactive market charts and a model-driven simulation; fetch only on submit."""

from datetime import date, timedelta
from pathlib import Path
import tomllib

import pandas as pd
import streamlit as st

from src.PySimFin import (
    PySimFin, predict_company_classification, run_strategy_backtest, transform_share_prices,
)

ROOT = Path(__file__).resolve().parents[2]
COMPANIES = {
    "AMZN": "AMAZON COM INC", "AAPL": "APPLE INC", "MSFT": "MICROSOFT CORP",
    "AAON": "AAON INC", "ALL": "ALLSTATE CORP",
}


def main():
    st.set_page_config(page_title="Live Analysis", page_icon="📈", layout="wide")
    st.title("Live Analysis")
    st.caption("Fetch daily SimFin prices, explore market trends, and simulate model signals.")
    st.info("Daily price data may be delayed. Simulations exclude fees and slippage and do not execute trades.")
    try:
        with (ROOT / "src/config.toml").open("rb") as config_file:
            config = tomllib.load(config_file)
        tickers = [ticker for ticker in config.get("tickers", []) if ticker in COMPANIES]
    except (OSError, tomllib.TOMLDecodeError) as error:
        st.error(f"Unable to load company configuration: {error}")
        return
    if not tickers:
        st.warning("No supported tickers are configured.")
        return

    today = date.today()
    with st.sidebar.form("market_request"):
        ticker = st.selectbox("Ticker", tickers)
        start = st.date_input("Start date", today - timedelta(days=120), max_value=today)
        end = st.date_input("End date", today, max_value=today)
        submitted = st.form_submit_button("Load analysis", type="primary")
    st.sidebar.caption("Price requests run only when you press Load analysis. Chart and strategy controls reuse the loaded data.")

    if submitted:
        st.session_state.pop("live_result", None)
        if start > end or len(pd.bdate_range(start, end)) < 30:
            st.warning("Choose an ordered range covering at least 30 business days.")
        else:
            try:
                with st.spinner(f"Loading {ticker} daily prices…"):
                    try:
                        hosted_key = st.secrets.get("API_KEY")
                    except (FileNotFoundError, KeyError):
                        hosted_key = None
                    client = PySimFin(api_key=hosted_key) if hosted_key else PySimFin()
                    raw = client.get_share_prices(ticker, start.isoformat(), end.isoformat())
                    raw.index = pd.to_datetime(raw.index)
                    raw = raw.sort_index()
                    if raw.empty:
                        st.warning("SimFin returned no prices for this selection.")
                        return
                    result = {"ticker": ticker, "raw": raw, "predictions": None, "model_message": None}
                    transformed = transform_share_prices(raw)
                    if transformed.empty:
                        result["model_message"] = "Not enough complete price rows for the model features. Load a longer date range."
                    else:
                        try:
                            predictions = predict_company_classification(
                                transformed, COMPANIES[ticker], str(ROOT / config["GOLD_DIR"] / "trained_models")
                            )
                            aligned = raw.loc[predictions.index].copy()
                            aligned["Prediction"] = predictions["Prediction"]
                            aligned["Prediction Confidence"] = predictions["Prediction Confidence"]
                            result["predictions"] = aligned
                        except (FileNotFoundError, ValueError) as error:
                            result["model_message"] = (
                                f"Model signals unavailable: {error} Generate corrected features and retrain models to enable signals."
                            )
                    st.session_state["live_result"] = result
            except Exception as error:
                # Handle failures once at the UI boundary; preserve no partially loaded result.
                st.error(f"Unable to load this analysis: {error}")

    result = st.session_state.get("live_result")
    if result is None:
        st.write("Choose a company and date range, then load an analysis to see charts.")
        return

    raw = result["raw"]
    st.subheader(f"{result['ticker']} · loaded analysis")
    st.caption(f"{len(raw)} daily rows · {raw.index.min().date()} to {raw.index.max().date()}")
    dates = st.date_input(
        "Chart date range", (raw.index.min().date(), raw.index.max().date()),
        min_value=raw.index.min().date(), max_value=raw.index.max().date(),
        key=f"chart_dates_{result['ticker']}_{raw.index.min()}_{raw.index.max()}",
    )
    if len(dates) != 2:
        st.write("Select both ends of the chart range.")
        return
    view = raw.loc[str(dates[0]):str(dates[1])]
    if view.empty:
        st.warning("No trading rows fall inside the selected chart dates.")
        return

    price_tab, signal_tab = st.tabs(["Prices & volume", "Signals & strategy"])
    with price_tab:
        series = st.multiselect("Price chart series", ["Close", "MA 5d", "MA 20d"], default=["Close", "MA 20d"])
        price_data = raw[["Close"]].copy()
        for window in (5, 20):
            price_data[f"MA {window}d"] = raw["Close"].rolling(window).mean()
        if series:
            st.line_chart(price_data.loc[view.index, series])
        else:
            st.write("Select a price series to display the chart.")
        if "Volume" in view:
            st.write("**Trading volume**")
            st.bar_chart(view[["Volume"]])
        st.dataframe(view)
        st.download_button("Download selected prices", view.to_csv().encode(), file_name=f"{result['ticker']}_prices.csv", mime="text/csv")

    with signal_tab:
        predictions = result["predictions"]
        if predictions is None:
            st.warning(result["model_message"] or "No model predictions are available.")
            return
        # Strategy controls rerun the simulation locally, without new API or model calls.
        confidence = st.slider("Minimum signal confidence", 0.50, 0.90, 0.55, 0.01)
        initial_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10000.0, step=100.0)
        selected_predictions = predictions.loc[str(dates[0]):str(dates[1])]
        if selected_predictions.empty:
            st.warning("No model predictions fall in this chart range. Try a later start or a longer request.")
            return
        backtest, summary = run_strategy_backtest(
            selected_predictions, initial_cash=initial_cash,
            buy_confidence_threshold=confidence, sell_confidence_threshold=confidence,
            high_confidence_threshold=max(0.70, confidence),
        )
        metrics = st.columns(3)
        metrics[0].metric("Final simulated value", f"${summary['final_portfolio_value']:,.2f}")
        metrics[1].metric("Simulated return", f"{summary['total_return']:.2%}")
        metrics[2].metric("Executed trades", str(summary["buy_trades"] + summary["sell_trades"]))
        portfolio = backtest[["Portfolio Value"]].copy()
        portfolio["Buy and hold (no costs)"] = initial_cash * backtest["Close"] / backtest["Close"].iloc[0]
        st.write("**Simulated portfolio and buy-and-hold comparison**")
        st.line_chart(portfolio)
        st.write("**Prediction confidence**")
        st.line_chart(backtest[["Prediction Confidence"]])
        st.write("**Executed action counts**")
        st.bar_chart(backtest["Executed Action"].value_counts().reindex(["BUY", "SELL", "HOLD"], fill_value=0))
        actions = st.multiselect("Actions in table and export", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
        filtered = backtest[backtest["Executed Action"].isin(actions)]
        st.dataframe(filtered)
        st.download_button("Download filtered simulation", filtered.to_csv().encode(), file_name=f"{result['ticker']}_simulation.csv", mime="text/csv")
        st.caption("Simulation resets cash and holdings at the start of the selected chart range. Model training periods may overlap the requested dates; this live simulation is not an independent holdout evaluation.")


if __name__ == "__main__":
    main()
