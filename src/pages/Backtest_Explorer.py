"""Explore locally generated backtests without downloading data or running trades."""

from pathlib import Path
import tomllib

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


def main():
    st.set_page_config(page_title="Stock Market ML | Results", page_icon="📈", layout="wide")
    st.title("Stock Market ML")
    st.caption("Explore the offline pipeline's simulated trading results.")
    st.info("Historical simulation. Model and execution limitations are documented in the README.")

    try:
        with (ROOT / "src/config.toml").open("rb") as config_file:
            config = tomllib.load(config_file)
        backtest_dir = ROOT / config["GOLD_DIR"] / "backtests"
        company_files = {}
        for company in config["companies"]:
            folder = company.split()[0].title()
            path = backtest_dir / folder / "holdout_backtest.csv"
            if path.is_file():
                company_files[company] = path
    except (OSError, KeyError, IndexError, tomllib.TOMLDecodeError) as error:
        st.error(f"Unable to load the results configuration: {error}")
        return

    if not company_files:
        st.warning("No backtest results are available yet.")
        st.write("Generate results with the offline pipeline, then refresh this page.")
        st.code("python manage.py --stage all", language="bash")
        return

    import pandas as pd

    company = st.sidebar.selectbox("Company", list(company_files))
    try:
        data = pd.read_csv(company_files[company])
    except (OSError, ValueError) as error:
        st.error(f"Unable to read results for {company}: {error}")
        return

    required = {"portfolio_value", "Close", "predicted_return", "Target Return Metric", "executed_action"}
    if missing := required.difference(data.columns):
        st.error("This backtest is missing required columns: " + ", ".join(sorted(missing)))
        return
    if data.empty:
        st.warning("This company's backtest contains no rows.")
        return
    for column in required - {"executed_action"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any() or data[column].isin([float("inf"), float("-inf")]).any():
            st.error(f"The backtest has invalid numeric values in {column}. Regenerate the results.")
            return

    data.index.name = "Holdout row"
    full_start, full_end = data.index.min(), data.index.max()
    if len(data) > 1:
        start, end = st.sidebar.slider(
            "Chart row range", int(full_start), int(full_end), (int(full_start), int(full_end))
        )
    else:
        start, end = full_start, full_end
    window = data.loc[start:end]
    st.sidebar.caption("Charts use row order because the current pipeline does not preserve dates in its backtest output.")
    st.subheader(company)
    st.caption(f"Showing holdout rows {start}–{end}. Hover over a chart to inspect values.")

    first = float(window["portfolio_value"].iloc[0])
    last = float(window["portfolio_value"].iloc[-1])
    change = (last / first - 1) if first > 0 else None
    trades = window["executed_action"].isin(["BUY", "SELL"]).sum()
    cards = st.columns(3)
    cards[0].metric("Portfolio at window end", f"${last:,.2f}")
    cards[1].metric("Portfolio change in window", f"{change:.2%}" if change is not None else "Unavailable")
    cards[2].metric("Executed trades in window", str(trades))

    price_tab, return_tab, trade_tab = st.tabs(["Price & portfolio", "Predicted vs actual", "Trades"])
    with price_tab:
        st.write("**Closing price**")
        st.line_chart(window[["Close"]])
        st.write("**Simulated portfolio value**")
        st.line_chart(window[["portfolio_value"]])
    with return_tab:
        st.write("**Next-day return (%)**")
        chart = window[["predicted_return", "Target Return Metric"]].rename(
            columns={"predicted_return": "Predicted", "Target Return Metric": "Actual"}
        ) * 100
        st.line_chart(chart)
        st.caption("The predictions and actual targets come from the saved holdout backtest.")
    with trade_tab:
        counts = window["executed_action"].value_counts().reindex(["BUY", "SELL", "HOLD"], fill_value=0)
        st.bar_chart(counts.rename("Rows"))
        selected = st.multiselect("Actions in table and export", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
        filtered = window[window["executed_action"].isin(selected)]
        st.caption(f"{len(filtered)} rows match the action filter. The filter affects the table and export; charts use the selected row range.")
        st.dataframe(filtered)
        st.download_button(
            "Download filtered rows", filtered.to_csv(index=True).encode("utf-8"),
            file_name=f"{company.split()[0].lower()}_filtered_backtest.csv", mime="text/csv",
        )


if __name__ == "__main__":
    main()
