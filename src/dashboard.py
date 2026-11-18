"""Portfolio-facing demo dashboard, backed by the project's real feature pipeline."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.demo import COMPANIES, build_demo_analysis
from src.PySimFin import run_strategy_backtest

ROOT = Path(__file__).resolve().parents[1]
TEAL = "#087F8C"
CORAL = "#C65B47"
GOLD = "#BD8D32"


@st.cache_data(show_spinner=False)
def demo_analysis(ticker):
    return build_demo_analysis(ticker)


def style_page():
    st.markdown("""<style>
    .block-container {padding-top: 2.1rem; padding-bottom: 2.5rem; max-width: 1400px;}
    .lab-eyebrow {color: #087F8C; font-size: .76rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .65rem;}
    .lab-hero {display: flex; gap: 20px; align-items: center; justify-content: space-between; margin-bottom: 1rem; flex-wrap: wrap;}
    .lab-hero h1 {font-size: clamp(2rem, 3vw, 2.8rem); letter-spacing: -.04em; margin: 0; padding: 0; color: #142D36;}
    .lab-hero p {color: #61747D; margin: .6rem 0 .2rem; font-size: 1rem;}
    .lab-badge {background: #FFF3D7; color: #866319; padding: 9px 14px; border-radius: 25px; font-size: .72rem; font-weight: 700; letter-spacing: .05em;}
    .lab-step {font-size: .85rem; color: #61747D;}
    [data-testid="stMetric"] {background: white; border: 1px solid #DFE7EB; border-radius: 12px; padding: 16px 20px;}
    [data-testid="stMetricLabel"] {color: #61747D;}
    </style>""", unsafe_allow_html=True)


def clean_chart(chart):
    return chart.configure_view(strokeWidth=0).configure_axis(
        gridColor="#EAF0F2", domain=False, labelColor="#71828B", titleColor="#71828B",
        labelFontSize=11, titleFontSize=11,
    ).configure_legend(title=None, orient="top", labelColor="#61747D")


def render_dashboard():
    style_page()
    st.markdown('<div class="lab-eyebrow">STOCK MARKET ML / INTERACTIVE LAB</div>', unsafe_allow_html=True)
    st.markdown('''<div class="lab-hero"><div><h1>Market dashboard</h1>
    <p>Explore prices. Inspect a model. Test a strategy.</p></div>
    <div class="lab-badge">DEMO · SYNTHETIC DATA</div></div>''', unsafe_allow_html=True)
    st.info("Demo data and models are synthetic. This walkthrough uses the project's feature pipeline and XGBoost, but does not show real market prices or project performance.")

    with st.sidebar:
        st.markdown("### Your workspace")
        ticker = st.selectbox("Company", list(COMPANIES), format_func=lambda value: f"{value} · {COMPANIES[value][0]}")
        st.caption("Illustrative company labels · no API key needed")
        st.divider()
        st.markdown("#### Strategy settings")
        confidence = st.slider("Minimum signal confidence", 0.50, 0.90, 0.60, 0.01)
        initial_cash = st.number_input("Starting cash ($)", min_value=100.0, max_value=1_000_000.0, value=10000.0, step=500.0)
        fee = st.number_input("Fee per executed trade ($)", min_value=0.0, max_value=25.0, value=1.0, step=0.5)
        st.caption("Long-only simulation · up to five shares per trade")
        st.divider()
        st.page_link("pages/Live_Analysis.py", label="Connect to SimFin", icon="↗")
        st.page_link("pages/Backtest_Explorer.py", label="Explore saved results", icon="↗")

    try:
        with st.spinner("Preparing the synthetic dataset and demo models…"):
            analysis = demo_analysis(ticker)
    except Exception as error:
        st.error(f"The demo could not initialize: {error}")
        st.caption("Install requirements.txt and the XGBoost OpenMP runtime described in the README, then restart the app.")
        return

    heading, selector = st.columns([2, 1])
    heading.subheader(f"{COMPANIES[ticker][0]} / {ticker}")
    period = selector.radio("History", ["1M", "3M", "6M", "1Y"], index=1, horizontal=True)
    count = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}[period]
    selected = analysis.predictions.tail(count)
    backtest, summary = run_strategy_backtest(
        selected, initial_cash=initial_cash, buy_confidence_threshold=confidence,
        sell_confidence_threshold=confidence, high_confidence_threshold=max(0.70, confidence),
        transaction_fee=fee,
    )
    latest = selected.iloc[-1]
    price_change = selected["Close"].iloc[-1] / selected["Close"].iloc[0] - 1
    direction = "Bullish" if int(latest["Prediction"]) else "Bearish"
    cards = st.columns(4)
    cards[0].metric("Illustrative closing price", f"${latest['Close']:,.2f}", f"{price_change:.2%} over window")
    cards[1].metric("Demo model direction", direction, f"{latest['Prediction Confidence']:.0%} model confidence", delta_color="off")
    cards[2].metric("Simulated portfolio", f"${summary['final_portfolio_value']:,.2f}", f"{summary['total_return']:.2%} after fees")
    cards[3].metric("Executed trades", str(summary["buy_trades"] + summary["sell_trades"]), f"${backtest['Transaction Fee'].sum():,.2f} total fees", delta_color="off")
    st.caption(f"Synthetic observations · {selected.index.min():%d %b %Y} – {selected.index.max():%d %b %Y} · {len(selected)} sessions. All displayed predictions are after the demo training cutoff.")

    market_tab, strategy_tab, model_tab, about_tab = st.tabs(["Market overview", "Strategy simulator", "Model & pipeline", "About this project"])
    with market_tab:
        st.markdown("#### Price trend & executed trades")
        overlays = st.multiselect("Moving-average overlays", [5, 10, 20], default=[20], format_func=lambda value: f"{value} sessions")
        # Calculate overlays on earlier history as well, so the chart does not reset at the window boundary.
        context = analysis.raw.loc[:selected.index.max()].tail(len(selected) + 20)
        chart_prices = context[["Close"]].copy()
        for window in overlays:
            chart_prices[f"MA {window}d"] = chart_prices["Close"].rolling(window).mean()
        st.altair_chart(price_chart_with_history(chart_prices.loc[selected.index], backtest, overlays))
        st.caption("Drag to pan; scroll to zoom. ▲ executed buy · ▼ executed sell. Price labels are illustrative, not market quotes.")
        volume = selected[["Volume"]].reset_index()
        volume_chart = alt.Chart(volume).mark_bar(color="#A0C9CD", cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("Volume:Q", title="Synthetic volume", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("Volume:Q", format=",")],
        ).properties(height=100)
        st.altair_chart(clean_chart(volume_chart))

    with strategy_tab:
        st.markdown("#### Portfolio vs. buy and hold")
        portfolio = backtest[["Portfolio Value"]].copy()
        portfolio["Buy and hold"] = (initial_cash - fee) * backtest["Close"] / backtest["Close"].iloc[0]
        chart_data = portfolio.reset_index().melt("Date", var_name="Strategy", value_name="Value")
        chart = alt.Chart(chart_data).mark_line(strokeWidth=2.5).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title="Simulated value ($)"),
            color=alt.Color("Strategy:N", scale=alt.Scale(domain=["Portfolio Value", "Buy and hold"], range=[TEAL, "#9BA9B7"])),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), "Strategy:N", alt.Tooltip("Value:Q", format="$.2f")],
        ).properties(height=330).interactive()
        st.altair_chart(clean_chart(chart))
        st.caption("The model strategy trades whole shares. The benchmark invests fractional shares once, paying one entry fee. Both mark open holdings to the final close and exclude slippage, taxes, and exit fees.")
        controls, details = st.columns([1, 2])
        with controls:
            st.markdown("#### Executed actions")
            st.bar_chart(backtest["Executed Action"].value_counts().reindex(["BUY", "SELL", "HOLD"], fill_value=0))
        with details:
            st.markdown("#### Trade log")
            actions = st.multiselect("Actions in log & export", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL"])
            log = backtest.loc[backtest["Executed Action"].isin(actions), ["Close", "Prediction Confidence", "Executed Action", "Executed Trade Size", "Transaction Fee", "Cash", "Shares Held", "Portfolio Value"]]
            st.dataframe(log)
            export = log.copy()
            export["Data Source"] = "synthetic demo"
            st.download_button("Download demo trade log", export.to_csv().encode(), file_name=f"{ticker.lower()}_synthetic_demo_trades.csv", mime="text/csv")
        st.caption("Changing a strategy setting resets this window to the chosen starting cash and no shares. Trades execute at the same row's close; this is an educational simplification.")

    with model_tab:
        st.markdown("#### From raw prices to model signals")
        st.image(str(ROOT / "docs/images/pipeline.svg"))
        a, b, c = st.columns(3)
        a.metric("Predictor features", str(analysis.feature_count))
        b.metric("Synthetic training rows", str(analysis.training_rows))
        c.metric("Post-training observations", str(len(analysis.predictions)))
        st.write(f"The demo fits compact XGBoost classification and regression models on synthetic inputs through **{analysis.training_end:%d %b %Y}**. Evaluation starts **{analysis.evaluation_start:%d %b %Y}**, after a one-row gap. These models stay in memory and never replace real-company artifacts.")
        st.markdown("#### Most-used demo features")
        importance = analysis.feature_importance.head(8).rename_axis("Feature").reset_index(name="Importance")
        chart = alt.Chart(importance).mark_bar(color=TEAL, cornerRadiusEnd=3).encode(
            y=alt.Y("Feature:N", sort="-x", title=None), x=alt.X("Importance:Q", title="XGBoost feature importance"),
            tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".3f")],
        ).properties(height=260)
        st.altair_chart(clean_chart(chart))
        st.caption("Importance describes this synthetic model only; it is not evidence that a feature predicts real prices.")
        st.dataframe(selected[["Probability Up", "Prediction Confidence", "Predicted Return", "Actual Next Return"]].tail(10))
        st.caption("The last observed row has no known next-row actual return. Regression values are fractional returns; 0.01 means 1%.")

    with about_tab:
        st.markdown("#### A collaborative Python for Data Analysis II project")
        st.write("The project combines SimFin ingestion, Polars transformations, XGBoost modeling, and a Streamlit interface. This dashboard makes the engineering workflow explorable before credentials and real trained artifacts are configured.")
        st.markdown("**Recorded contributors:** [Isha Shah](https://github.com/ishashahwork) · [nachimoreno](https://github.com/nachimoreno)")
        st.caption("Both development histories and contributor credits are preserved. Demo outputs are not the group's original project results.")
        report = ROOT / "docs/reports/executive-summary.pdf"
        if report.is_file():
            st.download_button("Read the executive summary (PDF)", report.read_bytes(), file_name="executive-summary.pdf", mime="application/pdf")
        st.link_button("View the GitHub project", "https://github.com/ishashahwork/stock-market-ml-pipeline")
        st.write("For real prices, open **Live Analysis** in the page menu. It uses your SimFin API key and models retrained with the corrected feature definition. Saved real-data backtests are available in **Backtest Explorer**.")
        st.caption("The supplied report describes the original system. Public deployment and corrected real-data model performance remain unverified.")


def price_chart_with_history(prices, trades, overlays):
    """Use precomputed rolling means so overlays include pre-window history."""
    series = prices.reset_index().melt("Date", var_name="Series", value_name="Price")
    base = alt.Chart(series).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Price:Q", scale=alt.Scale(zero=False), title="Illustrative price ($)"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Close"] + [f"MA {n}d" for n in overlays], range=[TEAL, GOLD, "#7D8FA9", "#B59CB7"])),
        tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), "Series:N", alt.Tooltip("Price:Q", format="$.2f")],
    )
    executed = trades.loc[trades["Executed Action"].isin(["BUY", "SELL"])].reset_index()
    points = alt.Chart(executed).mark_point(filled=True, size=65).encode(
        x="Date:T", y="Close:Q",
        color=alt.Color("Executed Action:N", scale=alt.Scale(domain=["BUY", "SELL"], range=[TEAL, CORAL]), legend=None),
        shape=alt.Shape("Executed Action:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle-up", "triangle-down"]), legend=None),
        tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), "Executed Action:N", alt.Tooltip("Close:Q", format="$.2f")],
    )
    return clean_chart(alt.layer(base, points).resolve_scale(color="independent").properties(height=330).interactive())
