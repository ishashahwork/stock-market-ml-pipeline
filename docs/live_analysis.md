# Live Analysis

Entry point: `src/pages/Live_Analysis.py`, reached through `python -m streamlit run src/Home.py`.

## Workflow

1. Select a configured ticker and request dates in the sidebar form.
2. Press **Load analysis**. The range must cover at least 30 business days; enough complete market rows must also be available for rolling features.
3. Fetch daily SimFin prices once and store the result in the Streamlit session.
4. Sort by date, build the shared features, and align predictions to the actual surviving dates.
5. Explore price/moving-average and volume charts, then model confidence, portfolio comparison, and action counts when compatible models are available.

The **Chart date range**, price-series selector, minimum confidence, starting cash, action filter, and export controls operate on loaded data. They do not trigger a new market request or model prediction. The displayed ticker/date span identifies the loaded result even if the form has been edited before resubmission.

The strategy is recomputed for the selected chart period with new starting cash and no initial holdings. The buy-and-hold comparison invests that cash at the period's first closing price and also excludes costs. The action filter affects the log/export, not the simulation calculation.

## Models and missing data

The corrected preprocessing uses 5/10/20-day moving averages. Models must have matching `feature_version` metadata; older models require feature regeneration and retraining. If compatible models are missing, price and volume exploration still works and the page explains how to enable model signals.

Live inference uses the same feature builder as training but omits future targets, retaining the latest usable row. Row identifiers preserve dates when incomplete features are dropped. This replaces the original fixed `20:-1` alignment in the live page. The separate offline backtest still has its original positional alignment limitation.

## Verification and limits

Streamlit AppTest checks the initial empty state, missing-model behavior, date validation, strategy/filter updates, and that ordinary controls reuse the initial request. Service calls and models are mocked in those checks.

The simulator does not execute trades, account for fees/slippage, or establish an independent holdout score. Requested dates may overlap model training. Daily SimFin prices are not guaranteed real-time. Actual network access requires your own valid API key and suitable data access.

UI controls use Streamlit's [native charts](https://docs.streamlit.io/develop/api-reference/charts/st.line_chart); interaction checks use [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
