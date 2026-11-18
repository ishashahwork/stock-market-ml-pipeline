# Assessment feedback and response

The user supplied an original assessment total of **90/100**. These are the original marks; the changes below have not been regraded.

| Criterion | Original mark | Feedback | Local response |
| --- | --- | --- | --- |
| 1.1 ETL (20%) | 18/20 | Minor feature-engineering bug; verbose error handling | Corrected 5/10/20-day moving-average windows in shared training/live preprocessing; removed repetitive print-and-reraise blocks; added clear input errors and regression checks. |
| 2.1 API Wrapper (20%) | 20/20 | No penalty | Recovered the original wrapper; retained its public interface; reused corrected features, added missing-key validation and request timeouts. Live service access is not retested without credentials. |
| 2.2 Web App (25%) | 21/25 | More charts and interactivity | Added price/moving-average, volume, confidence, portfolio/baseline, and action-count charts; chart-date, confidence, cash, and action controls; downloads and a saved-backtest explorer. Verified controls with mocked data and Streamlit AppTest. |
| 1.2 ML Model (10%) | 10/10 | No penalty | Retained the XGBoost training approach. Saved metadata now identifies the corrected feature version; old models are rejected until retrained. |
| 2.3 Deployment (15%) | 13/15 | URL missing from README | Added a visible deployment-status section and working local launch instructions. The user confirmed no old URL is available. A runnable demo dashboard and new deployment guide now support a fresh publication. No public deployment has been performed, so a verified new URL is still needed. |
| Deliverables (10%) | 8/10 | AI usage log missing | Added `AI_USAGE_LOG.md` with verified post-assessment assistance. The user confirmed that original submission-time AI records are unavailable; the log states that limitation. |

## Required follow-up

- Publish this new app using [the deployment guide](deployment.md), verify its returned URL, and add it to the README. No remote publication was performed during local preparation.
- Regenerate real-data features and retrain models after the moving-average fix. The old fitted artifacts are archived locally and removed from the proposed publication tree; their historical Git versions remain preserved.
- Review the AI log and add team-authored reflections. The original submission history has not been fabricated.

See [live-page documentation](live_analysis.md), [feature documentation](feature_engineering.md), and [verification/provenance notes](publication-notes.md).
