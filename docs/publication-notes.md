# Publication and feedback implementation record

Updated November 18, 2026. Working portfolio title: **Stock Market ML Pipeline**. The initial preparation was local. The user subsequently authorized GitHub publication and cloud deployment; the configuration and publication steps are recorded below.

## Source and history

The supplied local folder initially contained only `Executive Summary.pdf`, with no Git history. The first supplied repository, `ishashahwork/python-ii-group-assignment`, contained the older source at `5fd24b5` (March 20, 2026).

The user then supplied `nachimoreno/python-ii-group-assignment`. It contains the later Streamlit app, `PySimFin` wrapper, and original models, plus an earlier Part 2 development branch by Isha. The latest `main` is `858523a` (March 23, 2026), with 28 commits by nachimoreno. `part2` ends at `b872a32` and contains 11 additional commits by ishashahwork. The available histories contain 39 unique commits in total; `part2` is a separate development approach, not the final main source.

The local `main` baseline was fast-forwarded to the later main history, with working file edits preserved. `origin` remains the user's portfolio repository; `upstream` records the collaborator repository. The original baseline is retained on `archive/portfolio-start`; Isha's development branch is retained as local `part2`. No authorship or commit graph was rewritten. See [credits](../CREDITS.md) for evidence-backed contributions.

The PDF is the only supplied report, with internal metadata dated March 22, 2026. Its status as the submitted final report remains unconfirmed. Its contents are unchanged at `docs/reports/executive-summary.pdf`.

## Implemented feedback

- Corrected the 5/10/20-day moving-average windows and replaced duplicate training/live transformations with one shared builder.
- Added numerical regression checks for those windows and shared-feature parity checks. Live inference preserves actual dates after row removal and retains its latest usable observation by omitting future targets.
- Added feature-version metadata and guards to prevent old models from silently running against corrected features. Historical models must be retrained.
- Removed redundant ETL print-and-reraise blocks, added actionable missing-key/company/data checks, and fixed the cleaning row-count message.
- Added live price/moving-average, volume, confidence, portfolio/buy-and-hold, and executed-action charts, plus date, cash, confidence, action-filter, and download controls.
- Made API requests explicit form submissions and reused loaded data/predictions during normal chart and strategy changes.
- Added a saved-backtest explorer with company/range selection, return charts, filters, downloads, empty states, and malformed-data checks.
- Added the required `AI_USAGE_LOG.md`, recording this session and the user's confirmation that original AI-use records are unavailable. No original tools/prompts/reflections were invented.
- Rewrote the README, updated module/in-app documentation, and recorded the original 90/100 assessment and remaining items in [feedback response](feedback-response.md).
- Separated pipeline execution from app launch; app launch uses the active Python environment and a checked subprocess. The devcontainer installs requirements without a system upgrade and retains Streamlit's default request protections.

## Publication organization

The source/configuration paths and ETL filenames are preserved. The report is linked from the README. A source-based pipeline diagram illustrates the available code; it is not a performance chart.

The course brief and machine-specific Conda export are kept in ignored `.local-archive/`, rather than the current publication tree. Historical notebook source is preserved, with generated outputs/execution counters cleared.

The five companies' historical fitted models and metadata are retained in `.local-archive/legacy-trained-models/` and deleted from the proposed working publication tree. Those models used the earlier feature definitions and are unsuitable for corrected inference. Their original Git versions remain available. Newly trained models and data belong under ignored `ETL/`; no secrets or datasets were added.

`.gitignore` excludes credentials, environments, caches, checkpoints, local archives, downloaded data, fitted models, and default generated reports. `.env.example` is an empty, publishable template. The previous broad JSON exclusion was replaced with path/credential exclusions so legitimate project JSON can be committed.

## Verification

| Check | Outcome |
| --- | --- |
| Git integrity (`git fsck --full`) | Passed; 39 original commits preserved across available histories |
| Python syntax | All source and test modules parsed |
| Python 3.11 installation and `pip check` | Passed; no broken requirements |
| CLI help | Passed, including the `app` choice |
| ETL regression checks | Moving-average arithmetic, targets/warmup, and clear missing-input errors passed |
| Feature parity | Training/live predictor equality, date alignment, latest inference row, and old-model rejection passed |
| Streamlit UI checks | Company/range/action filters, empty/malformed results, invalid request dates, missing models, and request reuse passed |
| Combined suite | 22 checks passed using `python -m unittest discover -s tests -v` |
| Offline pipeline smoke | All five stages passed with synthetic input, mocked SimFin calls, and reduced model search |
| PDF | Three pages rendered and visually inspected; file hash unchanged |
| Architecture image | SVG parsed, rendered, and visually inspected |
| Markdown local links, ignore rules, whitespace | Checked locally |
| Credential-pattern scan | 131 historical file blobs and current publication text checked; no credential-pattern matches found |

The synthetic smoke check used 180 rows for one synthetic company and one 10-tree search candidate per model across the original five folds. It verified 10 output files, 31 predictors, 31 holdout rows, finite portfolio values, and nonnegative cash/holdings. It exercised real ingestion conversion, cleaning, features, training, serialization/reload, prediction, and backtesting functions. Temporary data/models were discarded. These checks do not establish real model performance.

The Python 3.11 environment required an OpenMP runtime for XGBoost. Verification used an isolated temporary copy of the machine's existing Anaconda runtime, without changing global packages. The README documents a conventional macOS `libomp` installation. The local Python 3.11 environment resolved pandas 2.2.3, SimFin 1.0.2, Polars 2.1.0, python-dotenv 1.2.4, XGBoost 3.2.0, scikit-learn 1.9.1, PyArrow 26.0.1, NumPy 2.4.6, and Streamlit 1.67.0. These are observations, not a cross-platform lockfile.

Deployment lookup checked both repository histories and public GitHub repository/deployment metadata. Neither repository has a homepage URL or recorded GitHub deployments; the only historical Streamlit URL is a placeholder. This does not prove that a deployment never existed. The actual app URL is still needed.

Original report SHA-256:

```text
53d80bcaeed55811006be2cd70b11b49d48673c1be8c04949251ff15f18760c2
```

## Remaining work

1. Supply and verify the real deployment URL, then link it from the README. Do not substitute a guessed or placeholder URL.
2. Run actual SimFin ingestion with the owner's credentials, regenerate features, retrain models, and regenerate backtests. Live service access and corrected-model performance were not tested.
3. Review offline chronological ordering, fixed-offset price alignment, target overlap at split boundaries, same-close execution assumptions, fees/slippage, and independent evaluation before publishing performance claims.
4. Review the AI log and add team-authored reflections; original submission-time records are unavailable.
5. Confirm the title, report submission status, full team roster, and reuse license if desired.
6. Review and commit the local changes when ready. Publishing both `main` and `part2` will retain both original development histories on the portfolio remote; no force-push or replacement repository is needed. The archived original baseline can also be retained remotely if desired.

## Subsequent runnable app build

The user confirmed no existing deployment URL was available and requested a working app. The home page is now an interactive synthetic-data dashboard with five company labels, four history windows, price overlays, executed-trade markers, a fee-aware portfolio comparison, feature importance, model details, filtered exports, project credits, and the original report download. Models are trained in memory on earlier synthetic observations; displayed predictions are later than the training cutoff with one row left between inputs. No synthetic artifacts replace real-company models. Real SimFin access remains a separate page and accepts host-managed secrets.

Direct dependencies were pinned to the verified local versions, a Streamlit theme and Linux OpenMP package declaration were added, and `docs/deployment.md` documents new publication using `src/Home.py`. No previous URL is needed, but no cloud deployment was performed. The expanded suite passed 22 checks. A local Streamlit server returned HTTP 200 for the app and health endpoint. Browser visual QA remained incomplete because the browser integration was unavailable and its native fallback stalled.

## Publication authorization and Render configuration

The user asked the assistant to carry out deployment. The existing public GitHub repository is accessible through the owner account. `render.yaml` defines one Free-plan Python service using the reviewed application, explicit Python version, assigned port, health endpoint, and bounded numerical-library threads. No paid resources or credentials are included. A confirmed Render connection is required before service creation, and a successful returned URL is required before claiming the app is live. Both original main and Part 2 histories will be preserved by fast-forward publication.
