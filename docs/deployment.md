# Run and publish the app

This project remains a Python/Streamlit application. The current revision has been built and checked locally. GitHub publication and a new cloud deployment were subsequently authorized; a public app URL will be documented only after hosting succeeds. No old public URL is required to deploy this version.

## Start locally

From the repository root, using Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run src/Home.py
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate `.\.venv\Scripts\Activate.ps1`.

The terminal prints the local URL, normally `http://localhost:8501`. The home dashboard works immediately without credentials, downloaded files, or pretrained artifacts. It generates deterministic synthetic prices and fits compact XGBoost models in memory. That demonstration is explicitly labeled throughout the app and its trade-log export.

On macOS, install `libomp` if XGBoost cannot load its OpenMP runtime. `packages.txt` declares `libgomp1` for a Debian/Linux deployment. Direct Python dependencies are pinned to the versions verified in the local Python 3.11 environment; this is not a complete transitive lockfile.

## Deploy on Render

The root `render.yaml` defines a Python web service on the **Free** compute plan, using Python 3.11.15 and the existing `src/Home.py` entry point. It binds Streamlit to `0.0.0.0` and Render's assigned `PORT`, checks `/_stcore/health`, and limits numerical-library threads for the small instance. Automatic deployments are disabled; subsequent releases are deliberate.

The demo does not require an API key, database, persistent disk, or saved model files. Do not add paid resources to this deployment without the owner's approval. The free service can sleep after inactivity, so its first response after an idle period may be delayed.

If deploying manually, create a Render Blueprint from this GitHub repository and review the explicit Free plan before deploying. When the Render connection is available to the assistant, the same service settings can be used directly through that connection. The real app URL must come from the hosting service; no guessed URL belongs in the README.

Configuration references: [Render web services](https://render.com/docs/web-services), [Blueprint fields](https://render.com/docs/blueprint-spec), [Python version](https://render.com/docs/python-version), and [Free plan behavior](https://render.com/docs/free).

## Create a new Streamlit Community Cloud deployment

After reviewing and publishing the local changes to GitHub:

1. Sign in to Streamlit Community Cloud with the GitHub account that can access the repository.
2. Create an app from `ishashahwork/stock-market-ml-pipeline`, branch `main`.
3. Set the entrypoint/main file path to **`src/Home.py`**.
4. In advanced settings, choose **Python 3.11**, matching the verified environment.
5. Deploy. Demo mode requires no secrets and no generated `ETL/` directory.
6. Open the returned app URL, check the company/period/strategy controls, and add that real URL to the README's demo section and the personal website.

The repository root contains `requirements.txt`, `.streamlit/config.toml`, and Linux `packages.txt`. Keep this single Python dependency manifest for hosting. The old machine-specific Conda export is archived locally and is not a competing deployment manifest.

These steps follow Streamlit's [deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency handling](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), and [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization). An actual cloud build has not been verified from this local task.

## Enable real SimFin prices

The **Live Analysis** page is separate from the synthetic demonstration.

Locally, copy `.env.example` to `.env` and set your own `API_KEY`. On Community Cloud, configure a root-level `API_KEY` in the app's private Secrets settings. The app reads it through `st.secrets`; do not add the secret to Git or put it in a website URL. See Streamlit's [secrets-management instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

Real price charts can run without models. To enable real-company model signals, regenerate real-data features and retrain the models using this revision's feature version. The supplied old models were archived because they used the previous preprocessing. Keep generated data and model artifacts out of the portfolio source tree. For a hosted real-model deployment, provide matching trained artifacts through an owner-managed artifact store or protected deployment process; that distribution is not configured here. Demo models are never substituted for real-company predictions.

## Verification

```bash
python -m unittest discover -s tests -v
```

22 checks passed locally, including app interactions without external requests, shared feature parity, date alignment, stale-model rejection, transaction-fee arithmetic, and CSV-filter behavior. The localhost server and health endpoint returned HTTP 200. Automated Streamlit rendering passed; native browser visual review could not be completed because the browser integration was unavailable and the fallback stalled. Real SimFin service access and a public cloud deployment remain untested.
