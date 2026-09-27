# Campus Night Safety — full-stack app

One backend, three sections, matching what you asked for:

1. **Upload & Classify** — upload an actual photo. The backend sends it to Claude's vision model, which extracts lighting, isolation, vegetation density, and foot traffic directly from the image. You only fill in what a photo can't show: Wi-Fi availability, signal strength, lat/lon, and route. The point is scored immediately (rule-based), and with the best trained ML model as soon as one exists.
2. **Dashboard** — Leaflet map of every point, colored by risk, plus per-route average risk and a ranked list.
3. **Compare with Form** — add anonymous-form responses (location text + tags); the backend trains 3 regression models (Linear Regression, Random Forest, Gradient Boosting) that predict the **form-reported risk** from **photo features**, shows which one is best, explains it with SHAP, and retrains automatically every time you add a point or a form response.

## Run it locally

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...

uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — FastAPI serves the frontend directly, no separate server needed.

**About the API key:** photo classification calls the Anthropic API directly from the backend, using whatever key is in `ANTHROPIC_API_KEY`. Get one from https://console.anthropic.com — it needs billing set up, and each photo upload is a small paid API call (a single-image classification like this is inexpensive, but it isn't free). If the key isn't set, uploads will fail with a clear error message rather than silently doing nothing.

## How the ML actually works here

- **Target (y):** the average form-derived risk score for a route (from tags like "Isolated area", "Lighting issue").
- **Features (X):** lighting, isolation score, vegetation density, foot traffic, Wi-Fi availability, signal strength — all from the photo-classification step.
- Each new point or form entry triggers `retrain_models()`, which refits all 3 models and picks the one with the lowest error.
- With fewer than 5 matched samples, there's no held-out validation — metrics are in-sample only, and the API flags this (`insufficient_data: true`).
- SHAP explains the **best** model's predictions: which photo features push the predicted risk up or down, both per-point and on average.

## Known limitations to fix before this is "real"

- **Route matching is keyword-based** (e.g. "disang" → Disang Main Road). Your real form has many free-text locations that won't match any of the 5 known routes — either add more routes/keywords, or put a route dropdown on the form itself.
- **SQLite file resets** on most free hosts' redeploys (ephemeral disk). Fine for now; move to a hosted Postgres (e.g. Supabase, Render Postgres) before this holds real production data.
- **No actual photo → feature pipeline yet.** Right now you type in the classified features by hand. Wiring a real image-classification step (e.g. via the Claude API with vision) is the next piece — happy to add it once you're ready.
- **Small data = noisy models.** Until you have real volume, treat model outputs and SHAP explanations as directional, not authoritative.

## Deploying (e.g. Render, free tier)

1. Push this folder to a GitHub repo.
2. New Web Service on Render → connect the repo, root directory `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. In Render's dashboard, add an environment variable: `ANTHROPIC_API_KEY` = your key.
6. Done — same URL serves both the API and the dashboard.

Uploaded photos are saved to `backend/uploads/` and served at `/uploads/<file>`. On most free hosts this disk is ephemeral (wiped on redeploy), same caveat as the SQLite file below — fine while testing, worth moving to persistent/object storage before this is load-bearing.
