"""
Campus Night Safety — backend
Trains regression models for TWO targets:
  1. overall risk   — predicted from photo scene features + network readings,
                       matched against form-reported risk per route.
  2. network risk    — predicted from photo SCENE features alone (lighting,
                       isolation, vegetation, foot traffic), to see whether the
                       physical character of a spot correlates with dead zones.
                       The label itself is computed directly from the Wi-Fi/
                       signal readings taken at that point (not guessed).

Photo classification (including isolation/enclosure) is done ENTIRELY by
Groq's vision API when GROQ_API_KEY is set — there is no manual isolation
override anymore; that judgment call belongs to the backend model, not a
human dropdown. If Groq is unavailable, a local pixel-analysis fallback
still gives lighting/vegetation/foot-traffic, and isolation defaults to a
neutral placeholder (with a clear label so nobody mistakes it for a real
read).

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY=gsk_...          # optional but recommended
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000
(index.html must sit in this same folder, next to main.py)
"""
import base64
import io
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

import shap

DB_PATH = os.path.join(os.path.dirname(__file__), "campus_safety.db")
FRONTEND_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Groq's free-tier multimodal lineup changes fairly often — if this model gets
# deprecated, check https://console.groq.com/docs/vision for the current one
# and swap the string below. Everything else stays the same.
##GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
app = FastAPI(title="Campus Night Safety API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# ---------------------------------------------------------------- encoding
LIGHT_ORD = {"bright": 0, "moderate": 1, "dim": 2, "dark": 3}
VEG_ORD = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}
FOOT_ORD = {"high": 0, "moderate": 1, "low": 2, "none": 3, "not_visible": 1}
SIGNAL_ORD = {"strong": 0, "moderate": 1, "weak": 2}

# Full feature set for the OVERALL RISK model — scene features plus the
# network readings taken at the spot (Wi-Fi availability, signal strength).
FEATURE_NAMES = ["lighting", "isolation_score", "vegetation", "foot_traffic", "wifi_absent", "signal_weak"]

# Feature set for the NETWORK RISK model — scene-only. Wi-Fi/signal are
# deliberately excluded here since they're what we're trying to predict;
# including them would make the "prediction" trivial.
SCENE_FEATURE_NAMES = ["lighting", "isolation_score", "vegetation", "foot_traffic"]

KNOWN_ROUTES = ["Disang Main Road", "Dhansiri Main Road", "Hospital Road", "Unnamed route", "Backside of Lecture Hall 2"]
ROUTE_KEYWORDS = {
    "Disang Main Road": ["disang", "dishang"],
    "Dhansiri Main Road": ["dhansiri"],
    "Hospital Road": ["hospital"],
}
TAG_WEIGHT = {"Isolated area": 3, "Lighting issue": 3, "Animals": 1, "Network issues": 1}


def encode_features(p: dict) -> List[float]:
    """Full 6-feature vector: scene features + network readings."""
    return [
        LIGHT_ORD.get(p["lighting_level"], 1),
        float(p["isolation_score"]),
        VEG_ORD.get(p["vegetation_density"], 1),
        FOOT_ORD.get(p["foot_traffic_observed"], 1),
        0 if p.get("wifi_available") else 1,
        SIGNAL_ORD.get(p.get("signal_strength", "moderate"), 1),
    ]


def encode_scene_features(p: dict) -> List[float]:
    """First 4 entries only — scene features, no network readings."""
    return encode_features(p)[:4]


def rule_based_risk(p: dict) -> float:
    """Fallback overall-safety score, always available even before any model
    is trained. Deliberately folds in Wi-Fi/signal so connectivity counts
    toward overall risk, same as before."""
    f = encode_features(p)
    raw = f[0] * 2 + f[1] * 1.5 + f[2] * 1.5 + (1 if f[3] >= 2 else 0) + f[4] + f[5]
    return round(min(10, raw * (10 / 19)), 1)


def network_risk_score(p: dict) -> float:
    """0–10 risk score for connectivity alone, derived directly from the
    Wi-Fi/signal readings taken at the point (not guessed from the photo).
    This is the ground-truth label the NETWORK RISK model learns to predict
    from scene features."""
    signal_component = SIGNAL_ORD.get(p.get("signal_strength", "moderate"), 1)  # 0 strong .. 2 weak
    wifi_component = 0 if p.get("wifi_available") else 1
    raw = signal_component * 2 + wifi_component * 3  # max 2*2 + 1*3 = 7
    return round(min(10, raw * (10 / 7)), 1)


def match_route(text: str) -> Optional[str]:
    t = text.lower()
    for route, kws in ROUTE_KEYWORDS.items():
        if any(k in t for k in kws):
            return route
    return None


def form_risk(tags: List[str]) -> float:
    s = sum(TAG_WEIGHT.get(t, 0) for t in tags)
    return round(min(10, s * 1.25), 1)


def parse_isolation_score(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"\d+", str(val))
    return float(m.group()) if m else 2.0


# ---------------------------------------------------------------- photo analysis
CLASSIFY_SYSTEM_PROMPT = """You are analyzing a photo taken at night on a campus road, as part of a
night-safety data collection project. Extract the following features based
ONLY on what is visible in the image. Do not guess GPS coordinates,
timestamps, signal strength, or Wi-Fi availability — those come from
device sensors and manual input, not the photo.

Return ONLY valid JSON, no preamble, no markdown fences, in exactly this shape:
{
  "lighting_level": "dark | dim | moderate | bright",
  "light_source_functional": "yes | no | unclear",
  "isolation_score": 1,
  "sightline_obstruction": "none | partial | full",
  "enclosure_type": "open | narrow_passage | blind_corner | fenced_enclosed",
  "vegetation_density": "none | light | moderate | heavy",
  "surface_condition": "good | uneven | broken | not_visible",
  "foot_traffic_observed": "none | low | moderate | high",
  "weather_condition": "clear | rainy_wet | foggy | unclear",
  "nearby_landmark": "brief description of a visible sign, building, or distinguishing feature",
  "notable_features": "anything else relevant to safety: obstructions, people, vehicles, construction",
  "confidence_note": "one line flagging anything ambiguous or hard to judge from this image"
}
isolation_score must be the integer 1 (open sightlines), 2 (partial), or 3 (enclosed/blind spot) — a bare number, not a description.
This isolation_score is used directly by the backend with no human review, so judge it as
carefully and consistently as you can from the visual evidence alone.
Base every judgment strictly on visible evidence. If something can't be determined, say so in confidence_note rather than guessing."""



def classify_photo_groq(image_bytes: bytes, media_type: str) -> dict:
    import requests  # plain HTTP call — avoids the groq SDK's client-construction bug

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{media_type};base64,{b64}"

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Classify this campus night photo per the schema in the system prompt."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    result = resp.json()
    text = (result["choices"][0]["message"]["content"] or "").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Groq did not return parseable JSON: {text[:200]}")
    data = json.loads(match.group())
    data["isolation_score"] = parse_isolation_score(data.get("isolation_score", 2))
    return data


def analyze_photo_local(image_bytes: bytes) -> dict:
    """Free, fully offline fallback — brightness for lighting, green-pixel ratio
    for vegetation, optional OpenCV person-count for foot traffic. Isolation
    can't be read from pixel stats alone, so it's defaulted to a neutral
    placeholder (2 — partial) rather than asked of the user; this is always
    a backend decision, never a manual override."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    small = img.resize((200, max(1, int(200 * img.height / max(img.width, 1)))))
    arr = np.asarray(small).astype(float)

    luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    brightness = float(luminance.mean())
    if brightness < 45:
        lighting_level = "dark"
    elif brightness < 90:
        lighting_level = "dim"
    elif brightness < 150:
        lighting_level = "moderate"
    else:
        lighting_level = "bright"

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_mask = (g > r + 12) & (g > b + 12) & (g > 40)
    green_ratio = float(green_mask.mean())
    if green_ratio < 0.06:
        vegetation_density = "none"
    elif green_ratio < 0.18:
        vegetation_density = "light"
    elif green_ratio < 0.35:
        vegetation_density = "moderate"
    else:
        vegetation_density = "heavy"

    people_count = None
    foot_traffic_observed = "not_visible"
    try:
        import cv2
        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, _ = hog.detectMultiScale(cv_img, winStride=(8, 8))
        people_count = len(rects)
        foot_traffic_observed = "none" if people_count == 0 else "low" if people_count <= 2 else "moderate" if people_count <= 5 else "high"
    except ImportError:
        pass

    return {
        "lighting_level": lighting_level,
        "isolation_score": 2,
        "vegetation_density": vegetation_density,
        "foot_traffic_observed": foot_traffic_observed,
        "avg_brightness": round(brightness, 1),
        "green_ratio": round(green_ratio, 3),
        "people_detected": people_count,
        "confidence_note": "Local pixel-stat fallback (Groq unavailable) — isolation defaulted to 'partial' (2); enable Groq for a real read.",
    }


def get_photo_features(image_bytes: bytes, media_type: str) -> dict:
    """Groq vision first (free tier), silently falls back to local pixel analysis
    if no key is set or the Groq call fails for any reason (rate limit, model
    deprecation, network issue, etc). The caller can check `_source` to see which
    path was used. Isolation always comes from one of these two — never from a
    manual field."""
    if os.environ.get("GROQ_API_KEY"):
        try:
            result = classify_photo_groq(image_bytes, media_type)
            result["_source"] = "groq"
            return result
        except Exception as e:
            fallback = analyze_photo_local(image_bytes)
            fallback["_source"] = "local-fallback"
            fallback["_groq_error"] = str(e)
            return fallback
    fallback = analyze_photo_local(image_bytes)
    fallback["_source"] = "local-no-key"
    return fallback


# ---------------------------------------------------------------- db
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS points(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route TEXT, segment TEXT, lat REAL, lon REAL,
                lighting_level TEXT, isolation_score REAL, vegetation_density TEXT,
                foot_traffic_observed TEXT, wifi_available INTEGER, signal_strength TEXT,
                rule_risk REAL, ml_risk REAL, verified INTEGER DEFAULT 0,
                photo_path TEXT, classification_json TEXT,
                created_at TEXT)"""
        )
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(points)")}
        for col, coltype in [
            ("photo_path", "TEXT"), ("classification_json", "TEXT"),
            ("network_risk", "REAL"), ("network_risk_ml", "REAL"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE points ADD COLUMN {col} {coltype}")
        if "network_risk" not in existing_cols:
            # one-time backfill for rows inserted before this column existed
            for row in conn.execute("SELECT id, wifi_available, signal_strength FROM points"):
                nr = network_risk_score({"wifi_available": bool(row[1]), "signal_strength": row[2]})
                conn.execute("UPDATE points SET network_risk=? WHERE id=?", (nr, row[0]))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS form_responses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_text TEXT, matched_route TEXT, tags TEXT,
                form_risk REAL, created_at TEXT)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS model_metrics(
                model_name TEXT PRIMARY KEY, mae REAL, r2 REAL, is_best INTEGER,
                n_samples INTEGER, insufficient_data INTEGER, trained_at TEXT)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS network_model_metrics(
                model_name TEXT PRIMARY KEY, mae REAL, r2 REAL, is_best INTEGER,
                n_samples INTEGER, insufficient_data INTEGER, trained_at TEXT)"""
        )


init_db()
STATE = {
    "models": {}, "shap_summary": {}, "shap_per_point": {}, "trained": False, "best_model_name": None,
    "network_models": {}, "network_shap_summary": {}, "network_shap_per_point": {},
    "network_trained": False, "network_best_model_name": None,
}


# ---------------------------------------------------------------- training
def build_training_set():
    """Overall risk target: photo points matched to their route's average
    form-reported risk."""
    with db() as conn:
        points = [dict(r) for r in conn.execute("SELECT * FROM points")]
        forms = [dict(r) for r in conn.execute("SELECT * FROM form_responses")]
    route_form_avg = {}
    for route in set(f["matched_route"] for f in forms if f["matched_route"]):
        vals = [f["form_risk"] for f in forms if f["matched_route"] == route]
        route_form_avg[route] = sum(vals) / len(vals)
    X, y, ids = [], [], []
    for p in points:
        if p["route"] in route_form_avg:
            X.append(encode_features(p))
            y.append(route_form_avg[p["route"]])
            ids.append(p["id"])
    return np.array(X, dtype=float), np.array(y, dtype=float), ids


def build_network_training_set():
    """Network risk target: every point's scene features, labeled with the
    network_risk computed from its own Wi-Fi/signal reading. No form data
    needed — the label lives on the point itself."""
    with db() as conn:
        points = [dict(r) for r in conn.execute("SELECT * FROM points WHERE network_risk IS NOT NULL")]
    X, y, ids = [], [], []
    for p in points:
        X.append(encode_scene_features(p))
        y.append(p["network_risk"])
        ids.append(p["id"])
    return np.array(X, dtype=float), np.array(y, dtype=float), ids


def _fresh_candidates():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, random_state=42),
    }


def _fit_target_models(X: np.ndarray, y: np.ndarray, label: str):
    """Train the 3 candidate regressors for one target. Returns a dict with
    either {'trained': False, 'reason': ...} or {'trained': True, 'results':
    {name: {'model', 'mae', 'r2'}}, 'best_name': ..., 'n': ..., 'insufficient': ...}."""
    n = len(y)
    if n < 2:
        return {"trained": False, "reason": f"Only {n} sample(s) available for {label} — need at least 2 to train anything."}

    candidates = _fresh_candidates()
    insufficient = n < 5
    results = {}
    for name, model in candidates.items():
        if n >= 5:
            loo = LeaveOneOut()
            preds = np.zeros(n)
            for tr, te in loo.split(X):
                m = clone(model)
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
        else:
            model.fit(X, y)
            preds = model.predict(X)
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds) if n > 1 else 0.0
        model.fit(X, y)  # final fit on everything
        results[name] = {"model": model, "mae": float(mae), "r2": float(r2)}

    best_name = min(results, key=lambda k: results[k]["mae"])
    return {"trained": True, "results": results, "best_name": best_name, "n": n, "insufficient": insufficient}


def _compute_shap(model, X: np.ndarray, ids: list, feature_names: list):
    try:
        explainer = shap.Explainer(model.predict, X)
        sv = explainer(X)
        importance = np.abs(sv.values).mean(axis=0)
        summary = {feature_names[i]: round(float(importance[i]), 3) for i in range(len(feature_names))}
        per_point = {
            ids[i]: {feature_names[j]: round(float(sv.values[i][j]), 3) for j in range(len(feature_names))}
            for i in range(len(ids))
        }
        return summary, per_point
    except Exception:
        return {}, {}


def _store_metrics(table: str, results: dict, best_name: str, n: int, insufficient: bool):
    with db() as conn:
        conn.execute(f"DELETE FROM {table}")
        for name, r in results.items():
            conn.execute(
                f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?)",
                (name, round(r["mae"], 3), round(r["r2"], 3), 1 if name == best_name else 0,
                 n, 1 if insufficient else 0, datetime.utcnow().isoformat()),
            )


def retrain_models():
    # ---------------------------------------------------- overall risk target
    X, y, ids = build_training_set()
    overall = _fit_target_models(X, y, "overall risk")
    if not overall["trained"]:
        STATE["trained"] = False
        STATE["best_model_name"] = None
        STATE["shap_summary"], STATE["shap_per_point"] = {}, {}
        with db() as conn:
            conn.execute("DELETE FROM model_metrics")
        overall_out = {"trained": False, "reason": overall["reason"]}
    else:
        best_name = overall["best_name"]
        results = overall["results"]
        STATE["models"] = {k: v["model"] for k, v in results.items()}
        STATE["best_model_name"] = best_name
        STATE["trained"] = True
        _store_metrics("model_metrics", results, best_name, overall["n"], overall["insufficient"])

        best_model = results[best_name]["model"]
        STATE["shap_summary"], STATE["shap_per_point"] = _compute_shap(best_model, X, ids, FEATURE_NAMES)

        with db() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM points")]
            for p in rows:
                pred = float(best_model.predict([encode_features(p)])[0])
                conn.execute("UPDATE points SET ml_risk=? WHERE id=?", (round(pred, 1), p["id"]))

        overall_out = {
            "trained": True, "best_model": best_name, "n_samples": overall["n"], "insufficient_data": overall["insufficient"],
            "metrics": {k: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)} for k, v in results.items()},
        }

    # ---------------------------------------------------- network risk target
    Xn, yn, idsn = build_network_training_set()
    network = _fit_target_models(Xn, yn, "network risk")
    if not network["trained"]:
        STATE["network_trained"] = False
        STATE["network_best_model_name"] = None
        STATE["network_shap_summary"], STATE["network_shap_per_point"] = {}, {}
        with db() as conn:
            conn.execute("DELETE FROM network_model_metrics")
        network_out = {"trained": False, "reason": network["reason"]}
    else:
        best_name_n = network["best_name"]
        results_n = network["results"]
        STATE["network_models"] = {k: v["model"] for k, v in results_n.items()}
        STATE["network_best_model_name"] = best_name_n
        STATE["network_trained"] = True
        _store_metrics("network_model_metrics", results_n, best_name_n, network["n"], network["insufficient"])

        best_model_n = results_n[best_name_n]["model"]
        STATE["network_shap_summary"], STATE["network_shap_per_point"] = _compute_shap(best_model_n, Xn, idsn, SCENE_FEATURE_NAMES)

        with db() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM points")]
            for p in rows:
                pred = float(best_model_n.predict([encode_scene_features(p)])[0])
                conn.execute("UPDATE points SET network_risk_ml=? WHERE id=?", (round(pred, 1), p["id"]))

        network_out = {
            "trained": True, "best_model": best_name_n, "n_samples": network["n"], "insufficient_data": network["insufficient"],
            "metrics": {k: {"mae": round(v["mae"], 3), "r2": round(v["r2"], 3)} for k, v in results_n.items()},
        }

    return {**overall_out, "network": network_out}


# ---------------------------------------------------------------- schemas
class PointIn(BaseModel):
    route: str
    segment: str
    lat: float
    lon: float
    lighting_level: str
    isolation_score: float
    vegetation_density: str
    foot_traffic_observed: str
    wifi_available: bool = False
    signal_strength: str = "moderate"
    verified: bool = False


class FormIn(BaseModel):
    location_text: str
    tags: List[str] = []


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "trained": STATE["trained"],
        "best_model": STATE.get("best_model_name"),
        "network_trained": STATE["network_trained"],
        "network_best_model": STATE.get("network_best_model_name"),
        "known_routes": KNOWN_ROUTES,
        "vision_backend": "groq" if os.environ.get("GROQ_API_KEY") else "local-pixel-analysis",
    }


@app.post("/api/points")
def add_point(p: PointIn):
    d = p.dict()
    rr = rule_based_risk(d)
    nr = network_risk_score(d)
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO points
               (route,segment,lat,lon,lighting_level,isolation_score,vegetation_density,
                foot_traffic_observed,wifi_available,signal_strength,rule_risk,ml_risk,verified,
                network_risk,network_risk_ml,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (p.route, p.segment, p.lat, p.lon, p.lighting_level, p.isolation_score, p.vegetation_density,
             p.foot_traffic_observed, int(p.wifi_available), p.signal_strength, rr, None, int(p.verified),
             nr, None, datetime.utcnow().isoformat()),
        )
        new_id = cur.lastrowid
    train_result = retrain_models()
    with db() as conn:
        row = dict(conn.execute("SELECT * FROM points WHERE id=?", (new_id,)).fetchone())
    return {
        "point": row, "retrain": train_result,
        "shap": STATE.get("shap_per_point", {}).get(new_id),
        "network_shap": STATE.get("network_shap_per_point", {}).get(new_id),
    }


@app.post("/api/points/from-photo")
async def add_point_from_photo(
    route: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    wifi_available: bool = Form(False),
    signal_strength: str = Form("moderate"),
    segment: Optional[str] = Form(None),
    photo: UploadFile = File(...),
):
    """Isolation/enclosure is NEVER taken from the client here — it's decided
    entirely by classify_photo_groq / analyze_photo_local on the backend.
    Wi-Fi availability and signal strength stay manual/sensor inputs since a
    photo genuinely can't show those."""
    image_bytes = await photo.read()
    media_type = photo.content_type or "image/jpeg"

    try:
        classification = get_photo_features(image_bytes, media_type)
    except Exception as e:
        raise HTTPException(500, f"Photo analysis failed (is the file a valid image?): {e}")

    ext = os.path.splitext(photo.filename or "")[1] or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    with open(os.path.join(UPLOADS_DIR, fname), "wb") as f:
        f.write(image_bytes)
    photo_path = f"/uploads/{fname}"

    iso = parse_isolation_score(classification.get("isolation_score", 2))

    point_features = {
        "lighting_level": classification.get("lighting_level", "moderate"),
        "isolation_score": iso,
        "vegetation_density": classification.get("vegetation_density", "light"),
        "foot_traffic_observed": classification.get("foot_traffic_observed", "low"),
        "wifi_available": wifi_available,
        "signal_strength": signal_strength,
    }
    rr = rule_based_risk(point_features)
    nr = network_risk_score(point_features)
    final_segment = segment or classification.get("nearby_landmark") or "unnamed point"

    with db() as conn:
        cur = conn.execute(
            """INSERT INTO points
               (route,segment,lat,lon,lighting_level,isolation_score,vegetation_density,
                foot_traffic_observed,wifi_available,signal_strength,rule_risk,ml_risk,verified,
                photo_path,classification_json,network_risk,network_risk_ml,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (route, final_segment, lat, lon, point_features["lighting_level"], point_features["isolation_score"],
             point_features["vegetation_density"], point_features["foot_traffic_observed"], int(wifi_available),
             signal_strength, rr, None, 0, photo_path, json.dumps(classification), nr, None,
             datetime.utcnow().isoformat()),
        )
        new_id = cur.lastrowid

    train_result = retrain_models()
    with db() as conn:
        row = dict(conn.execute("SELECT * FROM points WHERE id=?", (new_id,)).fetchone())
    return {
        "point": row,
        "classification": classification,
        "retrain": train_result,
        "shap": STATE.get("shap_per_point", {}).get(new_id),
        "network_shap": STATE.get("network_shap_per_point", {}).get(new_id),
    }


@app.get("/api/points")
def list_points():
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM points ORDER BY rule_risk DESC")]
    
@app.delete("/api/points/{point_id}")
def delete_point(point_id: int):
    with db() as conn:
        row = conn.execute("SELECT id, photo_path FROM points WHERE id=?", (point_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Point not found.")
        # clean up the stored photo file if one exists
        if row["photo_path"]:
            fpath = os.path.join(os.path.dirname(__file__), row["photo_path"].lstrip("/"))
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        conn.execute("DELETE FROM points WHERE id=?", (point_id,))
    train_result = retrain_models()
    return {"deleted": point_id, "retrain": train_result}

@app.get("/api/routes/summary")
def routes_summary():
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM points")]
    summary = {}
    for p in rows:
        s = summary.setdefault(p["route"], {
            "route": p["route"], "count": 0,
            "rule_avg": 0.0, "ml_avg": 0.0, "ml_count": 0,
            "network_avg": 0.0, "network_ml_avg": 0.0, "network_ml_count": 0,
        })
        s["count"] += 1
        s["rule_avg"] += p["rule_risk"]
        if p["ml_risk"] is not None:
            s["ml_avg"] += p["ml_risk"]
            s["ml_count"] += 1
        if p.get("network_risk") is not None:
            s["network_avg"] += p["network_risk"]
        if p.get("network_risk_ml") is not None:
            s["network_ml_avg"] += p["network_risk_ml"]
            s["network_ml_count"] += 1
    out = []
    for s in summary.values():
        s["rule_avg"] = round(s["rule_avg"] / s["count"], 1)
        s["ml_avg"] = round(s["ml_avg"] / s["ml_count"], 1) if s["ml_count"] else None
        s["network_avg"] = round(s["network_avg"] / s["count"], 1)
        s["network_ml_avg"] = round(s["network_ml_avg"] / s["network_ml_count"], 1) if s["network_ml_count"] else None
        del s["ml_count"], s["network_ml_count"]
        out.append(s)
    return out


@app.post("/api/form")
def add_form(f: FormIn):
    route = match_route(f.location_text)
    fr = form_risk(f.tags)
    with db() as conn:
        conn.execute(
            "INSERT INTO form_responses (location_text,matched_route,tags,form_risk,created_at) VALUES (?,?,?,?,?)",
            (f.location_text, route, json.dumps(f.tags), fr, datetime.utcnow().isoformat()),
        )
    train_result = retrain_models()
    return {"matched_route": route, "form_risk": fr, "retrain": train_result}


@app.get("/api/form")
def list_form():
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM form_responses ORDER BY id DESC")]
    for r in rows:
        r["tags"] = json.loads(r["tags"])
    return rows

@app.post("/api/form/bulk")
async def add_form_bulk(file: UploadFile = File(...)):
    import pandas as pd

    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not read Excel file: {e}")

    loc_col = None
    for col in df.columns:
        if str(col).strip().lower() in ("location_text", "location"):
            loc_col = col
            break
    if loc_col is None:
        for col in df.columns:
            c = str(col).lower()
            if "location" in c or "spot" in c or "unsafe" in c:
                loc_col = col
                break
    if loc_col is None:
        raise HTTPException(
            400,
            f"No location column found. Columns in your file: {list(df.columns)}. "
            f"Rename the relevant column to 'location_text', or tell the dev which column to use."
        )

    tag_col = None
    for col in df.columns:
        c = str(col).lower()
        if "tag" in c or "why" in c or "reason" in c:
            tag_col = col
            break

    KNOWN_TAGS = {"isolated area": "Isolated area", "lighting issue": "Lighting issue",
                  "animals": "Animals", "network issues": "Network issues"}

    def parse_tags(raw: str) -> List[str]:
        if not raw or str(raw).strip().lower() in ("nan", "na", "n/a", ""):
            return []
        parts = re.split(r"[;,]", str(raw))
        tags = []
        for p in parts:
            key = p.strip().lower()
            if key in KNOWN_TAGS:
                tags.append(KNOWN_TAGS[key])
        return tags

    added = []
    with db() as conn:
        for _, row in df.iterrows():
            loc = str(row.get(loc_col, "")).strip()
            if not loc or loc.lower() == "nan":
                continue

            tags = parse_tags(row.get(tag_col, "")) if tag_col else []
            route = match_route(loc)
            fr = form_risk(tags)
            conn.execute(
                "INSERT INTO form_responses (location_text,matched_route,tags,form_risk,created_at) VALUES (?,?,?,?,?)",
                (loc, route, json.dumps(tags), fr, datetime.utcnow().isoformat()),
            )
            added.append({"location_text": loc, "matched_route": route, "tags": tags, "form_risk": fr})

    train_result = retrain_models()
    return {"added": len(added), "rows": added, "retrain": train_result}

@app.post("/api/retrain")
def manual_retrain():
    return retrain_models()


@app.get("/api/models/compare")
def models_compare():
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM model_metrics")]


@app.get("/api/models/compare/network")
def network_models_compare():
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM network_model_metrics")]


@app.get("/api/shap/summary")
def shap_summary():
    return STATE.get("shap_summary", {})


@app.get("/api/shap/point/{point_id}")
def shap_point(point_id: int):
    d = STATE.get("shap_per_point", {}).get(point_id)
    if d is None:
        raise HTTPException(404, "No SHAP explanation for this point yet (needs to be in the trained set).")
    return d


@app.get("/api/shap/network/summary")
def network_shap_summary():
    return STATE.get("network_shap_summary", {})


@app.get("/api/shap/network/point/{point_id}")
def network_shap_point(point_id: int):
    d = STATE.get("network_shap_per_point", {}).get(point_id)
    if d is None:
        raise HTTPException(404, "No network SHAP explanation for this point yet (needs to be in the trained set).")
    return d


# ---------------------------------------------------------------- serve frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")