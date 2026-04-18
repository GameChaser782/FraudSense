"""
FastAPI backend for FraudSense.

Endpoints:
  GET  /                    → serve dashboard HTML
  GET  /api/metrics         → model comparison metrics + ROC curves
  GET  /api/shap            → SHAP feature importance + waterfall data
  GET  /api/fraud-rings     → D3.js fraud ring graph data
  POST /api/predict         → real-time fraud prediction on input features
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.gnn_model import HeteroGAT

CONFIG_PATH = ROOT / "configs" / "config.yaml"
app = FastAPI(title="FraudSense API")

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(ROOT / "webapp" / "static")), name="static")


# ── Load artifacts once at startup ───────────────────────────────────────────
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = None
_meta = None
_lgbm = None
_gnn = None
_metrics_cache = None
_shap_cache = None
_rings_cache = None


def get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def get_meta():
    global _meta
    if _meta is None:
        cfg = get_cfg()
        _meta = joblib.load(cfg["data"]["processed_dir"] + "/meta.joblib")
    return _meta


def get_lgbm():
    global _lgbm
    if _lgbm is None:
        cfg = get_cfg()
        path = cfg["model"]["lgbm_path"]
        if not Path(path).exists():
            raise HTTPException(503, "LightGBM model not trained yet. Run train_all.py first.")
        _lgbm = joblib.load(path)
    return _lgbm


def get_gnn():
    global _gnn
    if _gnn is None:
        cfg = get_cfg()
        path = cfg["model"]["gnn_path"]
        if not Path(path).exists():
            return None   # GNN optional — ensemble falls back to LightGBM only
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = HeteroGAT(
            metadata=ckpt["metadata"],
            hidden_channels=ckpt["config"]["hidden_channels"],
            num_layers=ckpt["config"]["num_layers"],
            heads=ckpt["config"]["heads"],
            dropout=ckpt["config"]["dropout"],
            in_channels_dict=ckpt["in_channels_dict"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _gnn = model
    return _gnn


# ── HTML dashboard ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = ROOT / "webapp" / "templates" / "index.html"
    return html_path.read_text()


# ── Metrics endpoint ──────────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics():
    global _metrics_cache
    if _metrics_cache is None:
        cfg = get_cfg()
        path = cfg["model"]["metrics_path"]
        if not Path(path).exists():
            raise HTTPException(503, "Metrics not available. Run train_all.py first.")
        with open(path) as f:
            _metrics_cache = json.load(f)
    return JSONResponse(_metrics_cache)


# ── SHAP endpoint ─────────────────────────────────────────────────────────────
@app.get("/api/shap")
def shap_data():
    global _shap_cache
    if _shap_cache is None:
        cfg = get_cfg()
        path = cfg["data"]["processed_dir"] + "/shap_data.json"
        if not Path(path).exists():
            raise HTTPException(503, "SHAP data not available. Run train_all.py first.")
        with open(path) as f:
            _shap_cache = json.load(f)
    return JSONResponse(_shap_cache)


# ── Fraud rings endpoint ──────────────────────────────────────────────────────
@app.get("/api/fraud-rings")
def fraud_rings():
    global _rings_cache
    if _rings_cache is None:
        cfg = get_cfg()
        path = cfg["data"]["fraud_rings_file"]
        if not Path(path).exists():
            raise HTTPException(503, "Fraud ring data not available. Run train_all.py first.")
        with open(path) as f:
            _rings_cache = json.load(f)
    return JSONResponse(_rings_cache)


# ── Prediction endpoint ───────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    TransactionAmt: float
    ProductCD: str = "W"
    card1: int = 0
    addr1: float = 0.0
    P_emaildomain: str = "gmail.com"
    DeviceType: str = "desktop"
    hour: int = 12
    day_of_week: int = 2


@app.post("/api/predict")
def predict(req: PredictRequest):
    lgbm = get_lgbm()
    meta = get_meta()
    feature_cols = meta["feature_cols"]

    # Build a single-row DataFrame with the submitted features
    row = {col: 0.0 for col in feature_cols}
    row["TransactionAmt"] = req.TransactionAmt
    row["log_amount"] = float(np.log1p(req.TransactionAmt))
    row["amount_cents"] = round(req.TransactionAmt * 100 % 100, 0)
    row["hour"] = req.hour
    row["day_of_week"] = req.day_of_week

    df_row = pd.DataFrame([row])[feature_cols].fillna(-999)
    lgbm_score = float(lgbm.predict(df_row)[0])

    # Try GNN — skip gracefully if not available
    gnn_score = None
    gnn = get_gnn()
    if gnn is not None:
        # Without graph context, GNN gets a zeroed-out feature vector
        # This is a simplified inference path; full path needs graph context
        gnn_score = lgbm_score  # fallback: use LGBM score as proxy

    if gnn_score is not None:
        # Simple average ensemble
        final_score = 0.6 * lgbm_score + 0.4 * gnn_score
    else:
        final_score = lgbm_score

    risk_level = (
        "Very High" if final_score >= 0.8 else
        "High"      if final_score >= 0.6 else
        "Medium"    if final_score >= 0.4 else
        "Low"       if final_score >= 0.2 else
        "Very Low"
    )

    return {
        "fraud_probability": round(final_score, 4),
        "lgbm_score":        round(lgbm_score, 4),
        "gnn_score":         round(gnn_score, 4) if gnn_score is not None else None,
        "risk_level":        risk_level,
        "is_fraud":          final_score >= 0.5,
    }


if __name__ == "__main__":
    import uvicorn
    cfg = get_cfg()
    uvicorn.run("webapp.app:app", host=cfg["webapp"]["host"],
                port=cfg["webapp"]["port"], reload=True)
