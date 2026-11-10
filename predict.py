"""
RouteMind - Step 7: score one new order with the frozen Random Forest.

This is the function the agent calls for every new order:

    result = RiskPredictor().score(order)

It returns the risk score plus everything the agent needs to act safely:
  - decision:  HIGH_RISK (risk >= threshold) or MONITOR (below threshold;
               NOT "safe" - about 40% of these orders are late in training)
  - route_to_human + reasons:
        * a critical feature is missing or was never seen in training
        * the score is inside the uncertainty band around the threshold
  - input_issues: any gaps the Pipeline had to fill in
  - top_features: the model's global feature importances (for the
               reasoning card). These are the same for every order; a
               per-order explanation would need an extra library (SHAP).

Run directly to score the orders in sample_orders.json:
    python predict.py
Writes logs/predict_<timestamp>.log
"""

import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from features import build_features, categorical_columns, load_config


class RiskPredictor:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        ev = self.cfg["evaluation"]
        self.threshold = ev["decision_threshold"]
        self.band = ev["uncertainty_band"]
        self.critical = self.cfg["prediction"]["critical_features"]
        self.model = joblib.load(self.cfg["paths"]["rf_model_out"])

        d = self.cfg["data"]
        self.raw_columns = (d["numeric_features"] + d["categorical_features"]
                            + [d["date_column"]])

        # Categories the model saw during training, per categorical column
        prep = self.model.named_steps["prep"]
        encoder = prep.named_transformers_["cat"].named_steps["onehot"]
        self.known = {col: set(cats) for col, cats
                      in zip(categorical_columns(self.cfg), encoder.categories_)}

        # Global feature importance, one-hot columns summed per feature
        totals = {}
        importances = self.model.named_steps["model"].feature_importances_
        for name, imp in zip(prep.get_feature_names_out(), importances):
            part, col = name.split("__", 1)
            if part == "cat":
                col = next(c for c in self.known if col.startswith(c + "_"))
            totals[col] = totals.get(col, 0.0) + float(imp)
        self.top_features = sorted(totals.items(), key=lambda kv: -kv[1])[:3]

    def _check_inputs(self, order):
        """List every gap or unseen value, and which ones are critical."""
        issues, critical_hits = [], []
        date_col = self.cfg["data"]["date_column"]
        for col in self.raw_columns:
            value = order.get(col)
            missing = value is None or (isinstance(value, float) and np.isnan(value))
            if missing:
                issues.append(f"'{col}' is missing")
            elif col in self.known and value not in self.known[col]:
                issues.append(f"'{col}' = '{value}' was never seen in training")
            elif col == date_col and pd.isna(pd.to_datetime(
                    value, format=self.cfg["data"]["date_format"], errors="coerce")):
                issues.append(f"'{col}' = '{value}' is not a readable date")
            else:
                continue
            if col in self.critical:
                critical_hits.append(col)
        return issues, critical_hits

    def score(self, order):
        issues, critical_hits = self._check_inputs(order)

        # One-row table with exactly the columns the model was trained on.
        # None -> NaN so the Pipeline's imputers recognise the gap.
        row = {col: order.get(col) for col in self.raw_columns}
        row = {k: (np.nan if v is None else v) for k, v in row.items()}
        X = build_features(pd.DataFrame([row]), self.cfg)
        risk = float(self.model.predict_proba(X)[0, 1])

        reasons = []
        if critical_hits:
            reasons.append("critical feature missing or unseen: "
                           + ", ".join(critical_hits)
                           + " (risk score not reliable)")
        if abs(risk - self.threshold) <= self.band:
            reasons.append(f"risk {risk:.2f} is inside the uncertainty band "
                           f"{self.threshold - self.band:.2f}-"
                           f"{self.threshold + self.band:.2f}")

        return {
            "order_id": order.get("order_id", "unknown"),
            "risk": round(risk, 3),
            "threshold": self.threshold,
            "decision": "HIGH_RISK" if risk >= self.threshold else "MONITOR",
            "route_to_human": bool(reasons),
            "human_reasons": reasons,
            "input_issues": issues,
            "top_features": [(name, round(share, 3))
                             for name, share in self.top_features],
        }


def main():
    cfg = load_config()
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg["paths"]["log_dir"], f"predict_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    log = logging.getLogger()

    predictor = RiskPredictor(cfg)
    with open(cfg["prediction"]["sample_orders"], "r", encoding="utf-8") as f:
        orders = json.load(f)

    log.info("=" * 64)
    log.info("ROUTEMIND PREDICTION (frozen Random Forest)")
    log.info("=" * 64)
    log.info(f"Model: {cfg['paths']['rf_model_out']} | threshold "
             f"{predictor.threshold} | band +/-{predictor.band}")
    log.info("Top global features: " + ", ".join(
        f"{n} {s:.1%}" for n, s in predictor.top_features))

    for order in orders:
        r = predictor.score(order)
        log.info("-" * 64)
        log.info(f"{r['order_id']}  ({order.get('_note', '')})")
        log.info(f"  Shipping Mode: {order.get('Shipping Mode')} | scheduled "
                 f"days: {order.get('Days for shipment (scheduled)')}")
        log.info(f"  Risk score: {r['risk']:.3f}  ->  {r['decision']}")
        for issue in r["input_issues"]:
            log.info(f"  Input issue: {issue}")
        if r["route_to_human"]:
            for reason in r["human_reasons"]:
                log.info(f"  ROUTE TO HUMAN: {reason}")
        else:
            log.info("  Route to human: no")
    log.info("-" * 64)
    log.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
