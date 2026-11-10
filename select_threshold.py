"""
RouteMind - Step 5: choose the decision threshold on TRAINING data only.

The Random Forest outputs a risk between 0 and 1. The threshold is the
cut-off at which the agent treats an order as "late".

How we get honest scores without touching the test set:
  cross_val_predict splits the training set into 5 folds. Each order's risk
  is predicted by a model that was trained on the OTHER 4 folds, so every
  training order gets a score from a model that never saw it.

Outputs
  1. Candidate thresholds under different stated rules (F1, F2, precision floor)
  2. Precision-recall curve
  3. Histogram of risk scores (explains why thresholds behave in steps)
  4. Calibration curve (does a risk of 0.8 really mean ~80% late?)
  5. How many orders fall inside each uncertainty band width

Writes logs/threshold_<timestamp>.log and charts to outputs/
"""

import logging
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from features import (build_features, load_config, load_dataset,
                      make_preprocessor, split_data)


def metrics_at(y, risk, t):
    flagged = risk >= t
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(beta):
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {"threshold": t, "recall": recall, "precision": precision,
            "f1": f_beta(1), "f2": f_beta(2),
            "flagged_share": flagged.mean(), "late_missed": fn,
            "false_alarms": fp}


def main():
    cfg = load_config()
    paths, sel = cfg["paths"], cfg["threshold_selection"]
    out = paths["output_dir"]
    os.makedirs(paths["log_dir"], exist_ok=True)
    os.makedirs(out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(paths["log_dir"], f"threshold_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    log = logging.getLogger()

    # Training data only
    df = load_dataset(cfg)
    X = build_features(df, cfg)
    y = df[cfg["data"]["label"]]
    X_train, _, y_train, _ = split_data(X, y, cfg)
    y_arr = y_train.to_numpy()

    model = Pipeline([
        ("prep", make_preprocessor(cfg, scale_numbers=False)),
        ("model", RandomForestClassifier(**cfg["random_forest"])),
    ])
    folds = cfg["evaluation"]["cv_folds"]
    seed = cfg["split"]["random_state"]

    log.info("=" * 64)
    log.info("ROUTEMIND THRESHOLD SELECTION (training data only)")
    log.info("=" * 64)
    log.info(f"Training orders: {len(X_train):,} | late rate {y_arr.mean():.1%} "
             f"| test set not loaded")
    log.info(f"RF settings: {cfg['random_forest']}")
    log.info(f"Out-of-fold probabilities from {folds}-fold cross_val_predict...")

    risk = cross_val_predict(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        method="predict_proba", n_jobs=1)[:, 1]

    # 1. Score every threshold from 0.05 to 0.95
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    table = pd.DataFrame([metrics_at(y_arr, risk, t) for t in grid])
    table.round(4).to_csv(os.path.join(out, "threshold_candidates.csv"), index=False)

    show = table[table["threshold"].isin(
        [0.10, 0.20, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.50,
         0.60, 0.70, 0.75, 0.80])]
    log.info("\n1. THRESHOLD TABLE (selected rows; full grid in "
             "outputs/threshold_candidates.csv)")
    log.info(show.round(3).to_string(index=False))

    always = metrics_at(y_arr, risk, 0.0)
    log.info(f"\n   Reference - flag every order as late: recall "
             f"{always['recall']:.3f}, precision {always['precision']:.3f}, "
             f"F1 {always['f1']:.3f}, F2 {always['f2']:.3f}")

    # 2. What each candidate rule would pick
    log.info("\n2. WHAT EACH RULE WOULD CHOOSE")
    picks = {}
    for name in ["f1", "f2"]:
        best = table.loc[table[name].idxmax()]
        picks[f"Highest {name.upper()}"] = best
    for floor in sel["precision_floors"]:
        ok = table[table["precision"] >= floor]
        if len(ok):
            # highest recall; if tied, the lowest threshold reaching it
            best = ok.sort_values(["recall", "threshold"],
                                  ascending=[False, True]).iloc[0]
            picks[f"Highest recall with precision >= {floor:.2f}"] = best
    for rule, r in picks.items():
        log.info(f"   {rule:<42} -> threshold {r['threshold']:.2f} | recall "
                 f"{r['recall']:.3f} | precision {r['precision']:.3f} | "
                 f"flags {r['flagged_share']:.1%} of orders")

    # 3. Uncertainty band: share of orders within +/- width of each pick
    log.info("\n3. UNCERTAINTY BAND (share of training orders inside "
             "threshold +/- width; these would go to a human)")
    cands = sorted({round(float(r["threshold"]), 2) for r in picks.values()} | {0.5})
    band = pd.DataFrame(
        {f"+/-{w:.2f}": [(np.abs(risk - t) <= w).mean() for t in cands]
         for w in sel["band_widths"]}, index=[f"t={t:.2f}" for t in cands])
    log.info(band.map("{:.1%}".format).to_string())

    # 4. Calibration
    bins = sel["calibration_bins"]
    frac_late, mean_risk = calibration_curve(y_arr, risk, n_bins=bins)
    log.info("\n4. CALIBRATION (predicted risk vs actual late rate, by bin)")
    counts, _ = np.histogram(risk, bins=np.linspace(0, 1, bins + 1))
    counts = counts[counts > 0]
    for p, a, n in zip(mean_risk, frac_late, counts):
        log.info(f"   predicted {p:.2f} -> actually late {a:.2f} "
                 f"(difference {a - p:+.2f}, {n:,} orders)")
    log.info(f"   Brier score: {brier_score_loss(y_arr, risk):.4f} "
             f"(0 = perfect; always predicting the base rate would give "
             f"{y_arr.mean() * (1 - y_arr.mean()):.4f})")
    log.info(f"   Mean predicted risk {risk.mean():.3f} vs actual late rate "
             f"{y_arr.mean():.3f}")

    # Charts ---------------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_arr, risk)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(rec, prec, color="tab:blue")
    markers = {"t = 0.50 (default)": table[table["threshold"] == 0.5].iloc[0]}
    for rule, r in picks.items():
        markers[f"t = {r['threshold']:.2f} ({rule})"] = r
    seen = set()
    for label, r in markers.items():
        key = round(float(r["threshold"]), 2)
        if key in seen:
            continue
        seen.add(key)
        ax.scatter(r["recall"], r["precision"], s=60, zorder=3, label=label)
    ax.axhline(y_arr.mean(), color="grey", linestyle=":",
               label=f"Flag everything (precision {y_arr.mean():.2f})")
    ax.set_xlabel("Recall (share of late orders caught)")
    ax.set_ylabel("Precision (share of flags that are right)")
    ax.set_title("Precision-recall curve (training set, 5-fold out-of-fold)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pr_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(risk[y_arr == 0], bins=50, range=(0, 1), alpha=0.6, label="On time")
    ax.hist(risk[y_arr == 1], bins=50, range=(0, 1), alpha=0.6, label="Late")
    ax.set_xlabel("Predicted late-delivery risk")
    ax.set_ylabel("Number of training orders")
    ax.set_title("Risk scores cluster into groups (training set, out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "risk_score_histogram.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(mean_risk, frac_late, marker="o", label="Random Forest")
    ax.set_xlabel("Predicted risk (bin average)")
    ax.set_ylabel("Actual share late")
    ax.set_title("Calibration curve (training set, out-of-fold)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration_curve.png"), dpi=150)
    plt.close(fig)

    log.info(f"\nCharts -> {out}/pr_curve.png, risk_score_histogram.png, "
             f"calibration_curve.png")
    log.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
