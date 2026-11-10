"""
RouteMind - Step 1.4: Evaluate the trained models on the untouched test set.

  1. Compare baseline vs Random Forest vs MLP (recall, precision, accuracy)
  2. Confusion matrix for the Random Forest
  3. Threshold trade-off: how recall and precision change with the cut-off
  4. Feature importance: which order details drive the risk score

Headline metric is recall on "late", but precision is always shown next to
it: a model that flags every order as late gets 100% recall and is useless.

Every run writes logs/evaluate_<timestamp>.log and charts to outputs/
"""

import logging
import os
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_score, recall_score)

from features import categorical_columns, load_config


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"evaluate_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"),
                  logging.StreamHandler()])
    return path


def scores(y_true, y_pred):
    return {
        "Recall (late)": recall_score(y_true, y_pred),
        "Precision (late)": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def main():
    cfg = load_config()
    paths, ev = cfg["paths"], cfg["evaluation"]
    out = paths["output_dir"]
    os.makedirs(out, exist_ok=True)
    log_path = setup_logging(paths["log_dir"])
    log = logging.getLogger()

    rf = joblib.load(paths["rf_model_out"])
    mlp = joblib.load(paths["mlp_model_out"])
    X_test, y_test = joblib.load(paths["test_set_out"])
    threshold = ev["decision_threshold"]

    log.info("=" * 60)
    log.info("ROUTEMIND EVALUATION (test set: %s orders, %.1f%% late)",
             f"{len(y_test):,}", 100 * y_test.mean())
    log.info("=" * 60)

    # 1. Model comparison
    rf_risk = rf.predict_proba(X_test)[:, 1]
    mlp_risk = mlp.predict_proba(X_test)[:, 1]
    results = {
        "Baseline (always 'late')": scores(y_test, np.ones(len(y_test), dtype=int)),
        f"Random Forest (t={threshold})": scores(y_test, (rf_risk >= threshold).astype(int)),
        f"MLP (t={threshold})": scores(y_test, (mlp_risk >= threshold).astype(int)),
    }
    table = pd.DataFrame(results).T.round(3)
    log.info("\n1. MODEL COMPARISON\n" + table.to_string())
    table.to_csv(os.path.join(out, "model_comparison.csv"))

    # 2. Confusion matrix (Random Forest)
    rf_pred = (rf_risk >= threshold).astype(int)
    cm = confusion_matrix(y_test, rf_pred)
    tn, fp, fn, tp = cm.ravel()
    log.info("\n2. RANDOM FOREST CONFUSION MATRIX (threshold %.2f)", threshold)
    log.info(f"  Late orders caught (true positives):   {tp:,}")
    log.info(f"  Late orders missed (false negatives):  {fn:,}")
    log.info(f"  False alarms (false positives):        {fp:,}")
    log.info(f"  On-time correctly cleared (true neg.): {tn:,}")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["On time", "Late"]).plot(
        ax=ax, cmap="Blues", values_format=",")
    ax.set_title(f"Random Forest - test set (threshold {threshold})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix_rf.png"), dpi=150)
    plt.close(fig)

    # 3. Threshold trade-off
    rows = []
    for t in ev["thresholds_to_compare"]:
        s = scores(y_test, (rf_risk >= t).astype(int))
        s["Late missed"] = int(((rf_risk < t) & (y_test == 1)).sum())
        s["False alarms"] = int(((rf_risk >= t) & (y_test == 0)).sum())
        rows.append({"Threshold": t, **s})
    tt = pd.DataFrame(rows).set_index("Threshold").round(3)
    log.info("\n3. THRESHOLD TRADE-OFF (Random Forest)\n" + tt.to_string())
    tt.to_csv(os.path.join(out, "threshold_tradeoff.csv"))

    grid = np.linspace(0.05, 0.95, 91)
    rec = [recall_score(y_test, (rf_risk >= t).astype(int)) for t in grid]
    pre = [precision_score(y_test, (rf_risk >= t).astype(int), zero_division=0)
           for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, rec, label="Recall (late caught)")
    ax.plot(grid, pre, label="Precision (flags that are right)")
    ax.axvline(threshold, color="grey", linestyle="--",
               label=f"Current threshold {threshold}")
    ax.set_xlabel("Decision threshold (flag 'late' if risk >= threshold)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Random Forest: recall vs precision by threshold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "threshold_tradeoff.png"), dpi=150)
    plt.close(fig)

    # 4. Feature importance, one-hot columns summed back to the original feature
    prep = rf.named_steps["prep"]
    importances = rf.named_steps["model"].feature_importances_
    names = prep.get_feature_names_out()
    totals = {}
    for name, imp in zip(names, importances):
        part, col = name.split("__", 1)
        if part == "cat":
            col = next(c for c in categorical_columns(cfg) if col.startswith(c + "_"))
        totals[col] = totals.get(col, 0) + imp
    fi = pd.Series(totals).sort_values(ascending=False)
    log.info("\n4. FEATURE IMPORTANCE (Random Forest)\n"
             + fi.map("{:.1%}".format).to_string())
    fi.to_csv(os.path.join(out, "feature_importance.csv"), header=["importance"])

    fig, ax = plt.subplots(figsize=(7, 5))
    fi.sort_values().plot.barh(ax=ax, color="tab:blue")
    ax.set_xlabel("Share of total importance")
    ax.set_title("What drives RouteMind's late-delivery risk score")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feature_importance.png"), dpi=150)
    plt.close(fig)

    log.info("\nCharts saved to %s/: confusion_matrix_rf.png, "
             "threshold_tradeoff.png, feature_importance.png", out)
    log.info("Log: %s", log_path)


if __name__ == "__main__":
    main()
