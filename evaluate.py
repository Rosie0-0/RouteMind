"""
RouteMind - Step 6: FINAL evaluation on the untouched test set.

Run this ONCE, after every setting in config.yaml is frozen (model
settings, operating threshold, uncertainty band). Nothing may be tuned
from these results.

  1. Baseline vs Random Forest (operating + comparison threshold) vs MLP
  2. Random Forest confusion matrix and classification report
  3. Uncertainty band: how many test orders would go to a human
  4. Feature importance of the frozen Random Forest (for explainability)

Headline metric is recall on "late", always shown with precision and the
baseline: a model that flags every order gets 100% recall and is useless.

Writes logs/evaluate_final_<timestamp>.log and charts to outputs/
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
                             classification_report, confusion_matrix,
                             precision_score, recall_score)

from features import categorical_columns, load_config


def scores(y_true, y_pred):
    return {
        "Recall (late)": recall_score(y_true, y_pred),
        "Precision (late)": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Orders flagged": float(np.mean(y_pred)),
    }


def main():
    cfg = load_config()
    paths, ev = cfg["paths"], cfg["evaluation"]
    out = paths["output_dir"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(paths["log_dir"], f"evaluate_final_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    log = logging.getLogger()

    rf = joblib.load(paths["rf_model_out"])
    mlp = joblib.load(paths["mlp_model_out"])
    X_test, y_test = joblib.load(paths["test_set_out"])
    t_op = ev["decision_threshold"]
    t_cmp = ev["comparison_threshold"]
    band = ev["uncertainty_band"]

    log.info("=" * 64)
    log.info("ROUTEMIND FINAL TEST EVALUATION (single run, settings frozen)")
    log.info("=" * 64)
    log.info(f"Test set: {len(y_test):,} orders, {y_test.mean():.1%} late")
    log.info(f"Frozen RF settings: {cfg['random_forest']}")
    log.info(f"Operating threshold {t_op} | comparison threshold {t_cmp} | "
             f"uncertainty band +/-{band}")
    log.info("Disclosure: a preliminary test run at threshold 0.5 was made on "
             "10 Nov 16:29\n(logs/evaluate_20261110_162939.log) before the "
             "threshold analysis. No setting was\nchosen from it: max_depth, "
             "threshold and band were all chosen on training\ncross-validation "
             "(logs/tune_*.log, logs/threshold_*.log).")

    # 1. Model comparison
    rf_risk = rf.predict_proba(X_test)[:, 1]
    mlp_risk = mlp.predict_proba(X_test)[:, 1]
    rf_pred = (rf_risk >= t_op).astype(int)
    results = {
        "Baseline (flag every order)": scores(y_test, np.ones(len(y_test), dtype=int)),
        f"Random Forest, t={t_op} (operating)": scores(y_test, rf_pred),
        f"Random Forest, t={t_cmp} (comparison)": scores(y_test, (rf_risk >= t_cmp).astype(int)),
        f"MLP, t={t_op}": scores(y_test, (mlp_risk >= t_op).astype(int)),
    }
    table = pd.DataFrame(results).T.round(3)
    log.info("\n1. MODEL COMPARISON (test set)\n" + table.to_string())
    table.to_csv(os.path.join(out, "final_model_comparison.csv"))

    # 2. Confusion matrix + classification report (RF, operating threshold)
    cm = confusion_matrix(y_test, rf_pred)
    tn, fp, fn, tp = cm.ravel()
    log.info(f"\n2. RANDOM FOREST AT OPERATING THRESHOLD {t_op}")
    log.info(f"  Late orders caught (true positives):   {tp:,}")
    log.info(f"  Late orders missed (false negatives):  {fn:,}")
    log.info(f"  False alarms (false positives):        {fp:,}")
    log.info(f"  On-time correctly cleared (true neg.): {tn:,}")
    log.info("\n  Classification report:\n" + classification_report(
        y_test, rf_pred, target_names=["On time (0)", "Late (1)"], digits=3))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["On time", "Late"]).plot(
        ax=ax, cmap="Blues", values_format=",")
    ax.set_title(f"Random Forest - final test set (threshold {t_op})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "final_confusion_matrix_rf.png"), dpi=150)
    plt.close(fig)

    # Headline chart: recall shown with precision, against the baseline
    fig, ax = plt.subplots(figsize=(8, 4.2))
    table[["Recall (late)", "Precision (late)"]].plot.bar(ax=ax, rot=0, width=0.75)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score on the test set")
    ax.set_title("Final test result: recall is always read together with precision")
    ax.set_xticklabels([label.replace(", ", "\n").replace(" (", "\n(")
                        for label in table.index], fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "final_headline_metric.png"), dpi=150)
    plt.close(fig)

    # 3. Uncertainty band
    uncertain = np.abs(rf_risk - t_op) <= band
    log.info(f"\n3. UNCERTAINTY BAND (risk within {t_op} +/- {band})")
    log.info(f"  Orders routed to a human as 'uncertain': {int(uncertain.sum()):,} "
             f"of {len(y_test):,} ({uncertain.mean():.2%})")

    # 4. Feature importance (property of the frozen model, not of the test set)
    prep = rf.named_steps["prep"]
    importances = rf.named_steps["model"].feature_importances_
    totals = {}
    for name, imp in zip(prep.get_feature_names_out(), importances):
        part, col = name.split("__", 1)
        if part == "cat":
            col = next(c for c in categorical_columns(cfg) if col.startswith(c + "_"))
        totals[col] = totals.get(col, 0) + imp
    fi = pd.Series(totals).sort_values(ascending=False)
    log.info("\n4. FEATURE IMPORTANCE (frozen Random Forest)\n"
             + fi.map("{:.1%}".format).to_string())
    fi.to_csv(os.path.join(out, "final_feature_importance.csv"), header=["importance"])

    fig, ax = plt.subplots(figsize=(7, 5))
    fi.sort_values().plot.barh(ax=ax, color="tab:blue")
    ax.set_xlabel("Share of total importance")
    ax.set_title("What drives RouteMind's late-delivery risk score")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "final_feature_importance.png"), dpi=150)
    plt.close(fig)

    log.info(f"\nCharts -> {out}/final_headline_metric.png, "
             f"final_confusion_matrix_rf.png, final_feature_importance.png")
    log.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
